#include <errno.h>
#include <fcntl.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <sys/mman.h>
#include <unistd.h>
#include <liburing.h>
#include <ionic/ionic.h>
#include <ionic/platform/linux/iouring.h>

#define IONIC_IOURING_TAG "io-uring"

#define IONIC_IOURING_QD_ENV_VAR   "IONIC_IOURING_QUEUE_DEPTH"
#define IONIC_IOURING_QD_DEFAULT   128U

#define IONIC_IOURING_BLOCK_SHIFT  12
#define IONIC_IOURING_BLOCK_SIZE   (1U << IONIC_IOURING_BLOCK_SHIFT) /* 4096 */
#define IONIC_IOURING_BLOCK_MASK   (IONIC_IOURING_BLOCK_SIZE - 1U)

#define IONIC_IOURING_SLOT_BLOCKS  32U
#define IONIC_IOURING_SLOT_SIZE    (IONIC_IOURING_SLOT_BLOCKS * IONIC_IOURING_BLOCK_SIZE) /* 128 KiB per slot */

/* ── slot bitmap helpers ──────────────────────────────────────────── */

typedef unsigned long slotword_t;
#define SLOT_BITS (sizeof(slotword_t) * 8)

struct ionic_arena {
    unsigned char  *buf;        /* mmap'd, page-aligned staging area      */
    size_t          buf_len;    /* total byte length of buf               */
    slotword_t     *free_map;   /* 1 = free, 0 = in-flight               */
    size_t          nslots;     /* total slot count                       */
    size_t          nwords;     /* number of words in free_map            */
};

static inline void arena_slot_set_free(struct ionic_arena *a, size_t idx)
{
    a->free_map[idx / SLOT_BITS] |= (slotword_t)1 << (idx % SLOT_BITS);
}

static inline void arena_slot_set_busy(struct ionic_arena *a, size_t idx)
{
    a->free_map[idx / SLOT_BITS] &= ~((slotword_t)1 << (idx % SLOT_BITS));
}

/*
 * Find and claim the first free slot.  Returns the slot index or (size_t)-1
 * when the arena is full.
 */
static size_t arena_slot_acquire(struct ionic_arena *a)
{
    for (size_t w = 0; w < a->nwords; ++w) {
        if (a->free_map[w] == 0)
            continue;
        int bit = __builtin_ctzl(a->free_map[w]);
        size_t idx = w * SLOT_BITS + (size_t)bit;
        if (idx >= a->nslots)
            return (size_t)-1;
        arena_slot_set_busy(a, idx);
        return idx;
    }
    return (size_t)-1;
}

static inline unsigned char *arena_slot_ptr(struct ionic_arena *a, size_t idx)
{
    return a->buf + idx * IONIC_IOURING_SLOT_SIZE;
}

/* ── completion cookie packed into sqe->user_data ─────────────────
 *
 *   63        52 51     40 39                     0
 *  ┌────────────┬─────────┬────────────────────────┐
 *  │  slot (12) │ pad (12)│      length (40)        │
 *  └────────────┴─────────┴────────────────────────┘
 *
 *  slot  : max 4096 (we cap at 1024)
 *  pad   : max 4095 (block alignment residual)
 *  length: max ~1 TiB (more than enough for a slot)
 *
 *  The destination pointer is stored in a side-table indexed by slot.
 */

struct ionic_iouring_flight {
    unsigned char *dst;
};

static inline __u64 pack_cookie(unsigned slot, unsigned pad, size_t len)
{
    return (((__u64)(slot & 0xFFF)) << 52) |
           (((__u64)(pad  & 0xFFF)) << 40) |
           ((__u64)(len & 0xFFFFFFFFFFULL));
}

static inline void unpack_cookie(__u64 cookie,
                                 unsigned *slot, unsigned *pad, size_t *len)
{
    *slot = (unsigned)(cookie >> 52) & 0xFFF;
    *pad  = (unsigned)(cookie >> 40) & 0xFFF;
    *len  = (size_t)(cookie & 0xFFFFFFFFFFULL);
}

/* ── registered file descriptors (required for SQPOLL) ────────────── */

#define IONIC_IOURING_MAX_FDS 64

struct ionic_fd_table {
    int     fds[IONIC_IOURING_MAX_FDS];
    size_t  count;
};

static void ionic_fd_table_init(struct ionic_fd_table *t)
{
    t->count = 0;
    for (size_t i = 0; i < IONIC_IOURING_MAX_FDS; ++i)
        t->fds[i] = -1;
}

/* ── backend structure ────────────────────────────────────────────── */

struct ionic_backend_iouring {
    struct ionic_backend          base;
    struct io_uring              *ring;
    struct ionic_arena            arena;
    struct ionic_iouring_flight  *flights;  /* indexed by slot */
    struct ionic_fd_table         fd_table;
    size_t                        depth;
    size_t                        inflight;
};

typedef struct ionic_backend_iouring ionic_backend_iouring_t;

static const struct ionic_backend ionic_iouring_vtable = {
    .destroy = ionic_iouring_destroy,
    .read    = ionic_iouring_read,
};

/* ── helpers ──────────────────────────────────────────────────────── */

static inline size_t align_down(size_t v, size_t a) { return v & ~(a - 1); }
static inline size_t align_up(size_t v, size_t a)   { return (v + a - 1) & ~(a - 1); }

/*
 * Look up an fd in the registered file table.  If not yet registered,
 * register it now via io_uring_register_files_update.
 * Returns the fixed-file index or -1 on failure.
 */
static int ionic_iouring_register_fd(struct ionic_context *ctx, struct ionic_backend_iouring *be, int fd)
{
    struct ionic_fd_table *t = &be->fd_table;

    for (size_t i = 0; i < t->count; ++i) {
        if (t->fds[i] == fd)
            return (int)i;
    }

    if (t->count >= IONIC_IOURING_MAX_FDS) {
        IONIC_ERROR(&ctx->logger, IONIC_IOURING_TAG, "fd_table_full max=%d", IONIC_IOURING_MAX_FDS);
        return -1;
    }

    int idx = (int)t->count;
    int ret = io_uring_register_files_update(be->ring, (unsigned)idx, &fd, 1);
    if (ret < 0) {
        IONIC_ERROR(&ctx->logger, IONIC_IOURING_TAG, "register_files_update_failed fd=%d idx=%d errno=%d", fd, idx, -ret);
        return -1;
    }

    t->fds[idx] = fd;
    t->count++;

    IONIC_DEBUG(&ctx->logger, IONIC_IOURING_TAG, "fd_registered fd=%d idx=%d total=%zu", fd, idx, t->count);
    return idx;
}

/* ── queue depth ─────────────────────────────────────────────────── */

static unsigned short ionic_iouring_get_queue_depth(struct ionic_context *ctx)
{
    char *qd_env = getenv(IONIC_IOURING_QD_ENV_VAR);
    if (!qd_env) {
        IONIC_DEBUG(&ctx->logger, IONIC_IOURING_TAG,
                    "queue_depth_env_not_set env_var=%s default=%u",
                    IONIC_IOURING_QD_ENV_VAR, IONIC_IOURING_QD_DEFAULT);
        return IONIC_IOURING_QD_DEFAULT;
    }

    unsigned long depth = strtoul(qd_env, NULL, 10);
    if (depth == 0) {
        IONIC_WARN(&ctx->logger, IONIC_IOURING_TAG,
                   "queue_depth_env_invalid requested=%lu default=%u",
                   depth, IONIC_IOURING_QD_DEFAULT);
        return IONIC_IOURING_QD_DEFAULT;
    }
    if (depth > 1024) {
        IONIC_WARN(&ctx->logger, IONIC_IOURING_TAG,
                   "queue_depth_env_clamped requested=%lu max=1024", depth);
        depth = 1024;
    }
    return (unsigned short)depth;
}

/* ── SQ poll CPU selection ───────────────────────────────────────── */

static unsigned short ionic_iouring_get_sq_thread_cpu(struct ionic_context *ctx)
{
    FILE *fp = fopen("/proc/stat", "r");
    if (!fp) {
        IONIC_WARN(&ctx->logger, IONIC_IOURING_TAG,
                   "proc_stat_open_failed default_sq_cpu=0");
        return 0;
    }

    unsigned short best_cpu = 0;
    double best_idle_ratio = -1.0;
    char line[256];

    while (fgets(line, sizeof(line), fp)) {
        unsigned int cpu_id;
        unsigned long long user, nice, system, idle, iowait, irq, softirq, steal;

        if (strncmp(line, "cpu", 3) != 0 || line[3] == ' ')
            continue;

        int matched = sscanf(
            line, "cpu%u %llu %llu %llu %llu %llu %llu %llu %llu",
            &cpu_id, &user, &nice, &system, &idle, &iowait, &irq, &softirq, &steal);
        if (matched < 5)
            continue;
        if (cpu_id >= ctx->topology.num_possible_cpus)
            continue;
        if (ctx->topology.node_of_core[cpu_id] == IONIC_NODE_NONE)
            continue;

        unsigned long long total = user + nice + system + idle;
        if (matched >= 6) total += iowait;
        if (matched >= 7) total += irq;
        if (matched >= 8) total += softirq;
        if (matched >= 9) total += steal;
        if (total == 0)
            continue;

        unsigned long long idle_total = idle;
        if (matched >= 6) idle_total += iowait;

        double idle_ratio = (double)idle_total / (double)total;
        if (idle_ratio > best_idle_ratio) {
            best_idle_ratio = idle_ratio;
            best_cpu = (unsigned short)cpu_id;
        }
    }

    fclose(fp);
    IONIC_DEBUG(&ctx->logger, IONIC_IOURING_TAG,
                "sq_cpu_selection sq_thread_cpu=%u idle_%=%.1f",
                best_cpu, best_idle_ratio * 100.0);
    return best_cpu;
}

/* ── arena lifecycle ─────────────────────────────────────────────── */

static int ionic_arena_init(struct ionic_context *ctx, struct ionic_arena *arena, size_t nslots)
{
    arena->nslots  = nslots;
    arena->buf_len = nslots * IONIC_IOURING_SLOT_SIZE;
    arena->nwords  = (nslots + SLOT_BITS - 1) / SLOT_BITS;

    arena->buf = mmap(NULL, arena->buf_len,
                      PROT_READ | PROT_WRITE,
                      MAP_PRIVATE | MAP_ANONYMOUS | MAP_POPULATE,
                      -1, 0);

    if (arena->buf == MAP_FAILED) {
        IONIC_ERROR(&ctx->logger, IONIC_IOURING_TAG, "arena_mmap_failed size=%zu errno=%d", arena->buf_len, errno);
        return -1;
    }

    arena->free_map = calloc(arena->nwords, sizeof(slotword_t));
    if (!arena->free_map) {
        munmap(arena->buf, arena->buf_len);
        return -1;
    }

    for (size_t i = 0; i < nslots; ++i)
        arena_slot_set_free(arena, i);

    IONIC_INFO(&ctx->logger, IONIC_IOURING_TAG, "arena_ready slots=%zu slot_size=%u total=%zu", nslots, IONIC_IOURING_SLOT_SIZE, arena->buf_len);
    return 0;
}

static void ionic_arena_destroy(struct ionic_arena *arena)
{
    if (arena->buf && arena->buf != MAP_FAILED)
        munmap(arena->buf, arena->buf_len);
    free(arena->free_map);
    arena->buf      = NULL;
    arena->free_map = NULL;
}

/* ── completion reaping ──────────────────────────────────────────── */

/*
 * Drain up to `max` completions.  For each CQE the useful payload
 * (skipping the O_DIRECT alignment padding) is copied to the
 * destination that was recorded before submission, then the slot is
 * returned to the arena.
 *
 * *bytes_out is incremented by the number of useful bytes copied.
 * Returns the number of completions reaped.
 */
static size_t ionic_iouring_reap(struct ionic_context *ctx, struct ionic_backend_iouring *be, unsigned max, int block, size_t *bytes_out)
{
    struct io_uring_cqe *cqe;
    size_t reaped = 0;

    for (unsigned i = 0; i < max; ++i) {
        int ret;
        if (block && i == 0)
            ret = io_uring_wait_cqe(be->ring, &cqe);
        else
            ret = io_uring_peek_cqe(be->ring, &cqe);

        if (ret < 0)
            break;

        unsigned slot, pad;
        size_t len;
        unpack_cookie(cqe->user_data, &slot, &pad, &len);

        if (cqe->res < 0) {
            IONIC_ERROR(&ctx->logger, IONIC_IOURING_TAG,
                        "cqe_error slot=%u res=%d", slot, cqe->res);
        } else {
            size_t avail = (size_t)cqe->res > pad
                           ? (size_t)cqe->res - pad : 0;
            size_t to_copy = avail < len ? avail : len;

            if (to_copy > 0) {
                unsigned char *src = arena_slot_ptr(&be->arena, slot) + pad;
                memcpy(be->flights[slot].dst, src, to_copy);
            }

            *bytes_out += to_copy;

            IONIC_TRACE(&ctx->logger, IONIC_IOURING_TAG,
                        "cqe_ok slot=%u pad=%u len=%zu res=%d copied=%zu dst=%p",
                        slot, pad, len, cqe->res, to_copy,
                        (void *)be->flights[slot].dst);
        }

        io_uring_cqe_seen(be->ring, cqe);
        arena_slot_set_free(&be->arena, slot);
        be->inflight--;
        reaped++;
    }
    return reaped;
}

/* ── public: read ─────────────────────────────────────────────────── */

size_t ionic_iouring_read(struct ionic_context *ctx, int fd, unsigned char *dst, size_t len, size_t offset)
{
    struct ionic_backend_iouring *be = (struct ionic_backend_iouring *)ctx->backend;

    IONIC_DEBUG(&ctx->logger, IONIC_IOURING_TAG, "read fd=%d dst=%p len=%zu offset=%zu", fd, dst, len, offset);

    int fixed_fd = ionic_iouring_register_fd(ctx, be, fd);
    if (fixed_fd < 0) {
        ctx->error = IONIC_ERR_WITH_MSG(IONIC_ERROR_BACKEND_INITIALIZATION_FAILED, "failed to register fd for SQPOLL");
        return 0;
    }

    size_t total_read = 0;
    size_t pos = 0;

    while (pos < len) {
        size_t file_off    = offset + pos;
        size_t aligned_off = align_down(file_off, IONIC_IOURING_BLOCK_SIZE);
        size_t pad         = file_off - aligned_off;
        size_t remaining   = len - pos;
        size_t chunk       = remaining;

        if (chunk + pad > IONIC_IOURING_SLOT_SIZE)
            chunk = IONIC_IOURING_SLOT_SIZE - pad;

        size_t io_len = align_up(pad + chunk, IONIC_IOURING_BLOCK_SIZE);

        size_t slot = arena_slot_acquire(&be->arena);
        while (slot == (size_t)-1) {
            ionic_iouring_reap(ctx, be, 1, /*block=*/1, &total_read);
            slot = arena_slot_acquire(&be->arena);
        }

        be->flights[slot].dst = dst + pos;

        struct io_uring_sqe *sqe = io_uring_get_sqe(be->ring);
        while (!sqe) {
            io_uring_submit(be->ring);
            ionic_iouring_reap(ctx, be, be->depth, /*block=*/0, &total_read);
            sqe = io_uring_get_sqe(be->ring);
        }

        io_uring_prep_read_fixed(sqe, fixed_fd, arena_slot_ptr(&be->arena, slot), io_len, aligned_off, /*buf_index=*/0);
        
        sqe->flags |= IOSQE_FIXED_FILE;
        sqe->user_data = pack_cookie((unsigned)slot, (unsigned)pad, chunk);
        be->inflight++;
        pos += chunk;

        IONIC_TRACE(&ctx->logger, IONIC_IOURING_TAG,
                    "sqe slot=%zu file_off=%zu aligned=%zu pad=%zu "
                    "io_len=%zu chunk=%zu inflight=%zu",
                    slot, file_off, aligned_off, pad, io_len, chunk, be->inflight);

        if (be->inflight >= be->depth / 2)
            io_uring_submit(be->ring);
    }

    io_uring_submit(be->ring);
    while (be->inflight > 0)
        ionic_iouring_reap(ctx, be, be->depth, /*block=*/1, &total_read);

    IONIC_DEBUG(&ctx->logger, IONIC_IOURING_TAG, "read_done fd=%d requested=%zu got=%zu", fd, len, total_read);

    return total_read;
}

/* ── public: destroy ──────────────────────────────────────────────── */

void ionic_iouring_destroy(struct ionic_context *ctx)
{
    struct ionic_backend_iouring *be =
        (struct ionic_backend_iouring *)ctx->backend;

    io_uring_unregister_buffers(be->ring);
    ionic_arena_destroy(&be->arena);
    free(be->flights);
    io_uring_queue_exit(be->ring);
    free(be->ring);
    free(be);
    ctx->backend = NULL;
    IONIC_INFO(&ctx->logger, IONIC_IOURING_TAG, "destroyed");
}

/* ── public: init ─────────────────────────────────────────────────── */

void ionic_iouring_init(struct ionic_context *ctx)
{
    ionic_backend_iouring_t *be = calloc(1, sizeof(*be));
    if (!be) {
        ctx->error = IONIC_ERR_WITH_MSG(IONIC_ERROR_ALLOCATION_FAILED, "failed to allocate io-uring backend");
        return;
    }

    be->depth = ionic_iouring_get_queue_depth(ctx);

    be->ring = malloc(sizeof(struct io_uring));
    if (!be->ring) {
        free(be);
        ctx->error = IONIC_ERR_WITH_MSG(IONIC_ERROR_ALLOCATION_FAILED, "failed to allocate io-uring ring");
        return;
    }

    unsigned short sq_cpu = ionic_iouring_get_sq_thread_cpu(ctx);
    struct io_uring_params params = {
        .flags         = IORING_SETUP_SQPOLL,
        .sq_thread_cpu = sq_cpu,
        .sq_thread_idle = 1000,
    };

    int ret = io_uring_queue_init_params(be->depth, be->ring, &params);
    if (ret < 0) {
        free(be->ring);
        free(be);
        ctx->error = IONIC_ERR_WITH_MSG(IONIC_ERROR_BACKEND_INITIALIZATION_FAILED, "io_uring_queue_init_params failed");
        return;
    }

    /* Arena: one slot per SQ entry so every in-flight op has guaranteed space. */
    if (ionic_arena_init(ctx, &be->arena, be->depth) < 0) {
        io_uring_queue_exit(be->ring);
        free(be->ring);
        free(be);
        ctx->error = IONIC_ERR(IONIC_ERROR_ALLOCATION_FAILED);
        return;
    }

    /* Register the arena as a single fixed buffer. */
    struct iovec iov = {
        .iov_base = be->arena.buf,
        .iov_len  = be->arena.buf_len,
    };
    ret = io_uring_register_buffers(be->ring, &iov, 1);
    if (ret < 0)
        IONIC_WARN(&ctx->logger, IONIC_IOURING_TAG, "register_buffers_failed errno=%d (continuing without fixed buffers)", -ret);

    /* Pre-register a sparse fd table so fds can be added at read time. */
    ionic_fd_table_init(&be->fd_table);
    ret = io_uring_register_files_sparse(be->ring, IONIC_IOURING_MAX_FDS);
    if (ret < 0) {
        IONIC_WARN(&ctx->logger, IONIC_IOURING_TAG, "register_files_sparse_failed errno=%d (SQPOLL may not work)", -ret);
    }

    be->flights = calloc(be->depth, sizeof(*be->flights));
    if (!be->flights) {
        io_uring_unregister_buffers(be->ring);
        ionic_arena_destroy(&be->arena);
        io_uring_queue_exit(be->ring);
        free(be->ring);
        free(be);
        ctx->error = IONIC_ERR(IONIC_ERROR_ALLOCATION_FAILED);
        return;
    }

    be->base = ionic_iouring_vtable;
    ctx->backend = &be->base;

    IONIC_INFO(&ctx->logger, IONIC_IOURING_TAG,
               "ready qd=%zu sq_thread_cpu=%u sq_thread_idle=%u "
               "arena_slots=%zu arena_size=%zu slot_size=%u",
               be->depth, params.sq_thread_cpu, params.sq_thread_idle,
               be->arena.nslots, be->arena.buf_len, IONIC_IOURING_SLOT_SIZE);
}
