#include <errno.h>
#include <fcntl.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/utsname.h>
#include <unistd.h>
#include <liburing.h>
#include "ionic/ionic.h"
#include "ionic/platform/linux/iouring.h"

#define IONIC_EVENT_TAG_IOURING "io-uring"

#define IONIC_IOURING_QD_ENV_VAR   "IONIC_IOURING_QUEUE_DEPTH"

#ifndef IONIC_IOURING_QD_DEFAULT
#define IONIC_IOURING_QD_DEFAULT     256U
#endif

#define IONIC_IOURING_READ_ALIGNMENT 4096U
#define IONIC_IOURING_SLOT_SIZE      512U * 1024U /* 512 kiB per slot */

/* ── kernel version detection ───────────────────────────────────────── */

static void ionic_iouring_get_kernel_version(unsigned *major, unsigned *minor)
{
    struct utsname uts;
    *major = 0;
    *minor = 0;

    if (uname(&uts) == 0) {
        sscanf(uts.release, "%u.%u", major, minor);
    }
}

/* Sparse file registration requires kernel 5.19+ */
#define IONIC_KERNEL_SPARSE_FILES_MIN_MAJOR 5
#define IONIC_KERNEL_SPARSE_FILES_MIN_MINOR 19

static inline bool ionic_iouring_kernel_supports_sparse_files(void)
{
    unsigned major, minor;
    ionic_iouring_get_kernel_version(&major, &minor);

    if (major > IONIC_KERNEL_SPARSE_FILES_MIN_MAJOR)
        return 1;
    if (major == IONIC_KERNEL_SPARSE_FILES_MIN_MAJOR && minor >= IONIC_KERNEL_SPARSE_FILES_MIN_MINOR)
        return 1;
    return 0;
}

/* ── slot bitmap helpers ──────────────────────────────────────────── */

typedef unsigned long slotword_t;
#define SLOT_BITS (sizeof(slotword_t) * 8)

struct ionic_staging_arena {
    slotword_t     *free_map;   /* 1 = free, 0 = in-flight                */
    size_t          nslots;     /* total slot count                       */
    size_t          nwords;     /* number of words in free_map            */
    size_t          length;     /* total byte length of buf               */
    unsigned char  *buf;        /* mmap'd, page-aligned staging area      */
};

static inline void arena_slot_set_free(struct ionic_staging_arena *a, size_t idx)
{
    a->free_map[idx / SLOT_BITS] |= 1ul << (idx % SLOT_BITS);
}

static inline void arena_slot_set_busy(struct ionic_staging_arena *a, size_t idx)
{
    a->free_map[idx / SLOT_BITS] &= ~(1ul << (idx % SLOT_BITS));
}

/*
 * Find and claim the first free slot.  Returns the slot index or (size_t)-1
 * when the arena is full.
 */
static size_t arena_slot_acquire(struct ionic_staging_arena *a)
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

static inline unsigned char *arena_slot_ptr(struct ionic_staging_arena *a, size_t idx)
{
    return a->buf + idx * IONIC_IOURING_SLOT_SIZE;
}

/* ── completion cookie packed into sqe->user_data ─────────────────
 *
 *   63        52 51     40 39                     0
 *  ┌────────────┬─────────┬────────────────────────┐
 *  │  slot (12) │ pad (12)│      length (40)       │
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

#define IONIC_IOURING_COOKIE_CHUNKED (1ULL << 63)

static inline __u64 pack_cookie(unsigned slot, unsigned pad, size_t len)
{
    return (((__u64)(slot & 0xFFF)) << 52) |
           (((__u64)(pad  & 0xFFF)) << 40) |
           ((__u64)(len & 0xFFFFFFFFFFULL));
}

static inline __u64 pack_cookie_chunked(unsigned cuda_slot, unsigned pad, size_t len)
{
    return IONIC_IOURING_COOKIE_CHUNKED |
           (((__u64)(cuda_slot & 0xFF)) << 52) |
           (((__u64)(pad & 0xFFF)) << 40) |
           ((__u64)(len & 0xFFFFFFFFFFULL));
}

static inline void unpack_cookie(__u64 cookie, unsigned *slot, unsigned *pad, size_t *len)
{
    *slot = (unsigned)(cookie >> 52) & 0xFFF;
    *pad  = (unsigned)(cookie >> 40) & 0xFFF;
    *len  = (size_t)(cookie & 0xFFFFFFFFFFULL);
}

/* ── registered file descriptors (required for SQPOLL) ────────────── */

#define IONIC_IOURING_MAX_FDS 256
#define IONIC_IOURING_FD_UNUSED -1

struct ionic_fd_table {
    size_t  count;
    int     fds[IONIC_IOURING_MAX_FDS];
};

static void ionic_fd_table_init(struct ionic_fd_table *t)
{
    t->count = 0;
    memset(t->fds, IONIC_IOURING_FD_UNUSED, sizeof(t->fds));
}

/* ── backend structure ────────────────────────────────────────────── */

struct ionic_backend_iouring {
    struct ionic_backend          base;
    struct io_uring              *ring;
    struct ionic_staging_arena    arena;
    struct ionic_iouring_flight  *flights;  /* indexed by slot */
    struct ionic_fd_table         fds;
    size_t                        depth;
    size_t                        inflight;
    bool                          use_fixed_buffers;
    bool                          use_fixed_files;
};

typedef struct ionic_backend_iouring ionic_backend_iouring_t;

/* Forward declarations for async I/O functions */
int ionic_iouring_submit_read(struct ionic_context *ctx, int fd, size_t offset, size_t len, void *userdata);
size_t ionic_iouring_poll_completions(struct ionic_context *ctx, struct ionic_io_completion *completions, size_t max_completions);
size_t ionic_iouring_get_inflight(struct ionic_context *ctx);
void *ionic_iouring_acquire_staging(struct ionic_context *ctx, size_t len, size_t *slot);
void ionic_iouring_release_staging(struct ionic_context *ctx, size_t slot);

static const struct ionic_backend ionic_iouring_vtable = {
    .destroy            = ionic_iouring_destroy,
    .read               = ionic_iouring_read,
    .submit_read        = ionic_iouring_submit_read,
    .poll_completions   = ionic_iouring_poll_completions,
    .get_inflight       = ionic_iouring_get_inflight,
    .acquire_staging    = ionic_iouring_acquire_staging,
    .release_staging    = ionic_iouring_release_staging,
};

/* ── helpers ──────────────────────────────────────────────────────── */

static inline size_t align_down(size_t v, size_t a) { return v & ~(a - 1); }
static inline size_t align_up(size_t v, size_t a)   { return (v + a - 1) & ~(a - 1); }

/*
 * Look up an fd in the registered file table. If not yet registered,
 * register it now via io_uring_register_files_update.
 * Returns the fixed-file index, or the original fd if fixed files not available.
 * Returns -1 on failure (only possible when fixed files are required but fail).
 */
static int ionic_iouring_register_fd(struct ionic_context *ctx, struct ionic_backend_iouring *be, int fd)
{
    /* If sparse file table not registered, just use the regular fd */
    if (!be->use_fixed_files)
        return fd;

    struct ionic_fd_table *table = &be->fds;

    for (size_t i = 0; i < table->count; ++i) {
        if (table->fds[i] == fd) {
            IONIC_TRACE(&ctx->logger, IONIC_EVENT_TAG_IOURING, "fd already registered fd=%d idx=%d", fd, i);
            return (int)i;
        }
    }

    if (table->count >= IONIC_IOURING_MAX_FDS) {
        IONIC_ERROR(&ctx->logger, IONIC_EVENT_TAG_IOURING, "fd table full max=%d", IONIC_IOURING_MAX_FDS);
        return -1;
    }

    int idx = (int) table->count;
    int ret = io_uring_register_files_update(be->ring, (unsigned)idx, &fd, 1);
    if (ret < 0) {
        IONIC_ERROR(&ctx->logger, IONIC_EVENT_TAG_IOURING, "fd register failed fd=%d idx=%d errno=%d", fd, idx, -ret);
        return -1;
    }

    table->fds[idx] = fd;
    table->count++;

    IONIC_DEBUG(&ctx->logger, IONIC_EVENT_TAG_IOURING, "fd registered fd=%d idx=%d total=%zu", fd, idx, table->count);
    return idx;
}

/* ── queue depth ─────────────────────────────────────────────────── */

static unsigned short ionic_iouring_get_queue_depth(struct ionic_context *ctx)
{
    char *qd_env = getenv(IONIC_IOURING_QD_ENV_VAR);
    if (!qd_env) {
        IONIC_DEBUG(&ctx->logger, IONIC_EVENT_TAG_IOURING, "queue depth env not set env_var=%s default=%u", IONIC_IOURING_QD_ENV_VAR, IONIC_IOURING_QD_DEFAULT);
        return IONIC_IOURING_QD_DEFAULT;
    }

    unsigned long depth = strtoul(qd_env, NULL, 10);
    if (depth == 0) {
        IONIC_WARN(&ctx->logger, IONIC_EVENT_TAG_IOURING, "queue depth env invalid requested=%lu default=%u", depth, IONIC_IOURING_QD_DEFAULT);
        return IONIC_IOURING_QD_DEFAULT;
    }
    if (depth > 1024) {
        IONIC_WARN(&ctx->logger, IONIC_EVENT_TAG_IOURING, "queue depth env clamped requested=%lu max=1024", depth);
        depth = 1024;
    }
    return (unsigned short)depth;
}

/* ── SQ poll CPU selection ───────────────────────────────────────── */

static unsigned short ionic_iouring_get_sq_thread_cpu(struct ionic_context *ctx)
{
    FILE *fp = fopen("/proc/stat", "r");
    if (!fp) {
        IONIC_WARN(&ctx->logger, IONIC_EVENT_TAG_IOURING, "/proc/stat open failed");
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
    IONIC_DEBUG(&ctx->logger, IONIC_EVENT_TAG_IOURING, "sq_thread_cpu selected core=%u idle_%=%.1f", best_cpu, best_idle_ratio * 100.0);
    return best_cpu;
}

/* ── arena lifecycle ─────────────────────────────────────────────── */

static int ionic_arena_init(struct ionic_context *ctx, struct ionic_staging_arena *arena, size_t nslots)
{
    arena->nslots  = nslots;
    arena->length = nslots * IONIC_IOURING_SLOT_SIZE;
    arena->nwords  = (nslots + SLOT_BITS - 1) / SLOT_BITS;

    arena->buf = mmap(NULL, arena->length,
                      PROT_READ | PROT_WRITE,
                      MAP_PRIVATE | MAP_ANONYMOUS | MAP_POPULATE,
                      -1, 0);

    if (arena->buf == MAP_FAILED) {
        IONIC_ERROR(&ctx->logger, IONIC_EVENT_TAG_IOURING, "arena mmap failed size=%zu errno=%d", arena->length, errno);
        return -1;
    }

    arena->free_map = calloc(arena->nwords, sizeof(slotword_t));
    if (!arena->free_map) {
        munmap(arena->buf, arena->length);
        return -1;
    }

    for (size_t i = 0; i < nslots; ++i)
        arena_slot_set_free(arena, i);

    IONIC_INFO(&ctx->logger, IONIC_EVENT_TAG_IOURING, "arena ready slots=%zu slot_size=%u total=%zu", nslots, IONIC_IOURING_SLOT_SIZE, arena->length);
    return 0;
}

static void ionic_arena_destroy(struct ionic_staging_arena *arena)
{
    if (arena->buf && arena->buf != MAP_FAILED)
        munmap(arena->buf, arena->length);

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
static size_t ionic_iouring_handle_completions(struct ionic_context *ctx, struct ionic_backend_iouring *be, unsigned max, int block, size_t *bytes_out)
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

        if (cqe->user_data & IONIC_IOURING_COOKIE_CHUNKED) {
            IONIC_ERROR(&ctx->logger, IONIC_EVENT_TAG_IOURING, "unexpected chunked cqe in sync read path");
            ionic_set_error(ctx, IONIC_ERR(IONIC_ERROR_IOURING));
            io_uring_cqe_seen(be->ring, cqe);
            be->inflight--;
            reaped++;
            continue;
        }

        if (cqe->res < 0) {
            IONIC_ERROR(&ctx->logger, IONIC_EVENT_TAG_IOURING, "cqe nack slot=%u res=%d", slot, cqe->res);
            ionic_set_error(ctx, IONIC_ERR(IONIC_ERROR_IOURING));
            goto exit;
        }
        
        
        size_t avail = (size_t)cqe->res > pad ? (size_t)cqe->res - pad : 0;
        size_t to_copy = avail < len ? avail : len;

        if (to_copy > 0) {
            unsigned char *src = arena_slot_ptr(&be->arena, slot) + pad;
            memcpy(be->flights[slot].dst, src, to_copy);
        }

        *bytes_out += to_copy;

        IONIC_TRACE(&ctx->logger, IONIC_EVENT_TAG_IOURING,
                    "cqe ack slot=%u pad=%u len=%zu res=%d copied=%zu dst=%p",
                    slot, pad, len, cqe->res, to_copy, (void *)be->flights[slot].dst);

        io_uring_cqe_seen(be->ring, cqe);
        arena_slot_set_free(&be->arena, slot);
        be->inflight--;
        reaped++;
    }

exit:
    return reaped;
}

/* ── public: read ─────────────────────────────────────────────────── */

size_t ionic_iouring_read(struct ionic_context *ctx, int fd, unsigned char *dst, size_t len, size_t offset)
{
    struct ionic_backend_iouring *be = (struct ionic_backend_iouring *)ctx->backend;

    IONIC_DEBUG(&ctx->logger, IONIC_EVENT_TAG_IOURING, "read fd=%d dst=%p len=%zu offset=%zu", fd, dst, len, offset);

    int fixed_fd = ionic_iouring_register_fd(ctx, be, fd);
    if (fixed_fd < 0) {
        ionic_set_error(ctx, IONIC_ERR_WITH_MSG(IONIC_ERROR_BACKEND_INITIALIZATION_FAILED, "failed to register fd for SQPOLL"));
        return 0;
    }

    size_t pos = 0, read_bytes = 0;
    size_t pending_submits = 0;

    while (pos < len) {
        size_t file_off    = offset + pos;
        size_t aligned_off = align_down(file_off, IONIC_IOURING_READ_ALIGNMENT);
        size_t pad         = file_off - aligned_off;
        size_t remaining   = len - pos;
        size_t chunk       = remaining;

        if (chunk + pad > IONIC_IOURING_SLOT_SIZE)
            chunk = IONIC_IOURING_SLOT_SIZE - pad;

        size_t io_len = align_up(pad + chunk, IONIC_IOURING_READ_ALIGNMENT);

        size_t slot = arena_slot_acquire(&be->arena);
        while (slot == (size_t)-1) {
            size_t reaped = ionic_iouring_handle_completions(ctx, be, 16, /*block=*/0, &read_bytes);
            if (reaped == 0) {
                ionic_iouring_handle_completions(ctx, be, 1, /*block=*/1, &read_bytes);
            }
            slot = arena_slot_acquire(&be->arena);
        }

        be->flights[slot].dst = dst + pos;

        struct io_uring_sqe *sqe = io_uring_get_sqe(be->ring);
        while (!sqe) {
            /* Ring full - submit and reap completions to free SQEs */
            io_uring_submit(be->ring);
            pending_submits = 0;
            ionic_iouring_handle_completions(ctx, be, be->depth, /*block=*/1, &read_bytes);
            sqe = io_uring_get_sqe(be->ring);
        }

        if (be->use_fixed_buffers)
            io_uring_prep_read_fixed(sqe, fixed_fd, arena_slot_ptr(&be->arena, slot), io_len, aligned_off, /*buf_index=*/0);
        else
            io_uring_prep_read(sqe, fixed_fd, arena_slot_ptr(&be->arena, slot), io_len, aligned_off);
        
        if (be->use_fixed_files)
            sqe->flags |= IOSQE_FIXED_FILE;
        
        sqe->user_data = pack_cookie((unsigned)slot, (unsigned)pad, chunk);
        be->inflight++;
        pending_submits++;
        pos += chunk;

        IONIC_TRACE(&ctx->logger, IONIC_EVENT_TAG_IOURING,
                    "sqe slot=%zu file_off=%zu aligned=%zu pad=%zu "
                    "io_len=%zu chunk=%zu inflight=%zu",
                    slot, file_off, aligned_off, pad, io_len, chunk, be->inflight);

        if (pending_submits >= 32 || be->inflight >= be->depth - 4) {
            io_uring_submit(be->ring);
            pending_submits = 0;
        }
    }

    if (pending_submits > 0) {
        io_uring_submit(be->ring);
    }

    while (be->inflight > 0)
        ionic_iouring_handle_completions(ctx, be, be->depth, /*block=*/1, &read_bytes);
    

    IONIC_DEBUG(&ctx->logger, IONIC_EVENT_TAG_IOURING, "read done fd=%d requested=%zu got=%zu", fd, len, read_bytes);

    return read_bytes;
}

/* ═══════════════════════════════════════════════════════════════════════════
 * Async I/O Operations
 * ═══════════════════════════════════════════════════════════════════════════ */

/* In-flight async operation tracking (stored in flights[slot].dst) */
struct ionic_async_op {
    void   *userdata;
    size_t  slot;
};

int ionic_iouring_submit_read(struct ionic_context *ctx, int fd, size_t offset, size_t len, void *userdata)
{
    struct ionic_backend_iouring *be = (struct ionic_backend_iouring *)ctx->backend;
    if (!be || !be->ring) {
        ionic_set_error(ctx, IONIC_ERR(IONIC_ERROR_BACKEND_INITIALIZATION_FAILED));
        return -1;
    }

    int fixed_fd = ionic_iouring_register_fd(ctx, be, fd);
    if (fixed_fd < 0 && be->use_fixed_files) {
        ionic_set_error(ctx, IONIC_ERR_WITH_MSG(IONIC_ERROR_BACKEND_INITIALIZATION_FAILED, "failed to register fd"));
        return -1;
    }
    if (fixed_fd < 0) fixed_fd = fd;

    /* Compute aligned I/O parameters */
    size_t aligned_off = align_down(offset, IONIC_IOURING_READ_ALIGNMENT);
    size_t pad = offset - aligned_off;
    size_t io_len = align_up(pad + len, IONIC_IOURING_READ_ALIGNMENT);

    /* Acquire a staging slot */
    size_t slot = arena_slot_acquire(&be->arena);
    if (slot == (size_t)-1) {
        IONIC_TRACE(&ctx->logger, IONIC_EVENT_TAG_IOURING, "no slots available for async read");
        return -1;
    }

    /* Allocate tracking for this async op */
    struct ionic_async_op *op = malloc(sizeof(*op));
    if (!op) {
        arena_slot_set_free(&be->arena, slot);
        ionic_set_error(ctx, IONIC_ERR(IONIC_ERROR_ALLOCATION_FAILED));
        return -1;
    }
    op->userdata = userdata;
    op->slot = slot;

    /* Get an SQE */
    struct io_uring_sqe *sqe = io_uring_get_sqe(be->ring);
    if (!sqe) {
        arena_slot_set_free(&be->arena, slot);
        free(op);
        IONIC_ERROR(&ctx->logger, IONIC_EVENT_TAG_IOURING, "SQ ring full for async read");
        return -1;
    }

    /* Prepare the read */
    unsigned char *buf = arena_slot_ptr(&be->arena, slot);
    if (be->use_fixed_buffers)
        io_uring_prep_read_fixed(sqe, fixed_fd, buf, io_len, aligned_off, /*buf_index=*/0);
    else
        io_uring_prep_read(sqe, fixed_fd, buf, io_len, aligned_off);
    
    if (be->use_fixed_files)
        sqe->flags |= IOSQE_FIXED_FILE;
    
    /* Pack: slot (12 bits) | pad (12 bits) | len (40 bits), with async flag */
    sqe->user_data = (((__u64)(slot & 0xFFF)) << 52) |
                     (((__u64)(pad & 0xFFF)) << 40) |
                     ((__u64)(len & 0xFFFFFFFFFFULL)) |
                     (1ULL << 63);  /* Async flag */
    
    /* Store the async op pointer in flights */
    be->flights[slot].dst = (unsigned char *)op;
    be->inflight++;

    IONIC_TRACE(&ctx->logger, IONIC_EVENT_TAG_IOURING,
                "async submit slot=%zu offset=%zu len=%zu io_len=%zu",
                slot, offset, len, io_len);

    return 0;
}

size_t ionic_iouring_poll_completions(struct ionic_context *ctx, struct ionic_io_completion *completions, size_t max_completions)
{
    struct ionic_backend_iouring *be = (struct ionic_backend_iouring *)ctx->backend;
    if (!be || !be->ring) return 0;

    struct io_uring_cqe *cqe;
    size_t completed = 0;

    for (size_t i = 0; i < max_completions; i++) {
        if (io_uring_peek_cqe(be->ring, &cqe) != 0)
            break;

        __u64 ud = cqe->user_data;
        bool is_async = (ud >> 63) & 1;
        
        unsigned slot = (unsigned)((ud >> 52) & 0xFFF);
        unsigned pad = (unsigned)((ud >> 40) & 0xFFF);
        size_t len = (size_t)(ud & 0xFFFFFFFFFFULL);

        io_uring_cqe_seen(be->ring, cqe);
        be->inflight--;

        if (cqe->res < 0) {
            IONIC_ERROR(&ctx->logger, IONIC_EVENT_TAG_IOURING, 
                        "async cqe error slot=%u res=%d", slot, cqe->res);
            if (is_async) {
                struct ionic_async_op *op = (struct ionic_async_op *)be->flights[slot].dst;
                free(op);
                arena_slot_set_free(&be->arena, slot);
            }
            continue;
        }

        if (is_async && completions) {
            struct ionic_async_op *op = (struct ionic_async_op *)be->flights[slot].dst;
            unsigned char *buf = arena_slot_ptr(&be->arena, slot);
            
            size_t actual = (size_t)cqe->res > pad ? (size_t)cqe->res - pad : 0;
            size_t to_deliver = actual < len ? actual : len;
            
            completions[completed].staging_buffer = buf + pad;
            completions[completed].bytes_read = to_deliver;
            completions[completed].userdata = op->userdata;
            
            free(op);
            /* Don't release slot - caller must do that via release_staging */
            completed++;
        }
    }

    return completed;
}

size_t ionic_iouring_get_inflight(struct ionic_context *ctx)
{
    struct ionic_backend_iouring *be = (struct ionic_backend_iouring *)ctx->backend;
    return be ? be->inflight : 0;
}

void *ionic_iouring_acquire_staging(struct ionic_context *ctx, size_t len, size_t *slot_out)
{
    struct ionic_backend_iouring *be = (struct ionic_backend_iouring *)ctx->backend;
    if (!be) return NULL;
    
    size_t slot = arena_slot_acquire(&be->arena);
    if (slot == (size_t)-1) return NULL;
    
    if (slot_out) *slot_out = slot;
    return arena_slot_ptr(&be->arena, slot);
}

void ionic_iouring_release_staging(struct ionic_context *ctx, size_t slot)
{
    struct ionic_backend_iouring *be = (struct ionic_backend_iouring *)ctx->backend;
    if (!be) return;
    arena_slot_set_free(&be->arena, slot);
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
    IONIC_INFO(&ctx->logger, IONIC_EVENT_TAG_IOURING, "destroyed");
}

/* ── public: init ─────────────────────────────────────────────────── */
void ionic_iouring_init(struct ionic_context *ctx)
{
    ionic_backend_iouring_t *be = calloc(1, sizeof(*be));
    if (!be) {
        ionic_set_error(ctx, IONIC_ERR_WITH_MSG(IONIC_ERROR_ALLOCATION_FAILED, "failed to allocate io-uring backend"));
        return;
    }

    be->depth = ionic_iouring_get_queue_depth(ctx);
    be->ring = malloc(sizeof(struct io_uring));
    if (!be->ring) {
        free(be);
        ionic_set_error(ctx, IONIC_ERR_WITH_MSG(IONIC_ERROR_ALLOCATION_FAILED, "failed to allocate io-uring ring"));
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
        ionic_set_error(ctx, IONIC_ERR_WITH_MSG(IONIC_ERROR_BACKEND_INITIALIZATION_FAILED, "io_uring_queue_init_params failed"));
        return;
    }

    /* Arena: one slot per SQ entry so every in-flight op has guaranteed space. */
    if (ionic_arena_init(ctx, &be->arena, be->depth) < 0) {
        io_uring_queue_exit(be->ring);
        free(be->ring);
        free(be);
        ionic_set_error(ctx, IONIC_ERR(IONIC_ERROR_ALLOCATION_FAILED));
        return;
    }

    /* Register the arena as a single fixed buffer. */
    struct iovec iov = {
        .iov_base = be->arena.buf,
        .iov_len  = be->arena.length,
    };

    ret = io_uring_register_buffers(be->ring, &iov, 1);
    be->use_fixed_buffers = (ret >= 0);
    if (ret < 0)
        IONIC_WARN(&ctx->logger, IONIC_EVENT_TAG_IOURING, "register_buffers_failed errno=%d (continuing without fixed buffers)", -ret);

    /* Pre-register a file table for fixed file support.
     * Sparse file registration requires kernel 5.19+.
     * On older kernels, use non-fixed files (regular fd).
     */
    ionic_fd_table_init(&be->fds);

    if (ionic_iouring_kernel_supports_sparse_files()) {
        ret = io_uring_register_files_sparse(be->ring, IONIC_IOURING_MAX_FDS);
        if (ret >= 0) {
            be->use_fixed_files = true;
            IONIC_DEBUG(&ctx->logger, IONIC_EVENT_TAG_IOURING, "sparse file table registered max=%d", IONIC_IOURING_MAX_FDS);
        } else {
            IONIC_WARN(&ctx->logger, IONIC_EVENT_TAG_IOURING, "sparse file registration failed errno=%d", -ret);
            be->use_fixed_files = false;
        }
    } else {
        unsigned kver_major, kver_minor;
        ionic_iouring_get_kernel_version(&kver_major, &kver_minor);
        IONIC_WARN(&ctx->logger, IONIC_EVENT_TAG_IOURING, "io_uring_register_files_sparse not available with kernel %u.%u (5.19+)", kver_major, kver_minor);
        be->use_fixed_files = false;
    }

    be->flights = calloc(be->depth, sizeof(*be->flights));
    if (!be->flights) {
        io_uring_unregister_buffers(be->ring);
        ionic_arena_destroy(&be->arena);
        io_uring_queue_exit(be->ring);
        free(be->ring);
        free(be);
        ionic_set_error(ctx, IONIC_ERR(IONIC_ERROR_ALLOCATION_FAILED));
        return;
    }

    be->base = ionic_iouring_vtable;
    ctx->backend = &be->base;

    IONIC_INFO(&ctx->logger, IONIC_EVENT_TAG_IOURING,
               "ready qd=%zu sq_thread_cpu=%u sq_thread_idle=%u arena=%zu slots=%zu chunk=%u fixed_bufs=%s fixed_files=%s",
               be->depth, params.sq_thread_cpu, params.sq_thread_idle,
               be->arena.length, be->arena.nslots, IONIC_IOURING_SLOT_SIZE,
               be->use_fixed_buffers ? "yes" : "no",
               be->use_fixed_files ? "yes" : "no");

    unsigned kver_major, kver_minor;
    ionic_iouring_get_kernel_version(&kver_major, &kver_minor);
    IONIC_DEBUG(&ctx->logger, IONIC_EVENT_TAG_IOURING, "kernel version=%u.%u", kver_major, kver_minor);
}

#ifdef __IONIC_CUDA_ENABLED__

#include <cuda_runtime.h>
#include <ionic/utils.h>
#include <numa.h>
#include <stdatomic.h>

static void unpack_cookie_chunked(__u64 cookie, unsigned *cuda_slot, unsigned *pad, size_t *len)
{
    *cuda_slot = (unsigned)((cookie >> 52) & 0xFF);
    *pad       = (unsigned)((cookie >> 40) & 0xFFF);
    *len       = (size_t)(cookie & 0xFFFFFFFFFFULL);
}

static int ionic_cuda_numa_node_for_gpu(struct ionic_context *ctx)
{
    char busid[20];
    cudaError_t err = cudaDeviceGetPCIBusId(busid, sizeof(busid), (int)ctx->device.ordinal);
    if (err != cudaSuccess)
        return -1;

    char path[128];
    snprintf(path, sizeof(path), "/sys/bus/pci/devices/%s/numa_node", busid);
    FILE *fp = fopen(path, "r");
    if (!fp)
        return -1;
    int node = -1;
    if (fscanf(fp, "%d", &node) != 1)
        node = -1;
    fclose(fp);
    return node;
}

static void ionic_cuda_bind_numa_for_staging_alloc(struct ionic_context *ctx)
{
    if (!numa_available() || numa_available() < 0)
        return;
    int node = ionic_cuda_numa_node_for_gpu(ctx);
    if (node < 0)
        return;
    struct bitmask *bm = numa_allocate_nodemask();
    if (!bm)
        return;
    numa_bitmask_setbit(bm, node);
    numa_set_membind(bm);
    numa_bitmask_free(bm);
}


// void ionic_iouring_cuda_pipeline_destroy(struct ionic_context *ctx)
// {
//     if (!ctx || !ctx->cuda_pipeline_inited)
//         return;

//     (void)cudaSetDevice((int)ctx->device.ordinal);

//     for (unsigned i = 0; i < IONIC_STAGING_SLOTS; i++) {
//         struct ionic_staging_slot_cuda *sl = &ctx->cuda_slots[i];
//         if (sl->h2d_done) {
//             (void)cudaEventDestroy(sl->h2d_done);
//             sl->h2d_done = NULL;
//         }
//         if (sl->scatter_done) {
//             (void)cudaEventDestroy(sl->scatter_done);
//             sl->scatter_done = NULL;
//         }
//         if (sl->staging_ptr) {
//             ionic_free_device(ctx, sl->staging_ptr, IONIC_ALLOC_STAGING);
//             sl->staging_ptr = NULL;
//         }
//         if (sl->device_ptr) {
//             ionic_free_device(ctx, sl->device_ptr, IONIC_ALLOC_DEVICE);
//             sl->device_ptr = NULL;
//         }
//         atomic_store_explicit(&sl->state, IONIC_SLOT_EMPTY, memory_order_relaxed);
//         sl->bound_chunk = NULL;
//     }

//     if (ctx->cuda_stream) {
//         (void)cudaStreamDestroy(ctx->cuda_stream);
//         ctx->cuda_stream = NULL;
//     }

//     ctx->cuda_pipeline_inited = 0;
// }

// static int ionic_iouring_cuda_pipeline_ensure(struct ionic_context *ctx)
// {
//     if (ctx->cuda_pipeline_inited)
//         return 0;
//     if (ctx->device.kind != IONIC_DEVICE_CUDA) {
//         ionic_set_error(ctx, IONIC_ERR(IONIC_ERROR_UNSUPPORTED_DEVICE));
//         return -1;
//     }

//     cudaError_t ce = cudaSetDevice((int)ctx->device.ordinal);
//     if (ce != cudaSuccess) {
//         ionic_set_error(ctx, IONIC_CUDA_ERR((int)ce));
//         return -1;
//     }

//     ce = cudaStreamCreate(&ctx->cuda_stream);
//     if (ce != cudaSuccess) {
//         ionic_set_error(ctx, IONIC_CUDA_ERR((int)ce));
//         return -1;
//     }

//     for (unsigned i = 0; i < IONIC_STAGING_SLOTS; i++) {
//         struct ionic_staging_slot_cuda *sl = &ctx->cuda_slots[i];
//         atomic_store_explicit(&sl->state, IONIC_SLOT_EMPTY, memory_order_relaxed);
//         sl->bound_chunk = NULL;

//         ce = cudaEventCreateWithFlags(&sl->h2d_done, cudaEventDisableTiming);
//         if (ce != cudaSuccess)
//             goto fail;
//         ce = cudaEventCreateWithFlags(&sl->scatter_done, cudaEventDisableTiming);
//         if (ce != cudaSuccess)
//             goto fail;

//         ionic_cuda_bind_numa_for_staging_alloc(ctx);
//         sl->staging_ptr = ionic_allocate_device(ctx, IONIC_CHUNK_SIZE, IONIC_ALLOC_STAGING);
//         if (!sl->staging_ptr)
//             goto fail;

//         sl->device_ptr = ionic_allocate_device(ctx, IONIC_CHUNK_SIZE, IONIC_ALLOC_DEVICE);
//         if (!sl->device_ptr)
//             goto fail;
//     }

//     ctx->cuda_pipeline_inited = 1;
//     return 0;

// fail:
//     ionic_iouring_cuda_pipeline_destroy(ctx);
//     if (!ionic_has_error(&ctx->error))
//     ionic_set_error(ctx, IONIC_ERR(IONIC_ERROR_ALLOCATION_FAILED));
//     return -1;
// }

// static int ionic_iouring_submit_chunk_read(
//     struct ionic_context *ctx, struct ionic_backend_iouring *be, int fixed_fd, unsigned cuda_slot, struct ionic_read_chunk *ch, size_t chunk_index)
// {
//     struct ionic_staging_slot_cuda *sl = &ctx->cuda_slots[cuda_slot];
//     sl->bound_chunk          = ch;
//     sl->submitted_chunk_index = chunk_index;
//     atomic_store_explicit(&sl->state, IONIC_SLOT_FILLING, memory_order_release);

//     struct io_uring_sqe *sqe = io_uring_get_sqe(be->ring);
//     unsigned spins = 0;
//     while (!sqe) {
//         io_uring_submit(be->ring);
//         if (++spins > 1000000) {
//             ionic_set_error(ctx, IONIC_ERR(IONIC_ERROR_IOURING));
//             atomic_store_explicit(&sl->state, IONIC_SLOT_EMPTY, memory_order_release);
//             return -1;
//         }
//         sqe = io_uring_get_sqe(be->ring);
//     }

//     io_uring_prep_read(sqe, fixed_fd, sl->staging_ptr, (unsigned)ch->io_len, (__s64)ch->file_offset);
//     sqe->flags |= IOSQE_FIXED_FILE;
//     sqe->user_data = pack_cookie_chunked(cuda_slot, ch->pad, ch->io_len);
//     be->inflight++;
//     io_uring_submit(be->ring);
//     return 0;
// }

// static int ionic_iouring_poll_chunk_cqe(struct ionic_context *ctx, struct ionic_backend_iouring *be, unsigned *cuda_slot_out)
// {
//     struct io_uring_cqe *cqe;
//     if (io_uring_peek_cqe(be->ring, &cqe) != 0)
//         return 0;

//     __u64 ud = cqe->user_data;
//     if (!(ud & IONIC_IOURING_COOKIE_CHUNKED))
//         return 0;

//     unsigned cuda_slot, pad;
//     size_t expect_len;
//     unpack_cookie_chunked(ud, &cuda_slot, &pad, &expect_len);

//     if (cuda_slot >= IONIC_STAGING_SLOTS) {
//         IONIC_ERROR(&ctx->logger, IONIC_EVENT_TAG_IOURING, "chunked cqe bad slot=%u", cuda_slot);
//         ionic_set_error(ctx, IONIC_ERR(IONIC_ERROR_IOURING));
//         io_uring_cqe_seen(be->ring, cqe);
//         be->inflight--;
//         return -1;
//     }

//     if (cqe->res < 0) {
//         IONIC_ERROR(&ctx->logger, IONIC_EVENT_TAG_IOURING, "chunked cqe nack slot=%u res=%d", cuda_slot, cqe->res);
//         ionic_set_error(ctx, IONIC_ERR(IONIC_ERROR_IOURING));
//         io_uring_cqe_seen(be->ring, cqe);
//         be->inflight--;
//         atomic_store_explicit(&ctx->cuda_slots[cuda_slot].state, IONIC_SLOT_EMPTY, memory_order_release);
//         return -1;
//     }

//     if ((size_t)cqe->res < expect_len) {
//         IONIC_ERROR(&ctx->logger, IONIC_EVENT_TAG_IOURING, "chunked cqe short read slot=%u got=%d want=%zu", cuda_slot, cqe->res, expect_len);
//         ionic_set_error(ctx, IONIC_ERR(IONIC_ERROR_IOURING));
//         io_uring_cqe_seen(be->ring, cqe);
//         be->inflight--;
//         atomic_store_explicit(&ctx->cuda_slots[cuda_slot].state, IONIC_SLOT_EMPTY, memory_order_release);
//         return -1;
//     }

//     struct ionic_staging_slot_cuda *sl = &ctx->cuda_slots[cuda_slot];
//     if (atomic_load_explicit(&sl->state, memory_order_acquire) != IONIC_SLOT_FILLING) {
//         IONIC_ERROR(&ctx->logger, IONIC_EVENT_TAG_IOURING, "chunked cqe bad state slot=%u", cuda_slot);
//         ionic_set_error(ctx, IONIC_ERR(IONIC_ERROR_IOURING));
//         io_uring_cqe_seen(be->ring, cqe);
//         be->inflight--;
//         return -1;
//     }

//     io_uring_cqe_seen(be->ring, cqe);
//     be->inflight--;
//     atomic_store_explicit(&sl->state, IONIC_SLOT_FILLED, memory_order_release);
//     *cuda_slot_out = cuda_slot;
//     return 1;
// }

// void ionic_iouring_read_chunked(struct ionic_context *ctx, struct ionic_read_plan *plan, ionic_chunk_ready_cb on_ready, void *userdata)
// {
//     if (!ctx || !plan || !plan->chunks || plan->n_chunks == 0) {
//         if (ctx)
//         ionic_set_error(ctx, IONIC_ERR(IONIC_ERROR_ALLOCATION_FAILED));
//         return;
//     }
//     if (ctx->device.kind != IONIC_DEVICE_CUDA) {
//         ionic_set_error(ctx, IONIC_ERR(IONIC_ERROR_UNSUPPORTED_DEVICE));
//         return;
//     }
//     if (ctx->n_ranks != 1u) {
//         ionic_set_error(ctx, IONIC_ERR(IONIC_ERROR_UNSUPPORTED));
//         return;
//     }
//     if (!ctx->weight_bufs || ctx->n_weight_bufs == 0) {
//         ionic_set_error(ctx, IONIC_ERR(IONIC_ERROR_ALLOCATION_FAILED));
//         return;
//     }

//     struct ionic_backend_iouring *be = (struct ionic_backend_iouring *)ctx->backend;
//     if (!be || !be->ring) {
//         ionic_set_error(ctx, IONIC_ERR(IONIC_ERROR_BACKEND_INITIALIZATION_FAILED));
//         return;
//     }

//     if (ionic_iouring_cuda_pipeline_ensure(ctx) != 0)
//         return;

//     int fixed_fd = ionic_iouring_register_fd(ctx, be, plan->fd);
//     if (fixed_fd < 0) {
//         ionic_set_error(ctx, IONIC_ERR_WITH_MSG(IONIC_ERROR_BACKEND_INITIALIZATION_FAILED, "failed to register fd for SQPOLL"));
//         return;
//     }

//     unsigned char *chunk_done = calloc(plan->n_chunks, 1);
//     if (!chunk_done) {
//         ionic_set_error(ctx, IONIC_ERR(IONIC_ERROR_ALLOCATION_FAILED));
//         return;
//     }

//     size_t submitted  = 0;
//     size_t next_emit  = 0;
//     size_t n_finished = 0;

//     while (n_finished < plan->n_chunks || next_emit < plan->n_chunks) {
//         if (ionic_has_error(&ctx->error))
//             break;

//         while (submitted < plan->n_chunks) {
//             unsigned free_slot = IONIC_STAGING_SLOTS;
//             for (unsigned s = 0; s < IONIC_STAGING_SLOTS; s++) {
//                 if (atomic_load_explicit(&ctx->cuda_slots[s].state, memory_order_acquire) == IONIC_SLOT_EMPTY) {
//                     free_slot = s;
//                     break;
//                 }
//             }
//             if (free_slot >= IONIC_STAGING_SLOTS)
//                 break;

//             struct ionic_read_chunk *ch = &plan->chunks[submitted];
//             if (ionic_iouring_submit_chunk_read(ctx, be, fixed_fd, free_slot, ch, submitted) != 0)
//                 goto loop_end;
//             submitted++;
//         }

//         unsigned cs;
//         while (ionic_iouring_poll_chunk_cqe(ctx, be, &cs) == 1) {
//             if (ionic_has_error(&ctx->error))
//                 goto loop_end;

//             struct ionic_staging_slot_cuda *sl = &ctx->cuda_slots[cs];
//             struct ionic_read_chunk *ch     = sl->bound_chunk;
//             if (!ch)
//                 continue;

//             cudaError_t ce = cudaMemcpyAsync(sl->device_ptr, sl->staging_ptr, ch->io_len, cudaMemcpyHostToDevice, ctx->cuda_stream);
//             if (ce != cudaSuccess) {
//                 ionic_set_error(ctx, (ionic_error_t){ .kind = IONIC_ERROR_CUDA, .res = (int)ce, .what = NULL });
//                 goto loop_end;
//             }
//             ce = cudaEventRecord(sl->h2d_done, ctx->cuda_stream);
//             if (ce != cudaSuccess) {
//                 ionic_set_error(ctx, (ionic_error_t){ .kind = IONIC_ERROR_CUDA, .res = (int)ce, .what = NULL });
//                 goto loop_end;
//             }
//             atomic_store_explicit(&sl->state, IONIC_SLOT_COPYING, memory_order_release);
//         }

//         for (unsigned s = 0; s < IONIC_STAGING_SLOTS; s++) {
//             struct ionic_staging_slot_cuda *sl = &ctx->cuda_slots[s];
//             if (atomic_load_explicit(&sl->state, memory_order_acquire) != IONIC_SLOT_COPYING)
//                 continue;
//             cudaError_t q = cudaEventQuery(sl->h2d_done);
//             if (q == cudaErrorNotReady)
//                 continue;
//             if (q != cudaSuccess) {
//                 ionic_set_error(ctx, (ionic_error_t){ .kind = IONIC_ERROR_CUDA, .res = (int)q, .what = NULL });
//                 goto loop_end;
//             }

//             struct ionic_read_chunk *ch = sl->bound_chunk;
//             for (uint32_t t = 0; t < ch->n_tensors; t++) {
//                 uint32_t ti = ch->tensor_idx[t];
//                 if (ti >= ctx->n_weight_bufs || !ctx->weight_bufs[ti]) {
//                     ionic_set_error(ctx, IONIC_ERR(IONIC_ERROR_ALLOCATION_FAILED));
//                     goto loop_end;
//                 }
//                 unsigned char *dst = (unsigned char *)ctx->weight_bufs[ti];
//                 unsigned char *src  = (unsigned char *)sl->device_ptr + ch->tensor_offset[t];
//                 size_t sz           = ch->tensor_size[t];
                
//                 cudaError_t ce = cudaMemcpyAsync(dst, src, sz, cudaMemcpyDeviceToDevice, ctx->cuda_stream);
//                 if (ce != cudaSuccess) {
//                     ionic_set_error(ctx, (ionic_error_t){ .kind = IONIC_ERROR_CUDA, .res = (int)ce, .what = NULL });
//                     goto loop_end;
//                 }
//             }

//             cudaError_t ce = cudaEventRecord(sl->scatter_done, ctx->cuda_stream);
//             if (ce != cudaSuccess) {
//                 ionic_set_error(ctx, (ionic_error_t){ .kind = IONIC_ERROR_CUDA, .res = (int)ce, .what = NULL });
//                 goto loop_end;
//             }
//             atomic_store_explicit(&sl->state, IONIC_SLOT_SCATTERING, memory_order_release);
//         }

//         for (unsigned s = 0; s < IONIC_STAGING_SLOTS; s++) {
//             struct ionic_staging_slot_cuda *sl = &ctx->cuda_slots[s];
//             if (atomic_load_explicit(&sl->state, memory_order_acquire) != IONIC_SLOT_SCATTERING)
//                 continue;

//             cudaError_t q = cudaEventQuery(sl->scatter_done);
//             if (q == cudaErrorNotReady)
//                 continue;
            
//             if (q != cudaSuccess) {
//                 ionic_set_error(ctx, (ionic_error_t){ .kind = IONIC_ERROR_CUDA, .res = (int)q, .what = NULL });
//                 goto loop_end;
//             }

//             size_t idx = sl->submitted_chunk_index;
//             chunk_done[idx] = 1;
//             atomic_store_explicit(&sl->state, IONIC_SLOT_EMPTY, memory_order_release);
//             sl->bound_chunk = NULL;
//             n_finished++;
//         }

//         while (next_emit < plan->n_chunks && chunk_done[next_emit]) {
//             if (on_ready)
//                 on_ready(ctx, &plan->chunks[next_emit], userdata);
//             next_emit++;
//         }

//         ionic_cpu_relax();
//     }

// loop_end:
//     free(chunk_done);
// }

#else /* !__IONIC_CUDA_ENABLED__ */

void ionic_iouring_cuda_pipeline_destroy(struct ionic_context *ctx) { (void)ctx; }

#endif
