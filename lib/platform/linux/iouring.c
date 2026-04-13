#include <ionic/platform/linux/iouring.h>

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <threads.h>
#include <sys/resource.h>
#include <ionic/utils.h>

#define IONIC_EVENT_TAG_IOENGINE_IOURING "ioengine(iouring)"

static struct ionic_iouring_registered_file *ionic_iouring_engine_find_registered_file(const struct ionic_iouring_engine *engine, const char *path) {
    for (size_t i = 0; i < engine->config.n_files; ++i) {
        if (strcmp(path, engine->files[i].path) == 0) return engine->files + i;
    }

    return NULL;
}

static void ionic_iouring_engine_register_files(struct ionic_context *ctx, struct ionic_iouring_engine *engine) {
    if (ionic_has_error(&ctx->error)) return;

    const struct ionic_iouring_engine_config *config = &engine->config;

    if (config->n_files > 0)
    {
        IONIC_DEBUG(&ctx->logger, IONIC_EVENT_TAG_IOENGINE_IOURING, "registering files=%u", config->n_files);

        struct ionic_iouring_registered_file *files = calloc(config->n_files, sizeof(struct ionic_iouring_registered_file));
        int *fds = calloc(config->n_files, sizeof(struct ionic_iouring_registered_file));
        if (!files || !fds) {
            if (files) free(files);
            if (fds) free(fds);

            ionic_set_error(ctx, IONIC_ERR(IONIC_ERROR_ALLOCATION_FAILED));
            return;
        }

        for (unsigned i = 0; i < config->n_files; ++i) {
            const char *file = engine->config.files[i];
            int fd = open(file, O_RDONLY| O_DIRECT);
            if (fd < 0) {
                struct ionic_error err = IONIC_SYS_ERR_WITH_MSG(-fd, "open failed");
                ionic_set_error(ctx, err);
                IONIC_ERROR(&ctx->logger, IONIC_EVENT_TAG_IOENGINE_IOURING,
                    "open failed path=%s, res=%u (%s)", file, err.res, strerror(err.res));
                for (unsigned j = 0; j < i; ++j) close(fds[j]);
                free(files);
                free(fds);
                return;
            }
            fds[i] = fd;
            files[i] = (struct ionic_iouring_registered_file) { file, fd };
            IONIC_TRACE(&ctx->logger, IONIC_EVENT_TAG_IOENGINE_IOURING, "\t opened file %s, fd=%u", file, fd);
        }

        int res = io_uring_register_files(&engine->ring, fds, config->n_files);
        if (res < 0) {
            struct ionic_error err = IONIC_SYS_ERR_WITH_MSG(res, "io_uring_register_files failed");
            ionic_set_error(ctx, err);
            IONIC_ERROR(&ctx->logger, IONIC_EVENT_TAG_IOENGINE_IOURING,
                "io_uring_register_files failed res=%u (%s)", err.res, strerror(err.res));
            free(files);
            free(fds);
            return;
        }

        engine->files = files;
        free(fds);
    }
}

static void ionic_iouring_engine_register_buffers(struct ionic_context *ctx, struct ionic_iouring_engine *engine) {
    if (ionic_has_error(&ctx->error)) return;

    const struct ionic_iouring_engine_config config = engine->config;
    IONIC_DEBUG(&ctx->logger, IONIC_EVENT_TAG_IOENGINE_IOURING, "registering staging buffers n=%u, size=%ukiB", config.qd, config.st_size / 1024);

    if (!ionic_is_power_of_two(config.qd)) {
        IONIC_ERROR(&ctx->logger, IONIC_EVENT_TAG_IOENGINE_IOURING, "invalid queue depth value %hu, needs to be power of two", config.qd);
        ionic_set_error(ctx, IONIC_ERR_WITH_MSG(IONIC_ERROR_INVALID_VALUE, "qd needs to be power of two"));
        return;
    }

    // one buffer per SQ entry
    void *base = ctx->dalloc.allocate(ctx, config.qd * config.st_size, IONIC_ALLOC_STAGING);
    if (!base) goto ko; // error set by ctx->dalloc.allocate

    engine->iovecs = calloc(config.qd, sizeof(struct iovec));
    if (!engine->iovecs) {
        ionic_set_error(ctx, IONIC_ERR(IONIC_ERROR_ALLOCATION_FAILED));
        return;
    }

    if (((uintptr_t)base & 0xFF) != 0) {
        ionic_set_error(ctx, IONIC_ERR(IONIC_ERROR_ALLOCATION_NOT_ALIGNED));
        goto ko;
    }

    for (unsigned i = 0; i < config.qd; ++i) {
        engine->iovecs[i].iov_base = base + i * config.st_size;
        engine->iovecs[i].iov_len  = config.st_size;
    }

    int res = 0;
    if ((res = io_uring_register_buffers(&engine->ring, engine->iovecs, engine->config.qd))) {
        struct ionic_error err = IONIC_SYS_ERR_WITH_MSG(-res, "io_uring_register_buffers failed");
        ionic_set_error(ctx, err);

        // RLIMIT_MEMLOCK advise
        if (err.res == ENOMEM) {
            struct rlimit rl;
            getrlimit(RLIMIT_MEMLOCK, &rl);
            IONIC_WARN(&ctx->logger, IONIC_EVENT_TAG_IOENGINE_IOURING,
                "io_uring_register_buffers failed with ENOMEM (12) is certainly related to RLIMIT_MEMLOCK set too low "
                "(cur=%lu max=%lu, needed=%zu bytes)", rl.rlim_cur, rl.rlim_max, (size_t)engine->config.qd * 2 * 1024 * 1024
            );
        }

        IONIC_ERROR(&ctx->logger, IONIC_EVENT_TAG_IOENGINE_IOURING, "%s, res=%i (%s)", err.what, err.res, strerror(err.res));
        goto ko;
    }

    IONIC_INFO(&ctx->logger, IONIC_EVENT_TAG_IOENGINE_IOURING, "registered staging buffers n=%u, size=%ukiB", config.qd, config.st_size / 1024);
    return;

ko:
    if (engine->iovecs) free(engine->iovecs);
    if (base) ctx->dalloc.free(ctx, base, IONIC_ALLOC_STAGING);
}


static size_t ionic_iouring_engine_get_sqe_chunks(
    struct ionic_context *ctx, 
    struct ionic_iouring_engine *engine, 
    struct ionic_io_file_segment *segments,
    unsigned count,
    struct ionic_io_fetch_result **results) 
{
    const size_t st_size = engine->config.st_size;

    size_t n_chunks = 0;
    for (unsigned i = 0; i < count; ++i) {
        size_t aligned_from = ionic_align_down_sz(segments[i].from, 4096);
        size_t aligned_to = ionic_align_up_sz(segments[i].to, 4096);
        size_t segment_size = aligned_to - aligned_from;
        n_chunks += (segment_size + st_size - 1) / st_size;
    }

    if (n_chunks == 0) {
        *results = NULL;
        return 0;
    }

    *results = calloc(n_chunks, sizeof(struct ionic_io_fetch_result));
    if (!*results) {
        ionic_set_error(ctx, IONIC_ERR(IONIC_ERROR_ALLOCATION_FAILED));
        IONIC_ERROR(&ctx->logger, IONIC_EVENT_TAG_IOENGINE_IOURING, "failed to allocate sqe chunks");
        return 0;
    }

    size_t chunk_idx = 0;
    for (unsigned seg_idx = 0; seg_idx < count; ++seg_idx) {
        const struct ionic_io_file_segment *seg = &segments[seg_idx];
        size_t aligned_from = ionic_align_down_sz(seg->from, 4096);
        size_t aligned_to = ionic_align_up_sz(seg->to, 4096);
        
        size_t offset = aligned_from;
        while (offset < aligned_to) {
            size_t chunk_end = offset + st_size;
            if (chunk_end > aligned_to) {
                chunk_end = aligned_to;
            }

            struct ionic_io_fetch_result *result = &(*results)[chunk_idx];
            result->fragment.file.from = offset;
            result->fragment.file.to = chunk_end;
            result->fragment.file.path = seg->path;
            result->fragment.len = chunk_end - offset;
            result->fragment.data = seg->dst + (offset - aligned_from);
            result->userdata = NULL;
            
            offset = chunk_end;
            chunk_idx++;
        }
    }

    if (chunk_idx != n_chunks) {
        IONIC_WARN(&ctx->logger, IONIC_EVENT_TAG_IOENGINE_IOURING, 
            "chunk count mismatch: computed=%zu, actual=%zu", n_chunks, chunk_idx);
    }

    IONIC_DEBUG(&ctx->logger, IONIC_EVENT_TAG_IOENGINE_IOURING, 
        "computed sqe chunks: segments=%u, chunks=%zu", count, n_chunks);

    return n_chunks;
}


static void ionic_iouring_engine_destroy(struct ionic_ioengine *engine) {
    struct ionic_iouring_engine *engine_ = (struct ionic_iouring_engine *)engine;
    if (engine_->slot_infos) free(engine_->slot_infos);
}

static void ionic_iouring_engine_init(struct ionic_context *ctx, struct ionic_ioengine *engine) {
    if (ionic_has_error(&ctx->error)) return;

    struct ionic_iouring_engine *engine_ = (struct ionic_iouring_engine *)engine;

    unsigned qd = engine_->config.qd;
    if (qd >= BITS_PER_WORD) {
        engine_->slots[0] = SLOT_ALL_AVAILABLE;
        engine_->slots[1] = qd >= 2 * BITS_PER_WORD ? SLOT_ALL_AVAILABLE : 0;
    } else {
        engine_->slots[0] = (qd == BITS_PER_WORD) ? SLOT_ALL_AVAILABLE : ((1UL << qd) - 1);
        engine_->slots[1] = 0;
    }
    atomic_store_explicit(&engine_->pending[0], 0, memory_order_relaxed);
    atomic_store_explicit(&engine_->pending[1], 0, memory_order_relaxed);
    atomic_store_explicit(&engine_->done[0], 0, memory_order_relaxed);
    atomic_store_explicit(&engine_->done[1], 0, memory_order_relaxed);

    engine_->slot_infos = calloc(engine_->config.qd, sizeof(struct ionic_iouring_slot_info));
    if (!engine_->slot_infos) {
        ionic_set_error(ctx, IONIC_ERR(IONIC_ERROR_ALLOCATION_FAILED));
        return;
    }

    io_uring_queue_init_params(engine_->config.qd, &engine_->ring, &engine_->config.params);
    ionic_iouring_engine_register_files(ctx, engine_);
    ionic_iouring_engine_register_buffers(ctx, engine_);

    IONIC_INFO(&ctx->logger, IONIC_EVENT_TAG_IOENGINE_IOURING, "initialized ring");
}

static size_t ionic_iouring_engine_fetch(
    struct ionic_context *ctx, struct ionic_ioengine *engine, struct ionic_io_file_segment *segments, const unsigned count) {

    if (ionic_has_error(&ctx->error)) return 0;

    struct ionic_iouring_engine *engine_ = (struct ionic_iouring_engine *)engine;
    struct ionic_io_fetch_result *results = NULL;
    const size_t n_chunks = ionic_iouring_engine_get_sqe_chunks(ctx, engine_, segments, count, &results);

    IONIC_DEBUG(&ctx->logger, IONIC_EVENT_TAG_IOENGINE_IOURING, "fetch segments=%u, chunks=%zu", count, n_chunks);

    unsigned long (*slots)[2] = &engine_->slots;
    struct iovec *iovecs = engine_->iovecs;
    struct io_uring_cqe **cqes = calloc(engine_->config.qd, sizeof(struct io_uring_cqe *));
    struct io_uring *ring = &engine_->ring;

    size_t submitted = 0, completed = 0;
    do {
        while (submitted < n_chunks) {
            const int slot = get_available_slot(slots);
            if (slot < 0) break;

            set_slot_busy(slots, slot);

            struct ionic_io_fetch_result res = results[submitted];
            struct ionic_iouring_registered_file *file = ionic_iouring_engine_find_registered_file(engine_, res.fragment.file.path);
            if (!file) {
                ionic_set_error(ctx, IONIC_ERR_WITH_MSG(IONIC_ERROR_IOENGINE_INVALID_PARAMETER, "file not registered"));
                return 0;
            }

            struct iovec dst = iovecs[slot];
            struct io_uring_sqe *sqe = io_uring_get_sqe(ring);
            struct ionic_io_fragment fragment = res.fragment;

            io_uring_prep_read_fixed(sqe, file->fd, dst.iov_base, fragment.len, fragment.file.from, slot);
            sqe->user_data = (unsigned long)slot;

            // engine_->slot_infos[slot] = (struct ionic_iouring_slot_info) {
            //     .dst = fragment.data,
            //     .len = fragment.len,
            // };

            io_uring_submit(ring);
            ++submitted;
        }

        const unsigned n_ready = io_uring_peek_batch_cqe(ring, cqes, engine_->config.qd);
        if (n_ready > 0) {
            completed += n_ready;
            for (unsigned i = 0; i < n_ready; ++i) {
                const unsigned slot = cqes[i]->user_data;
                // struct ionic_iouring_slot_info info = engine_->slot_infos[slot];
                set_slot_available(&engine_->slots, slot);
                io_uring_cqe_seen(ring, cqes[i]);
            }
        }

        // if (submitted == 0) {
        //     struct timespec ts = { .tv_sec = 0, .tv_nsec = 1000 };
        //     io_uring_submit_and_wait(ring, 1);
        // } else {
        //     struct io_uring_cqe *cqe;
        //     int ret = io_uring_wait_cqe(ring, &cqe);
        //     if (ret < 0) {
        //         if (ret == -EINTR) continue;
        //         ionic_cpu_relax();
        //         continue;
        //     }
        //
        //     unsigned slot_ = (unsigned)cqe->user_data;
        //     if (slot_ < engine_->config.qd) {
        //         unsigned w = slot_ / BITS_PER_WORD;
        //         unsigned bit = slot_ % BITS_PER_WORD;
        //         atomic_fetch_or_explicit(&engine_->pending[w], 1UL << bit, memory_order_release);
        //     }
        //     io_uring_cqe_seen(ring, cqe);
        // }

        // for (int w = 0; w < 2; ++w) {
        //     unsigned long bits = atomic_load_explicit(&engine_->done[w], memory_order_acquire);
        //     while (bits) {
        //         unsigned bit = __builtin_ctzl(bits);
        //         unsigned slot_ = w * BITS_PER_WORD + bit;
        //         set_slot_available(slots, slot_);
        //         atomic_fetch_and_explicit(&engine_->done[w], ~(1UL << bit), memory_order_release);
        //         bits = atomic_load_explicit(&engine_->done[w], memory_order_acquire);
        //         ++completed;
        //     }
        // }

    } while (completed < n_chunks);

    if (results) free(results);
    return completed;
}

static void ionic_iouring_engine_mark_done(struct ionic_ioengine *engine, struct ionic_io_fetch_result *res) {
    struct ionic_iouring_engine *engine_ = (struct ionic_iouring_engine *)engine;
    unsigned slot = (unsigned)(uintptr_t)res->userdata;
    unsigned w = slot / BITS_PER_WORD;
    unsigned bit = slot % BITS_PER_WORD;
    atomic_fetch_or_explicit(&engine_->done[w], 1UL << bit, memory_order_release);
}

static void ionic_iouring_engine_probe_ring(struct ionic_context *ctx, struct io_uring_params *params, unsigned qd) {
    struct io_uring ring;
    int res = 0;

    params->flags |= IORING_SETUP_SQPOLL;
    if ((res = io_uring_queue_init_params(qd, &ring, params))) {
        struct ionic_error err = IONIC_SYS_ERR_WITH_MSG(-res, "io_uring_queue_init_params failed");
        ionic_set_error(ctx, err);
        IONIC_ERROR(&ctx->logger, IONIC_EVENT_TAG_IOENGINE_IOURING, "io_uring_queue_init failed flags=IORING_SETUP_SQPOLL res=%i (%s)", err.res, strerror(err.res));
        return;
    }

    struct io_uring_probe *probe = io_uring_get_probe_ring(&ring);
    if (!probe) {
        return;
    }

    if (!io_uring_opcode_supported(probe, IORING_OP_READ)) {
        struct ionic_error err = IONIC_ERR_WITH_MSG(IONIC_ERROR_UNSUPPORTED, "probe failed feature=IORING_OP_READ not supported");
        ionic_set_error(ctx, err);
        IONIC_ERROR(&ctx->logger, IONIC_EVENT_TAG_IOENGINE_IOURING, err.what);
        io_uring_free_probe(probe);
        return;
    }

    if (!io_uring_opcode_supported(probe, IORING_OP_READ_FIXED)) {
        struct ionic_error err = IONIC_ERR_WITH_MSG(IONIC_ERROR_UNSUPPORTED, "probe failed feature=IORING_OP_READ_FIXED not supported");
        ionic_set_error(ctx, err);
        IONIC_ERROR(&ctx->logger, IONIC_EVENT_TAG_IOENGINE_IOURING, err.what);
        io_uring_free_probe(probe);
        return;
    }

    io_uring_free_probe(probe);
}

struct ionic_iouring_engine *ionic_iouring_engine_create(struct ionic_context *ctx, struct ionic_iouring_engine_config *config) {
    if (ionic_has_error(&ctx->error)) goto ko;
    IONIC_INFO(&ctx->logger, IONIC_EVENT_TAG_IOENGINE_IOURING, "create qd=%hu", config->qd);

    ionic_iouring_engine_probe_ring(ctx, &config->params, config->qd);
    if (ionic_has_error(&ctx->error)) goto ko;

    struct ionic_iouring_engine *engine = malloc(sizeof(struct ionic_iouring_engine));
    engine->config          = *config;
    engine->base.initialize = ionic_iouring_engine_init;
    engine->base.destroy    = ionic_iouring_engine_destroy;
    engine->base.fetch      = ionic_iouring_engine_fetch;
    engine->base.mark_done  = ionic_iouring_engine_mark_done;

    return engine;
ko:
    return NULL;
}
