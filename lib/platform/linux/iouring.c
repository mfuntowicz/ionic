#include <ionic/platform/linux/iouring.h>

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/resource.h>
#include <ionic/utils.h>

#define IONIC_EVENT_TAG_IOENGINE_IOURING "ioengine(iouring)"

static void atomic_set_bit(atomic_ulong *bits, unsigned slot) {
    atomic_fetch_or_explicit(bits, 1UL << slot, memory_order_acq_rel);
}

static void atomic_clear_bit(atomic_ulong *bits, unsigned slot) {
    atomic_fetch_and_explicit(bits, ~(1UL << slot), memory_order_acq_rel);
}

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

        IONIC_INFO(&ctx->logger, IONIC_EVENT_TAG_IOENGINE_IOURING, "registered files n=%u", config->n_files);
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

    void *base = ctx->dalloc.allocate(ctx, config.qd * config.st_size, IONIC_ALLOC_STAGING);
    if (!base) goto ko;

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


struct ionic_chunk_key {
    const char *path;
    size_t offset;
};

static int ionic_chunk_key_compare(const void *a, const void *b) {
    const struct ionic_chunk_key *ka = a, *kb = b;
    int cmp = strcmp(ka->path, kb->path);
    if (cmp != 0) return cmp;
    if (ka->offset < kb->offset) return -1;
    if (ka->offset > kb->offset) return 1;
    return 0;
}

static ssize_t ionic_chunk_key_bsearch(const struct ionic_chunk_key *keys, size_t n, const char *path, size_t offset) {
    size_t lo = 0, hi = n;
    while (lo < hi) {
        size_t mid = lo + (hi - lo) / 2;
        int cmp = strcmp(keys[mid].path, path);
        if (cmp < 0 || (cmp == 0 && keys[mid].offset < offset))
            lo = mid + 1;
        else if (cmp > 0 || (cmp == 0 && keys[mid].offset > offset))
            hi = mid;
        else
            return (ssize_t)mid;
    }
    return -1;
}

static size_t ionic_iouring_engine_get_seq_chunks(
    struct ionic_context *ctx,
    struct ionic_iouring_engine *engine,
    struct ionic_logical_segment *segments,
    size_t count,
    struct ionic_io_fetch_result **out)
{
    const size_t st_size = engine->config.st_size;

    if (count == 0) {
        *out = NULL;
        return 0;
    }

    size_t max_keys = 0;
    for (size_t i = 0; i < count; ++i) {
        size_t aligned_from = ionic_align_down_sz(segments[i].from, st_size);
        size_t aligned_to = ionic_align_up_sz(segments[i].to, st_size);
        max_keys += (aligned_to - aligned_from + st_size - 1) / st_size;
    }

    if (max_keys == 0) {
        *out = NULL;
        return 0;
    }

    struct ionic_chunk_key *keys = calloc(max_keys, sizeof(*keys));
    if (!keys) {
        ionic_set_error(ctx, IONIC_ERR(IONIC_ERROR_ALLOCATION_FAILED));
        return 0;
    }

    size_t n_keys = 0;
    for (size_t i = 0; i < count; ++i) {
        size_t aligned_from = ionic_align_down_sz(segments[i].from, st_size);
        size_t aligned_to = ionic_align_up_sz(segments[i].to, st_size);
        for (size_t offset = aligned_from; offset < aligned_to; offset += st_size) {
            keys[n_keys++] = (struct ionic_chunk_key){ segments[i].path, offset };
        }
    }

    qsort(keys, n_keys, sizeof(*keys), ionic_chunk_key_compare);

    size_t n_unique = 0;
    for (size_t i = 0; i < n_keys; ++i) {
        if (n_unique == 0 || ionic_chunk_key_compare(&keys[i], &keys[n_unique - 1]) != 0) {
            keys[n_unique++] = keys[i];
        }
    }

    *out = calloc(n_unique, sizeof(struct ionic_io_fetch_result));
    if (!*out) {
        ionic_set_error(ctx, IONIC_ERR(IONIC_ERROR_ALLOCATION_FAILED));
        free(keys);
        return 0;
    }

    for (size_t c = 0; c < n_unique; ++c) {
        (*out)[c] = (struct ionic_io_fetch_result){
            .path = keys[c].path,
            .offset = keys[c].offset,
            .len = st_size,
            .entries = NULL,
            .n_entries = 0,
            .userdata = ULONG_MAX,
        };
    }

    size_t *entry_counts = calloc(n_unique, sizeof(size_t));
    if (!entry_counts) {
        ionic_set_error(ctx, IONIC_ERR(IONIC_ERROR_ALLOCATION_FAILED));
        for (size_t c = 0; c < n_unique; ++c) free((*out)[c].entries);
        free(*out);
        *out = NULL;
        free(keys);
        return 0;
    }

    for (size_t i = 0; i < count; ++i) {
        const struct ionic_logical_segment *seg = &segments[i];
        size_t aligned_from = ionic_align_down_sz(seg->from, st_size);
        size_t aligned_to = ionic_align_up_sz(seg->to, st_size);

        for (size_t offset = aligned_from; offset < aligned_to; offset += st_size) {
            ssize_t idx = ionic_chunk_key_bsearch(keys, n_unique, seg->path, offset);
            if (idx < 0) continue;

            size_t overlap_start = seg->from > offset ? seg->from : offset;
            size_t overlap_end = seg->to < (offset + st_size) ? seg->to : (offset + st_size);
            if (overlap_start >= overlap_end) continue;

            entry_counts[idx]++;
        }
    }

    for (size_t c = 0; c < n_unique; ++c) {
        if (entry_counts[c] > 0) {
            (*out)[c].entries = calloc(entry_counts[c], sizeof(struct ionic_scatter_entry));
            (*out)[c].n_entries = entry_counts[c];
            if (!(*out)[c].entries) {
                ionic_set_error(ctx, IONIC_ERR(IONIC_ERROR_ALLOCATION_FAILED));
                for (size_t j = 0; j < c; ++j) free((*out)[j].entries);
                free(*out);
                *out = NULL;
                free(entry_counts);
                free(keys);
                return 0;
            }
        }
    }

    size_t *fill_idx = calloc(n_unique, sizeof(size_t));
    if (!fill_idx) {
        ionic_set_error(ctx, IONIC_ERR(IONIC_ERROR_ALLOCATION_FAILED));
        for (size_t c = 0; c < n_unique; ++c) free((*out)[c].entries);
        free(*out);
        *out = NULL;
        free(entry_counts);
        free(keys);
        return 0;
    }

    for (size_t i = 0; i < count; ++i) {
        const struct ionic_logical_segment *seg = &segments[i];
        size_t aligned_from = ionic_align_down_sz(seg->from, st_size);
        size_t aligned_to = ionic_align_up_sz(seg->to, st_size);

        for (size_t offset = aligned_from; offset < aligned_to; offset += st_size) {
            ssize_t idx = ionic_chunk_key_bsearch(keys, n_unique, seg->path, offset);
            if (idx < 0) continue;

            size_t overlap_start = seg->from > offset ? seg->from : offset;
            size_t overlap_end = seg->to < (offset + st_size) ? seg->to : (offset + st_size);
            if (overlap_start >= overlap_end) continue;

            struct ionic_io_fetch_result *result = &(*out)[idx];
            size_t ei = fill_idx[idx]++;
            result->entries[ei] = (struct ionic_scatter_entry){
                .staging_offset = overlap_start - offset,
                .len = overlap_end - overlap_start,
                .dst = seg->dst ? (char *)seg->dst + (overlap_start - seg->from) : NULL,
                .userdata = seg->userdata,
            };
        }
    }

    IONIC_DEBUG(&ctx->logger, IONIC_EVENT_TAG_IOENGINE_IOURING,
        "computed seq chunks: segments=%zu, chunks=%zu", count, n_unique);

    free(fill_idx);
    free(entry_counts);
    free(keys);
    return n_unique;
}


static void ionic_iouring_engine_destroy(struct ionic_ioengine *engine) {
    struct ionic_iouring_engine *engine_ = (struct ionic_iouring_engine *)engine;
    if (engine_->results) {
        for (size_t i = 0; i < engine_->config.qd; ++i)
            if (engine_->results[i].entries) free(engine_->results[i].entries);
        free(engine_->results);
    }
}

static void ionic_iouring_engine_init(struct ionic_context *ctx, struct ionic_ioengine *engine) {
    if (ionic_has_error(&ctx->error)) return;

    struct ionic_iouring_engine *engine_ = (struct ionic_iouring_engine *)engine;

    if (engine_->config.qd == 0) {
        ionic_set_error(ctx, IONIC_ERR(IONIC_ERROR_INVALID_VALUE));
        IONIC_ERROR(&ctx->logger, IONIC_EVENT_TAG_IOENGINE_IOURING, "invalid queue depth == 0 (got %u)", engine_->config.qd);
        return;
    }

    if (engine_->config.qd > 64) {
        ionic_set_error(ctx, IONIC_ERR(IONIC_ERROR_INVALID_VALUE));
        IONIC_ERROR(&ctx->logger, IONIC_EVENT_TAG_IOENGINE_IOURING, "invalid queue depth should < 64 (got %u)", engine_->config.qd);
        return;
    }

    if (__builtin_popcount(engine_->config.qd) > 1) {
        ionic_set_error(ctx, IONIC_ERR(IONIC_ERROR_INVALID_VALUE));
        IONIC_ERROR(&ctx->logger, IONIC_EVENT_TAG_IOENGINE_IOURING, "invalid queue depth should be a power of 2 (got %u)", engine_->config.qd);
        return;
    }

    engine_->slots = ULONG_MAX; // TODO(mfuntowicz): use a mask to the number of actual slots from config->qd
    atomic_store_explicit(&engine_->pending, 0, memory_order_relaxed);
    atomic_store_explicit(&engine_->done, 0, memory_order_relaxed);

    engine_->results = calloc(engine_->config.qd, sizeof(struct ionic_io_fetch_result));
    if (!engine_->results) {
        ionic_set_error(ctx, IONIC_ERR(IONIC_ERROR_ALLOCATION_FAILED));
        return;
    }

    io_uring_queue_init_params(engine_->config.qd, &engine_->ring, &engine_->config.params);
    ionic_iouring_engine_register_files(ctx, engine_);
    ionic_iouring_engine_register_buffers(ctx, engine_);

    IONIC_INFO(&ctx->logger, IONIC_EVENT_TAG_IOENGINE_IOURING, "initialized ring");
}

static size_t ionic_iouring_engine_fetch(
    struct ionic_context *ctx, struct ionic_ioengine *engine, struct ionic_logical_segment *segments, const size_t count) {

    if (ionic_has_error(&ctx->error)) return 0;

    struct ionic_iouring_engine *engine_ = (struct ionic_iouring_engine *)engine;
    struct ionic_io_fetch_result *results;
    const size_t n_chunks = ionic_iouring_engine_get_seq_chunks(ctx, engine_, segments, count, &results);

    if (ionic_has_error(&ctx->error)) return 0;

    IONIC_DEBUG(&ctx->logger, IONIC_EVENT_TAG_IOENGINE_IOURING, "fetch segments=%zu, chunks=%zu", count, n_chunks);

    unsigned long *slots = &engine_->slots;
    struct iovec *iovecs = engine_->iovecs;
    struct io_uring_cqe **cqes = calloc(engine_->config.qd, sizeof(struct io_uring_cqe *));
    struct io_uring *ring = &engine_->ring;

    size_t submitted = 0, completed = 0;
    do {
        unsigned char has_new_entries = 0;
        while (submitted < n_chunks) {
            const int slot = get_slot_available(slots);
            if (slot < 0) break;

            struct io_uring_sqe *sqe = io_uring_get_sqe(ring);
            if (!sqe) break;

            mark_slot_busy(slots, slot);

            struct iovec dst = iovecs[slot];
            struct ionic_io_fetch_result *res = results + submitted;
            struct ionic_iouring_registered_file *file = ionic_iouring_engine_find_registered_file(engine_, res->path);
            if (!file) {
                ionic_set_error(ctx, IONIC_ERR_WITH_MSG(IONIC_ERROR_IOENGINE_INVALID_PARAMETER, "file not registered"));
                goto cleanup;
            }

            sqe->user_data = (unsigned long)slot;
            res->userdata = (unsigned long)slot;
            res->data = dst.iov_base;
            if (engine_->results[slot].entries) {
                free(engine_->results[slot].entries);
            }
            engine_->results[slot] = *res;

            io_uring_prep_read_fixed(sqe, file->fd, dst.iov_base, res->len, res->offset, slot);
            ++submitted;
            has_new_entries = 1;
        }

        if (has_new_entries)
            io_uring_submit(ring);

        const unsigned n_ready = io_uring_peek_batch_cqe(ring, cqes, engine_->config.qd);
        if (n_ready > 0) {
            completed += n_ready;
            for (unsigned i = 0; i < n_ready; ++i) {
                struct io_uring_cqe *cqe = cqes[i];
                const unsigned long slot = cqe->user_data;

                if (slot < engine_->config.qd)
                    atomic_set_bit(&engine_->pending, slot); // inform dma worker some are pending

                io_uring_cqe_seen(ring, cqe);  // release iouring cqe marking for reuse
            }
        }

        unsigned long ready;
        while ((ready = atomic_load_explicit(&engine_->done, memory_order_acquire))) {
            if (has_slot_available(ready)) {
                unsigned long slot = get_slot_available(&ready);
                atomic_clear_bit(&engine_->done, slot);
                set_slot_available(slots, slot);
            }
        }
    } while (completed < n_chunks);

cleanup:
    if (cqes) free(cqes);
    if (results) {
        for (size_t i = submitted; i < n_chunks; ++i)
            if (results[i].entries) free(results[i].entries);
        free(results);
    }

    return completed;
}

static size_t ionic_iouring_engine_peek(struct ionic_context *ctx, struct ionic_ioengine *engine, struct ionic_io_fetch_result **res, size_t count) {
    struct ionic_iouring_engine *engine_ = (struct ionic_iouring_engine *)engine;
    unsigned long pending = atomic_load_explicit(&engine_->pending, memory_order_acquire);;

    if (pending == 0) return 0;

    size_t filled = 0;
    while (pending && filled < count) {
        int slot = get_slot_available(&pending);
        res[filled] = &engine_->results[slot];

        atomic_clear_bit(&engine_->pending, slot);
        pending = atomic_load_explicit(&engine_->pending, memory_order_acquire);
        ++filled;
    }

    return filled;
}

static void ionic_iouring_engine_mark_done(struct ionic_context *ctx, struct ionic_ioengine *engine, const struct ionic_io_fetch_result *res) {
    struct ionic_iouring_engine *engine_ = (struct ionic_iouring_engine *)engine;
    const unsigned long slot = (unsigned long)res->userdata;
    atomic_set_bit(&engine_->done, slot);

    IONIC_DEBUG(&ctx->logger, IONIC_EVENT_TAG_IOENGINE_IOURING, "slot done slot=%u", slot);
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
    if (!probe)
        return;

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
    engine->config             = *config;
    engine->base.initialize    = ionic_iouring_engine_init;
    engine->base.destroy       = ionic_iouring_engine_destroy;
    engine->base.fetch         = ionic_iouring_engine_fetch;
    engine->base.peek          = ionic_iouring_engine_peek;
    engine->base.mark_done     = ionic_iouring_engine_mark_done;

    return engine;
ko:
    return NULL;
}
