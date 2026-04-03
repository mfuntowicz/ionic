#include <ionic/platform/linux/iouring.h>

#include <stdlib.h>
#include <string.h>
#include <sys/resource.h>
#include <ionic/utils.h>

#define IONIC_EVENT_TAG_IOENGINE_IOURING "ioengine(iouring)"

static void ionic_iouring_engine_register_files(struct ionic_context *ctx, struct ionic_iouring_engine *engine) {
    if (ionic_has_error(&ctx->error)) return;

    const struct ionic_iouring_engine_config *config = &engine->config;

    if (config->n_files > 0)
    {
        IONIC_DEBUG(&ctx->logger, IONIC_EVENT_TAG_IOENGINE_IOURING, "registering files=%u", config->n_files);

        int *fds = calloc(config->n_files, sizeof(int));
        if (!fds) {
            ionic_set_error(ctx, IONIC_ERR(IONIC_ERROR_ALLOCATION_FAILED));
            return;
        }

        for (unsigned i = 0; i < config->n_files; ++i) {
            const char *file = engine->config.files + i;
            int fd = open(file, O_RDONLY| O_DIRECT);
            if (fd < 0) {
                struct ionic_error err = IONIC_SYS_ERR_WITH_MSG(-fd, "open failed");
                ionic_set_error(ctx, err);
                IONIC_ERROR(&ctx->logger, IONIC_EVENT_TAG_IOENGINE_IOURING, "open failed path=%s, res=%u (%s)", file, err.res, strerror(err.res));
                goto ko;
            }
            fds[i] = fd;
            IONIC_TRACE(&ctx->logger, IONIC_EVENT_TAG_IOENGINE_IOURING, "\t opened file %s, fd=%u", file, fd);
        }

        io_uring_register_files(&engine->ring, fds, config->n_files);
ko:
        free(fds);
    }
}

static void ionic_iouring_engine_register_buffers(struct ionic_context *ctx, struct ionic_iouring_engine *engine) {
    if (ionic_has_error(&ctx->error)) return;

    const struct ionic_iouring_engine_config *config = &engine->config;
    IONIC_DEBUG(&ctx->logger, IONIC_EVENT_TAG_IOENGINE_IOURING, "registering staging buffers n=%u, size=%ukiB", config->qd, 2 * 1024);

    if (!ionic_is_power_of_two(config->qd)) {
        IONIC_ERROR(&ctx->logger, IONIC_EVENT_TAG_IOENGINE_IOURING, "invalid queue depth value %hu, needs to be power of two", config->qd);
        ionic_set_error(ctx, IONIC_ERR_WITH_MSG(IONIC_ERROR_INVALID_VALUE, "qd needs to be power of two"));
        return;
    }

    // one buffer per SQ entry
    void *base = ctx->dalloc.allocate(ctx, config->qd * 2 * 1024 * 1024, IONIC_ALLOC_STAGING);
    if (!base) goto ko; // error set by ctx->dalloc.allocate

    engine->iovecs = calloc(config->qd, sizeof(struct iovec));
    if (!engine->iovecs) {
        ionic_set_error(ctx, IONIC_ERR(IONIC_ERROR_ALLOCATION_FAILED));
        return;
    }

    for (unsigned i = 0; i < config->qd; ++i) {
        engine->iovecs[i].iov_base = base + i * 2 * 1024 * 1024; //todo(mfuntowicz): configure with env variable
        engine->iovecs[i].iov_len  = 2 * 1024 * 1024;
    if (((uintptr_t)base & 0xFF) != 0) {
        ionic_set_error(ctx, IONIC_ERR(IONIC_ERROR_ALLOCATION_NOT_ALIGNED));
        goto ko;
    }
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

    IONIC_INFO(&ctx->logger, IONIC_EVENT_TAG_IOENGINE_IOURING, "registered staging buffers n=%u, size=%ukiB", config->qd, 2 * 1024);
    return;

ko:
    if (engine->iovecs) free(engine->iovecs);
    if (base) ctx->dalloc.free(ctx, base, IONIC_ALLOC_STAGING);
}

static void ionic_iouring_engine_destroy(struct ionic_ioengine *engine) {
    struct ionic_iouring_engine *engine_ = (struct ionic_iouring_engine *)engine;
}

static void ionic_iouring_engine_init(struct ionic_context *ctx, struct ionic_ioengine *engine) {
    if (ionic_has_error(&ctx->error)) return;

    struct ionic_iouring_engine *engine_ = (struct ionic_iouring_engine *)engine;

    engine_->slots[0] = engine_->slots[1] = SLOT_ALL_AVAILABLE;

    ionic_iouring_engine_register_files(ctx, engine_);
    ionic_iouring_engine_register_buffers(ctx, engine_);

    IONIC_INFO(&ctx->logger, IONIC_EVENT_TAG_IOENGINE_IOURING, "initialized ring");
}

static void ionic_iouring_engine_fetch(
    struct ionic_context *ctx, struct ionic_ioengine *engine, struct ionic_io_fragment *segments, const unsigned count) {

    if (ionic_has_error(&ctx->error)) return;
    IONIC_DEBUG(&ctx->logger, IONIC_EVENT_TAG_IOENGINE_IOURING, "fetch count=%u", count);


}

static void ionic_iouring_engine_probe_ring(struct ionic_context *ctx, struct io_uring_params *params, unsigned qd) {
    struct io_uring ring;
    int res = 0;

    params->flags |= IORING_SETUP_SQPOLL | IORING_SETUP_IOPOLL;
    if ((res = io_uring_queue_init_params(qd, &ring, params))) {
        IONIC_WARN(&ctx->logger, IONIC_EVENT_TAG_IOENGINE_IOURING, "io_uring_queue_init failed flags=IORING_SETUP_SQPOLL|IORING_SETUP_IOPOLL res=%i (%s)", -res, strerror(-res));

        params->flags = IORING_SETUP_SQPOLL;
        if ((res = io_uring_queue_init_params(qd, &ring, params))) {
            struct ionic_error err = IONIC_SYS_ERR_WITH_MSG(-res, "io_uring_queue_init_params failed");
            ionic_set_error(ctx, err);
            IONIC_ERROR(&ctx->logger, IONIC_EVENT_TAG_IOENGINE_IOURING, "io_uring_queue_init failed flags=IORING_SETUP_SQPOLL res=%i (%s)", err.res, strerror(err.res));
            return;
        }
    }

    IONIC_DEBUG(&ctx->logger, IONIC_EVENT_TAG_IOENGINE_IOURING, "ioring created flags=%u", params->flags);

    struct io_uring_probe *probe = io_uring_get_probe_ring(&ring);
    if (!io_uring_opcode_supported(probe, IORING_OP_READ)) {
        struct ionic_error err = IONIC_ERR_WITH_MSG(IONIC_ERROR_UNSUPPORTED, "probe failed feature=IORING_OP_READ not supported");
        ionic_set_error(ctx, err);
        IONIC_ERROR(&ctx->logger, IONIC_EVENT_TAG_IOENGINE_IOURING, err.what);
        return;
    }

    if (!io_uring_opcode_supported(probe, IORING_OP_READ_FIXED)) {
        struct ionic_error err = IONIC_ERR_WITH_MSG(IONIC_ERROR_UNSUPPORTED, "probe failed feature=IORING_OP_READ_FIXED not supported");
        ionic_set_error(ctx, err);
        IONIC_ERROR(&ctx->logger, IONIC_EVENT_TAG_IOENGINE_IOURING, err.what);
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

    return engine;
ko:
    return NULL;
}