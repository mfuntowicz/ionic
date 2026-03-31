#include <ionic/platform/linux/iouring.h>

#include <stdlib.h>
#include <sys/mman.h>

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
                IONIC_ERROR(&ctx->logger, IONIC_EVENT_TAG_IOENGINE_IOURING, "open failed path=%s, res=%u", file, -fd);
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

    if (config->qd % 2 != 0) {
        IONIC_ERROR(&ctx->logger, IONIC_EVENT_TAG_IOENGINE_IOURING, "invalid queue depth value %hu, needs to be power of two", config->qd);
        ionic_set_error(ctx, IONIC_ERR_WITH_MSG(IONIC_ERROR_INVALID_VALUE, "qd needs to be power of two"));
        return;
    }

    struct iovec *iovecs = calloc(config->qd, sizeof(struct iovec));
    if (!iovecs) {
        ionic_set_error(ctx, IONIC_ERR(IONIC_ERROR_ALLOCATION_FAILED));
        return;
    }

    // one buffer per SQ entry
    void *base = ctx->dalloc.allocate(ctx, config->qd * 2 * 1024 * 1024, IONIC_ALLOC_STAGING);
    if (!base) goto ko; // error set by ctx->dalloc.allocate

    // register the buffer ring to work along with IORING_OP_PROVIDE_BUFFERS
    struct io_uring_buf_ring *bring = mmap(NULL, sizeof(struct io_uring_buf_ring), PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (bring == MAP_FAILED) goto ko;

    IONIC_DEBUG(&ctx->logger, IONIC_EVENT_TAG_IOENGINE_IOURING, "allocated io_uring_buf_ring ptr=%p", bring);

    struct io_uring_buf_reg breg = {
        .ring_addr    = (__u64)bring,
        .ring_entries = config->qd,
        .bgid         = ctx->device.ordinal,  // buffer group id — you pick this
    };

    io_uring_buf_ring_init(bring);
    int res = io_uring_register_buf_ring(&engine->ring, &breg, 0);
    if (res < 0) {
        struct ionic_error err = IONIC_SYS_ERR_WITH_MSG(-res, "io_uring_register_buf_ring failed");
        IONIC_ERROR(&ctx->logger, IONIC_EVENT_TAG_IOENGINE_IOURING, "%s res=%u", err.what, -res);
        ionic_set_error(ctx, err);
        goto ko_munmap;
    }

    IONIC_DEBUG(&ctx->logger, IONIC_EVENT_TAG_IOENGINE_IOURING, "registered io_uring_buf_ring ptr=%p", bring);

    int mask = io_uring_buf_ring_mask(config->qd);
    for (unsigned i = 0; i < config->qd; ++i) {
        iovecs[i].iov_base = base + i * 2 * 1024 * 1024;
        iovecs[i].iov_len  = 2 * 1024 * 1024;
        io_uring_buf_ring_add(bring, iovecs[i].iov_base, iovecs[i].iov_len, i, mask, i);
    }

    io_uring_buf_ring_advance(bring, config->qd); // commit all at once
    engine->iovecs = iovecs;
    engine->bring  = bring;

    IONIC_INFO(&ctx->logger, IONIC_EVENT_TAG_IOENGINE_IOURING, "registered staging buffers n=%u, size=%ukiB", config->qd, 2 * 1024);
    return;

ko_munmap:
    munmap(bring, sizeof(struct io_uring_buf_ring));
ko:
    free(iovecs);
}

static void ionic_iouring_engine_probe_ring(
    struct ionic_context *ctx, struct ionic_iouring_engine *engine, struct io_uring_params *params, unsigned qd) {
    int res = 0;

    params->flags |= IORING_SETUP_SQPOLL | IORING_SETUP_IOPOLL;
    if ((res = io_uring_queue_init_params(qd, &engine->ring, params)) < 0) {
        IONIC_WARN(&ctx->logger, IONIC_EVENT_TAG_IOENGINE_IOURING, "io_uring_queue_init failed flags=IORING_SETUP_SQPOLL|IORING_SETUP_IOPOLL (res=%i)", -res);

        params->flags = IORING_SETUP_SQPOLL;
        if ((res = io_uring_queue_init_params(qd, &engine->ring, params)) < 0) {
            IONIC_ERROR(&ctx->logger, IONIC_EVENT_TAG_IOENGINE_IOURING, "io_uring_queue_init failed flags=IORING_SETUP_SQPOLL (res=%i)", -res);
            ionic_set_error(ctx, IONIC_SYS_ERR(-res));
            return;
        }
    }

    IONIC_DEBUG(&ctx->logger, IONIC_EVENT_TAG_IOENGINE_IOURING, "ioring created flags=%u", params->flags);

    struct io_uring_probe *probe = io_uring_get_probe_ring(&engine->ring);
    if (!io_uring_opcode_supported(probe, IORING_OP_PROVIDE_BUFFERS)) {
        struct ionic_error err = IONIC_ERR_WITH_MSG(IONIC_ERROR_UNSUPPORTED, "probe failed feature=IORING_OP_PROVIDE_BUFFERS not supported");
        IONIC_ERROR(&ctx->logger, IONIC_EVENT_TAG_IOENGINE_IOURING, err.what);
        ionic_set_error(ctx, err);
        return;
    }

    io_uring_free_probe(probe);

    ionic_iouring_engine_register_files(ctx, engine);
    ionic_iouring_engine_register_buffers(ctx, engine);
}

struct ionic_iouring_engine *ionic_iouring_engine_create(struct ionic_context *ctx, struct ionic_iouring_engine_config *config) {
    if (ionic_has_error(&ctx->error)) goto ko;
    IONIC_INFO(&ctx->logger, IONIC_EVENT_TAG_IOENGINE_IOURING, "create qd=%hu", config->qd);

    struct ionic_iouring_engine *engine = malloc(sizeof(struct ionic_iouring_engine));
    engine->config = *config;

    ionic_iouring_engine_probe_ring(ctx, engine, &config->params, config->qd);
    return engine;
ko:
    return NULL;
}