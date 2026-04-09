#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <threads.h>
#include <ionic/engine.h>
#include <ionic/pipelines/cuda.h>
#include <ionic/logging.h>
#include "ionic/utils.h"

#ifdef __linux__
#include <ionic/platform/linux/iouring.h>
#endif

static void ionic_pipeline_cuda_allocate_plan_memory(
    struct ionic_context *ctx, const struct ionic_pipeline_cuda *pipeline, const struct ionic_sharding_plan *plan, unsigned short rank) {

    size_t total = 0;
    for (unsigned i = 0; i < plan->n; ++i) {
        const struct ionic_sharded_tensor target = plan->tensors[i];
        const size_t nbytes = ionic_tensor_nbytes(target.tensor);

        if ((target.specs->dst = ctx->dalloc.allocate(ctx, nbytes, IONIC_ALLOC_DEVICE)) == NULL) {
            ionic_set_error(ctx, IONIC_ERR(IONIC_ERROR_ALLOCATION_FAILED));
            IONIC_ERROR(&ctx->logger, pipeline->tag, "destination memory allocation failed");

            //todo(mfuntowicz): clean memory
            return;
        }

        total += nbytes;
    }

    IONIC_DEBUG(&ctx->logger, pipeline->tag,
        "allocated destinations memory n=%zu, size=%zu (%.4f GiB)", plan->n, total, (float)total / 1024.0 / 1024.0 / 1024.0);
}

static void *ionic_pipeline_probe_ioengine(struct ionic_context *ctx, struct ionic_pipeline *pipeline)
{
    struct ionic_pipeline_cuda *pipeline_ = (struct ionic_pipeline_cuda *)pipeline;
    IONIC_TRACE(&ctx->logger, pipeline_->tag, "probing ioengine");
#ifdef __linux__
    struct ionic_iouring_engine_config p = { .qd = 32, .st_size = 2 * 1024 * 1024, .files = pipeline->files, .n_files = pipeline->n_files}; // todo(mfuntowicz): move to iouring + override with envvar
    return ionic_iouring_engine_create(ctx, &p);
#else
    ionic_set_error(ctx, IONIC_ERR_WITH_MSG(IONIC_ERROR_UNSUPPORTED, "platform not supported yet."))
#endif
    return NULL;
}

static void ionic_pipeline_cuda_destroy(struct ionic_pipeline *pipeline) {
    if (!pipeline) return;

    struct ionic_pipeline_cuda *pipeline_ = (struct ionic_pipeline_cuda *)pipeline;

    if (pipeline_->events) {
        for (unsigned char i = 0; i < pipeline_->concurrency; ++i) {
            cudaEventSynchronize(pipeline_->events[i]);
            cudaEventDestroy(pipeline_->events[i]);
        }

        free(pipeline_->events);
        pipeline_->events = NULL;
    }

    if (pipeline_->streams) {
        for (unsigned char i = 0; i < pipeline_->concurrency; ++i) {
            cudaStreamSynchronize(pipeline_->streams[i]);
            cudaStreamDestroy(pipeline_->streams[i]);
        }

        free(pipeline_->streams);
        pipeline_->streams = NULL;
    }

    if (pipeline_->ioengine) {
        pipeline_->ioengine->destroy(pipeline_->ioengine);
        pipeline_->ioengine = NULL;
    }
}

static void ionic_pipeline_cuda_initialize(struct ionic_context *ctx, struct ionic_pipeline *pipeline) {
    if (ionic_has_error(&ctx->error)) return;

    struct ionic_pipeline_cuda *pipeline_ = (struct ionic_pipeline_cuda *)pipeline;

    const char *threshold_env = getenv("IONIC_COALESCED_READ_THRESHOLD");
    if (threshold_env) {
        char *end;
        const unsigned long threshold = strtoul(threshold_env, &end, 10);
        if (end != threshold_env && threshold > 0)
            pipeline_->threshold = threshold;
        else
            pipeline_->threshold = IONIC_COALESCED_READ_THRESHOLD_DEFAULT;
    } else {
        pipeline_->threshold = IONIC_COALESCED_READ_THRESHOLD_DEFAULT;
    }

    enum cudaError cuErr;
    if ((cuErr = cudaSetDevice(ctx->device.ordinal)) != cudaSuccess) {
        struct ionic_error err = IONIC_CUDA_ERR_WITH_MSG(cuErr, cudaGetErrorString(cuErr));
        IONIC_ERROR(&ctx->logger, pipeline_->tag, "cudaSetDevice(device=%hhu) failed err=%s", ctx->device.ordinal, err.what);
        ionic_set_error(ctx, err);
        goto ko;
    }

    struct cudaDeviceProp props;
    if ((cuErr = cudaGetDeviceProperties(&props, ctx->device.ordinal)) != cudaSuccess) {
        IONIC_WARN(&ctx->logger, pipeline_->tag, "cudaGetDeviceProperties failed (%s)", cudaGetErrorString(cuErr));
        pipeline_->concurrency = 1; // default to 1, safe
    } else {
        pipeline_->concurrency = props.asyncEngineCount < 1 ? 1 : props.asyncEngineCount;
    }

    pipeline_->events = calloc(pipeline_->concurrency, sizeof(cudaEvent_t));
    pipeline_->streams = calloc(pipeline_->concurrency, sizeof(cudaStream_t));

    if (!pipeline_->events || !pipeline_->streams) {
        ionic_set_error(ctx, IONIC_ERR(IONIC_ERROR_ALLOCATION_FAILED));
        IONIC_ERROR(&ctx->logger, pipeline_->tag,
            "allocation failed cudaEvent_t=%p, cudaStream_t=%p, n=%hhu",
            pipeline_->events, pipeline_->streams, pipeline_->concurrency);
        goto ko;
    }

    for (unsigned char i = 0; i < pipeline_->concurrency; ++i) {
        if ((cuErr = cudaStreamCreateWithFlags(pipeline_->streams + i, cudaStreamNonBlocking)) != cudaSuccess) {
            struct ionic_error err = IONIC_CUDA_ERR_WITH_MSG(cuErr, cudaGetErrorString(cuErr));
            IONIC_WARN(&ctx->logger, pipeline_->tag, "cudaStreamCreateWithFlags failed err=%s", err.what);
            ionic_set_error(ctx, err);
            goto ko;
        }

        if ((cuErr = cudaEventCreateWithFlags(pipeline_->events + i, cudaEventDisableTiming)) != cudaSuccess) {
            struct ionic_error err = IONIC_CUDA_ERR_WITH_MSG(cuErr, cudaGetErrorString(cuErr));
            IONIC_WARN(&ctx->logger, pipeline_->tag, "cudaEventCreateWithFlags failed err=%s", err.what);
            ionic_set_error(ctx, err);
            goto ko;
        }
    }

    void *ioengine = ionic_pipeline_probe_ioengine(ctx, pipeline);
    if (!ioengine) {
        struct ionic_error err = IONIC_ERR_WITH_MSG(IONIC_ERROR_IOENGINE_INITIALIZATION_FAILED, "failed to get cuda ioengine");
        IONIC_ERROR(&ctx->logger, pipeline_->tag, "probe failed: %s", err.what);
        ionic_set_error(ctx, err);
        goto ko;
    }

    ionic_ioengine_initialize(ctx, ioengine);
    pipeline_->ioengine = ioengine;
    IONIC_TRACE(&ctx->logger, pipeline_->tag, "initialized concurrency=%hhu, threshold=%zu",
        pipeline_->concurrency, pipeline_->threshold);
    return;

ko:
    ionic_pipeline_cuda_destroy(&pipeline_->base);
}

struct ionic_io_file_segment *ionic_pipeline_cuda_fragments_from_files(
    struct ionic_context *ctx, const struct ionic_pipeline_cuda *pipeline, unsigned short rank) {
    IONIC_TRACE(&ctx->logger, pipeline->tag, "computing fragments files=%zu, rank=%hu", pipeline->base.n_files, rank);

    const struct ionic_sharding_plan *plan = pipeline->base.plan;

    size_t n_files = pipeline->base.n_files;
    imaxdiv_t n_files_per_rank = imaxdiv((intmax_t)n_files, (intmax_t)plan->world_size);
    size_t n_files_for_rank = n_files_per_rank.quot;

    if (n_files_per_rank.rem > 0 && rank < n_files_per_rank.rem)
        n_files_for_rank += 1;

    struct ionic_io_file_segment *segments = calloc(n_files_for_rank, sizeof(struct ionic_io_file_segment));
    if (!segments) {
        ionic_set_error(ctx, IONIC_ERR(IONIC_ERROR_ALLOCATION_FAILED));
        IONIC_ERROR(&ctx->logger, pipeline->tag, "fragments allocation failed n_files=%zu", n_files);
        goto exit;
    }

    for (size_t i = 0; i < n_files_for_rank; ++i) {
        const size_t idx = i * plan->world_size + rank;
        const char *f = pipeline->base.files[idx];
        const unsigned short location = pipeline->base.locations[idx];

        // look for the start/end of all the tensors in the file
        size_t from = SIZE_MAX, to = -SIZE_MAX;
        for (size_t j = 0; j < plan->n; ++j) {
            const struct ionic_tensor *t = plan->tensors[j].tensor;
            if (t->file == location && t->start < from) from = t->start;
            if (t->file == location && t->end > to) to = t->end;
        }

        segments[i].path = f;
        segments[i].from = from;
        segments[i].to   = to;
    }

exit:
    return segments;
}

struct dma_worker_params {
    struct ionic_context *ctx;
    struct ionic_ioengine *engine;
    atomic_uchar running;
};

static void ionic_pipeline_cuda_dma_worker(const struct dma_worker_params *params) {
    const thrd_t self = thrd_current();

    char tag[32];
    const struct ionic_context *ctx = params->ctx;
    snprintf(tag, sizeof(tag), "dma_worker[%lu](%s:%u)",
        self, IONIC_DEVICE_LITERAL[ctx->device.kind], ctx->device.ordinal);

    IONIC_INFO(&params->ctx->logger, tag, "started");
    while(atomic_load_explicit(&params->running, memory_order_acquire)) {
        thrd_yield();
    }

    IONIC_INFO(&params->ctx->logger, tag, "exited");
}

static size_t ionic_pipeline_cuda_scheduler_loop(
    struct ionic_context *ctx, const struct ionic_pipeline_cuda *pipeline, const struct ionic_sharding_plan *plan, const unsigned short rank){
    if (ionic_has_error(&ctx->error)) return 0;

    thrd_t dma_worker;
    struct dma_worker_params dma_params = { .ctx = ctx, .engine = pipeline->ioengine, .running = 1 };
    int dma_worker_status = thrd_create(&dma_worker, (thrd_start_t)ionic_pipeline_cuda_dma_worker, &dma_params);
    if (dma_worker_status != 0) {
        IONIC_ERROR(&ctx->logger, pipeline->tag, "dma_worker thread launch failed res=%i", dma_worker_status);
        ionic_set_error(ctx, IONIC_SYS_ERR_WITH_MSG(-dma_worker_status, "dma_worker thread launch failed"));
        return 0;
    }

    struct ionic_io_file_segment *segments = ionic_pipeline_cuda_fragments_from_files(ctx, pipeline, rank);
    size_t n = ionic_ioengine_fetch(ctx, pipeline->ioengine, segments, pipeline->base.n_files);
    atomic_store_explicit(&dma_params.running, 0, memory_order_release);

    int status = 0;
    thrd_join(dma_worker, &status);

    if (segments) free(segments);
    return n;
}

static void ionic_pipeline_cuda_execute(struct ionic_context *ctx, struct ionic_pipeline *pipeline, const struct ionic_sharding_plan *plan, const unsigned short rank) {
    if (ionic_has_error(&ctx->error)) return;

    const struct ionic_pipeline_cuda *pipeline_ = (struct ionic_pipeline_cuda *)pipeline;

    ionic_pipeline_cuda_allocate_plan_memory(ctx, pipeline_, plan, rank);
    if (ionic_has_error(&ctx->error)) return;

    IONIC_INFO(&ctx->logger, pipeline_->tag, "execute rank=%hu, n=%zu", rank, plan->n);

    ionic_pipeline_cuda_scheduler_loop(ctx, pipeline_, plan, rank);
}

struct ionic_pipeline *ionic_pipeline_cuda_create(struct ionic_context *ctx, unsigned short world_size) {
    if (ionic_has_error(&ctx->error)) goto ko;

    struct ionic_pipeline_cuda *pipeline = malloc(sizeof(struct ionic_pipeline_cuda));
    snprintf(pipeline->tag, sizeof(pipeline->tag), "pipeline(cuda:%hhu)", ctx->device.ordinal);

    pipeline->base.destroy = ionic_pipeline_cuda_destroy;
    pipeline->base.initialize = ionic_pipeline_cuda_initialize;
    pipeline->base.execute = ionic_pipeline_cuda_execute;

    IONIC_INFO(&ctx->logger, pipeline->tag, "create world_size=%hu", world_size);

    return &pipeline->base;

ko:
    return NULL;
}


