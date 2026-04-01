#include <stdio.h>
#include <stdlib.h>
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
    struct ionic_iouring_engine_config p = { .qd = 32 };
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
    //todo(mfuntowicz) free ioengine memory
}

static void ionic_pipeline_cuda_initialize(struct ionic_context *ctx, struct ionic_pipeline *pipeline) {
    if (ionic_has_error(&ctx->error)) return;

    struct ionic_pipeline_cuda *pipeline_ = (struct ionic_pipeline_cuda *)pipeline;

    enum cudaError cuErr;
    if ((cuErr = cudaSetDevice(ctx->device.ordinal)) != cudaSuccess) {
        struct ionic_error err = IONIC_CUDA_ERR_WITH_MSG(cuErr, cudaGetErrorString(cuErr));
        IONIC_ERROR(&ctx->logger, pipeline_->tag, "cudaSetDevice(device=%hhu) failed err=%s", ctx->device.ordinal, err.what);
        ionic_set_error(ctx, err);
        return;
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

        if ((cuErr = cudaEventCreateWithFlags(pipeline_->events +i, cudaEventDisableTiming)) != cudaSuccess) {
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
        return;
    }

    pipeline_->ioengine = ioengine;  //todo(mfuntowicz): move to struct ionic_pipeline?
    IONIC_TRACE(&ctx->logger, pipeline_->tag, "initialized concurrency=%hhu", pipeline_->concurrency);
    return;

ko:
    ionic_pipeline_cuda_destroy(&pipeline_->base);
}

static void ionic_pipeline_cuda_scheduler_loop(
    struct ionic_context *ctx, const struct ionic_pipeline_cuda *pipeline, const struct ionic_sharding_plan *plan, const unsigned short rank) {

    struct ionic_io_uring_engine *ioengine = (struct ionic_io_uring_engine *)pipeline->ioengine;

    size_t done = 0;
    while (done < plan->n) {
        while (ionic_ioengine_can_submit(ctx, pipeline->ioengine)) {

        }

        ionic_ioengine_poll(ctx, pipeline->ioengine);

        ++done;
        IONIC_DEBUG(&ctx->logger, pipeline->tag, "tensor ack done=%zu", done);
    }
}

static void ionic_pipeline_cuda_execute(
    struct ionic_context *ctx, struct ionic_pipeline *pipeline, const struct ionic_sharding_plan *plan, const unsigned short rank) {

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


