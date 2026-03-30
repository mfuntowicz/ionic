#include <stdio.h>
#include <ionic/pipelines/cuda.h>
#include <ionic/logging.h>
#include <stdlib.h>

static void ionic_pipeline_cuda_destroy(struct ionic_pipeline *pipeline) {
    if (!pipeline) return;

    struct ionic_pipeline_cuda *pipeline_ = (struct ionic_pipeline_cuda *)pipeline;
    if (pipeline_->stream) {
        cudaStreamSynchronize(pipeline_->stream);
        cudaStreamDestroy(pipeline_->stream);
    }
    if (pipeline_->event) {
        cudaEventSynchronize(pipeline_->event);
        cudaEventDestroy(pipeline_->event);
    }
}

static void ionic_pipeline_cuda_initialize(struct ionic_context *ctx, struct ionic_pipeline *pipeline) {
    if (ionic_has_error(&ctx->error)) return;

    struct ionic_pipeline_cuda *pipeline_ = (struct ionic_pipeline_cuda *)pipeline;
    IONIC_TRACE(&ctx->logger, pipeline_->tag, "initialize");
}

static void ionic_pipeline_cuda_execute(
    struct ionic_context *ctx, struct ionic_pipeline *pipeline, const struct ionic_sharding_plan *plan, unsigned short rank) {
    if (ionic_has_error(&ctx->error)) return;

    struct ionic_pipeline_cuda *pipeline_ = (struct ionic_pipeline_cuda *)pipeline;
    IONIC_INFO(&ctx->logger, pipeline_->tag, "execute rank=%hu", rank);
}

struct ionic_pipeline *ionic_pipeline_cuda_create(struct ionic_context *ctx, unsigned short world_size) {
    if (ionic_has_error(&ctx->error)) goto ko;

    struct ionic_pipeline_cuda *pipeline = malloc(sizeof(struct ionic_pipeline_cuda));
    snprintf(pipeline->tag, sizeof(pipeline->tag), "pipeline(cuda:%hhu)", ctx->device.ordinal);

    cudaStreamCreateWithFlags(&pipeline->stream, cudaStreamNonBlocking);
    cudaEventCreateWithFlags(&pipeline->event, cudaEventDisableTiming);

    pipeline->base.destroy = ionic_pipeline_cuda_destroy;
    pipeline->base.initialize = ionic_pipeline_cuda_initialize;
    pipeline->base.execute = ionic_pipeline_cuda_execute;

    IONIC_INFO(&ctx->logger, pipeline->tag, "create world_size=%hu", world_size);

    return &pipeline->base;

ko:
    return NULL;
}


