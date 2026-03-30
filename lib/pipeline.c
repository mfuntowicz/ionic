#include "ionic/pipeline.h"

#include <time.h>
#include <ionic/utils.h>

#ifdef __IONIC_CUDA_ENABLED__
#include <ionic/pipelines/cuda.h>
#endif

#define IONIC_EVENT_TAG_PIPELINE "pipeline"

struct ionic_pipeline *ionic_pipeline_probe(struct ionic_context *ctx, unsigned short world_size)
{
#ifdef __IONIC_CUDA_ENABLED__
    if(world_size == 1 && ctx->device.kind == IONIC_DEVICE_CUDA) {
        IONIC_INFO(&ctx->logger, IONIC_EVENT_TAG_PLANNER, "probe -> device=CUDA, world_size=1");
        return ionic_pipeline_cuda_create(ctx, 1);
    }
#endif

    IONIC_ERROR(&ctx->logger, IONIC_EVENT_TAG_PLANNER, "unsupported topology device=%s, world_size=%hu", IONIC_DEVICE_LITERAL[ctx->device.kind], world_size);
    ionic_set_error(ctx, IONIC_ERR_WITH_MSG(IONIC_ERROR_UNSUPPORTED_TOPOLOGY, "only world size = 1 is supported"));
    return NULL;
}

void ionic_pipeline_init(
    struct ionic_context *ctx,
    struct ionic_pipeline *pipeline,
    const unsigned short *locations,
    char * const *files,
    size_t n_tensors,
    size_t n_files
){
    if (ionic_has_error(&ctx->error)) return;
    if (!pipeline) return;

    pipeline->initialize(ctx, pipeline);
}

void ionic_pipeline_destroy(struct ionic_pipeline *pipeline) {
    if (pipeline) {
        if (pipeline->destroy) pipeline->destroy(pipeline);
        free(pipeline);
    }
}

size_t ionic_pipeline_execute(struct ionic_context *ctx, struct ionic_pipeline *pipeline, const struct ionic_sharding_plan *plan, unsigned short rank) {
    IONIC_INFO(&ctx->logger, IONIC_EVENT_TAG_PIPELINE, "executing plan n=%zu", plan->n);

    if (ionic_has_error(&ctx->error)) return 0;
    if (!pipeline) return 0;

    struct timespec start, end;
    clock_gettime(CLOCK_MONOTONIC_COARSE, &start);

    pipeline->execute(ctx, pipeline, plan, rank);

    clock_gettime(CLOCK_MONOTONIC_COARSE, &end);
    double duration = (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9;
    IONIC_INFO(&ctx->logger, IONIC_EVENT_TAG_PIPELINE, "executed plan n=%zu, duration=%.6fs", plan->n, duration);

    return 0;
}