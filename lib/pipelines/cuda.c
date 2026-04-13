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

static void ionic_pipeline_cuda_probe_ioengine(struct ionic_context *ctx, struct ionic_pipeline_cuda *pipeline) {
    IONIC_TRACE(&ctx->logger, pipeline->tag, "probing ioengine");
#ifdef __linux__
    struct ionic_iouring_engine_config p = { .qd = 32, .st_size = 2 * 1024 * 1024, .files = (const char **)pipeline->base.files, .n_files = pipeline->base.n_files}; // todo(mfuntowicz): move to iouring + override with envvar
    pipeline->ioengine = &ionic_iouring_engine_create(ctx, &p)->base;
#else
    ionic_set_error(ctx, IONIC_ERR_WITH_MSG(IONIC_ERROR_UNSUPPORTED, "platform not supported yet."));
    pipeline->ioengine = NULL;
#endif
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

    ionic_pipeline_cuda_probe_ioengine(ctx, pipeline_);
    if (!pipeline_->ioengine) {
        struct ionic_error err = IONIC_ERR_WITH_MSG(IONIC_ERROR_IOENGINE_INITIALIZATION_FAILED, "failed to get cuda ioengine");
        IONIC_ERROR(&ctx->logger, pipeline_->tag, "probe failed: %s", err.what);
        ionic_set_error(ctx, err);
        goto ko;
    }

    ionic_ioengine_initialize(ctx, pipeline_->ioengine);
    IONIC_TRACE(&ctx->logger, pipeline_->tag, "initialized concurrency=%hhu, threshold=%zu",
        pipeline_->concurrency, pipeline_->threshold);
    return;

ko:
    ionic_pipeline_cuda_destroy(&pipeline_->base);
}

size_t ionic_pipeline_cuda_fragments_from_files(
    struct ionic_context *ctx, const struct ionic_pipeline_cuda *pipeline, unsigned short rank, struct ionic_io_file_segment **out) {
    const struct ionic_sharding_plan *plan = pipeline->base.plan;
    IONIC_TRACE(&ctx->logger, pipeline->tag, "computing fragments tensors=%zu, rank=%hu", plan->n, rank);

    if (plan->n == 0) return 0;

    size_t n_segments = 0;
    struct ionic_io_file_segment *segments = calloc(plan->n, sizeof(struct ionic_io_file_segment));
    if (!segments) {
        ionic_set_error(ctx, IONIC_ERR(IONIC_ERROR_ALLOCATION_FAILED));
        IONIC_ERROR(&ctx->logger, pipeline->tag, "segments allocation failed n=%zu", plan->n);
        return 0;
    }

    for (; n_segments < plan->n; ++n_segments) {
        const struct ionic_sharded_tensor st = plan->tensors[n_segments];
        const struct ionic_tensor *t = st.tensor;

        segments[n_segments] = (struct ionic_io_file_segment){
            .path = pipeline->base.files[t->file],
            .from = t->start,
            .to = t->end,
            .dst = NULL
        };
    }

    *out = segments;
    return n_segments;
}

struct dma_worker_params {
    struct ionic_context *ctx;
    struct ionic_pipeline_cuda *pipeline;
    struct ionic_iouring_engine *engine;
    atomic_uchar running;
};

static int ionic_pipeline_cuda_dma_worker(void *arg) {
    const struct dma_worker_params *params = (struct dma_worker_params *)arg;

    char tag[32];
    snprintf(tag, sizeof(tag), "dma_worker(%s:%u)",
        IONIC_DEVICE_LITERAL[params->ctx->device.kind], params->ctx->device.ordinal);

    IONIC_INFO(&params->ctx->logger, tag, "started");

    struct ionic_pipeline_cuda *pipeline = params->pipeline;
    struct ionic_iouring_engine *engine = params->engine;
    const unsigned char concurrency = pipeline->concurrency;

    int *active_slots = malloc(concurrency * sizeof(int));
    if (!active_slots) {
        IONIC_ERROR(&params->ctx->logger, tag, "allocation failed for active_slots");
        goto exit;
    }
    for (unsigned char i = 0; i < concurrency; ++i)
        active_slots[i] = -1;

    while (atomic_load_explicit(&params->running, memory_order_acquire)) {
        for (int w = 0; w < 2; ++w) {
            unsigned long bits = atomic_load_explicit(&engine->pending[w], memory_order_acquire);
            while (bits) {
                unsigned bit = __builtin_ctzl(bits);
                unsigned slot = w * BITS_PER_WORD + bit;

                unsigned char s;
                for (s = 0; s < concurrency; ++s)
                    if (active_slots[s] < 0) break;
                if (s == concurrency) break;

                void *src = engine->iovecs[slot].iov_base;
                void *dst = engine->slot_infos[slot].dst;
                size_t len = engine->slot_infos[slot].len;

                if (!dst || len == 0) {
                    atomic_fetch_and_explicit(&engine->pending[w], ~(1UL << bit), memory_order_release);
                    atomic_fetch_or_explicit(&engine->done[w], 1UL << bit, memory_order_release);
                    bits = atomic_load_explicit(&engine->pending[w], memory_order_acquire);
                    continue;
                }

                cudaMemcpyAsync(dst, src, len, cudaMemcpyHostToDevice, pipeline->streams[s]);
                cudaEventRecord(pipeline->events[s], pipeline->streams[s]);
                active_slots[s] = (int)slot;

                atomic_fetch_and_explicit(&engine->pending[w], ~(1UL << bit), memory_order_release);
                bits = atomic_load_explicit(&engine->pending[w], memory_order_acquire);
            }
        }

        for (unsigned char s = 0; s < concurrency; ++s) {
            if (active_slots[s] < 0) continue;
            if (cudaEventQuery(pipeline->events[s]) == cudaSuccess) {
                unsigned slot = (unsigned)active_slots[s];
                unsigned w = slot / BITS_PER_WORD;
                unsigned bit = slot % BITS_PER_WORD;
                atomic_fetch_or_explicit(&engine->done[w], 1UL << bit, memory_order_release);
                active_slots[s] = -1;
            }
        }

        ionic_cpu_relax();
    }

    for (unsigned char s = 0; s < concurrency; ++s) {
        if (active_slots[s] >= 0) {
            cudaEventSynchronize(pipeline->events[s]);
            unsigned slot = (unsigned)active_slots[s];
            unsigned w = slot / BITS_PER_WORD;
            unsigned bit = slot % BITS_PER_WORD;
            atomic_fetch_or_explicit(&engine->done[w], 1UL << bit, memory_order_release);
            active_slots[s] = -1;
        }
    }

    free(active_slots);

exit:
    IONIC_INFO(&params->ctx->logger, tag, "exited");
    return 0;
}

static size_t ionic_pipeline_cuda_scheduler_loop(
    struct ionic_context *ctx, const struct ionic_pipeline_cuda *pipeline, const struct ionic_sharding_plan *plan, const unsigned short rank) {
    if (ionic_has_error(&ctx->error)) return 0;

    thrd_t dma_worker;
    struct dma_worker_params dma_params = {
        .ctx = ctx,
        .pipeline = (struct ionic_pipeline_cuda *)pipeline,
        .engine = (struct ionic_iouring_engine *)pipeline->ioengine,
        .running = 1
    };
    int dma_worker_status = thrd_create(&dma_worker, ionic_pipeline_cuda_dma_worker, &dma_params);
    if (dma_worker_status != 0) {
        IONIC_ERROR(&ctx->logger, pipeline->tag, "dma_worker thread launch failed res=%i", dma_worker_status);
        ionic_set_error(ctx, IONIC_SYS_ERR_WITH_MSG(-dma_worker_status, "dma_worker thread launch failed"));
        return 0;
    }

    struct ionic_io_file_segment *segments;
    size_t n_segments = ionic_pipeline_cuda_fragments_from_files(ctx, pipeline, rank, &segments);
    size_t n_fetched = 0;
    if (ionic_has_error(&ctx->error)) goto terminate;

    n_fetched = ionic_ioengine_fetch(ctx, pipeline->ioengine, segments, n_segments);

terminate:
    atomic_store_explicit(&dma_params.running, 0, memory_order_release);

    int status = 0;
    thrd_join(dma_worker, &status);

    if (segments) free(segments);
    return n_fetched;
}

static void ionic_pipeline_cuda_execute(struct ionic_context *ctx, struct ionic_pipeline *pipeline, const struct ionic_sharding_plan *plan, const unsigned short rank) {
    if (ionic_has_error(&ctx->error)) return;

    const struct ionic_pipeline_cuda *pipeline_ = (struct ionic_pipeline_cuda *)pipeline;

    IONIC_INFO(&ctx->logger, pipeline_->tag, "execute rank=%hu, n=%zu", rank, plan->n);

    ionic_pipeline_cuda_scheduler_loop(ctx, pipeline_, plan, rank);
}

struct ionic_pipeline *ionic_pipeline_cuda_create(struct ionic_context *ctx, unsigned short world_size) {
    if (ionic_has_error(&ctx->error)) goto ko;

    struct ionic_pipeline_cuda *pipeline = calloc(1, sizeof(struct ionic_pipeline_cuda));
    snprintf(pipeline->tag, sizeof(pipeline->tag), "pipeline(cuda:%hhu)", ctx->device.ordinal);

    pipeline->base.destroy = ionic_pipeline_cuda_destroy;
    pipeline->base.initialize = ionic_pipeline_cuda_initialize;
    pipeline->base.execute = ionic_pipeline_cuda_execute;

    IONIC_INFO(&ctx->logger, pipeline->tag, "create world_size=%hu", world_size);

    return &pipeline->base;

ko:
    return NULL;
}
