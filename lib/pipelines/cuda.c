#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
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
    struct ionic_iouring_engine_config p = {
        .qd = 64,
        .st_size = 128 * 1024,
        .files = (const char **)pipeline->base.files,
        .n_files = pipeline->base.n_files
    };
    pipeline->ioengine = &ionic_iouring_engine_create(ctx, &p)->base;
#else
    ionic_set_error(ctx, IONIC_ERR_WITH_MSG(IONIC_ERROR_UNSUPPORTED, "platform not supported yet."));
    pipeline->ioengine = NULL;
#endif
}

static void ionic_pipeline_cuda_destroy(struct ionic_pipeline *pipeline) {
    if (!pipeline) return;

    struct ionic_pipeline_cuda *pipeline_ = (struct ionic_pipeline_cuda *)pipeline;

    if (pipeline_->device_buffer) {
        cudaFree(pipeline_->device_buffer);
        pipeline_->device_buffer = NULL;
        pipeline_->device_buffer_size = 0;
    }

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
        pipeline_->concurrency = 1;
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

struct dma_worker_params {
    struct ionic_context *ctx;
    struct ionic_pipeline_cuda *pipeline;
    struct ionic_iouring_engine *engine;
    atomic_uchar running;
};

static int ionic_pipeline_cuda_dma_worker(void *arg) {
    const struct dma_worker_params *params = arg;

    struct ionic_context *ctx = params->ctx;
    struct ionic_pipeline_cuda *pipeline = params->pipeline;
    struct ionic_ioengine *engine = pipeline->ioengine;

    char tag[32];
    snprintf(tag, sizeof(tag), "dma_worker(%s:%u)",
        IONIC_DEVICE_LITERAL[ctx->device.kind], ctx->device.ordinal);

    IONIC_INFO(&ctx->logger, tag, "started");

    // there is always a maximum of 2 async copy engine(s) on Nvidia hardware
    struct ionic_io_fetch_result *dmas[2] = {  NULL, NULL };
    struct ionic_io_fetch_result *res[8];
    while (atomic_load_explicit(&params->running, memory_order_acquire)) {

        size_t count = engine->peek(ctx, engine, res, 8);
        if (count > 0) {
            IONIC_INFO(&ctx->logger, tag, "ioengine peek count=%zu", count);

            while (count > 0){
                const struct ionic_io_fetch_result *r = res[count - 1];
                if (!dmas[0] || !dmas[1]) {
                    const int dma = !dmas[0] ? 0 : 1;

                    for (size_t n = 0; n < r->n_entries; ++n) {
                        const struct ionic_scatter_entry *entry = &r->entries[n];
                        if (entry->dst) {
                            const cudaError_t err = cudaMemcpyAsync(
                                entry->dst,
                                r->data + entry->staging_offset,
                                entry->len,
                                cudaMemcpyHostToDevice,
                                pipeline->streams[dma]
                            );

                            if (err != cudaSuccess) {
                                // TODO(mfuntowicz): what do we do?
                                IONIC_ERROR(&ctx->logger, pipeline->tag, "memcpy failed err=%s", cudaGetErrorString(err));
                                continue;
                            }

                            cudaEventRecord(pipeline->events[dma], pipeline->streams[dma]);
                            dmas[dma] = r;
                        }
                    }

                    --count;
                }

                for (unsigned i = 0; i < 2; ++i) {
                    if (dmas[i]) {
                        if (cudaEventQuery(pipeline->events[i]) == cudaSuccess) {
                            for (size_t n = 0; n < dmas[i]->n_entries; ++n) {
                                const struct ionic_scatter_entry *entry = &dmas[i]->entries[n];
                                if (entry->dst && entry->userdata) {
                                    struct ionic_sharded_tensor_specs *specs = entry->userdata;
                                    atomic_fetch_add_explicit(&specs->loaded, entry->len, memory_order_release);
                                }
                            }
                            engine->mark_done(ctx, engine, dmas[i]);
                            dmas[i] = NULL;

                            IONIC_DEBUG(&ctx->logger, tag, "memcpy HtoD async done");
                        }
                    }
                }

                ionic_cpu_relax();
            }
        }

        ionic_cpu_relax();
    }

    for (unsigned i = 0; i < 2; ++i) {
        if (dmas[i]) {
            cudaEventSynchronize(pipeline->events[i]);
            for (size_t n = 0; n < dmas[i]->n_entries; ++n) {
                const struct ionic_scatter_entry *entry = &dmas[i]->entries[n];
                if (entry->dst && entry->userdata) {
                    struct ionic_sharded_tensor_specs *specs = entry->userdata;
                    atomic_fetch_add_explicit(&specs->loaded, entry->len, memory_order_release);
                }
            }
            engine->mark_done(ctx, engine, dmas[i]);
            dmas[i] = NULL;

            IONIC_DEBUG(&ctx->logger, pipeline->tag, "finalized memcpy HtoD async");
        }
    }

    IONIC_INFO(&ctx->logger, tag, "exited");
    return 0;
}

static size_t ionic_pipeline_cuda_scheduler_loop(
    struct ionic_context *ctx,
    struct ionic_pipeline_cuda *pipeline,
    const struct ionic_sharding_plan *plan,
    const unsigned short rank)
{
    if (ionic_has_error(&ctx->error)) return 0;

    size_t n_bytes = 0;
    for (size_t i = 0; i < plan->n; ++i) {
        const struct ionic_tensor *t = plan->tensors[i].tensor;
        n_bytes += t->end - t->start;
    }

    if (n_bytes == 0) return 0;

    void *device_buffer = ctx->dalloc.allocate(ctx, n_bytes, IONIC_ALLOC_DEVICE);
    if (!device_buffer) {
        ionic_set_error(ctx, IONIC_ERR(IONIC_ERROR_ALLOCATION_FAILED));
        IONIC_ERROR(&ctx->logger, pipeline->tag, "device allocation failed size=%zu", n_bytes);
        return 0;
    }

    if (pipeline->device_buffer) {
        ctx->dalloc.free(ctx, pipeline->device_buffer, IONIC_ALLOC_DEVICE);
    }
    pipeline->device_buffer = device_buffer;
    pipeline->device_buffer_size = n_bytes;

    struct ionic_logical_segment *segments = calloc(plan->n, sizeof(struct ionic_logical_segment));
    if (!segments) {
        ionic_set_error(ctx, IONIC_ERR(IONIC_ERROR_ALLOCATION_FAILED));
        return 0;
    }

    size_t running_offset = 0;
    for (size_t i = 0; i < plan->n; ++i) {
        const struct ionic_tensor *t = plan->tensors[i].tensor;
        struct ionic_sharded_tensor_specs *specs = plan->tensors[i].specs + rank;
        atomic_store_explicit(&specs->loaded, 0, memory_order_relaxed);
        segments[i] = (struct ionic_logical_segment){
            .path = t->file,
            .from = t->start,
            .to = t->end,
            .dst = (char *)device_buffer + running_offset,
            .userdata = specs,
        };
        running_offset += t->end - t->start;
    }

    struct ionic_iouring_engine *iouring_engine = (struct ionic_iouring_engine *)pipeline->ioengine;

    thrd_t dma_worker;
    struct dma_worker_params dma_params = {
        .ctx = ctx,
        .pipeline = pipeline,
        .engine = iouring_engine,
        .running = 1
    };
    int dma_worker_status = thrd_create(&dma_worker, ionic_pipeline_cuda_dma_worker, &dma_params);
    if (dma_worker_status != 0) {
        IONIC_ERROR(&ctx->logger, pipeline->tag, "dma_worker thread launch failed res=%i", dma_worker_status);
        ionic_set_error(ctx, IONIC_SYS_ERR_WITH_MSG(-dma_worker_status, "dma_worker thread launch failed"));
        free(segments);
        return 0;
    }

    size_t n_fetched = ionic_ioengine_fetch(ctx, pipeline->ioengine, segments, plan->n);

    atomic_store_explicit(&dma_params.running, 0, memory_order_release);

    int status = 0;
    thrd_join(dma_worker, &status);

    free(segments);
    return n_fetched;
}

static void ionic_pipeline_cuda_execute(struct ionic_context *ctx, struct ionic_pipeline *pipeline, const struct ionic_sharding_plan *plan, const unsigned short rank) {
    if (ionic_has_error(&ctx->error)) return;

    struct ionic_pipeline_cuda *pipeline_ = (struct ionic_pipeline_cuda *)pipeline;

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
