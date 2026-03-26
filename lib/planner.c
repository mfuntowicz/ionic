#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>
#include "ionic/error.h"
#include "ionic/ionic.h"
#include "ionic/logging.h"
#include "ionic/types.h"
#include "ionic/utils.h"
#include "ionic/planner.h"

#define IONIC_FILE_ALIGN_BYTES 4096u


static struct ionic_sharded_tensor ionic_sharding_rule_replicate(const struct ionic_tensor *tensor, struct ionic_device device, int world_size) {
    struct ionic_sharded_tensor sharded = {0};
    sharded.tensor = tensor;
    sharded.specs = calloc(world_size, sizeof(struct ionic_sharded_tensor_specs));

    for (int i = 0; i < world_size; ++i) {
        struct ionic_sharded_tensor_specs *specs = sharded.specs + i;
        specs->device.kind    = device.kind;
        specs->device.ordinal = i;
        specs->start          = tensor->start;
        specs->end            = tensor->end;
    }

    return sharded;
} 

void ionic_planner_destroy(ionic_planner_t *planner) {
    if(planner) {
        if(planner->destroy) planner->destroy(planner);
        if(planner->infos) free(planner->infos);
        
        planner->infos = NULL;
        planner->n = 0;
        planner->n_registered = 0;
        planner->world_size = 0;
        planner->rank = 0;
    }
}

struct ionic_planner *ionic_planner_init(struct ionic_context *ctx, size_t num_tensors, unsigned short rank, unsigned short world_size) {
    if(ionic_has_error(&ctx->error)) return NULL;

    struct ionic_planner *planner = NULL;
    if(world_size == 1 && ctx->device.kind == IONIC_DEVICE_CUDA) {
        planner = malloc(sizeof(struct ionic_planner_cuda));
        planner->initialize = ionic_planner_cuda_single_gpu_init;
        planner->destroy    = ionic_planner_cuda_single_gpu_destroy;
        planner->execute    = ionic_planner_cuda_execute;
    }
     
    planner->n = num_tensors;
    planner->n_registered = 0;
    planner->rank = rank;
    planner->world_size = world_size;

    planner->infos = calloc(num_tensors, sizeof(struct ionic_sharding_info));
    if(!planner->infos) {
        ctx->error = IONIC_ERR(IONIC_ERROR_ALLOCATION_FAILED);
        return NULL;
    }

    planner->initialize(ctx, planner);
    return planner;
}

void ionic_planner_register_sharding(ionic_context_t *ctx, ionic_planner_t *planner, const struct ionic_tensor *tensor, enum ionic_sharding_kind kind, int fd) {
    if (ionic_has_error(&ctx->error) || !planner || !tensor)
        return;

    if (planner->n_registered >= planner->n) {
        ionic_set_error(ctx, IONIC_ERR(IONIC_ERROR_PLANNER_INVALID_SHARDING));
        return;
    }

    if (planner->world_size == 1 && kind != IONIC_SHARDING_REPLICATED) {
        ionic_set_error(ctx, IONIC_ERR(IONIC_ERROR_PLANNER_INVALID_SHARDING));
        return;
    }
    
    struct ionic_sharding_info *info = planner->infos + planner->n_registered;
    info->tensor = tensor;
    info->kind = kind;
    info->fd = fd;
    planner->n_registered++;
}

ionic_sharding_plan_t ionic_planner_materialize_plan(ionic_context_t *ctx, ionic_planner_t *planner, int fd)
{
    ionic_sharding_plan_t out = {0};

    if(!planner || planner->n_registered != planner->n) {
        IONIC_ERROR(&ctx->logger, IONIC_EVENT_TAG_PLANNER, "n_registered != n_tensors n_registered=%zu, n_tensors=%zu", planner->n_registered, planner->n);
        ionic_set_error(ctx, IONIC_ERR_WITH_MSG(IONIC_ERROR_PLANNER_INVALID_SHARDING, "uneven number of tensors and sharding rules"));
        return out;
    }

    out.tensors = malloc(planner->n * sizeof(struct ionic_sharded_tensor));

    for (size_t i = 0; i < planner->n; ++i) {
        struct ionic_sharding_info *info = &planner->infos[i];
        struct ionic_sharded_tensor shard;

        switch(info->kind) {
            case IONIC_SHARDING_REPLICATED:
                shard = ionic_sharding_rule_replicate(info->tensor, ctx->device, planner->world_size);    
                break;
            
            default:
                break;
        }

        shard.fd = info->fd;
        out.tensors[i] = shard;
        out.n++;
    }
    return out;
}

void ionic_planner_execute_plan(struct ionic_context *ctx, struct ionic_planner *planner, struct ionic_sharding_plan *plan)
{
    IONIC_INFO(&ctx->logger, IONIC_EVENT_TAG_PLANNER, "executing plan n=%zu", plan->n);

    struct timespec start, end;
    clock_gettime(CLOCK_MONOTONIC_COARSE, &start);

    planner->execute(ctx, planner, plan);

    clock_gettime(CLOCK_MONOTONIC_COARSE, &end);
    double duration = (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9;
    IONIC_INFO(&ctx->logger, IONIC_EVENT_TAG_PLANNER, "executed plan n=%zu, duration=%.6fs", plan->n, duration);
}

#ifdef __IONIC_CUDA_ENABLED__

#include <cuda_runtime.h>
#include <ionic/devices/cuda.h>

void ionic_planner_cuda_single_gpu_init(struct ionic_context *ctx, struct ionic_planner *planner)
{
    struct ionic_planner_cuda *planner_ = (struct ionic_planner_cuda *) planner;
    
    planner_->stream = NULL;
    planner_->staging_size = 0;
    for (int i = 0; i < IONIC_PLANNER_CUDA_STAGING_COUNT; ++i) {
        planner_->staging[i] = NULL;
        planner_->staging_done[i] = NULL;
    }
    
    if (ionic_has_error(&ctx->error))
        return;
    
    cudaError_t ce = cudaStreamCreateWithFlags(&planner_->stream, cudaStreamNonBlocking);
    if (ce != cudaSuccess) {
        ionic_set_error(ctx, IONIC_CUDA_ERR((int)ce));
        return;
    }

    // Create events for staging buffer synchronization
    for (int i = 0; i < IONIC_PLANNER_CUDA_STAGING_COUNT; ++i) {
        ce = cudaEventCreateWithFlags(&planner_->staging_done[i], cudaEventDisableTiming);
        if (ce != cudaSuccess) {
            ionic_set_error(ctx, IONIC_CUDA_ERR((int)ce));
            return;
        }
    }

    IONIC_INFO(&ctx->logger, IONIC_EVENT_TAG_PLANNER, 
        "initialized device=cuda:%hhu n=%zu rank=%hhu world_size=%hhu", ctx->device.ordinal,
        planner_->base.n, planner_->base.rank, planner_->base.world_size);
}

void ionic_planner_cuda_single_gpu_destroy(struct ionic_planner *planner)
{
    struct ionic_planner_cuda *planner_ = (struct ionic_planner_cuda *) planner;
    
    if (planner_->stream) {
        (void)cudaStreamSynchronize(planner_->stream);
        (void)cudaStreamDestroy(planner_->stream);
        planner_->stream = NULL;
    }
    
    for (int i = 0; i < IONIC_PLANNER_CUDA_STAGING_COUNT; ++i) {
        if (planner_->staging_done[i]) {
            (void)cudaEventDestroy(planner_->staging_done[i]);
            planner_->staging_done[i] = NULL;
        }
        if (planner_->staging[i]) {
            (void)cudaFreeHost(planner_->staging[i]);
            planner_->staging[i] = NULL;
        }
    }
    planner_->staging_size = 0;
}

static int ionic_planner_cuda_ensure_staging(struct ionic_context *ctx, struct ionic_planner_cuda *planner, size_t required_size)
{
    if (planner->staging_size >= required_size)
        return 0;
    
    // Free old buffers if they exist
    for (int i = 0; i < IONIC_PLANNER_CUDA_STAGING_COUNT; ++i) {
        if (planner->staging[i]) {
            cudaFreeHost(planner->staging[i]);
            planner->staging[i] = NULL;
        }
    }
    
    // Allocate new pinned host buffers
    for (int i = 0; i < IONIC_PLANNER_CUDA_STAGING_COUNT; ++i) {
        cudaError_t ce = cudaHostAlloc(&planner->staging[i], required_size, cudaHostAllocDefault);
        if (ce != cudaSuccess) {
            ionic_set_error(ctx, IONIC_CUDA_ERR((int)ce));
            return -1;
        }
    }
    
    planner->staging_size = required_size;
    IONIC_DEBUG(&ctx->logger, IONIC_EVENT_TAG_PLANNER, "allocated staging buffers size=%zu count=%d", required_size, IONIC_PLANNER_CUDA_STAGING_COUNT);
    return 0;
}

void ionic_planner_cuda_execute(struct ionic_context *ctx, struct ionic_planner *planner, struct ionic_sharding_plan *plan) {
    struct ionic_planner_cuda *planner_cuda = (struct ionic_planner_cuda *) planner;
    
    const size_t chunk_size = 512 * 1024;  // 512 KiB chunks
    
    // Allocate device memory for each tensor
    for (size_t i = 0; i < plan->n; ++i) {
        size_t nbytes = ionic_tensor_nbytes(plan->tensors[i].tensor);
        void *dst = ionic_allocate_device(ctx, nbytes, IONIC_ALLOC_DEVICE);
        if (ionic_has_error(&ctx->error)) {
            IONIC_ERROR(&ctx->logger, IONIC_EVENT_TAG_PLANNER, "failed to allocate device memory for tensor %zu", i);
            return;
        }
        plan->tensors[i].specs[planner->rank].dst = dst;
    }
    
    // Ensure staging buffers are large enough
    if (ionic_planner_cuda_ensure_staging(ctx, planner_cuda, chunk_size) != 0)
        return;
    
    size_t total_bytes = 0;
    size_t total_chunks = 0;
    
    // Process each tensor
    for (size_t tensor_idx = 0; tensor_idx < plan->n; ++tensor_idx) {
        struct ionic_sharded_tensor *sharded = &plan->tensors[tensor_idx];
        const struct ionic_tensor *tensor = sharded->tensor;
        struct ionic_sharded_tensor_specs *spec = &sharded->specs[planner->rank];
        int fd = sharded->fd;
        
        size_t offset = spec->start;
        size_t remaining = spec->end - spec->start;
        unsigned char *dst_ptr = (unsigned char *)spec->dst;
        
        size_t chunk_idx = 0;
        
        while (remaining > 0) {
            size_t to_read = remaining < chunk_size ? remaining : chunk_size;
            int slot = (int)(chunk_idx % IONIC_PLANNER_CUDA_STAGING_COUNT);
            
            // Wait for this staging buffer's previous use to complete
            cudaError_t ce = cudaEventSynchronize(planner_cuda->staging_done[slot]);
            if (ce != cudaSuccess) {
                ionic_set_error(ctx, IONIC_CUDA_ERR((int)ce));
                return;
            }
            
            // Read from file via io_uring
            size_t read = ctx->backend->read(ctx, fd, planner_cuda->staging[slot], to_read, offset);
            if (read != to_read) {
                IONIC_ERROR(&ctx->logger, IONIC_EVENT_TAG_PLANNER, 
                    "short read tensor=%zu offset=%zu requested=%zu got=%zu", 
                    tensor_idx, offset, to_read, read);
                return;
            }
            
            // Async H2D copy
            ce = cudaMemcpyAsync(
                dst_ptr,
                planner_cuda->staging[slot],
                to_read,
                cudaMemcpyHostToDevice,
                planner_cuda->stream
            );
            if (ce != cudaSuccess) {
                ionic_set_error(ctx, IONIC_CUDA_ERR((int)ce));
                return;
            }
            
            // Record event to track when this staging buffer can be reused
            ce = cudaEventRecord(planner_cuda->staging_done[slot], planner_cuda->stream);
            if (ce != cudaSuccess) {
                ionic_set_error(ctx, IONIC_CUDA_ERR((int)ce));
                return;
            }
            
            offset += to_read;
            dst_ptr += to_read;
            remaining -= to_read;
            total_bytes += to_read;
            chunk_idx++;
            total_chunks++;
        }
    }
    
    // Final synchronization - wait for all copies to complete
    cudaError_t ce = cudaStreamSynchronize(planner_cuda->stream);
    if (ce != cudaSuccess) {
        ionic_set_error(ctx, IONIC_CUDA_ERR((int)ce));
        return;
    }
    
    IONIC_INFO(&ctx->logger, IONIC_EVENT_TAG_PLANNER, 
        "executed plan tensors=%zu chunks=%zu bytes=%zu", plan->n, total_chunks, total_bytes);
}

#endif