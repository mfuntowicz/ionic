#ifndef IONIC_DEVICES_CUDA_H
#define IONIC_DEVICES_CUDA_H

#include "ionic/ionic.h"
#include "ionic/logging.h"
#include <ionic/types.h>

#define IONIC_EVENT_TAG_DEVICES_CUDA "cuda"


#ifdef __IONIC_CUDA_ENABLED__
#include <cuda_runtime.h>

static inline void *ionic_cuda_naive_allocate(struct ionic_context *ctx, size_t size, enum ionic_allocation_kind kind) {
    void *dst = NULL;
    cudaError_t err;
    switch (kind) {
    /* todo(mfuntowicz): cudaMallocManaged path for Grace unified memory / Blackwell */
    case IONIC_ALLOC_DEVICE:
        IONIC_DEBUG(&ctx->logger, IONIC_EVENT_TAG_DEVICES_CUDA, "allocating memory size=%zu, kind=device, device=%u", size, ctx->device.ordinal);
        if((err = cudaMalloc(&dst, size)) != cudaSuccess) {
            IONIC_ERROR(&ctx->logger, IONIC_EVENT_TAG_DEVICES_CUDA, "allocation failed size=%zu, kind=device, error=%u", size, err);
            return NULL;
        }
        break;
    case IONIC_ALLOC_STAGING:
        IONIC_DEBUG(&ctx->logger, IONIC_EVENT_TAG_DEVICES_CUDA, "allocating memory size=%zu, kind=staging, device=%u", size, ctx->device.ordinal);
        if ((err = cudaHostAlloc(&dst, size, cudaHostAllocDefault)) != cudaSuccess) {
            IONIC_ERROR(&ctx->logger, IONIC_EVENT_TAG_DEVICES_CUDA, "allocation failed size=%zu, kind=staging, error=%u", size, (unsigned)err);
            return NULL;
        }
        break;
    default:
        return NULL;
    }
    return dst;
}

static inline void ionic_cuda_naive_free(struct ionic_context *ctx, void *ptr, enum ionic_allocation_kind kind) {
    if (!ptr)
        return;

    switch (kind) {
    case IONIC_ALLOC_DEVICE:
        IONIC_DEBUG(&ctx->logger, IONIC_EVENT_TAG_DEVICES_CUDA, "freeing cuda memory ptr=%p, kind=device", ptr);

        (void)cudaFree(ptr);
        break;
    case IONIC_ALLOC_STAGING:
        IONIC_DEBUG(&ctx->logger, IONIC_EVENT_TAG_DEVICES_CUDA, "freeing cuda memory ptr=%p, kind=staging", ptr);
        (void)cudaFreeHost(ptr);
        break;
    default:
        break;
    }
}

static inline struct ionic_allocator ionic_get_cuda_allocator(struct ionic_context *ctx, struct ionic_device device) {
    IONIC_TRACE(&ctx->logger, IONIC_EVENT_TAG_DEVICES_CUDA, "dalloc=cuda_naive");
    return (struct ionic_allocator) { .allocate = ionic_cuda_naive_allocate, .free = ionic_cuda_naive_free };
}

#endif // __IONIC_CUDA_ENABLED__
#endif // IONIC_DEVICES_CUDA_H