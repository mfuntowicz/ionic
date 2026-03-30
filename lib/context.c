#include "ionic/types.h"
#include "ionic/error.h"
#include "ionic/ionic.h"
#include "ionic/topology.h"

#ifdef __IONIC_CUDA_ENABLED__
#include <ionic/devices/cuda.h>
#endif

#if defined(__linux__)
void *ionic_linux_cpu_allocate(struct ionic_context *ctx, size_t size, enum ionic_allocation_kind kind);
void ionic_linux_cpu_free(struct ionic_context *ctx, void *ptr, enum ionic_allocation_kind kind);
#else
void *ionic_malloc_cpu_allocate(struct ionic_context *ctx, size_t size, enum ionic_allocation_kind kind);
void ionic_malloc_cpu_free(struct ionic_context *ctx, void *ptr, enum ionic_allocation_kind kind);
#endif

#define IONIC_EVENT_TAG_CONTEXT "context"

void ionic_allocator_init(struct ionic_context *ctx, struct ionic_device device) {
#if defined(__linux__)
    struct ionic_allocator cpu_alloc = { .allocate = ionic_linux_cpu_allocate, .free = ionic_linux_cpu_free };
#else
    struct ionic_allocator cpu_alloc = { .allocate = ionic_malloc_cpu_allocate, .free = ionic_malloc_cpu_free };
#endif

    switch (device.kind) {
    case IONIC_DEVICE_CPU:
        ctx->halloc = cpu_alloc;
        ctx->dalloc = cpu_alloc;
        break;

    case IONIC_DEVICE_CUDA:
#ifndef __IONIC_CUDA_ENABLED__
        ctx->error = IONIC_ERR_WITH_MSG(IONIC_ERROR_UNSUPPORTED_DEVICE, "CUDA not enabled");
        break;
#else
        ctx->halloc = cpu_alloc;
        ctx->dalloc = ionic_get_cuda_allocator(ctx, device);
        break;
#endif
    }
}

void ionic_context_init(struct ionic_context *ctx, struct ionic_device device) {
    ctx->error = IONIC_SUCCESS;
    ctx->device = device;

    ionic_logger_init(&ctx->logger);
    ionic_allocator_init(ctx, device);
    ionic_topology_init(ctx);

#ifdef __IONIC_CUDA_ENABLED__
    if (device.kind != IONIC_DEVICE_CPU && device.kind != IONIC_DEVICE_CUDA) {
        ctx->error = IONIC_ERR(IONIC_ERROR_UNSUPPORTED_DEVICE);
        IONIC_ERROR(&ctx->logger, IONIC_EVENT_TAG_CONTEXT, "initialization failed reason=%s unsupported", IONIC_DEVICE_LITERAL[device.kind]);
        return;
    }
#else
    if (device.kind != IONIC_DEVICE_CPU) {
        ctx->error = IONIC_ERR(IONIC_ERROR_UNSUPPORTED_DEVICE);
        IONIC_ERROR(&ctx->logger, IONIC_CONTEXT_EVENT_TAG, "initialization failed reason=%s not enabled", IONIC_DEVICE_LITERAL[device.kind]);
        return;
    }
#endif

    IONIC_DEBUG(&ctx->logger, IONIC_EVENT_TAG_CONTEXT, "initialized");
}

void ionic_context_destroy(ionic_context_t *ctx) {
    ionic_topology_destroy(ctx);
    IONIC_DEBUG(&ctx->logger, IONIC_EVENT_TAG_CONTEXT, "destroyed");
}