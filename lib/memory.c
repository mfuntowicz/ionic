#include <stdlib.h>
#include "ionic/ionic.h"

#if !defined(__linux__)

void *ionic_malloc_cpu_allocate(struct ionic_context *ctx, size_t size, enum ionic_allocation_kind kind) {
    (void)ctx;
    (void)kind;
    return malloc(size);
}

void ionic_malloc_cpu_free(struct ionic_context *ctx, void *ptr, enum ionic_allocation_kind kind) {
    (void)ctx;
    (void)kind;
    free(ptr);
}

#endif

void ionic_set_host_allocator(struct ionic_context *ctx, struct ionic_allocator alloc) {
    if (ctx)
        ctx->halloc = alloc;
}

void ionic_set_device_allocator(struct ionic_context *ctx, struct ionic_allocator alloc) {
    if (ctx)
        ctx->dalloc = alloc;
}

void *ionic_allocate_host(struct ionic_context *ctx, size_t size, enum ionic_allocation_kind kind) {
    if (!ctx || !ctx->halloc.allocate)
        return NULL;
    return ctx->halloc.allocate(ctx, size, kind);
}

void *ionic_allocate_device(struct ionic_context *ctx, size_t size, enum ionic_allocation_kind kind) {
    if (!ctx || !ctx->dalloc.allocate)
        return NULL;
    return ctx->dalloc.allocate(ctx, size, kind);
}

void ionic_free_host(struct ionic_context *ctx, void *ptr, enum ionic_allocation_kind kind) {
    if (!ptr || !ctx || !ctx->halloc.free)
        return;
    ctx->halloc.free(ctx, ptr, kind);
}

void ionic_free_device(struct ionic_context *ctx, void *ptr, enum ionic_allocation_kind kind) {
    if (!ptr || !ctx || !ctx->dalloc.free)
        return;
    ctx->dalloc.free(ctx, ptr, kind);
}
