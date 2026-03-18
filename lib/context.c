#include <ionic/ionic.h>
#include <ionic/topology.h>
#include <ionic/platform/linux/iouring.h>

void ionic_context_init(struct ionic_context *ctx) {
    ctx->error = IONIC_SUCCESS;
    ionic_logger_init(&ctx->logger);
    ionic_topology_init(ctx);
    ionic_iouring_init(ctx);
    IONIC_DEBUG(&ctx->logger, "context", "initialized");
}

void ionic_context_destroy(ionic_context_t *ctx) {
    if(ctx->backend && ctx->backend->destroy) ctx->backend->destroy(ctx);
    ionic_topology_destroy(ctx);
    IONIC_DEBUG(&ctx->logger, "context", "destroyed");
}