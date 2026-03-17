#include <ionic/ionic.h>
#include <ionic/topology.h>

void ionic_context_init(struct ionic_context *ctx) {
    ctx->error = IONIC_SUCCESS;
    ionic_logger_init(&ctx->logger);
    ionic_topology_init(ctx);
    IONIC_DEBUG(&ctx->logger, "Context initialized");
}

void ionic_context_destroy(ionic_context_t *ctx) {
    ionic_topology_destroy(ctx);
    IONIC_DEBUG(&ctx->logger, "Context destroyed");
}