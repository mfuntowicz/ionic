#include <ionic/ionic.h>


void ionic_context_init(ionic_context_t *ctx) {
    ctx->error = IONIC_SUCCESS;
    ionic_logger_init(&ctx->logger);
    IONIC_DEBUG(&ctx->logger, "Context initialized");
}

void ionic_context_destroy(ionic_context_t *ctx) {
    IONIC_DEBUG(&ctx->logger, "Context destroyed");
}