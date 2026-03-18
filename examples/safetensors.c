#include <stdio.h>
#include <ionic/ionic.h>
#include <ionic/safetensors.h>

int main(int argc, char **argv) {
    ionic_context_t ctx;
    ionic_safetensors_t registry;

    if(argc != 2) {
        fprintf(stderr, "Usage: %s <model_path>\n", argv[0]);
        return 1;
    }

    ionic_context_init(&ctx);
    ionic_safetensors_init(&registry);
    ionic_safetensors_discover_tensors(&ctx, &registry, argv[1]);

    if(ionic_has_error(&ctx.error)) {
        fprintf(stderr, "Failed to populate safetensors tensors registry: %s\n", ctx.error.what);
        return 1;
    }

    ionic_safetensors_destroy(&registry);
    ionic_context_destroy(&ctx);
    return 0;
}