#include "ionic/pipeline.h"
#include <errno.h>
#include <stdio.h>
#include <ionic/ionic.h>
#include <ionic/safetensors.h>
#include <stdlib.h>
#include <string.h>

#ifdef __IONIC_CUDA_ENABLED__
#include <ionic/planner.h>
#endif

int main(int argc, char **argv)
{
    ionic_context_t ctx;
    ionic_safetensors_t registry;
    ionic_device_t device = ionic_cuda_device(0);

    if (argc < 2) {
        fprintf(stderr, "Usage: %s <model_path> [--world-size <size>]\n", argv[0]);
        return 1;
    }

    int world_size = 0;
    if(argc == 4 && strcmp(argv[2], "--world-size") == 0) {
        errno = 0;
        char *end;
        char *val = argv[3];
        world_size = strtoul(val, &end, 10);
        if(val == end || errno == ERANGE) {
            fprintf(stderr, "Failed to interpret world_size provided argument: %s\n", val);
            return 2;
        }

        if(world_size == 0) {
            fprintf(stderr, "Invalid world size %ul, should be >= 1", world_size);
            return 2;
        }
    } else {
        world_size = 1;
    }

    ionic_context_init(&ctx, device);
    ionic_safetensors_init(&registry);
    ionic_safetensors_discover_tensors(&ctx, &registry, argv[1]);

    if (ionic_has_error(&ctx.error)) {
        fprintf(stderr, "Failed to populate safetensors tensors registry: %s\n", ctx.error.what);
        return 3;
    }

    struct ionic_planner *planner = ionic_planner_init(&ctx, registry.n, 0, world_size);

    if (!ionic_has_error(&ctx.error)) {
        for (size_t i = 0; i < registry.n; i++)
            ionic_planner_register_sharding(&ctx, planner, &registry.tensors[i], IONIC_SHARDING_REPLICATED, registry.fds[i]);
    }

    struct ionic_sharding_plan plan = ionic_planner_materialize_plan(&ctx, planner, 0);
    struct ionic_pipeline *pipeline = ionic_pipeline_init(&ctx, ionic_pipeline_config_default(ctx.device));
    int num_tensors = ionic_pipeline_execute_plan(&ctx, pipeline, &plan, planner->rank);

    ionic_planner_destroy(planner);
    ionic_safetensors_destroy(&registry);
    ionic_context_destroy(&ctx);
    
    return 0;
}
