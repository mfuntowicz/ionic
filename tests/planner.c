#include <string.h>
#include "ionic/ionic.h"
#include "ionic/planner.h"
#include "helpers.h"

static int test_planner_init(void)
{
    ionic_context_t ctx;
    ionic_context_init(&ctx, ionic_cpu_device(0));

    struct ionic_planner *planner = ionic_planner_init(&ctx, 4, 0, 1);
    if (!planner)
        return IONIC_RESULT_FAILURE;
    if (planner->n != 4)
        return IONIC_RESULT_FAILURE;
    if (planner->n_registered != 0)
        return IONIC_RESULT_FAILURE;
    if (planner->rank != 0)
        return IONIC_RESULT_FAILURE;
    if (planner->world_size != 1)
        return IONIC_RESULT_FAILURE;

    ionic_planner_destroy(planner);
    ionic_context_destroy(&ctx);
    return IONIC_RESULT_SUCCESS;
}

static int test_planner_shard_replicated(void)
{
    ionic_context_t ctx;
    ionic_context_init(&ctx, ionic_cpu_device(0));

    struct ionic_planner *planner = ionic_planner_init(&ctx, 2, 0, 1);
    if (!planner)
        return IONIC_RESULT_FAILURE;

    struct ionic_tensor tensors[2] = {
        { .start = 0,   .end = 100, .dtype = IONIC_DATA_TYPE_FLOAT32, .rank = 1, .shape = {10}, .file = "test.bin" },
        { .start = 100, .end = 300, .dtype = IONIC_DATA_TYPE_FLOAT32, .rank = 1, .shape = {20}, .file = "test.bin" },
    };

    ionic_planner_shard(&ctx, planner, &tensors[0], IONIC_SHARDING_REPLICATED);
    if (ctx.error.kind != IONIC_ERROR_SUCCESS)
        return IONIC_RESULT_FAILURE;
    if (planner->n_registered != 1)
        return IONIC_RESULT_FAILURE;

    ionic_planner_shard(&ctx, planner, &tensors[1], IONIC_SHARDING_REPLICATED);
    if (ctx.error.kind != IONIC_ERROR_SUCCESS)
        return IONIC_RESULT_FAILURE;
    if (planner->n_registered != 2)
        return IONIC_RESULT_FAILURE;

    ionic_planner_destroy(planner);
    ionic_context_destroy(&ctx);
    return IONIC_RESULT_SUCCESS;
}

static int test_planner_shard_non_replicated_fails_on_world_size_1(void)
{
    ionic_context_t ctx;
    ionic_context_init(&ctx, ionic_cpu_device(0));

    struct ionic_planner *planner = ionic_planner_init(&ctx, 1, 0, 1);
    if (!planner)
        return IONIC_RESULT_FAILURE;

    struct ionic_tensor tensor = { .start = 0, .end = 100, .dtype = IONIC_DATA_TYPE_FLOAT32, .rank = 1, .shape = {10}, .file = "test.bin" };
    ionic_planner_shard(&ctx, planner, &tensor, IONIC_SHARDING_COLWISE);

    if (ctx.error.kind == IONIC_ERROR_SUCCESS)
        return IONIC_RESULT_FAILURE;

    ionic_planner_destroy(planner);
    ionic_context_destroy(&ctx);
    return IONIC_RESULT_SUCCESS;
}

static int test_planner_too_many_shards(void)
{
    ionic_context_t ctx;
    ionic_context_init(&ctx, ionic_cpu_device(0));

    struct ionic_planner *planner = ionic_planner_init(&ctx, 1, 0, 1);
    if (!planner)
        return IONIC_RESULT_FAILURE;

    struct ionic_tensor tensors[2] = {
        { .start = 0,   .end = 100, .dtype = IONIC_DATA_TYPE_FLOAT32, .rank = 1, .shape = {10}, .file = "test.bin" },
        { .start = 100, .end = 300, .dtype = IONIC_DATA_TYPE_FLOAT32, .rank = 1, .shape = {20}, .file = "test.bin" },
    };

    ionic_planner_shard(&ctx, planner, &tensors[0], IONIC_SHARDING_REPLICATED);
    if (ctx.error.kind != IONIC_ERROR_SUCCESS)
        return IONIC_RESULT_FAILURE;

    ionic_planner_shard(&ctx, planner, &tensors[1], IONIC_SHARDING_REPLICATED);
    if (ctx.error.kind == IONIC_ERROR_SUCCESS)
        return IONIC_RESULT_FAILURE;

    ionic_planner_destroy(planner);
    ionic_context_destroy(&ctx);
    return IONIC_RESULT_SUCCESS;
}

static int test_materialize_plan_replicated(void)
{
    ionic_context_t ctx;
    ionic_context_init(&ctx, ionic_cpu_device(0));

    struct ionic_planner *planner = ionic_planner_init(&ctx, 2, 0, 1);
    if (!planner)
        return IONIC_RESULT_FAILURE;

    struct ionic_tensor tensors[2] = {
        { .start = 0,   .end = 100, .dtype = IONIC_DATA_TYPE_FLOAT32, .rank = 1, .shape = {10}, .file = "test.bin" },
        { .start = 100, .end = 300, .dtype = IONIC_DATA_TYPE_FLOAT32, .rank = 1, .shape = {20}, .file = "test.bin" },
    };

    ionic_planner_shard(&ctx, planner, &tensors[0], IONIC_SHARDING_REPLICATED);
    ionic_planner_shard(&ctx, planner, &tensors[1], IONIC_SHARDING_REPLICATED);

    ionic_sharding_plan_t plan = ionic_planner_materialize_plan(&ctx, planner);
    if (ctx.error.kind != IONIC_ERROR_SUCCESS)
        return IONIC_RESULT_FAILURE;
    if (plan.n != 2)
        return IONIC_RESULT_FAILURE;
    if (plan.world_size != 1)
        return IONIC_RESULT_FAILURE;
    if (!plan.tensors)
        return IONIC_RESULT_FAILURE;

    ionic_sharding_plan_destroy(&ctx, &plan);
    ionic_planner_destroy(planner);
    ionic_context_destroy(&ctx);
    return IONIC_RESULT_SUCCESS;
}

static int test_materialize_plan_without_all_shards_fails(void)
{
    ionic_context_t ctx;
    ionic_context_init(&ctx, ionic_cpu_device(0));

    struct ionic_planner *planner = ionic_planner_init(&ctx, 2, 0, 1);
    if (!planner)
        return IONIC_RESULT_FAILURE;

    struct ionic_tensor tensor = { .start = 0, .end = 100, .dtype = IONIC_DATA_TYPE_FLOAT32, .rank = 1, .shape = {10}, .file = "test.bin" };
    ionic_planner_shard(&ctx, planner, &tensor, IONIC_SHARDING_REPLICATED);

    ionic_sharding_plan_t plan = ionic_planner_materialize_plan(&ctx, planner);
    if (ctx.error.kind == IONIC_ERROR_SUCCESS)
        return IONIC_RESULT_FAILURE;
    if (plan.n != 0)
        return IONIC_RESULT_FAILURE;
    if (plan.tensors != NULL)
        return IONIC_RESULT_FAILURE;

    ionic_sharding_plan_destroy(&ctx, &plan);
    ionic_planner_destroy(planner);
    ionic_context_destroy(&ctx);
    return IONIC_RESULT_SUCCESS;
}

static int test_plan_specs_values(void)
{
    ionic_context_t ctx;
    ionic_context_init(&ctx, ionic_cpu_device(0));

    struct ionic_planner *planner = ionic_planner_init(&ctx, 1, 0, 1);
    if (!planner)
        return IONIC_RESULT_FAILURE;

    struct ionic_tensor tensor = { .start = 1784, .end = 2034239224, .dtype = IONIC_DATA_TYPE_FLOAT32, .rank = 1, .shape = {10}, .file = "model.safetensors" };
    ionic_planner_shard(&ctx, planner, &tensor, IONIC_SHARDING_REPLICATED);

    ionic_sharding_plan_t plan = ionic_planner_materialize_plan(&ctx, planner);
    if (ctx.error.kind != IONIC_ERROR_SUCCESS)
        return IONIC_RESULT_FAILURE;

    struct ionic_sharded_tensor_specs *specs = plan.tensors[0].specs;
    if (!specs)
        return IONIC_RESULT_FAILURE;

    if (specs->start != 1784)
        return IONIC_RESULT_FAILURE;
    if (specs->end != 2034239224)
        return IONIC_RESULT_FAILURE;
    if (specs->device.kind != IONIC_DEVICE_CPU)
        return IONIC_RESULT_FAILURE;
    if (specs->device.ordinal != 0)
        return IONIC_RESULT_FAILURE;

    ionic_sharding_plan_destroy(&ctx, &plan);
    ionic_planner_destroy(planner);
    ionic_context_destroy(&ctx);
    return IONIC_RESULT_SUCCESS;
}

static int test_plan_specs_is_loaded_initially_false(void)
{
    ionic_context_t ctx;
    ionic_context_init(&ctx, ionic_cpu_device(0));

    struct ionic_planner *planner = ionic_planner_init(&ctx, 1, 0, 1);
    if (!planner)
        return IONIC_RESULT_FAILURE;

    struct ionic_tensor tensor = { .start = 0, .end = 100, .dtype = IONIC_DATA_TYPE_FLOAT32, .rank = 1, .shape = {10}, .file = "test.bin" };
    ionic_planner_shard(&ctx, planner, &tensor, IONIC_SHARDING_REPLICATED);

    ionic_sharding_plan_t plan = ionic_planner_materialize_plan(&ctx, planner);
    if (ctx.error.kind != IONIC_ERROR_SUCCESS)
        return IONIC_RESULT_FAILURE;

    struct ionic_sharded_tensor_specs *specs = plan.tensors[0].specs;
    if (ionic_sharded_tensor_is_loaded(specs))
        return IONIC_RESULT_FAILURE;

    ionic_sharding_plan_destroy(&ctx, &plan);
    ionic_planner_destroy(planner);
    ionic_context_destroy(&ctx);
    return IONIC_RESULT_SUCCESS;
}

static int test_plan_specs_is_loaded_after_update(void)
{
    ionic_context_t ctx;
    ionic_context_init(&ctx, ionic_cpu_device(0));

    struct ionic_planner *planner = ionic_planner_init(&ctx, 1, 0, 1);
    if (!planner)
        return IONIC_RESULT_FAILURE;

    struct ionic_tensor tensor = { .start = 0, .end = 100, .dtype = IONIC_DATA_TYPE_FLOAT32, .rank = 1, .shape = {10}, .file = "test.bin" };
    ionic_planner_shard(&ctx, planner, &tensor, IONIC_SHARDING_REPLICATED);

    ionic_sharding_plan_t plan = ionic_planner_materialize_plan(&ctx, planner);
    if (ctx.error.kind != IONIC_ERROR_SUCCESS)
        return IONIC_RESULT_FAILURE;

    struct ionic_sharded_tensor_specs *specs = plan.tensors[0].specs;

    /* simulate partial load */
    atomic_fetch_add_explicit(&specs->loaded, 50, memory_order_release);
    if (ionic_sharded_tensor_is_loaded(specs))
        return IONIC_RESULT_FAILURE;

    /* complete the load */
    atomic_fetch_add_explicit(&specs->loaded, 50, memory_order_release);
    if (!ionic_sharded_tensor_is_loaded(specs))
        return IONIC_RESULT_FAILURE;

    ionic_sharding_plan_destroy(&ctx, &plan);
    ionic_planner_destroy(planner);
    ionic_context_destroy(&ctx);
    return IONIC_RESULT_SUCCESS;
}

static int test_plan_specs_world_size_greater_than_one(void)
{
    ionic_context_t ctx;
    ionic_context_init(&ctx, ionic_cpu_device(0));

    struct ionic_planner *planner = ionic_planner_init(&ctx, 1, 0, 4);
    if (!planner)
        return IONIC_RESULT_FAILURE;

    struct ionic_tensor tensor = { .start = 0, .end = 400, .dtype = IONIC_DATA_TYPE_FLOAT32, .rank = 1, .shape = {10}, .file = "test.bin" };
    ionic_planner_shard(&ctx, planner, &tensor, IONIC_SHARDING_REPLICATED);

    ionic_sharding_plan_t plan = ionic_planner_materialize_plan(&ctx, planner);
    if (ctx.error.kind != IONIC_ERROR_SUCCESS)
        return IONIC_RESULT_FAILURE;

    struct ionic_sharded_tensor_specs *specs = plan.tensors[0].specs;
    if (!specs)
        return IONIC_RESULT_FAILURE;

    /* all 4 ranks should have the same start/end for replicated */
    for (int i = 0; i < 4; ++i) {
        if (specs[i].start != 0)
            return IONIC_RESULT_FAILURE;
        if (specs[i].end != 400)
            return IONIC_RESULT_FAILURE;
        if (specs[i].device.ordinal != i)
            return IONIC_RESULT_FAILURE;
    }

    ionic_sharding_plan_destroy(&ctx, &plan);
    ionic_planner_destroy(planner);
    ionic_context_destroy(&ctx);
    return IONIC_RESULT_SUCCESS;
}

int main(void)
{
    if (test_planner_init() != IONIC_RESULT_SUCCESS)
        return IONIC_RESULT_FAILURE;
    if (test_planner_shard_replicated() != IONIC_RESULT_SUCCESS)
        return IONIC_RESULT_FAILURE;
    if (test_planner_shard_non_replicated_fails_on_world_size_1() != IONIC_RESULT_SUCCESS)
        return IONIC_RESULT_FAILURE;
    if (test_planner_too_many_shards() != IONIC_RESULT_SUCCESS)
        return IONIC_RESULT_FAILURE;
    if (test_materialize_plan_replicated() != IONIC_RESULT_SUCCESS)
        return IONIC_RESULT_FAILURE;
    if (test_materialize_plan_without_all_shards_fails() != IONIC_RESULT_SUCCESS)
        return IONIC_RESULT_FAILURE;
    if (test_plan_specs_values() != IONIC_RESULT_SUCCESS)
        return IONIC_RESULT_FAILURE;
    if (test_plan_specs_is_loaded_initially_false() != IONIC_RESULT_SUCCESS)
        return IONIC_RESULT_FAILURE;
    if (test_plan_specs_is_loaded_after_update() != IONIC_RESULT_SUCCESS)
        return IONIC_RESULT_FAILURE;
    if (test_plan_specs_world_size_greater_than_one() != IONIC_RESULT_SUCCESS)
        return IONIC_RESULT_FAILURE;

    return IONIC_RESULT_SUCCESS;
}
