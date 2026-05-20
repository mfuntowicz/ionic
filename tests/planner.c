#include <string.h>
#include "ionic/ionic.h"
#include "ionic/planner.h"
#include "helpers.h"

static int test_planner_init(void)
{
    ionic_context_t ctx;
    ionic_context_init(&ctx, ionic_cpu_device(0), NULL);

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
    ionic_context_init(&ctx, ionic_cpu_device(0), NULL);

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
    ionic_context_init(&ctx, ionic_cpu_device(0), NULL);

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
    ionic_context_init(&ctx, ionic_cpu_device(0), NULL);

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
    ionic_context_init(&ctx, ionic_cpu_device(0), NULL);

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
    ionic_context_init(&ctx, ionic_cpu_device(0), NULL);

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
    ionic_context_init(&ctx, ionic_cpu_device(0), NULL);

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
    ionic_context_init(&ctx, ionic_cpu_device(0), NULL);

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
    ionic_context_init(&ctx, ionic_cpu_device(0), NULL);

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
    ionic_context_init(&ctx, ionic_cpu_device(0), NULL);

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

static int test_materialize_plan_rowwise(void)
{
    ionic_context_t ctx;
    ionic_context_init(&ctx, ionic_cpu_device(0), NULL);

    struct ionic_planner *planner = ionic_planner_init(&ctx, 1, 0, 2);
    if (!planner)
        return IONIC_RESULT_FAILURE;

    /* 4 rows, 10 cols, float32 = 4 bytes -> 40 elements, 160 bytes total */
    struct ionic_tensor tensor = {
        .start = 0, .end = 160, .dtype = IONIC_DATA_TYPE_FLOAT32,
        .rank = 2, .shape = {4, 10}, .file = "test.bin"
    };
    ionic_planner_shard(&ctx, planner, &tensor, IONIC_SHARDING_ROWWISE);

    ionic_sharding_plan_t plan = ionic_planner_materialize_plan(&ctx, planner);
    if (ctx.error.kind != IONIC_ERROR_SUCCESS)
        return IONIC_RESULT_FAILURE;
    if (plan.n != 1)
        return IONIC_RESULT_FAILURE;
    if (plan.world_size != 2)
        return IONIC_RESULT_FAILURE;

    struct ionic_sharded_tensor_specs *specs = plan.tensors[0].specs;
    if (!specs)
        return IONIC_RESULT_FAILURE;

    /* rank 0: rows 0-1 -> 2 * 10 * 4 = 80 bytes */
    if (specs[0].start != 0)
        return IONIC_RESULT_FAILURE;
    if (specs[0].end != 80)
        return IONIC_RESULT_FAILURE;

    /* rank 1: rows 2-3 -> 2 * 10 * 4 = 80 bytes */
    if (specs[1].start != 80)
        return IONIC_RESULT_FAILURE;
    if (specs[1].end != 160)
        return IONIC_RESULT_FAILURE;

    ionic_sharding_plan_destroy(&ctx, &plan);
    ionic_planner_destroy(planner);
    ionic_context_destroy(&ctx);
    return IONIC_RESULT_SUCCESS;
}

static int test_materialize_plan_rowwise_uneven(void)
{
    ionic_context_t ctx;
    ionic_context_init(&ctx, ionic_cpu_device(0), NULL);

    struct ionic_planner *planner = ionic_planner_init(&ctx, 1, 0, 3);
    if (!planner)
        return IONIC_RESULT_FAILURE;

    /* 5 rows, 4 cols, float32 = 4 bytes -> 20 elements, 80 bytes total */
    struct ionic_tensor tensor = {
        .start = 0, .end = 80, .dtype = IONIC_DATA_TYPE_FLOAT32,
        .rank = 2, .shape = {5, 4}, .file = "test.bin"
    };
    ionic_planner_shard(&ctx, planner, &tensor, IONIC_SHARDING_ROWWISE);

    ionic_sharding_plan_t plan = ionic_planner_materialize_plan(&ctx, planner);
    if (ctx.error.kind != IONIC_ERROR_SUCCESS)
        return IONIC_RESULT_FAILURE;

    struct ionic_sharded_tensor_specs *specs = plan.tensors[0].specs;
    if (!specs)
        return IONIC_RESULT_FAILURE;

    /* 5 rows / 3 ranks -> chunk=1, remainder=2 */
    /* rank 0: 2 rows -> 2 * 4 * 4 = 32 bytes */
    if (specs[0].start != 0 || specs[0].end != 32)
        return IONIC_RESULT_FAILURE;
    /* rank 1: 2 rows -> 2 * 4 * 4 = 32 bytes */
    if (specs[1].start != 32 || specs[1].end != 64)
        return IONIC_RESULT_FAILURE;
    /* rank 2: 1 row -> 1 * 4 * 4 = 16 bytes */
    if (specs[2].start != 64 || specs[2].end != 80)
        return IONIC_RESULT_FAILURE;

    ionic_sharding_plan_destroy(&ctx, &plan);
    ionic_planner_destroy(planner);
    ionic_context_destroy(&ctx);
    return IONIC_RESULT_SUCCESS;
}

static int test_materialize_plan_colwise(void)
{
    ionic_context_t ctx;
    ionic_context_init(&ctx, ionic_cpu_device(0), NULL);

    struct ionic_planner *planner = ionic_planner_init(&ctx, 1, 0, 2);
    if (!planner)
        return IONIC_RESULT_FAILURE;

    /* 4 rows, 10 cols, float32 = 4 bytes -> 40 elements, 160 bytes total */
    struct ionic_tensor tensor = {
        .start = 0, .end = 160, .dtype = IONIC_DATA_TYPE_FLOAT32,
        .rank = 2, .shape = {4, 10}, .file = "test.bin"
    };
    ionic_planner_shard(&ctx, planner, &tensor, IONIC_SHARDING_COLWISE);

    ionic_sharding_plan_t plan = ionic_planner_materialize_plan(&ctx, planner);
    if (ctx.error.kind != IONIC_ERROR_SUCCESS)
        return IONIC_RESULT_FAILURE;
    if (plan.n != 1)
        return IONIC_RESULT_FAILURE;
    if (plan.world_size != 2)
        return IONIC_RESULT_FAILURE;

    struct ionic_sharded_tensor_specs *specs = plan.tensors[0].specs;
    if (!specs)
        return IONIC_RESULT_FAILURE;

    /* rank 0: cols 0-4 -> 5 * 4 * 4 = 80 bytes (column-major assumption) */
    if (specs[0].start != 0)
        return IONIC_RESULT_FAILURE;
    if (specs[0].end != 80)
        return IONIC_RESULT_FAILURE;

    /* rank 1: cols 5-9 -> 5 * 4 * 4 = 80 bytes */
    if (specs[1].start != 80)
        return IONIC_RESULT_FAILURE;
    if (specs[1].end != 160)
        return IONIC_RESULT_FAILURE;

    ionic_sharding_plan_destroy(&ctx, &plan);
    ionic_planner_destroy(planner);
    ionic_context_destroy(&ctx);
    return IONIC_RESULT_SUCCESS;
}

static int test_materialize_plan_colwise_1d(void)
{
    ionic_context_t ctx;
    ionic_context_init(&ctx, ionic_cpu_device(0), NULL);

    struct ionic_planner *planner = ionic_planner_init(&ctx, 1, 0, 3);
    if (!planner)
        return IONIC_RESULT_FAILURE;

    /* 1D tensor: 10 elements, float32 = 4 bytes -> 40 bytes total */
    struct ionic_tensor tensor = {
        .start = 0, .end = 40, .dtype = IONIC_DATA_TYPE_FLOAT32,
        .rank = 1, .shape = {10}, .file = "test.bin"
    };
    ionic_planner_shard(&ctx, planner, &tensor, IONIC_SHARDING_COLWISE);

    ionic_sharding_plan_t plan = ionic_planner_materialize_plan(&ctx, planner);
    if (ctx.error.kind != IONIC_ERROR_SUCCESS)
        return IONIC_RESULT_FAILURE;

    struct ionic_sharded_tensor_specs *specs = plan.tensors[0].specs;
    if (!specs)
        return IONIC_RESULT_FAILURE;

    /* 10 elements / 3 ranks -> chunk=3, remainder=1 */
    /* rank 0: 4 elements -> 16 bytes */
    if (specs[0].start != 0 || specs[0].end != 16)
        return IONIC_RESULT_FAILURE;
    /* rank 1: 3 elements -> 12 bytes */
    if (specs[1].start != 16 || specs[1].end != 28)
        return IONIC_RESULT_FAILURE;
    /* rank 2: 3 elements -> 12 bytes */
    if (specs[2].start != 28 || specs[2].end != 40)
        return IONIC_RESULT_FAILURE;

    ionic_sharding_plan_destroy(&ctx, &plan);
    ionic_planner_destroy(planner);
    ionic_context_destroy(&ctx);
    return IONIC_RESULT_SUCCESS;
}

static int test_materialize_plan_unsupported_sharding(void)
{
    ionic_context_t ctx;
    ionic_context_init(&ctx, ionic_cpu_device(0), NULL);

    struct ionic_planner *planner = ionic_planner_init(&ctx, 1, 0, 2);
    if (!planner)
        return IONIC_RESULT_FAILURE;

    struct ionic_tensor tensor = {
        .start = 0, .end = 100, .dtype = IONIC_DATA_TYPE_FLOAT32,
        .rank = 1, .shape = {10}, .file = "test.bin"
    };
    ionic_planner_shard(&ctx, planner, &tensor, IONIC_SHARDING_EXPERT);

    ionic_sharding_plan_t plan = ionic_planner_materialize_plan(&ctx, planner);
    if (ctx.error.kind == IONIC_ERROR_SUCCESS)
        return IONIC_RESULT_FAILURE;
    if (plan.tensors != NULL)
        return IONIC_RESULT_FAILURE;

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
    if (test_materialize_plan_rowwise() != IONIC_RESULT_SUCCESS)
        return IONIC_RESULT_FAILURE;
    if (test_materialize_plan_rowwise_uneven() != IONIC_RESULT_SUCCESS)
        return IONIC_RESULT_FAILURE;
    if (test_materialize_plan_colwise() != IONIC_RESULT_SUCCESS)
        return IONIC_RESULT_FAILURE;
    if (test_materialize_plan_colwise_1d() != IONIC_RESULT_SUCCESS)
        return IONIC_RESULT_FAILURE;
    if (test_materialize_plan_unsupported_sharding() != IONIC_RESULT_SUCCESS)
        return IONIC_RESULT_FAILURE;

    return IONIC_RESULT_SUCCESS;
}
