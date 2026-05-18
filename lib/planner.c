#include <stdlib.h>
#include <string.h>
#include "ionic/error.h"
#include "ionic/ionic.h"
#include "ionic/logging.h"
#include "ionic/types.h"
#include "ionic/planner.h"
#include "ionic/utils.h"

#define IONIC_FILE_ALIGN_BYTES 4096u

static void compute_shard_bounds(size_t dim_size, int world_size, int rank, size_t *out_start, size_t *out_count) {
    size_t chunk = dim_size / world_size;
    size_t remainder = dim_size % world_size;
    size_t start = rank * chunk + (rank < (int)remainder ? rank : remainder);
    size_t count = chunk + (rank < (int)remainder ? 1 : 0);
    *out_start = start;
    *out_count = count;
}

static struct ionic_sharded_tensor ionic_sharding_rule_replicate(const struct ionic_tensor *tensor, struct ionic_device device, int world_size) {
    struct ionic_sharded_tensor sharded = {0};
    sharded.tensor = tensor;
    sharded.specs = calloc(world_size, sizeof(struct ionic_sharded_tensor_specs));

    for (int i = 0; i < world_size; ++i) {
        struct ionic_sharded_tensor_specs *specs = sharded.specs + i;
        specs->device.kind    = device.kind;
        specs->device.ordinal = i;
        specs->start          = tensor->start;
        specs->end            = tensor->end;
    }

    return sharded;
}

static struct ionic_sharded_tensor ionic_sharding_rule_rowwise(const struct ionic_tensor *tensor, struct ionic_device device, int world_size) {
    struct ionic_sharded_tensor sharded = {0};
    sharded.tensor = tensor;
    sharded.specs = calloc(world_size, sizeof(struct ionic_sharded_tensor_specs));

    size_t elem_size = ionic_dtype_size(tensor->dtype);
    size_t elems_per_row = 1;
    for (unsigned char d = 1; d < tensor->rank; ++d) {
        elems_per_row *= tensor->shape[d];
    }
    size_t row_bytes = elems_per_row * elem_size;

    for (int i = 0; i < world_size; ++i) {
        struct ionic_sharded_tensor_specs *specs = sharded.specs + i;
        size_t row_start, row_count;
        compute_shard_bounds(tensor->shape[0], world_size, i, &row_start, &row_count);

        specs->device.kind    = device.kind;
        specs->device.ordinal = i;
        specs->start          = tensor->start + row_start * row_bytes;
        specs->end            = specs->start + row_count * row_bytes;
    }

    return sharded;
}

static struct ionic_sharded_tensor ionic_sharding_rule_colwise(const struct ionic_tensor *tensor, struct ionic_device device, int world_size) {
    struct ionic_sharded_tensor sharded = {0};
    sharded.tensor = tensor;
    sharded.specs = calloc(world_size, sizeof(struct ionic_sharded_tensor_specs));

    size_t elem_size = ionic_dtype_size(tensor->dtype);
    unsigned char last_dim = tensor->rank > 0 ? tensor->rank - 1 : 0;
    size_t elems_per_col_slice = 1;
    for (unsigned char d = 0; d + 1 < tensor->rank; ++d) {
        elems_per_col_slice *= tensor->shape[d];
    }
    size_t col_bytes = elems_per_col_slice * elem_size;

    for (int i = 0; i < world_size; ++i) {
        struct ionic_sharded_tensor_specs *specs = sharded.specs + i;
        size_t col_start, col_count;
        compute_shard_bounds(tensor->shape[last_dim], world_size, i, &col_start, &col_count);

        specs->device.kind    = device.kind;
        specs->device.ordinal = i;
        specs->start          = tensor->start + col_start * col_bytes;
        specs->end            = specs->start + col_count * col_bytes;
    }

    return sharded;
}

struct ionic_planner *ionic_planner_init(struct ionic_context *ctx, size_t num_tensors, unsigned short rank, unsigned short world_size) {
    if(ionic_has_error(&ctx->error)) goto ko;

    struct ionic_planner *planner = malloc(sizeof(struct ionic_planner));
    if (!planner) goto oom;

    planner->n = num_tensors;
    planner->n_registered = 0;
    planner->rank = rank;
    planner->world_size = world_size;

    planner->infos = calloc(num_tensors, sizeof(struct ionic_sharding_info));
    if(!planner->infos) goto oom;

    IONIC_INFO(&ctx->logger, IONIC_EVENT_TAG_PLANNER,
        "initialized rank=%hu, world_size=%hu, tensors=%zu", planner->rank, planner->world_size, planner->n);
    return planner;

oom:
    ionic_set_error(ctx, IONIC_ERR(IONIC_ERROR_ALLOCATION_FAILED));
ko:
    return NULL;
}

void ionic_planner_destroy(ionic_planner_t *planner) {
    if(planner) {
        if(planner->infos) free(planner->infos);
        planner->infos = NULL;
        planner->n = 0;
        planner->n_registered = 0;
        planner->world_size = 0;
        planner->rank = 0;
    }
}

void ionic_sharding_plan_destroy(ionic_context_t *ctx, ionic_sharding_plan_t *plan) {
    IONIC_UNUSED(ctx);
    if (plan) {
        if (plan->tensors) {
            for (size_t i = 0; i < plan->n; ++i)
                if (plan->tensors[i].specs) free(plan->tensors[i].specs);

            free(plan->tensors);
        }
        plan->tensors = NULL;
        plan->n = 0;
        plan->world_size = 0;
    }
}

void ionic_planner_shard(ionic_context_t *ctx, ionic_planner_t *planner, const struct ionic_tensor *tensor, enum ionic_sharding_kind kind) {
    if (ionic_has_error(&ctx->error) || !planner || !tensor)
        return;

    if (planner->n_registered >= planner->n) {
        ionic_set_error(ctx, IONIC_ERR(IONIC_ERROR_PLANNER_INVALID_SHARDING));
        return;
    }

    if (planner->world_size == 1 && kind != IONIC_SHARDING_REPLICATED) {
        ionic_set_error(ctx, IONIC_ERR(IONIC_ERROR_PLANNER_INVALID_SHARDING));
        return;
    }

    struct ionic_sharding_info *info = planner->infos + planner->n_registered;
    info->tensor = tensor;
    info->kind = kind;
    planner->n_registered++;
}

ionic_sharding_plan_t ionic_planner_materialize_plan(ionic_context_t *ctx, ionic_planner_t *planner)
{
    ionic_sharding_plan_t out = {0};
    if (ionic_has_error(&ctx->error)) return out;

    if(!planner || planner->n_registered != planner->n) {
        IONIC_ERROR(&ctx->logger, IONIC_EVENT_TAG_PLANNER, "n_registered != n_tensors n_registered=%zu, n_tensors=%zu", planner->n_registered, planner->n);
        ionic_set_error(ctx, IONIC_ERR_WITH_MSG(IONIC_ERROR_PLANNER_INVALID_SHARDING, "uneven number of tensors and sharding rules"));
        return out;
    }

    out.tensors = malloc(planner->n * sizeof(struct ionic_sharded_tensor));
    if (!out.tensors) {
        ionic_set_error(ctx, IONIC_ERR(IONIC_ERROR_ALLOCATION_FAILED));
        return out;
    }

    size_t i = 0;
    for (; i < planner->n; ++i) {
        struct ionic_sharding_info *info = &planner->infos[i];
        struct ionic_sharded_tensor shard = {0};

        switch(info->kind) {
            case IONIC_SHARDING_REPLICATED:
                shard = ionic_sharding_rule_replicate(info->tensor, ctx->device, planner->world_size);
                break;
            case IONIC_SHARDING_ROWWISE:
                shard = ionic_sharding_rule_rowwise(info->tensor, ctx->device, planner->world_size);
                break;
            case IONIC_SHARDING_COLWISE:
                shard = ionic_sharding_rule_colwise(info->tensor, ctx->device, planner->world_size);
                break;
            default:
                ionic_set_error(ctx, IONIC_ERR_WITH_MSG(IONIC_ERROR_UNSUPPORTED, "unsupported sharding kind"));
                goto cleanup;
        }

        out.tensors[i] = shard;
    }

    out.world_size = planner->world_size;
    out.n = planner->n;
    return out;

cleanup:
    for (size_t j = 0; j < i; ++j) {
        if (out.tensors[j].specs) free(out.tensors[j].specs);
    }
    free(out.tensors);
    out.tensors = NULL;
    return out;
}
