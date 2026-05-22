#ifndef IONIC_PLANNER_H
#define IONIC_PLANNER_H

#include <stdatomic.h>
#include "ionic/types.h"

#define IONIC_EVENT_TAG_PLANNER "planner"

enum ionic_sharding_kind {
    IONIC_SHARDING_COLWISE = 0,
    IONIC_SHARDING_ROWWISE = 1,
    IONIC_SHARDING_REPLICATED,
    IONIC_SHARDING_EXPERT,
    IONIC_SHARDING_EXPERT_TP,
};

static const char* IONIC_SHARDING_KIND_NAMES[] = {
    [IONIC_SHARDING_COLWISE] = "COLWISE",
    [IONIC_SHARDING_ROWWISE] = "ROWWISE",
    [IONIC_SHARDING_REPLICATED] = "REPLICATED",
    [IONIC_SHARDING_EXPERT] = "EXPERT",
    [IONIC_SHARDING_EXPERT_TP] = "EXPERT_TP",
};


struct ionic_sharding_info {
    const struct ionic_tensor *tensor;
    enum ionic_sharding_kind  kind;
};

struct ionic_sharded_tensor_specs {
    struct ionic_device device;
    size_t              start;
    size_t              end;
    void                *dst;  // nullable
    atomic_size_t       loaded;
};

IONIC_INLINE ionic_bool_t ionic_sharded_tensor_is_loaded(const struct ionic_sharded_tensor_specs *specs) NO_EXCEPT {
    return atomic_load_explicit(&specs->loaded, memory_order_acquire) == (specs->end - specs->start);
}

struct ionic_sharded_tensor {
    struct ionic_sharded_tensor_specs *specs;  // [world_size], specs for each rank
    const struct ionic_tensor         *tensor;
};

struct ionic_sharding_plan {
    struct ionic_sharded_tensor *tensors;
    size_t                      n;
    size_t                      world_size;
};

struct ionic_planner {
    struct ionic_sharding_info *infos;
    size_t                     n;
    size_t                     n_registered;
    unsigned short             world_size;
    unsigned short             rank;
};

typedef enum ionic_allocation_kind ionic_allocation_kind_t;
typedef enum ionic_sharding_kind ionic_sharding_kind_t;
typedef struct ionic_sharding_info ionic_sharding_info_t;
typedef struct ionic_sharding_specs ionic_sharding_specs_t;
typedef struct ionic_sharding_plan ionic_sharding_plan_t;
typedef struct ionic_planner ionic_planner_t;

#endif // IONIC_PLANNER_H