#ifndef IONIC_PLANNER_H
#define IONIC_PLANNER_H

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
    enum ionic_sharding_kind kind;
    int fd;
};

struct ionic_sharded_tensor_specs {
    struct ionic_device device;
    size_t start;
    size_t end;
    void *dst;  // nullable
};

struct ionic_sharded_tensor {
    struct ionic_sharded_tensor_specs *specs;  // [world_size], specs for each rank
    const struct ionic_tensor *tensor;
    int fd;
};

struct ionic_sharding_plan {
    struct ionic_sharded_tensor *tensors;
    size_t n;
    size_t world_size;
};

struct ionic_planner {
    struct ionic_sharding_info *infos;
    size_t n;
    size_t n_registered;
    unsigned short world_size;
    unsigned short rank;

    void (*initialize)(struct ionic_context *, struct ionic_planner *);
    void (*destroy)(struct ionic_planner *);
    void (*execute)(struct ionic_context *, struct ionic_planner *, struct ionic_sharding_plan *);
};

typedef enum ionic_allocation_kind ionic_allocation_kind_t;
typedef enum ionic_sharding_kind ionic_sharding_kind_t;
typedef struct ionic_sharding_info ionic_sharding_info_t;
typedef struct ionic_sharding_specs ionic_sharding_specs_t;
typedef struct ionic_sharding_plan ionic_sharding_plan_t;
typedef struct ionic_planner ionic_planner_t;


#ifdef __IONIC_CUDA_ENABLED__
#include <cuda_runtime.h>

#define IONIC_PLANNER_CUDA_STAGING_COUNT 2

struct ionic_planner_cuda {
    struct ionic_planner base;
    cudaStream_t         stream;
    void                *staging[IONIC_PLANNER_CUDA_STAGING_COUNT];  // Pinned host buffers
    size_t               staging_size;                                // Per-buffer capacity
    cudaEvent_t          staging_done[IONIC_PLANNER_CUDA_STAGING_COUNT];  // Events per buffer
};

typedef struct ionic_planner_cuda ionic_planner_cuda_t;

void ionic_planner_cuda_single_gpu_init(struct ionic_context *, struct ionic_planner *);
void ionic_planner_cuda_single_gpu_destroy(struct ionic_planner *);
void ionic_planner_cuda_execute(struct ionic_context *, struct ionic_planner *, struct ionic_sharding_plan *);

#endif // __IONIC_CUDA_ENABLED__
#endif // IONIC_PLANNER_H