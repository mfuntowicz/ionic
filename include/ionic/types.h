#ifndef IONIC_TYPES_H
#define IONIC_TYPES_H

#include <assert.h>
#include <stddef.h>
#include <stdatomic.h>


typedef enum ionic_device_kind {
    IONIC_DEVICE_CPU,
    IONIC_DEVICE_CUDA
} ionic_device_kind_t;

static const char *IONIC_DEVICE_LITERAL[] = {
    [IONIC_DEVICE_CPU] = "CPU",
    [IONIC_DEVICE_CUDA] = "CUDA"
};

typedef struct ionic_device {
    enum ionic_device_kind kind;
    unsigned char ordinal;
} ionic_device_t;

enum ionic_allocation_kind {
    IONIC_ALLOC_DEVICE,
    IONIC_ALLOC_STAGING,
};

struct ionic_context;

typedef struct ionic_allocator {
    void *(*allocate)(struct ionic_context *ctx, size_t size, enum ionic_allocation_kind kind);
    void (*free)(struct ionic_context *ctx, void *ptr, enum ionic_allocation_kind kind);
} ionic_allocator_t;

typedef struct ionic_barrier {
    atomic_uint   steps;
    atomic_uchar  ready; // max 16 devices per hosts
    unsigned char total; // max 16 devices per hosts
} ionic_barrier_t;

typedef enum ionic_data_type {
    IONIC_DATA_TYPE_BOOL,
    IONIC_DATA_TYPE_BFLOAT16,
    IONIC_DATA_TYPE_COMPLEX,
    IONIC_DATA_TYPE_FLOAT4,
    IONIC_DATA_TYPE_FLOAT6_E2M3,
    IONIC_DATA_TYPE_FLOAT6_E3M2,
    IONIC_DATA_TYPE_FLOAT8_E5M2,
    IONIC_DATA_TYPE_FLOAT8_E4M3,
    IONIC_DATA_TYPE_FLOAT8_E8M0,
    IONIC_DATA_TYPE_FLOAT16,
    IONIC_DATA_TYPE_FLOAT32,
    IONIC_DATA_TYPE_FLOAT64,
    IONIC_DATA_TYPE_SIGNED_INT4,
    IONIC_DATA_TYPE_SIGNED_INT8,
    IONIC_DATA_TYPE_SIGNED_INT16,
    IONIC_DATA_TYPE_SIGNED_INT32,
    IONIC_DATA_TYPE_SIGNED_INT64,
    IONIC_DATA_TYPE_UNSIGNED_INT4,
    IONIC_DATA_TYPE_UNSIGNED_INT8,
    IONIC_DATA_TYPE_UNSIGNED_INT16,
    IONIC_DATA_TYPE_UNSIGNED_INT32,
    IONIC_DATA_TYPE_UNSIGNED_INT64,
    IONIC_DATA_TYPE_UNKNOWN
} ionic_data_type_t;

#ifndef IONIC_MAX_RANK
#define IONIC_MAX_RANK 8
#endif

typedef struct ionic_tensor {
    size_t start;
    size_t end;
    ionic_data_type_t dtype;
    unsigned int shape[IONIC_MAX_RANK];
    unsigned short file;
    unsigned char rank;
    unsigned char padding[8];
} ionic_tensor_t;

static_assert(sizeof(ionic_tensor_t) == 64, "ionic_tensor_t must be cache line sized");

#endif // IONIC_TYPES_H