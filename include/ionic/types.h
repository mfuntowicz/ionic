#ifndef IONIC_TYPES_H
#define IONIC_TYPES_H

#include <assert.h>
#include <stddef.h>

#ifndef IONIC_MAX_RANK
#define IONIC_MAX_RANK 8
#endif

enum ionic_data_type {
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
};
typedef enum ionic_data_type ionic_data_type_t;

struct ionic_tensor {
    size_t start;
    size_t end;
    ionic_data_type_t dtype;
    unsigned int shape[IONIC_MAX_RANK];
    unsigned short file;
    unsigned char rank;
    unsigned char padding[8];
};
typedef struct ionic_tensor ionic_tensor_t;
static_assert(sizeof(ionic_tensor_t) == 64, "ionic_tensor_t must be cache line sized");

#endif // IONIC_TYPES_H