#ifndef IONIC_PIPELINE_H
#define IONIC_PIPELINE_H

#include <ionic/types.h>
#include <stdbool.h>
#include <stdatomic.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/*
 * Async Tensor Status
 * ====================
 * 
 * Tracks loading state of a single tensor within a sharding plan.
 * FFI-safe and observable from Python/Rust futures.
 * 
 * State machine:
 * 
 *   PENDING ──► LOADING ──► TRANSFERRING ──► READY
 *                               │
 *                               ▼
 *                            FAILED
 */

typedef enum ionic_tensor_state {
    IONIC_TENSOR_STATE_PENDING      = 0,
    IONIC_TENSOR_STATE_LOADING      = 1,
    IONIC_TENSOR_STATE_TRANSFERRING = 2,
    IONIC_TENSOR_STATE_READY        = 3,
    IONIC_TENSOR_STATE_FAILED       = 4,
} ionic_tensor_state_t;

/*
 * Tensor status - 64 bytes, cache-aligned
 */
struct ionic_tensor_status {
    atomic_int  state;
    atomic_long bytes_loaded;

    void   *device_ptr;
    size_t offset;
    size_t size;
    int    fd;
    int    error_code;
    
    unsigned char _pad[12];
};

static_assert(sizeof(struct ionic_tensor_status) == 64, "ionic_tensor_status must be 64 bytes");

/*
 * Pipeline Configuration
 */
struct ionic_pipeline_config {
    size_t staging_buffer_size;
    unsigned staging_slot_count;
    unsigned io_uring_batch_size;
    unsigned max_inflight_reads;
};

static inline struct ionic_pipeline_config ionic_pipeline_config_default(struct ionic_device device) {
    return (struct ionic_pipeline_config) {
        .staging_buffer_size = 512 * 1024UL,   /* 512 KiB per slot */
        .staging_slot_count  = 64,              /* More parallel H2D */
        .io_uring_batch_size = 128,             /* Larger batch submits */
        .max_inflight_reads  = 256               /* Match io_uring depth */
    };
}

/*
 * Opaque handles (forward declarations)
 */
struct ionic_pipeline;
struct ionic_context;
struct ionic_sharding_plan;

typedef struct ionic_pipeline ionic_pipeline_t;

#ifdef __cplusplus
}
#endif

#endif /* IONIC_PIPELINE_H */
