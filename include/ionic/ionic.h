#ifndef IONIC_H
#define IONIC_H
#ifdef __cplusplus
#define NO_EXCEPT noexcept
extern "C" {
#else
#define NO_EXCEPT
#endif

#ifndef IONIC_EXTERN
    #ifndef IONIC_STATIC
        #ifdef _WIN32
            #ifdef __IONIC_EXPORTS__
                #define IONIC_EXTERN __declspec(dllexport)
            #else
                #define IONIC_EXTERN __declspec(dllimport)
            #endif
        #elif defined(__GNUC__) && __GNUC__ >= 4
            #define IONIC_EXTERN __attribute__((visibility("default")))
        #else
            #define IONIC_EXTERN
        #endif
    #else
        #define IONIC_EXTERN
    #endif
#endif

#ifndef IONIC_INLINE
#if defined(__cplusplus) && __cplusplus >= 202002L
#define IONIC_INLINE inline
#else
#define IONIC_INLINE static inline
#endif
#endif

#define IONIC_FALSE        0u
#define IONIC_UNUSED(expr) (void)(expr)

#if defined(_MSC_VER)
    #define __builtin_unreachable() __assume(0)
#endif

#define IONIC_MAJOR_VERSION 1
#define IONIC_MINOR_VERSION 0
#define IONIC_PATCH_VERSION 0
#define IONIC_VERSION (IONIC_MAJOR_VERSION * 10000U + IONIC_MINOR_VERSION * 100 + IONIC_PATCH_VERSION)

/* Returns the version of the library as a single integer as MAJOR * 10000 + MINOR * 100 + PATCH */
IONIC_INLINE unsigned ionic_version(void) NO_EXCEPT { return IONIC_VERSION; }

#include "error.h"
#include "logging.h"

/*
 * Initialise a logger from the IONIC_LOG environment variable.
 * If the variable is absent or unrecognised the level defaults to OFF.
 */
IONIC_EXTERN void ionic_logger_init(ionic_logger_t *) NO_EXCEPT;
IONIC_EXTERN void ionic_log(ionic_logger_t *, ionic_log_level_t, const char *tag, const char *fmt, ...) NO_EXCEPT;
IONIC_EXTERN unsigned char ionic_log_level_is_enabled(ionic_logger_t *, ionic_log_level_t) NO_EXCEPT;

#include "types.h"
#include "topology.h"

static inline struct ionic_device ionic_cpu_device(unsigned char ordinal) NO_EXCEPT { return (struct ionic_device) { .kind = IONIC_DEVICE_CPU, .ordinal = ordinal }; }
static inline struct ionic_device ionic_cuda_device(unsigned char ordinal) NO_EXCEPT { return (struct ionic_device) { .kind = IONIC_DEVICE_CUDA, .ordinal = ordinal }; } 

IONIC_EXTERN void ionic_set_host_allocator(struct ionic_context *, struct ionic_allocator) NO_EXCEPT;
IONIC_EXTERN void ionic_set_device_allocator(struct ionic_context *, struct ionic_allocator) NO_EXCEPT;
IONIC_EXTERN void *ionic_allocate_host(struct ionic_context *, size_t, enum ionic_allocation_kind) NO_EXCEPT;
IONIC_EXTERN void *ionic_allocate_device(struct ionic_context *, size_t, enum ionic_allocation_kind) NO_EXCEPT;
IONIC_EXTERN void ionic_free_host(struct ionic_context *, void *, enum ionic_allocation_kind) NO_EXCEPT;
IONIC_EXTERN void ionic_free_device(struct ionic_context *, void *, enum ionic_allocation_kind) NO_EXCEPT;

IONIC_EXTERN void ionic_barrier_wait(struct ionic_barrier *) NO_EXCEPT;

#include "pipeline.h"
#include "planner.h"

/* ═══════════════════════════════════════════════════════════════════════════
 * Pipeline API
 * ═══════════════════════════════════════════════════════════════════════════ */

IONIC_EXTERN struct ionic_pipeline *ionic_pipeline_init(struct ionic_context *, struct ionic_pipeline_config) NO_EXCEPT;
IONIC_EXTERN void ionic_pipeline_destroy(struct ionic_pipeline *) NO_EXCEPT;
IONIC_EXTERN int ionic_pipeline_execute_plan(struct ionic_context *, struct ionic_pipeline *, struct ionic_sharding_plan *, unsigned short rank) NO_EXCEPT;
IONIC_EXTERN ionic_tensor_state_t ionic_tensor_status_get_state(const struct ionic_tensor_status *) NO_EXCEPT;
IONIC_EXTERN size_t ionic_tensor_status_get_bytes_loaded(const struct ionic_tensor_status *) NO_EXCEPT;
IONIC_EXTERN size_t ionic_tensor_status_get_total_bytes(const struct ionic_tensor_status *) NO_EXCEPT;
IONIC_EXTERN bool ionic_tensor_status_is_complete(const struct ionic_tensor_status *) NO_EXCEPT;
IONIC_EXTERN bool ionic_tensor_status_has_error(const struct ionic_tensor_status *, int *err_code) NO_EXCEPT;
IONIC_EXTERN const struct ionic_tensor_status *ionic_pipeline_get_tensor_status(const struct ionic_pipeline *, size_t) NO_EXCEPT;
IONIC_EXTERN size_t ionic_pipeline_get_bytes_total(const struct ionic_pipeline *) NO_EXCEPT;
IONIC_EXTERN size_t ionic_pipeline_get_bytes_loaded(const struct ionic_pipeline *) NO_EXCEPT;

/* ═══════════════════════════════════════════════════════════════════════════
 * Backend Interface
 * ═══════════════════════════════════════════════════════════════════════════ */

/* I/O completion result */
struct ionic_io_completion {
    void   *staging_buffer;   /* Pointer to staging buffer containing data */
    size_t   bytes_read;      /* Number of bytes read */
    void    *userdata;        /* User data passed to submit_read */
    size_t   staging_slot;    /* Slot index for release_staging() */
};

struct ionic_backend {
    void (*destroy)(struct ionic_context *);
    
    /* Synchronous read (legacy) */
    size_t (*read)(struct ionic_context *, int fd, unsigned char *dst, size_t len, size_t offset);
    
    /* Async I/O operations */
    int (*submit_read)(struct ionic_context *, int fd, size_t offset, size_t len, void *userdata);
    size_t (*poll_completions)(struct ionic_context *, struct ionic_io_completion *completions, size_t max_completions);
    size_t (*get_inflight)(struct ionic_context *);
    void *(*acquire_staging)(struct ionic_context *, size_t len, size_t *slot);
    void (*release_staging)(struct ionic_context *, size_t slot);
};
typedef struct ionic_backend ionic_backend_t;

enum ionic_staging_slot_state {
    IONIC_SLOT_EMPTY = 0,
    IONIC_SLOT_FILLING,
    IONIC_SLOT_FILLED,
    IONIC_SLOT_COPYING,
    IONIC_SLOT_SCATTERING,
};

#ifdef __IONIC_CUDA_ENABLED__
#include <cuda_runtime.h>
#include <stdatomic.h>

struct ionic_staging_slot_cuda {
    void *staging_ptr;
    void *device_ptr;
    cudaEvent_t h2d_done;
    cudaEvent_t scatter_done;
    atomic_uint state;
    struct ionic_read_chunk *bound_chunk;
    size_t submitted_chunk_index;
};
#endif

struct ionic_context {
    struct ionic_error  error;
    struct ionic_device device;
    struct ionic_allocator halloc;
    struct ionic_allocator dalloc;
    struct ionic_logger logger;
    struct ionic_topology topology;
    struct ionic_backend *backend;
};

typedef struct ionic_context ionic_context_t;

IONIC_EXTERN void ionic_allocator_init(struct ionic_context *, struct ionic_device) NO_EXCEPT;
IONIC_EXTERN void ionic_context_init(struct ionic_context *, struct ionic_device) NO_EXCEPT;
IONIC_EXTERN void ionic_context_destroy(struct ionic_context *) NO_EXCEPT;

IONIC_EXTERN IONIC_INLINE void ionic_set_error(struct ionic_context *ctx, struct ionic_error error) NO_EXCEPT { ctx->error = error; };
IONIC_EXTERN IONIC_INLINE unsigned char ionic_has_error(struct ionic_error *error) NO_EXCEPT { return error->kind != IONIC_ERROR_SUCCESS; }

#include "planner.h"

IONIC_EXTERN struct ionic_planner *ionic_planner_init(struct ionic_context *, size_t num_tensors, unsigned short rank, unsigned short world_size) NO_EXCEPT;
IONIC_EXTERN void ionic_planner_destroy(struct ionic_planner *) NO_EXCEPT;
IONIC_EXTERN void ionic_planner_register_sharding(struct ionic_context *, struct ionic_planner *, const struct ionic_tensor *, enum ionic_sharding_kind, int) NO_EXCEPT;
IONIC_EXTERN void ionic_planner_execute_plan(struct ionic_context *, struct ionic_planner *, struct ionic_sharding_plan *) NO_EXCEPT;
IONIC_EXTERN struct ionic_sharding_plan ionic_planner_materialize_plan(struct ionic_context *, struct ionic_planner *, int fd) NO_EXCEPT;
IONIC_EXTERN void ionic_sharding_plan_destroy(struct ionic_context *, struct ionic_sharding_plan *) NO_EXCEPT;

#ifdef __cplusplus
}
#endif

#endif // IONIC_H