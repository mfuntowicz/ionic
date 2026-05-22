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
 * Initialize a logger from the IONIC_LOG environment variable.
 * If the variable is absent or not recognized, the level defaults to OFF.
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

#define IONIC_GROUP_MAX_LEN 32

struct ionic_context {
    struct ionic_error  error;
    struct ionic_device device;
    char                group[IONIC_GROUP_MAX_LEN];
    unsigned short      rank;
    unsigned short      world_size;
    struct ionic_group *pg;
    struct ionic_allocator halloc;
    struct ionic_allocator dalloc;
    struct ionic_logger logger;
    struct ionic_topology topology;
};

typedef struct ionic_context ionic_context_t;

IONIC_EXTERN void ionic_allocator_init(struct ionic_context *, struct ionic_device) NO_EXCEPT;
IONIC_EXTERN void ionic_context_init(struct ionic_context *, struct ionic_device, const char *group, unsigned short rank, unsigned short world_size) NO_EXCEPT;
IONIC_EXTERN void ionic_context_destroy(struct ionic_context *) NO_EXCEPT;

IONIC_EXTERN IONIC_INLINE void ionic_set_error(struct ionic_context *ctx, struct ionic_error error) NO_EXCEPT { ctx->error = error; };
IONIC_EXTERN IONIC_INLINE unsigned char ionic_has_error(struct ionic_error *error) NO_EXCEPT { return error->kind != IONIC_ERROR_SUCCESS; }

#include "planner.h"
IONIC_EXTERN struct ionic_planner *ionic_planner_init(struct ionic_context *, size_t num_tensors, unsigned short rank, unsigned short world_size) NO_EXCEPT;
IONIC_EXTERN void ionic_planner_destroy(struct ionic_planner *) NO_EXCEPT;
IONIC_EXTERN void ionic_planner_shard(struct ionic_context *, struct ionic_planner *, const struct ionic_tensor *, enum ionic_sharding_kind) NO_EXCEPT;
IONIC_EXTERN struct ionic_sharding_plan ionic_planner_materialize_plan(struct ionic_context *, struct ionic_planner *) NO_EXCEPT;
IONIC_EXTERN void ionic_sharding_plan_destroy(struct ionic_context *, struct ionic_sharding_plan *) NO_EXCEPT;


/* ═══════════════════════════════════════════════════════════════════════════
 * Pipeline API
 * ═══════════════════════════════════════════════════════════════════════════ */
#include "pipeline.h"
IONIC_EXTERN struct ionic_pipeline *ionic_pipeline_probe(struct ionic_context *, unsigned short world_size);
IONIC_EXTERN void ionic_pipeline_init(struct ionic_context *, struct ionic_pipeline *, char * const *, size_t, size_t) NO_EXCEPT;
IONIC_EXTERN void ionic_pipeline_destroy(struct ionic_pipeline *) NO_EXCEPT;
IONIC_EXTERN size_t ionic_pipeline_execute(struct ionic_context *, struct ionic_pipeline *, const struct ionic_sharding_plan *, unsigned short rank) NO_EXCEPT;

#ifdef __cplusplus
}
#endif

#endif // IONIC_H