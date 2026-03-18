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

IONIC_EXTERN IONIC_INLINE unsigned char ionic_has_error(ionic_error_t *error) NO_EXCEPT { return error->kind != IONIC_ERROR_SUCCESS; }

#include "logging.h"

/*
 * Initialise a logger from the IONIC_LOG environment variable.
 * If the variable is absent or unrecognised the level defaults to OFF.
 */
IONIC_EXTERN void ionic_logger_init(ionic_logger_t *) NO_EXCEPT;
IONIC_EXTERN void ionic_log(ionic_logger_t *logger, ionic_log_level_t level, const char *tag, const char *fmt, ...) NO_EXCEPT;
IONIC_EXTERN unsigned char ionic_log_level_is_enabled(ionic_logger_t *, ionic_log_level_t) NO_EXCEPT;

#include "types.h"
#include "topology.h"

struct ionic_backend {
    void (*destroy)(struct ionic_context *ctx);
    size_t (*read)(struct ionic_context *ctx, int fd, unsigned char *dst, size_t len, size_t offset);
};
typedef struct ionic_backend ionic_backend_t;

struct ionic_context {
    struct ionic_logger logger;
    struct ionic_error  error;
    struct ionic_topology topology;
    struct ionic_backend *backend;
};

typedef struct ionic_context ionic_context_t;

IONIC_EXTERN void ionic_context_init(ionic_context_t *) NO_EXCEPT;
IONIC_EXTERN void ionic_context_destroy(ionic_context_t *) NO_EXCEPT;

#ifdef __cplusplus
}
#endif

#endif // IONIC_H