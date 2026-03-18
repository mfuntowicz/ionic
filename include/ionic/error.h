#ifndef IONIC_ERROR_H
#define IONIC_ERROR_H

#include <stddef.h>

enum ionic_error_kind {
    IONIC_ERROR_SUCCESS = 0,
    IONIC_ERROR_SYSTEM = 1,
    IONIC_ERROR_ALLOCATION_FAILED,
    IONIC_ERROR_BACKEND_INITIALIZATION_FAILED,
    IONIC_ERROR_SAFETENSORS_JSON_MALFORMED_HEADER,
    IONIC_ERROR_SAFETENSORS_UNKNOWN_DTYPE,
};

typedef enum ionic_error_kind ionic_error_kind_t;

struct ionic_error {
    enum ionic_error_kind kind;
    int res;
    char *what;
};

typedef struct ionic_error ionic_error_t;

#define IONIC_SUCCESS (ionic_error_t){ .kind = IONIC_ERROR_SUCCESS, .res = 0 }
#define IONIC_ERR(k) (ionic_error_t){ .kind = (k), .res = 0, .what = NULL }
#define IONIC_SYS_ERR(err) (ionic_error_t){ .kind = IONIC_ERROR_SYSTEM, .res = (err), .what = NULL }
#define IONIC_ERR_WITH_MSG(k, msg) (ionic_error_t){ .kind = (k), .res = 0, .what = (msg) }
#define IONIC_SYS_ERR_WITH_MSG(err, msg) (ionic_error_t){ .kind = IONIC_ERROR_SYSTEM, .res = (err), .what = (msg) }

#endif // IONIC_ERROR_H