#ifndef IONIC_ERROR_H
#define IONIC_ERROR_H

#include <stddef.h>

enum ionic_error_code {
    IONIC_ERROR_CODE_SUCCESS = 0,
};

typedef enum ionic_error_code ionic_error_code_t;

struct ionic_error {
    enum ionic_error_code kind;
    const char *what;
};

typedef struct ionic_error ionic_error_t;

#define IONIC_SUCCESS (ionic_error_t){ .kind = IONIC_ERROR_CODE_SUCCESS, .what = NULL }

#endif // IONIC_ERROR_H