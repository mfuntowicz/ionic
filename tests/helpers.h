#ifndef IONIC_TESTS_HELPERS_H
#define IONIC_TESTS_HELPERS_H

#ifdef __cplusplus
extern "C" {
#endif

#include <ionic/ionic.h>

enum t_ionic_result {
    IONIC_RESULT_SUCCESS = 0,
    IONIC_RESULT_FAILURE = 1,
    IONIC_RESULT_SKIP
};

#ifdef __cplusplus
}
#endif

#endif // IONIC_TESTS_HELPERS_H