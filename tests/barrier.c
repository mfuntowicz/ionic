#include <assert.h>
#include <stdatomic.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>

#include <ionic/ionic.h>
#include <ionic/types.h>
#include "helpers.h"

extern struct ionic_barrier *ionic_barrier_create(struct ionic_context *ctx, const char *identifier, unsigned char count);
extern void ionic_barrier_close(struct ionic_barrier *barrier);
extern void ionic_barrier_destroy(struct ionic_barrier *barrier);
extern void ionic_barrier_wait(struct ionic_barrier *barrier);

static char test_shm_name[IONIC_BARRIER_MAX_IDENT];

static void make_shm_name(void) {
    snprintf(test_shm_name, IONIC_BARRIER_MAX_IDENT, "/ionic_test_barrier_%d", getpid());
}

static int test_barrier_create_and_destroy(void) {
    make_shm_name();

    struct ionic_context ctx;
    ionic_context_init(&ctx, ionic_cpu_device(0));
    if (ionic_has_error(&ctx.error))
        return IONIC_RESULT_FAILURE;

    struct ionic_barrier *barrier = ionic_barrier_create(&ctx, test_shm_name, 4);
    if (!barrier)
        return IONIC_RESULT_FAILURE;

    if (atomic_load_explicit(&barrier->steps, memory_order_acquire) != 0)
        return IONIC_RESULT_FAILURE;

    if (atomic_load_explicit(&barrier->ready, memory_order_acquire) != 0)
        return IONIC_RESULT_FAILURE;

    if (barrier->total != 4)
        return IONIC_RESULT_FAILURE;

    if (strncmp(barrier->identifier, test_shm_name, IONIC_BARRIER_MAX_IDENT) != 0)
        return IONIC_RESULT_FAILURE;

    ionic_barrier_destroy(barrier);
    ionic_context_destroy(&ctx);
    return IONIC_RESULT_SUCCESS;
}

static int test_barrier_single_wait(void) {
    make_shm_name();

    struct ionic_context ctx;
    ionic_context_init(&ctx, ionic_cpu_device(0));
    if (ionic_has_error(&ctx.error))
        return IONIC_RESULT_FAILURE;

    struct ionic_barrier *barrier = ionic_barrier_create(&ctx, test_shm_name, 1);
    if (!barrier)
        return IONIC_RESULT_FAILURE;

    ionic_barrier_wait(barrier);

    if (atomic_load_explicit(&barrier->steps, memory_order_acquire) != 1)
        return IONIC_RESULT_FAILURE;

    if (atomic_load_explicit(&barrier->ready, memory_order_acquire) != 0)
        return IONIC_RESULT_FAILURE;

    ionic_barrier_destroy(barrier);
    ionic_context_destroy(&ctx);
    return IONIC_RESULT_SUCCESS;
}

static int test_barrier_multi_process(void) {
    make_shm_name();

    struct ionic_context ctx;
    ionic_context_init(&ctx, ionic_cpu_device(0));
    if (ionic_has_error(&ctx.error))
        return IONIC_RESULT_FAILURE;

    const unsigned char n_procs = 4;
    struct ionic_barrier *barrier = ionic_barrier_create(&ctx, test_shm_name, n_procs);
    if (!barrier)
        return IONIC_RESULT_FAILURE;

    for (unsigned char i = 1; i < n_procs; i++) {
        pid_t pid = fork();
        if (pid < 0)
            return IONIC_RESULT_FAILURE;

        if (pid == 0) {
            ionic_barrier_wait(barrier);
            ionic_barrier_close(barrier);
            _exit(IONIC_RESULT_SUCCESS);
        }
    }

    ionic_barrier_wait(barrier);

    for (unsigned char i = 1; i < n_procs; i++) {
        int status;
        if (wait(&status) < 0)
            return IONIC_RESULT_FAILURE;
        if (!WIFEXITED(status) || WEXITSTATUS(status) != IONIC_RESULT_SUCCESS)
            return IONIC_RESULT_FAILURE;
    }

    if (atomic_load_explicit(&barrier->steps, memory_order_acquire) != 1)
        return IONIC_RESULT_FAILURE;

    ionic_barrier_destroy(barrier);
    ionic_context_destroy(&ctx);
    return IONIC_RESULT_SUCCESS;
}

static int test_barrier_join_existing(void) {
    make_shm_name();

    struct ionic_context ctx;
    ionic_context_init(&ctx, ionic_cpu_device(0));
    if (ionic_has_error(&ctx.error))
        return IONIC_RESULT_FAILURE;

    struct ionic_barrier *barrier = ionic_barrier_create(&ctx, test_shm_name, 2);
    if (!barrier)
        return IONIC_RESULT_FAILURE;

    pid_t pid = fork();
    if (pid < 0) {
        ionic_barrier_destroy(barrier);
        ionic_context_destroy(&ctx);
        return IONIC_RESULT_FAILURE;
    }

    if (pid == 0) {
        struct ionic_context child_ctx;
        ionic_context_init(&child_ctx, ionic_cpu_device(0));
        struct ionic_barrier *joined = ionic_barrier_create(&child_ctx, test_shm_name, 2);
        if (!joined) {
            ionic_context_destroy(&child_ctx);
            _exit(IONIC_RESULT_FAILURE);
        }

        if (joined->total != 2) {
            ionic_barrier_close(joined);
            ionic_context_destroy(&child_ctx);
            _exit(IONIC_RESULT_FAILURE);
        }

        ionic_barrier_wait(joined);
        ionic_barrier_close(joined);
        ionic_context_destroy(&child_ctx);
        _exit(IONIC_RESULT_SUCCESS);
    }

    ionic_barrier_wait(barrier);

    int status;
    if (wait(&status) < 0)
        return IONIC_RESULT_FAILURE;
    if (!WIFEXITED(status) || WEXITSTATUS(status) != IONIC_RESULT_SUCCESS)
        return IONIC_RESULT_FAILURE;

    ionic_barrier_destroy(barrier);
    ionic_context_destroy(&ctx);
    return IONIC_RESULT_SUCCESS;
}

static int test_barrier_multiple_waits(void) {
    make_shm_name();

    struct ionic_context ctx;
    ionic_context_init(&ctx, ionic_cpu_device(0));
    if (ionic_has_error(&ctx.error))
        return IONIC_RESULT_FAILURE;

    const unsigned char n_procs = 3;
    struct ionic_barrier *barrier = ionic_barrier_create(&ctx, test_shm_name, n_procs);
    if (!barrier)
        return IONIC_RESULT_FAILURE;

    for (unsigned char i = 1; i < n_procs; i++) {
        pid_t pid = fork();
        if (pid < 0)
            return IONIC_RESULT_FAILURE;

        if (pid == 0) {
            ionic_barrier_wait(barrier);
            ionic_barrier_wait(barrier);
            ionic_barrier_close(barrier);
            _exit(IONIC_RESULT_SUCCESS);
        }
    }

    ionic_barrier_wait(barrier);

    if (atomic_load_explicit(&barrier->steps, memory_order_acquire) != 1)
        return IONIC_RESULT_FAILURE;

    ionic_barrier_wait(barrier);

    if (atomic_load_explicit(&barrier->steps, memory_order_acquire) != 2)
        return IONIC_RESULT_FAILURE;

    for (unsigned char i = 1; i < n_procs; i++) {
        int status;
        if (wait(&status) < 0)
            return IONIC_RESULT_FAILURE;
        if (!WIFEXITED(status) || WEXITSTATUS(status) != IONIC_RESULT_SUCCESS)
            return IONIC_RESULT_FAILURE;
    }

    ionic_barrier_destroy(barrier);
    ionic_context_destroy(&ctx);
    return IONIC_RESULT_SUCCESS;
}

int main(void) {
    if (test_barrier_size() != IONIC_RESULT_SUCCESS)
        return IONIC_RESULT_FAILURE;
    if (test_barrier_create_and_destroy() != IONIC_RESULT_SUCCESS)
        return IONIC_RESULT_FAILURE;
    if (test_barrier_single_wait() != IONIC_RESULT_SUCCESS)
        return IONIC_RESULT_FAILURE;
    if (test_barrier_multi_process() != IONIC_RESULT_SUCCESS)
        return IONIC_RESULT_FAILURE;
    if (test_barrier_join_existing() != IONIC_RESULT_SUCCESS)
        return IONIC_RESULT_FAILURE;
    if (test_barrier_multiple_waits() != IONIC_RESULT_SUCCESS)
        return IONIC_RESULT_FAILURE;

    return IONIC_RESULT_SUCCESS;
}
