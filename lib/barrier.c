#include <stdatomic.h>
#include <errno.h>
#include <string.h>

#include <ionic/ionic.h>
#include <ionic/platform/posix/shm.h>


static inline void ionic_sched_yield_intrinsic(void) {
    #if defined(__x86_64__) || defined(__i386__)
        __builtin_ia32_pause();
    #elif defined(__aarch64__) || defined(__arm__)
        __asm__ volatile("yield" ::: "memory");
    #else
        ((void)0); 
    #endif
}


struct ionic_barrier *ionic_barrier_create(struct ionic_context *ctx, const char *identifier, unsigned char count) {
    if (ionic_has_error(&ctx->error)) return NULL;

    struct ionic_barrier *barrier = ionic_shm_create(identifier, sizeof(*barrier));
    if (!barrier) {
        ionic_set_error(ctx, IONIC_SYS_ERR(errno));
        return NULL;
    }

    atomic_store_explicit(&barrier->steps, 0, memory_order_relaxed);
    atomic_store_explicit(&barrier->ready, 0, memory_order_relaxed);
    barrier->total = count;
    strncpy(barrier->identifier, identifier, IONIC_BARRIER_MAX_IDENT - 1);
    barrier->identifier[IONIC_BARRIER_MAX_IDENT - 1] = '\0';
    atomic_thread_fence(memory_order_release);

    return barrier;
}

struct ionic_barrier *ionic_barrier_open(struct ionic_context *ctx, const char *identifier) {
    if (ionic_has_error(&ctx->error)) return NULL;

    struct ionic_barrier *barrier = ionic_shm_open(identifier, sizeof(*barrier));
    if (!barrier) {
        ionic_set_error(ctx, IONIC_SYS_ERR(errno));
        return NULL;
    }

    atomic_thread_fence(memory_order_acquire);

    return barrier;
}

void ionic_barrier_close(struct ionic_barrier *barrier) {
    if (!barrier) return;
    ionic_shm_close(barrier, sizeof(*barrier));
}

void ionic_barrier_destroy(struct ionic_barrier *barrier) {
    if (!barrier) return;

    char ident[IONIC_BARRIER_MAX_IDENT];
    strncpy(ident, barrier->identifier, IONIC_BARRIER_MAX_IDENT);

    ionic_shm_close(barrier, sizeof(*barrier));
    ionic_shm_unlink(ident);
}

void ionic_barrier_wait(struct ionic_barrier *barrier) {
    
    unsigned int current = atomic_load_explicit(&barrier->steps, memory_order_acquire); 
    unsigned char count = atomic_fetch_add_explicit(&barrier->ready, 1, memory_order_acq_rel);

    if(count == (barrier->total - 1)) {
        atomic_store_explicit(&barrier->ready, 0, memory_order_release);
        atomic_fetch_add_explicit(&barrier->steps, 1, memory_order_acq_rel);
    } else {
        while(atomic_load_explicit(&barrier->steps, memory_order_acquire) == current)
            ionic_sched_yield_intrinsic();
    }
}
