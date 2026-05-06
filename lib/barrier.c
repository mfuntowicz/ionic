#include <stdatomic.h>
#include "ionic/types.h"


static inline void ionic_sched_yield_intrinsic(void) {
    #if defined(__x86_64__) || defined(__i386__)
        __builtin_ia32_pause();
    #elif defined(__aarch64__) || defined(__arm__)
        __asm__ volatile("yield" ::: "memory");
    #else
        ((void)0); 
    #endif
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