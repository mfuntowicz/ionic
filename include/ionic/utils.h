#ifndef IONIC_UTILS_H
#define IONIC_UTILS_H

#include <ionic/types.h>

#if defined(__x86_64__) || defined(__i386__)
#include <immintrin.h>
#define ionic_cpu_relax() _mm_pause()
#elif defined(__aarch64__)
#define ionic_cpu_relax() __asm__ __volatile__("yield" ::: "memory")
#else
#define ionic_cpu_relax() ((void)0)
#endif

static inline int ionic_sort_tensors_by_offset(const void *a, const void *b) {
    const struct ionic_tensor *lhs = (const struct ionic_tensor *)a;
    const struct ionic_tensor *rhs = (const struct ionic_tensor *)b;
    return lhs->start - rhs->start;
}

static inline size_t ionic_tensor_nbytes(const struct ionic_tensor *tensor) {
    if(!tensor) return 0;
    return tensor->end - tensor->start;
}

static inline unsigned char ionic_is_power_of_two(const unsigned int n) { return n && !(n & (n - 1)); }
static inline size_t ionic_align_down_sz(const size_t v, const size_t a) { return v & ~(a - 1); }
static inline size_t ionic_align_up_sz(const size_t v, const size_t a) { return (v + a - 1) & ~(a - 1); }

#endif // IONIC_UTILS_H