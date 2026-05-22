#include <errno.h>
#include <string.h>

#include "group.h"
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

static size_t ionic_group_shm_size(unsigned char count) {
    return sizeof(ionic_group_t) + count * sizeof(void *);
}

struct ionic_group *ionic_group_create(struct ionic_context *ctx, const char *identifier, unsigned char count) {
    if (ionic_has_error(&ctx->error)) return NULL;

    const size_t size = ionic_group_shm_size(count);
    struct ionic_group *group = ionic_shm_create(identifier, size);
    if (!group) {
        ionic_set_error(ctx, IONIC_SYS_ERR(errno));
        return NULL;
    }

    atomic_store_explicit(&group->steps, 0, memory_order_relaxed);
    atomic_store_explicit(&group->ready, 0, memory_order_relaxed);
    group->count = count;
    strncpy(group->identifier, identifier, IONIC_GROUP_MAX_IDENT - 1);
    group->identifier[IONIC_GROUP_MAX_IDENT - 1] = '\0';
    memset(group->device_ptrs, 0, count * sizeof(void *));
    atomic_thread_fence(memory_order_release);

    return group;
}

struct ionic_group *ionic_group_open(struct ionic_context *ctx, const char *identifier, unsigned char count) {
    if (ionic_has_error(&ctx->error)) return NULL;

    const size_t size = ionic_group_shm_size(count);
    struct ionic_group *group = ionic_shm_open(identifier, size);
    if (!group) {
        ionic_set_error(ctx, IONIC_SYS_ERR(errno));
        return NULL;
    }

    atomic_thread_fence(memory_order_acquire);

    return group;
}

void ionic_group_wait(struct ionic_group *group) {
    unsigned int current = atomic_load_explicit(&group->steps, memory_order_acquire);
    unsigned char count = atomic_fetch_add_explicit(&group->ready, 1, memory_order_acq_rel);

    if (count == (group->count - 1)) {
        atomic_store_explicit(&group->ready, 0, memory_order_release);
        atomic_fetch_add_explicit(&group->steps, 1, memory_order_acq_rel);
    } else {
        while (atomic_load_explicit(&group->steps, memory_order_acquire) == current)
            ionic_sched_yield_intrinsic();
    }
}

void ionic_group_close(struct ionic_group *group) {
    if (!group) return;
    const size_t size = ionic_group_shm_size(group->count);
    ionic_shm_close(group, size);
}

void ionic_group_destroy(struct ionic_group *group) {
    if (!group) return;

    char ident[IONIC_GROUP_MAX_IDENT];
    strncpy(ident, group->identifier, IONIC_GROUP_MAX_IDENT);
    const size_t size = ionic_group_shm_size(group->count);

    ionic_shm_close(group, size);
    ionic_shm_unlink(ident);
}