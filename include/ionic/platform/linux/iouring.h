#ifndef IONIC_PLATFORM_LINUX_IO_URING_H
#define IONIC_PLATFORM_LINUX_IO_URING_H

#include <limits.h>
#include <stdatomic.h>
#include <liburing.h>
#include <ionic/ionic.h>
#include <ionic/engine.h>

#define SLOT_AVAILABLE 1
#define SLOT_UNAVAILABLE 0
#define SLOT_ALL_AVAILABLE ULONG_MAX

#define BITS_PER_WORD (sizeof(unsigned long) * CHAR_BIT)

struct ionic_iouring_registered_file {
    const char *path;
    int fd;
};


struct ionic_iouring_engine_config {
    struct io_uring_params params;
    const char **files;
    unsigned n_files;
    unsigned qd;
    size_t   st_size;
};

struct ionic_iouring_slot_info {
    void *dst;
    size_t len;
};

struct ionic_iouring_engine {
    struct ionic_ioengine base;
    struct io_uring ring;
    struct ionic_iouring_engine_config config;
    struct iovec *iovecs;
    unsigned long slots[2];
    _Atomic unsigned long pending[2];
    _Atomic unsigned long done[2];
    struct ionic_iouring_slot_info *slot_infos;
    struct ionic_iouring_registered_file *files;
};


static inline unsigned char has_free_slot(const unsigned long slots) {
    return slots != SLOT_UNAVAILABLE;
}

static inline int get_available_slot(const unsigned long (*slots)[2]) {
    if ((*slots)[0] != 0) {
        return (int)__builtin_ctzl((*slots)[0]);
    }
    if ((*slots)[1] != 0) {
        return (int)(BITS_PER_WORD + __builtin_ctzl((*slots)[1]));
    }
    return -1;
}

static inline void set_slot_busy(unsigned long (*slots)[2], const size_t i) {
    unsigned s = i / BITS_PER_WORD;
    unsigned bit = i % BITS_PER_WORD;
    (*slots)[s] &= ~(1UL << bit);
}

static inline void set_slot_available(unsigned long (*slots)[2], const size_t i) {
    unsigned s = i / BITS_PER_WORD;
    unsigned bit = i % BITS_PER_WORD;
    (*slots)[s] |= (1UL << bit);
}

struct ionic_iouring_engine *ionic_iouring_engine_create(struct ionic_context *ctx, struct ionic_iouring_engine_config *config);

#endif
