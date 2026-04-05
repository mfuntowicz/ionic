#ifndef IONIC_PLATFORM_LINUX_IO_URING_H
#define IONIC_PLATFORM_LINUX_IO_URING_H

#include <limits.h>
#include <liburing.h>
#include <ionic/ionic.h>
#include <ionic/engine.h>

#define SLOT_AVAILABLE 1
#define SLOT_UNAVAILABLE 0
#define SLOT_ALL_AVAILABLE ULONG_MAX

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

struct ionic_iouring_engine {
    struct ionic_ioengine base;
    struct io_uring ring;
    struct ionic_iouring_engine_config config;
    struct iovec *iovecs;
    unsigned long slots[2];
    struct ionic_iouring_registered_file *files;
};


static inline unsigned char has_free_slot(const unsigned long slots) {
    return slots != SLOT_UNAVAILABLE;
}

static inline int get_available_slot(const unsigned long (*slots)[2]) {
    const unsigned long mask = *slots[0] | *slots[1];
    return __builtin_ctzl(mask | (~mask + 1)) ^ (mask == 0);   // todo(mfuntowicz): limit to the actual queue depth
}

static inline void set_slot_busy(unsigned long (*slots)[2], const size_t i) {
    unsigned s = i % (sizeof(unsigned long) * 8);
    *slots[s] &= !(1UL << i);
}

struct ionic_iouring_engine *ionic_iouring_engine_create(struct ionic_context *ctx, struct ionic_iouring_engine_config *config);

#endif