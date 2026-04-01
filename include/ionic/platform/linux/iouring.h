#ifndef IONIC_PLATFORM_LINUX_IO_URING_H
#define IONIC_PLATFORM_LINUX_IO_URING_H

#include <limits.h>
#include <liburing.h>
#include <ionic/ionic.h>
#include <ionic/engine.h>

#define SLOT_AVAILABLE 1
#define SLOT_UNAVAILABLE 0
#define SLOT_ALL_AVAILABLE ULONG_MAX

struct ionic_iouring_engine_slots {

};

struct ionic_iouring_engine_config {
    struct io_uring_params params;
    const char *files;
    unsigned n_files;
    unsigned qd;
};

struct ionic_iouring_engine {
    struct ionic_ioengine base;
    struct io_uring ring;
    struct ionic_iouring_engine_config config;
    struct io_uring_buf_ring *bring;
    struct iovec *iovecs;
    unsigned long slots[2];
    int fds;
};


static inline unsigned char ionic_iouring_engine_has_free_slot(const unsigned long slots) {
    return slots != SLOT_UNAVAILABLE;
}

static inline int ionic_iouring_engine_get_slot(const unsigned long slots[2]) {
    const unsigned long mask = slots[0] | slots[1];
    return __builtin_ctzl(mask | (~mask + 1)) ^ (mask == 0);
}

struct ionic_iouring_engine *ionic_iouring_engine_create(struct ionic_context *ctx, struct ionic_iouring_engine_config *config);

#endif