#ifndef IONIC_PLATFORM_LINUX_IO_URING_H
#define IONIC_PLATFORM_LINUX_IO_URING_H

#include <ionic/ionic.h>
#include <liburing.h>

struct ionic_iouring_engine_config {
    struct io_uring_params params;
    const char *files;
    unsigned n_files;
    unsigned qd;
};

struct ionic_iouring_engine {
    struct io_uring ring;
    struct ionic_iouring_engine_config config;
    struct io_uring_buf_ring *bring;
    struct iovec *iovecs;
    int fds;
};

void ionic_iouring_engine_destroy(struct ionic_iouring_engine *);
struct ionic_iouring_engine *ionic_iouring_engine_create(struct ionic_context *ctx, struct ionic_iouring_engine_config *config);

#endif