#ifndef IONIC_PLATFORM_LINUX_IO_URING_H
#define IONIC_PLATFORM_LINUX_IO_URING_H

#include <limits.h>
#include <stdatomic.h>
#include <liburing.h>
#include <ionic/ionic.h>
#include <ionic/engine.h>

#define SLOT_AVAILABLE 1
#define SLOT_UNAVAILABLE 0

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
    struct ionic_scatter_entry *entries;
    size_t n_entries;
};

struct ionic_iouring_engine {
    struct ionic_ioengine base;
    struct io_uring ring;
    struct ionic_iouring_engine_config config;
    struct iovec *iovecs;
    unsigned long slots;
    atomic_ulong pending;
    atomic_ulong done;
    struct ionic_io_fetch_result *results;
    struct ionic_iouring_registered_file *files;
};

static inline unsigned char has_slot_available(const unsigned long slots) { return slots != SLOT_UNAVAILABLE; }
static inline int get_slot_available(const unsigned long *slots) { return __builtin_ffsl((long)*slots) - 1; }
static inline void mark_slot_busy(unsigned long *slots, const size_t i) { *slots &= ~(1UL << i); }
static inline void set_slot_available(unsigned long *slots, const size_t i) { *slots |= 1UL << i; }

struct ionic_iouring_engine *ionic_iouring_engine_create(struct ionic_context *ctx, struct ionic_iouring_engine_config *config);

#endif
