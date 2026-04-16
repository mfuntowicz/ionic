#ifndef IONIC_ENGINE_H
#define IONIC_ENGINE_H

#include <ionic/ionic.h>

struct ionic_scatter_entry {
    size_t staging_offset;
    size_t len;
    void  *dst;
};

struct ionic_logical_segment {
    const char *path;
    size_t from;
    size_t to;
    void  *dst;
};

struct ionic_io_fetch_result {
    const char *path;
    const void *data;
    size_t offset;
    size_t len;
    struct ionic_scatter_entry *entries;
    size_t n_entries;
    size_t userdata;
};

struct ionic_ioengine {
    void(*mark_done)(struct ionic_context *, struct ionic_ioengine *, const struct ionic_io_fetch_result *);
    size_t(*peek)(struct ionic_context *, struct ionic_ioengine *, struct ionic_io_fetch_result **, size_t count);
    size_t(*fetch)(struct ionic_context *, struct ionic_ioengine *, struct ionic_logical_segment *, size_t);
    void(*initialize)(struct ionic_context *, struct ionic_ioengine *);
    void(*destroy)(struct ionic_ioengine *);
};

static inline void ionic_ioengine_destroy(struct ionic_ioengine *engine) {
    return engine->destroy(engine);
}

static inline void ionic_ioengine_initialize(struct ionic_context *ctx, struct ionic_ioengine *engine) {
    return engine->initialize(ctx, engine);
}

static inline size_t ionic_ioengine_fetch(struct ionic_context *ctx, struct ionic_ioengine *engine, struct ionic_logical_segment *segments, size_t count) {
    return engine->fetch(ctx, engine, segments, count);
}

#endif