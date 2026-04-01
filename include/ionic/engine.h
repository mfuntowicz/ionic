#ifndef IONIC_ENGINE_H
#define IONIC_ENGINE_H

#include <ionic/ionic.h>

struct ionic_io_fragment {
    size_t from;
    size_t to;
    size_t len;
    void *data;
    const char *path;
};

struct ionic_ioengine {
    void(*fetch)(struct ionic_context *, struct ionic_ioengine *, struct ionic_io_fragment *, unsigned);
    unsigned(*peek)(struct ionic_context *, struct ionic_ioengine *, struct ionic_io_fragment **, unsigned);
    void(*initialize)(struct ionic_context *, struct ionic_ioengine *);
    void(*destroy)(struct ionic_ioengine *);
};

static inline void ionic_ioengine_destroy(struct ionic_ioengine *engine) {
    return engine->destroy(engine);
}

static inline void ionic_ioengine_initialize(struct ionic_context *ctx, struct ionic_ioengine *engine) {
    return engine->initialize(ctx, engine);
}

static inline void ionic_ioengine_fetch(struct ionic_context *ctx, struct ionic_ioengine *engine, struct ionic_io_fragment *fragments, unsigned count) {
    return engine->fetch(ctx, engine, fragments, count);
}

static inline unsigned ionic_ioengine_peek(struct ionic_context *ctx, struct ionic_ioengine *engine, struct ionic_io_fragment **fragments, unsigned count) {
    return engine->peek(ctx, engine, fragments, count);
}

#endif