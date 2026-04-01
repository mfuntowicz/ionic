#ifndef IONIC_ENGINE_H
#define IONIC_ENGINE_H

#include <ionic/ionic.h>

struct ionic_ioengine {
    ionic_bool(*can_submit)(const struct ionic_context *, const struct ionic_ioengine *);
    void(*submit)(struct ionic_context *, struct ionic_ioengine *, size_t, size_t);
    void(*poll)(struct ionic_context *, struct ionic_ioengine *);
    void(*initialize)(struct ionic_context *, struct ionic_ioengine *);
    void(*destroy)(struct ionic_ioengine *);
};

typedef struct ionic_ioengine ionic_ioengine_t;

static inline ionic_bool ionic_ioengine_can_submit(const struct ionic_context *ctx, const struct ionic_ioengine *engine) {
    return engine->can_submit(ctx, engine);
}

static inline void ionic_ioengine_poll(struct ionic_context *ctx, struct ionic_ioengine *engine) {
    return engine->poll(ctx, engine);
}

#endif