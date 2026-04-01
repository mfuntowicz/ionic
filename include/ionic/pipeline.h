#ifndef IONIC_PIPELINE_H
#define IONIC_PIPELINE_H

#include <ionic/ionic.h>
#include <ionic/planner.h>

struct ionic_pipeline {
    const struct ionic_sharding_plan *plan;
    const unsigned short *locations;
    char * const *files;
    size_t n_tensors;
    size_t n_files;

    void(*initialize)(struct ionic_context *, struct ionic_pipeline *);
    void(*execute)(struct ionic_context *, struct ionic_pipeline *pipeline, const struct ionic_sharding_plan *, unsigned short);
    void(*destroy)(struct ionic_pipeline *);
};

typedef struct ionic_pipeline ionic_pipeline_t;

#endif
