#ifndef IONIC_PIPELINES_CUDA_H
#define IONIC_PIPELINES_CUDA_H

#include <ionic/types.h>
#include <ionic/pipeline.h>
#include <cuda_runtime.h>

#define IONIC_EVENT_TAG_PIPELINE_CUDA "pipeline(cuda)"

struct ionic_pipeline_cuda {
    struct ionic_pipeline base;
    struct ionic_device   device;
    cudaStream_t stream;
    cudaEvent_t  event;

    char tag[24];
};


struct ionic_pipeline *ionic_pipeline_cuda_create(struct ionic_context *, unsigned short world_size);


#endif