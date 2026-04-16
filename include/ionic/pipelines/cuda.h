#ifndef IONIC_PIPELINES_CUDA_H
#define IONIC_PIPELINES_CUDA_H

#include <ionic/types.h>
#include <ionic/pipeline.h>
#include <cuda_runtime.h>

#define IONIC_EVENT_TAG_PIPELINE_CUDA "pipeline(cuda)"
#define IONIC_COALESCED_READ_THRESHOLD_DEFAULT (128 * 1024)

struct ionic_pipeline_cuda {
    struct ionic_pipeline base;
    struct ionic_device   device;
    struct ionic_ioengine *ioengine;

    cudaStream_t *streams;
    cudaEvent_t  *events;

    void  *device_buffer;
    size_t device_buffer_size;

    size_t threshold;
    unsigned char concurrency;
    char tag[24];
};

struct ionic_pipeline *ionic_pipeline_cuda_create(struct ionic_context *, unsigned short world_size);


#endif