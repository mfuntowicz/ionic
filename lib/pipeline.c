#include <string.h>
#include <time.h>
#include <unistd.h>
#include <stdbool.h>
#include <stdint.h>
#include "ionic/ionic.h"
#include "ionic/pipeline.h"
#include "ionic/utils.h"

#define IONIC_EVENT_TAG_PIPELINE "pipeline"

/*
 * Read Chunk Metadata (internal)
 * 
 * Design: Fixed-size I/O chunks (STAGING_BUFFER_SIZE) driven by file ranges,
 * not tensor boundaries. A single chunk may span multiple tensors.
 * During completion, we resolve which tensors each chunk covers.
 */
struct ionic_chunk_meta {
    uint64_t file_offset;      /* Aligned start offset in file */
    uint32_t io_size;          /* Actual I/O size (aligned) */
    uint32_t payload_offset;   /* Offset of valid data within I/O (alignment padding) */
    uint32_t payload_size;     /* Valid data size */
    int32_t  fd;               /* File descriptor for this chunk */
    int32_t  first_tensor;     /* First tensor index this chunk covers */
    int32_t  last_tensor;      /* Last tensor index this chunk covers (inclusive) */
    uint16_t staging_slot;     /* Assigned staging slot during execution */
    uint16_t flags;
};

enum ionic_chunk_flag {
    IONIC_CHUNK_FLAG_NONE = 0,
};

/*
 * Internal Pipeline Structure
 */
struct ionic_pipeline {
    struct ionic_pipeline_config config;
    
    struct ionic_tensor_status *statuses;
    size_t                      n_tensors;
    
    struct ionic_chunk_meta    *chunks;
    size_t                      n_chunks;
    
    void                      **staging_buffers;
    _Atomic uint32_t           *slot_state;
    
    _Atomic uint64_t            total_bytes;
    _Atomic uint64_t            loaded_bytes;
    
#ifdef __IONIC_CUDA_ENABLED__
    cudaStream_t                stream;
    cudaEvent_t                *transfer_done;
#endif
};

enum {
    SLOT_FREE       = 0,
    SLOT_READING    = 1,
    SLOT_TRANSFERRING = 2,
};

/* ═══════════════════════════════════════════════════════════════════════════
 * Status Accessor Functions (FFI-safe)
 * ═══════════════════════════════════════════════════════════════════════════ */

ionic_tensor_state_t ionic_tensor_status_get_state(const struct ionic_tensor_status *s)
{
    return (ionic_tensor_state_t)atomic_load_explicit(&s->state, memory_order_acquire);
}

size_t ionic_tensor_status_get_bytes_loaded(const struct ionic_tensor_status *s)
{
    return (size_t)atomic_load_explicit(&s->bytes_loaded, memory_order_acquire);
}

size_t ionic_tensor_status_get_total_bytes(const struct ionic_tensor_status *s)
{
    return (size_t)s->size;
}

bool ionic_tensor_status_is_complete(const struct ionic_tensor_status *s)
{
    ionic_tensor_state_t st = ionic_tensor_status_get_state(s);
    return st == IONIC_TENSOR_STATE_READY || st == IONIC_TENSOR_STATE_FAILED;
}

bool ionic_tensor_status_has_error(const struct ionic_tensor_status *s, int *error_code_out)
{
    if (ionic_tensor_status_get_state(s) != IONIC_TENSOR_STATE_FAILED)
        return false;
    
    if (error_code_out)
        *error_code_out = s->error_code;
    return true;
}

const struct ionic_tensor_status *ionic_pipeline_get_tensor_status(
    const struct ionic_pipeline *p, size_t tensor_index)
{
    if (!p || tensor_index >= p->n_tensors)
        return NULL;
    return &p->statuses[tensor_index];
}

/* ═══════════════════════════════════════════════════════════════════════════
 * Progress Query Functions
 * ═══════════════════════════════════════════════════════════════════════════ */

float ionic_pipeline_get_progress(const struct ionic_pipeline *p)
{
    uint64_t total = atomic_load_explicit(&p->total_bytes, memory_order_acquire);
    if (total == 0) return 0.0f;
    uint64_t loaded = atomic_load_explicit(&p->loaded_bytes, memory_order_acquire);
    return (float)loaded / (float)total;
}

size_t ionic_pipeline_get_bytes_total(const struct ionic_pipeline *p)
{
    return (size_t)atomic_load_explicit(&p->total_bytes, memory_order_acquire);
}

size_t ionic_pipeline_get_bytes_loaded(const struct ionic_pipeline *p)
{
    return (size_t)atomic_load_explicit(&p->loaded_bytes, memory_order_acquire);
}

/* ═══════════════════════════════════════════════════════════════════════════
 * Pipeline Lifecycle
 * ═══════════════════════════════════════════════════════════════════════════ */

struct ionic_pipeline *ionic_pipeline_init(struct ionic_context *ctx, struct ionic_pipeline_config config)
{
    if (ionic_has_error(&ctx->error))
        return NULL;
    
    struct ionic_pipeline *p = calloc(1, sizeof(*p));
    if (!p) {
        ionic_set_error(ctx, IONIC_ERR(IONIC_ERROR_ALLOCATION_FAILED));
        return NULL;
    }
    
    p->config = config;
    
    p->staging_buffers = calloc(config.staging_slot_count, sizeof(void*));
    p->slot_state = calloc(config.staging_slot_count, sizeof(_Atomic uint32_t));
    if (!p->staging_buffers || !p->slot_state) {
        free(p->staging_buffers);
        free(p->slot_state);
        free(p);
        ionic_set_error(ctx, IONIC_ERR(IONIC_ERROR_ALLOCATION_FAILED));
        return NULL;
    }
    
#ifdef __IONIC_CUDA_ENABLED__
    if (ctx->device.kind == IONIC_DEVICE_CUDA) {
        cudaError_t ce = cudaSetDevice((int)ctx->device.ordinal);
        if (ce != cudaSuccess) {
            ionic_set_error(ctx, IONIC_CUDA_ERR((int)ce));
            free(p->staging_buffers);
            free(p->slot_state);
            free(p);
            return NULL;
        }
        
        ce = cudaStreamCreateWithFlags(&p->stream, cudaStreamNonBlocking);
        if (ce != cudaSuccess) {
            ionic_set_error(ctx, IONIC_CUDA_ERR((int)ce));
            free(p->staging_buffers);
            free(p->slot_state);
            free(p);
            return NULL;
        }
        
        p->transfer_done = calloc(config.staging_slot_count, sizeof(cudaEvent_t));
        if (!p->transfer_done) {
            cudaStreamDestroy(p->stream);
            free(p->staging_buffers);
            free(p->slot_state);
            free(p);
            ionic_set_error(ctx, IONIC_ERR(IONIC_ERROR_ALLOCATION_FAILED));
            return NULL;
        }
        
        for (uint32_t i = 0; i < config.staging_slot_count; i++) {
            ce = cudaEventCreateWithFlags(&p->transfer_done[i], cudaEventDisableTiming);
            if (ce != cudaSuccess) {
                for (uint32_t j = 0; j < i; j++)
                    cudaEventDestroy(p->transfer_done[j]);
                cudaStreamDestroy(p->stream);
                free(p->transfer_done);
                free(p->staging_buffers);
                free(p->slot_state);
                free(p);
                ionic_set_error(ctx, IONIC_CUDA_ERR((int)ce));
                return NULL;
            }
            /* Record event immediately so cudaEventQuery returns cudaSuccess */
            ce = cudaEventRecord(p->transfer_done[i], p->stream);
            if (ce != cudaSuccess) {
                for (uint32_t j = 0; j <= i; j++)
                    cudaEventDestroy(p->transfer_done[j]);
                cudaStreamDestroy(p->stream);
                free(p->transfer_done);
                free(p->staging_buffers);
                free(p->slot_state);
                free(p);
                ionic_set_error(ctx, IONIC_CUDA_ERR((int)ce));
                return NULL;
            }
        }
        
        for (uint32_t i = 0; i < config.staging_slot_count; i++) {
            ce = cudaHostAlloc(&p->staging_buffers[i], config.staging_buffer_size, cudaHostAllocDefault);
            if (ce != cudaSuccess) {
                for (uint32_t j = 0; j < i; j++)
                    cudaFreeHost(p->staging_buffers[j]);
                for (uint32_t j = 0; j < config.staging_slot_count; j++)
                    cudaEventDestroy(p->transfer_done[j]);
                cudaStreamDestroy(p->stream);
                free(p->transfer_done);
                free(p->staging_buffers);
                free(p->slot_state);
                free(p);
                ionic_set_error(ctx, IONIC_CUDA_ERR((int)ce));
                return NULL;
            }
            atomic_store_explicit(&p->slot_state[i], SLOT_FREE, memory_order_release);
        }
    }
#endif
    
    IONIC_INFO(&ctx->logger, IONIC_EVENT_TAG_PIPELINE,
               "initialized slots=%u slot_size=%llu batch=%u max_inflight=%u",
               config.staging_slot_count, 
               (unsigned long long)config.staging_buffer_size,
               config.io_uring_batch_size,
               config.max_inflight_reads);
    
    return p;
}

void ionic_pipeline_destroy(struct ionic_pipeline *p)
{
    if (!p) return;
    
#ifdef __IONIC_CUDA_ENABLED__
    if (p->stream) {
        cudaStreamSynchronize(p->stream);
        cudaStreamDestroy(p->stream);
    }
    if (p->transfer_done) {
        for (uint32_t i = 0; i < p->config.staging_slot_count; i++) {
            if (p->transfer_done[i])
                cudaEventDestroy(p->transfer_done[i]);
        }
        free(p->transfer_done);
    }
    for (uint32_t i = 0; i < p->config.staging_slot_count; i++) {
        if (p->staging_buffers[i])
            cudaFreeHost(p->staging_buffers[i]);
    }
#else
    for (uint32_t i = 0; i < p->config.staging_slot_count; i++) {
        free(p->staging_buffers[i]);
    }
#endif
    
    free(p->staging_buffers);
    free(p->slot_state);
    free(p->chunks);
    free(p->statuses);
    free(p);
}

/* ═══════════════════════════════════════════════════════════════════════════
 * Initialize from Sharding Plan
 * ═══════════════════════════════════════════════════════════════════════════ */

static int ionic_pipeline_init_from_plan(
    struct ionic_context *ctx,
    struct ionic_pipeline *p,
    struct ionic_sharding_plan *plan,
    unsigned short rank)
{
    p->n_tensors = plan->n;
    p->statuses = calloc(plan->n, sizeof(struct ionic_tensor_status));
    if (!p->statuses) {
        ionic_set_error(ctx, IONIC_ERR(IONIC_ERROR_ALLOCATION_FAILED));
        return -1;
    }
    
    atomic_store_explicit(&p->total_bytes, 0, memory_order_release);
    atomic_store_explicit(&p->loaded_bytes, 0, memory_order_release);
    
    for (size_t i = 0; i < plan->n; i++) {
        struct ionic_sharded_tensor *st = &plan->tensors[i];
        struct ionic_sharded_tensor_specs *spec = &st->specs[rank];
        
        struct ionic_tensor_status *s = &p->statuses[i];
        
        atomic_store_explicit(&s->state, IONIC_TENSOR_STATE_PENDING, memory_order_release);
        atomic_store_explicit(&s->bytes_loaded, 0, memory_order_release);
        
        s->size = spec->end - spec->start;
        s->offset = spec->start;
        s->fd = st->fd;
        s->error_code = 0;
        
        /* Allocate device memory if not provided */
        if (spec->dst) {
            s->device_ptr = spec->dst;
        } else {
            s->device_ptr = ionic_allocate_device(ctx, s->size, IONIC_ALLOC_DEVICE);
            if (!s->device_ptr) {
                IONIC_ERROR(&ctx->logger, IONIC_EVENT_TAG_PIPELINE, "device memory allocation failed tensor=%zu size=%zu", i, s->size);
                for (size_t j = 0; j < i; j++) {
                    if (p->statuses[j].device_ptr && !plan->tensors[j].specs[rank].dst) {
                        ionic_free_device(ctx, p->statuses[j].device_ptr, IONIC_ALLOC_DEVICE);
                    }
                }
                free(p->statuses);
                p->statuses = NULL;
                return -1;
            }
            /* Update plan so caller can access the pointer */
            spec->dst = s->device_ptr;
        }
        
        atomic_fetch_add_explicit(&p->total_bytes, s->size, memory_order_relaxed);
    }
    
    return 0;
}

static int ionic_pipeline_build_chunks(struct ionic_context *ctx, struct ionic_pipeline *p)
{
    const size_t alignment = 4096;
    const size_t chunk_io_size = p->config.staging_buffer_size;  /* Fixed I/O size for all chunks */
    
    if (p->n_tensors == 0) {
        p->chunks = NULL;
        p->n_chunks = 0;
        return 0;
    }
    
    /*
     * Strategy: Build fixed-size chunks based on tensor coverage.
     * Only create chunks that actually overlap with at least one tensor.
     * 
     * Each chunk is STAGING_BUFFER_SIZE, starting from the first tensor's aligned offset.
     */
    int32_t primary_fd = p->statuses[0].fd;  /* Assume single file for now */
    
    /* Find the starting point (aligned offset of first tensor) */
    uint64_t min_offset = UINT64_MAX;
    uint64_t max_end_offset = 0;
    
    for (size_t i = 0; i < p->n_tensors; i++) {
        if (p->statuses[i].offset < min_offset)
            min_offset = p->statuses[i].offset;
        uint64_t end = p->statuses[i].offset + p->statuses[i].size;
        if (end > max_end_offset)
            max_end_offset = end;
    }
    
    uint64_t aligned_start = ionic_align_down_sz(min_offset, alignment);
    
    /* First pass: count chunks that actually overlap with tensors */
    size_t total_chunks = 0;
    uint64_t chunk_start = aligned_start;
    
    while (chunk_start < max_end_offset) {
        uint64_t chunk_end = chunk_start + chunk_io_size;
        if (chunk_end > max_end_offset + alignment)  /* Allow slight overshoot for last chunk */
            chunk_end = max_end_offset + alignment;
        
        /* Check if this chunk overlaps with any tensor */
        bool has_overlap = false;
        for (size_t t = 0; t < p->n_tensors; t++) {
            uint64_t tensor_start = p->statuses[t].offset;
            uint64_t tensor_end = tensor_start + p->statuses[t].size;
            
            if (tensor_start < chunk_end && tensor_end > chunk_start) {
                has_overlap = true;
                break;
            }
        }
        
        if (has_overlap)
            total_chunks++;
        
        chunk_start = chunk_end;
    }
    
    IONIC_DEBUG(&ctx->logger, IONIC_EVENT_TAG_PIPELINE,
                "planning range: tensors=%zu range=%llu-%llu aligned_start=%llu chunks=%zu",
                p->n_tensors, (unsigned long long)min_offset, (unsigned long long)max_end_offset,
                (unsigned long long)aligned_start, total_chunks);
    
    p->chunks = calloc(total_chunks, sizeof(struct ionic_chunk_meta));
    if (!p->chunks) {
        ionic_set_error(ctx, IONIC_ERR(IONIC_ERROR_ALLOCATION_FAILED));
        return -1;
    }
    p->n_chunks = total_chunks;
    
    /* Second pass: build chunks that overlap with tensors */
    size_t chunk_idx = 0;
    chunk_start = aligned_start;
    
    while (chunk_start < max_end_offset && chunk_idx < total_chunks) {
        uint64_t chunk_end = chunk_start + chunk_io_size;
        if (chunk_end > max_end_offset + alignment)
            chunk_end = max_end_offset + alignment;
        
        /* Check if this chunk overlaps with any tensor */
        bool has_overlap = false;
        for (size_t t = 0; t < p->n_tensors; t++) {
            uint64_t tensor_start = p->statuses[t].offset;
            uint64_t tensor_end = tensor_start + p->statuses[t].size;
            
            if (tensor_start < chunk_end && tensor_end > chunk_start) {
                has_overlap = true;
                break;
            }
        }
        
        if (!has_overlap) {
            chunk_start = chunk_end;
            continue;
        }
        
        struct ionic_chunk_meta *c = &p->chunks[chunk_idx];
        
        /* Align the I/O start down and size up for direct I/O */
        uint64_t aligned_off = ionic_align_down_sz(chunk_start, alignment);
        uint32_t pad = (uint32_t)(chunk_start - aligned_off);
        uint32_t payload = (uint32_t)(chunk_end - chunk_start);
        
        uint64_t io_end = ionic_align_up_sz(chunk_end, alignment);
        uint32_t io_size = (uint32_t)(io_end - aligned_off);
        if (io_size > chunk_io_size + (uint32_t)alignment)
            io_size = (uint32_t)(chunk_io_size + alignment);
        
        c->file_offset = aligned_off;
        c->io_size = io_size;
        c->payload_offset = pad;
        c->payload_size = payload;
        c->fd = primary_fd;
        c->first_tensor = -1;
        c->last_tensor = -1;
        c->staging_slot = 0;
        c->flags = 0;
        
        /* Resolve which tensors this chunk covers */
        for (size_t t = 0; t < p->n_tensors; t++) {
            struct ionic_tensor_status *ts = &p->statuses[t];
            uint64_t tensor_start = ts->offset;
            uint64_t tensor_end = tensor_start + ts->size;
            
            if (tensor_start < chunk_end && tensor_end > chunk_start) {
                /* This tensor overlaps with the chunk */
                if (c->first_tensor < 0)
                    c->first_tensor = (int32_t)t;
                c->last_tensor = (int32_t)t;
            }
        }
        
        chunk_idx++;
        chunk_start = chunk_end;
    }
    
    IONIC_DEBUG(&ctx->logger, IONIC_EVENT_TAG_PIPELINE,
                "built chunks tensors=%zu chunks=%zu total_bytes=%llu io_size=%zuKB",
                p->n_tensors, p->n_chunks, 
                (unsigned long long)atomic_load(&p->total_bytes),
                chunk_io_size / 1024);
    
    return 0;
}

/* ═══════════════════════════════════════════════════════════════════════════
 * Execution (CUDA path)
 * ═══════════════════════════════════════════════════════════════════════════ */

#ifdef __IONIC_CUDA_ENABLED__

int ionic_pipeline_execute_plan(struct ionic_context *ctx, struct ionic_pipeline *p, struct ionic_sharding_plan *plan, unsigned short rank) {
    if(ionic_has_error(&ctx->error)) return -1; 

    if (!p || !plan || plan->n == 0) {
        IONIC_ERROR(&ctx->logger, IONIC_EVENT_TAG_PIPELINE, "invalid plan");
        return -1;
    }
    
    if(p->statuses) free(p->statuses);
    if(p->chunks)   free(p->chunks);
    p->statuses = NULL;
    p->chunks = NULL;
    p->n_tensors = 0;
    p->n_chunks = 0;
    
    if (ionic_pipeline_init_from_plan(ctx, p, plan, rank) != 0)
        return -1;
    
    if (ionic_pipeline_build_chunks(ctx, p) != 0)
        return -1;
    
    struct ionic_backend *backend = ctx->backend;
    
    IONIC_DEBUG(&ctx->logger, IONIC_EVENT_TAG_PIPELINE, "execution starting backend=%p", (void*)backend);
    
    /* Track chunk state: 0=pending, 1=io_done, 2=cuda_done */
    int *chunk_state = calloc(p->n_chunks, sizeof(*chunk_state));
    uint32_t *cuda_slots = calloc(p->n_chunks, sizeof(*cuda_slots));
    
    IONIC_DEBUG(&ctx->logger, IONIC_EVENT_TAG_PIPELINE, "allocated chunk_state=%p cuda_slots=%p", (void*)chunk_state, (void*)cuda_slots);
    
    if (!chunk_state || !cuda_slots) {
        free(chunk_state);
        free(cuda_slots);
        ionic_set_error(ctx, IONIC_ERR(IONIC_ERROR_ALLOCATION_FAILED));
        return -1;
    }
    
    /* Completion buffer for polling */
    struct ionic_io_completion *completions = malloc(64 * sizeof(*completions));
    if (!completions) {
        free(chunk_state);
        free(cuda_slots);
        ionic_set_error(ctx, IONIC_ERR(IONIC_ERROR_ALLOCATION_FAILED));
        return -1;
    }
    
    size_t io_submitted = 0;
    size_t completed = 0;
    size_t io_inflight = 0;
    
    IONIC_DEBUG(&ctx->logger, IONIC_EVENT_TAG_PIPELINE, "starting execution chunks=%zu", p->n_chunks);
    
    while (completed < p->n_chunks) {
        /* Phase 1: Submit I/O requests aggressively (decoupled from CUDA slots) */
        while (io_submitted < p->n_chunks && io_inflight < p->config.max_inflight_reads) {
            struct ionic_chunk_meta *c = &p->chunks[io_submitted];
            
            /* Mark all covered tensors as loading */
            if (c->first_tensor >= 0) {
                for (int32_t t = c->first_tensor; t <= c->last_tensor; t++) {
                    atomic_store_explicit(&p->statuses[t].state, IONIC_TENSOR_STATE_LOADING, memory_order_release);
                }
            }
            
            /* Submit async read for the full chunk - pass chunk index as userdata */
            int rc = backend->submit_read(ctx, c->fd, c->file_offset, 
                                          c->io_size, (void *)(uintptr_t)io_submitted);
            IONIC_TRACE(&ctx->logger, IONIC_EVENT_TAG_PIPELINE, 
                       "submit chunk=%zu io_size=%u io_off=%llu", 
                       io_submitted, c->io_size, (unsigned long long)c->file_offset);
            if (rc != 0) {
                IONIC_ERROR(&ctx->logger, IONIC_EVENT_TAG_PIPELINE, "async submit failed chunk=%zu", io_submitted);
                /* Mark all covered tensors as failed */
                if (c->first_tensor >= 0) {
                    for (int32_t t = c->first_tensor; t <= c->last_tensor; t++) {
                        p->statuses[t].error_code = rc;
                        atomic_store_explicit(&p->statuses[t].state, IONIC_TENSOR_STATE_FAILED, memory_order_release);
                    }
                }
                free(chunk_state);
                free(cuda_slots);
                free(completions);
                return -1;
            }
            
            io_submitted++;
            io_inflight++;
        }
        
        /* Phase 2: Poll I/O completions and start CUDA transfers */
        size_t n = backend->poll_completions(ctx, completions, 64);
        if (n > 0)
            io_inflight -= n;  /* Decrease inflight count */
        
        for (size_t i = 0; i < n; i++) {
            size_t chunk_idx = (size_t)(uintptr_t)completions[i].userdata;
            
            /* Validate chunk_idx */
            if (chunk_idx >= p->n_chunks) {
                IONIC_ERROR(&ctx->logger, IONIC_EVENT_TAG_PIPELINE, "invalid chunk_idx=%zu", chunk_idx);
                backend->release_staging(ctx, completions[i].staging_slot);
                completed++;  /* Count as complete (failed) */
                continue;
            }
            
            struct ionic_chunk_meta *c = &p->chunks[chunk_idx];
            
            /* Skip chunks with no tensors (shouldn't happen but safety check) */
            if (c->first_tensor < 0 || c->first_tensor >= (int32_t)p->n_tensors) {
                IONIC_ERROR(&ctx->logger, IONIC_EVENT_TAG_PIPELINE, "chunk has no valid tensors chunk=%zu", chunk_idx);
                backend->release_staging(ctx, completions[i].staging_slot);
                completed++;
                continue;
            }
            
            /* Find a free CUDA slot for H2D transfer */
            uint32_t cuda_slot = UINT32_MAX;
            for (uint32_t s = 0; s < p->config.staging_slot_count; s++) {
                uint32_t state = atomic_load_explicit(&p->slot_state[s], memory_order_acquire);
                if (state == SLOT_FREE) {
                    cudaError_t q = cudaEventQuery(p->transfer_done[s]);
                    if (q == cudaSuccess) {
                        cuda_slot = s;
                        break;
                    }
                }
            }
            
            if (cuda_slot == UINT32_MAX) {
                /* No CUDA slot available - process CUDA completions to free slots */
                while (cuda_slot == UINT32_MAX) {
                    for (uint32_t s = 0; s < p->config.staging_slot_count; s++) {
                        uint32_t state = atomic_load_explicit(&p->slot_state[s], memory_order_acquire);
                        if (state == SLOT_TRANSFERRING) {
                            cudaError_t q = cudaEventQuery(p->transfer_done[s]);
                            if (q == cudaSuccess) {
                                /* Find and complete the chunk using this slot */
                                for (size_t ci = 0; ci < p->n_chunks; ci++) {
                                    if (cuda_slots[ci] == s && chunk_state[ci] == 1) {
                                        chunk_state[ci] = 2;
                                        completed++;
                                        
                                        struct ionic_chunk_meta *cc = &p->chunks[ci];
                                        
                                        /* Update all tensors covered by this chunk */
                                        for (int32_t t = cc->first_tensor; t <= cc->last_tensor && t < (int32_t)p->n_tensors; t++) {
                                            struct ionic_tensor_status *tsc = &p->statuses[t];
                                            
                                            /* Calculate overlap between chunk and tensor */
                                            uint64_t chunk_start = cc->file_offset + cc->payload_offset;
                                            uint64_t chunk_end = chunk_start + cc->payload_size;
                                            uint64_t tensor_start = tsc->offset;
                                            uint64_t tensor_end = tensor_start + tsc->size;
                                            
                                            uint64_t overlap_start = (chunk_start > tensor_start) ? chunk_start : tensor_start;
                                            uint64_t overlap_end = (chunk_end < tensor_end) ? chunk_end : tensor_end;
                                            
                                            if (overlap_start < overlap_end) {
                                                uint64_t overlap_bytes = overlap_end - overlap_start;
                                                uint64_t loaded = atomic_fetch_add_explicit(&tsc->bytes_loaded, overlap_bytes, memory_order_relaxed) + overlap_bytes;
                                                atomic_fetch_add_explicit(&p->loaded_bytes, overlap_bytes, memory_order_relaxed);
                                                
                                                if (loaded >= tsc->size) {
                                                    atomic_store_explicit(&tsc->state, IONIC_TENSOR_STATE_READY, memory_order_release);
                                                }
                                            }
                                        }
                                        break;
                                    }
                                }
                                atomic_store_explicit(&p->slot_state[s], SLOT_FREE, memory_order_release);
                                /* Now check if the newly freed slot is usable */
                                q = cudaEventQuery(p->transfer_done[s]);
                                if (q == cudaSuccess) {
                                    cuda_slot = s;
                                    break;
                                }
                            }
                        } else if (state == SLOT_FREE) {
                            cudaError_t q = cudaEventQuery(p->transfer_done[s]);
                            if (q == cudaSuccess) {
                                cuda_slot = s;
                                break;
                            }
                        }
                    }
                    
                    if (cuda_slot == UINT32_MAX) {
                        ionic_cpu_relax();
                    }
                }
            }
            
            /* Validate cuda_slot before using */
            if (cuda_slot >= p->config.staging_slot_count) {
                IONIC_ERROR(&ctx->logger, IONIC_EVENT_TAG_PIPELINE, "failed to acquire cuda slot for chunk=%zu (cuda_slot=%u, slot_count=%u)", 
                           chunk_idx, cuda_slot, p->config.staging_slot_count);
                backend->release_staging(ctx, completions[i].staging_slot);
                completed++;
                continue;
            }
            
            IONIC_TRACE(&ctx->logger, IONIC_EVENT_TAG_PIPELINE, "cuda slot acquired chunk=%zu cuda_slot=%u", chunk_idx, cuda_slot);
            
            cuda_slots[chunk_idx] = cuda_slot;
            atomic_store_explicit(&p->slot_state[cuda_slot], SLOT_READING, memory_order_release);
            
            /* Copy from io_uring staging to CUDA staging */
            memcpy(p->staging_buffers[cuda_slot], completions[i].staging_buffer, c->payload_size);
            
            /* Release the io_uring staging slot */
            backend->release_staging(ctx, completions[i].staging_slot);
            
            /* Calculate chunk's data range (where valid data starts and ends in the staging buffer) */
            uint64_t chunk_data_start = c->file_offset + c->payload_offset;
            uint64_t chunk_data_end = chunk_data_start + c->payload_size;
            
            /* Issue H2D transfers for each tensor covered by this chunk */
            bool any_transfer_failed = false;
            for (int32_t t = c->first_tensor; t <= c->last_tensor && t < (int32_t)p->n_tensors; t++) {
                struct ionic_tensor_status *ts = &p->statuses[t];
                
                /* Calculate overlap between chunk and tensor */
                uint64_t tensor_start = ts->offset;
                uint64_t tensor_end = tensor_start + ts->size;
                
                uint64_t overlap_start = (chunk_data_start > tensor_start) ? chunk_data_start : tensor_start;
                uint64_t overlap_end = (chunk_data_end < tensor_end) ? chunk_data_end : tensor_end;
                
                if (overlap_start >= overlap_end)
                    continue;  /* No overlap (shouldn't happen but safety) */
                
                uint64_t overlap_bytes = overlap_end - overlap_start;
                
                /* Calculate source offset in staging buffer */
                size_t src_offset = (size_t)(overlap_start - chunk_data_start);
                
                /* Calculate destination offset in tensor's device memory */
                size_t dst_offset = (size_t)(overlap_start - tensor_start);
                
                unsigned char *dst = (unsigned char *)ts->device_ptr + dst_offset;
                unsigned char *src = (unsigned char *)p->staging_buffers[cuda_slot] + src_offset;
                
                cudaError_t ce = cudaMemcpyAsync(dst, src, overlap_bytes, 
                                                 cudaMemcpyHostToDevice, p->stream);
                if (ce != cudaSuccess) {
                    IONIC_ERROR(&ctx->logger, IONIC_EVENT_TAG_PIPELINE, "cudaMemcpyAsync failed: %d", ce);
                    ts->error_code = (int)ce;
                    atomic_store_explicit(&ts->state, IONIC_TENSOR_STATE_FAILED, memory_order_release);
                    any_transfer_failed = true;
                    break;
                }
                
                /* Mark tensor as transferring */
                ionic_tensor_state_t current = atomic_load_explicit(&ts->state, memory_order_acquire);
                if (current != IONIC_TENSOR_STATE_FAILED) {
                    atomic_store_explicit(&ts->state, IONIC_TENSOR_STATE_TRANSFERRING, memory_order_release);
                }
            }
            
            if (any_transfer_failed) {
                free(chunk_state);
                free(cuda_slots);
                free(completions);
                return -1;
            }
            
            /* Record completion event for the whole set of transfers */
            cudaError_t ce = cudaEventRecord(p->transfer_done[cuda_slot], p->stream);
            if (ce != cudaSuccess) {
                IONIC_ERROR(&ctx->logger, IONIC_EVENT_TAG_PIPELINE, "cudaEventRecord failed: %d", ce);
                free(chunk_state);
                free(cuda_slots);
                free(completions);
                return -1;
            }
            
            atomic_store_explicit(&p->slot_state[cuda_slot], SLOT_TRANSFERRING, memory_order_release);
            chunk_state[chunk_idx] = 1;  /* I/O done */
        }
        
        /* Phase 3: Poll CUDA transfer completions */
        for (uint32_t slot = 0; slot < p->config.staging_slot_count; slot++) {
            uint32_t state = atomic_load_explicit(&p->slot_state[slot], memory_order_acquire);
            if (state != SLOT_TRANSFERRING)
                continue;
            
            cudaError_t q = cudaEventQuery(p->transfer_done[slot]);
            if (q == cudaErrorNotReady)
                continue;
            
            if (q != cudaSuccess) {
                ionic_set_error(ctx, IONIC_CUDA_ERR((int)q));
                free(chunk_state);
                free(cuda_slots);
                free(completions);
                return -1;
            }
            
            /* Find which chunk this slot belongs to */
            for (size_t i = 0; i < p->n_chunks; i++) {
                if (cuda_slots[i] == slot && chunk_state[i] == 1) {
                    chunk_state[i] = 2;  /* CUDA done */
                    completed++;
                    
                    struct ionic_chunk_meta *c = &p->chunks[i];
                    
                    /* Calculate chunk's data range */
                    uint64_t chunk_data_start = c->file_offset + c->payload_offset;
                    uint64_t chunk_data_end = chunk_data_start + c->payload_size;
                    
                    /* Update all tensors covered by this chunk */
                    for (int32_t t = c->first_tensor; t <= c->last_tensor && t < (int32_t)p->n_tensors; t++) {
                        struct ionic_tensor_status *ts = &p->statuses[t];
                        
                        /* Calculate overlap between chunk and tensor */
                        uint64_t tensor_start = ts->offset;
                        uint64_t tensor_end = tensor_start + ts->size;
                        
                        uint64_t overlap_start = (chunk_data_start > tensor_start) ? chunk_data_start : tensor_start;
                        uint64_t overlap_end = (chunk_data_end < tensor_end) ? chunk_data_end : tensor_end;
                        
                        if (overlap_start < overlap_end) {
                            uint64_t overlap_bytes = overlap_end - overlap_start;
                            uint64_t loaded = atomic_fetch_add_explicit(&ts->bytes_loaded, overlap_bytes, memory_order_relaxed) + overlap_bytes;
                            atomic_fetch_add_explicit(&p->loaded_bytes, overlap_bytes, memory_order_relaxed);
                            
                            if (loaded >= ts->size) {
                                atomic_store_explicit(&ts->state, IONIC_TENSOR_STATE_READY, memory_order_release);
                            }
                        }
                    }
                    break;
                }
            }
            
            atomic_store_explicit(&p->slot_state[slot], SLOT_FREE, memory_order_release);
        }
        
        if (completed < p->n_chunks)
            ionic_cpu_relax();
    }
    
    cudaError_t ce = cudaStreamSynchronize(p->stream);
    if (ce != cudaSuccess) {
        ionic_set_error(ctx, IONIC_CUDA_ERR((int)ce));
        free(chunk_state);
        free(cuda_slots);
        free(completions);
        return -1;
    }
    
    free(chunk_state);
    free(cuda_slots);
    free(completions);
    
    IONIC_INFO(&ctx->logger, IONIC_EVENT_TAG_PIPELINE,
               "completed tensors=%zu chunks=%zu bytes=%llu",
               p->n_tensors, p->n_chunks, 
               (unsigned long long)atomic_load(&p->loaded_bytes));
    
    return (int)p->n_tensors;
}

#else

int ionic_pipeline_execute_plan(
    struct ionic_context *ctx,
    struct ionic_pipeline *p,
    struct ionic_sharding_plan *plan,
    unsigned short rank)
{
    if (!p || !plan || plan->n == 0)
        return -1;
    
    free(p->statuses);
    free(p->chunks);
    p->statuses = NULL;
    p->chunks = NULL;
    
    if (ionic_pipeline_init_from_plan(ctx, p, plan, rank) != 0)
        return -1;
    
    if (ionic_pipeline_build_chunks(ctx, p) != 0)
        return -1;
    
    struct ionic_backend *backend = ctx->backend;
    
    for (size_t i = 0; i < p->n_tensors; i++) {
        struct ionic_tensor_status *ts = &p->statuses[i];
        atomic_store_explicit(&ts->state, IONIC_TENSOR_STATE_LOADING, memory_order_release);
        
        size_t read = backend->read(ctx, ts->fd, ts->device_ptr, ts->total_bytes, ts->start_offset);
        
        if (read != ts->total_bytes) {
            ts->error_code = -1;
            atomic_store_explicit(&ts->state, IONIC_TENSOR_STATE_FAILED, memory_order_release);
            return -1;
        }
        
        atomic_store_explicit(&ts->bytes_loaded, ts->total_bytes, memory_order_release);
        atomic_store_explicit(&ts->state, IONIC_TENSOR_STATE_READY, memory_order_release);
        atomic_fetch_add_explicit(&p->loaded_bytes, ts->total_bytes, memory_order_relaxed);
    }
    
    return (int)p->n_tensors;
}

#endif
