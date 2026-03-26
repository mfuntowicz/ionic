#include <string.h>
#include <time.h>
#include <unistd.h>
#include <stdbool.h>

#include "ionic/ionic.h"
#include "ionic/pipeline.h"
#include "ionic/utils.h"

#define IONIC_EVENT_TAG_PIPELINE "pipeline"

/*
 * Read Chunk Metadata (internal)
 */
struct ionic_chunk_meta {
    uint64_t file_offset;
    uint32_t io_size;
    uint32_t payload_offset;
    uint32_t payload_size;
    uint32_t tensor_index;
    uint64_t tensor_offset;
    uint16_t staging_slot;
    uint16_t flags;
    uint32_t _pad;
};

enum ionic_chunk_flag {
    IONIC_CHUNK_FLAG_FIRST = 1 << 0,
    IONIC_CHUNK_FLAG_LAST  = 1 << 1,
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
    size_t total_chunks = 0;
    
    // account for alignment
    for (size_t i = 0; i < p->n_tensors; i++) {
        size_t remaining = p->statuses[i].size;
        size_t file_offset = p->statuses[i].offset;
        
        while (remaining > 0) {
            size_t aligned_off = ionic_align_down_sz(file_offset, alignment);
            size_t pad = file_offset - aligned_off;
            size_t payload = remaining;
            if (payload + pad > p->config.staging_buffer_size)
                payload = p->config.staging_buffer_size - pad;
            
            total_chunks++;
            file_offset += payload;
            remaining -= payload;
        }
    }
    
    p->chunks = calloc(total_chunks, sizeof(struct ionic_chunk_meta));
    if (!p->chunks) {
        ionic_set_error(ctx, IONIC_ERR(IONIC_ERROR_ALLOCATION_FAILED));
        return -1;
    }
    p->n_chunks = total_chunks;
    
    size_t chunk_idx = 0;
    for (size_t tensor_idx = 0; tensor_idx < p->n_tensors; tensor_idx++) {
        struct ionic_tensor_status *s = &p->statuses[tensor_idx];
        
        size_t tensor_offset = 0;
        size_t file_offset = s->offset;
        size_t remaining = s->size;
        
        while (remaining > 0) {
            struct ionic_chunk_meta *c = &p->chunks[chunk_idx];
            
            size_t aligned_off = ionic_align_down_sz(file_offset, alignment);
            size_t pad = file_offset - aligned_off;
            size_t payload = remaining;
            if (payload + pad > p->config.staging_buffer_size)
                payload = p->config.staging_buffer_size - pad;
            
            size_t io_size = ionic_align_up_sz(pad + payload, alignment);
            
            c->file_offset    = aligned_off;
            c->io_size        = (uint32_t)io_size;
            c->payload_offset = (uint32_t)pad;
            c->payload_size   = (uint32_t)payload;
            c->tensor_index   = (uint32_t)tensor_idx;
            c->tensor_offset  = tensor_offset;
            c->staging_slot   = 0;
            c->flags          = 0;
            
            if (tensor_offset == 0)
                c->flags |= IONIC_CHUNK_FLAG_FIRST;
            if (remaining == payload)
                c->flags |= IONIC_CHUNK_FLAG_LAST;
            
            file_offset += payload;
            tensor_offset += payload;
            remaining -= payload;
            chunk_idx++;
        }
    }
    
    IONIC_DEBUG(&ctx->logger, IONIC_EVENT_TAG_PIPELINE,
                "built chunks tensors=%zu chunks=%zu total_bytes=%llu",
                p->n_tensors, p->n_chunks, 
                (unsigned long long)atomic_load(&p->total_bytes));
    
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
    
    /* Track chunk state: 0=pending, 1=io_done, 2=cuda_done */
    int *chunk_state = calloc(p->n_chunks, sizeof(*chunk_state));
    uint32_t *cuda_slots = calloc(p->n_chunks, sizeof(*cuda_slots));
    
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
    
    while (completed < p->n_chunks) {
        /* Phase 1: Submit all pending I/O requests */
        while (io_submitted < p->n_chunks) {
            struct ionic_chunk_meta *c = &p->chunks[io_submitted];
            struct ionic_tensor_status *ts = &p->statuses[c->tensor_index];
            
            /* Find a free CUDA slot for post-I/O processing */
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
            
            if (cuda_slot == UINT32_MAX)
                break;
            
            atomic_store_explicit(&ts->state, IONIC_TENSOR_STATE_LOADING, memory_order_release);
            atomic_store_explicit(&p->slot_state[cuda_slot], SLOT_READING, memory_order_release);
            cuda_slots[io_submitted] = cuda_slot;
            
            /* Submit async read - pass chunk index as userdata */
            int rc = backend->submit_read(ctx, ts->fd, c->file_offset + c->payload_offset, 
                                          c->payload_size, (void *)(uintptr_t)io_submitted);
            if (rc != 0) {
                IONIC_ERROR(&ctx->logger, IONIC_EVENT_TAG_PIPELINE, "async submit failed chunk=%zu", io_submitted);
                ts->error_code = rc;
                atomic_store_explicit(&ts->state, IONIC_TENSOR_STATE_FAILED, memory_order_release);
                atomic_store_explicit(&p->slot_state[cuda_slot], SLOT_FREE, memory_order_release);
                free(chunk_state);
                free(cuda_slots);
                free(completions);
                return -1;
            }
            
            io_submitted++;
        }
        
        /* Phase 2: Poll I/O completions and start CUDA transfers */
        size_t n = backend->poll_completions(ctx, completions, 64);
        
        for (size_t i = 0; i < n; i++) {
            size_t chunk_idx = (size_t)(uintptr_t)completions[i].userdata;
            struct ionic_chunk_meta *c = &p->chunks[chunk_idx];
            struct ionic_tensor_status *ts = &p->statuses[c->tensor_index];
            uint32_t cuda_slot = cuda_slots[chunk_idx];
            
            /* Copy from io_uring staging to CUDA staging */
            memcpy(p->staging_buffers[cuda_slot], completions[i].staging_buffer, completions[i].bytes_read);
            
            /* Start H2D transfer */
            unsigned char *dst = (unsigned char *)ts->device_ptr + c->tensor_offset;
            cudaError_t ce = cudaMemcpyAsync(dst, p->staging_buffers[cuda_slot], c->payload_size, 
                                             cudaMemcpyHostToDevice, p->stream);
            if (ce != cudaSuccess) {
                IONIC_ERROR(&ctx->logger, IONIC_EVENT_TAG_PIPELINE, "cudaMemcpyAsync failed: %d", ce);
                ts->error_code = (int)ce;
                atomic_store_explicit(&ts->state, IONIC_TENSOR_STATE_FAILED, memory_order_release);
                free(chunk_state);
                free(cuda_slots);
                free(completions);
                return -1;
            }
            
            ce = cudaEventRecord(p->transfer_done[cuda_slot], p->stream);
            if (ce != cudaSuccess) {
                IONIC_ERROR(&ctx->logger, IONIC_EVENT_TAG_PIPELINE, "cudaEventRecord failed: %d", ce);
                ts->error_code = (int)ce;
                atomic_store_explicit(&ts->state, IONIC_TENSOR_STATE_FAILED, memory_order_release);
                free(chunk_state);
                free(cuda_slots);
                free(completions);
                return -1;
            }
            
            atomic_store_explicit(&ts->state, IONIC_TENSOR_STATE_TRANSFERRING, memory_order_release);
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
                    struct ionic_tensor_status *ts = &p->statuses[c->tensor_index];
                    
                    uint64_t loaded = atomic_fetch_add_explicit(&ts->bytes_loaded, c->payload_size, memory_order_relaxed) + c->payload_size;
                    atomic_fetch_add_explicit(&p->loaded_bytes, c->payload_size, memory_order_relaxed);
                    
                    if (loaded >= ts->size) {
                        atomic_store_explicit(&ts->state, IONIC_TENSOR_STATE_READY, memory_order_release);
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
