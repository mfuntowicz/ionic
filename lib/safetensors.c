#include "ionic/safetensors.h"

#include <errno.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <yyjson.h>

#include "ionic/error.h"
#include "ionic/file.h"
#include "ionic/ionic.h"
#include "ionic/logging.h"
#include "ionic/types.h"


#define IONIC_EVENT_TAG_SAFETENSORS "safetensors"


static inline size_t ionic_path_parent(const char *path, char *const dst, size_t dst_size)
{
    const char *sep = strrchr(path, '/');
    if (!sep)
        return 0;

    size_t len = (size_t)(sep - path + 1);
    if (len >= dst_size)
        return 0;

    memcpy(dst, path, len);
    dst[len] = '\0';
    return len;
}

static inline char *ionic_path_join(const char *parent, size_t parent_len, const char *filename, size_t filename_len)
{
    char *path = malloc(parent_len + filename_len + 1);
    if (!path)
        return NULL;

    memcpy(path, parent, parent_len);
    memcpy(path + parent_len, filename, filename_len);
    path[parent_len + filename_len] = '\0';
    return path;
}


static inline enum ionic_data_type ionic_safetensors_dtype_from_str(const char *dtype, const size_t size)
{
    if (size == 0) return IONIC_DATA_TYPE_UNKNOWN;
    if (strncmp(dtype, "BOOL", size) == 0) return IONIC_DATA_TYPE_BOOL;
    if (strncmp(dtype, "BF16", size) == 0) return IONIC_DATA_TYPE_BFLOAT16;
    if (strncmp(dtype, "C64", size) == 0) return IONIC_DATA_TYPE_COMPLEX;
    if (strncmp(dtype, "F4", size) == 0) return IONIC_DATA_TYPE_FLOAT4;
    if (strncmp(dtype, "F6_E2M3", size) == 0) return IONIC_DATA_TYPE_FLOAT6_E2M3;
    if (strncmp(dtype, "F6_E3M2", size) == 0) return IONIC_DATA_TYPE_FLOAT6_E3M2;
    if (strncmp(dtype, "F8_E8M0", size) == 0) return IONIC_DATA_TYPE_FLOAT8_E8M0;
    if (strncmp(dtype, "F8_E4M3", size) == 0) return IONIC_DATA_TYPE_FLOAT8_E4M3;
    if (strncmp(dtype, "F8_E5M2", size) == 0) return IONIC_DATA_TYPE_FLOAT8_E5M2;
    if (strncmp(dtype, "F16", size) == 0) return IONIC_DATA_TYPE_FLOAT16;
    if (strncmp(dtype, "F32", size) == 0) return IONIC_DATA_TYPE_FLOAT32;
    if (strncmp(dtype, "F64", size) == 0) return IONIC_DATA_TYPE_FLOAT64;
    if (strncmp(dtype, "I8", size) == 0) return IONIC_DATA_TYPE_SIGNED_INT8;
    if (strncmp(dtype, "I16", size) == 0) return IONIC_DATA_TYPE_SIGNED_INT16;
    if (strncmp(dtype, "I32", size) == 0) return IONIC_DATA_TYPE_SIGNED_INT32;
    if (strncmp(dtype, "I64", size) == 0) return IONIC_DATA_TYPE_SIGNED_INT64;
    if (strncmp(dtype, "U8", size) == 0) return IONIC_DATA_TYPE_UNSIGNED_INT8;
    if (strncmp(dtype, "U16", size) == 0) return IONIC_DATA_TYPE_UNSIGNED_INT16;
    if (strncmp(dtype, "U32", size) == 0) return IONIC_DATA_TYPE_UNSIGNED_INT32;
    if (strncmp(dtype, "U64", size) == 0) return IONIC_DATA_TYPE_UNSIGNED_INT64;

    return IONIC_DATA_TYPE_UNKNOWN;
}


static inline struct ionic_error ionic_safetensors_parse_dtype(yyjson_val *dtype, struct ionic_tensor *tensor)
{
    if (dtype && yyjson_is_str(dtype)) {
        const char* dtype_str = yyjson_get_str(dtype);
        const size_t dtype_len = yyjson_get_len(dtype);
        tensor->dtype = ionic_safetensors_dtype_from_str(dtype_str, dtype_len);
    } else {
        tensor->dtype = IONIC_DATA_TYPE_UNKNOWN;
    }

    if (tensor->dtype == IONIC_DATA_TYPE_UNKNOWN)
        return IONIC_ERR(IONIC_ERROR_SAFETENSORS_UNKNOWN_DTYPE);

    return IONIC_SUCCESS;
}

static inline struct ionic_error ionic_safetensors_parse_offsets(yyjson_val *offsets, struct ionic_tensor *tensor)
{
    if (offsets && yyjson_is_arr(offsets)) {
        const size_t length = yyjson_arr_size(offsets);
        if (length >= 2) {
            yyjson_val* start_val = yyjson_arr_get(offsets, 0);
            yyjson_val* end_val = yyjson_arr_get(offsets, 1);

            if (yyjson_is_uint(start_val))
                tensor->start = yyjson_get_uint(start_val);

            if (yyjson_is_uint(end_val))
                tensor->end = yyjson_get_uint(end_val);

            return IONIC_SUCCESS;
        }
        return IONIC_ERR(IONIC_ERROR_SAFETENSORS_JSON_MALFORMED_HEADER);
    }
    return IONIC_ERR(IONIC_ERROR_SAFETENSORS_JSON_MALFORMED_HEADER);
}

static inline struct ionic_error ionic_safetensors_parse_shape(yyjson_val *shape, struct ionic_tensor *tensor)
{
    if (shape && yyjson_is_arr(shape)) {
        const size_t rank = yyjson_arr_size(shape);
        tensor->rank = (uint8_t)rank;

        if (rank > 0) {
            size_t shape_idx = 0, shape_max = 0;
            yyjson_val* dim_val;
            yyjson_arr_foreach(shape, shape_idx, shape_max, dim_val) {
                if (yyjson_is_uint(dim_val))
                    tensor->shape[shape_idx] = yyjson_get_uint(dim_val);
            }
            return IONIC_SUCCESS;
        }
        return IONIC_SUCCESS;
    }
    return IONIC_ERR(IONIC_ERROR_SAFETENSORS_JSON_MALFORMED_HEADER);
}

static inline struct ionic_error ionic_safetensors_parse_tensor(yyjson_val *specs, struct ionic_tensor *tensor)
{
    struct ionic_error error = IONIC_SUCCESS;

    // Parse dtype
    yyjson_val* dtype_val = yyjson_obj_get(specs, "dtype");
    error = ionic_safetensors_parse_dtype(dtype_val, tensor);

    if (ionic_has_error(&error)) return error;

    // Parse shape
    yyjson_val* shape_val = yyjson_obj_get(specs, "shape");
    error = ionic_safetensors_parse_shape(shape_val, tensor);

    if (ionic_has_error(&error)) return error;

    // Parse offsets
    yyjson_val* data_offsets_val = yyjson_obj_get(specs, "data_offsets");
    return ionic_safetensors_parse_offsets(data_offsets_val, tensor);
}


void ionic_safetensors_init(ionic_safetensors_t *registry) {
    registry->tensors = NULL;
    registry->names = NULL;
    registry->files = NULL;
    registry->n_tensors = 0;
    registry->n_files   = 0;
    registry->hdr_size  = 0;
}

void ionic_safetensors_destroy(ionic_safetensors_t *registry) {
    if(registry->tensors) free(registry->tensors);
    if(registry->names) free(registry->names);
    if(registry->files) free(registry->files);
    registry->n_tensors = 0;
    registry->n_files   = 0;
    registry->hdr_size  = 0;
}

static size_t ionic_safetensors_extract_registry(
    struct ionic_context *ctx, struct ionic_safetensors *registry, yyjson_val *const root, size_t offset, unsigned short file) {
    if(ionic_has_error(&ctx->error) || !root) goto ko;

    size_t n = yyjson_obj_size(root);
    IONIC_INFO(&ctx->logger, IONIC_EVENT_TAG_SAFETENSORS, "discovery tensors=%zu", n);

    // if coming from an index file, we preallocate the tensors and names arrays across all the files
    if (!registry->tensors || !registry->names) {
        registry->n_tensors = n;
        registry->tensors   = calloc(n, sizeof(ionic_tensor_t));
        registry->names     = calloc(n, sizeof(char *));
        if(!registry->tensors || !registry->names) {
            IONIC_ERROR(&ctx->logger, IONIC_EVENT_TAG_SAFETENSORS, "allocation_failed");
            goto ko;
        }
    }
    
    size_t i = 0, max = 0, tidx = 0;
    yyjson_val *key, *val;
    yyjson_obj_foreach(root, i, max, key, val) {
        const char *name = yyjson_get_str(key);
        const unsigned is_metadata = strcmp(name, "__metadata__") == 0;

        if(!name || is_metadata) continue;
        if (!yyjson_is_obj(val)) {
            ctx->error = IONIC_ERR(IONIC_ERROR_SAFETENSORS_JSON_MALFORMED_HEADER);
            goto ko;
        }

        struct ionic_tensor *tensor = registry->tensors + offset + tidx;
        struct ionic_error error = ionic_safetensors_parse_tensor(val, tensor);
        if (ionic_has_error(&error)) {
            ctx->error = IONIC_ERR(IONIC_ERROR_SAFETENSORS_JSON_MALFORMED_HEADER);
            goto ko;
        }

        tensor->start += registry->hdr_size + sizeof(registry->hdr_size);
        tensor->end += registry->hdr_size + sizeof(registry->hdr_size);
        registry->names[offset + tidx] = strndup(name, yyjson_get_len(key));
        registry->locations[offset + tidx] = file;
        tidx++;

        IONIC_TRACE(
            &ctx->logger, IONIC_EVENT_TAG_SAFETENSORS, "\tstart=%-12zu end=%-12zu tensor name=%s", tensor->start,  tensor->end, name);
    }
    return tidx;
ko:
    return 0;
}

static size_t ionic_safetensors_discover_tensors_from_file(ionic_context_t *ctx, ionic_safetensors_t *registry, const char *const path, size_t offset, unsigned short file) {
    IONIC_INFO(&ctx->logger, IONIC_EVENT_TAG_SAFETENSORS, "file path=%s", path);
    
    if(ionic_has_error(&ctx->error)) goto ko;

    struct ionic_file f = ionic_ro_mmap(path);
    if(f.fd < 0) {
        IONIC_ERROR(&ctx->logger, IONIC_EVENT_TAG_SAFETENSORS, "open failed errno=%d path=%s", errno, path);
        goto ko;
    }
    
    memcpy(&registry->hdr_size, f.content, sizeof(registry->hdr_size));
    if(registry->hdr_size > 0) {
        IONIC_DEBUG(&ctx->logger, IONIC_EVENT_TAG_SAFETENSORS, "read success hsize=%-12zu path=%s", registry->hdr_size, path);

        yyjson_doc *doc = yyjson_read(f.content + sizeof(uint64_t), registry->hdr_size, YYJSON_READ_NOFLAG);
        if(!doc){
            IONIC_ERROR(&ctx->logger, IONIC_EVENT_TAG_SAFETENSORS, "read failed errno=%-2d path=%s", errno, path);
            goto ko;
        }

        yyjson_val *root = yyjson_doc_get_root(doc);
        if(!root){
            IONIC_ERROR(&ctx->logger, IONIC_EVENT_TAG_SAFETENSORS, "read failed errno=%-2d path=%s", errno, path);
            goto ko;
        }

        size_t count = ionic_safetensors_extract_registry(ctx, registry, root, offset, file);
        yyjson_doc_free(doc);
        return count;
    }

ko:
    registry->hdr_size = 0;
    return 0;
}

static size_t ionic_safetensors_discover_tensors_from_index(
    ionic_context_t *ctx, ionic_safetensors_t *registry, yyjson_val *const root, const char *const workspace, size_t workspace_len) {
    if(ionic_has_error(&ctx->error) || !root) goto ko;
    
    yyjson_val *weights = yyjson_obj_get(root, "weight_map");
    if(!weights) {
        IONIC_ERROR(&ctx->logger, IONIC_EVENT_TAG_SAFETENSORS, "weight_map not found");
        ctx->error = IONIC_ERR(IONIC_ERROR_SAFETENSORS_JSON_MALFORMED_HEADER);
        goto ko;
    }
    
    registry->n_tensors = yyjson_obj_size(weights);
    IONIC_INFO(&ctx->logger, IONIC_EVENT_TAG_SAFETENSORS, "index items=%zu", registry->n_tensors);

    if(registry->n_tensors <= 0) {
        IONIC_ERROR(&ctx->logger, IONIC_EVENT_TAG_SAFETENSORS, "index weight_map=none");
        ctx->error = IONIC_ERR(IONIC_ERROR_SAFETENSORS_JSON_MALFORMED_HEADER);
        goto ko;
    }

    registry->tensors     = calloc(registry->n_tensors, sizeof(ionic_tensor_t));
    registry->locations   = calloc(registry->n_tensors, sizeof(unsigned short));
    registry->names       = calloc(registry->n_tensors, sizeof(char *));
    registry->files = calloc(registry->n_tensors, sizeof(char *));
    if (!registry->tensors || !registry->locations || !registry->names || !registry->files) {
        ionic_set_error(ctx, IONIC_ERR(IONIC_ERROR_ALLOCATION_FAILED));
        goto freeup;
    }

    size_t idx = 0, max = 0;
    yyjson_val *key, *val;
    yyjson_obj_foreach(weights, idx, max, key, val) {
        const char *file = yyjson_get_str(val);
        if (!file) continue;

        size_t fi = 0;
        for (; fi < registry->n_files; fi++) {
            if (strcmp(registry->files[fi], file) == 0)
                break;
        }

        if (fi == registry->n_files) {
            registry->files[registry->n_files] = strndup(file, yyjson_get_len(val));
            ++registry->n_files;
        }
    }

    char **files = realloc(registry->files, registry->n_files * sizeof(char *));
    if (files)
        registry->files = files;

    IONIC_INFO(&ctx->logger, IONIC_EVENT_TAG_SAFETENSORS, "index files=%zu tensors=%zu", registry->n_files, registry->n_tensors);

    size_t offset = 0;
    for (size_t fi = 0; fi < registry->n_files; fi++) {
        char *shard_path = ionic_path_join(workspace, workspace_len, registry->files[fi], strlen(registry->files[fi]));
        if (!shard_path) goto freeup_files;

        offset += ionic_safetensors_discover_tensors_from_file(ctx, registry, shard_path, offset, fi);
        free(shard_path);

        if (ionic_has_error(&ctx->error))
            break;
    }

    return offset;

freeup_files:
    for (size_t fi = 0; fi < registry->n_files; fi++)
        free(registry->files[fi]);
    free(registry->files);

freeup:
    if(registry->tensors) free(registry->tensors);
    if(registry->names) free(registry->names);

ko:
    registry->n_tensors = 0;
    return 0;
}

size_t ionic_safetensors_discover_tensors(ionic_context_t *ctx, ionic_safetensors_t *registry, const char *path) {
    if(ionic_has_error(&ctx->error)) return 0;

    // we first try to read the file as a JSON index file
    size_t n_tensors = 0;
    yyjson_doc *doc = yyjson_read_file(path, YYJSON_READ_NOFLAG, NULL, NULL);
    if(doc) {
        IONIC_INFO(&ctx->logger, IONIC_EVENT_TAG_SAFETENSORS, "index path=%s", path);

        char cwd[4096];  // max path length
        size_t length = ionic_path_parent(path, cwd, sizeof(cwd));
        yyjson_val *root = yyjson_doc_get_root(doc);
        n_tensors = ionic_safetensors_discover_tensors_from_index(ctx, registry, root, cwd, length);

        yyjson_doc_free(doc);
    } else {
        n_tensors = ionic_safetensors_discover_tensors_from_file(ctx, registry, path, 0, 0);
    }

    return n_tensors;
}