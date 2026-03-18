#include <ionic/error.h>
#include <ionic/safetensors.h>

#include <errno.h>
#include <fcntl.h>
#include <stdint.h>
#include <stdlib.h>
#include <yyjson.h>

#include "ionic/error.h"
#include "ionic/ionic.h"
#include "ionic/logging.h"
#include "ionic/types.h"


#define IONIC_SAFETENSORS_EVENT_TAG "safetensors"


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
    registry->n = 0;
    registry->hsize = 0;
}

void ionic_safetensors_destroy(ionic_safetensors_t *registry) {
    if(registry->tensors) free(registry->tensors);
    if(registry->names) free(registry->names);
    registry->n = 0;
    registry->hsize = 0;
}

size_t ionic_safetensors_extract_registry(struct ionic_context *context, struct ionic_safetensors *registry, yyjson_val *const root, size_t offset) {
    if(ionic_has_error(&context->error) || !root) goto ko;

    size_t n = yyjson_obj_size(root);
    IONIC_INFO(&context->logger, IONIC_SAFETENSORS_EVENT_TAG, "discovery tensors=%zu", n);

    registry->n = n;
    registry->tensors = calloc(n, sizeof(ionic_tensor_t));
    registry->names = calloc(n, sizeof(char *));
    if(!registry->tensors || !registry->names) {
        IONIC_ERROR(&context->logger, IONIC_SAFETENSORS_EVENT_TAG, "allocation_failed");
        goto ko;
    }
    
    size_t i = 0, max = 0, tidx = 0;
    yyjson_val *key, *val;
    yyjson_obj_foreach(root, i, max, key, val) {
        const char *name = yyjson_get_str(key);
        const unsigned is_metadata = strcmp(name, "__metadata__") == 0;

        if(!name) continue;
        if (!yyjson_is_obj(val)) {
            context->error = IONIC_ERR(IONIC_ERROR_SAFETENSORS_JSON_MALFORMED_HEADER);
            goto ko;
        }

        struct ionic_tensor *tensor = registry->tensors + offset + tidx;
        struct ionic_error error = ionic_safetensors_parse_tensor(val, tensor);
        if (ionic_has_error(&error)) {
            context->error = IONIC_ERR(IONIC_ERROR_SAFETENSORS_JSON_MALFORMED_HEADER);
            goto ko;
        }

        tensor->start += registry->hsize + sizeof(registry->hsize);
        tensor->end += registry->hsize + sizeof(registry->hsize);
        registry->names[offset + tidx] = strndup(name, yyjson_get_len(val));
        tidx++;

        IONIC_TRACE(
            &context->logger, IONIC_SAFETENSORS_EVENT_TAG, "\ttensor name=\"%s\" start=%zu end=%zu", name, tensor->start,  tensor->end);
    }
ko:
    return 0;
}

size_t ionic_safetensors_discover_tensors_from_file(ionic_context_t *context, ionic_safetensors_t *registry, const char *const path) {
    IONIC_INFO(&context->logger, IONIC_SAFETENSORS_EVENT_TAG, "file path=\"%s\"", path);
    
    if(ionic_has_error(&context->error)) goto ko;
    
    int fd = open(path, O_RDONLY | O_DIRECT);
    if(fd < 0) {
        IONIC_ERROR(&context->logger, IONIC_SAFETENSORS_EVENT_TAG, "file_open_failed path=\"%s\" errno=%d", path, errno);
        goto ko;
    }

    char dst[64 * 1024];
    size_t n = context->backend->read(context, fd, (unsigned char *)dst, sizeof(dst), 0);
    
    if(n > 0) {
        memcpy(&registry->hsize, dst, sizeof(registry->hsize));
        if(registry->hsize > 0) {
            IONIC_DEBUG(&context->logger, IONIC_SAFETENSORS_EVENT_TAG, "file_read_success path=\"%s\" hsize=%zu", path, registry->hsize);

            yyjson_doc *doc = yyjson_read(dst + sizeof(uint64_t), registry->hsize, YYJSON_READ_NOFLAG);
            if(!doc){
                IONIC_ERROR(&context->logger, IONIC_SAFETENSORS_EVENT_TAG, "file_read_failed path=\"%s\" errno=%d", path, errno);
                goto ko;
            }
            

            yyjson_val *root = yyjson_doc_get_root(doc);
            if(!root){
                IONIC_ERROR(&context->logger, IONIC_SAFETENSORS_EVENT_TAG, "file_read_failed path=\"%s\" errno=%d", path, errno);
                goto ko;
            }
            
            ionic_safetensors_extract_registry(context, registry, root, 0);
            yyjson_doc_free(doc);
            return n;
        }
    }


ko:
    registry->hsize = 0;
    return 0;
}

size_t ionic_safetensors_discover_tensors_from_index(ionic_context_t *context, ionic_safetensors_t *registry, yyjson_doc *const doc) {
    if(ionic_has_error(&context->error) || !doc) goto ko;
    
    size_t n = yyjson_doc_get_val_count(doc);
    IONIC_INFO(&context->logger, IONIC_SAFETENSORS_EVENT_TAG, "discovery count=%zu", n);
    return n;

ko:
    return 0;
}

size_t ionic_safetensors_discover_tensors(ionic_context_t *context, ionic_safetensors_t *registry, const char *path) {
    if(ionic_has_error(&context->error)) goto ko;

    // we first try to read the file as a json index file
    yyjson_doc *doc = yyjson_read_file(path, YYJSON_READ_NOFLAG, NULL, NULL);
    if(doc) {
        IONIC_INFO(&context->logger, IONIC_SAFETENSORS_EVENT_TAG, "index path=\"%s\"", path);
        return ionic_safetensors_discover_tensors_from_index(context, registry, doc);
    }
    
    return ionic_safetensors_discover_tensors_from_file(context, registry, path);

free_doc:
    yyjson_doc_free(doc);

ko:
    return 0;
}