#include "ionic/logging.h"
#include <ionic/safetensors.h>
#include <stdint.h>
#include <stdlib.h>
#include <yyjson.h>

void ionic_safetensors_init(ionic_safetensors_t *registry) {
    registry->tensors = NULL;
    registry->names = NULL;
    registry->n = 0;
}

void ionic_safetensors_destroy(ionic_safetensors_t *registry) {
    if(registry->tensors) free(registry->tensors);
    if(registry->names) free(registry->names);
    registry->n = 0;
}

size_t ionic_safetensors_extract_registry(struct ionic_context *context, struct ionic_safetensors *registry, yyjson_doc *const doc) {
    if(ionic_has_error(&context->error) || !doc) goto ko;

    yyjson_val *root = yyjson_doc_get_root(doc);

ko:
    return 0;
}

size_t ionic_safetensors_discover_tensors_from_file(ionic_context_t *context, ionic_safetensors_t *registry, const char *const path) {
    IONIC_INFO(&context->logger, "Reading safetensors file %s", path);
    
    if(ionic_has_error(&context->error)) goto ko;
    
    

ko:
    return 0;
}

size_t ionic_safetensors_discover_tensors_from_index(ionic_context_t *context, ionic_safetensors_t *registry, yyjson_doc *const doc) {
    if(ionic_has_error(&context->error) || !doc) goto ko;
    
    size_t n = yyjson_doc_get_val_count(doc);
    IONIC_INFO(&context->logger, "Discovered %zu tensors", n);
    return n;

ko:
    return 0;
}

size_t ionic_safetensors_discover_tensors(ionic_context_t *context, ionic_safetensors_t *registry, const char *path) {
    if(ionic_has_error(&context->error)) goto ko;

    // we first try to read the file as a json index file
    yyjson_doc *doc = yyjson_read_file(path, YYJSON_READ_NOFLAG, NULL, NULL);
    if(doc) {
        IONIC_INFO(&context->logger, "Reading safetensors index file %s", path);
        return ionic_safetensors_discover_tensors_from_index(context, registry, doc);
    }
    
    return ionic_safetensors_discover_tensors_from_file(context, registry, path);

free_doc:
    yyjson_doc_free(doc);

ko:
    return 0;
}