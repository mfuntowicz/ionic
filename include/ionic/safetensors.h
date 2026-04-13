#ifndef IONIC_SAFETENSORS_H
#define IONIC_SAFETENSORS_H

#ifdef __cplusplus
extern "C" {
#endif

#define SAFETENSORS_MAX_HEADER_SIZE

#include <stddef.h>
#include <ionic/types.h>

struct ionic_safetensors {
    struct ionic_tensor *tensors;
    // unsigned short      *locations;
    char                **names;
    char                **files;
    size_t              n_tensors;
    size_t              n_files;
    size_t              hdr_size;
    unsigned char       _pad[9];
};

typedef struct ionic_safetensors ionic_safetensors_t;

static_assert(sizeof(struct ionic_safetensors) == 64, "struct ionic_safetensors is");

void ionic_safetensors_init(struct ionic_safetensors *);
void ionic_safetensors_destroy(struct ionic_safetensors *);
size_t ionic_safetensors_discover_tensors(struct ionic_context *, struct ionic_safetensors *, const char *);

#ifdef __cplusplus
}
#endif

#endif // IONIC_SAFETENSORS_H