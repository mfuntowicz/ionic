#ifndef IONIC_SAFETENSORS_H
#define IONIC_SAFETENSORS_H

#ifdef __cplusplus
extern "C" {
#endif

#include <stddef.h>
#include <ionic/ionic.h>

struct ionic_safetensors {
    ionic_tensor_t *tensors;
    size_t n;
    char **names;
};

typedef struct ionic_safetensors ionic_safetensors_t;

void ionic_safetensors_init(struct ionic_safetensors *);
void ionic_safetensors_destroy(struct ionic_safetensors *);
size_t ionic_safetensors_discover_tensors(struct ionic_context *, struct ionic_safetensors *, const char *);

#ifdef __cplusplus
}
#endif

#endif // IONIC_SAFETENSORS_H