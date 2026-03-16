#include <stddef.h>
#include <ionic/types.h>

struct ionic_safetensors {
    ionic_tensor_t *tensors;
    size_t n;
    char *names[];
};

typedef struct ionic_safetensors ionic_safetensors_t;

size_t ionic_safetensors_init(ionic_safetensors_t *safetensors);