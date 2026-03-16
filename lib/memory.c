#include <stdlib.h>
#include <stdint.h>

void *ionic_alloc(size_t size) {
    return malloc(size);
}

void ionic_free(void *ptr) {
    free(ptr);
}