#ifndef IONIC_PLATFORM_POSIX_SHM_H
#define IONIC_PLATFORM_POSIX_SHM_H

#include <stddef.h>

void *ionic_shm_create(const char *name, size_t size);
void *ionic_shm_open(const char *name, size_t size);
void  ionic_shm_close(void *ptr, size_t size);
int   ionic_shm_unlink(const char *name);

#endif // IONIC_PLATFORM_POSIX_SHM_H
