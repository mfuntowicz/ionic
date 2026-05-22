#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#include <errno.h>

#include <ionic/platform/posix/shm.h>

static inline void ionic_sched_yield_intrinsic(void) {
    #if defined(__x86_64__) || defined(__i386__)
        __builtin_ia32_pause();
    #elif defined(__aarch64__) || defined(__arm__)
        __asm__ volatile("yield" ::: "memory");
    #else
        ((void)0);
    #endif
}

void *ionic_shm_create(const char *name, size_t size) {
    int fd = shm_open(name, O_CREAT | O_EXCL | O_RDWR, 0600);
    if (fd < 0)
        return NULL;

    if (ftruncate(fd, (off_t)size) != 0) {
        close(fd);
        shm_unlink(name);
        return NULL;
    }

    void *ptr = mmap(NULL, size, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
    close(fd);

    if (ptr == MAP_FAILED) {
        shm_unlink(name);
        return NULL;
    }

    return ptr;
}

void *ionic_shm_open(const char *name, size_t size) {
    int fd;
    for (;;) {
        fd = shm_open(name, O_RDWR, 0);
        if (fd >= 0)
            break;
        if (errno != ENOENT)
            return NULL;
        ionic_sched_yield_intrinsic();
    }

    void *ptr = mmap(NULL, size, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
    close(fd);

    if (ptr == MAP_FAILED)
        return NULL;

    return ptr;
}

void ionic_shm_close(void *ptr, size_t size) {
    if (ptr)
        munmap(ptr, size);
}

int ionic_shm_unlink(const char *name) {
    return shm_unlink(name);
}
