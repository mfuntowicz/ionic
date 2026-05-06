//
// Created by mfuntowicz on 3/30/26.
//

#ifndef IONIC_PLATFORM_LINUX_FILE_H
#define IONIC_PLATFORM_LINUX_FILE_H

#include <fcntl.h>
#include <unistd.h>
#include <sys/mman.h>
#include <ionic/types.h>
#include <sys/stat.h>

static inline struct ionic_file ionic_ro_mmap(const char *path) {
    int fd = open(path, O_RDONLY);
    if (fd == -1) goto ko;

    struct stat stats;
    if (fstat(fd, &stats) < 0)
        goto cleanup;

    void *addr = NULL;
    if ((addr = mmap(NULL, stats.st_size, PROT_READ | PROT_WRITE, MAP_PRIVATE, fd, 0)) != MAP_FAILED)
        return (struct ionic_file) { .fd = fd, .content = addr };

cleanup:
    close(fd);
ko:
    return (struct ionic_file) { .fd = -1, .content = NULL};
}

#endif //IONIC_PLATFORM_LINUX_FILE_H