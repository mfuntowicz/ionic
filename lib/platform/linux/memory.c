#include <stdalign.h>
#include <stddef.h>
#include <stdint.h>
#include <string.h>
#include <sys/mman.h>
#include <unistd.h>
#include "ionic/types.h"

#define IONIC_HOST_MAP_MAGIC 0x31434948u /* "HIC1" little-endian tag */

struct ionic_host_map_hdr {
    uint32_t magic;
    uint32_t reserved;
    size_t map_bytes;
};

static const size_t ionic_host_hdr_user_off =
    (sizeof(struct ionic_host_map_hdr) + alignof(max_align_t) - 1) & ~(alignof(max_align_t) - 1);

#ifndef MAP_HUGE_SHIFT
#define MAP_HUGE_SHIFT 26
#endif
#ifndef MAP_HUGE_2MB
#define MAP_HUGE_2MB (21 << MAP_HUGE_SHIFT)
#endif

static size_t ionic_round_up(size_t n, size_t align) {
    if (align == 0u)
        return n;
    size_t r = n % align;
    if (r == 0u)
        return n;
    size_t add = align - r;
    if (n > SIZE_MAX - add)
        return 0u;
    return n + add;
}

static void *ionic_linux_host_mmap_allocate_impl(size_t user_size, enum ionic_allocation_kind kind) {
    (void)kind;
    if (user_size == 0u)
        return NULL;

    const size_t huge_2mb = 2u * 1024u * 1024u;
    const size_t page = (size_t)getpagesize();
    if (page == 0u)
        return NULL;

    size_t need = ionic_host_hdr_user_off + user_size;
    if (need < user_size)
        return NULL;

    size_t map_sz = ionic_round_up(need, huge_2mb);
    if (map_sz == 0u)
        map_sz = need;

    void *p = MAP_FAILED;

#if defined(MAP_HUGETLB) && defined(MAP_HUGE_2MB)
    p = mmap(NULL, map_sz, PROT_READ | PROT_WRITE,
             MAP_PRIVATE | MAP_ANONYMOUS | MAP_HUGETLB | MAP_HUGE_2MB, -1, 0);
#endif

    if (p == MAP_FAILED) {
        map_sz = ionic_round_up(need, page);
        if (map_sz == 0u)
            return NULL;
        p = mmap(NULL, map_sz, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
        if (p == MAP_FAILED)
            return NULL;
#ifdef MADV_HUGEPAGE
        (void)madvise(p, map_sz, MADV_HUGEPAGE);
#endif
    }

    struct ionic_host_map_hdr *hdr = (struct ionic_host_map_hdr *)p;
    hdr->magic = IONIC_HOST_MAP_MAGIC;
    hdr->reserved = 0u;
    hdr->map_bytes = map_sz;
    if (ionic_host_hdr_user_off > sizeof(struct ionic_host_map_hdr))
        memset((unsigned char *)p + sizeof(struct ionic_host_map_hdr), 0,
               ionic_host_hdr_user_off - sizeof(struct ionic_host_map_hdr));

    return (unsigned char *)p + ionic_host_hdr_user_off;
}

static void ionic_linux_host_mmap_free_impl(void *user_ptr, enum ionic_allocation_kind kind) {
    (void)kind;
    if (!user_ptr)
        return;

    unsigned char *base = (unsigned char *)user_ptr - ionic_host_hdr_user_off;
    struct ionic_host_map_hdr *hdr = (struct ionic_host_map_hdr *)base;
    if (hdr->magic != IONIC_HOST_MAP_MAGIC || hdr->map_bytes == 0u)
        return;

    (void)munmap(base, hdr->map_bytes);
}

void *ionic_linux_cpu_allocate(struct ionic_context *ctx, size_t size, enum ionic_allocation_kind kind) {
    (void)ctx;
    return ionic_linux_host_mmap_allocate_impl(size, kind);
}

void ionic_linux_cpu_free(struct ionic_context *ctx, void *ptr, enum ionic_allocation_kind kind) {
    (void)ctx;
    ionic_linux_host_mmap_free_impl(ptr, kind);
}
