#ifndef IONIC_GROUP_H
#define IONIC_GROUP_H

#include <stdatomic.h>
#include <stddef.h>

#define IONIC_GROUP_MAX_IDENT 58

struct ionic_context;

typedef struct ionic_group {
    atomic_uint   steps;
    atomic_uchar  ready;
    unsigned char count;
    char          identifier[IONIC_GROUP_MAX_IDENT];
    void          *device_ptrs[];
} ionic_group_t;

struct ionic_group *ionic_group_create(struct ionic_context *, const char *identifier, unsigned char count);
struct ionic_group *ionic_group_open(struct ionic_context *, const char *identifier, unsigned char count);
void ionic_group_wait(struct ionic_group *);
void ionic_group_close(struct ionic_group *);
void ionic_group_destroy(struct ionic_group *);

#endif