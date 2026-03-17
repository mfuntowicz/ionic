#ifndef IONIC_TOPOLOGY_H
#define IONIC_TOPOLOGY_H

#ifdef __cplusplus
extern "C" {
#endif

struct ionic_context;

/** Sentinel value: core is not available to the current task (not in cpuset) */
#define IONIC_NODE_NONE (unsigned char)0xFF

struct ionic_topology {
    unsigned short num_cores;
    unsigned char num_nodes;        
    unsigned int num_possible_cpus; /* Length of node_of_core (max CPU id + 1) */
    unsigned char *node_of_core;
};
typedef struct ionic_topology ionic_topology_t;


void ionic_topology_init(struct ionic_context *);
void ionic_topology_destroy(struct ionic_context *);

#ifdef __cplusplus
}
#endif

#endif // IONIC_TOPOLOGY_H