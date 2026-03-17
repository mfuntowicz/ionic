#include <ionic/ionic.h>
#include <numa.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

/*
 * numa_all_cpus_ptr: CPUs the calling task may execute on
 * (from /proc/self/status Cpus_allowed). Respects cgroups/cpusets;
 * when unrestricted this equals the online CPU set.
 */

/** Try to populate topology from NUMA. Returns 0 on success, -1 on fallback. */
static int ionic_numa_probe(struct ionic_topology *topo, struct ionic_logger *log) {
    if (!numa_available()) {
        IONIC_WARN(log, "NUMA not available on the platform, assuming single node topology");
        return -1;
    }

    struct bitmask *allowed = numa_all_cpus_ptr;
    if (!allowed) {
        IONIC_WARN(log, "numa_all_cpus_ptr not available, falling back to single node");
        return -1;
    }

    int max_cpu = numa_num_possible_cpus();
    int n_nodes = numa_num_configured_nodes();
    if (n_nodes <= 0 || max_cpu <= 0)
        return -1;

    unsigned char *map = (unsigned char *)malloc((size_t)max_cpu);
    if (!map)
        return -1;
    memset(map, IONIC_NODE_NONE, (size_t)max_cpu);

    struct bitmask *node_cpus = numa_allocate_cpumask();
    if (!node_cpus) {
        free(map);
        return -1;
    }

    unsigned short total = 0;
    for (int node = 0; node < n_nodes; node++) {
        if (numa_node_to_cpus(node, node_cpus) != 0)
            continue;
        
        for (int cpu = 0; cpu < max_cpu; cpu++) {
            if (numa_bitmask_isbitset(node_cpus, cpu) && numa_bitmask_isbitset(allowed, cpu)) {
                map[cpu] = (unsigned char)node;
                total++;
            }
        }
    }
    numa_bitmask_free(node_cpus);

    topo->node_of_core = map;
    topo->num_possible_cpus = (unsigned int)max_cpu;
    topo->num_nodes = (unsigned char)n_nodes;
    topo->num_cores = total;
    return 0;
}

void ionic_topology_init(struct ionic_context *ctx) {
    struct ionic_topology *topology = &ctx->topology;

    if (ionic_numa_probe(topology, &ctx->logger) != 0) {
        topology->num_cores = (unsigned short)sysconf(_SC_NPROCESSORS_ONLN);
        topology->num_nodes = 1;
    }

    IONIC_INFO(&ctx->logger, "Topology nodes: %u, cores: %u", topology->num_nodes, topology->num_cores);
}

void ionic_topology_destroy(struct ionic_context *ctx) {
    free(ctx->topology.node_of_core);
    ctx->topology = (struct ionic_topology){0};
}
