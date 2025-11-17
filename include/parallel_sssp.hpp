#ifndef PARALLEL_SSSP_HPP
#define PARALLEL_SSSP_HPP

#include "graph.hpp"
#include <mpi.h>
#include <omp.h>
#include <vector>
#include <memory>

namespace sssp {
namespace parallel {

// MPI + OpenMP parallel SSSP implementation
class DistributedSSSP {
private:
    const Graph& graph;
    vertex_t source;

    // MPI parameters
    int rank;
    int world_size;

    // Graph partitioning
    std::vector<vertex_t> local_vertices;
    vertex_t local_start, local_end;

    // Distance arrays
    std::vector<weight_t> local_db;
    std::vector<vertex_t> local_pred;
    std::vector<bool> local_complete;

    // Global distance array (replicated)
    std::vector<weight_t> global_db;

    // Communication buffers
    struct UpdateMessage {
        vertex_t vertex;
        weight_t distance;
        vertex_t predecessor;
    };
    std::vector<UpdateMessage> send_buffer;
    std::vector<UpdateMessage> recv_buffer;

    // Partition graph by vertices
    void partition_graph();

    // Synchronize distance updates across processes
    void synchronize_distances();

    // Local relaxation with OpenMP
    void parallel_relax(const std::set<vertex_t>& active_vertices);

public:
    DistributedSSSP(const Graph& g, vertex_t src);
    ~DistributedSSSP();

    // Compute SSSP with MPI + OpenMP
    void compute();

    // Get results (only valid on rank 0 after gather)
    const std::vector<weight_t>& get_distances() const { return global_db; }

    // Gather results to rank 0
    void gather_results();
};

// OpenMP-only parallel SSSP
class SharedMemorySSSP {
private:
    const Graph& graph;
    vertex_t source;

    std::vector<weight_t> db;
    std::vector<vertex_t> pred;
    std::vector<bool> complete;

    uint32_t k, t;
    int num_threads;

    // Parallel edge relaxation
    void parallel_relax_edges(const std::vector<vertex_t>& vertices);

public:
    SharedMemorySSSP(const Graph& g, vertex_t src, int threads = 0);

    void compute();

    weight_t get_distance(vertex_t v) const { return db[v]; }
    std::vector<vertex_t> get_path(vertex_t v) const;
};

} // namespace parallel
} // namespace sssp

#endif // PARALLEL_SSSP_HPP
