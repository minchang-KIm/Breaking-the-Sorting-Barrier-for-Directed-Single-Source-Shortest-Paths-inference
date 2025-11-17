#ifndef CUDA_SSSP_CUH
#define CUDA_SSSP_CUH

#include "graph.hpp"
#include <cuda_runtime.h>
#include <vector>

namespace sssp {
namespace cuda {

// CSR (Compressed Sparse Row) format for GPU
struct CSRGraph {
    vertex_t n;
    uint64_t m;

    // Host arrays
    std::vector<uint64_t> row_ptr;
    std::vector<vertex_t> col_idx;
    std::vector<weight_t> weights;

    // Device arrays
    uint64_t* d_row_ptr;
    vertex_t* d_col_idx;
    weight_t* d_weights;

    CSRGraph(const Graph& g);
    ~CSRGraph();

    void copy_to_device();
    void free_device();
};

// GPU-accelerated SSSP using CUDA
class CudaSSSP {
private:
    const Graph& graph;
    vertex_t source;
    CSRGraph csr;

    // Device arrays
    weight_t* d_distances;
    vertex_t* d_predecessors;
    bool* d_complete;
    bool* d_updated;

    // Host arrays
    std::vector<weight_t> h_distances;
    std::vector<vertex_t> h_predecessors;

    // Allocate GPU memory
    void allocate_device_memory();

    // Free GPU memory
    void free_device_memory();

    // Copy results from device to host
    void copy_results_to_host();

public:
    CudaSSSP(const Graph& g, vertex_t src);
    ~CudaSSSP();

    // Compute SSSP on GPU
    void compute();

    // Get results
    weight_t get_distance(vertex_t v) const { return h_distances[v]; }
    const std::vector<weight_t>& get_distances() const { return h_distances; }
    std::vector<vertex_t> get_path(vertex_t v) const;
};

// CUDA kernels (declared here, defined in .cu file)
__global__ void initialize_kernel(weight_t* distances, vertex_t* predecessors,
                                  bool* complete, vertex_t n, vertex_t source);

__global__ void relax_edges_kernel(const uint64_t* row_ptr, const vertex_t* col_idx,
                                   const weight_t* weights, weight_t* distances,
                                   vertex_t* predecessors, bool* complete,
                                   bool* updated, vertex_t n);

__global__ void bellman_ford_kernel(const uint64_t* row_ptr, const vertex_t* col_idx,
                                    const weight_t* weights, weight_t* distances,
                                    vertex_t* predecessors, const bool* active,
                                    bool* updated, vertex_t n, weight_t B);

} // namespace cuda
} // namespace sssp

#endif // CUDA_SSSP_CUH
