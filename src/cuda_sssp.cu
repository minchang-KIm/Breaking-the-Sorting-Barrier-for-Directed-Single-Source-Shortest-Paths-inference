#include "cuda_sssp.cuh"
#include <algorithm>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <limits>
#include <iostream>

#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            std::cerr << "CUDA error in " << __FILE__ << ":" << __LINE__ \
                      << " - " << cudaGetErrorString(err) << std::endl; \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

namespace sssp {
namespace cuda {

// CUDA kernel implementations

__global__ void initialize_kernel(weight_t* distances, vertex_t* predecessors,
                                  bool* complete, vertex_t n, vertex_t source) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid < n) {
        if (tid == source) {
            distances[tid] = 0.0;
            complete[tid] = true;
        } else {
            distances[tid] = INFINITY;
            complete[tid] = false;
        }
        predecessors[tid] = UINT32_MAX;
    }
}

__global__ void relax_edges_kernel(const uint64_t* row_ptr, const vertex_t* col_idx,
                                   const weight_t* weights, weight_t* distances,
                                   vertex_t* predecessors, bool* complete,
                                   bool* updated, vertex_t n) {
    int u = blockIdx.x * blockDim.x + threadIdx.x;

    if (u < n && complete[u]) {
        uint64_t start = row_ptr[u];
        uint64_t end = row_ptr[u + 1];

        for (uint64_t i = start; i < end; i++) {
            vertex_t v = col_idx[i];
            weight_t new_dist = distances[u] + weights[i];

            // Atomic min operation
            weight_t old_dist = atomicMin(&distances[v], new_dist);

            if (new_dist < old_dist) {
                predecessors[v] = u;
                *updated = true;
            }
        }
    }
}

__global__ void bellman_ford_kernel(const uint64_t* row_ptr, const vertex_t* col_idx,
                                    const weight_t* weights, weight_t* distances,
                                    vertex_t* predecessors, const bool* active,
                                    bool* updated, vertex_t n, weight_t B) {
    int u = blockIdx.x * blockDim.x + threadIdx.x;

    if (u < n && active[u]) {
        uint64_t start = row_ptr[u];
        uint64_t end = row_ptr[u + 1];

        for (uint64_t i = start; i < end; i++) {
            vertex_t v = col_idx[i];
            weight_t new_dist = distances[u] + weights[i];

            if (new_dist < B) {
                weight_t old_dist = atomicMin(&distances[v], new_dist);

                if (new_dist < old_dist) {
                    predecessors[v] = u;
                    *updated = true;
                }
            }
        }
    }
}

__global__ void update_complete_kernel(weight_t* distances, bool* complete,
                                       bool* updated, vertex_t n) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid < n && !complete[tid] && distances[tid] < INFINITY) {
        complete[tid] = true;
        *updated = true;
    }
}

// CSRGraph implementation

CSRGraph::CSRGraph(const Graph& g) : n(g.n), m(g.m) {
    row_ptr.resize(n + 1, 0);
    col_idx.reserve(m);
    weights.reserve(m);

    // Build CSR format
    for (vertex_t u = 0; u < n; u++) {
        row_ptr[u + 1] = row_ptr[u] + g.adj[u].size();
        for (const Edge& e : g.adj[u]) {
            col_idx.push_back(e.to);
            weights.push_back(e.weight);
        }
    }

    d_row_ptr = nullptr;
    d_col_idx = nullptr;
    d_weights = nullptr;
}

CSRGraph::~CSRGraph() {
    free_device();
}

void CSRGraph::copy_to_device() {
    CUDA_CHECK(cudaMalloc(&d_row_ptr, (n + 1) * sizeof(uint64_t)));
    CUDA_CHECK(cudaMalloc(&d_col_idx, m * sizeof(vertex_t)));
    CUDA_CHECK(cudaMalloc(&d_weights, m * sizeof(weight_t)));

    CUDA_CHECK(cudaMemcpy(d_row_ptr, row_ptr.data(), (n + 1) * sizeof(uint64_t),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_col_idx, col_idx.data(), m * sizeof(vertex_t),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_weights, weights.data(), m * sizeof(weight_t),
                          cudaMemcpyHostToDevice));
}

void CSRGraph::free_device() {
    if (d_row_ptr) cudaFree(d_row_ptr);
    if (d_col_idx) cudaFree(d_col_idx);
    if (d_weights) cudaFree(d_weights);

    d_row_ptr = nullptr;
    d_col_idx = nullptr;
    d_weights = nullptr;
}

// CudaSSSP implementation

CudaSSSP::CudaSSSP(const Graph& g, vertex_t src)
    : graph(g), source(src), csr(g) {

    h_distances.resize(g.n);
    h_predecessors.resize(g.n);

    allocate_device_memory();
}

CudaSSSP::~CudaSSSP() {
    free_device_memory();
}

void CudaSSSP::allocate_device_memory() {
    csr.copy_to_device();

    CUDA_CHECK(cudaMalloc(&d_distances, graph.n * sizeof(weight_t)));
    CUDA_CHECK(cudaMalloc(&d_predecessors, graph.n * sizeof(vertex_t)));
    CUDA_CHECK(cudaMalloc(&d_complete, graph.n * sizeof(bool)));
    CUDA_CHECK(cudaMalloc(&d_updated, sizeof(bool)));

    // Initialize
    int block_size = 256;
    int num_blocks = (graph.n + block_size - 1) / block_size;

    initialize_kernel<<<num_blocks, block_size>>>(
        d_distances, d_predecessors, d_complete, graph.n, source);

    CUDA_CHECK(cudaDeviceSynchronize());
}

void CudaSSSP::free_device_memory() {
    if (d_distances) cudaFree(d_distances);
    if (d_predecessors) cudaFree(d_predecessors);
    if (d_complete) cudaFree(d_complete);
    if (d_updated) cudaFree(d_updated);

    d_distances = nullptr;
    d_predecessors = nullptr;
    d_complete = nullptr;
    d_updated = nullptr;
}

void CudaSSSP::compute() {
    int block_size = 256;
    int num_blocks = (graph.n + block_size - 1) / block_size;

    bool h_updated = true;
    uint32_t max_iterations = graph.n;

    for (uint32_t iter = 0; iter < max_iterations && h_updated; iter++) {
        h_updated = false;
        CUDA_CHECK(cudaMemcpy(d_updated, &h_updated, sizeof(bool),
                              cudaMemcpyHostToDevice));

        // Relax edges
        relax_edges_kernel<<<num_blocks, block_size>>>(
            csr.d_row_ptr, csr.d_col_idx, csr.d_weights,
            d_distances, d_predecessors, d_complete, d_updated, graph.n);

        CUDA_CHECK(cudaDeviceSynchronize());

        // Update complete flags
        update_complete_kernel<<<num_blocks, block_size>>>(
            d_distances, d_complete, d_updated, graph.n);

        CUDA_CHECK(cudaDeviceSynchronize());

        // Check if any updates occurred
        CUDA_CHECK(cudaMemcpy(&h_updated, d_updated, sizeof(bool),
                              cudaMemcpyDeviceToHost));
    }

    copy_results_to_host();
}

void CudaSSSP::copy_results_to_host() {
    CUDA_CHECK(cudaMemcpy(h_distances.data(), d_distances,
                          graph.n * sizeof(weight_t), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_predecessors.data(), d_predecessors,
                          graph.n * sizeof(vertex_t), cudaMemcpyDeviceToHost));
}

std::vector<vertex_t> CudaSSSP::get_path(vertex_t v) const {
    std::vector<vertex_t> path;
    vertex_t current = v;

    while (current != UINT32_MAX) {
        path.push_back(current);
        current = h_predecessors[current];
    }

    std::reverse(path.begin(), path.end());
    return path;
}

} // namespace cuda
} // namespace sssp
