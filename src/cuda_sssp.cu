/**
 * @file cuda_sssp.cu
 * @brief GPU-accelerated SSSP implementation using CUDA
 *
 * GPU 가속 SSSP 구현 (CUDA 사용)
 *
 * This file implements SSSP on NVIDIA GPUs using CUDA
 * Uses CSR (Compressed Sparse Row) format for efficient GPU memory access
 *
 * BUG FIXED: Implemented proper atomicMin for double using atomicCAS
 * Standard CUDA doesn't provide atomicMin for double, only for int/uint
 */

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

/**
 * @brief Atomic minimum for double precision floating point
 *
 * double 정밀도 부동 소수점을 위한 원자적 최소값
 *
 * CUDA doesn't provide atomicMin for double, so we implement it using atomicCAS
 * Uses compare-and-swap loop to ensure atomicity
 *
 * FIXED: Proper implementation that works on all CUDA architectures
 *
 * @param address Pointer to the value to update
 * @param val Value to compare against
 * @return Old value before update
 */
__device__ __forceinline__ double atomicMinDouble(double* address, double val) {
    unsigned long long int* address_as_ull = (unsigned long long int*)address;
    unsigned long long int old = *address_as_ull;
    unsigned long long int assumed;

    do {
        assumed = old;
        // Convert to double, compare, convert back / double로 변환, 비교, 다시 변환
        double old_double = __longlong_as_double(assumed);
        if (old_double <= val) break; // Current value is already smaller / 현재 값이 이미 작음

        old = atomicCAS(address_as_ull, assumed, __double_as_longlong(val));
    } while (assumed != old);

    return __longlong_as_double(old);
}

/******************************************************************************
 * CUDA Kernels
 ******************************************************************************/

/**
 * @brief Initialize distances, predecessors, and completion status
 *
 * 거리, 선행자 및 완료 상태 초기화
 *
 * Each thread initializes one vertex
 * Source vertex gets distance 0, others get infinity
 */
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

/**
 * @brief Relax edges from completed vertices
 *
 * 완료된 정점에서 간선 완화
 *
 * FIXED: Uses custom atomicMinDouble instead of non-existent atomicMin(double)
 *
 * Each thread processes one source vertex and relaxes all its outgoing edges
 * Uses atomicMinDouble for thread-safe distance updates
 */
__global__ void relax_edges_kernel(const uint64_t* row_ptr, const vertex_t* col_idx,
                                   const weight_t* weights, weight_t* distances,
                                   vertex_t* predecessors, bool* complete,
                                   bool* updated, vertex_t n) {
    int u = blockIdx.x * blockDim.x + threadIdx.x;

    if (u < n && complete[u]) {
        uint64_t start = row_ptr[u];
        uint64_t end = row_ptr[u + 1];

        weight_t dist_u = distances[u]; // Cache for efficiency / 효율성을 위해 캐시

        for (uint64_t i = start; i < end; i++) {
            vertex_t v = col_idx[i];
            weight_t new_dist = dist_u + weights[i];

            // FIXED: Use custom atomicMinDouble instead of atomicMin
            // 수정됨: atomicMin 대신 사용자 정의 atomicMinDouble 사용
            weight_t old_dist = atomicMinDouble(&distances[v], new_dist);

            if (new_dist < old_dist) {
                // Note: predecessor update is not atomic, but acceptable for SSSP
                // 참고: 선행자 업데이트는 원자적이지 않지만 SSSP에서는 허용 가능
                predecessors[v] = u;
                *updated = true;
            }
        }
    }
}

/**
 * @brief Bellman-Ford style relaxation with bound B
 *
 * 경계 B를 사용한 벨만-포드 스타일 완화
 *
 * FIXED: Uses custom atomicMinDouble
 *
 * Only relaxes edges where new distance < B (bounded relaxation)
 */
__global__ void bellman_ford_kernel(const uint64_t* row_ptr, const vertex_t* col_idx,
                                    const weight_t* weights, weight_t* distances,
                                    vertex_t* predecessors, const bool* active,
                                    bool* updated, vertex_t n, weight_t B) {
    int u = blockIdx.x * blockDim.x + threadIdx.x;

    if (u < n && active[u]) {
        uint64_t start = row_ptr[u];
        uint64_t end = row_ptr[u + 1];

        weight_t dist_u = distances[u];

        for (uint64_t i = start; i < end; i++) {
            vertex_t v = col_idx[i];
            weight_t new_dist = dist_u + weights[i];

            if (new_dist < B) {
                // FIXED: Use custom atomicMinDouble
                // 수정됨: 사용자 정의 atomicMinDouble 사용
                weight_t old_dist = atomicMinDouble(&distances[v], new_dist);

                if (new_dist < old_dist) {
                    predecessors[v] = u;
                    *updated = true;
                }
            }
        }
    }
}

/**
 * @brief Update completion status for vertices
 *
 * 정점에 대한 완료 상태 업데이트
 *
 * Marks vertices as complete if they have finite distance
 */
__global__ void update_complete_kernel(weight_t* distances, bool* complete,
                                       bool* updated, vertex_t n) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid < n && !complete[tid] && distances[tid] < INFINITY) {
        complete[tid] = true;
        *updated = true;
    }
}

/******************************************************************************
 * CSRGraph implementation
 ******************************************************************************/

/**
 * @brief Convert adjacency list to CSR format
 *
 * 인접 리스트를 CSR 형식으로 변환
 *
 * CSR (Compressed Sparse Row) format is optimal for GPU memory access
 * Provides coalesced memory access patterns for better GPU performance
 *
 * Structure:
 * - row_ptr[i] = starting index of edges for vertex i
 * - col_idx[j] = destination vertex of edge j
 * - weights[j] = weight of edge j
 */
CSRGraph::CSRGraph(const Graph& g) : n(g.n), m(g.m) {
    row_ptr.resize(n + 1, 0);
    col_idx.reserve(m);
    weights.reserve(m);

    // Build CSR format / CSR 형식 구축
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

/**
 * @brief Copy CSR graph to GPU device memory
 *
 * CSR 그래프를 GPU 장치 메모리에 복사
 *
 * Allocates GPU memory and copies graph data
 */
void CSRGraph::copy_to_device() {
    // Allocate device memory / 장치 메모리 할당
    CUDA_CHECK(cudaMalloc(&d_row_ptr, (n + 1) * sizeof(uint64_t)));
    CUDA_CHECK(cudaMalloc(&d_col_idx, m * sizeof(vertex_t)));
    CUDA_CHECK(cudaMalloc(&d_weights, m * sizeof(weight_t)));

    // Copy data to device / 장치로 데이터 복사
    CUDA_CHECK(cudaMemcpy(d_row_ptr, row_ptr.data(), (n + 1) * sizeof(uint64_t),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_col_idx, col_idx.data(), m * sizeof(vertex_t),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_weights, weights.data(), m * sizeof(weight_t),
                          cudaMemcpyHostToDevice));
}

/**
 * @brief Free device memory
 *
 * 장치 메모리 해제
 */
void CSRGraph::free_device() {
    if (d_row_ptr) CUDA_CHECK(cudaFree(d_row_ptr));
    if (d_col_idx) CUDA_CHECK(cudaFree(d_col_idx));
    if (d_weights) CUDA_CHECK(cudaFree(d_weights));

    d_row_ptr = nullptr;
    d_col_idx = nullptr;
    d_weights = nullptr;
}

/******************************************************************************
 * CudaSSSP implementation
 ******************************************************************************/

CudaSSSP::CudaSSSP(const Graph& g, vertex_t src)
    : graph(g), source(src), csr(g) {

    h_distances.resize(g.n);
    h_predecessors.resize(g.n);

    allocate_device_memory();
}

CudaSSSP::~CudaSSSP() {
    free_device_memory();
}

/**
 * @brief Allocate and initialize GPU memory
 *
 * GPU 메모리 할당 및 초기화
 *
 * FIXED: Added error checking for kernel launches
 */
void CudaSSSP::allocate_device_memory() {
    csr.copy_to_device();

    // Allocate device arrays / 장치 배열 할당
    CUDA_CHECK(cudaMalloc(&d_distances, graph.n * sizeof(weight_t)));
    CUDA_CHECK(cudaMalloc(&d_predecessors, graph.n * sizeof(vertex_t)));
    CUDA_CHECK(cudaMalloc(&d_complete, graph.n * sizeof(bool)));
    CUDA_CHECK(cudaMalloc(&d_updated, sizeof(bool)));

    // Initialize / 초기화
    int block_size = 256;
    int num_blocks = (graph.n + block_size - 1) / block_size;

    initialize_kernel<<<num_blocks, block_size>>>(
        d_distances, d_predecessors, d_complete, graph.n, source);

    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaGetLastError()); // FIXED: Check for kernel launch errors
}

/**
 * @brief Free GPU memory
 *
 * GPU 메모리 해제
 */
void CudaSSSP::free_device_memory() {
    if (d_distances) CUDA_CHECK(cudaFree(d_distances));
    if (d_predecessors) CUDA_CHECK(cudaFree(d_predecessors));
    if (d_complete) CUDA_CHECK(cudaFree(d_complete));
    if (d_updated) CUDA_CHECK(cudaFree(d_updated));

    d_distances = nullptr;
    d_predecessors = nullptr;
    d_complete = nullptr;
    d_updated = nullptr;
}

/**
 * @brief Main SSSP computation on GPU
 *
 * GPU에서 주요 SSSP 계산
 *
 * Iteratively relaxes edges until convergence
 * Uses Bellman-Ford style with early termination
 *
 * FIXED: Added comprehensive error checking
 */
void CudaSSSP::compute() {
    int block_size = 256;
    int num_blocks = (graph.n + block_size - 1) / block_size;

    bool h_updated = true;
    uint32_t max_iterations = graph.n;

    for (uint32_t iter = 0; iter < max_iterations && h_updated; iter++) {
        h_updated = false;
        CUDA_CHECK(cudaMemcpy(d_updated, &h_updated, sizeof(bool),
                              cudaMemcpyHostToDevice));

        // Relax edges / 간선 완화
        relax_edges_kernel<<<num_blocks, block_size>>>(
            csr.d_row_ptr, csr.d_col_idx, csr.d_weights,
            d_distances, d_predecessors, d_complete, d_updated, graph.n);

        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaGetLastError()); // FIXED: Check for kernel errors

        // Update complete flags / 완료 플래그 업데이트
        update_complete_kernel<<<num_blocks, block_size>>>(
            d_distances, d_complete, d_updated, graph.n);

        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaGetLastError()); // FIXED: Check for kernel errors

        // Check if any updates occurred / 업데이트 발생 여부 확인
        CUDA_CHECK(cudaMemcpy(&h_updated, d_updated, sizeof(bool),
                              cudaMemcpyDeviceToHost));
    }

    copy_results_to_host();
}

/**
 * @brief Copy results from GPU to CPU
 *
 * GPU에서 CPU로 결과 복사
 */
void CudaSSSP::copy_results_to_host() {
    CUDA_CHECK(cudaMemcpy(h_distances.data(), d_distances,
                          graph.n * sizeof(weight_t), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_predecessors.data(), d_predecessors,
                          graph.n * sizeof(vertex_t), cudaMemcpyDeviceToHost));
}

/**
 * @brief Reconstruct shortest path to vertex v
 *
 * 정점 v까지의 최단 경로 재구성
 */
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
