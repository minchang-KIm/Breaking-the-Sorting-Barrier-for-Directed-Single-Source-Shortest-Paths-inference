#include <iostream>
#include <fstream>
#include <chrono>
#include <vector>
#include <string>
#include <iomanip>

#include "graph.hpp"
#include "sssp_algorithm.hpp"

#ifdef ENABLE_OPENMP
#include "parallel_sssp.hpp"
#endif

#ifdef ENABLE_CUDA
#include "cuda_sssp.cuh"
#endif

#ifdef ENABLE_MPI
#include <mpi.h>
#endif

using namespace sssp;

struct BenchmarkResult {
    std::string name;
    double time_ms;
    bool success;
};

Graph load_graph(const std::string& filename) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error: Cannot open file " << filename << std::endl;
        exit(1);
    }

    vertex_t n;
    uint64_t m;
    file >> n >> m;

    Graph g(n);

    for (uint64_t i = 0; i < m; i++) {
        vertex_t u, v;
        weight_t w;
        file >> u >> v >> w;
        g.add_edge(u, v, w);
    }

    return g;
}

BenchmarkResult benchmark_sequential(const Graph& g, vertex_t source) {
    BenchmarkResult result;
    result.name = "Sequential";

    auto start = std::chrono::high_resolution_clock::now();

    try {
        SSSPAlgorithm algo(g, source);
        algo.compute_shortest_paths();
        result.success = true;
    } catch (...) {
        result.success = false;
    }

    auto end = std::chrono::high_resolution_clock::now();
    result.time_ms = std::chrono::duration<double, std::milli>(end - start).count();

    return result;
}

#ifdef ENABLE_OPENMP
BenchmarkResult benchmark_openmp(const Graph& g, vertex_t source, int threads) {
    BenchmarkResult result;
    result.name = "OpenMP (" + std::to_string(threads) + " threads)";

    auto start = std::chrono::high_resolution_clock::now();

    try {
        parallel::SharedMemorySSSP algo(g, source, threads);
        algo.compute();
        result.success = true;
    } catch (...) {
        result.success = false;
    }

    auto end = std::chrono::high_resolution_clock::now();
    result.time_ms = std::chrono::duration<double, std::milli>(end - start).count();

    return result;
}
#endif

#ifdef ENABLE_CUDA
BenchmarkResult benchmark_cuda(const Graph& g, vertex_t source) {
    BenchmarkResult result;
    result.name = "CUDA GPU";

    auto start = std::chrono::high_resolution_clock::now();

    try {
        cuda::CudaSSSP algo(g, source);
        algo.compute();
        result.success = true;
    } catch (...) {
        result.success = false;
    }

    auto end = std::chrono::high_resolution_clock::now();
    result.time_ms = std::chrono::duration<double, std::milli>(end - start).count();

    return result;
}
#endif

void print_results(const std::vector<BenchmarkResult>& results) {
    std::cout << "\n" << std::string(60, '=') << "\n";
    std::cout << "Benchmark Results\n";
    std::cout << std::string(60, '=') << "\n\n";

    std::cout << std::left << std::setw(30) << "Implementation"
              << std::right << std::setw(15) << "Time (ms)"
              << std::setw(15) << "Status" << "\n";
    std::cout << std::string(60, '-') << "\n";

    for (const auto& r : results) {
        std::cout << std::left << std::setw(30) << r.name
                  << std::right << std::setw(15) << std::fixed
                  << std::setprecision(2) << r.time_ms
                  << std::setw(15) << (r.success ? "OK" : "FAILED") << "\n";
    }

    std::cout << std::string(60, '=') << "\n";

    // Compute speedups
    if (results.size() > 1 && results[0].success) {
        double baseline = results[0].time_ms;
        std::cout << "\nSpeedups (relative to " << results[0].name << "):\n";
        for (size_t i = 1; i < results.size(); i++) {
            if (results[i].success) {
                double speedup = baseline / results[i].time_ms;
                std::cout << "  " << results[i].name << ": "
                          << std::fixed << std::setprecision(2)
                          << speedup << "x\n";
            }
        }
    }
}

int main(int argc, char** argv) {
#ifdef ENABLE_MPI
    MPI_Init(&argc, &argv);
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    if (rank != 0) {
        MPI_Finalize();
        return 0;
    }
#endif

    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <graph_file> [source_vertex]\n";
        return 1;
    }

    std::string graph_file = argv[1];
    vertex_t source = (argc > 2) ? std::stoul(argv[2]) : 0;

    std::cout << "Loading graph from " << graph_file << "...\n";
    Graph g = load_graph(graph_file);

    std::cout << "Graph loaded: " << g.n << " vertices, " << g.m << " edges\n";
    std::cout << "Source vertex: " << source << "\n";
    std::cout << "k = " << Graph::compute_k(g.n) << ", t = "
              << Graph::compute_t(g.n) << "\n";

    std::vector<BenchmarkResult> results;

    // Sequential benchmark
    std::cout << "\nRunning sequential benchmark...\n";
    results.push_back(benchmark_sequential(g, source));

#ifdef ENABLE_OPENMP
    // OpenMP benchmarks with different thread counts
    std::vector<int> thread_counts = {1, 2, 4, 8};
    for (int threads : thread_counts) {
        std::cout << "Running OpenMP benchmark (" << threads << " threads)...\n";
        results.push_back(benchmark_openmp(g, source, threads));
    }
#endif

#ifdef ENABLE_CUDA
    // CUDA benchmark
    std::cout << "Running CUDA benchmark...\n";
    results.push_back(benchmark_cuda(g, source));
#endif

    print_results(results);

#ifdef ENABLE_MPI
    MPI_Finalize();
#endif

    return 0;
}
