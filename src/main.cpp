#include <iostream>
#include <fstream>
#include <chrono>
#include <string>
#include <cstring>

#include "graph.hpp"
#include "sssp_algorithm.hpp"

#ifdef ENABLE_MPI
#include "parallel_sssp.hpp"
#include <mpi.h>
#endif

#ifdef ENABLE_CUDA
#include "cuda_sssp.cuh"
#endif

using namespace sssp;

enum class Mode {
    SEQUENTIAL,
    PARALLEL_MPI,
    PARALLEL_OPENMP,
    CUDA
};

void print_usage(const char* program) {
    std::cout << "Usage: " << program << " [OPTIONS]\n"
              << "Options:\n"
              << "  -i, --input FILE       Input graph file\n"
              << "  -s, --source VERTEX    Source vertex (default: 0)\n"
              << "  -m, --mode MODE        Execution mode:\n"
              << "                           seq      - Sequential\n"
              << "                           mpi      - MPI distributed\n"
              << "                           openmp   - OpenMP parallel\n"
              << "                           cuda     - CUDA GPU\n"
              << "  -t, --threads NUM      Number of OpenMP threads (default: max)\n"
              << "  -o, --output FILE      Output file for distances\n"
              << "  -v, --verbose          Verbose output\n"
              << "  -h, --help             Show this help message\n"
              << "\nGraph file format:\n"
              << "  First line: n m (vertices, edges)\n"
              << "  Following m lines: u v w (edge from u to v with weight w)\n";
}

Graph load_graph(const std::string& filename, bool verbose) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error: Cannot open file " << filename << std::endl;
        exit(1);
    }

    vertex_t n;
    uint64_t m;
    file >> n >> m;

    if (verbose) {
        std::cout << "Loading graph: " << n << " vertices, " << m << " edges\n";
    }

    Graph g(n);

    for (uint64_t i = 0; i < m; i++) {
        vertex_t u, v;
        weight_t w;
        file >> u >> v >> w;
        g.add_edge(u, v, w);
    }

    file.close();

    if (verbose) {
        std::cout << "Graph loaded successfully\n";
    }

    return g;
}

void save_results(const std::string& filename, vertex_t n,
                  const std::vector<weight_t>& distances, bool verbose) {
    std::ofstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error: Cannot write to file " << filename << std::endl;
        return;
    }

    for (vertex_t v = 0; v < n; v++) {
        file << v << " " << distances[v] << "\n";
    }

    file.close();

    if (verbose) {
        std::cout << "Results saved to " << filename << "\n";
    }
}

int main(int argc, char** argv) {
    // Parse command line arguments
    std::string input_file;
    std::string output_file;
    vertex_t source = 0;
    Mode mode = Mode::SEQUENTIAL;
    int num_threads = 0;
    bool verbose = false;

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "-i") == 0 || strcmp(argv[i], "--input") == 0) {
            if (i + 1 < argc) input_file = argv[++i];
        } else if (strcmp(argv[i], "-s") == 0 || strcmp(argv[i], "--source") == 0) {
            if (i + 1 < argc) source = std::stoul(argv[++i]);
        } else if (strcmp(argv[i], "-m") == 0 || strcmp(argv[i], "--mode") == 0) {
            if (i + 1 < argc) {
                std::string m = argv[++i];
                if (m == "seq") mode = Mode::SEQUENTIAL;
                else if (m == "mpi") mode = Mode::PARALLEL_MPI;
                else if (m == "openmp") mode = Mode::PARALLEL_OPENMP;
                else if (m == "cuda") mode = Mode::CUDA;
                else {
                    std::cerr << "Unknown mode: " << m << std::endl;
                    return 1;
                }
            }
        } else if (strcmp(argv[i], "-t") == 0 || strcmp(argv[i], "--threads") == 0) {
            if (i + 1 < argc) num_threads = std::stoi(argv[++i]);
        } else if (strcmp(argv[i], "-o") == 0 || strcmp(argv[i], "--output") == 0) {
            if (i + 1 < argc) output_file = argv[++i];
        } else if (strcmp(argv[i], "-v") == 0 || strcmp(argv[i], "--verbose") == 0) {
            verbose = true;
        } else if (strcmp(argv[i], "-h") == 0 || strcmp(argv[i], "--help") == 0) {
            print_usage(argv[0]);
            return 0;
        }
    }

    if (input_file.empty()) {
        std::cerr << "Error: Input file required\n";
        print_usage(argv[0]);
        return 1;
    }

#ifdef ENABLE_MPI
    if (mode == Mode::PARALLEL_MPI) {
        MPI_Init(&argc, &argv);
    }
#endif

    // Load graph
    Graph g = load_graph(input_file, verbose);

    if (source >= g.n) {
        std::cerr << "Error: Source vertex " << source << " is out of range\n";
        return 1;
    }

    // Run algorithm
    auto start_time = std::chrono::high_resolution_clock::now();

    std::vector<weight_t> distances;

    switch (mode) {
        case Mode::SEQUENTIAL: {
            if (verbose) std::cout << "Running sequential SSSP algorithm\n";
            SSSPAlgorithm algo(g, source);
            algo.compute_shortest_paths();

            distances.resize(g.n);
            for (vertex_t v = 0; v < g.n; v++) {
                distances[v] = algo.get_distance(v);
            }
            break;
        }

#ifdef ENABLE_MPI
        case Mode::PARALLEL_MPI: {
            int rank;
            MPI_Comm_rank(MPI_COMM_WORLD, &rank);
            if (rank == 0 && verbose) {
                std::cout << "Running MPI distributed SSSP algorithm\n";
            }

            parallel::DistributedSSSP algo(g, source);
            algo.compute();
            algo.gather_results();

            if (rank == 0) {
                distances = algo.get_distances();
            }
            break;
        }
#endif

#ifdef ENABLE_OPENMP
        case Mode::PARALLEL_OPENMP: {
            if (verbose) std::cout << "Running OpenMP parallel SSSP algorithm\n";
            parallel::SharedMemorySSSP algo(g, source, num_threads);
            algo.compute();

            distances.resize(g.n);
            for (vertex_t v = 0; v < g.n; v++) {
                distances[v] = algo.get_distance(v);
            }
            break;
        }
#endif

#ifdef ENABLE_CUDA
        case Mode::CUDA: {
            if (verbose) std::cout << "Running CUDA GPU SSSP algorithm\n";
            cuda::CudaSSSP algo(g, source);
            algo.compute();
            distances = algo.get_distances();
            break;
        }
#endif

        default:
            std::cerr << "Error: Selected mode not compiled in this build\n";
            return 1;
    }

    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(
        end_time - start_time).count();

#ifdef ENABLE_MPI
    int rank = 0;
    if (mode == Mode::PARALLEL_MPI) {
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    }

    if (rank == 0) {
#endif
        if (verbose) {
            std::cout << "Computation time: " << duration << " ms\n";
        }

        // Save results if output file specified
        if (!output_file.empty()) {
            save_results(output_file, g.n, distances, verbose);
        }

        // Print sample results
        if (verbose) {
            std::cout << "\nSample distances from source " << source << ":\n";
            for (vertex_t v = 0; v < std::min(g.n, (vertex_t)10); v++) {
                std::cout << "  Vertex " << v << ": ";
                if (distances[v] < INF) {
                    std::cout << distances[v];
                } else {
                    std::cout << "unreachable";
                }
                std::cout << "\n";
            }
        } else {
            std::cout << duration << "\n";
        }
#ifdef ENABLE_MPI
    }

    if (mode == Mode::PARALLEL_MPI) {
        MPI_Finalize();
    }
#endif

    return 0;
}
