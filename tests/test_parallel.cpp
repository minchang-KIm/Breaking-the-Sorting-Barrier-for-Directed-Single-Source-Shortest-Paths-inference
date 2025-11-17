#include "graph.hpp"
#include "parallel_sssp.hpp"
#include <iostream>
#include <cassert>
#include <cmath>
#include <mpi.h>

using namespace sssp;

void test_simple_graph() {
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    if (rank == 0) {
        std::cout << "Testing simple graph (MPI)...\n";
    }

    Graph g(4);
    g.add_edge(0, 1, 1.0);
    g.add_edge(0, 2, 4.0);
    g.add_edge(1, 2, 2.0);
    g.add_edge(1, 3, 5.0);
    g.add_edge(2, 3, 1.0);

    parallel::DistributedSSSP algo(g, 0);
    algo.compute();
    algo.gather_results();

    if (rank == 0) {
        const auto& distances = algo.get_distances();
        assert(std::fabs(distances[0] - 0.0) < 1e-9);
        assert(std::fabs(distances[1] - 1.0) < 1e-9);
        assert(std::fabs(distances[2] - 3.0) < 1e-9);
        assert(std::fabs(distances[3] - 4.0) < 1e-9);
        std::cout << "  PASSED\n";
    }
}

void test_openmp() {
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    if (rank == 0) {
        std::cout << "Testing OpenMP implementation...\n";

        Graph g(4);
        g.add_edge(0, 1, 1.0);
        g.add_edge(0, 2, 4.0);
        g.add_edge(1, 2, 2.0);
        g.add_edge(1, 3, 5.0);
        g.add_edge(2, 3, 1.0);

        parallel::SharedMemorySSSP algo(g, 0, 4);
        algo.compute();

        assert(std::fabs(algo.get_distance(0) - 0.0) < 1e-9);
        assert(std::fabs(algo.get_distance(1) - 1.0) < 1e-9);
        assert(std::fabs(algo.get_distance(2) - 3.0) < 1e-9);
        assert(std::fabs(algo.get_distance(3) - 4.0) < 1e-9);

        std::cout << "  PASSED\n";
    }
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    if (rank == 0) {
        std::cout << "Running parallel SSSP tests\n";
        std::cout << "============================\n\n";
    }

    test_simple_graph();
    test_openmp();

    if (rank == 0) {
        std::cout << "\nAll tests passed!\n";
    }

    MPI_Finalize();
    return 0;
}
