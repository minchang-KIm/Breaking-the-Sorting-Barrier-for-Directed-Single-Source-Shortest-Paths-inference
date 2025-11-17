#include "graph.hpp"
#include "cuda_sssp.cuh"
#include <iostream>
#include <cassert>
#include <cmath>

using namespace sssp;

void test_simple_graph() {
    std::cout << "Testing simple graph (CUDA)...\n";

    Graph g(4);
    g.add_edge(0, 1, 1.0);
    g.add_edge(0, 2, 4.0);
    g.add_edge(1, 2, 2.0);
    g.add_edge(1, 3, 5.0);
    g.add_edge(2, 3, 1.0);

    cuda::CudaSSSP algo(g, 0);
    algo.compute();

    assert(std::fabs(algo.get_distance(0) - 0.0) < 1e-5);
    assert(std::fabs(algo.get_distance(1) - 1.0) < 1e-5);
    assert(std::fabs(algo.get_distance(2) - 3.0) < 1e-5);
    assert(std::fabs(algo.get_distance(3) - 4.0) < 1e-5);

    std::cout << "  PASSED\n";
}

void test_path_recovery() {
    std::cout << "Testing path recovery (CUDA)...\n";

    Graph g(4);
    g.add_edge(0, 1, 1.0);
    g.add_edge(1, 2, 1.0);
    g.add_edge(2, 3, 1.0);

    cuda::CudaSSSP algo(g, 0);
    algo.compute();

    auto path = algo.get_path(3);
    assert(path.size() == 4);
    assert(path[0] == 0);
    assert(path[1] == 1);
    assert(path[2] == 2);
    assert(path[3] == 3);

    std::cout << "  PASSED\n";
}

int main() {
    std::cout << "Running CUDA SSSP tests\n";
    std::cout << "=======================\n\n";

    test_simple_graph();
    test_path_recovery();

    std::cout << "\nAll tests passed!\n";
    return 0;
}
