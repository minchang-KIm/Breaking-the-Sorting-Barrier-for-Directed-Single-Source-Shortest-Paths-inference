#include "graph.hpp"
#include "sssp_algorithm.hpp"
#include <iostream>
#include <cassert>
#include <cmath>

using namespace sssp;

void test_simple_graph() {
    std::cout << "Testing simple graph...\n";

    Graph g(4);
    g.add_edge(0, 1, 1.0);
    g.add_edge(0, 2, 4.0);
    g.add_edge(1, 2, 2.0);
    g.add_edge(1, 3, 5.0);
    g.add_edge(2, 3, 1.0);

    SSSPAlgorithm algo(g, 0);
    algo.compute_shortest_paths();

    assert(std::fabs(algo.get_distance(0) - 0.0) < 1e-9);
    assert(std::fabs(algo.get_distance(1) - 1.0) < 1e-9);
    assert(std::fabs(algo.get_distance(2) - 3.0) < 1e-9);
    assert(std::fabs(algo.get_distance(3) - 4.0) < 1e-9);

    std::cout << "  PASSED\n";
}

void test_disconnected_graph() {
    std::cout << "Testing disconnected graph...\n";

    Graph g(5);
    g.add_edge(0, 1, 1.0);
    g.add_edge(1, 2, 2.0);
    g.add_edge(3, 4, 1.0);

    SSSPAlgorithm algo(g, 0);
    algo.compute_shortest_paths();

    assert(std::fabs(algo.get_distance(0) - 0.0) < 1e-9);
    assert(std::fabs(algo.get_distance(1) - 1.0) < 1e-9);
    assert(std::fabs(algo.get_distance(2) - 3.0) < 1e-9);
    assert(algo.get_distance(3) == INF);
    assert(algo.get_distance(4) == INF);

    std::cout << "  PASSED\n";
}

void test_single_vertex() {
    std::cout << "Testing single vertex...\n";

    Graph g(1);
    SSSPAlgorithm algo(g, 0);
    algo.compute_shortest_paths();

    assert(std::fabs(algo.get_distance(0) - 0.0) < 1e-9);

    std::cout << "  PASSED\n";
}

void test_path_recovery() {
    std::cout << "Testing path recovery...\n";

    Graph g(4);
    g.add_edge(0, 1, 1.0);
    g.add_edge(1, 2, 1.0);
    g.add_edge(2, 3, 1.0);

    SSSPAlgorithm algo(g, 0);
    algo.compute_shortest_paths();

    auto path = algo.get_path(3);
    assert(path.size() == 4);
    assert(path[0] == 0);
    assert(path[1] == 1);
    assert(path[2] == 2);
    assert(path[3] == 3);

    std::cout << "  PASSED\n";
}

int main() {
    std::cout << "Running sequential SSSP tests\n";
    std::cout << "==============================\n\n";

    test_simple_graph();
    test_disconnected_graph();
    test_single_vertex();
    test_path_recovery();

    std::cout << "\nAll tests passed!\n";
    return 0;
}
