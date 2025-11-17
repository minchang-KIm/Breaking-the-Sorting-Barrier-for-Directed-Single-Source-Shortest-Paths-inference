#ifndef SSSP_ALGORITHM_HPP
#define SSSP_ALGORITHM_HPP

#include "graph.hpp"
#include "partial_sort_ds.hpp"
#include <vector>
#include <set>

namespace sssp {

class SSSPAlgorithm {
private:
    const Graph& graph;
    vertex_t source;

    // Global distance and predecessor arrays
    std::vector<weight_t> db; // distance bounds
    std::vector<vertex_t> pred; // predecessors
    std::vector<bool> complete; // completion status

    // Algorithm parameters
    uint32_t k; // floor(log^(1/3) n)
    uint32_t t; // floor(log^(2/3) n)

    // Helper functions
    void relax_edge(vertex_t u, vertex_t v, weight_t w);

    // FindPivots (Algorithm 1)
    struct PivotsResult {
        std::set<vertex_t> pivots;
        std::set<vertex_t> W;
    };
    PivotsResult find_pivots(weight_t B, const std::set<vertex_t>& S);

    // BaseCase (Algorithm 2)
    struct BMSSPResult {
        weight_t B_prime;
        std::set<vertex_t> U;
    };
    BMSSPResult base_case(weight_t B, const std::set<vertex_t>& S);

    // BMSSP (Algorithm 3)
    BMSSPResult bmssp(uint32_t level, weight_t B, const std::set<vertex_t>& S);

public:
    SSSPAlgorithm(const Graph& g, vertex_t src)
        : graph(g), source(src),
          db(g.n, INF), pred(g.n, UINT32_MAX), complete(g.n, false) {
        k = Graph::compute_k(g.n);
        t = Graph::compute_t(g.n);
        db[source] = 0.0;
        complete[source] = true;
    }

    // Main algorithm
    void compute_shortest_paths();

    // Get results
    weight_t get_distance(vertex_t v) const { return db[v]; }
    vertex_t get_predecessor(vertex_t v) const { return pred[v]; }

    // Get shortest path to vertex v
    std::vector<vertex_t> get_path(vertex_t v) const;
};

} // namespace sssp

#endif // SSSP_ALGORITHM_HPP
