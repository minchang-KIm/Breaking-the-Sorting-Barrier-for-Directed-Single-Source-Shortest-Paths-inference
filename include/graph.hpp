#ifndef GRAPH_HPP
#define GRAPH_HPP

#include <vector>
#include <limits>
#include <cmath>
#include <algorithm>
#include <cstdint>

namespace sssp {

using vertex_t = uint32_t;
using weight_t = double;
constexpr weight_t INF = std::numeric_limits<weight_t>::infinity();

struct Edge {
    vertex_t to;
    weight_t weight;

    Edge(vertex_t t, weight_t w) : to(t), weight(w) {}
};

class Graph {
public:
    vertex_t n; // number of vertices
    uint64_t m; // number of edges
    std::vector<std::vector<Edge>> adj; // adjacency list

    Graph(vertex_t num_vertices) : n(num_vertices), m(0) {
        adj.resize(n);
    }

    void add_edge(vertex_t from, vertex_t to, weight_t weight) {
        adj[from].emplace_back(to, weight);
        m++;
    }

    // Get out-degree of a vertex
    size_t out_degree(vertex_t v) const {
        return adj[v].size();
    }

    // Convert to constant degree graph (Algorithm preprocessing)
    static Graph to_constant_degree(const Graph& g);

    // Parameters from the paper
    static uint32_t compute_k(vertex_t n) {
        return static_cast<uint32_t>(std::floor(std::pow(std::log(n), 1.0/3.0)));
    }

    static uint32_t compute_t(vertex_t n) {
        return static_cast<uint32_t>(std::floor(std::pow(std::log(n), 2.0/3.0)));
    }
};

// Path information
struct PathInfo {
    weight_t distance;
    vertex_t predecessor;
    bool is_complete;

    PathInfo() : distance(INF), predecessor(UINT32_MAX), is_complete(false) {}
};

} // namespace sssp

#endif // GRAPH_HPP
