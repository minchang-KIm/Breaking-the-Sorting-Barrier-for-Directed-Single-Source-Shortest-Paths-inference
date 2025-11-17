#include "graph.hpp"
#include <queue>

namespace sssp {

// Convert to constant degree graph as described in the paper
Graph Graph::to_constant_degree(const Graph& g) {
    // First pass: compute in-degrees for each vertex
    std::vector<size_t> in_degree(g.n, 0);
    for (vertex_t u = 0; u < g.n; u++) {
        for (const auto& e : g.adj[u]) {
            in_degree[e.to]++;
        }
    }

    // Second pass: count total vertices needed (cycle size = in-degree + out-degree)
    vertex_t total_vertices = 0;
    std::vector<vertex_t> vertex_offsets(g.n + 1);

    for (vertex_t v = 0; v < g.n; v++) {
        size_t out_deg = g.adj[v].size();
        size_t total_deg = in_degree[v] + out_deg;
        vertex_t needed = std::max(total_deg, (size_t)2); // At least 2 for cycle
        vertex_offsets[v] = total_vertices;
        total_vertices += needed;
    }
    vertex_offsets[g.n] = total_vertices;

    Graph result(total_vertices);

    // Build cycles for each original vertex
    for (vertex_t v = 0; v < g.n; v++) {
        vertex_t start = vertex_offsets[v];
        vertex_t count = vertex_offsets[v + 1] - start;

        // Create cycle
        for (vertex_t i = 0; i < count; i++) {
            vertex_t current = start + i;
            vertex_t next = start + ((i + 1) % count);
            result.add_edge(current, next, 0.0);
        }

        // Add outgoing edges
        size_t edge_idx = 0;
        for (const auto& e : g.adj[v]) {
            vertex_t from = start + edge_idx;
            vertex_t to_orig = e.to;
            vertex_t to = vertex_offsets[to_orig]; // First vertex of target's cycle
            result.add_edge(from, to, e.weight);
            edge_idx++;
        }
    }

    return result;
}

} // namespace sssp
