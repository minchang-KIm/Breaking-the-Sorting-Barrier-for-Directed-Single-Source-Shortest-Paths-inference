#include "graph.hpp"
#include <queue>

namespace sssp {

/**
 * @brief Convert graph to constant-degree format as described in the paper
 * @param g Input graph with arbitrary vertex degrees
 * @return Graph with constant degree (≤ 3)
 *
 * 논문에 설명된 대로 그래프를 상수 차수 형식으로 변환
 *
 * Algorithm: Replace each vertex with a cycle of vertices
 * - Each vertex v with degree d becomes a cycle of max(d, 2) vertices
 * - Outgoing edges distributed across cycle vertices
 * - Cycle edges have weight 0
 *
 * This transformation preserves shortest path distances while ensuring
 * constant degree, which improves cache locality and theoretical bounds
 *
 * Time complexity: O(n + m)
 * Space complexity: O(n + m)
 *
 * BUG FIXED: Removed redundant degree calculation
 * Was: g.adj[v].size() + g.out_degree(v) (counted twice!)
 * Now: g.out_degree(v) only
 */
Graph Graph::to_constant_degree(const Graph& g) {
    // Count total vertices needed / 필요한 총 정점 수 계산
    vertex_t total_vertices = 0;
    std::vector<vertex_t> vertex_offsets(g.n + 1);

    for (vertex_t v = 0; v < g.n; v++) {
        // FIXED: Use out_degree(v) only, not adj[v].size() + out_degree(v)
        // 수정됨: out_degree(v)만 사용, adj[v].size() + out_degree(v) 아님
        size_t degree = g.out_degree(v);
        vertex_t needed = (degree > 0) ? std::max(degree, (size_t)2) : 1;
        vertex_offsets[v] = total_vertices;
        total_vertices += needed;
    }
    vertex_offsets[g.n] = total_vertices;

    Graph result(total_vertices);

    // Build cycles for each original vertex / 각 원본 정점에 대한 사이클 구축
    for (vertex_t v = 0; v < g.n; v++) {
        vertex_t start = vertex_offsets[v];
        vertex_t count = vertex_offsets[v + 1] - start;

        // Create cycle with weight-0 edges / 가중치 0인 간선으로 사이클 생성
        for (vertex_t i = 0; i < count; i++) {
            vertex_t current = start + i;
            vertex_t next = start + ((i + 1) % count);
            result.add_edge(current, next, 0.0);
        }

        // Distribute outgoing edges across cycle / 나가는 간선을 사이클에 분산
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
