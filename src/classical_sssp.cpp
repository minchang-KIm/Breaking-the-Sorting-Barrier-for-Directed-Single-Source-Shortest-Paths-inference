/**
 * @file classical_sssp.cpp
 * @brief Implementation of classical SSSP algorithms
 *
 * 고전적인 SSSP 알고리즘 구현
 *
 * Provides textbook implementations of:
 * - Dijkstra's algorithm (1959)
 * - Bellman-Ford algorithm (1956-1958)
 */

#include "classical_sssp.hpp"
#include <queue>
#include <algorithm>
#include <limits>

namespace sssp {
namespace classical {

/******************************************************************************
 * Dijkstra's Algorithm
 *
 * Named after Edsger W. Dijkstra (1959)
 * One of the most famous algorithms in computer science
 *
 * Key idea: Greedy algorithm that maintains a priority queue
 * Always processes the vertex with smallest tentative distance
 *
 * Complexity:
 * - Time: O((m + n) log n) with binary heap
 * - Space: O(n)
 *
 * Limitation: Requires non-negative edge weights
 ******************************************************************************/

Dijkstra::Dijkstra(const Graph& g, vertex_t src)
    : graph(g), source(src), dist(g.n, INF), pred(g.n, UINT32_MAX) {
}

/**
 * @brief Main Dijkstra's algorithm implementation
 *
 * 다익스트라 알고리즘 주요 구현
 *
 * Algorithm:
 * 1. Initialize: dist[source] = 0, all others = ∞
 * 2. Priority queue with (distance, vertex) pairs
 * 3. While queue not empty:
 *    a. Extract minimum distance vertex u
 *    b. For each edge (u,v):
 *       - If dist[u] + weight(u,v) < dist[v]:
 *         - Update dist[v]
 *         - Add v to queue
 *
 * Correctness: Greedy choice property - once a vertex is processed,
 * its distance is optimal
 */
void Dijkstra::compute() {
    // Initialize source distance / 소스 거리 초기화
    dist[source] = 0.0;

    // Priority queue: (distance, vertex) / 우선순위 큐: (거리, 정점)
    // std::priority_queue is max-heap, so we negate or use greater<>
    using PQElement = std::pair<weight_t, vertex_t>;
    std::priority_queue<PQElement, std::vector<PQElement>, std::greater<PQElement>> pq;

    pq.push({0.0, source});

    // Track which vertices have been processed / 처리된 정점 추적
    std::vector<bool> processed(graph.n, false);

    while (!pq.empty()) {
        auto [d, u] = pq.top();
        pq.pop();

        // Skip if already processed / 이미 처리된 경우 건너뛰기
        if (processed[u]) continue;
        processed[u] = true;

        // Skip if distance is outdated / 거리가 오래된 경우 건너뛰기
        if (d > dist[u]) continue;

        // Relax all outgoing edges / 모든 나가는 간선 완화
        for (const Edge& e : graph.adj[u]) {
            vertex_t v = e.to;
            weight_t new_dist = dist[u] + e.weight;

            // Relaxation step / 완화 단계
            if (new_dist < dist[v]) {
                dist[v] = new_dist;
                pred[v] = u;
                pq.push({new_dist, v});
            }
        }
    }
}

/**
 * @brief Reconstruct shortest path from source to v
 *
 * 소스에서 v까지의 최단 경로 재구성
 */
std::vector<vertex_t> Dijkstra::get_path(vertex_t v) const {
    std::vector<vertex_t> path;

    // Check if vertex is reachable / 정점이 도달 가능한지 확인
    if (dist[v] == INF) {
        return path; // Empty path = unreachable / 빈 경로 = 도달 불가
    }

    // Follow predecessors backward / 선행자를 역으로 따라가기
    vertex_t current = v;
    while (current != UINT32_MAX) {
        path.push_back(current);
        current = pred[current];
    }

    // Reverse to get source-to-target order / 소스에서 목표 순서로 역전
    std::reverse(path.begin(), path.end());
    return path;
}

/******************************************************************************
 * Bellman-Ford Algorithm
 *
 * Named after Richard Bellman and Lester Ford Jr.
 * Bellman (1958): Dynamic Programming formulation
 * Ford (1956): Earlier version
 *
 * Key idea: Dynamic programming with relaxation
 * Process all edges n-1 times, guarantees optimality
 *
 * Complexity:
 * - Time: O(nm) worst case
 * - Space: O(n)
 *
 * Advantages over Dijkstra:
 * - Works with negative edge weights
 * - Detects negative cycles
 *
 * Slower but more general
 ******************************************************************************/

BellmanFord::BellmanFord(const Graph& g, vertex_t src)
    : graph(g), source(src), dist(g.n, INF), pred(g.n, UINT32_MAX),
      has_negative_cycle(false) {
}

/**
 * @brief Main Bellman-Ford algorithm implementation
 *
 * 벨만-포드 알고리즘 주요 구현
 *
 * Algorithm:
 * 1. Initialize: dist[source] = 0, all others = ∞
 * 2. Repeat n-1 times:
 *    - For each edge (u,v):
 *      - Relax: if dist[u] + weight(u,v) < dist[v], update dist[v]
 * 3. Check for negative cycles:
 *    - If any edge can still be relaxed, negative cycle exists
 *
 * Correctness: After k iterations, shortest paths using ≤k edges are correct
 * After n-1 iterations, all shortest paths are correct (assuming no negative cycles)
 */
void BellmanFord::compute() {
    // Initialize source distance / 소스 거리 초기화
    dist[source] = 0.0;

    // Main relaxation loop: repeat n-1 times / 주요 완화 루프: n-1회 반복
    // After k iterations, shortest paths with ≤k edges are optimal
    // k번 반복 후, ≤k개 간선을 사용하는 최단 경로가 최적
    for (vertex_t iter = 0; iter < graph.n - 1; iter++) {
        bool updated = false;

        // Relax all edges / 모든 간선 완화
        for (vertex_t u = 0; u < graph.n; u++) {
            // Skip vertices that haven't been reached yet / 아직 도달하지 않은 정점 건너뛰기
            if (dist[u] == INF) continue;

            for (const Edge& e : graph.adj[u]) {
                vertex_t v = e.to;
                weight_t new_dist = dist[u] + e.weight;

                // Relaxation step / 완화 단계
                if (new_dist < dist[v]) {
                    dist[v] = new_dist;
                    pred[v] = u;
                    updated = true;
                }
            }
        }

        // Early termination: if no updates, we're done / 조기 종료: 업데이트 없으면 완료
        if (!updated) break;
    }

    // Check for negative cycles / 음수 사이클 확인
    // If we can still relax edges, there's a negative cycle
    // 여전히 간선을 완화할 수 있으면 음수 사이클 존재
    for (vertex_t u = 0; u < graph.n; u++) {
        if (dist[u] == INF) continue;

        for (const Edge& e : graph.adj[u]) {
            vertex_t v = e.to;
            weight_t new_dist = dist[u] + e.weight;

            if (new_dist < dist[v]) {
                has_negative_cycle = true;
                return; // Negative cycle detected / 음수 사이클 감지됨
            }
        }
    }
}

/**
 * @brief Reconstruct shortest path from source to v
 *
 * 소스에서 v까지의 최단 경로 재구성
 *
 * Note: If negative cycle exists, paths may not be well-defined
 */
std::vector<vertex_t> BellmanFord::get_path(vertex_t v) const {
    std::vector<vertex_t> path;

    // Check for negative cycle / 음수 사이클 확인
    if (has_negative_cycle) {
        return path; // Paths not well-defined with negative cycles
    }

    // Check if vertex is reachable / 정점이 도달 가능한지 확인
    if (dist[v] == INF) {
        return path; // Empty path = unreachable / 빈 경로 = 도달 불가
    }

    // Follow predecessors backward / 선행자를 역으로 따라가기
    vertex_t current = v;
    std::set<vertex_t> visited; // Prevent infinite loops / 무한 루프 방지

    while (current != UINT32_MAX) {
        // Check for cycle in path (shouldn't happen without negative cycles)
        // 경로의 사이클 확인 (음수 사이클 없이는 발생하지 않아야 함)
        if (visited.count(current)) {
            path.clear();
            return path; // Cycle detected / 사이클 감지됨
        }

        visited.insert(current);
        path.push_back(current);
        current = pred[current];

        // Safety limit / 안전 한계
        if (path.size() > graph.n) {
            path.clear();
            return path;
        }
    }

    // Reverse to get source-to-target order / 소스에서 목표 순서로 역전
    std::reverse(path.begin(), path.end());
    return path;
}

} // namespace classical
} // namespace sssp
