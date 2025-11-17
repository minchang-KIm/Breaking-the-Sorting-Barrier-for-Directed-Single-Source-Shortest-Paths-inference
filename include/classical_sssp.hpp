/**
 * @file classical_sssp.hpp
 * @brief Classical SSSP algorithms for comparison and validation
 *
 * 비교 및 검증을 위한 고전적인 SSSP 알고리즘
 *
 * This file implements standard textbook algorithms:
 * - Dijkstra's algorithm: O(m + n log n) with binary heap
 * - Bellman-Ford algorithm: O(nm) worst case
 *
 * These implementations serve as:
 * 1. Baseline for performance comparison
 * 2. Correctness validation
 * 3. Reference implementations
 */

#ifndef CLASSICAL_SSSP_HPP
#define CLASSICAL_SSSP_HPP

#include "graph.hpp"
#include <vector>

namespace sssp {
namespace classical {

/**
 * @brief Dijkstra's algorithm implementation
 *
 * 다익스트라 알고리즘 구현
 *
 * Classic algorithm using binary heap priority queue
 * Time complexity: O((m + n) log n)
 * Space complexity: O(n)
 *
 * Requirements: Non-negative edge weights
 *
 * This is the standard algorithm taught in algorithms courses
 * Published by Edsger W. Dijkstra in 1959
 */
class Dijkstra {
private:
    const Graph& graph;
    vertex_t source;

    std::vector<weight_t> dist;
    std::vector<vertex_t> pred;

public:
    Dijkstra(const Graph& g, vertex_t src);

    // Main algorithm
    void compute();

    // Get results
    weight_t get_distance(vertex_t v) const { return dist[v]; }
    vertex_t get_predecessor(vertex_t v) const { return pred[v]; }
    const std::vector<weight_t>& get_distances() const { return dist; }

    // Get shortest path
    std::vector<vertex_t> get_path(vertex_t v) const;
};

/**
 * @brief Bellman-Ford algorithm implementation
 *
 * 벨만-포드 알고리즘 구현
 *
 * Classic dynamic programming algorithm
 * Time complexity: O(nm)
 * Space complexity: O(n)
 *
 * Advantages:
 * - Works with negative edge weights
 * - Detects negative cycles
 *
 * Slower than Dijkstra but more general
 *
 * Published by Richard Bellman (1958) and Lester Ford Jr. (1956)
 */
class BellmanFord {
private:
    const Graph& graph;
    vertex_t source;

    std::vector<weight_t> dist;
    std::vector<vertex_t> pred;
    bool has_negative_cycle;

public:
    BellmanFord(const Graph& g, vertex_t src);

    // Main algorithm
    void compute();

    // Get results
    weight_t get_distance(vertex_t v) const { return dist[v]; }
    vertex_t get_predecessor(vertex_t v) const { return pred[v]; }
    const std::vector<weight_t>& get_distances() const { return dist; }
    bool has_negative_cycle_detected() const { return has_negative_cycle; }

    // Get shortest path
    std::vector<vertex_t> get_path(vertex_t v) const;
};

} // namespace classical
} // namespace sssp

#endif // CLASSICAL_SSSP_HPP
