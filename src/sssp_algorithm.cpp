/**
 * @file sssp_algorithm.cpp
 * @brief Implementation of the O(m log^(2/3) n) SSSP algorithm
 *
 * Based on "Breaking the Sorting Barrier for Directed Single-Source Shortest Paths"
 * by Duan et al., 2025
 *
 * This file implements three key algorithms from the paper:
 * 1. FindPivots (Algorithm 1): Identifies vertices with large shortest-path trees
 * 2. BaseCase (Algorithm 2): Mini-Dijkstra for small subproblems
 * 3. BMSSP (Algorithm 3): Bounded Multi-Source Shortest Path solver
 *
 * Key innovation: Breaks the O(m + n log n) barrier using a divide-and-conquer
 * approach with partial sorting data structures
 *
 * 주요 알고리즘 구현 파일:
 * - FindPivots: 큰 최단 경로 트리를 가진 정점 식별
 * - BaseCase: 작은 부분 문제를 위한 미니 다익스트라
 * - BMSSP: 경계가 있는 다중 소스 최단 경로 해결기
 */

#include "sssp_algorithm.hpp"
#include <queue>
#include <algorithm>
#include <map>
#include <cmath>
#include <stdexcept>

namespace sssp {

/**
 * @brief Relax an edge (u, v) with weight w
 * @param u Source vertex / 시작 정점
 * @param v Target vertex / 도착 정점
 * @param w Edge weight / 간선 가중치
 *
 * 간선 (u, v)를 가중치 w로 완화
 *
 * Standard relaxation operation from Dijkstra/Bellman-Ford
 * Updates distance if a shorter path is found
 */
void SSSPAlgorithm::relax_edge(vertex_t u, vertex_t v, weight_t w) {
    if (db[u] + w < db[v]) {
        db[v] = db[u] + w;
        pred[v] = u;
    }
}

/******************************************************************************
 * Algorithm 1: FindPivots
 *
 * Purpose: Reduce frontier size by identifying "pivot" vertices
 * Input: Bound B, source set S
 * Output: Pivot set P ⊆ S, reachable set W
 *
 * Strategy: Perform k steps of relaxation from S
 * If W grows too large (|W| > k|S|), return S as pivots (early termination)
 * Otherwise, build forest and select pivots as roots of large trees (size ≥ k)
 *
 * Time complexity: O(k·|edges explored|)
 *
 * 목적: "피벗" 정점을 식별하여 프론티어 크기 축소
 * 입력: 경계 B, 소스 집합 S
 * 출력: 피벗 집합 P ⊆ S, 도달 가능 집합 W
 *
 * 전략: S에서 k 단계의 완화 수행
 * W가 너무 크면 (|W| > k|S|) S를 피벗으로 반환 (조기 종료)
 * 그렇지 않으면 숲을 구축하고 큰 트리(크기 ≥ k)의 루트를 피벗으로 선택
 ******************************************************************************/
SSSPAlgorithm::PivotsResult SSSPAlgorithm::find_pivots(weight_t B, const std::set<vertex_t>& S) {
    PivotsResult result;
    std::set<vertex_t> W = S;
    std::set<vertex_t> W_prev = S;

    // Relax for k steps / k 단계 완화
    for (uint32_t step = 0; step < k; step++) {
        std::set<vertex_t> W_next;

        // Explore from all vertices in current frontier
        // 현재 프론티어의 모든 정점에서 탐색
        for (vertex_t u : W_prev) {
            for (const Edge& e : graph.adj[u]) {
                vertex_t v = e.to;
                weight_t new_dist = db[u] + e.weight;

                // Relax edge and add to frontier if improves and within bound
                // 간선 완화 및 개선되고 경계 내이면 프론티어에 추가
                if (new_dist < db[v]) {
                    db[v] = new_dist;
                    pred[v] = u;

                    if (new_dist < B) {
                        W_next.insert(v);
                        W.insert(v);
                    }
                }
            }
        }

        W_prev = W_next;

        // Early termination if W grows too large
        // W가 너무 크면 조기 종료
        if (W.size() > k * S.size()) {
            result.pivots = S;
            result.W = W;
            return result;
        }
    }

    // Build forest F and find pivots
    // 숲 F 구축 및 피벗 찾기
    std::map<vertex_t, std::vector<vertex_t>> children; // Tree structure / 트리 구조
    std::map<vertex_t, vertex_t> parent;
    std::set<vertex_t> roots;

    // Build parent-child relationships within W
    // W 내에서 부모-자식 관계 구축
    for (vertex_t v : W) {
        if (pred[v] != UINT32_MAX && W.count(pred[v])) {
            parent[v] = pred[v];
            children[pred[v]].push_back(v);
        } else {
            roots.insert(v);
        }
    }

    // Compute tree sizes using DFS
    // DFS를 사용하여 트리 크기 계산
    std::map<vertex_t, uint32_t> tree_size;
    std::function<uint32_t(vertex_t)> compute_size = [&](vertex_t u) -> uint32_t {
        uint32_t size = 1;
        if (children.count(u)) {
            for (vertex_t child : children[u]) {
                size += compute_size(child);
            }
        }
        tree_size[u] = size;
        return size;
    };

    for (vertex_t root : roots) {
        compute_size(root);
    }

    // Select pivots: vertices in S that are roots of trees with ≥ k vertices
    // 피벗 선택: k개 이상의 정점을 가진 트리의 루트인 S의 정점
    for (vertex_t u : S) {
        if (tree_size.count(u) && tree_size[u] >= k) {
            result.pivots.insert(u);
        }
    }

    // If no pivots found, use S as fallback
    // 피벗이 없으면 S를 대체로 사용
    if (result.pivots.empty() && !S.empty()) {
        result.pivots = S;
    }

    result.W = W;
    return result;
}

/******************************************************************************
 * Algorithm 2: BaseCase
 *
 * Purpose: Solve small subproblems using mini-Dijkstra
 * Input: Bound B, singleton source set S = {x}
 * Output: Bound B', vertex set U
 *
 * Strategy: Run Dijkstra from x until:
 * - k+1 vertices explored, OR
 * - All reachable vertices within B found
 *
 * If |U| > k, set B' to max distance in U and filter
 * Otherwise, B' = B
 *
 * Time complexity: O(k log k) using binary heap
 *
 * 목적: 미니 다익스트라를 사용하여 작은 부분 문제 해결
 * 전략: x에서 다익스트라 실행:
 * - k+1 정점 탐색, 또는
 * - B 내의 모든 도달 가능한 정점 발견
 ******************************************************************************/
SSSPAlgorithm::BMSSPResult SSSPAlgorithm::base_case(weight_t B, const std::set<vertex_t>& S) {
    BMSSPResult result;

    if (S.size() != 1) {
        // Error: S should be singleton / 오류: S는 단일 원소여야 함
        result.B_prime = B;
        return result;
    }

    vertex_t x = *S.begin();
    result.U.insert(x);

    // Mini Dijkstra's algorithm / 미니 다익스트라 알고리즘
    // Priority queue: (distance, vertex) / 우선순위 큐: (거리, 정점)
    using PQElement = std::pair<weight_t, vertex_t>;
    std::priority_queue<PQElement, std::vector<PQElement>, std::greater<PQElement>> heap;

    heap.push({db[x], x});

    while (!heap.empty() && result.U.size() < k + 1) {
        auto [dist, u] = heap.top();
        heap.pop();

        if (dist > db[u]) continue; // Outdated entry / 오래된 항목

        result.U.insert(u);
        complete[u] = true;

        // Relax outgoing edges / 나가는 간선 완화
        for (const Edge& e : graph.adj[u]) {
            vertex_t v = e.to;
            weight_t new_dist = db[u] + e.weight;

            if (new_dist < db[v] && new_dist < B) {
                db[v] = new_dist;
                pred[v] = u;
                heap.push({db[v], v});
            }
        }
    }

    // Determine B' / B' 결정
    if (result.U.size() <= k) {
        result.B_prime = B;
    } else {
        // Find max distance in U and create B' / U에서 최대 거리 찾기 및 B' 생성
        weight_t max_dist = 0.0;
        for (vertex_t v : result.U) {
            max_dist = std::max(max_dist, db[v]);
        }
        result.B_prime = max_dist;

        // Remove vertices with distance ≥ B' / 거리 ≥ B'인 정점 제거
        std::set<vertex_t> U_filtered;
        for (vertex_t v : result.U) {
            if (db[v] < result.B_prime) {
                U_filtered.insert(v);
            }
        }
        result.U = U_filtered;
    }

    return result;
}

/******************************************************************************
 * Algorithm 3: BMSSP (Bounded Multi-Source Shortest Path)
 *
 * Purpose: Main recursive algorithm for SSSP
 * Input: Recursion level ℓ, bound B, source set S
 * Output: Bound B', completed vertex set U
 *
 * Strategy: Divide-and-conquer with partial sorting
 * - Base case (ℓ=0): Use mini-Dijkstra
 * - Recursive case:
 *   1. Find pivots P ⊆ S using FindPivots
 *   2. Use partial sorting data structure D to maintain frontier
 *   3. Repeatedly pull M smallest elements from D
 *   4. Recursively solve subproblems
 *   5. Relax edges and update D
 *
 * Time complexity: O(m log^(2/3) n)
 *
 * 목적: SSSP를 위한 주요 재귀 알고리즘
 * 전략: 부분 정렬을 사용한 분할 정복
 ******************************************************************************/
SSSPAlgorithm::BMSSPResult SSSPAlgorithm::bmssp(uint32_t level, weight_t B, const std::set<vertex_t>& S) {
    BMSSPResult result;

    // Base case: level 0 / 기본 케이스: 레벨 0
    if (level == 0) {
        return base_case(B, S);
    }

    // Find pivots / 피벗 찾기
    auto [P, W] = find_pivots(B, S);

    // Initialize data structure D / 자료 구조 D 초기화
    // Size parameters from paper / 논문의 크기 매개변수
    uint32_t M = 1u << ((level - 1) * t);
    uint32_t N = 4 * k * (1u << (level * t));
    PartialSortDS D(N, M, B);

    // Insert pivots into D / D에 피벗 삽입
    for (vertex_t x : P) {
        D.insert(x, db[x]);
    }

    // Track minimum separator / 최소 구분자 추적
    weight_t B_prime_prev = INF;
    for (vertex_t x : P) {
        B_prime_prev = std::min(B_prime_prev, db[x]);
    }
    if (P.empty()) B_prime_prev = B;

    std::set<vertex_t> U;

    // Main iteration loop / 주요 반복 루프
    // Iterate until U is large enough or D is empty
    // U가 충분히 크거나 D가 비어있을 때까지 반복
    uint32_t max_vertices = k * (1u << (level * t));

    while (U.size() < max_vertices && !D.empty()) {
        // Pull M smallest elements from D / D에서 M개의 최소 요소 추출
        auto [S_i, B_i] = D.pull();
        std::set<vertex_t> S_i_set(S_i.begin(), S_i.end());

        // Recursive call at level ℓ-1 / 레벨 ℓ-1에서 재귀 호출
        auto [B_prime_i, U_i] = bmssp(level - 1, B_i, S_i_set);

        // Add U_i to U / U에 U_i 추가
        U.insert(U_i.begin(), U_i.end());

        // Mark U_i as complete / U_i를 완료로 표시
        for (vertex_t v : U_i) {
            complete[v] = true;
        }

        // Relax edges and update D / 간선 완화 및 D 업데이트
        // K contains vertices to be batch-prepended / K는 배치 전위될 정점 포함
        std::vector<std::pair<vertex_t, weight_t>> K;

        for (vertex_t u : U_i) {
            for (const Edge& e : graph.adj[u]) {
                vertex_t v = e.to;
                weight_t new_dist = db[u] + e.weight;

                if (new_dist < db[v]) {
                    db[v] = new_dist;
                    pred[v] = u;

                    // Categorize by distance range / 거리 범위로 분류
                    if (new_dist >= B_i && new_dist < B) {
                        D.insert(v, db[v]);
                    } else if (new_dist >= B_prime_i && new_dist < B_i) {
                        K.emplace_back(v, db[v]);
                    }
                }
            }
        }

        // Batch prepend K and unpulled S_i vertices / K와 추출되지 않은 S_i 정점 배치 전위
        for (vertex_t x : S_i) {
            if (db[x] >= B_prime_i && db[x] < B_i) {
                K.emplace_back(x, db[x]);
            }
        }

        if (!K.empty()) {
            D.batch_prepend(K);
        }

        B_prime_prev = B_prime_i;
    }

    // Add complete vertices from W / W에서 완료된 정점 추가
    for (vertex_t x : W) {
        if (db[x] < B_prime_prev) {
            U.insert(x);
            complete[x] = true;
        }
    }

    result.B_prime = std::min(B_prime_prev, B);
    result.U = U;

    return result;
}

/**
 * @brief Main entry point - Compute shortest paths from source
 *
 * 주요 진입점 - 소스에서 최단 경로 계산
 *
 * Determines maximum recursion level and calls BMSSP
 * FIXED: Correct recursion depth calculation
 */
void SSSPAlgorithm::compute_shortest_paths() {
    // FIXED: Correct maximum level calculation
    // 수정됨: 올바른 최대 레벨 계산
    // max_level = ⌈log_t(log n)⌉
    if (graph.n <= 1) {
        return; // Trivial case / 자명한 경우
    }

    double log_n = std::log(graph.n);
    uint32_t max_level = static_cast<uint32_t>(std::ceil(log_n / (t * std::log(2.0))));

    // Ensure at least level 1 / 최소 레벨 1 보장
    max_level = std::max(max_level, 1u);

    std::set<vertex_t> S = {source};
    auto [B_prime, U] = bmssp(max_level, INF, S);

    // Mark all vertices in U as complete / U의 모든 정점을 완료로 표시
    for (vertex_t v : U) {
        complete[v] = true;
    }
}

/**
 * @brief Reconstruct shortest path to vertex v
 * @param v Target vertex / 목표 정점
 * @return Vector of vertices in path from source to v
 *
 * v까지의 최단 경로 재구성
 *
 * Follows predecessor pointers backwards, then reverses
 */
std::vector<vertex_t> SSSPAlgorithm::get_path(vertex_t v) const {
    std::vector<vertex_t> path;
    vertex_t current = v;

    // Follow predecessors / 선행자 따라가기
    while (current != UINT32_MAX) {
        path.push_back(current);
        current = pred[current];
    }

    // Reverse to get source-to-target order / 소스에서 목표 순서로 역전
    std::reverse(path.begin(), path.end());
    return path;
}

} // namespace sssp
