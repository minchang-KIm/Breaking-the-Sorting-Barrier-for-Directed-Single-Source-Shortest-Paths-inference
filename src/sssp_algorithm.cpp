#include "sssp_algorithm.hpp"
#include <queue>
#include <algorithm>
#include <map>

namespace sssp {

void SSSPAlgorithm::relax_edge(vertex_t u, vertex_t v, weight_t w) {
    // Remark 3.4: Use non-strict inequality (≤) to allow edge reuse across levels
    if (db[u] + w <= db[v]) {
        db[v] = db[u] + w;
        pred[v] = u;
    }
}

// Algorithm 1: FindPivots
SSSPAlgorithm::PivotsResult SSSPAlgorithm::find_pivots(weight_t B, const std::set<vertex_t>& S) {
    PivotsResult result;
    std::set<vertex_t> W = S;
    std::set<vertex_t> W_prev = S;

    // Relax for k steps
    for (uint32_t step = 0; step < k; step++) {
        std::set<vertex_t> W_next;

        for (vertex_t u : W_prev) {
            for (const Edge& e : graph.adj[u]) {
                vertex_t v = e.to;
                weight_t new_dist = db[u] + e.weight;

                if (new_dist <= db[v]) {
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
        if (W.size() > k * S.size()) {
            result.pivots = S;
            result.W = W;
            return result;
        }
    }

    // Build forest F and find pivots
    std::map<vertex_t, std::vector<vertex_t>> children; // Tree structure
    std::map<vertex_t, vertex_t> parent;
    std::set<vertex_t> roots;

    for (vertex_t v : W) {
        if (pred[v] != UINT32_MAX && W.count(pred[v])) {
            parent[v] = pred[v];
            children[pred[v]].push_back(v);
        } else {
            roots.insert(v);
        }
    }

    // Find tree sizes using DFS
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

    // Select pivots: roots of trees with >= k vertices
    for (vertex_t u : S) {
        if (tree_size.count(u) && tree_size[u] >= k) {
            result.pivots.insert(u);
        }
    }

    result.W = W;
    return result;
}

// Algorithm 2: BaseCase
SSSPAlgorithm::BMSSPResult SSSPAlgorithm::base_case(weight_t B, const std::set<vertex_t>& S) {
    BMSSPResult result;

    if (S.size() != 1) {
        // Error: S should be singleton
        result.B_prime = B;
        return result;
    }

    vertex_t x = *S.begin();
    result.U.insert(x);

    // Mini Dijkstra's algorithm
    using PQElement = std::pair<weight_t, vertex_t>;
    std::priority_queue<PQElement, std::vector<PQElement>, std::greater<PQElement>> heap;

    heap.push({db[x], x});

    while (!heap.empty() && result.U.size() < k + 1) {
        auto [dist, u] = heap.top();
        heap.pop();

        if (dist > db[u]) continue; // Outdated entry

        result.U.insert(u);
        complete[u] = true;

        for (const Edge& e : graph.adj[u]) {
            vertex_t v = e.to;
            weight_t new_dist = db[u] + e.weight;

            if (new_dist <= db[v] && new_dist < B) {
                db[v] = new_dist;
                pred[v] = u;
                heap.push({db[v], v});
            }
        }
    }

    if (result.U.size() <= k) {
        result.B_prime = B;
    } else {
        // Find max distance in U and create B_prime
        weight_t max_dist = 0.0;
        for (vertex_t v : result.U) {
            max_dist = std::max(max_dist, db[v]);
        }
        result.B_prime = max_dist;

        // Remove vertices with distance >= B_prime
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

// Algorithm 3: BMSSP
SSSPAlgorithm::BMSSPResult SSSPAlgorithm::bmssp(uint32_t level, weight_t B, const std::set<vertex_t>& S) {
    BMSSPResult result;

    // Base case
    if (level == 0) {
        return base_case(B, S);
    }

    // Find pivots
    auto [P, W] = find_pivots(B, S);

    // Initialize data structure D
    uint32_t M = 1u << ((level - 1) * t);
    PartialSortDS D(4 * k * (1u << (level * t)), M, B);

    // Insert pivots into D
    for (vertex_t x : P) {
        D.insert(x, db[x]);
    }

    weight_t B_prime_prev = INF;
    for (vertex_t x : P) {
        B_prime_prev = std::min(B_prime_prev, db[x]);
    }
    if (P.empty()) B_prime_prev = B;

    std::set<vertex_t> U;

    // Iterate until termination
    while (U.size() < k * (1u << (level * t)) && !D.empty()) {
        // Pull from D
        auto [S_i, B_i] = D.pull();
        std::set<vertex_t> S_i_set(S_i.begin(), S_i.end());

        // Recursive call
        auto [B_prime_i, U_i] = bmssp(level - 1, B_i, S_i_set);

        // Add U_i to U
        U.insert(U_i.begin(), U_i.end());

        // Mark U_i as complete
        for (vertex_t v : U_i) {
            complete[v] = true;
        }

        // Relax edges and update D
        std::vector<std::pair<vertex_t, weight_t>> K;

        for (vertex_t u : U_i) {
            for (const Edge& e : graph.adj[u]) {
                vertex_t v = e.to;
                weight_t new_dist = db[u] + e.weight;

                if (new_dist <= db[v]) {
                    db[v] = new_dist;
                    pred[v] = u;

                    if (new_dist >= B_i && new_dist < B) {
                        D.insert(v, db[v]);
                    } else if (new_dist >= B_prime_i && new_dist < B_i) {
                        K.emplace_back(v, db[v]);
                    }
                }
            }
        }

        // Batch prepend K and S_i vertices
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

    // Add complete vertices from W
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

void SSSPAlgorithm::compute_shortest_paths() {
    uint32_t max_level = static_cast<uint32_t>(std::ceil(std::log(graph.n) / t));

    std::set<vertex_t> S = {source};
    auto [B_prime, U] = bmssp(max_level, INF, S);

    // Mark all vertices in U as complete
    for (vertex_t v : U) {
        complete[v] = true;
    }
}

std::vector<vertex_t> SSSPAlgorithm::get_path(vertex_t v) const {
    std::vector<vertex_t> path;
    vertex_t current = v;

    while (current != UINT32_MAX) {
        path.push_back(current);
        current = pred[current];
    }

    std::reverse(path.begin(), path.end());
    return path;
}

} // namespace sssp
