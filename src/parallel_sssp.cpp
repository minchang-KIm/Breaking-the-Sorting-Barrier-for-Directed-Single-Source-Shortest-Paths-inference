#include "parallel_sssp.hpp"
#include <algorithm>
#include <cmath>
#include <queue>

namespace sssp {
namespace parallel {

// DistributedSSSP implementation
DistributedSSSP::DistributedSSSP(const Graph& g, vertex_t src)
    : graph(g), source(src) {
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    global_db.resize(graph.n, INF);
    global_db[source] = 0.0;

    partition_graph();

    local_db.resize(local_vertices.size(), INF);
    local_pred.resize(local_vertices.size(), UINT32_MAX);
    local_complete.resize(local_vertices.size(), false);

    // Initialize source
    if (source >= local_start && source < local_end) {
        size_t local_idx = source - local_start;
        local_db[local_idx] = 0.0;
        local_complete[local_idx] = true;
    }
}

DistributedSSSP::~DistributedSSSP() {
    // MPI cleanup handled by MPI_Finalize
}

void DistributedSSSP::partition_graph() {
    // Simple range partitioning
    vertex_t vertices_per_proc = (graph.n + world_size - 1) / world_size;
    local_start = rank * vertices_per_proc;
    local_end = std::min(local_start + vertices_per_proc, graph.n);

    for (vertex_t v = local_start; v < local_end; v++) {
        local_vertices.push_back(v);
    }
}

void DistributedSSSP::synchronize_distances() {
    // Gather all local distance updates
    std::vector<weight_t> all_distances(graph.n);

    #pragma omp parallel for
    for (size_t i = 0; i < local_vertices.size(); i++) {
        all_distances[local_vertices[i]] = local_db[i];
    }

    // MPI Allreduce with MIN operation
    MPI_Allreduce(all_distances.data(), global_db.data(), graph.n,
                  MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD);

    // Update local distances from global
    #pragma omp parallel for
    for (size_t i = 0; i < local_vertices.size(); i++) {
        local_db[i] = global_db[local_vertices[i]];
    }
}

void DistributedSSSP::parallel_relax(const std::set<vertex_t>& active_vertices) {
    std::vector<vertex_t> active_vec(active_vertices.begin(), active_vertices.end());

    #pragma omp parallel for schedule(dynamic, 64)
    for (size_t i = 0; i < active_vec.size(); i++) {
        vertex_t u = active_vec[i];

        // Check if u is in local partition
        if (u < local_start || u >= local_end) continue;

        size_t local_idx = u - local_start;
        if (!local_complete[local_idx]) continue;

        for (const Edge& e : graph.adj[u]) {
            vertex_t v = e.to;
            weight_t new_dist = local_db[local_idx] + e.weight;

            // Atomic update
            #pragma omp critical
            {
                if (new_dist < global_db[v]) {
                    global_db[v] = new_dist;
                }
            }
        }
    }
}

void DistributedSSSP::compute() {
    uint32_t k = Graph::compute_k(graph.n);
    uint32_t max_iterations = graph.n;

    for (uint32_t iter = 0; iter < max_iterations; iter++) {
        synchronize_distances();

        // Find active vertices (incomplete with finite distance)
        std::set<vertex_t> active;

        #pragma omp parallel
        {
            std::set<vertex_t> local_active;

            #pragma omp for nowait
            for (size_t i = 0; i < local_vertices.size(); i++) {
                vertex_t v = local_vertices[i];
                if (!local_complete[i] && global_db[v] < INF) {
                    local_active.insert(v);
                }
            }

            #pragma omp critical
            active.insert(local_active.begin(), local_active.end());
        }

        if (active.empty()) break;

        // Parallel relaxation
        parallel_relax(active);

        // Update completion status
        #pragma omp parallel for
        for (size_t i = 0; i < local_vertices.size(); i++) {
            vertex_t v = local_vertices[i];
            if (global_db[v] == local_db[i] && global_db[v] < INF) {
                local_complete[i] = true;
            }
        }

        // Check global convergence
        int local_done = active.empty() ? 1 : 0;
        int global_done = 0;
        MPI_Allreduce(&local_done, &global_done, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);

        if (global_done) break;
    }

    synchronize_distances();
}

void DistributedSSSP::gather_results() {
    synchronize_distances();
}

// SharedMemorySSSP implementation
SharedMemorySSSP::SharedMemorySSSP(const Graph& g, vertex_t src, int threads)
    : graph(g), source(src), db(g.n, INF), pred(g.n, UINT32_MAX), complete(g.n, false) {

    k = Graph::compute_k(g.n);
    t = Graph::compute_t(g.n);

    db[source] = 0.0;
    complete[source] = true;

    num_threads = (threads > 0) ? threads : omp_get_max_threads();
    omp_set_num_threads(num_threads);
}

void SharedMemorySSSP::parallel_relax_edges(const std::vector<vertex_t>& vertices) {
    #pragma omp parallel for schedule(dynamic, 64)
    for (size_t i = 0; i < vertices.size(); i++) {
        vertex_t u = vertices[i];

        for (const Edge& e : graph.adj[u]) {
            vertex_t v = e.to;
            weight_t new_dist = db[u] + e.weight;

            // Atomic compare-and-swap for distance update
            #pragma omp critical
            {
                if (new_dist < db[v]) {
                    db[v] = new_dist;
                    pred[v] = u;
                }
            }
        }
    }
}

void SharedMemorySSSP::compute() {
    // Parallel Bellman-Ford with k iterations
    std::vector<bool> in_queue(graph.n, false);
    std::vector<vertex_t> current_frontier;

    current_frontier.push_back(source);
    in_queue[source] = true;

    for (uint32_t iter = 0; iter < graph.n; iter++) {
        if (current_frontier.empty()) break;

        // Relax edges in parallel
        parallel_relax_edges(current_frontier);

        // Build next frontier
        std::vector<vertex_t> next_frontier;

        #pragma omp parallel
        {
            std::vector<vertex_t> local_frontier;

            #pragma omp for nowait
            for (vertex_t v = 0; v < graph.n; v++) {
                if (!complete[v] && db[v] < INF) {
                    local_frontier.push_back(v);
                }
            }

            #pragma omp critical
            {
                next_frontier.insert(next_frontier.end(),
                                   local_frontier.begin(), local_frontier.end());
            }
        }

        // Mark current frontier as complete
        #pragma omp parallel for
        for (size_t i = 0; i < current_frontier.size(); i++) {
            complete[current_frontier[i]] = true;
        }

        current_frontier = std::move(next_frontier);
        std::fill(in_queue.begin(), in_queue.end(), false);

        for (vertex_t v : current_frontier) {
            in_queue[v] = true;
        }
    }
}

std::vector<vertex_t> SharedMemorySSSP::get_path(vertex_t v) const {
    std::vector<vertex_t> path;
    vertex_t current = v;

    while (current != UINT32_MAX) {
        path.push_back(current);
        current = pred[current];
    }

    std::reverse(path.begin(), path.end());
    return path;
}

} // namespace parallel
} // namespace sssp
