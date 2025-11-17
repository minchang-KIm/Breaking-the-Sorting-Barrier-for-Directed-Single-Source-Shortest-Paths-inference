#include <iostream>
#include <fstream>
#include <random>
#include <string>
#include <cstring>

void print_usage(const char* program) {
    std::cout << "Usage: " << program << " [OPTIONS]\n"
              << "Options:\n"
              << "  -n NUM          Number of vertices\n"
              << "  -m NUM          Number of edges\n"
              << "  -t TYPE         Graph type:\n"
              << "                    random    - Random graph\n"
              << "                    grid      - 2D grid\n"
              << "                    dag       - Random DAG\n"
              << "  -w MAX_WEIGHT   Maximum edge weight (default: 100.0)\n"
              << "  -o FILE         Output file\n"
              << "  -s SEED         Random seed (default: random)\n"
              << "  -h, --help      Show this help message\n";
}

void generate_random_graph(std::ofstream& out, uint32_t n, uint64_t m,
                          double max_weight, std::mt19937& rng) {
    out << n << " " << m << "\n";

    std::uniform_int_distribution<uint32_t> vertex_dist(0, n - 1);
    std::uniform_real_distribution<double> weight_dist(0.1, max_weight);

    for (uint64_t i = 0; i < m; i++) {
        uint32_t u = vertex_dist(rng);
        uint32_t v = vertex_dist(rng);
        double w = weight_dist(rng);
        out << u << " " << v << " " << w << "\n";
    }
}

void generate_grid_graph(std::ofstream& out, uint32_t n, double max_weight,
                        std::mt19937& rng) {
    uint32_t grid_size = (uint32_t)std::sqrt(n);
    uint32_t actual_n = grid_size * grid_size;

    std::uniform_real_distribution<double> weight_dist(0.1, max_weight);

    // Count edges first
    uint64_t edge_count = 0;
    for (uint32_t i = 0; i < grid_size; i++) {
        for (uint32_t j = 0; j < grid_size; j++) {
            if (j < grid_size - 1) edge_count++; // Right
            if (i < grid_size - 1) edge_count++; // Down
        }
    }

    out << actual_n << " " << edge_count << "\n";

    auto get_vertex = [grid_size](uint32_t i, uint32_t j) {
        return i * grid_size + j;
    };

    for (uint32_t i = 0; i < grid_size; i++) {
        for (uint32_t j = 0; j < grid_size; j++) {
            uint32_t u = get_vertex(i, j);

            // Right edge
            if (j < grid_size - 1) {
                uint32_t v = get_vertex(i, j + 1);
                double w = weight_dist(rng);
                out << u << " " << v << " " << w << "\n";
            }

            // Down edge
            if (i < grid_size - 1) {
                uint32_t v = get_vertex(i + 1, j);
                double w = weight_dist(rng);
                out << u << " " << v << " " << w << "\n";
            }
        }
    }
}

void generate_dag(std::ofstream& out, uint32_t n, uint64_t m,
                 double max_weight, std::mt19937& rng) {
    out << n << " " << m << "\n";

    std::uniform_real_distribution<double> weight_dist(0.1, max_weight);

    for (uint64_t i = 0; i < m; i++) {
        std::uniform_int_distribution<uint32_t> u_dist(0, n - 2);
        uint32_t u = u_dist(rng);

        std::uniform_int_distribution<uint32_t> v_dist(u + 1, n - 1);
        uint32_t v = v_dist(rng);

        double w = weight_dist(rng);
        out << u << " " << v << " " << w << "\n";
    }
}

int main(int argc, char** argv) {
    uint32_t n = 1000;
    uint64_t m = 5000;
    double max_weight = 100.0;
    std::string graph_type = "random";
    std::string output_file;
    uint32_t seed = std::random_device{}();

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
            n = std::stoul(argv[++i]);
        } else if (strcmp(argv[i], "-m") == 0 && i + 1 < argc) {
            m = std::stoull(argv[++i]);
        } else if (strcmp(argv[i], "-t") == 0 && i + 1 < argc) {
            graph_type = argv[++i];
        } else if (strcmp(argv[i], "-w") == 0 && i + 1 < argc) {
            max_weight = std::stod(argv[++i]);
        } else if (strcmp(argv[i], "-o") == 0 && i + 1 < argc) {
            output_file = argv[++i];
        } else if (strcmp(argv[i], "-s") == 0 && i + 1 < argc) {
            seed = std::stoul(argv[++i]);
        } else if (strcmp(argv[i], "-h") == 0 || strcmp(argv[i], "--help") == 0) {
            print_usage(argv[0]);
            return 0;
        }
    }

    if (output_file.empty()) {
        std::cerr << "Error: Output file required\n";
        print_usage(argv[0]);
        return 1;
    }

    std::mt19937 rng(seed);
    std::ofstream out(output_file);

    if (!out.is_open()) {
        std::cerr << "Error: Cannot write to file " << output_file << std::endl;
        return 1;
    }

    std::cout << "Generating " << graph_type << " graph with n=" << n
              << ", m=" << m << ", seed=" << seed << "\n";

    if (graph_type == "random") {
        generate_random_graph(out, n, m, max_weight, rng);
    } else if (graph_type == "grid") {
        generate_grid_graph(out, n, max_weight, rng);
    } else if (graph_type == "dag") {
        generate_dag(out, n, m, max_weight, rng);
    } else {
        std::cerr << "Error: Unknown graph type " << graph_type << std::endl;
        return 1;
    }

    out.close();
    std::cout << "Graph saved to " << output_file << "\n";

    return 0;
}
