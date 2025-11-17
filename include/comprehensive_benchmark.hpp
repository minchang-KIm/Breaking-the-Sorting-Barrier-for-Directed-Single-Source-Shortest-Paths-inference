/**
 * @file comprehensive_benchmark.hpp
 * @brief Comprehensive benchmarking framework for SSSP algorithms
 *
 * 포괄적인 SSSP 알고리즘 벤치마킹 프레임워크
 *
 * This framework provides:
 * - Multi-algorithm comparison
 * - Detailed performance metrics collection
 * - Statistical analysis
 * - Result export for paper figures
 */

#ifndef COMPREHENSIVE_BENCHMARK_HPP
#define COMPREHENSIVE_BENCHMARK_HPP

#include "graph.hpp"
#include <string>
#include <vector>
#include <map>
#include <chrono>

namespace sssp {
namespace benchmark {

/**
 * @brief Performance metrics for a single algorithm run
 *
 * 단일 알고리즘 실행에 대한 성능 지표
 */
struct PerformanceMetrics {
    std::string algorithm_name;      // Algorithm identifier / 알고리즘 식별자
    double wall_time_ms;             // Total execution time (ms) / 총 실행 시간
    double computation_time_ms;      // Pure computation time / 순수 계산 시간
    double communication_time_ms;    // Communication overhead / 통신 오버헤드

    size_t memory_usage_bytes;       // Peak memory usage / 최대 메모리 사용량
    size_t memory_allocated_bytes;   // Total allocated / 총 할당량

    uint64_t edges_relaxed;          // Number of edge relaxations / 간선 완화 횟수
    uint64_t vertices_visited;       // Vertices processed / 처리된 정점 수

    double speedup;                  // vs baseline / 기준 대비 속도 향상
    double efficiency;               // Parallel efficiency / 병렬 효율성

    bool correctness_verified;       // Result validation / 결과 검증
    std::string error_message;       // Error if any / 오류 메시지
};

/**
 * @brief Graph characteristics for analysis
 *
 * 분석을 위한 그래프 특성
 */
struct GraphCharacteristics {
    std::string name;
    vertex_t num_vertices;
    uint64_t num_edges;
    double avg_degree;
    double max_degree;
    double density;
    std::string graph_type;  // "road", "social", "random", etc.
};

/**
 * @brief Comprehensive benchmark suite
 *
 * 포괄적인 벤치마크 스위트
 */
class BenchmarkSuite {
private:
    std::vector<GraphCharacteristics> graphs;
    std::map<std::string, std::vector<PerformanceMetrics>> results;

    // Timing utilities
    std::chrono::high_resolution_clock::time_point start_time;

    // Validation
    bool verify_correctness(const Graph& g, vertex_t source,
                           const std::vector<weight_t>& distances);

    // Statistics
    double compute_mean(const std::vector<double>& values);
    double compute_stddev(const std::vector<double>& values);
    double compute_median(std::vector<double> values);

public:
    BenchmarkSuite();

    // Add graph to benchmark suite / 벤치마크 스위트에 그래프 추가
    void add_graph(const std::string& name, const Graph& g, const std::string& type);

    // Run benchmarks / 벤치마크 실행
    void run_dijkstra(const Graph& g, vertex_t source);
    void run_bellman_ford(const Graph& g, vertex_t source);
    void run_sequential_sssp(const Graph& g, vertex_t source);
    void run_parallel_sssp(const Graph& g, vertex_t source, int threads);
    void run_cuda_sssp(const Graph& g, vertex_t source);

    // Run all benchmarks / 모든 벤치마크 실행
    void run_all_benchmarks(const Graph& g, vertex_t source);

    // Analysis and reporting / 분석 및 보고
    void print_summary();
    void print_detailed_results();
    void export_csv(const std::string& filename);
    void export_latex_table(const std::string& filename);
    void export_json(const std::string& filename);

    // Statistical analysis / 통계 분석
    void compute_speedups(const std::string& baseline);
    void analyze_scalability();
    void generate_plots();
};

/**
 * @brief Dataset generator for benchmarks
 *
 * 벤치마크를 위한 데이터셋 생성기
 */
class DatasetGenerator {
public:
    // Generate random graph / 랜덤 그래프 생성
    static Graph generate_random(vertex_t n, uint64_t m, uint32_t seed = 42);

    // Generate grid graph / 그리드 그래프 생성
    static Graph generate_grid(uint32_t width, uint32_t height);

    // Generate scale-free graph (Barabási-Albert) / 척도 없는 그래프 생성
    static Graph generate_scale_free(vertex_t n, uint32_t m0, uint32_t m);

    // Generate road network-like graph / 도로 네트워크형 그래프 생성
    static Graph generate_road_network(vertex_t n, double avg_degree);

    // Generate DAG / DAG 생성
    static Graph generate_dag(vertex_t n, uint64_t m, uint32_t seed = 42);
};

} // namespace benchmark
} // namespace sssp

#endif // COMPREHENSIVE_BENCHMARK_HPP
