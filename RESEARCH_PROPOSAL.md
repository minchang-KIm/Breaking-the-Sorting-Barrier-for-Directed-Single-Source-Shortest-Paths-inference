# Research Proposal: HPC-Optimized SSSP with Multi-GPU NVLINK and Asynchronous Pipeline
## 연구 제안서: Multi-GPU NVLINK 및 비동기 파이프라인을 활용한 HPC 최적화 SSSP

**Author:** [Your Name]
**Date:** 2025-11-17
**Target:** Winter Domestic Conference Paper
**Repository:** Breaking-the-Sorting-Barrier-for-Directed-Single-Source-Shortest-Paths-inference

---

## 1. Research Motivation and Objectives
## 1. 연구 동기 및 목표

### 1.1 Background
### 1.1 배경

The recent breakthrough algorithm by Duan et al. (2025) achieves **O(m log^(2/3) n)** time complexity for directed single-source shortest paths (SSSP), breaking the O(m + n log n) sorting barrier that has stood since Dijkstra's algorithm (1959). However, the original paper focuses on theoretical complexity and sequential implementation, leaving significant opportunities for HPC optimization.

Duan 등(2025)의 최근 획기적인 알고리즘은 방향 그래프의 단일 출발점 최단 경로(SSSP) 문제에서 **O(m log^(2/3) n)** 시간 복잡도를 달성하여, Dijkstra 알고리즘(1959) 이후 유지되어 온 O(m + n log n) 정렬 장벽을 돌파했습니다. 하지만 원 논문은 이론적 복잡도와 순차 구현에 초점을 맞추고 있어, HPC 최적화를 위한 상당한 기회가 남아있습니다.

### 1.2 Research Gap
### 1.2 연구 공백

**Current State:**
- Sequential implementation available
- Basic OpenMP/MPI parallelization exists
- Single-GPU CUDA implementation with atomicMin optimization
- Limited scalability for billion-edge graphs

**현재 상태:**
- 순차 구현 가능
- 기본 OpenMP/MPI 병렬화 존재
- atomicMin 최적화를 적용한 단일 GPU CUDA 구현
- 수십억 간선 그래프에 대한 제한적 확장성

**Missing:**
- Multi-GPU coordination with NVLINK high-bandwidth interconnect
- Overlapping computation and communication (asynchronous pipeline)
- Optimized graph partitioning for minimal edge-cut
- Comprehensive performance analysis on HPC clusters

**부족한 부분:**
- NVLINK 고대역폭 인터커넥트를 활용한 Multi-GPU 조정
- 계산과 통신을 중첩하는 비동기 파이프라인
- 최소 간선 절단을 위한 최적화된 그래프 분할
- HPC 클러스터에서의 포괄적 성능 분석

### 1.3 Research Objectives
### 1.3 연구 목표

**Primary Objective:**
Propose and implement a **Multi-GPU Asynchronous Pipeline (MGAP)** optimization technique that:
1. Exploits NVLINK's 600GB/s bandwidth for direct GPU-to-GPU communication
2. Overlaps computation phases with inter-GPU data transfer
3. Reduces edge-cut through intelligent graph partitioning
4. Achieves **10-50x speedup** over sequential baseline on HPC systems

**주요 목표:**
다음을 수행하는 **Multi-GPU 비동기 파이프라인(MGAP)** 최적화 기법을 제안하고 구현:
1. NVLINK의 600GB/s 대역폭을 활용한 직접 GPU-to-GPU 통신
2. 계산 단계와 GPU 간 데이터 전송을 중첩
3. 지능적 그래프 분할을 통한 간선 절단 감소
4. HPC 시스템에서 순차 베이스라인 대비 **10-50배 속도 향상** 달성

**Secondary Objectives:**
- Compare against classical algorithms (Dijkstra, Bellman-Ford)
- Analyze time/space complexity theoretically and empirically
- Measure communication volume and edge-cut metrics
- Provide ease-of-use analysis for adoption in production systems

**부차 목표:**
- 고전 알고리즘(Dijkstra, Bellman-Ford)과 비교
- 이론적 및 경험적으로 시간/공간 복잡도 분석
- 통신량 및 간선 절단 메트릭 측정
- 프로덕션 시스템 도입을 위한 사용 용이성 분석

---

## 2. Proposed Technique: Multi-GPU Asynchronous Pipeline (MGAP)
## 2. 제안 기법: Multi-GPU 비동기 파이프라인 (MGAP)

### 2.1 Core Innovation
### 2.1 핵심 혁신

**Technique Name:** Multi-GPU Asynchronous Pipeline (MGAP) for SSSP
**기법 명칭:** SSSP를 위한 Multi-GPU 비동기 파이프라인 (MGAP)

**Key Components:**

1. **NVLINK-Accelerated Multi-GPU Coordination**
   - Direct P2P memory access between GPUs (bypass PCIe bottleneck)
   - Up to 600GB/s bandwidth vs 16GB/s PCIe Gen3
   - Expected: **3-5x communication speedup**

2. **Asynchronous Computation-Communication Pipeline**
   - CUDA streams for overlapping kernel execution and memory transfer
   - Triple-buffering strategy: compute(GPU_i) || transfer(GPU_j) || prepare(GPU_k)
   - Expected: **20-30% latency hiding improvement**

3. **METIS-Based Intelligent Graph Partitioning**
   - Minimize edge-cut using k-way partitioning
   - Balance vertex distribution across GPUs
   - Expected: **30-50% reduction in inter-GPU communication**

4. **Lock-Free Atomic Distance Updates**
   - Custom atomicMinDouble using CAS (Compare-And-Swap)
   - Eliminates mutex contention in parallel edge relaxation
   - Expected: **15-25% reduction in atomic operation overhead**

**주요 구성 요소:**

1. **NVLINK 가속 Multi-GPU 조정**
   - GPU 간 직접 P2P 메모리 액세스 (PCIe 병목 우회)
   - PCIe Gen3 16GB/s 대비 최대 600GB/s 대역폭
   - 예상: **통신 속도 3-5배 향상**

2. **비동기 계산-통신 파이프라인**
   - 커널 실행과 메모리 전송을 중첩하는 CUDA 스트림
   - 삼중 버퍼링 전략: compute(GPU_i) || transfer(GPU_j) || prepare(GPU_k)
   - 예상: **지연 시간 은닉 20-30% 개선**

3. **METIS 기반 지능형 그래프 분할**
   - k-way 분할을 사용한 간선 절단 최소화
   - GPU 간 정점 분포 균형 유지
   - 예상: **GPU 간 통신 30-50% 감소**

4. **락-프리 원자적 거리 업데이트**
   - CAS(Compare-And-Swap)를 사용한 커스텀 atomicMinDouble
   - 병렬 간선 완화에서 뮤텍스 경합 제거
   - 예상: **원자 연산 오버헤드 15-25% 감소**

### 2.2 Algorithm Design
### 2.2 알고리즘 설계

**High-Level Workflow:**

```
Algorithm: Multi-GPU Asynchronous Pipeline SSSP (MGAP-SSSP)
Input: Graph G(V, E, w), source vertex s, GPU count k
Output: Shortest distances d[v] for all v ∈ V

1. Preprocessing Phase:
   a. Partition graph using METIS k-way partitioning
   b. Distribute partitions to k GPUs with balanced load
   c. Build CSR representation on each GPU
   d. Enable NVLINK P2P access between all GPU pairs

2. Initialization Phase:
   a. Set d[s] = 0, d[v] = ∞ for v ≠ s on all GPUs
   b. Create CUDA streams: compute_stream[k], transfer_stream[k]
   c. Allocate triple-buffers for boundary vertices

3. Iterative Relaxation Phase (Asynchronous Pipeline):
   for iteration = 1 to max_iterations:
       // Stage 1: Local relaxation (parallel across GPUs)
       for each GPU_i in parallel:
           Launch relax_edges_kernel on compute_stream[i]
           Process local edges using atomicMinDouble

       // Stage 2: Boundary exchange (overlapped with next iteration)
       for each GPU_i in parallel:
           Async copy boundary distances to neighbors via NVLINK
           Use transfer_stream[i] for non-blocking transfer

       // Stage 3: Global convergence check (Allreduce)
       changed = MPI_Allreduce(local_changed, MPI_LOR)
       if not changed:
           break

4. Path Reconstruction Phase:
   Gather predecessor information from all GPUs
   Reconstruct path from s to target vertex

Return: Distance array d[]
```

**Technical Details:**

- **Graph Partitioning:** METIS multilevel k-way partitioning with edge-cut minimization
- **Load Balancing:** Balance vertices ±5% across GPUs
- **Communication Pattern:** Halo exchange for boundary vertices only
- **Synchronization:** Asynchronous barriers using CUDA events
- **Memory Management:** Unified memory with prefetching hints

---

## 3. Experimental Design
## 3. 실험 설계

### 3.1 Algorithms to Compare
### 3.1 비교 대상 알고리즘

| Algorithm | Time Complexity | Space Complexity | Implementation |
|-----------|----------------|------------------|----------------|
| Dijkstra (Sequential) | O((m+n) log n) | O(n) | CPU with binary heap |
| Bellman-Ford (Sequential) | O(nm) | O(n) | CPU iterative |
| Duan et al. (Sequential) | O(m log^(2/3) n) | O(n+m) | CPU baseline |
| Duan et al. (OpenMP) | O(m log^(2/3) n / p) | O(n+m) | Multi-core CPU |
| Duan et al. (Single-GPU) | O(m log^(2/3) n / p) | O(n+m) | CUDA baseline |
| **MGAP-SSSP (Proposed)** | **O(m log^(2/3) n / kp)** | **O(n+m)** | **Multi-GPU NVLINK** |

Where:
- m: number of edges
- n: number of vertices
- p: parallelism factor (GPU threads)
- k: number of GPUs

### 3.2 Datasets
### 3.2 데이터셋

**Small Scale (Correctness Validation):**
1. **Simple Graph:** 4 vertices, 5 edges (unit test)
2. **Grid Graph:** 1,000 vertices, ~2,000 edges
3. **Random DAG:** 10,000 vertices, 50,000 edges

**Medium Scale (Performance Baseline):**
4. **Road Network (USA-small):** 100K vertices, 250K edges
5. **Social Network (Twitter-sample):** 500K vertices, 2M edges
6. **Random Sparse:** 1M vertices, 5M edges (avg degree 5)

**Large Scale (HPC Scalability):**
7. **Road Network (USA-full):** 24M vertices, 58M edges
8. **Web Graph (Google):** 875K vertices, 5.1M edges
9. **Synthetic Scale-Free:** 100M vertices, 1B edges (power-law distribution)

**Dataset Sources:**
- Road networks: DIMACS Challenge (9th, 10th)
- Social networks: Stanford SNAP
- Synthetic: Custom graph generator with configurable parameters

### 3.3 Evaluation Metrics
### 3.3 평가 메트릭

**Performance Metrics:**
1. **Execution Time (ms):** Wall-clock time from start to convergence
2. **Speedup:** T_sequential / T_parallel
3. **Throughput (MTEPS):** Million Traversed Edges Per Second
4. **Scalability:** Weak/Strong scaling curves

**Resource Metrics:**
5. **Memory Usage (GB):** Peak GPU/CPU memory consumption
6. **Memory Bandwidth (GB/s):** Effective utilization of NVLINK/PCIe
7. **GPU Utilization (%):** Kernel execution time / total time

**Communication Metrics:**
8. **Edge-Cut:** Number of edges crossing partition boundaries
9. **Communication Volume (MB):** Total data transferred between GPUs
10. **Communication Time (%):** Ratio of communication to total time
11. **Message Count:** Number of inter-GPU synchronization events

**Quality Metrics:**
12. **Correctness:** Distance error compared to sequential baseline (tolerance 1e-5)
13. **Path Recovery:** Verification of shortest path reconstruction

### 3.4 Hardware Configuration
### 3.4 하드웨어 구성

**HPC System Specification (Expected):**
- **GPUs:** 4× NVIDIA A100 80GB with NVLINK (600GB/s per link)
- **CPU:** 2× AMD EPYC 7742 64-core (128 cores total)
- **RAM:** 512GB DDR4-3200
- **Interconnect:** NVLINK 3.0 (GPU-GPU), PCIe Gen4 (CPU-GPU)
- **Storage:** 4TB NVMe SSD for dataset staging

**Software Stack:**
- **OS:** Ubuntu 22.04 LTS
- **CUDA:** 12.0 or higher
- **MPI:** OpenMPI 4.1 with CUDA-aware support
- **Compiler:** GCC 11.4, nvcc 12.0
- **Libraries:** METIS 5.1, OpenMP 4.5

---

## 4. Expected Results and Contributions
## 4. 예상 결과 및 기여

### 4.1 Expected Performance Gains
### 4.1 예상 성능 향상

**Quantitative Targets:**

| Metric | Sequential Baseline | Single-GPU | **MGAP (4 GPUs)** | **Improvement** |
|--------|---------------------|------------|-------------------|-----------------|
| Execution Time (1M edges) | 1,200 ms | 45 ms | **12 ms** | **100× faster** |
| Edge-Cut (%) | N/A | 100% | **35%** | **65% reduction** |
| Communication Volume | N/A | 500 MB | **180 MB** | **64% reduction** |
| Memory Efficiency | 100% | 85% | **78%** | **22% overhead** |
| Scalability (4→8 GPUs) | N/A | N/A | **1.7× speedup** | **Linear trend** |

**Qualitative Contributions:**

1. **Breaking Theoretical Barriers in Practice:**
   - First HPC implementation of Duan et al.'s O(m log^(2/3) n) algorithm
   - Demonstrates practical viability of theoretical breakthrough
   - Bridges gap between algorithmic theory and high-performance computing

2. **Novel Parallelization Strategy:**
   - Asynchronous pipeline architecture for graph algorithms
   - Generalizable to other graph problems (BFS, PageRank, etc.)
   - Template for future GPU-accelerated divide-and-conquer algorithms

3. **Domestic Research Contribution:**
   - Advances South Korea's competitiveness in HPC and graph analytics
   - Provides open-source reference implementation for researchers
   - Applications in transportation networks, social media analysis, etc.

### 4.2 Paper Contributions
### 4.2 논문 기여도

**Main Contributions:**

1. **Algorithm Design:**
   - Multi-GPU asynchronous pipeline technique (MGAP)
   - Lock-free atomic distance updates with custom CUDA primitives
   - METIS-integrated graph partitioning for SSSP

2. **Implementation:**
   - Production-quality C++/CUDA codebase with full annotations
   - Comprehensive benchmark framework with 9 datasets
   - Correctness validation against classical algorithms

3. **Experimental Analysis:**
   - Detailed performance characterization on HPC cluster
   - Communication overhead analysis with edge-cut metrics
   - Scalability study (1-8 GPUs)

4. **Practical Insights:**
   - Best practices for multi-GPU graph algorithm implementation
   - Trade-offs between partitioning quality and overhead
   - Applicability analysis for different graph types

---

## 5. Paper Structure Outline
## 5. 논문 구조 개요

### English Version

**Title:** Breaking the Sorting Barrier in Practice: Multi-GPU Acceleration of O(m log^(2/3) n) SSSP with Asynchronous Pipeline

**Sections:**

1. **Introduction** (2 pages)
   - Background on SSSP and historical O(m + n log n) barrier
   - Recent theoretical breakthrough (Duan et al. 2025)
   - Motivation for HPC optimization
   - Paper contributions and structure

2. **Related Work** (1.5 pages)
   - Classical algorithms: Dijkstra, Bellman-Ford
   - Parallel SSSP: Δ-stepping, GraphLab
   - GPU implementations: Gunrock, CuSha
   - Multi-GPU techniques: NVLINK, CUDA-aware MPI

3. **Background and Preliminaries** (2 pages)
   - Graph definitions and notation
   - Duan et al. algorithm overview (Algorithms 1-3)
   - Time complexity proof sketch
   - HPC architecture: NVLINK, GPU memory hierarchy

4. **Proposed Technique: MGAP** (3 pages)
   - Architecture overview with diagrams
   - Component 1: NVLINK-based multi-GPU coordination
   - Component 2: Asynchronous pipeline design
   - Component 3: METIS graph partitioning
   - Component 4: Lock-free atomic operations
   - Theoretical analysis: complexity and communication cost

5. **Implementation** (2.5 pages)
   - Software architecture
   - CUDA kernel design
   - Memory management strategy
   - Synchronization mechanisms
   - Code annotations and best practices

6. **Experimental Evaluation** (4 pages)
   - Experimental setup (hardware, datasets)
   - Correctness validation
   - Performance results: time, speedup, throughput
   - Communication analysis: edge-cut, volume, overhead
   - Scalability study
   - Ablation study: impact of each component

7. **Discussion** (1.5 pages)
   - Strengths: performance gains, scalability
   - Weaknesses: partitioning overhead, memory constraints
   - Applicability: graph types, problem sizes
   - Ease of deployment

8. **Conclusion** (1 page)
   - Summary of contributions
   - Practical impact
   - Future work: dynamic graphs, distributed clusters

9. **References** (1 page)

**Total:** ~18-20 pages (excluding references and appendix)

### Korean Version (한국어 버전)

**제목:** 정렬 장벽을 실전에서 돌파하기: 비동기 파이프라인을 활용한 O(m log^(2/3) n) SSSP의 Multi-GPU 가속

**섹션:**

1. **서론** (2페이지)
2. **관련 연구** (1.5페이지)
3. **배경 및 예비 지식** (2페이지)
4. **제안 기법: MGAP** (3페이지)
5. **구현** (2.5페이지)
6. **실험 평가** (4페이지)
7. **논의** (1.5페이지)
8. **결론** (1페이지)
9. **참고문헌** (1페이지)

---

## 6. Implementation Timeline
## 6. 구현 일정

### Phase 1: Baseline Enhancement (Days 1-3)
- [ ] Review and annotate existing Dijkstra/Bellman-Ford implementations
- [ ] Add comprehensive Korean/English comments
- [ ] Verify correctness on all test cases

### Phase 2: MGAP Core Implementation (Days 4-7)
- [ ] Implement METIS graph partitioning integration
- [ ] Develop multi-GPU coordination layer with NVLINK P2P
- [ ] Create asynchronous pipeline with CUDA streams
- [ ] Implement lock-free atomicMinDouble kernel

### Phase 3: Benchmark Infrastructure (Days 8-10)
- [ ] Generate/download 9 benchmark datasets
- [ ] Extend benchmark framework with communication metrics
- [ ] Add memory profiling and GPU utilization tracking
- [ ] Create automated experiment runner scripts

### Phase 4: Experimental Evaluation (Days 11-14)
- [ ] Run correctness validation on all datasets
- [ ] Execute performance benchmarks (sequential, OpenMP, GPU, MGAP)
- [ ] Collect communication metrics (edge-cut, volume)
- [ ] Generate scalability data (1-8 GPUs if available)
- [ ] Create visualization scripts for graphs and tables

### Phase 5: Paper Writing (Days 15-21)
- [ ] Write English version (sections 1-9)
- [ ] Create figures and tables
- [ ] Translate to Korean
- [ ] Generate PDF using LaTeX
- [ ] Create Word-exportable version (Pandoc)

### Phase 6: Verification and Finalization (Days 22-25)
- [ ] Code-paper consistency check
- [ ] Reproducibility verification
- [ ] Peer review simulation
- [ ] Final polishing and formatting

---

## 7. Validation Checklist
## 7. 검증 체크리스트

### Code Implementation Checklist

- [ ] **Correctness:**
  - [ ] All unit tests pass (sequential, parallel, CUDA)
  - [ ] Distance accuracy within 1e-5 tolerance
  - [ ] Path reconstruction verified
  - [ ] Disconnected graph handling

- [ ] **Performance:**
  - [ ] Sequential baseline matches expected O(m log^(2/3) n)
  - [ ] MGAP achieves ≥10× speedup over sequential
  - [ ] NVLINK bandwidth ≥300 GB/s measured
  - [ ] Communication volume <40% of single-GPU

- [ ] **Scalability:**
  - [ ] Linear speedup for 1→2 GPUs (≥1.7×)
  - [ ] Sub-linear but positive for 2→4 GPUs (≥1.4×)
  - [ ] Weak scaling efficiency ≥70%

- [ ] **Code Quality:**
  - [ ] Korean + English annotations for all functions
  - [ ] Memory leak-free (verified with cuda-memcheck)
  - [ ] Proper error handling (CUDA_CHECK macros)
  - [ ] Compilation warnings resolved

### Paper Quality Checklist

- [ ] **Content:**
  - [ ] All claims supported by experimental data
  - [ ] Complexity analysis mathematically sound
  - [ ] Related work comprehensive and accurate
  - [ ] Figures/tables have clear captions

- [ ] **Consistency:**
  - [ ] Algorithm pseudocode matches implementation
  - [ ] Performance numbers match benchmark outputs
  - [ ] Graph sizes consistent across tables
  - [ ] References properly formatted

- [ ] **Language:**
  - [ ] English version grammatically correct
  - [ ] Korean version professionally translated
  - [ ] Technical terms consistently used
  - [ ] Abstract <250 words

- [ ] **Formatting:**
  - [ ] PDF renders correctly (fonts, equations)
  - [ ] Word export preserves formatting
  - [ ] Figures high-resolution (≥300 DPI)
  - [ ] Code listings syntax-highlighted

### Reproducibility Checklist

- [ ] **Documentation:**
  - [ ] README with build instructions
  - [ ] Dataset download links provided
  - [ ] Benchmark scripts included
  - [ ] Hardware requirements specified

- [ ] **Artifacts:**
  - [ ] Source code on GitHub with commit hash
  - [ ] Datasets archived (or generation scripts)
  - [ ] Benchmark results in CSV format
  - [ ] Visualization scripts for plots

---

## 8. Success Criteria
## 8. 성공 기준

**Minimum Viable Paper (MVP):**
1. ✅ MGAP implementation compiles and runs correctly
2. ✅ Achieves ≥10× speedup over sequential baseline
3. ✅ Reduces communication volume by ≥30%
4. ✅ Complete paper draft in English and Korean (15+ pages)
5. ✅ All experiments reproducible with provided scripts

**Stretch Goals:**
- 🎯 Achieve 50× speedup on billion-edge graphs
- 🎯 Strong scaling efficiency >80% (1→4 GPUs)
- 🎯 Published dataset contributions (new benchmark suite)
- 🎯 Acceptance at domestic conference (KCC, KSC)

---

## 9. References (Preliminary)
## 9. 참고문헌 (예비)

1. **Duan et al.** (2025). "Breaking the Sorting Barrier for Directed Single-Source Shortest Paths." arXiv:2504.17033v2.

2. **Dijkstra, E. W.** (1959). "A note on two problems in connexion with graphs." Numerische mathematik, 1(1), 269-271.

3. **Bellman, R.** (1958). "On a routing problem." Quarterly of applied mathematics, 16(1), 87-90.

4. **Meyer, U., & Sanders, P.** (2003). "Δ-stepping: a parallelizable shortest path algorithm." Journal of Algorithms, 49(1), 114-152.

5. **Wang, Y., et al.** (2017). "Gunrock: A high-performance graph processing library on the GPU." ACM SIGPLAN Notices, 52(8), 265-266.

6. **Karypis, G., & Kumar, V.** (1998). "A fast and high quality multilevel scheme for partitioning irregular graphs." SIAM Journal on scientific Computing, 20(1), 359-392.

7. **NVIDIA Corporation.** (2023). "NVLINK and NVSwitch Architecture Whitepaper."

8. **Besta, M., et al.** (2019). "Slim Graph: Practical Lossy Graph Compression for Approximate Graph Processing, Storage, and Analytics." SC19.

---

## Appendix A: Detailed Algorithm Pseudocode
## 부록 A: 상세 알고리즘 의사코드

```cpp
// MGAP-SSSP Detailed Implementation

// Phase 1: Graph Partitioning
void partition_graph(Graph& G, int k_gpus) {
    // Use METIS k-way partitioning
    idx_t nvtxs = G.n;
    idx_t ncon = 1;  // Number of constraints
    idx_t nparts = k_gpus;
    idx_t objval;  // Edge-cut value

    idx_t options[METIS_NOPTIONS];
    METIS_SetDefaultOptions(options);
    options[METIS_OPTION_OBJTYPE] = METIS_OBJTYPE_CUT;  // Minimize edge-cut
    options[METIS_OPTION_NUMBERING] = 0;  // C-style numbering

    // Partition graph
    int ret = METIS_PartGraphKway(
        &nvtxs, &ncon, G.xadj, G.adjncy,
        NULL, NULL, NULL,  // Vertex weights, sizes, edge weights
        &nparts, NULL, NULL,  // Target partition weights, ubvec
        options, &objval, G.partition
    );

    // Distribute partitions to GPUs
    for (int gpu_id = 0; gpu_id < k_gpus; ++gpu_id) {
        cudaSetDevice(gpu_id);
        build_local_graph(G, gpu_id);
        identify_boundary_vertices(G, gpu_id);
    }
}

// Phase 2: Multi-GPU SSSP with Asynchronous Pipeline
void mgap_sssp(Graph G[], int k_gpus, vertex_t source) {
    // Initialize CUDA streams for each GPU
    cudaStream_t compute_streams[k_gpus];
    cudaStream_t transfer_streams[k_gpus];
    cudaEvent_t events[k_gpus];

    for (int i = 0; i < k_gpus; ++i) {
        cudaSetDevice(i);
        cudaStreamCreate(&compute_streams[i]);
        cudaStreamCreate(&transfer_streams[i]);
        cudaEventCreate(&events[i]);

        // Enable P2P access to all other GPUs
        for (int j = 0; j < k_gpus; ++j) {
            if (i != j) cudaDeviceEnablePeerAccess(j, 0);
        }
    }

    // Initialize distances
    for (int i = 0; i < k_gpus; ++i) {
        cudaSetDevice(i);
        initialize_distances<<<blocks, threads, 0, compute_streams[i]>>>(
            G[i].d_distances, G[i].n, source
        );
    }

    bool global_changed = true;
    int iteration = 0;

    // Main iteration loop
    while (global_changed && iteration < MAX_ITER) {
        global_changed = false;

        // Stage 1: Local relaxation (parallel across GPUs)
        for (int i = 0; i < k_gpus; ++i) {
            cudaSetDevice(i);

            // Launch edge relaxation kernel
            relax_edges_kernel<<<blocks, threads, 0, compute_streams[i]>>>(
                G[i].d_row_offsets,
                G[i].d_col_indices,
                G[i].d_weights,
                G[i].d_distances,
                G[i].d_changed,
                G[i].n,
                G[i].m
            );

            // Record event for synchronization
            cudaEventRecord(events[i], compute_streams[i]);
        }

        // Stage 2: Boundary exchange (asynchronous)
        for (int i = 0; i < k_gpus; ++i) {
            cudaSetDevice(i);

            // Wait for local computation to finish
            cudaStreamWaitEvent(transfer_streams[i], events[i], 0);

            // Async transfer boundary distances to neighbors
            for (int j = 0; j < k_gpus; ++j) {
                if (i == j) continue;

                // Direct P2P copy via NVLINK (non-blocking)
                cudaMemcpyPeerAsync(
                    G[j].d_boundary_distances,  // Destination
                    j,                           // Destination device
                    G[i].d_boundary_distances,  // Source
                    i,                           // Source device
                    boundary_size * sizeof(weight_t),
                    transfer_streams[i]
                );
            }
        }

        // Stage 3: Check convergence
        for (int i = 0; i < k_gpus; ++i) {
            cudaSetDevice(i);
            cudaStreamSynchronize(compute_streams[i]);
            cudaStreamSynchronize(transfer_streams[i]);

            bool local_changed;
            cudaMemcpy(&local_changed, G[i].d_changed, sizeof(bool),
                       cudaMemcpyDeviceToHost);
            global_changed |= local_changed;
        }

        iteration++;
    }

    // Cleanup
    for (int i = 0; i < k_gpus; ++i) {
        cudaSetDevice(i);
        cudaStreamDestroy(compute_streams[i]);
        cudaStreamDestroy(transfer_streams[i]);
        cudaEventDestroy(events[i]);
    }
}

// Custom atomic operation for double precision
__device__ void atomicMinDouble(double* address, double val) {
    unsigned long long* address_as_ull = (unsigned long long*)address;
    unsigned long long old = *address_as_ull;
    unsigned long long assumed;

    do {
        assumed = old;
        old = atomicCAS(
            address_as_ull,
            assumed,
            __double_as_longlong(min(val, __longlong_as_double(assumed)))
        );
    } while (assumed != old);
}

// Edge relaxation kernel
__global__ void relax_edges_kernel(
    const uint32_t* row_offsets,
    const uint32_t* col_indices,
    const double* weights,
    double* distances,
    bool* changed,
    uint32_t n,
    uint64_t m
) {
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid >= n) return;

    double d_u = distances[tid];
    if (d_u == INFINITY) return;

    // Relax all outgoing edges from vertex tid
    for (uint64_t edge = row_offsets[tid]; edge < row_offsets[tid + 1]; ++edge) {
        uint32_t v = col_indices[edge];
        double w = weights[edge];
        double new_dist = d_u + w;

        // Atomic update with custom double atomic
        double old_dist = distances[v];
        if (new_dist < old_dist) {
            atomicMinDouble(&distances[v], new_dist);
            *changed = true;
        }
    }
}
```

---

**End of Research Proposal**
**연구 제안서 종료**

---

**Next Steps:**
1. Review and approve this proposal
2. Proceed with implementation (Phases 1-6)
3. Continuous verification at each milestone
4. Deliver complete paper package (code + datasets + paper + reproducibility scripts)

**다음 단계:**
1. 본 제안서 검토 및 승인
2. 구현 진행 (1-6단계)
3. 각 마일스톤에서 지속적 검증
4. 완전한 논문 패키지 전달 (코드 + 데이터셋 + 논문 + 재현성 스크립트)
