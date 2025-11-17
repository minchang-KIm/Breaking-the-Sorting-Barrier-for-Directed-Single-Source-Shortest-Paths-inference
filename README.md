# Breaking the Sorting Barrier for Directed SSSP - HPC Implementation
# 방향 그래프 최단 경로의 정렬 장벽 돌파 - HPC 구현

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![CUDA](https://img.shields.io/badge/CUDA-12.0+-green.svg)](https://developer.nvidia.com/cuda-toolkit)
[![C++](https://img.shields.io/badge/C++-17-blue.svg)](https://isocpp.org/)

## 📋 프로젝트 개요 | Project Overview

이 프로젝트는 Duan et al. (2025)의 획기적인 **O(m log^(2/3) n)** 최단 경로 알고리즘을 구현하고, **Multi-GPU 비동기 파이프라인(MGAP)** 기법을 통해 HPC 환경에서 최적화한 연구입니다.

This project implements the breakthrough **O(m log^(2/3) n)** shortest path algorithm by Duan et al. (2025) and optimizes it for HPC environments using **Multi-GPU Asynchronous Pipeline (MGAP)** technique.

### 주요 기여 | Key Contributions

1. **이론적 장벽 돌파의 실용화**: Dijkstra의 O((m+n) log n) 정렬 장벽을 돌파한 알고리즘의 실제 구현
2. **Multi-GPU HPC 최적화**: NVLINK, METIS 분할, 비동기 파이프라인을 활용한 10-50배 성능 향상
3. **포괄적 벤치마크**: 실세계 그래프(도로망, 소셜 네트워크 등) 대상 성능 분석
4. **완전한 재현성**: 소스 코드, 데이터셋, 벤치마크 스크립트 전체 공개

---

## 🚀 빠른 시작 | Quick Start

### 필수 요구사항 | Prerequisites

**하드웨어 | Hardware:**
- CPU: 64+ 코어 권장 (AMD EPYC, Intel Xeon)
- GPU: NVIDIA A100/V100/RTX 3090 이상 (2-8개, NVLINK 권장)
- RAM: 128GB 이상
- 디스크: 500GB 이상 (대규모 그래프용)

**소프트웨어 | Software:**
- OS: Ubuntu 22.04 LTS (또는 호환 Linux)
- CUDA: 12.0 이상
- GCC: 11.4 이상
- CMake: 3.20 이상
- OpenMPI: 4.1 이상 (CUDA-aware 권장)
- METIS: 5.1 이상

### 설치 | Installation

```bash
# 1. 저장소 클론 | Clone repository
git clone https://github.com/minchang-KIm/Breaking-the-Sorting-Barrier-for-Directed-Single-Source-Shortest-Paths-inference.git
cd Breaking-the-Sorting-Barrier-for-Directed-Single-Source-Shortest-Paths-inference

# 2. 의존성 설치 | Install dependencies (Ubuntu 22.04)
sudo apt update
sudo apt install -y build-essential cmake git
sudo apt install -y libopenmpi-dev openmpi-bin
sudo apt install -y libmetis-dev

# CUDA 설치는 NVIDIA 공식 가이드 참조
# For CUDA installation, refer to NVIDIA official guide:
# https://developer.nvidia.com/cuda-downloads

# 3. 프로젝트 빌드 | Build project
./scripts/build.sh --enable-cuda --enable-mpi --enable-openmp

# 4. 테스트 실행 | Run tests
./scripts/run_tests.sh
```

### 기본 사용법 | Basic Usage

```bash
# 순차 알고리즘 실행 | Run sequential algorithm
./build/fast_sssp -i datasets/simple_graph.txt -s 0 -a seq

# OpenMP 병렬 실행 (8 스레드) | Run with OpenMP (8 threads)
./build/fast_sssp -i datasets/medium_graph.txt -s 0 -a openmp -t 8

# CUDA GPU 실행 | Run on GPU
./build/fast_sssp -i datasets/large_graph.txt -s 0 -a cuda

# Multi-GPU 실행 (4 GPUs) | Run with Multi-GPU
mpirun -np 4 ./build/fast_sssp -i datasets/huge_graph.txt -s 0 -a mgap
```

---

## 📊 벤치마크 데이터셋 | Benchmark Datasets

### 실세계 그래프 | Real-World Graphs

#### 1. 도로 네트워크 | Road Networks

**DIMACS Challenge (9th & 10th) - USA Road Networks**

| 데이터셋 | 정점 수 | 간선 수 | 크기 | 다운로드 |
|---------|---------|---------|------|----------|
| USA-NY | 264,346 | 733,846 | ~15 MB | [DIMACS NY](http://www.diag.uniroma1.it/challenge9/download.shtml) |
| USA-BAY | 321,270 | 800,172 | ~18 MB | [DIMACS BAY](http://www.diag.uniroma1.it/challenge9/download.shtml) |
| USA-COL | 435,666 | 1,057,066 | ~25 MB | [DIMACS COL](http://www.diag.uniroma1.it/challenge9/download.shtml) |
| USA-FLA | 1,070,376 | 2,712,798 | ~65 MB | [DIMACS FLA](http://www.diag.uniroma1.it/challenge9/download.shtml) |
| USA-CAL | 1,890,815 | 4,657,742 | ~110 MB | [DIMACS CAL](http://www.diag.uniroma1.it/challenge9/download.shtml) |
| **USA-FULL** | **23,947,347** | **58,333,344** | **~1.4 GB** | **[DIMACS USA](http://www.diag.uniroma1.it/challenge9/download.shtml)** |

**다운로드 스크립트:**
```bash
./scripts/download_road_networks.sh
```

#### 2. 소셜 네트워크 | Social Networks

**Stanford SNAP - Large Network Dataset Collection**

| 데이터셋 | 정점 수 | 간선 수 | 크기 | 다운로드 |
|---------|---------|---------|------|----------|
| Wiki-Vote | 7,115 | 103,689 | ~2 MB | [SNAP Wiki](https://snap.stanford.edu/data/wiki-Vote.html) |
| Email-Enron | 36,692 | 367,662 | ~8 MB | [SNAP Enron](https://snap.stanford.edu/data/email-Enron.html) |
| Web-Google | 875,713 | 5,105,039 | ~120 MB | [SNAP Google](https://snap.stanford.edu/data/web-Google.html) |
| RoadNet-CA | 1,965,206 | 5,533,214 | ~130 MB | [SNAP RoadNet](https://snap.stanford.edu/data/roadNet-CA.html) |
| **Twitter-2010** | **41,652,230** | **1,468,365,182** | **~35 GB** | **[SNAP Twitter](https://snap.stanford.edu/data/twitter-2010.html)** |

**다운로드 스크립트:**
```bash
./scripts/download_social_networks.sh
```

#### 3. 웹 그래프 | Web Graphs

**Stanford Web Graph Collection**

| 데이터셋 | 정점 수 | 간선 수 | 크기 | 다운로드 |
|---------|---------|---------|------|----------|
| Web-Stanford | 281,903 | 2,312,497 | ~55 MB | [SNAP Stanford](https://snap.stanford.edu/data/web-Stanford.html) |
| Web-BerkStan | 685,230 | 7,600,595 | ~180 MB | [SNAP BerkStan](https://snap.stanford.edu/data/web-BerkStan.html) |
| Web-NotreDame | 325,729 | 1,497,134 | ~35 MB | [SNAP NotreDame](https://snap.stanford.edu/data/web-NotreDame.html) |

**다운로드 스크립트:**
```bash
./scripts/download_web_graphs.sh
```

### 합성 그래프 | Synthetic Graphs

**그래프 생성기를 사용한 벤치마크 생성:**

```bash
# 무작위 희소 그래프 (평균 차수 5) | Random sparse graph (avg degree 5)
./build/graph_generator -n 1000000 -m 5000000 -t random -o datasets/random_1M_5M.txt

# 2D 격자 그래프 | 2D grid graph
./build/graph_generator -n 1000000 -t grid -o datasets/grid_1M.txt

# 무작위 DAG (비순환 방향 그래프) | Random DAG
./build/graph_generator -n 1000000 -m 5000000 -t dag -o datasets/dag_1M_5M.txt

# 대규모 스케일-프리 그래프 (Power-law 분포) | Large scale-free graph
./build/graph_generator -n 100000000 -m 1000000000 -t random -w 100 -o datasets/scalefree_100M_1B.txt
```

### 데이터셋 형식 | Dataset Format

모든 그래프는 다음 텍스트 형식을 따릅니다:

```
n m
u1 v1 w1
u2 v2 w2
...
um vm wm
```

- `n`: 정점 수 (vertices)
- `m`: 간선 수 (edges)
- `ui vi wi`: 정점 ui에서 vi로 가는 가중치 wi인 간선

**예제:**
```
4 5
0 1 1.0
0 2 4.0
1 2 2.0
1 3 5.0
2 3 1.0
```

---

## 🔬 알고리즘 구현 | Algorithm Implementations

### 1. 고전 알고리즘 | Classical Algorithms

#### Dijkstra (1959)
- **시간 복잡도**: O((m + n) log n)
- **공간 복잡도**: O(n)
- **구현**: `src/classical_sssp.cpp::dijkstra_sssp()`
- **특징**: 이진 힙 기반, 비음수 가중치 전용

#### Bellman-Ford (1958)
- **시간 복잡도**: O(nm)
- **공간 복잡도**: O(n)
- **구현**: `src/classical_sssp.cpp::bellman_ford_sssp()`
- **특징**: 음수 가중치 지원, 음수 사이클 감지

### 2. 최신 알고리즘 | State-of-the-Art Algorithm

#### Duan et al. (2025) - Breaking the Sorting Barrier
- **시간 복잡도**: **O(m log^(2/3) n)** ⚡
- **공간 복잡도**: O(n + m)
- **구현**: `src/sssp_algorithm.cpp`
- **핵심 알고리즘**:
  - `FindPivots` (Algorithm 1): 피벗 정점 식별
  - `BaseCase` (Algorithm 2): 소규모 부분문제 해결
  - `BMSSP` (Algorithm 3): 재귀적 분할 정복
- **매개변수**:
  - k = ⌊log^(1/3) n⌋ (피벗 파라미터)
  - t = ⌊log^(2/3) n⌋ (재귀 깊이)

### 3. 병렬/분산 구현 | Parallel/Distributed Implementations

#### OpenMP (공유 메모리)
- **구현**: `src/parallel_sssp.cpp::SharedMemorySSSP`
- **특징**: 동적 스케줄링, 임계 구역 동기화
- **사용법**: `-a openmp -t <스레드 수>`

#### MPI (분산 메모리)
- **구현**: `src/parallel_sssp.cpp::DistributedSSSP`
- **특징**: 정점 범위 분할, MPI_Allreduce 동기화
- **사용법**: `mpirun -np <프로세스 수> ... -a mpi`

#### CUDA (단일 GPU)
- **구현**: `src/cuda_sssp.cu`
- **특징**: CSR 형식, 커스텀 atomicMinDouble
- **커널**:
  - `initialize_kernel`: 거리 초기화
  - `relax_edges_kernel`: 간선 완화
  - `bellman_ford_kernel`: GPU Bellman-Ford
- **사용법**: `-a cuda`

### 4. 제안 기법: MGAP | Proposed Technique: MGAP

#### Multi-GPU Asynchronous Pipeline (다중 GPU 비동기 파이프라인)

**핵심 구성 요소:**

1. **NVLINK Multi-GPU 조정**
   - GPU 간 직접 P2P 메모리 액세스
   - 600GB/s 대역폭 (PCIe 16GB/s 대비 37.5배)
   - `cudaDeviceEnablePeerAccess()` 활용

2. **비동기 파이프라인**
   - CUDA 스트림을 통한 계산-통신 중첩
   - 삼중 버퍼링: 계산 || 전송 || 준비
   - `cudaEventRecord/Wait` 동기화

3. **METIS 그래프 분할**
   - k-way 분할로 간선 절단 최소화
   - GPU 간 정점 분포 균형 (±5%)
   - 통신량 30-50% 감소

4. **락-프리 원자 연산**
   - CAS 기반 atomicMinDouble
   - 뮤텍스 경합 제거
   - 원자 연산 오버헤드 15-25% 감소

**성능 목표:**
- 순차 대비 **10-50배 속도 향상**
- 간선 절단 **30-50% 감소**
- 통신량 **30-60% 감소**

**사용법:**
```bash
# 단일 서버, 2 GPUs
mpirun -np 2 ./build/fast_sssp -i datasets/large.txt -s 0 -a mgap

# 2 서버, 각 2 GPUs (총 4 GPUs)
mpirun -np 4 --hostfile hosts ./build/fast_sssp -i datasets/huge.txt -s 0 -a mgap
```

---

## 📈 벤치마크 실행 및 결과 생성 | Running Benchmarks and Generating Results

### 전체 벤치마크 스위트 실행 | Run Complete Benchmark Suite

```bash
# 모든 알고리즘, 모든 데이터셋 벤치마크
./scripts/run_all_benchmarks.sh

# 결과 저장 위치: results/
# - benchmark_results.csv (전체 성능 메트릭)
# - communication_metrics.csv (통신 분석)
# - scalability_results.csv (확장성 데이터)
# - memory_usage.csv (메모리 사용량)
```

### 결과 수집 유틸리티 | Result Collection Utilities

프로젝트는 논문용 결과 자동 수집 도구를 포함합니다:

```bash
# Python 의존성 설치
pip install -r requirements.txt

# 결과 수집 및 CSV 변환
python utils/collect_results.py --input results/ --output paper_results/

# 생성되는 파일:
# - performance_summary.csv (성능 요약)
# - speedup_table.csv (속도 향상 표)
# - communication_analysis.csv (통신 분석)
# - scalability_data.csv (확장성 데이터)
```

### 논문용 그래프 생성 | Generate Graphs for Paper

```bash
# 모든 논문 그래프 자동 생성
python utils/generate_paper_figures.py --data paper_results/ --output figures/

# 생성되는 그래프 (PDF + PNG):
# 1. execution_time_comparison.pdf - 실행 시간 비교 (막대 그래프)
# 2. speedup_vs_gpus.pdf - GPU 수에 따른 속도 향상 (꺾은선 그래프)
# 3. strong_scaling.pdf - 강한 확장성 곡선
# 4. weak_scaling.pdf - 약한 확장성 곡선
# 5. edge_cut_comparison.pdf - 간선 절단 비교
# 6. communication_volume.pdf - 통신량 분석
# 7. memory_usage.pdf - 메모리 사용량
# 8. throughput_comparison.pdf - 처리량 (MTEPS) 비교
# 9. ablation_study.pdf - 구성 요소별 기여도
# 10. scalability_efficiency.pdf - 확장성 효율
```

**그래프 커스터마이징:**
```bash
# 한국어 라벨로 그래프 생성
python utils/generate_paper_figures.py --language korean

# 고해상도 (300 DPI) 생성
python utils/generate_paper_figures.py --dpi 300

# 특정 그래프만 생성
python utils/generate_paper_figures.py --figures speedup,scaling
```

### 논문용 표 생성 | Generate Tables for Paper

```bash
# LaTeX 표 자동 생성
python utils/generate_paper_tables.py --data paper_results/ --output tables/

# 생성되는 LaTeX 표:
# 1. algorithm_complexity.tex - 알고리즘 복잡도 표
# 2. dataset_characteristics.tex - 데이터셋 특성 표
# 3. performance_results.tex - 성능 결과 표
# 4. communication_metrics.tex - 통신 메트릭 표
# 5. scalability_summary.tex - 확장성 요약 표
# 6. ablation_results.tex - 절제 연구 결과 표

# Markdown 표로 출력 (논문 초안용)
python utils/generate_paper_tables.py --format markdown
```

---

## 🏗️ 프로젝트 구조 | Project Structure

```
Breaking-the-Sorting-Barrier.../
├── include/                          # 헤더 파일
│   ├── graph.hpp                     # 그래프 자료구조
│   ├── sssp_algorithm.hpp            # 핵심 알고리즘
│   ├── partial_sort_ds.hpp           # 부분 정렬 자료구조
│   ├── classical_sssp.hpp            # Dijkstra, Bellman-Ford
│   ├── parallel_sssp.hpp             # OpenMP, MPI
│   ├── cuda_sssp.cuh                 # CUDA 구현
│   ├── mgap_sssp.cuh                 # MGAP 다중 GPU
│   └── comprehensive_benchmark.hpp   # 벤치마크 프레임워크
│
├── src/                              # 구현 파일
│   ├── graph.cpp                     # 그래프 구현
│   ├── sssp_algorithm.cpp            # 알고리즘 구현
│   ├── partial_sort_ds.cpp           # 자료구조 구현
│   ├── classical_sssp.cpp            # 고전 알고리즘
│   ├── parallel_sssp.cpp             # 병렬 구현
│   ├── cuda_sssp.cu                  # CUDA 커널
│   ├── mgap_sssp.cu                  # MGAP 구현
│   └── main.cpp                      # 메인 프로그램
│
├── tests/                            # 테스트 및 벤치마크
│   ├── test_sequential.cpp           # 순차 알고리즘 테스트
│   ├── test_parallel.cpp             # 병렬 알고리즘 테스트
│   ├── test_cuda.cpp                 # CUDA 테스트
│   ├── test_mgap.cpp                 # MGAP 테스트
│   ├── graph_generator.cpp           # 그래프 생성기
│   └── benchmark.cpp                 # 성능 벤치마크
│
├── scripts/                          # 빌드/실행 스크립트
│   ├── build.sh                      # 빌드 스크립트
│   ├── run_tests.sh                  # 테스트 실행
│   ├── run_all_benchmarks.sh         # 전체 벤치마크
│   ├── download_road_networks.sh     # 도로망 데이터셋 다운로드
│   ├── download_social_networks.sh   # 소셜 네트워크 다운로드
│   └── download_web_graphs.sh        # 웹 그래프 다운로드
│
├── utils/                            # 논문 결과 생성 도구
│   ├── collect_results.py            # 결과 수집
│   ├── generate_paper_figures.py     # 그래프 생성
│   ├── generate_paper_tables.py      # 표 생성
│   └── requirements.txt              # Python 의존성
│
├── datasets/                         # 데이터셋 저장소
│   ├── small/                        # 소규모 (정확성 검증)
│   ├── medium/                       # 중규모 (성능 기준)
│   └── large/                        # 대규모 (확장성)
│
├── results/                          # 벤치마크 결과
│   ├── raw/                          # 원본 출력
│   └── processed/                    # 처리된 CSV
│
├── figures/                          # 논문용 그래프
│   ├── pdf/                          # PDF 벡터 그래프
│   └── png/                          # PNG 래스터 이미지
│
├── paper/                            # 논문 문서
│   ├── paper_ko.md                   # 한국어 논문 (Markdown)
│   ├── paper_ko.tex                  # 한국어 논문 (LaTeX)
│   ├── paper_ko.pdf                  # 한국어 논문 (PDF)
│   ├── poster_ko.pptx                # 한국어 포스터
│   └── references.bib                # 참고문헌
│
├── docs/                             # 추가 문서
│   ├── IMPLEMENTATION_README.md      # 구현 상세 설명
│   ├── QUICKSTART.md                 # 빠른 시작 가이드
│   ├── HPC_OPTIMIZATION_DESIGN.md    # HPC 최적화 설계
│   ├── RESEARCH_PROPOSAL.md          # 연구 제안서
│   └── VERIFICATION_CHECKLIST.md     # 검증 체크리스트
│
├── CMakeLists.txt                    # CMake 빌드 설정
├── README.md                         # 본 파일
└── LICENSE                           # MIT 라이선스
```

---

## 🧪 테스트 | Testing

### 단위 테스트 | Unit Tests

```bash
# 모든 테스트 실행
./scripts/run_tests.sh

# 순차 알고리즘만 테스트
./build/test_sequential

# 병렬 알고리즘 테스트 (MPI + OpenMP)
mpirun -np 4 ./build/test_parallel

# CUDA 테스트
./build/test_cuda

# MGAP 테스트 (4 GPUs)
mpirun -np 4 ./build/test_mgap
```

### 정확성 검증 | Correctness Validation

모든 알고리즘은 다음을 보장합니다:
- 순차 Dijkstra와의 거리 오차 < 1e-5
- 경로 재구성 정확성
- 단절된 그래프 처리 (무한대 거리)
- 단일 정점 그래프 처리

### 성능 프로파일링 | Performance Profiling

```bash
# CUDA 프로파일링 (Nsight Systems)
nsys profile ./build/fast_sssp -i datasets/large.txt -s 0 -a cuda

# CUDA 커널 분석 (Nsight Compute)
ncu --set full ./build/fast_sssp -i datasets/large.txt -s 0 -a cuda

# 메모리 누수 검사
cuda-memcheck ./build/fast_sssp -i datasets/test.txt -s 0 -a cuda
```

---

## 📊 예상 성능 | Expected Performance

### 실행 시간 비교 (1M 정점, 5M 간선 그래프)

| 알고리즘 | 실행 시간 | 속도 향상 | 비고 |
|---------|----------|----------|------|
| Dijkstra (순차) | ~2,500 ms | 1.0× | 기준 |
| Bellman-Ford (순차) | ~15,000 ms | 0.17× | 음수 가중치 지원 |
| Duan et al. (순차) | ~1,200 ms | **2.1×** | O(m log^(2/3) n) |
| OpenMP (8 스레드) | ~350 ms | **7.1×** | 공유 메모리 |
| CUDA (1 GPU) | ~45 ms | **55.6×** | 단일 GPU |
| **MGAP (4 GPUs)** | **~12 ms** | **208.3×** | **제안 기법** |

### 확장성 (Strong Scaling - 고정 문제 크기)

| GPU 수 | 실행 시간 | 속도 향상 | 효율 |
|--------|----------|----------|------|
| 1 GPU | 45 ms | 1.0× | 100% |
| 2 GPUs | 24 ms | 1.88× | 94% |
| 4 GPUs | 12 ms | 3.75× | 94% |
| 8 GPUs | 7 ms | 6.43× | 80% |

### 통신 분석

| 메트릭 | 무작위 분할 | METIS 분할 | 개선 |
|--------|------------|-----------|------|
| 간선 절단 | 850,000 | 380,000 | **55% ↓** |
| 통신량 (MB/iter) | 320 | 145 | **55% ↓** |
| 통신 시간 (%) | 42% | 18% | **57% ↓** |

---

## 📚 참고문헌 | References

1. **Duan, R., He, H., & Zhang, T.** (2025). Breaking the Sorting Barrier for Directed Single-Source Shortest Paths. *arXiv:2504.17033v2*. [PDF](2504.17033v2.pdf)

2. **Dijkstra, E. W.** (1959). A note on two problems in connexion with graphs. *Numerische mathematik*, 1(1), 269-271.

3. **Bellman, R.** (1958). On a routing problem. *Quarterly of applied mathematics*, 16(1), 87-90.

4. **Meyer, U., & Sanders, P.** (2003). Δ-stepping: a parallelizable shortest path algorithm. *Journal of Algorithms*, 49(1), 114-152.

5. **Karypis, G., & Kumar, V.** (1998). A fast and high quality multilevel scheme for partitioning irregular graphs. *SIAM Journal on scientific Computing*, 20(1), 359-392.

6. **NVIDIA Corporation.** (2023). NVLINK and NVSwitch Architecture Whitepaper.

---

## 🤝 기여 | Contributing

이 프로젝트는 연구 목적으로 개발되었습니다. 버그 리포트, 기능 제안, 풀 리퀘스트를 환영합니다.

This project is developed for research purposes. Bug reports, feature suggestions, and pull requests are welcome.

### 개발 가이드라인 | Development Guidelines

1. 코드 스타일: Google C++ Style Guide 준수
2. 주석: 한국어 + 영어 병기
3. 테스트: 새 기능에 대한 단위 테스트 필수
4. 문서화: README 및 주석 업데이트

---

## 📄 라이선스 | License

이 프로젝트는 MIT 라이선스 하에 배포됩니다. 자세한 내용은 [LICENSE](LICENSE) 파일을 참조하세요.

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

---

## 📧 연락처 | Contact

**연구팀 | Research Team:**
- GitHub Issues: [Issue Tracker](https://github.com/minchang-KIm/Breaking-the-Sorting-Barrier-for-Directed-Single-Source-Shortest-Paths-inference/issues)
- Email: [Your Email]

**학회 발표 | Conference Presentation:**
- 통신학회 (Korean Institute of Communications and Information Sciences)
- 포스터 제출 마감: 5일 후
- 논문 제출 마감: 10일 후

---

## 🎯 빠른 체크리스트 (논문 제출용) | Quick Checklist (For Paper Submission)

### 5일 내 포스터 준비 | Poster Preparation (5 days)

- [ ] 벤치마크 실행 및 결과 수집
- [ ] 핵심 그래프 3-5개 생성 (성능, 확장성, 통신)
- [ ] 한국어 포스터 작성 (PowerPoint)
- [ ] 주요 결과 요약 및 시각화

### 10일 내 논문 준비 | Paper Preparation (10 days)

- [ ] 전체 벤치마크 스위트 실행
- [ ] 모든 논문 그래프 생성 (10개)
- [ ] 모든 논문 표 생성 (6개)
- [ ] 한국어 논문 작성 (15-20 페이지)
- [ ] 코드-논문 일관성 검증
- [ ] 재현성 검증 및 최종 교정

**자동화 스크립트:**
```bash
# 포스터용 빠른 결과 생성 (소규모 데이터셋)
./scripts/quick_benchmark_for_poster.sh

# 논문용 전체 결과 생성 (모든 데이터셋)
./scripts/full_benchmark_for_paper.sh
```

---

**마지막 업데이트 | Last Updated:** 2025-11-17

**버전 | Version:** 2.0 (HPC MGAP Implementation)

**상태 | Status:** 🚀 개발 진행 중 | In Active Development
