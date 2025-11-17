# Breaking the Sorting Barrier for Directed Single-Source Shortest Paths

[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![C++17](https://img.shields.io/badge/C++-17-blue.svg)](https://isocpp.org/)
[![CUDA](https://img.shields.io/badge/CUDA-11.0+-green.svg)](https://developer.nvidia.com/cuda-toolkit)

High-performance implementation of the groundbreaking **O(m log^(2/3) n)** algorithm for single-source shortest paths (SSSP) on directed graphs, from the paper by Duan et al. (2025).

This is the **first algorithm to break the O(m + n log n) sorting barrier** for SSSP on directed graphs with real non-negative edge weights!

## 🚀 Quick Start

```bash
# Build with MPI + OpenMP support
./scripts/build.sh

# Generate a test graph
mkdir -p data
./build/graph_generator -n 10000 -m 50000 -t random -o data/graph.txt

# Run sequential algorithm
./build/fast_sssp -i data/graph.txt -s 0 -m seq -v

# Run parallel (4 threads)
./build/fast_sssp -i data/graph.txt -s 0 -m openmp -t 4 -v

# Run benchmark
./scripts/run_benchmark.sh -n 10000 -m 50000
```

📖 **[Full Quick Start Guide →](QUICKSTART.md)**

## 🌟 Key Features

### Multiple Implementation Variants
- **Sequential**: Reference implementation following the paper
- **MPI Distributed**: Scale across multiple nodes in a cluster
- **OpenMP Parallel**: Multi-threaded execution on multi-core CPUs
- **CUDA GPU**: Massively parallel execution on NVIDIA GPUs

### Performance Breakthrough
- **Theoretical**: O(m log^(2/3) n) time complexity
- **Practical**: Significant speedup over Dijkstra for sparse graphs
- **Parallel Efficiency**: Near-linear speedup with multiple cores/nodes

### Complete Toolkit
- Graph generators (random, grid, DAG)
- Comprehensive test suite
- Performance benchmarking tools
- Automated build scripts

## 📊 Algorithm Overview

The algorithm uses a novel divide-and-conquer approach that merges:
- **Dijkstra's algorithm**: Priority queue-based exploration
- **Bellman-Ford algorithm**: DP-based relaxation

Key innovation: **Frontier reduction** via pivot selection reduces the effective problem size by factor of log^(Ω(1))(n), enabling sublinear time per vertex.

### Complexity Comparison

| Algorithm | Time Complexity | Space | Graph Type |
|-----------|----------------|-------|------------|
| **This Work** | **O(m log^(2/3) n)** | O(n+m) | Directed, real weights |
| Dijkstra | O(m + n log n) | O(n+m) | Directed, real weights |
| Thorup 2004 | O(m + n log log C) | O(n+m) | Directed, integer weights |
| DMSY 2023 | O(m √log n log log n) | O(n+m) | Undirected, real weights |

## 🏗️ Project Structure

```
├── include/               # Header files
│   ├── graph.hpp         # Graph data structures
│   ├── sssp_algorithm.hpp    # Core SSSP algorithm
│   ├── partial_sort_ds.hpp   # Partial sorting structure (Lemma 3.3)
│   ├── parallel_sssp.hpp     # MPI + OpenMP implementations
│   └── cuda_sssp.cuh         # CUDA GPU kernels
├── src/                  # Implementation files
│   ├── graph.cpp
│   ├── sssp_algorithm.cpp    # Algorithms 1-3 from paper
│   ├── partial_sort_ds.cpp
│   ├── parallel_sssp.cpp
│   ├── cuda_sssp.cu
│   └── main.cpp
├── tests/                # Test programs
│   ├── test_sequential.cpp
│   ├── test_parallel.cpp
│   ├── test_cuda.cpp
│   ├── graph_generator.cpp
│   └── benchmark.cpp
├── scripts/              # Build and run scripts
│   ├── build.sh
│   ├── run_tests.sh
│   └── run_benchmark.sh
├── CMakeLists.txt        # Build configuration
├── README.md             # This file
├── QUICKSTART.md         # Quick start guide
└── IMPLEMENTATION_README.md  # Detailed documentation
```

## 🔧 Requirements

### Minimum
- C++17 compatible compiler (GCC 7+, Clang 5+, MSVC 2017+)
- CMake 3.18+

### Optional (for parallel versions)
- **MPI**: OpenMPI 3.0+ or MPICH 3.2+
- **OpenMP**: Usually bundled with compilers
- **CUDA**: CUDA Toolkit 11.0+ (for GPU version)

## 📦 Installation

### Clone Repository
```bash
git clone https://github.com/minchang-KIm/Breaking-the-Sorting-Barrier-for-Directed-Single-Source-Shortest-Paths-inference.git
cd Breaking-the-Sorting-Barrier-for-Directed-Single-Source-Shortest-Paths-inference
```

### Build Options

**Sequential only:**
```bash
./scripts/build.sh --no-mpi --no-openmp
```

**With MPI + OpenMP (recommended):**
```bash
./scripts/build.sh
```

**With CUDA support:**
```bash
./scripts/build.sh --cuda
```

**All features:**
```bash
./scripts/build.sh --cuda
```

## 🧪 Testing

```bash
# Run all tests
./scripts/run_tests.sh

# Run specific tests
./build/test_sequential
mpirun -np 4 ./build/test_parallel
./build/test_cuda
```

## 📈 Benchmarking

```bash
# Benchmark on random graph
./scripts/run_benchmark.sh -n 10000 -m 50000 -t random

# Benchmark on grid graph
./scripts/run_benchmark.sh -n 10000 -t grid

# Benchmark on DAG
./scripts/run_benchmark.sh -n 10000 -m 50000 -t dag
```

Example output:
```
============================================================
Benchmark Results
============================================================

Implementation                      Time (ms)         Status
------------------------------------------------------------
Sequential                            1245.32             OK
OpenMP (2 threads)                     640.55             OK
OpenMP (4 threads)                     335.21             OK
OpenMP (8 threads)                     180.44             OK
CUDA GPU                                45.23             OK
============================================================

Speedups (relative to Sequential):
  OpenMP (2 threads): 1.94x
  OpenMP (4 threads): 3.71x
  OpenMP (8 threads): 6.90x
  CUDA GPU: 27.53x
```

## 📚 Documentation

- **[Quick Start Guide](QUICKSTART.md)** - Get running in 5 minutes
- **[Implementation Details](IMPLEMENTATION_README.md)** - Comprehensive technical documentation
- **[Original Paper](2504.17033v2.pdf)** - Research paper with algorithm details

## 🎯 Use Cases

This implementation is ideal for:
- **Large-scale graph analytics** - Road networks, social networks
- **Scientific computing** - Molecular dynamics, physics simulations
- **Optimization problems** - Routing, scheduling, resource allocation
- **Machine learning** - Graph neural networks, shortest path features
- **Research** - Algorithm comparison, performance analysis

## 🔬 Algorithm Details

### Three Core Algorithms

1. **FindPivots** (Algorithm 1): Reduces frontier size by identifying pivot vertices
2. **BaseCase** (Algorithm 2): Mini-Dijkstra for small subproblems
3. **BMSSP** (Algorithm 3): Bounded multi-source shortest path solver

### Key Parameters
- **k** = ⌊log^(1/3) n⌋ - Pivot selection granularity
- **t** = ⌊log^(2/3) n⌋ - Recursion depth control

### Time Complexity Breakdown
```
T(n,m) = O((k + t²/k + t)(log n)/t · n + (t + (log n)·log k) · m)
       = O(m log^(2/3) n)  [with k = log^(1/3) n, t = log^(2/3) n]
```

## 🤝 Contributing

Contributions are welcome! Please feel free to submit pull requests or open issues for:
- Bug fixes
- Performance improvements
- Additional parallel implementations
- Documentation enhancements
- New test cases

## 📝 Citation

If you use this implementation in your research, please cite:

```bibtex
@inproceedings{duan2025breaking,
  title={Breaking the Sorting Barrier for Directed Single-Source Shortest Paths},
  author={Duan, Ran and Mao, Jiayi and Mao, Xiao and Shu, Xinkai and Yin, Longhui},
  booktitle={arXiv:2504.17033v2 [cs.DS]},
  year={2025}
}
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Original algorithm by Duan, Mao, Mao, Shu, and Yin
- MPI, OpenMP, and CUDA communities for parallel computing frameworks
- CMake for build system support

## 📞 Contact

For questions, issues, or collaboration:
- Open an issue on GitHub
- See [IMPLEMENTATION_README.md](IMPLEMENTATION_README.md) for detailed technical questions

---

**Made with ❤️ for the graph algorithms community**

*Breaking barriers, one log factor at a time* 🚀
