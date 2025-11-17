# Fast SSSP: Breaking the Sorting Barrier for Directed Single-Source Shortest Paths

This repository contains a high-performance implementation of the algorithm from the paper "Breaking the Sorting Barrier for Directed Single-Source Shortest Paths" (Duan et al., 2025).

## Algorithm Overview

The algorithm achieves **O(m log^(2/3) n)** time complexity, breaking the classical O(m + n log n) bound of Dijkstra's algorithm for the first time on sparse directed graphs with real non-negative edge weights.

### Key Features

- **Sequential Implementation**: Core algorithm from the paper
- **MPI Distributed**: Multi-node parallelization using MPI
- **OpenMP Shared-Memory**: Multi-threaded parallelization within a single node
- **CUDA GPU**: GPU-accelerated version for massive parallelism

## Project Structure

```
.
├── include/              # Header files
│   ├── graph.hpp        # Graph data structures
│   ├── sssp_algorithm.hpp   # Core SSSP algorithm
│   ├── partial_sort_ds.hpp  # Partial sorting data structure
│   ├── parallel_sssp.hpp    # MPI + OpenMP implementations
│   └── cuda_sssp.cuh        # CUDA GPU implementation
├── src/                 # Source files
│   ├── graph.cpp
│   ├── sssp_algorithm.cpp
│   ├── partial_sort_ds.cpp
│   ├── parallel_sssp.cpp
│   ├── cuda_sssp.cu
│   └── main.cpp         # Main executable
├── tests/               # Test programs
│   ├── test_sequential.cpp
│   ├── test_parallel.cpp
│   ├── test_cuda.cpp
│   ├── graph_generator.cpp
│   └── benchmark.cpp
├── scripts/             # Build and run scripts
└── CMakeLists.txt       # CMake build configuration
```

## Requirements

### Minimum Requirements

- C++17 compatible compiler (GCC 7+, Clang 5+, or MSVC 2017+)
- CMake 3.18+

### Optional Requirements

- **MPI**: OpenMPI 3.0+ or MPICH 3.2+ (for distributed parallelization)
- **OpenMP**: Usually bundled with modern compilers (for shared-memory parallelization)
- **CUDA**: CUDA Toolkit 11.0+ (for GPU acceleration)

## Building

### Basic Build (Sequential Only)

```bash
mkdir build
cd build
cmake .. -DENABLE_MPI=OFF -DENABLE_OPENMP=OFF -DENABLE_CUDA=OFF
make -j$(nproc)
```

### Build with MPI + OpenMP

```bash
mkdir build
cd build
cmake .. -DENABLE_MPI=ON -DENABLE_OPENMP=ON -DENABLE_CUDA=OFF
make -j$(nproc)
```

### Build with CUDA

```bash
mkdir build
cd build
cmake .. -DENABLE_MPI=OFF -DENABLE_OPENMP=OFF -DENABLE_CUDA=ON
make -j$(nproc)
```

### Build All Features

```bash
mkdir build
cd build
cmake .. -DENABLE_MPI=ON -DENABLE_OPENMP=ON -DENABLE_CUDA=ON
make -j$(nproc)
```

## Usage

### Generate Test Graphs

```bash
# Random graph: 10,000 vertices, 50,000 edges
./graph_generator -n 10000 -m 50000 -t random -o data/random_10k.txt

# 2D Grid graph: ~10,000 vertices (100x100 grid)
./graph_generator -n 10000 -t grid -o data/grid_10k.txt

# Random DAG: 10,000 vertices, 50,000 edges
./graph_generator -n 10000 -m 50000 -t dag -o data/dag_10k.txt
```

### Run Sequential SSSP

```bash
./fast_sssp -i data/random_10k.txt -s 0 -m seq -o data/distances.txt -v
```

### Run with OpenMP (4 threads)

```bash
./fast_sssp -i data/random_10k.txt -s 0 -m openmp -t 4 -o data/distances.txt -v
```

### Run with MPI (4 processes)

```bash
mpirun -np 4 ./fast_sssp -i data/random_10k.txt -s 0 -m mpi -o data/distances.txt -v
```

### Run with CUDA

```bash
./fast_sssp -i data/random_10k.txt -s 0 -m cuda -o data/distances.txt -v
```

### Command Line Options

```
Usage: fast_sssp [OPTIONS]
Options:
  -i, --input FILE       Input graph file
  -s, --source VERTEX    Source vertex (default: 0)
  -m, --mode MODE        Execution mode:
                           seq      - Sequential
                           mpi      - MPI distributed
                           openmp   - OpenMP parallel
                           cuda     - CUDA GPU
  -t, --threads NUM      Number of OpenMP threads (default: max)
  -o, --output FILE      Output file for distances
  -v, --verbose          Verbose output
  -h, --help             Show this help message

Graph file format:
  First line: n m (vertices, edges)
  Following m lines: u v w (edge from u to v with weight w)
```

## Running Tests

```bash
# Sequential tests
./test_sequential

# Parallel tests (requires MPI)
mpirun -np 2 ./test_parallel

# CUDA tests (requires GPU)
./test_cuda
```

## Benchmarking

```bash
# Run comprehensive benchmark
./benchmark data/random_10k.txt 0
```

Output example:
```
============================================================
Benchmark Results
============================================================

Implementation                      Time (ms)         Status
------------------------------------------------------------
Sequential                            1245.32             OK
OpenMP (1 threads)                    1250.18             OK
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

## Algorithm Details

### Core Algorithm (Algorithm 3: BMSSP)

The algorithm uses a divide-and-conquer approach with the following key components:

1. **FindPivots** (Algorithm 1): Reduces the frontier size by identifying "pivot" vertices with large shortest-path trees
2. **BaseCase** (Algorithm 2): Mini Dijkstra for small subproblems
3. **BMSSP** (Algorithm 3): Recursive bounded multi-source shortest path solver

### Key Parameters

- **k** = ⌊log^(1/3) n⌋: Controls pivot selection granularity
- **t** = ⌊log^(2/3) n⌋: Controls recursion depth

### Complexity

- **Time**: O(m log^(2/3) n)
- **Space**: O(n + m)

## Parallelization Strategy

### MPI (Distributed)

- Graph partitioning by vertex ranges
- Synchronization via MPI_Allreduce for distance updates
- Suitable for very large graphs that don't fit in single-node memory

### OpenMP (Shared-Memory)

- Parallel edge relaxation using dynamic scheduling
- Critical sections for atomic distance updates
- Efficient for medium-sized graphs on multi-core CPUs

### CUDA (GPU)

- CSR (Compressed Sparse Row) format for efficient GPU memory access
- Parallel Bellman-Ford relaxation
- Atomic operations for thread-safe distance updates
- Best for dense graphs with high parallelism

## Performance Tips

1. **Graph Format**: Convert graphs to constant-degree format for better cache locality
2. **Thread Count**: For OpenMP, use number of physical cores (not hyperthreads)
3. **GPU Memory**: Ensure graph fits in GPU memory for CUDA version
4. **MPI Processes**: Use powers of 2 for better load balancing

## Citation

If you use this implementation, please cite the original paper:

```bibtex
@inproceedings{duan2025breaking,
  title={Breaking the Sorting Barrier for Directed Single-Source Shortest Paths},
  author={Duan, Ran and Mao, Jiayi and Mao, Xiao and Shu, Xinkai and Yin, Longhui},
  booktitle={arXiv:2504.17033v2 [cs.DS]},
  year={2025}
}
```

## License

See LICENSE file for details.

## Contributing

Contributions are welcome! Please submit pull requests or open issues for bugs and feature requests.

## Contact

For questions and support, please open an issue on the GitHub repository.
