# Quick Start Guide

This guide will get you up and running with the Fast SSSP implementation in 5 minutes.

## Prerequisites

Ensure you have the following installed:
- C++17 compiler (g++ or clang++)
- CMake 3.18+
- (Optional) MPI library for distributed computing
- (Optional) CUDA Toolkit for GPU acceleration

## Build in 3 Steps

### 1. Clone and Navigate

```bash
cd Breaking-the-Sorting-Barrier-for-Directed-Single-Source-Shortest-Paths-inference
```

### 2. Build with Default Options (MPI + OpenMP)

```bash
./scripts/build.sh
```

Or build with specific options:

```bash
# Sequential only
./scripts/build.sh --no-mpi --no-openmp

# With CUDA support
./scripts/build.sh --cuda

# Debug build
./scripts/build.sh --debug
```

### 3. Run Tests

```bash
./scripts/run_tests.sh
```

## Your First SSSP Computation

### Generate a Test Graph

```bash
# Create a random graph with 1,000 vertices and 5,000 edges
mkdir -p data
./build/graph_generator -n 1000 -m 5000 -t random -o data/test_graph.txt
```

### Run Sequential Algorithm

```bash
./build/fast_sssp -i data/test_graph.txt -s 0 -m seq -v
```

Expected output:
```
Loading graph: 1000 vertices, 5000 edges
Graph loaded successfully
Running sequential SSSP algorithm
Computation time: 42 ms

Sample distances from source 0:
  Vertex 0: 0
  Vertex 1: 2.34
  Vertex 2: 1.56
  ...
```

### Run with OpenMP (Parallel)

```bash
# Use 4 threads
./build/fast_sssp -i data/test_graph.txt -s 0 -m openmp -t 4 -v
```

### Run with MPI (Distributed)

```bash
# Use 4 MPI processes
mpirun -np 4 ./build/fast_sssp -i data/test_graph.txt -s 0 -m mpi -v
```

### Run with CUDA (GPU)

```bash
./build/fast_sssp -i data/test_graph.txt -s 0 -m cuda -v
```

## Benchmark Different Implementations

```bash
./scripts/run_benchmark.sh -n 10000 -m 50000 -t random
```

This will:
1. Generate a random graph with 10,000 vertices and 50,000 edges
2. Run the sequential, OpenMP (with various thread counts), and CUDA implementations
3. Display timing results and speedups

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
CUDA GPU                                45.23             OK
============================================================

Speedups (relative to Sequential):
  OpenMP (2 threads): 1.94x
  OpenMP (4 threads): 3.71x
  CUDA GPU: 27.53x
```

## Graph File Format

The input graph format is simple:

```
n m
u1 v1 w1
u2 v2 w2
...
um vm wm
```

Where:
- `n` = number of vertices
- `m` = number of edges
- Each line `ui vi wi` represents an edge from vertex `ui` to vertex `vi` with weight `wi`

Example (`data/simple.txt`):
```
4 5
0 1 1.0
0 2 4.0
1 2 2.0
1 3 5.0
2 3 1.0
```

## Next Steps

- See [IMPLEMENTATION_README.md](IMPLEMENTATION_README.md) for detailed documentation
- Explore [tests/](tests/) for more examples
- Read the [original paper](2504.17033v2.pdf) for algorithm details

## Troubleshooting

### Build fails with "MPI not found"

```bash
# Build without MPI
./scripts/build.sh --no-mpi
```

### Build fails with "CUDA not found"

Make sure CUDA Toolkit is installed and in your PATH:
```bash
export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
```

Or build without CUDA:
```bash
./scripts/build.sh  # CUDA is OFF by default
```

### "Segmentation fault" when running

- Ensure graph file is properly formatted
- Check that source vertex is within valid range `[0, n-1]`
- Try with a smaller graph first

## Performance Tips

1. **Use OpenMP for medium graphs** (10K-100K vertices) on multi-core CPUs
2. **Use CUDA for large, dense graphs** if you have a GPU
3. **Use MPI for very large graphs** that don't fit in memory
4. **Set thread count** to number of physical cores for best OpenMP performance

## Support

For issues and questions:
- Check [IMPLEMENTATION_README.md](IMPLEMENTATION_README.md)
- Review test files in [tests/](tests/)
- Open an issue on GitHub
