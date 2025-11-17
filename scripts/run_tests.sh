#!/bin/bash
# Test runner script for Fast SSSP

set -e

BUILD_DIR="build"
MPI_PROCS=2

usage() {
    echo "Usage: $0 [OPTIONS]"
    echo "Options:"
    echo "  --build-dir DIR  Build directory (default: build)"
    echo "  --mpi-procs NUM  Number of MPI processes (default: 2)"
    echo "  -h, --help       Show this help"
}

while [[ $# -gt 0 ]]; do
    case $1 in
        --build-dir)
            BUILD_DIR="$2"
            shift 2
            ;;
        --mpi-procs)
            MPI_PROCS="$2"
            shift 2
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            usage
            exit 1
            ;;
    esac
done

cd "$BUILD_DIR"

echo "Running Fast SSSP Tests"
echo "======================="
echo ""

# Run sequential tests
if [ -f "test_sequential" ]; then
    echo "Running sequential tests..."
    ./test_sequential
    echo ""
else
    echo "Sequential tests not built"
fi

# Run parallel tests
if [ -f "test_parallel" ]; then
    echo "Running parallel tests with $MPI_PROCS MPI processes..."
    if command -v mpirun &> /dev/null; then
        mpirun -np "$MPI_PROCS" ./test_parallel
    else
        echo "mpirun not found, skipping MPI tests"
    fi
    echo ""
else
    echo "Parallel tests not built"
fi

# Run CUDA tests
if [ -f "test_cuda" ]; then
    echo "Running CUDA tests..."
    if command -v nvidia-smi &> /dev/null; then
        ./test_cuda
    else
        echo "CUDA not available, skipping GPU tests"
    fi
    echo ""
else
    echo "CUDA tests not built"
fi

echo "All tests complete!"
