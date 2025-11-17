#!/bin/bash
# Build script for Fast SSSP

set -e

# Parse arguments
BUILD_TYPE="Release"
ENABLE_MPI="ON"
ENABLE_OPENMP="ON"
ENABLE_CUDA="OFF"
BUILD_DIR="build"
CLEAN=0

usage() {
    echo "Usage: $0 [OPTIONS]"
    echo "Options:"
    echo "  --debug          Build in debug mode"
    echo "  --no-mpi         Disable MPI support"
    echo "  --no-openmp      Disable OpenMP support"
    echo "  --cuda           Enable CUDA support"
    echo "  --clean          Clean build directory first"
    echo "  --build-dir DIR  Build directory (default: build)"
    echo "  -h, --help       Show this help"
}

while [[ $# -gt 0 ]]; do
    case $1 in
        --debug)
            BUILD_TYPE="Debug"
            shift
            ;;
        --no-mpi)
            ENABLE_MPI="OFF"
            shift
            ;;
        --no-openmp)
            ENABLE_OPENMP="OFF"
            shift
            ;;
        --cuda)
            ENABLE_CUDA="ON"
            shift
            ;;
        --clean)
            CLEAN=1
            shift
            ;;
        --build-dir)
            BUILD_DIR="$2"
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

# Clean if requested
if [ $CLEAN -eq 1 ]; then
    echo "Cleaning build directory..."
    rm -rf "$BUILD_DIR"
fi

# Create build directory
mkdir -p "$BUILD_DIR"
cd "$BUILD_DIR"

# Configure
echo "Configuring build..."
echo "  Build type: $BUILD_TYPE"
echo "  MPI: $ENABLE_MPI"
echo "  OpenMP: $ENABLE_OPENMP"
echo "  CUDA: $ENABLE_CUDA"

cmake .. \
    -DCMAKE_BUILD_TYPE="$BUILD_TYPE" \
    -DENABLE_MPI="$ENABLE_MPI" \
    -DENABLE_OPENMP="$ENABLE_OPENMP" \
    -DENABLE_CUDA="$ENABLE_CUDA" \
    -DBUILD_TESTS=ON

# Build
echo "Building..."
make -j$(nproc)

echo "Build complete! Binaries are in $BUILD_DIR/"
