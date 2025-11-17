#!/bin/bash
# Benchmark script for Fast SSSP

set -e

# Default parameters
GRAPH_SIZE=10000
EDGE_COUNT=50000
GRAPH_TYPE="random"
SOURCE=0
DATA_DIR="data"
BUILD_DIR="build"

usage() {
    echo "Usage: $0 [OPTIONS]"
    echo "Options:"
    echo "  -n SIZE          Number of vertices (default: 10000)"
    echo "  -m EDGES         Number of edges (default: 50000)"
    echo "  -t TYPE          Graph type: random, grid, dag (default: random)"
    echo "  -s SOURCE        Source vertex (default: 0)"
    echo "  --data-dir DIR   Data directory (default: data)"
    echo "  --build-dir DIR  Build directory (default: build)"
    echo "  -h, --help       Show this help"
}

while [[ $# -gt 0 ]]; do
    case $1 in
        -n)
            GRAPH_SIZE="$2"
            shift 2
            ;;
        -m)
            EDGE_COUNT="$2"
            shift 2
            ;;
        -t)
            GRAPH_TYPE="$2"
            shift 2
            ;;
        -s)
            SOURCE="$2"
            shift 2
            ;;
        --data-dir)
            DATA_DIR="$2"
            shift 2
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

# Create data directory
mkdir -p "$DATA_DIR"

# Generate graph
GRAPH_FILE="$DATA_DIR/${GRAPH_TYPE}_${GRAPH_SIZE}.txt"

echo "Generating graph..."
echo "  Type: $GRAPH_TYPE"
echo "  Vertices: $GRAPH_SIZE"
echo "  Edges: $EDGE_COUNT"
echo "  Output: $GRAPH_FILE"

if [ ! -f "$GRAPH_FILE" ]; then
    "$BUILD_DIR/graph_generator" \
        -n "$GRAPH_SIZE" \
        -m "$EDGE_COUNT" \
        -t "$GRAPH_TYPE" \
        -o "$GRAPH_FILE"
else
    echo "  Graph already exists, skipping generation"
fi

# Run benchmark
echo ""
echo "Running benchmark..."
"$BUILD_DIR/benchmark" "$GRAPH_FILE" "$SOURCE"
