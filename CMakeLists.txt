cmake_minimum_required(VERSION 3.18)
project(FastSSSP LANGUAGES CXX C)

# C++17 required
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Options
option(ENABLE_MPI "Enable MPI support" ON)
option(ENABLE_OPENMP "Enable OpenMP support" ON)
option(ENABLE_CUDA "Enable CUDA support" OFF)
option(BUILD_TESTS "Build test programs" ON)

# Find packages
if(ENABLE_MPI)
    find_package(MPI REQUIRED)
    add_definitions(-DENABLE_MPI)
endif()

if(ENABLE_OPENMP)
    find_package(OpenMP REQUIRED)
    add_definitions(-DENABLE_OPENMP)
endif()

if(ENABLE_CUDA)
    enable_language(CUDA)
    find_package(CUDAToolkit REQUIRED)
    add_definitions(-DENABLE_CUDA)
    set(CMAKE_CUDA_STANDARD 14)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
endif()

# Include directories
include_directories(${CMAKE_SOURCE_DIR}/include)

# Source files
set(CORE_SOURCES
    src/graph.cpp
    src/partial_sort_ds.cpp
    src/sssp_algorithm.cpp
    src/classical_sssp.cpp
)

set(PARALLEL_SOURCES
    src/parallel_sssp.cpp
)

set(CUDA_SOURCES
    src/cuda_sssp.cu
)

# Core library (sequential)
add_library(sssp_core ${CORE_SOURCES})

# Parallel library (MPI + OpenMP)
if(ENABLE_MPI AND ENABLE_OPENMP)
    add_library(sssp_parallel ${PARALLEL_SOURCES})
    target_link_libraries(sssp_parallel sssp_core MPI::MPI_CXX OpenMP::OpenMP_CXX)
endif()

# CUDA library
if(ENABLE_CUDA)
    add_library(sssp_cuda ${CUDA_SOURCES})
    target_link_libraries(sssp_cuda sssp_core CUDA::cudart)
    set_target_properties(sssp_cuda PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
endif()

# Main executable
add_executable(fast_sssp src/main.cpp)
target_link_libraries(fast_sssp sssp_core)

if(ENABLE_MPI AND ENABLE_OPENMP)
    target_link_libraries(fast_sssp sssp_parallel)
endif()

if(ENABLE_CUDA)
    target_link_libraries(fast_sssp sssp_cuda)
endif()

# Test programs
if(BUILD_TESTS)
    add_executable(test_sequential tests/test_sequential.cpp)
    target_link_libraries(test_sequential sssp_core)

    if(ENABLE_MPI AND ENABLE_OPENMP)
        add_executable(test_parallel tests/test_parallel.cpp)
        target_link_libraries(test_parallel sssp_parallel)
    endif()

    if(ENABLE_CUDA)
        add_executable(test_cuda tests/test_cuda.cpp)
        target_link_libraries(test_cuda sssp_cuda)
    endif()

    add_executable(graph_generator tests/graph_generator.cpp)
    target_link_libraries(graph_generator sssp_core)

    add_executable(benchmark tests/benchmark.cpp)
    target_link_libraries(benchmark sssp_core)

    if(ENABLE_MPI AND ENABLE_OPENMP)
        target_link_libraries(benchmark sssp_parallel)
    endif()

    if(ENABLE_CUDA)
        target_link_libraries(benchmark sssp_cuda)
    endif()
endif()

# Compiler flags
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    add_compile_options(-Wall -Wextra -O3 -march=native)
endif()

# Installation
install(TARGETS fast_sssp DESTINATION bin)
install(DIRECTORY include/ DESTINATION include)
