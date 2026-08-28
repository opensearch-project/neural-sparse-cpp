# Copyright OpenSearch Contributors
# SPDX-License-Identifier: Apache-2.0
#
# The OpenSearch Contributors require contributions made to
# this file be licensed under the Apache-2.0 license or a
# compatible open source license.

set(NSPARSE_BENCHMARK_SRC
    index_search_benchmark.cpp
)
# Fetch Google Benchmark
include(FetchContent)
FetchContent_Declare(
    googlebenchmark
    GIT_REPOSITORY https://github.com/google/benchmark.git
    GIT_TAG v1.9.1
)
set(BENCHMARK_ENABLE_TESTING OFF CACHE BOOL "" FORCE)
set(BENCHMARK_ENABLE_INSTALL OFF CACHE BOOL "" FORCE)

# Enable libpfm for perf counters on Linux (requires libpfm4-dev)
if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
    set(BENCHMARK_ENABLE_LIBPFM ON CACHE BOOL "" FORCE)
endif()

FetchContent_MakeAvailable(googlebenchmark)

# Kernel selection is compile-time (nsparse/utils/distance_simd.h) and the -m
# flags live on the per-level library targets, so linking the plain `nsparse`
# gets the scalar kernels no matter what NSPARSE_OPT_LEVEL says. Pick the library
# that matches the requested level (as nsparse/python/CMakeLists.txt does) and
# print which one, so a mismatch is visible instead of silently scalar. The
# foreach over an explicit list (not nsparse_${NSPARSE_OPT_LEVEL}) makes a typo'd
# level fall back to a target that exists rather than fail at link time.
set(NSPARSE_BENCH_LIB nsparse)
foreach(_level avx2 avx512 sve)
    if(NSPARSE_OPT_LEVEL STREQUAL "${_level}")
        set(NSPARSE_BENCH_LIB nsparse_${_level})
    endif()
endforeach()
message(STATUS "benchmarks: NSPARSE_OPT_LEVEL=${NSPARSE_OPT_LEVEL} -> linking ${NSPARSE_BENCH_LIB}")

if(NSPARSE_BENCHMARK_SRC)
    find_package(OpenMP REQUIRED)

    add_executable(nsparse_benchmark ${NSPARSE_BENCHMARK_SRC})

    target_include_directories(nsparse_benchmark PRIVATE ${PROJECT_SOURCE_DIR})

    target_link_libraries(nsparse_benchmark PRIVATE
        ${NSPARSE_BENCH_LIB}
        OpenMP::OpenMP_CXX
        benchmark::benchmark
        benchmark::benchmark_main
        absl::flat_hash_set
        absl::flat_hash_map
    )
endif()

# Latency / residency / recall driver for a single serialized index. Not a
# google-benchmark target: it reports the per-query tail, the RssAnon/RssFile
# split and recall, none of which that harness can express.
add_executable(sq_residency_bench sq_residency_bench.cpp)
target_include_directories(sq_residency_bench PRIVATE ${PROJECT_SOURCE_DIR})
target_link_libraries(sq_residency_bench PRIVATE
    ${NSPARSE_BENCH_LIB}
    OpenMP::OpenMP_CXX
    absl::flat_hash_set
    absl::flat_hash_map
)

# Build-time benchmark: SeismicIndex build() on CPU vs GPU (cuSPARSE). The GPU
# case is compiled in only when the nsparse target carries CUDA support (the
# NSPARSE_WITH_GPU define propagates from the nsparse target).
add_executable(nsparse_build_benchmark index_build_benchmark.cpp)
target_include_directories(nsparse_build_benchmark PRIVATE ${PROJECT_SOURCE_DIR})
target_link_libraries(nsparse_build_benchmark PRIVATE
    ${NSPARSE_BENCH_LIB}
    OpenMP::OpenMP_CXX
    benchmark::benchmark
    absl::flat_hash_set
    absl::flat_hash_map
)
