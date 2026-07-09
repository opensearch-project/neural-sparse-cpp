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

if(NSPARSE_BENCHMARK_SRC)
    find_package(OpenMP REQUIRED)

    add_executable(nsparse_benchmark ${NSPARSE_BENCHMARK_SRC})

    target_include_directories(nsparse_benchmark PRIVATE ${PROJECT_SOURCE_DIR})

    target_link_libraries(nsparse_benchmark PRIVATE
        nsparse
        OpenMP::OpenMP_CXX
        benchmark::benchmark
        benchmark::benchmark_main
        absl::flat_hash_set
        absl::flat_hash_map
    )
endif()

# Build-time benchmark: SeismicIndex build() on CPU vs GPU (cuSPARSE). The GPU
# case is compiled in only when the nsparse target carries CUDA support (the
# NSPARSE_WITH_GPU define propagates from the nsparse target).
add_executable(nsparse_build_benchmark index_build_benchmark.cpp)
target_include_directories(nsparse_build_benchmark PRIVATE ${PROJECT_SOURCE_DIR})
target_link_libraries(nsparse_build_benchmark PRIVATE
    nsparse
    OpenMP::OpenMP_CXX
    benchmark::benchmark
    absl::flat_hash_set
    absl::flat_hash_map
)
