#
# Copyright OpenSearch Contributors
# SPDX-License-Identifier: Apache-2.0
#

# Require CMake 3.15+ (matching scikit-build-core) Use new versions of all
# policies up to CMake 3.27
cmake_minimum_required(VERSION 3.15...3.27)

project(nsparse
  VERSION 0.1
  LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(WIN32)
    # mmap_file.h defines both before including windows.h, but a dependency's
    # header may pull windows.h in first -- and its min/max macros break every
    # later std::numeric_limits<T>::max(). Set here as well so include order
    # cannot decide whether the build succeeds.
    add_compile_definitions(NOMINMAX WIN32_LEAN_AND_MEAN)
endif()

# Sanitizers, if requested. Included before any target exists so the flags reach
# every one of them, third-party dependencies included.
include(cmake/sanitizers.cmake)

# Third party dependencies
include(cmake/third_party.cmake)

# Valid values are "generic", "avx2", "avx512", "sve".
option(NSPARSE_OPT_LEVEL "" "generic")
option(NSPARSE_ENABLE_PYTHON "Build Python bindings" OFF)

# GPU acceleration for the index-building (build()) path via NVIDIA cuSPARSE.
# The search() path always stays on CPU. Default OFF so CPU-only builds and CI
# are unaffected when no CUDA toolkit is present. Toolchain setup and per-target
# wiring live in nsparse/gpu/CMakeLists.txt (included from nsparse/CMakeLists.txt).
option(NSPARSE_ENABLE_GPU "Build GPU-accelerated index building via cuSPARSE" OFF)

add_subdirectory(nsparse)

# Python bindings
if(NSPARSE_ENABLE_PYTHON)
    add_subdirectory(nsparse/python)
endif()

# CTest must be included in the top level to enable `make test` target.
include(CTest)

# Unit tests
option(NSPARSE_ENABLE_TESTS "Build unit tests" OFF)
if(NSPARSE_ENABLE_TESTS)
    add_subdirectory(tests)
endif()

# Benchmarks
option(NSPARSE_ENABLE_BENCHMARKS "Build benchmarks" OFF)
if(NSPARSE_ENABLE_BENCHMARKS)
    add_subdirectory(benchmarks)
endif()
