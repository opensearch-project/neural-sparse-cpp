# Copyright OpenSearch Contributors
# SPDX-License-Identifier: Apache-2.0
#
# The OpenSearch Contributors require contributions made to
# this file be licensed under the Apache-2.0 license or a
# compatible open source license.

# Unit tests for nsparse library

set(NSPARSE_TEST_SRC
    buf_test.cpp
    csr_layout_test.cpp
    distance_test.cpp
    exact_matcher_test.cpp
    id_map_index_test.cpp
    id_selector_test.cpp
    index_factory_test.cpp
    index_io_test.cpp
    index_test.cpp
    inline_forward_index_test.cpp
    inverted_index_test.cpp
    inverted_index_recall_test.cpp
    inverted_list_clusters_test.cpp
    inverted_lists_test.cpp
    kmeans_utils_test.cpp
    mmap_cursor_test.cpp
    mmap_index_test.cpp
    prefetch_test.cpp
    random_kmeans_test.cpp
    ranker_test.cpp
    scalar_quantizer_test.cpp
    seismic_common_test.cpp
    disk_seismic_index_test.cpp
    disk_seismic_scalar_quantized_index_test.cpp
    seismic_index_test.cpp
    seismic_batched_build_test.cpp
    seismic_invlists_writer_test.cpp
    seismic_scalar_quantized_index_test.cpp
    sparse_vectors_test.cpp
    vector_process_test.cpp
)

# GPU parity test is only meaningful when GPU support is compiled in.
if(NSPARSE_ENABLE_GPU)
    list(APPEND NSPARSE_TEST_SRC gpu_cluster_assigner_test.cpp)
endif()

# Fetch GoogleTest
include(FetchContent)
FetchContent_Declare(
    googletest
    GIT_REPOSITORY https://github.com/google/googletest.git
    GIT_TAG v1.15.2
)
set(BUILD_GMOCK OFF CACHE BOOL "" FORCE)
set(INSTALL_GTEST OFF CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

find_package(OpenMP REQUIRED)

add_executable(nsparse_test ${NSPARSE_TEST_SRC})

target_include_directories(nsparse_test PRIVATE ${PROJECT_SOURCE_DIR})

target_link_libraries(nsparse_test PRIVATE
    nsparse
    OpenMP::OpenMP_CXX
    GTest::gtest_main
    absl::flat_hash_set
)

include(GoogleTest)
# DISCOVERY_TIMEOUT: gtest_discover_tests runs the built executable with
# --gtest_list_tests as a post-build step. On cold Windows CI runners, loading
# the exe together with its deployed DLLs (e.g. abseil_dll.dll) and enumerating
# every test case can exceed the 5s default, causing a spurious build failure.
# Bump the timeout well above that to keep discovery robust.
gtest_discover_tests(nsparse_test DISCOVERY_TIMEOUT 120)

# SIMD kernel equivalence tests. distance_test.cpp above runs through the
# flag-less `nsparse`, so it only ever exercises the scalar distance.h; the SIMD
# kernels that ship in production (nsparse_avx2/avx512, selected at load time by
# the Python bindings) are otherwise untested. Each binary below compiles
# distance_kernel_equivalence_test.cpp with one ISA's -m flags so distance_simd.h
# resolves to that SIMD kernel, and checks it against a plain reference.
#
# Because the whole binary is compiled for the ISA, its static initializers can
# execute that ISA before main() -- so it must not even be launched on a CPU
# that lacks the ISA (that would SIGILL, ahead of any in-process guard). ctest
# therefore invokes each binary through run_if_isa_supported.cmake, which checks
# the host CPU first and reports a skip otherwise; this also keeps the AVX-512
# binary from faulting on an AVX-512-less host.
#
# Each binary links the matching library (the base `nsparse` for an arch's
# baseline kernel, else nsparse_avx2/avx512/sve), so every translation unit --
# the test and the library -- is compiled for the same ISA: the shared
# nsparse::detail kernel symbols then have one consistent definition (no ODR
# violation), and SparseVectors' out-of-line symbols resolve.
#
# isa: label (also NSPARSE_EXPECT_ISA); cpu_flag: the /proc/cpuinfo (or macOS
# sysctl) token to gate the launch on, or "" for an arch-baseline ISA that needs
# no gate; lib: the library target to link; ARGN: the compile flags.
function(nsparse_add_kernel_test isa cpu_flag lib)
    set(_tgt nsparse_${isa}_kernel_test)
    add_executable(${_tgt} distance_kernel_equivalence_test.cpp)
    target_include_directories(${_tgt} PRIVATE ${PROJECT_SOURCE_DIR})
    target_compile_options(${_tgt} PRIVATE ${ARGN})
    target_compile_definitions(${_tgt} PRIVATE NSPARSE_EXPECT_ISA="${isa}")
    target_link_libraries(${_tgt} PRIVATE
        ${lib}
        OpenMP::OpenMP_CXX
        GTest::gtest_main
        absl::flat_hash_set
    )
    if("${cpu_flag}" STREQUAL "")
        # Arch-baseline ISA (e.g. NEON on aarch64): always safe to run.
        add_test(NAME ${_tgt} COMMAND ${_tgt})
    else()
        # A specialized binary can execute its ISA in a static initializer --
        # before main(), before any in-process guard -- so it must not even be
        # launched on a CPU that lacks the ISA (that would SIGILL). ctest goes
        # through run_if_isa_supported.cmake, which checks the host and reports a
        # skip otherwise.
        add_test(NAME ${_tgt}
                 COMMAND ${CMAKE_COMMAND}
                         -DTEST_EXE=$<TARGET_FILE:${_tgt}>
                         -DCPU_FLAG=${cpu_flag}
                         -P ${CMAKE_CURRENT_SOURCE_DIR}/run_if_isa_supported.cmake)
    endif()
endfunction()

if(CMAKE_SYSTEM_PROCESSOR MATCHES "(x86_64|AMD64|i[3-6]86)" AND NOT WIN32)
    nsparse_add_kernel_test(avx2 avx2 nsparse_avx2 -mavx2 -mfma -mf16c -mpopcnt)
    nsparse_add_kernel_test(avx512 avx512f nsparse_avx512 -mavx2 -mfma -mf16c
                            -mavx512f -mavx512cd -mavx512vl -mavx512dq -mavx512bw
                            -mpopcnt)
elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "(aarch64|arm64|ARM64)" AND NOT WIN32)
    # NEON is the aarch64 baseline: the base `nsparse` already compiles
    # distance_neon.h (via __aarch64__), so no extra flags and no launch gate.
    nsparse_add_kernel_test(neon "" nsparse)
    # SVE is optional (Graviton3+) and, like AVX-512, can fault in a static
    # initializer on a CPU without it, so gate the launch on the "sve" flag. Not
    # built on Apple Silicon (no SVE) -- mirrors the nsparse_sve library target.
    if(NOT APPLE)
        nsparse_add_kernel_test(sve sve nsparse_sve -march=armv8-a+sve)
    endif()
endif()
