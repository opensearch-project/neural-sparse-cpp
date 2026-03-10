# Copyright OpenSearch Contributors
# SPDX-License-Identifier: Apache-2.0
#
# The OpenSearch Contributors require contributions made to
# this file be licensed under the Apache-2.0 license or a
# compatible open source license.

# Unit tests for nsparse library

set(NSPARSE_TEST_SRC
    distance_test.cpp
    exact_matcher_test.cpp
    id_map_index_test.cpp
    id_selector_test.cpp
    index_factory_test.cpp
    index_io_test.cpp
    inverted_index_test.cpp
    inverted_list_clusters_test.cpp
    inverted_lists_test.cpp
    kmeans_utils_test.cpp
    prefetch_test.cpp
    random_kmeans_test.cpp
    ranker_test.cpp
    scalar_quantizer_test.cpp
    seismic_common_test.cpp
    seismic_index_test.cpp
    seismic_invlists_writer_test.cpp
    seismic_scalar_quantized_index_test.cpp
    sparse_vectors_test.cpp
    vector_process_test.cpp
)

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
gtest_discover_tests(nsparse_test)
