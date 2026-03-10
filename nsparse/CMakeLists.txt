# Copyright OpenSearch Contributors
# SPDX-License-Identifier: Apache-2.0
#
# The OpenSearch Contributors require contributions made to
# this file be licensed under the Apache-2.0 license or a
# compatible open source license.

set(NSPARSE_SRC
    index.cpp
    brutal_index.cpp
    seismic_index.cpp
    seismic_scalar_quantized_index.cpp
    sparse_vectors.cpp
    index_factory.cpp
    id_map_index.cpp
    inverted_index.cpp
    cluster/random_kmeans.cpp
    cluster/kmeans_utils.cpp
    cluster/inverted_list_clusters.cpp
    invlists/inverted_lists.cpp
    io/buffered_io.cpp
    io/file_io.cpp
    io/index_io.cpp
)

set(NSPARSE_HEADERS
    index.h
    types.h
    sparse_vectors.h
    brutal_index.h
    seismic_index.h
    seismic_scalar_quantized_index.h
    index_factory.h
    id_map_index.h
    inverted_index.h
    utils/checks.h
    utils/distance.h
    utils/distance_avx512.h
    utils/distance_avx2.h
    utils/distance_neon.h
    utils/distance_sve.h
    utils/distance_simd.h
    utils/ranker.h
    cluster/random_kmeans.h
    cluster/inverted_list_clusters.h
    cluster/kmeans_utils.h
    cluster/inverted_list_clusters.h
    invlists/inverted_lists.h
    io/io.h
    io/buffered_io.h
    io/seismic_invlists_writer.h
    io/file_io.h
    io/index_io.h
)

add_library(nsparse ${NSPARSE_SRC} ${NSPARSE_HEADERS})

# Detect architecture - x86 SIMD targets only make sense on x86_64
if(CMAKE_SYSTEM_PROCESSOR MATCHES "(x86_64|AMD64|i[3-6]86)")
  set(NSPARSE_IS_X86 TRUE)
else()
  set(NSPARSE_IS_X86 FALSE)
endif()

# AVX2 target - x86 only
if(NSPARSE_IS_X86)
  add_library(nsparse_avx2 ${NSPARSE_SRC} ${NSPARSE_HEADERS})
  if(NOT NSPARSE_OPT_LEVEL STREQUAL "avx2" AND NOT NSPARSE_OPT_LEVEL STREQUAL "avx512")
    set_target_properties(nsparse_avx2 PROPERTIES EXCLUDE_FROM_ALL TRUE)
  endif()
  if(NOT WIN32)
    target_compile_options(nsparse_avx2 PRIVATE $<$<COMPILE_LANGUAGE:CXX>:-mavx2 -mfma -mf16c -mpopcnt>)
  else()
    target_compile_options(nsparse_avx2 PRIVATE $<$<COMPILE_LANGUAGE:CXX>:/arch:AVX2>)
    add_compile_options(/bigobj)
  endif()
else()
  # Create a dummy target that just aliases the base library on non-x86
  add_library(nsparse_avx2 ALIAS nsparse)
endif()

# AVX512 target - x86 only
if(NSPARSE_IS_X86)
  add_library(nsparse_avx512 ${NSPARSE_SRC} ${NSPARSE_HEADERS})
  if(NOT NSPARSE_OPT_LEVEL STREQUAL "avx512")
    set_target_properties(nsparse_avx512 PROPERTIES EXCLUDE_FROM_ALL TRUE)
  endif()
  if(NOT WIN32)
    target_compile_options(nsparse_avx512 PRIVATE $<$<COMPILE_LANGUAGE:CXX>:-mavx2 -mfma -mf16c -mavx512f -mavx512cd -mavx512vl -mavx512dq -mavx512bw -mpopcnt>)
  else()
    target_compile_options(nsparse_avx512 PRIVATE $<$<COMPILE_LANGUAGE:CXX>:/arch:AVX512>)
    add_compile_options(/bigobj)
  endif()
else()
  add_library(nsparse_avx512 ALIAS nsparse)
endif()

# SVE target - ARM only (but note: Apple Silicon doesn't support SVE)
# Detect ARM architecture
if(CMAKE_SYSTEM_PROCESSOR MATCHES "(aarch64|arm64|ARM64)")
  set(NSPARSE_IS_ARM TRUE)
else()
  set(NSPARSE_IS_ARM FALSE)
endif()

if(NSPARSE_IS_ARM AND NOT APPLE)
  # SVE is available on some ARM processors but NOT on Apple Silicon
  add_library(nsparse_sve ${NSPARSE_SRC} ${NSPARSE_HEADERS})
  if(NOT NSPARSE_OPT_LEVEL STREQUAL "sve")
    set_target_properties(nsparse_sve PROPERTIES EXCLUDE_FROM_ALL TRUE)
  endif()
  if(NOT WIN32)
    if("${CMAKE_CXX_FLAGS} ${CMAKE_CXX_FLAGS_DEBUG} " MATCHES "(^| )-march=native")
    elseif("${CMAKE_CXX_FLAGS} ${CMAKE_CXX_FLAGS_DEBUG} " MATCHES "(^| )(-march=armv[0-9]+(\\.[1-9]+)?-[^+ ](\\+[^+$ ]+)*)")
      target_compile_options(nsparse_sve PRIVATE $<$<AND:$<COMPILE_LANGUAGE:CXX>,$<CONFIG:DEBUG>>:${CMAKE_MATCH_2}+sve>)
    elseif(NOT "${CMAKE_CXX_FLAGS} ${CMAKE_CXX_FLAGS_DEBUG} " MATCHES "(^| )-march=armv")
      target_compile_options(nsparse_sve PRIVATE $<$<AND:$<COMPILE_LANGUAGE:CXX>,$<CONFIG:DEBUG>>:-march=armv8-a+sve>)
    endif()
    if("${CMAKE_CXX_FLAGS} ${CMAKE_CXX_FLAGS_RELEASE} " MATCHES "(^| )-march=native")
    elseif("${CMAKE_CXX_FLAGS} ${CMAKE_CXX_FLAGS_RELEASE} " MATCHES "(^| )(-march=armv[0-9]+(\\.[1-9]+)?-[^+ ](\\+[^+$ ]+)*)")
      target_compile_options(nsparse_sve PRIVATE $<$<AND:$<COMPILE_LANGUAGE:CXX>,$<CONFIG:RELEASE>>:${CMAKE_MATCH_2}+sve>)
    elseif(NOT "${CMAKE_CXX_FLAGS} ${CMAKE_CXX_FLAGS_RELEASE} " MATCHES "(^| )-march=armv")
      target_compile_options(nsparse_sve PRIVATE $<$<AND:$<COMPILE_LANGUAGE:CXX>,$<CONFIG:RELEASE>>:-march=armv8-a+sve>)
    endif()
  endif()
  set_target_properties(nsparse_sve PROPERTIES POSITION_INDEPENDENT_CODE ON)
  target_include_directories(nsparse_sve PUBLIC $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}>)
endif()

# Set properties for base library (always built)
set_target_properties(nsparse PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_include_directories(nsparse PUBLIC $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}>)

# Set properties for x86 SIMD targets (only if they're real targets, not aliases)
if(NSPARSE_IS_X86)
  set_target_properties(nsparse_avx2 nsparse_avx512 PROPERTIES POSITION_INDEPENDENT_CODE ON)
  target_include_directories(nsparse_avx2 PUBLIC $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}>)
  target_include_directories(nsparse_avx512 PUBLIC $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}>)
endif()

find_package(OpenMP REQUIRED)
# Link base library (always)
target_link_libraries(nsparse PUBLIC OpenMP::OpenMP_CXX absl::flat_hash_set absl::flat_hash_map)

# Link x86 SIMD targets (only if they're real targets)
if(NSPARSE_IS_X86)
  target_link_libraries(nsparse_avx2 PUBLIC OpenMP::OpenMP_CXX absl::flat_hash_set absl::flat_hash_map)
  target_link_libraries(nsparse_avx512 PUBLIC OpenMP::OpenMP_CXX absl::flat_hash_set absl::flat_hash_map)
endif()

# Link SVE target (only if it exists as a real target)
if(NSPARSE_IS_ARM AND NOT APPLE)
  target_link_libraries(nsparse_sve PUBLIC OpenMP::OpenMP_CXX absl::flat_hash_set absl::flat_hash_map)
endif()
