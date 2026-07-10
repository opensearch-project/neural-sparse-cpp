# Copyright OpenSearch Contributors
# SPDX-License-Identifier: Apache-2.0
#
# The OpenSearch Contributors require contributions made to
# this file be licensed under the Apache-2.0 license or a
# compatible open source license.
#
# CUDA / cuSPARSE toolchain setup and target wiring for GPU-accelerated index
# building (the build() path; search() always stays on CPU).
#
# Included from nsparse/CMakeLists.txt (before the nsparse targets are defined)
# only when -DNSPARSE_ENABLE_GPU=ON. It enables the CUDA language, finds the
# toolkit, and defines nsparse_attach_gpu(<target>) which the parent uses to
# compile gpu_cluster_assigner.cu into each nsparse variant and link cuSPARSE.
#
# When the CUDA toolkit lives in a non-standard location (for example, shipped
# inside a pip `nvidia-cu13` wheel), point CMake at it with:
#   -DNSPARSE_CUDA_TOOLKIT_ROOT=/path/to/nvidia/cu13

# Allow the caller to hint the toolkit location (nvcc lives in <root>/bin).
if(NSPARSE_CUDA_TOOLKIT_ROOT)
    set(CUDAToolkit_ROOT "${NSPARSE_CUDA_TOOLKIT_ROOT}" CACHE PATH "" FORCE)
    if(NOT CMAKE_CUDA_COMPILER)
        set(CMAKE_CUDA_COMPILER "${NSPARSE_CUDA_TOOLKIT_ROOT}/bin/nvcc" CACHE FILEPATH "" FORCE)
    endif()
    # Wheel-packaged toolkits keep shared libs under lib/ rather than lib64/.
    list(APPEND CMAKE_LIBRARY_PATH
        "${NSPARSE_CUDA_TOOLKIT_ROOT}/lib"
        "${NSPARSE_CUDA_TOOLKIT_ROOT}/lib64")
endif()

# Turn on the CUDA language now that the compiler is (optionally) hinted.
enable_language(CUDA)

# CUDAToolkit provides the CUDA::cusparse and CUDA::cudart imported targets.
find_package(CUDAToolkit REQUIRED)

# Default to the common data-center / Ada architectures. L4 (this project's
# reference GPU) is compute capability 8.9. Override with
# -DNSPARSE_CUDA_ARCHITECTURES=... if targeting different GPUs.
if(NSPARSE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES "${NSPARSE_CUDA_ARCHITECTURES}" CACHE STRING "" FORCE)
else()
    set(CMAKE_CUDA_ARCHITECTURES 80 89 CACHE STRING "CUDA architectures to build for" FORCE)
endif()

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Absolute paths to the CUDA translation units, resolved here so the helper can
# be called from the parent directory scope. This file is include()d from
# nsparse/CMakeLists.txt, so CMAKE_CURRENT_LIST_DIR is this gpu/ directory.
set(NSPARSE_GPU_SRC
    "${CMAKE_CURRENT_LIST_DIR}/gpu_cluster_assigner.cu"
    "${CMAKE_CURRENT_LIST_DIR}/gpu_summarizer.cu"
    CACHE INTERNAL "nsparse GPU sources")

message(STATUS "nsparse: GPU enabled (nvcc=${CMAKE_CUDA_COMPILER}, "
        "archs=${CMAKE_CUDA_ARCHITECTURES})")

# Compile the GPU sources into <target> and link cuSPARSE/cudart. Guarded by the
# NSPARSE_WITH_GPU compile define so the CPU sources pick up the GPU hooks.
function(nsparse_attach_gpu target)
    target_sources(${target} PRIVATE ${NSPARSE_GPU_SRC})
    target_compile_definitions(${target} PUBLIC NSPARSE_WITH_GPU)
    target_link_libraries(${target} PRIVATE CUDA::cusparse CUDA::cudart)
    # Resolve device symbols into the library itself so pure-C++ consumers
    # (tests, benchmarks) can link it without a separate CUDA device-link step.
    set_target_properties(${target} PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        CUDA_RESOLVE_DEVICE_SYMBOLS ON
        CUDA_ARCHITECTURES "${CMAKE_CUDA_ARCHITECTURES}")
endfunction()
