# Copyright OpenSearch Contributors
# SPDX-License-Identifier: Apache-2.0
#
# The OpenSearch Contributors require contributions made to
# this file be licensed under the Apache-2.0 license or a
# compatible open source license.
#
# CUDA / cuSPARSE toolchain setup for GPU-accelerated index building.
#
# Enable with -DNSPARSE_ENABLE_CUDA=ON. When the CUDA toolkit is installed in a
# non-standard location (for example, shipped inside a pip `nvidia-cu13` wheel),
# point CMake at it with:
#   -DNSPARSE_CUDA_TOOLKIT_ROOT=/path/to/nvidia/cu13
# and CMake will pick up nvcc, the headers and the runtime/cusparse libraries.

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
# reference GPU) is compute capability 8.9. CMake picks a conservative default
# during language enablement, so override it unless the caller pinned one
# explicitly via -DNSPARSE_CUDA_ARCHITECTURES.
if(NSPARSE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES "${NSPARSE_CUDA_ARCHITECTURES}" CACHE STRING "" FORCE)
else()
    set(CMAKE_CUDA_ARCHITECTURES 80 89 CACHE STRING "CUDA architectures to build for" FORCE)
endif()

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

message(STATUS "nsparse: CUDA enabled (nvcc=${CMAKE_CUDA_COMPILER}, "
        "archs=${CMAKE_CUDA_ARCHITECTURES})")
