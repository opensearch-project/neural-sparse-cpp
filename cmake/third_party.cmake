# Copyright OpenSearch Contributors
# SPDX-License-Identifier: Apache-2.0
#
# The OpenSearch Contributors require contributions made to
# this file be licensed under the Apache-2.0 license or a
# compatible open source license.

# Set custom download directory for third-party libraries
set(FETCHCONTENT_BASE_DIR "${CMAKE_SOURCE_DIR}/third_party" CACHE PATH "")

# Try system first, fetch if not found
find_package(absl QUIET)

if(NOT absl_FOUND)
    message(STATUS "Abseil not found, fetching from GitHub...")
    include(FetchContent)
    FetchContent_Declare(
        abseil-cpp
        GIT_REPOSITORY https://github.com/abseil/abseil-cpp.git
        GIT_TAG        20250814.1
    )
    set(ABSL_PROPAGATE_CXX_STD ON)
    set(CMAKE_POSITION_INDEPENDENT_CODE ON)
    FetchContent_MakeAvailable(abseil-cpp)
endif()