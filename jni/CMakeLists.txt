# Copyright OpenSearch Contributors
# SPDX-License-Identifier: Apache-2.0

find_package(JNI REQUIRED)

add_library(nsparse_jni SHARED nsparse_jni.cpp)

target_include_directories(nsparse_jni PRIVATE
    ${JNI_INCLUDE_DIRS}
    ${PROJECT_SOURCE_DIR}
)

target_link_libraries(nsparse_jni PRIVATE nsparse)

set_target_properties(nsparse_jni PROPERTIES
    POSITION_INDEPENDENT_CODE ON
    OUTPUT_NAME "nsparse_jni"
    PREFIX "lib"
)

# Strip debug symbols in release builds
if(CMAKE_BUILD_TYPE STREQUAL "Release")
    if(NOT WIN32)
        target_link_options(nsparse_jni PRIVATE -s)
    endif()
endif()
