# Copyright OpenSearch Contributors
# SPDX-License-Identifier: Apache-2.0
#
# The OpenSearch Contributors require contributions made to
# this file be licensed under the Apache-2.0 license or a
# compatible open source license.

# Python bindings for NSPARSE

project(pynsparse
  LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)

# Find Python first
find_package(Python REQUIRED
  COMPONENTS Interpreter Development.Module NumPy
)

find_package(SWIG REQUIRED COMPONENTS python)
include(${SWIG_USE_FILE})

set(UseSWIG_TARGET_NAME_PREFERENCE STANDARD)
set(SWIG_SOURCE_FILE_EXTENSIONS swig)

# Configure SWIG
macro(configure_swignsparse source)
  set_source_files_properties(${source} PROPERTIES
    CPLUSPLUS ON
    USE_TARGET_INCLUDE_DIRECTORIES TRUE
  )
  if(NOT ${CMAKE_SYSTEM_NAME} MATCHES "Darwin" AND NOT WIN32)
    set_source_files_properties(${source} PROPERTIES
      SWIG_FLAGS -DSWIGWORDSIZE64
    )
  endif()
  if(WIN32)
    set_source_files_properties(${source} PROPERTIES
      SWIG_FLAGS -DSWIGWIN
    )
  endif()
endmacro()

configure_file(swignsparse.swig ${CMAKE_CURRENT_SOURCE_DIR}/swignsparse_avx2.swig COPYONLY)
configure_file(swignsparse.swig ${CMAKE_CURRENT_SOURCE_DIR}/swignsparse_avx512.swig COPYONLY)
configure_file(swignsparse.swig ${CMAKE_CURRENT_SOURCE_DIR}/swignsparse_sve.swig COPYONLY)

configure_swignsparse(swignsparse.swig)
configure_swignsparse(swignsparse_avx2.swig)
configure_swignsparse(swignsparse_avx512.swig)
configure_swignsparse(swignsparse_sve.swig)


set(CMAKE_SWIG_FLAGS -c++ -I${PROJECT_SOURCE_DIR})
set_property(SOURCE swignsparse.swig PROPERTY CPLUSPLUS ON)
set_property(SOURCE swignsparse.swig PROPERTY USE_TARGET_INCLUDE_DIRECTORIES ON)

# Create SWIG Python module
swig_add_library(swignsparse
  TYPE SHARED
  LANGUAGE python
  SOURCES swignsparse.swig
)


set_property(TARGET swignsparse PROPERTY SWIG_COMPILE_OPTIONS -doxygen)

# Macro to add a SWIG module with optional exclusion based on optimization level
macro(add_swig_module MODULE_NAME OPT_LEVELS)
  set_property(SOURCE swig${MODULE_NAME}.swig
    PROPERTY SWIG_MODULE_NAME swig${MODULE_NAME})
  swig_add_library(swig${MODULE_NAME}
    TYPE SHARED
    LANGUAGE python
    SOURCES swig${MODULE_NAME}.swig
  )
  set_property(TARGET swig${MODULE_NAME} PROPERTY SWIG_COMPILE_OPTIONS -doxygen)

  set(_exclude TRUE)
  foreach(_level ${OPT_LEVELS})
    if(NSPARSE_OPT_LEVEL STREQUAL "${_level}")
      set(_exclude FALSE)
      break()
    endif()
  endforeach()

  if(_exclude)
    set_target_properties(swig${MODULE_NAME} PROPERTIES EXCLUDE_FROM_ALL TRUE)
  endif()
endmacro()

add_swig_module(nsparse_avx2 "avx2;avx512")
add_swig_module(nsparse_avx512 "avx512")
add_swig_module(nsparse_sve "sve")

if(NOT WIN32)
  # NOTE: Python does not recognize the dylib extension.
  set_target_properties(swignsparse PROPERTIES SUFFIX .so)
  set_target_properties(swignsparse_avx2 PROPERTIES SUFFIX .so)
  set_target_properties(swignsparse_avx512 PROPERTIES SUFFIX .so)
  set_target_properties(swignsparse_sve PROPERTIES SUFFIX .so)
endif()

foreach(_target swignsparse swignsparse_avx2 swignsparse_avx512 swignsparse_sve)
  target_include_directories(${_target} PRIVATE
      ${PROJECT_SOURCE_DIR}
      ${NUMPY_INCLUDE_DIR}
      ${Python_INCLUDE_DIRS}
  )
  string(REGEX REPLACE "^swig" "" _lib_name ${_target})
  target_link_libraries(${_target} PRIVATE
    ${_lib_name}
    Python::Module
    Python::NumPy
    absl::flat_hash_map
    absl::flat_hash_set
  )
endforeach()

# Set output properties
set_target_properties(swignsparse PROPERTIES
    OUTPUT_NAME "swignsparse"
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}"
    SWIG_USE_TARGET_INCLUDE_DIRECTORIES ON
)

# Copy Python package files to build directory
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/setup.py
    ${CMAKE_CURRENT_BINARY_DIR}/setup.py COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/pyproject.toml
    ${CMAKE_CURRENT_BINARY_DIR}/pyproject.toml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/__init__.py
    ${CMAKE_CURRENT_BINARY_DIR}/__init__.py COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/loader.py
    ${CMAKE_CURRENT_BINARY_DIR}/loader.py COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/class_wrappers.py
    ${CMAKE_CURRENT_BINARY_DIR}/class_wrappers.py COPYONLY)

