# Copyright OpenSearch Contributors
# SPDX-License-Identifier: Apache-2.0
#
# The OpenSearch Contributors require contributions made to
# this file be licensed under the Apache-2.0 license or a
# compatible open source license.

# Sanitizer instrumentation, off by default.
#
# nsparse parses binary index files that may be truncated or corrupt. The
# negative-path tests assert that an exception is thrown, but not that no
# out-of-bounds read, overflow or misaligned access happened on the way there --
# in an in-process native library those are far worse than an exception. ASan and
# UBSan are what turn those assertions into memory-safety assertions.
option(NSPARSE_ENABLE_SANITIZERS "Instrument the build with sanitizers" OFF)
set(NSPARSE_SANITIZERS "address,undefined" CACHE STRING
    "Comma-separated -fsanitize list used when NSPARSE_ENABLE_SANITIZERS=ON")

if(NOT NSPARSE_ENABLE_SANITIZERS)
    return()
endif()

if(MSVC)
    # MSVC only ships /fsanitize=address, and spells it differently; nothing
    # here would apply.
    message(FATAL_ERROR "NSPARSE_ENABLE_SANITIZERS is only supported with GCC and Clang")
endif()

# Applied globally, before any target (including the FetchContent'ed abseil and
# GoogleTest) is defined: an ASan-instrumented translation unit and an
# uninstrumented one disagree about libstdc++ container annotations, which shows
# up as container-overflow false positives rather than a link error.
#
# -fno-sanitize-recover=all: by default UBSan prints and continues, so a
# findings-only run still exits 0 and ctest reports a pass. Aborting is what
# makes a UBSan finding fail CI.
set(NSPARSE_SANITIZER_FLAGS
    -fsanitize=${NSPARSE_SANITIZERS}
    -fno-sanitize-recover=all
    -fno-omit-frame-pointer)

# GCC's null-pointer checks make the address of a function template
# instantiation non-constant, which breaks abseil's constexpr
# `get_hash_slot_fn() == nullptr` dispatch (hash_policy_traits.h) -- every
# translation unit that includes flat_hash_map fails to compile. Clang is
# unaffected. Dropping the three checks costs little here: a null dereference
# still surfaces, as an ASan SEGV report.
if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU" AND NSPARSE_SANITIZERS MATCHES "undefined")
    list(APPEND NSPARSE_SANITIZER_FLAGS
         -fno-sanitize=null,nonnull-attribute,returns-nonnull-attribute)
endif()

add_compile_options(${NSPARSE_SANITIZER_FLAGS})
add_link_options(${NSPARSE_SANITIZER_FLAGS})

message(STATUS "Sanitizers enabled: ${NSPARSE_SANITIZERS}")
