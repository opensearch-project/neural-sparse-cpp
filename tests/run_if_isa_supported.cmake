# Copyright OpenSearch Contributors
# SPDX-License-Identifier: Apache-2.0
#
# The OpenSearch Contributors require contributions made to
# this file be licensed under the Apache-2.0 license or a
# compatible open source license.

# Runs ${TEST_EXE} only when the host CPU exposes ${CPU_FLAG} (the /proc/cpuinfo
# flag name, e.g. "avx2" or "avx512f"); otherwise reports a skip and succeeds.
#
# An ISA-specialized binary is compiled entirely with that ISA's -m flags, so
# its static initializers (e.g. a namespace-scope object's constructor) can
# execute the ISA before main() -- and thus before any in-process CPU check.
# On a CPU without the ISA that is an illegal instruction (SIGILL). The only
# safe guard is to decide whether to launch the binary at all, here, before it
# starts. ctest invokes this via `cmake -P`.

if(NOT DEFINED TEST_EXE OR NOT DEFINED CPU_FLAG)
    message(FATAL_ERROR "run_if_isa_supported: TEST_EXE and CPU_FLAG required")
endif()

set(_supported FALSE)
if(EXISTS "/proc/cpuinfo")
    file(READ "/proc/cpuinfo" _cpuinfo)
    # Match CPU_FLAG as a whole token (flags are separated by spaces, and the
    # last one on a line by a newline -- both are non-word characters).
    if(" ${_cpuinfo} " MATCHES "[^a-zA-Z0-9_]${CPU_FLAG}[^a-zA-Z0-9_]")
        set(_supported TRUE)
    endif()
elseif(APPLE)
    execute_process(COMMAND sysctl -a OUTPUT_VARIABLE _sysctl ERROR_QUIET)
    string(TOLOWER "${_sysctl}" _sysctl)
    if(_sysctl MATCHES "${CPU_FLAG}")
        set(_supported TRUE)
    endif()
endif()

if(NOT _supported)
    message(STATUS "SKIP ${TEST_EXE}: host CPU does not expose '${CPU_FLAG}'")
    return()
endif()

execute_process(COMMAND "${TEST_EXE}" RESULT_VARIABLE _rc)
if(NOT _rc EQUAL 0)
    message(FATAL_ERROR "${TEST_EXE} exited with ${_rc}")
endif()
