# Copyright OpenSearch Contributors
# SPDX-License-Identifier: Apache-2.0
#
# The OpenSearch Contributors require contributions made to
# this file be licensed under the Apache-2.0 license or a
# compatible open source license.

# NSPARSE loader - runtime CPU feature detection and module loading

import platform
import subprocess
import logging
import os
import sys

from packaging.version import Version


def supported_instruction_sets():
    """
    Returns the set of supported CPU features, see
    https://github.com/numpy/numpy/blob/master/numpy/core/src/common/npy_cpu_features.h
    for the list of features that this set may contain per architecture.
    """

    def is_sve_supported():
        if platform.machine() != "aarch64":
            return False
        if platform.system() != "Linux":
            return False
        import numpy

        if Version(numpy.__version__) >= Version("2.0"):
            return False
        try:
            import numpy.distutils.cpuinfo

            return (
                "sve" in numpy.distutils.cpuinfo.cpu.info[0].get("Features", "").split()
            )
        except ImportError:
            return bool(__import__("ctypes").CDLL(None).getauxval(16) & (1 << 22))

    import numpy

    if Version(numpy.__version__) >= Version("1.19"):
        from numpy._core._multiarray_umath import __cpu_features__

        supported = {k for k, v in __cpu_features__.items() if v}
        if is_sve_supported():
            supported.add("SVE")
        for f in os.getenv("NSPARSE_DISABLE_CPU_FEATURES", "").split(", \t\n\r"):
            supported.discard(f)
        return supported

    # Legacy fallback before numpy 1.19
    if platform.system() == "Darwin":
        if (
            subprocess.check_output(["/usr/sbin/sysctl", "hw.optional.avx2_0"])[-1]
            == "1"
        ):
            return {"AVX2"}
    elif platform.system() == "Linux":
        import numpy.distutils.cpuinfo

        result = set()
        if "avx2" in numpy.distutils.cpuinfo.cpu.info[0].get("flags", ""):
            result.add("AVX2")
        if "avx512" in numpy.distutils.cpuinfo.cpu.info[0].get("flags", ""):
            result.add("AVX512")
        if is_sve_supported():
            result.add("SVE")
        for f in os.getenv("NSPARSE_DISABLE_CPU_FEATURES", "").split(", \t\n\r"):
            result.discard(f)
        return result
    return set()


logger = logging.getLogger(__name__)

instruction_sets = None

# Try to load optimization level from env variable
opt_env_variable_name = "NSPARSE_OPT_LEVEL"
opt_level = os.environ.get(opt_env_variable_name, None)

if opt_level is None:
    logger.debug(
        f"Environment variable {opt_env_variable_name} is not set, "
        "picking instruction set according to current CPU"
    )
    instruction_sets = supported_instruction_sets()
else:
    logger.debug(f"Using {opt_level} as instruction set.")
    instruction_sets = {opt_level}

loaded = False

has_AVX512 = any("AVX512" in x.upper() for x in instruction_sets)
if has_AVX512 and not loaded:
    try:
        logger.info("Loading nsparse with AVX512 support.")
        from .swignsparse_avx512 import *

        logger.info("Successfully loaded nsparse with AVX512 support.")
        loaded = True
    except ImportError as e:
        logger.info(f"Could not load library with AVX512 support: {e!r}")
        loaded = False

has_AVX2 = "AVX2" in instruction_sets
if has_AVX2 and not loaded:
    try:
        logger.info("Loading nsparse with AVX2 support.")
        from .swignsparse_avx2 import *

        logger.info("Successfully loaded nsparse with AVX2 support.")
        loaded = True
    except ImportError as e:
        logger.info(f"Could not load library with AVX2 support: {e!r}")
        loaded = False

has_SVE = "SVE" in instruction_sets
if has_SVE and not loaded:
    try:
        logger.info("Loading nsparse with SVE support.")
        from .swignsparse_sve import *

        logger.info("Successfully loaded nsparse with SVE support.")
        loaded = True
    except ImportError as e:
        logger.info(f"Could not load library with SVE support: {e!r}")
        loaded = False

if not loaded:
    try:
        logger.info("Loading nsparse (generic).")
        from .swignsparse import *

        logger.info("Successfully loaded nsparse.")
    except ModuleNotFoundError:
        formatted_ins_sets = ", ".join(supported_instruction_sets())
        message = (
            f"No nsparse SWIG module found. Supported instruction sets on this system:\n"
            f"{formatted_ins_sets}\n\n"
            f"Build with appropriate NSPARSE_OPT_LEVEL (avx512, avx2, sve, or generic)."
        )
        logger.error(message)
        sys.exit(1)

# Apply class wrappers to provide Pythonic interface
import sys
from importlib import import_module

# Use importlib to avoid circular import issues
class_wrappers = import_module(".class_wrappers", package=__package__)
class_wrappers.handle_all_classes(sys.modules[__name__])
