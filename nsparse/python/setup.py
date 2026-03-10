# Copyright OpenSearch Contributors
# SPDX-License-Identifier: Apache-2.0
#
# The OpenSearch Contributors require contributions made to
# this file be licensed under the Apache-2.0 license or a
# compatible open source license.

"""
Setup script for nsparse Python bindings.

This setup.py works with CMake-built extensions.
Build steps:
  cmake -B build -DNSPARSE_ENABLE_PYTHON=ON && cmake --build build -j
  cd build/nsparse/python && pip install .
"""

import os
import platform
import shutil
import sys
from setuptools import setup
from setuptools.dist import Distribution


class BinaryDistribution(Distribution):
    def has_ext_modules(self):
        return True


def prepare_package():
    """Prepare the nsparse package directory with all available SIMD variants."""
    ext = ".pyd" if platform.system() == "Windows" else ".so"
    target_ext = ".so" if platform.system() != "Windows" else ext

    # All possible SWIG module variants (name, .py file, .so file)
    variants = [
        "swignsparse",
        "swignsparse_avx2",
        "swignsparse_avx512",
        "swignsparse_sve",
    ]

    found_any = False
    for name in variants:
        if os.path.exists(f"_{name}{ext}"):
            found_any = True
            break

    if not found_any:
        print("Error: No extension module found! Build with CMake first:")
        print("  cmake -B build -DNSPARSE_ENABLE_PYTHON=ON -DNSPARSE_OPT_LEVEL=avx512")
        print("  cmake --build build -j")
        print("  cd build/nsparse/python && pip install .")
        sys.exit(1)

    # Create package directory
    shutil.rmtree("nsparse", ignore_errors=True)
    os.mkdir("nsparse")

    # Copy core files
    shutil.copyfile("__init__.py", "nsparse/__init__.py")
    shutil.copyfile("loader.py", "nsparse/loader.py")
    shutil.copyfile("class_wrappers.py", "nsparse/class_wrappers.py")

    # Copy all available variants
    for name in variants:
        lib_file = f"_{name}{ext}"
        py_file = f"{name}.py"
        if os.path.exists(lib_file):
            print(f"Copying {lib_file}")
            shutil.copyfile(py_file, f"nsparse/{py_file}")
            shutil.copyfile(lib_file, f"nsparse/_{name}{target_ext}")


prepare_package()
setup(distclass=BinaryDistribution)
