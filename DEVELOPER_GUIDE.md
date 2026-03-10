- [Developer Guide](#developer-guide)
  - [Getting Started](#getting-started)
    - [Fork neural-sparse-cpp Repo](#fork-neural-sparse-cpp-repo)
    - [Install Prerequisites](#install-prerequisites)
      - [C++ Compiler and Build Tools](#c-compiler-and-build-tools)
      - [Python (Optional)](#python-optional)
  - [Build](#build)
    - [Build Options](#build-options)
    - [SIMD Optimization Levels](#simd-optimization-levels)
  - [Run Tests](#run-tests)
  - [Run Benchmarks](#run-benchmarks)
  - [Python Bindings](#python-bindings)
    - [Build Python Bindings](#build-python-bindings)
    - [Python Environment Setup](#python-environment-setup)
    - [Python Usage](#python-usage)
  - [Debugging](#debugging)
    - [Major Dependencies](#major-dependencies)
  - [Submitting Changes](#submitting-changes)
  - [Code Guidelines](#code-guidelines)
    - [File and class names](#file-and-class-names)
    - [Modular code](#modular-code)
    - [Documentation](#documentation)
    - [Code style](#code-style)
    - [Style and Formatting Check](#style-and-formatting-check)
    - [Tests](#tests)
    - [Outdated or irrelevant code](#outdated-or-irrelevant-code)

# Developer Guide

So you want to contribute code to neural-sparse-cpp? Excellent! We're glad you're here. Here's what you need to do.

## Getting Started

### Fork neural-sparse-cpp Repo

Fork [opensearch-project/neural-sparse-cpp](https://github.com/opensearch-project/neural-sparse-cpp) and clone locally.

Example:
```bash
git clone https://github.com/[your username]/neural-sparse-cpp.git
```

### Install Prerequisites

#### C++ Compiler and Build Tools

neural-sparse-cpp requires C++20 and uses CMake as its build system. You will need:

- A C++20 compatible compiler (with OpenMP support version 2 or higher), such as GCC 11+, Clang 14+, or MSVC 2022+
- CMake 3.15 or higher
- OpenMP
- SWIG (only if building Python bindings)

**Linux (Ubuntu/Debian)**
```bash
sudo apt update
sudo apt install -y g++ cmake libomp-dev swig libabsl-dev
```

**macOS**
```bash
brew install cmake libomp swig abseil
```

> Note: [Abseil](https://github.com/abseil/abseil-cpp) is an optional system dependency. If not found, CMake will automatically fetch it from GitHub during the build.

#### Python (Optional)

If you plan to build or use the Python bindings, you will also need:

- Python 3.8+ with development headers
- pip

**Linux (Ubuntu/Debian)**
```bash
sudo apt install -y python3-dev python3-pip
```

> Note: Replace `python3-dev` with your specific version package (e.g., `python3.12-dev`) if needed.

## Build

Configure and build the project using CMake:

```bash
cmake -S . -B build
cmake --build build -j
```

### Build Options

| Option | Default | Description |
|---|---|---|
| `NSPARSE_OPT_LEVEL` | `generic` | SIMD optimization level |
| `NSPARSE_ENABLE_PYTHON` | `OFF` | Build Python bindings |
| `NSPARSE_ENABLE_TESTS` | `OFF` | Build unit tests |
| `NSPARSE_ENABLE_BENCHMARKS` | `OFF` | Build benchmarks |

Example with multiple options:
```bash
cmake -S . -B build -DNSPARSE_ENABLE_TESTS=ON -DNSPARSE_OPT_LEVEL=avx2
cmake --build build -j
```

### SIMD Optimization Levels

The `NSPARSE_OPT_LEVEL` option controls which SIMD instruction sets are compiled:

| Value | Architecture | Description |
|---|---|---|
| `generic` | Any | No SIMD specialization (default) |
| `avx2` | x86_64 | AVX2 + FMA + F16C + POPCNT |
| `avx512` | x86_64 | AVX-512 (F, CD, VL, DQ, BW) + AVX2 |
| `sve` | ARM (non-Apple) | Scalable Vector Extension |

> Note: ARM NEON is used automatically on ARM platforms. SVE is not supported on Apple Silicon.

## Run Tests

Build with tests enabled and run:

```bash
cmake -S . -B build -DNSPARSE_ENABLE_TESTS=ON
cmake --build build -j
ctest --test-dir build --output-on-failure
```

To run specific test suites using GoogleTest filters:

```bash
./build/tests/nsparse_test --gtest_filter="SparseVectors*"
./build/tests/nsparse_test --gtest_filter="SeismicIndex*"
```

## Run Benchmarks

Build with benchmarks enabled and run:

```bash
cmake -S . -B build -DNSPARSE_ENABLE_BENCHMARKS=ON
cmake --build build -j
./build/benchmarks/nsparse_benchmark
```

On Linux, the benchmarks support hardware performance counters via [libpfm](http://perfmon2.sourceforge.net/). Install `libpfm4-dev` to enable this.

## Python Bindings

### Build Python Bindings

```bash
cmake -S . -B build -DNSPARSE_ENABLE_PYTHON=ON -DNSPARSE_OPT_LEVEL=avx2
cmake --build build -j
cd build/nsparse/python
pip install .
```

### Python Environment Setup

**Using venv**
```bash
python3 -m venv venv
source venv/bin/activate
```

**Using Conda**
```bash
conda create -n nsparse python=3.12
conda activate nsparse
```

Then build and install the Python bindings as described above.

### Python Usage

After building and installing, you can run the demo scripts:

```bash
python demos/seismic_sq.py
python demos/seismic_sq_idmap.py
python demos/seismic_sq_idmap_idselector.py
```

## Debugging

For debugging with GDB or LLDB, build in Debug mode:

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Debug -DNSPARSE_ENABLE_TESTS=ON
cmake --build build -j
```

Then attach your debugger to the test binary:

```bash
# GDB
gdb ./build/tests/nsparse_test

# LLDB
lldb ./build/tests/nsparse_test
```

In CLion or VS Code, you can set breakpoints and debug directly from the IDE using the test or benchmark targets.

### Major Dependencies

| Dependency | Purpose | Acquisition |
|---|---|---|
| [Abseil](https://github.com/abseil/abseil-cpp) | Hash containers (`flat_hash_set`, `flat_hash_map`) | System or auto-fetched |
| [GoogleTest](https://github.com/google/googletest) | Unit testing framework | Auto-fetched via CMake |
| [Google Benchmark](https://github.com/google/benchmark) | Benchmarking framework | Auto-fetched via CMake |
| OpenMP | Parallelism | System package |
| SWIG | Python bindings generation | System package |

## Submitting Changes

See [CONTRIBUTING](CONTRIBUTING.md).

## Code Guidelines

### File and class names

Class names should use `CamelCase`. File names should use `snake_case`.

Header files use the `.h` extension and source files use `.cpp`.

Try to put new classes into existing directories if the directory name abstracts the purpose of the class. The project is organized as follows:

- `nsparse/` — Core library (index types, sparse vectors, inverted index)
- `nsparse/cluster/` — Clustering algorithms (k-means, inverted list clusters)
- `nsparse/invlists/` — Inverted list storage
- `nsparse/io/` — Serialization and I/O
- `nsparse/utils/` — Utilities (distance functions, SIMD, quantization, ranker)
- `nsparse/python/` — Python bindings (SWIG)

### Modular code

Organize code into small classes and methods with a single concise purpose. Prefer multiple small methods over a single long one that does everything.

### Documentation

Document your code. That includes the purpose of new classes, every public method, and code sections that have critical or non-trivial logic.

Use C++ style comments:
```cpp
/**
 * Brief description of the class/method.
 *
 * @param name Description of parameter
 * @return Description of return value
 */
```

### Code style

The project uses [Google C++ Style](https://google.github.io/styleguide/cppguide.html) as a base with 4-space indentation, configured via `.clang-format`:

```
BasedOnStyle: Google
IndentWidth: 4
AccessModifierOffset: -4
```

Additional conventions:
1. Use descriptive names for classes, methods, fields, and variables.
2. Avoid abbreviations unless they are widely accepted.
3. Use `const` wherever possible.
4. Prefer smart pointers (`std::unique_ptr`, `std::shared_ptr`) over raw pointers for ownership.
5. Use `override` on all overridden virtual methods.
6. SWIG `.i` files are excluded from formatting (see `.clang-format-ignore`).

### Style and Formatting Check

The project uses `clang-format` for code formatting and `clang-tidy` for static analysis.

To format code:
```bash
# Format a single file
clang-format -i nsparse/index.cpp

# Format all source files
find nsparse -name '*.cpp' -o -name '*.h' | xargs clang-format -i
```

To run static analysis:
```bash
# Run clang-tidy on a single file (requires compile_commands.json)
cmake -S . -B build -DCMAKE_EXPORT_COMPILE_COMMANDS=ON
clang-tidy -p build nsparse/index.cpp
```

The `.clang-tidy` configuration enables checks from `bugprone-*`, `modernize-*`, `performance-*`, and `readability-*` categories.

### Tests

Write unit tests for your new functionality using GoogleTest. Tests live in the `tests/` directory with the naming convention `<module>_test.cpp`.

Unit tests are preferred as they are fast and cheap. Try to cover all possible combinations of parameters.

If your changes could affect backward compatibility, please include relevant tests along with your PR.

### Outdated or irrelevant code

Do not submit code that is not used or needed, even if it's commented. We rely on GitHub as a version control system; code can be restored if needed.
