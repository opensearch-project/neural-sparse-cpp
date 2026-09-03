- [Developer Guide](#developer-guide)
  - [Getting Started](#getting-started)
    - [Fork neural-sparse-cpp Repo](#fork-neural-sparse-cpp-repo)
    - [Install Prerequisites](#install-prerequisites)
      - [C++ Compiler and Build Tools](#c-compiler-and-build-tools)
      - [Python (Optional)](#python-optional)
  - [Build](#build)
    - [Build Options](#build-options)
    - [SIMD Optimization Levels](#simd-optimization-levels)
    - [GPU Acceleration](#gpu-acceleration)
  - [Run Tests](#run-tests)
  - [Run Benchmarks](#run-benchmarks)
  - [Python Bindings](#python-bindings)
    - [Build Python Bindings](#build-python-bindings)
      - [Using venv](#using-venv)
      - [Using Conda](#using-conda)
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
| `NSPARSE_ENABLE_GPU` | `OFF` | GPU-accelerate index building via cuSPARSE (see [GPU Acceleration](#gpu-acceleration)) |

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

### GPU Acceleration

`NSPARSE_ENABLE_GPU=ON` compiles optional NVIDIA GPU acceleration for the
**index-building** (`build()`) path of `SeismicIndex`, using cuSPARSE plus a few
custom CUDA kernels. This offloads the two heaviest build phases — the k-means
document→centroid assignment and the `summarize()` per-term max-pool. Search is
unaffected and always runs on the CPU. The option is `OFF` by default, so
CPU-only builds and platforms without a CUDA toolkit are unaffected.

Requirements:

- NVIDIA CUDA Toolkit (nvcc, cuSPARSE, cudart) and a CUDA-capable GPU
- `float` (`U32`) weights — quantized (`seismic_sq`) indices fall back to the CPU

Build:

```bash
cmake -S . -B build -DNSPARSE_ENABLE_GPU=ON
cmake --build build -j
```

If the CUDA toolkit is in a non-standard location (for example, a pip-packaged
`nvidia-cu*` wheel), point CMake at it:

```bash
cmake -S . -B build -DNSPARSE_ENABLE_GPU=ON \
  -DNSPARSE_CUDA_TOOLKIT_ROOT=/path/to/nvidia/cuXX
```

Target GPU architectures default to `80;89` (Ampere / Ada); override with
`-DNSPARSE_CUDA_ARCHITECTURES=...`. When built with GPU support and a device is
present, assignment is offloaded unconditionally. The `summarize()` max-pool
offload is opt-in via the `NSPARSE_GPU_SUMMARIZE=1` environment variable; the
default keeps summarize on the CPU, which is faster on high-core hosts and
produces identical output. The GPU build wiring lives in
[`nsparse/gpu/CMakeLists.txt`](nsparse/gpu/CMakeLists.txt).

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

### Python integration tests

`python_tests/` holds black-box tests that drive the library the way a user of
the Python bindings does: only what SWIG exposes, with no access to internals.
They complement the GoogleTest suite rather than duplicating it -- the C++ tests
cover the internals SWIG deliberately hides (`MmapCursor`, borrowing
`map_vectors` views, in-memory `BufferedIOWriter` round-trips).

They need the bindings built and installed:

```bash
cmake -S . -B build -DNSPARSE_ENABLE_PYTHON=ON
cmake --build build -j
pip install "numpy<2.0" pytest
pip install --no-deps build/nsparse/python
pytest python_tests -v
```

One file per index type, named after the use case being exercised
(`test_happy_case`, `test_with_id_map`, `test_exact_match`, ...). Accuracy is
checked against an independent numpy brute-force oracle in
`python_tests/oracle.py`: exact indexes must match it outright, approximate ones
must clear a recall floor.

Seismic builds are nondeterministic by default -- every posting list draws fresh
entropy for its initial centroids -- so keep recall floors loose. Add `seed=` to
the factory description when a test needs the build to reproduce itself:

```python
nsparse.index_factory(dim, "seismic,lambda=25|beta=4|alpha=0.4|seed=42")
```

A seeded build is also independent of `OMP_NUM_THREADS`, because each list's
seed is derived from its own index rather than from the order OpenMP happens to
schedule the lists in.

`test_threading.py` runs each case under several `OMP_NUM_THREADS` values in
subprocesses, which is required because the OpenMP runtime reads that variable
when it initialises.

## Run Benchmarks

Build with benchmarks enabled and run:

```bash
cmake -S . -B build -DNSPARSE_ENABLE_BENCHMARKS=ON
cmake --build build -j
./build/benchmarks/nsparse_benchmark
```

On Linux, the benchmarks support hardware performance counters via [libpfm](http://perfmon2.sourceforge.net/). Install `libpfm4-dev` to enable this.

## Building an index larger than memory

A whole-corpus build holds two intermediates that scale with the corpus's
non-zeros — the inverted lists (every posting) and then the clustered posting
lists — so peak memory scales with the corpus, and a corpus whose posting lists
do not fit in RAM cannot be indexed at all.

Batching splits the term space into contiguous windows and finishes one window
before starting the next. It is a build option on the seismic family rather than
a separate entry point, so it is set in the factory description alongside
`lambda` and `beta`, and every type in the family gets it — `seismic`,
`seismic_sq`, `disk_seismic`, `disk_seismic_sq`:

| Option | Effect |
|---|---|
| `inverted_list_batch_size=N` | Build in `N` term windows. Bounds the inverted-list intermediate to one window; the index is still built in memory as usual. |
| `batch_file_output_path=P` | With `N > 1`, write the index to `P` as it is built rather than assembling it in memory, so the clustered lists are never all resident either, then borrow them back from `P` by mapping it. Unused at `N <= 1`, which is an ordinary build and already holds its own lists. |

`seismic` and `seismic_sq` end their payload with their posting lists, so each
window is serialized straight into `P` and dropped. The disk-resident pair cannot
be written that way: their summaries are followed by an inline forward index whose
blocks are laid out from the doc-id membership of *every* list, which is not known
until the last window is clustered. They spill the clustered lists to `P.lists`
instead, map them back, and write the payload from that mapping — same bound on
anonymous memory, at the cost of scratch disk the size of the lists. The spill is
deleted with the build; nothing else reads it.

```cpp
auto* index = nsparse::index_factory(
    dimension,
    "seismic,lambda=6000|beta=400|alpha=0.4"
    "|inverted_list_batch_size=10|batch_file_output_path=/data/index.dat");

// Corpus residency is SparseVectors' business, not the build's: read_csr can
// map a native-layout CSR instead of copying it, and the build is unchanged.
index->read_csr("corpus.mcsr", nsparse::Residency::kMmap);
index->build();   // streams to /data/index.dat, then maps its lists back in

// Ready to serve, with no reopening by path: the posting lists are borrowed from
// the file just written, and the corpus is still borrowed from its own mapping.
index->search(...);
```

The same from Python, since it is only a description string:

```python
native = nsparse.native_path("corpus.csr")
nsparse.convert("corpus.csr", native)
index = nsparse.index_factory(
    dim,
    "seismic,lambda=6000|beta=400|alpha=0.4"
    "|inverted_list_batch_size=10|batch_file_output_path=/data/index.dat",
)
index.read_csr(native, nsparse.Residency_kMmap)
index.build()          # streams out, then maps its lists back in
dists, labels = index.search(n, indptr, indices, values, k)
```

The file is an ordinary index of its type — byte-for-byte what `write_index`
would have produced from the equivalent whole-corpus build. That is asserted
rather than assumed: at a fixed `seed` the two are compared as files, for all four
types. Each posting list's k-means seed comes from its own *global* term id and
`lambda`/`beta` are resolved once from the whole corpus, so the window count
cannot change what is produced.

### Choosing `inverted_list_batch_size`

Windows are cut to equal estimated *memory*, not equal width. Term frequencies are
heavily skewed, and peak memory is set by the largest window, so an uneven split
wastes most of what batching could save.

A window has two memory peaks and the split has to weigh both. Filling it holds
every posting of its terms; clustering it holds what survives pruning
(`min(count, lambda)` per term) as clusters and summaries, an order of magnitude
bulkier per posting. Weighting either phase alone unbalances the other, both worse
than weighting their sum — see `make_windows` in `nsparse/seismic_common.cpp`.

`RssAnon` is the figure to watch, being what the process itself allocated;
`RssFile` is pages it touched of a mapping, which the kernel can reclaim under
pressure. Read as total RSS the win looks far smaller than it is: an index that
maps its corpus, and then maps its own output back, keeps most of its residency in
page cache, which total RSS counts and pressure reclaims. Batching moves the
anonymous column, so report the split rather than the total.

What to expect from the shape of it: anonymous memory falls faster than 1/N,
because the split comes from the real per-term costs rather than from term ids,
and build time is flat to around ten windows and then climbs, because every window
makes its own pass over the corpus. Ten to twenty windows is usually the useful
range — most of the memory saving for a few percent of build time. The floor is
what the build cannot batch: one window plus whatever the corpus itself costs.

The disk-resident pair behaves the same way, with two differences. Build time
climbs sooner, since the spill is written and read back and the payload write is
another pass over the lists; and `lambda` is the knob to watch for output size,
because their inline forward index copies every pruned posting's whole doc
vector — at a large `lambda` that section alone can dwarf the rest of the index,
and the spill needs scratch disk beside it. `disk_seismic_sq` cannot map a float
CSR (it searches over codes), so its corpus stays on the heap and shows up in the
same column the build's own growth does; subtract the reported
`start_rss_anon_mb`.

Mapping the finished lists back in is nearly free in the column that matters:
borrowing an index costs a fraction of a second and single-digit megabytes of
anonymous memory, against copying it, which costs its whole size — the cursor
reads the size header before each array and skips the bulk, so it faults in part
of the file as reclaimable page cache. That mapping is separate from the corpus's:
an index that mapped its corpus with `read_csr` keeps doing so, since it still
scores from it.

Compare like with like. A build that loads the corpus is not comparable to one
that maps it, and the difference is more than the corpus: a streaming ingest
stages a second copy that the allocator retains rather than returning to the OS,
so do not read a heap figure and a mapped figure as differing by a fixed offset.
Measure from the start of the build with the corpus already resident, or a
whole-process high-water mark reports the loader and hides everything below it.
Peak RSS comes from `VmHWM`, a kernel counter; the anon and file peaks have no
such counter and are sampled, so they are lower bounds and can disagree with
`VmHWM` by a hair.

Query performance does not move, because the index is the same index: identical
byte for byte at a fixed seed, and indistinguishable in latency, QPS and recall
for the random-seeded default.

Write the index to a real disk. On a tmpfs such as `/tmp` it is RAM, which
defeats the point.

### Measuring it

`benchmarks/batched_build_mem_bench` reports peak RSS (`VmHWM`) and wall time for
one configuration per process — `google-benchmark` measures throughput, and a
high-water mark is only clean in a process that has built nothing else. Use
`inmem` to compare against `baseline`: both then hold the corpus on the heap, so
the difference is the batching rather than the residency.

```bash
cmake -S . -B build -DNSPARSE_ENABLE_BENCHMARKS=ON && cmake --build build -j
B=./build/benchmarks/batched_build_mem_bench
$B convert corpus.csr corpus.mcsr
$B baseline corpus.csr 6000 400 0.4               # whole-corpus build, for reference
$B batched inmem corpus.csr 6000 400 0.4 10 /data # 10 windows, same corpus residency
$B batched mmap corpus.mcsr 6000 400 0.4 10 /data # ... or with the corpus mapped

# Any type in the family, as its factory name. Both arms need it, and the
# baseline's index path is positional -- pass "" to skip writing one.
$B baseline corpus.csr 6000 400 0.4 "" disk_seismic
$B batched mmap corpus.mcsr 6000 400 0.4 10 /data disk_seismic
```

## Python Bindings

### Build Python Bindings

#### Using venv

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r nsparse/python/requirements.txt
cmake -S . -B build -DNSPARSE_ENABLE_PYTHON=ON -DNSPARSE_OPT_LEVEL=avx2
cmake --build build -j
cd build/nsparse/python
pip install .
```

#### Using Conda

```bash
conda create -n nsparse python=3.12 numpy
conda activate nsparse
cmake -S . -B build -DNSPARSE_ENABLE_PYTHON=ON -DNSPARSE_OPT_LEVEL=avx2
cmake --build build -j
cd build/nsparse/python
pip install .
```

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

In VS Code, you can set breakpoints and debug directly from the IDE using the test or benchmark targets.

### Major Dependencies

| Dependency | Purpose | Acquisition |
|---|---|---|
| [Abseil](https://github.com/abseil/abseil-cpp) | Hash containers (`flat_hash_set`, `flat_hash_map`) | System or auto-fetched |
| [GoogleTest](https://github.com/google/googletest) | Unit testing framework | Auto-fetched via CMake |
| [Google Benchmark](https://github.com/google/benchmark) | Benchmarking framework | Auto-fetched via CMake |
| OpenMP | Parallelism | System package |
| SWIG | Python bindings generation | System package |
| CUDA Toolkit / cuSPARSE | GPU-accelerated index building (only with `NSPARSE_ENABLE_GPU=ON`) | System or CUDA pip wheel |

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
- `nsparse/gpu/` — Optional CUDA/cuSPARSE build-time acceleration (`NSPARSE_ENABLE_GPU`)
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

If your change alters what the Python bindings expose or how an index behaves end to end, add a black-box case to `python_tests/` as well. See [Python integration tests](#python-integration-tests).

If your changes could affect backward compatibility, please include relevant tests along with your PR.

### Index file format

Every serialized index starts with a fixed header (`nsparse::IndexHeader` in `nsparse/io/io.h`). It is written and parsed centrally, by `write_header`/`read_header` in `nsparse/io/index_io.cpp` — an index type never reads its own header, it receives the already-parsed one:

| Field | Type | Notes |
|---|---|---|
| id | `uint32` | fourcc of the index type, e.g. `SEIS` |
| version | `uint32` | layout revision of the payload that follows |
| dimension | `int32` | |

The payload follows immediately, and its layout is the index type's own business. A type parses it in `read_index` (copying), `mmap_index` (borrowed from a file mapping), or both — `DiskSeismicIndex` is mmap-only and its `read_index` just throws, while `IDMapIndex` has no `mmap_index` at all and instead threads `io_flags` down to its delegate.

Versions are numbered **per index type**, not per file: `IndexIO::format_version()` returns the type's own `kFormatVersion`, so revising one type's payload leaves the others' numbering alone. An `IDMapIndex` writes its own header for the id map and then a second, complete header for the delegate it wraps, each with its own version.

The same central code rejects a version outside `1..format_version()` before dispatching to either parse path — a file from a newer build fails with a clear error instead of consuming whatever its fields happen to align with.

To change a payload layout:

1. Bump that type's `kFormatVersion`.
2. Branch on `header.version` wherever that type actually parses its payload — its `read_index` and/or its `mmap_index` — keeping the older branch so existing files still load.
3. Add a round-trip test for the new version and a test that reads the old layout.

### Outdated or irrelevant code

Do not submit code that is not used or needed, even if it's commented. We rely on GitHub as a version control system; code can be restored if needed.
