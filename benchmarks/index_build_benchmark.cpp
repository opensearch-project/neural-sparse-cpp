/**
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 *
 * The OpenSearch Contributors require contributions made to
 * this file be licensed under the Apache-2.0 license or a
 * compatible open source license.
 *
 * Benchmarks SeismicIndex build() time on CPU vs GPU (cuSPARSE).
 *
 * The GPU path accelerates the k-means document->centroid assignment step of
 * build() via cuSPARSE SpMM; the rest of build() (inverted-list construction,
 * summarization) stays on CPU. GPU vs CPU is selected at run time through the
 * NSPARSE_GPU_MIN_DOCS gate so both paths run from the same binary:
 *   - CPU:  set the gate above the largest inverted list (no list offloads).
 *   - GPU:  set the gate to 1 (every eligible list offloads).
 *
 * Data is the Big-ANN sparse-vectors (SPLADE MS MARCO) CSR file, e.g.
 * data/base_small.csr from cpp-sparse-ann's dataset.py.
 *
 * Usage:
 *   NSPARSE_DATA_CSR=/path/to/base_small.csr \
 *     ./nsparse_build_benchmark --benchmark_min_time=1x
 */

#include <benchmark/benchmark.h>

#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "nsparse/seismic_common.h"
#include "nsparse/seismic_index.h"
#include "nsparse/types.h"

#ifdef NSPARSE_WITH_CUDA
#include "nsparse/gpu/gpu_cluster_assigner.h"
#endif

namespace {

struct CSRMatrix {
    int64_t nrow;
    int64_t ncol;
    int64_t nnz;
    std::vector<nsparse::idx_t> indptr;
    std::vector<nsparse::term_t> indices;
    std::vector<float> data;
};

CSRMatrix read_csr(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f.is_open()) {
        throw std::runtime_error("Cannot open CSR file: " + path);
    }

    CSRMatrix m;
    int64_t sizes[3];
    f.read(reinterpret_cast<char*>(sizes), sizeof(sizes));
    m.nrow = sizes[0];
    m.ncol = sizes[1];
    m.nnz = sizes[2];

    std::vector<int64_t> indptr64(m.nrow + 1);
    f.read(reinterpret_cast<char*>(indptr64.data()),
           static_cast<std::streamsize>((m.nrow + 1) * sizeof(int64_t)));
    m.indptr.resize(m.nrow + 1);
    for (int64_t i = 0; i <= m.nrow; ++i) {
        m.indptr[i] = static_cast<nsparse::idx_t>(indptr64[i]);
    }

    std::vector<int32_t> indices32(m.nnz);
    f.read(reinterpret_cast<char*>(indices32.data()),
           static_cast<std::streamsize>(m.nnz * sizeof(int32_t)));
    m.indices.resize(m.nnz);
    for (int64_t i = 0; i < m.nnz; ++i) {
        m.indices[i] = static_cast<nsparse::term_t>(indices32[i]);
    }

    m.data.resize(m.nnz);
    f.read(reinterpret_cast<char*>(m.data.data()),
           static_cast<std::streamsize>(m.nnz * sizeof(float)));

    return m;
}

std::string get_env_or_die(const char* name) {
    const char* val = std::getenv(name);
    if (val == nullptr || val[0] == '\0') {
        throw std::runtime_error(std::string("Environment variable not set: ") +
                                 name);
    }
    return val;
}

// Loads the data CSR exactly once and shares it across benchmark cases.
const CSRMatrix& shared_data() {
    static CSRMatrix data = [] {
        std::string path = get_env_or_die("NSPARSE_DATA_CSR");
        std::cout << "Loading data CSR: " << path << "\n";
        CSRMatrix m = read_csr(path);
        std::cout << "  rows=" << m.nrow << " cols=" << m.ncol
                  << " nnz=" << m.nnz << "\n";
        return m;
    }();
    return data;
}

// Seismic cluster parameters comparable to the search benchmark's index.
constexpr nsparse::SeismicClusterParameters kParams = {
    .lambda = 6000, .beta = 400, .alpha = 0.4F};

// Force the GPU-offload gate one way or the other for this process. Setting the
// minimum-docs threshold to 1 offloads every eligible list; setting it beyond
// the largest inverted list keeps the whole build on the CPU.
void set_gpu_offload(bool enabled) {
    if (enabled) {
        ::setenv("NSPARSE_GPU_MIN_DOCS", "1", /*overwrite=*/1);
    } else {
        // Larger than lambda (max pruned posting length), so no list offloads.
        ::setenv("NSPARSE_GPU_MIN_DOCS", "1000000000", /*overwrite=*/1);
    }
}

// Builds a fresh SeismicIndex from the shared corpus and times only build().
void run_build(benchmark::State& state, bool use_gpu) {
    const CSRMatrix& data = shared_data();
    for (auto _ : state) {
        state.PauseTiming();
        auto index = std::make_unique<nsparse::SeismicIndex>(
            static_cast<int>(data.ncol), kParams);
        index->add(static_cast<nsparse::idx_t>(data.nrow), data.indptr.data(),
                   data.indices.data(), data.data.data());
        set_gpu_offload(use_gpu);
        state.ResumeTiming();

        index->build();

        benchmark::DoNotOptimize(index.get());
        benchmark::ClobberMemory();
    }
    state.counters["docs"] = static_cast<double>(data.nrow);
    state.counters["nnz"] = static_cast<double>(data.nnz);
}

// Iterations / repetitions are configurable so the same binary can run a
// robust multi-sample sweep on small corpora and a single build on very large
// ones (where one full build already takes many minutes).
int env_int(const char* name, int fallback) {
    if (const char* v = std::getenv(name); v != nullptr && v[0] != '\0') {
        return std::atoi(v);
    }
    return fallback;
}

}  // namespace

// ---------------------------------------------------------------------------
// CPU build (assignment on CPU)
// ---------------------------------------------------------------------------
static void BM_Seismic_Build_CPU(benchmark::State& state) {
    run_build(state, /*use_gpu=*/false);
}
BENCHMARK(BM_Seismic_Build_CPU)
    ->Unit(benchmark::kMillisecond)
    ->UseRealTime()
    ->Iterations(env_int("NSPARSE_BENCH_ITERS", 3))
    ->Repetitions(env_int("NSPARSE_BENCH_REPS", 3));

// ---------------------------------------------------------------------------
// GPU build (assignment via cuSPARSE) - only when compiled with CUDA
// ---------------------------------------------------------------------------
#ifdef NSPARSE_WITH_CUDA
static void BM_Seismic_Build_GPU(benchmark::State& state) {
    if (!nsparse::detail::GpuClusterAssigner::available()) {
        state.SkipWithError("No CUDA-capable GPU available");
        return;
    }
    run_build(state, /*use_gpu=*/true);
}
BENCHMARK(BM_Seismic_Build_GPU)
    ->Unit(benchmark::kMillisecond)
    ->UseRealTime()
    ->Iterations(env_int("NSPARSE_BENCH_ITERS", 3))
    ->Repetitions(env_int("NSPARSE_BENCH_REPS", 3));
#endif

BENCHMARK_MAIN();
