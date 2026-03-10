/**
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 *
 * The OpenSearch Contributors require contributions made to
 * this file be licensed under the Apache-2.0 license or a
 * compatible open source license.
 */

#include <benchmark/benchmark.h>

#include <random>

#include "nsparse/utils/ranker.h"

namespace {

std::vector<std::pair<float, size_t>> GenerateRandomData(size_t n,
                                                         unsigned int seed) {
    std::mt19937 gen(seed);
    std::uniform_real_distribution<float> dist(0.0F, 1000.0F);
    std::vector<std::pair<float, size_t>> data(n);
    for (size_t i = 0; i < n; ++i) {
        data[i] = {dist(gen), i};
    }
    return data;
}

}  // namespace

static void BM_TopKHolder_Add(benchmark::State& state) {
    const size_t n = state.range(0);
    const int k = state.range(1);
    auto data = GenerateRandomData(n, 42);

    for (auto _ : state) {
        nsparse::detail::TopKHolder<size_t> holder(k);
        for (const auto& [score, id] : data) {
            holder.add(score, id);
        }
        benchmark::DoNotOptimize(holder);
    }
    state.SetItemsProcessed(state.iterations() * n);
}

static void BM_TopKHolder_TopK(benchmark::State& state) {
    const size_t n = state.range(0);
    const int k = state.range(1);
    auto data = GenerateRandomData(n, 42);

    for (auto _ : state) {
        state.PauseTiming();
        nsparse::detail::TopKHolder<size_t> holder(k);
        for (const auto& [score, id] : data) {
            holder.add(score, id);
        }
        state.ResumeTiming();
        auto result = holder.top_k();
        benchmark::DoNotOptimize(result);
    }
}

static void BM_TopKHolder_TopKDescending(benchmark::State& state) {
    const size_t n = state.range(0);
    const int k = state.range(1);
    auto data = GenerateRandomData(n, 42);

    for (auto _ : state) {
        state.PauseTiming();
        nsparse::detail::TopKHolder<size_t> holder(k);
        for (const auto& [score, id] : data) {
            holder.add(score, id);
        }
        state.ResumeTiming();
        auto result = holder.top_k_descending();
        benchmark::DoNotOptimize(result);
    }
}

static void BM_DedupeTopKHolder_Add(benchmark::State& state) {
    const size_t n = state.range(0);
    const int k = state.range(1);
    auto data = GenerateRandomData(n, 42);

    for (auto _ : state) {
        nsparse::detail::DedupeTopKHolder<size_t> holder(k);
        for (const auto& [score, id] : data) {
            holder.add(score, id, id);
        }
        benchmark::DoNotOptimize(holder);
    }
    state.SetItemsProcessed(state.iterations() * n);
}

static void BM_DedupeTopKHolder_AddWithDuplicates(benchmark::State& state) {
    const size_t n = state.range(0);
    const int k = state.range(1);
    auto data = GenerateRandomData(n, 42);
    // Create duplicates by modding the id
    for (auto& [score, id] : data) {
        id = id % (n / 4);  // 25% unique ids
    }

    for (auto _ : state) {
        nsparse::detail::DedupeTopKHolder<size_t> holder(k);
        for (const auto& [score, id] : data) {
            holder.add(score, id, id);
        }
        benchmark::DoNotOptimize(holder);
    }
    state.SetItemsProcessed(state.iterations() * n);
}

static void BM_DedupeTopKHolder_TopK(benchmark::State& state) {
    const size_t n = state.range(0);
    const int k = state.range(1);
    auto data = GenerateRandomData(n, 42);

    for (auto _ : state) {
        state.PauseTiming();
        nsparse::detail::DedupeTopKHolder<size_t> holder(k);
        for (const auto& [score, id] : data) {
            holder.add(score, id, id);
        }
        state.ResumeTiming();
        auto result = holder.top_k();
        benchmark::DoNotOptimize(result);
    }
}

// Benchmark configurations: (n_items, k)
BENCHMARK(BM_TopKHolder_Add)->Args({100000, 10});

BENCHMARK(BM_TopKHolder_TopK)->Args({100000, 10});

BENCHMARK(BM_TopKHolder_TopKDescending)->Args({10000, 10});

BENCHMARK(BM_DedupeTopKHolder_Add)->Args({100000, 10});

BENCHMARK(BM_DedupeTopKHolder_AddWithDuplicates)->Args({100000, 10});

BENCHMARK(BM_DedupeTopKHolder_TopK)->Args({100000, 10});
