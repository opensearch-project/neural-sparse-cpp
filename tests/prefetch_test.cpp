/**
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 *
 * The OpenSearch Contributors require contributions made to
 * this file be licensed under the Apache-2.0 license or a
 * compatible open source license.
 */

#include <gtest/gtest.h>

#include <cstddef>
#include <vector>

#include "nsparse/types.h"

// Mock infrastructure for prefetch
namespace prefetch_mock {
std::vector<const void*> prefetched_addresses;
int prefetch_rw = -1;
int prefetch_locality = -1;

void reset() {
    prefetched_addresses.clear();
    prefetch_rw = -1;
    prefetch_locality = -1;
}

void record(const void* addr, int rw, int locality) {
    prefetched_addresses.push_back(addr);
    prefetch_rw = rw;
    prefetch_locality = locality;
}
}  // namespace prefetch_mock

// Override NSPARSE_PREFETCH before including prefetch.h
#define NSPARSE_PREFETCH(addr, rw, locality) \
    prefetch_mock::record(addr, rw, locality)

// Force unique instantiations local to this TU by wrapping in anonymous
// namespace. This avoids ODR conflicts with other TUs that use the real
// __builtin_prefetch.
#include "nsparse/utils/prefetch.h"

namespace {

using nsparse::term_t;

template <class T>
__attribute__((noinline)) void test_prefetch_vector(const term_t* indices,
                                                    const T* values,
                                                    size_t len) {
    static constexpr size_t kCacheLineSize = 64;
    const char* indices_ptr = reinterpret_cast<const char*>(indices);
    const char* values_ptr = reinterpret_cast<const char*>(values);
    const size_t indices_bytes = len * sizeof(term_t);
    const size_t values_bytes = len * sizeof(T);
    for (size_t offset = 0; offset < indices_bytes; offset += kCacheLineSize) {
        NSPARSE_PREFETCH(indices_ptr + offset, 0, 0);
    }
    for (size_t offset = 0; offset < values_bytes; offset += kCacheLineSize) {
        NSPARSE_PREFETCH(values_ptr + offset, 0, 0);
    }
}

class PrefetchTest : public ::testing::Test {
protected:
    void SetUp() override { prefetch_mock::reset(); }
};

TEST_F(PrefetchTest, empty_vector_no_prefetch) {
    std::vector<term_t> indices;
    std::vector<float> values;

    test_prefetch_vector(indices.data(), values.data(), 0);

    ASSERT_TRUE(prefetch_mock::prefetched_addresses.empty());
}

TEST_F(PrefetchTest, small_vector_single_cacheline_each) {
    // Small enough to fit in one cache line each
    std::vector<term_t> indices(4);  // 4 * 2 = 8 bytes < 64
    std::vector<float> values(4);    // 4 * 4 = 16 bytes < 64

    test_prefetch_vector(indices.data(), values.data(), indices.size());

    // Should prefetch once for indices, once for values
    ASSERT_EQ(prefetch_mock::prefetched_addresses.size(), 2);
    ASSERT_EQ(prefetch_mock::prefetched_addresses[0], indices.data());
    ASSERT_EQ(prefetch_mock::prefetched_addresses[1], values.data());
}

TEST_F(PrefetchTest, prefetch_uses_read_hint) {
    std::vector<term_t> indices(4);
    std::vector<float> values(4);

    test_prefetch_vector(indices.data(), values.data(), indices.size());

    ASSERT_EQ(prefetch_mock::prefetch_rw, 0);  // read
}

TEST_F(PrefetchTest, prefetch_uses_no_temporal_locality) {
    std::vector<term_t> indices(4);
    std::vector<float> values(4);

    test_prefetch_vector(indices.data(), values.data(), indices.size());

    ASSERT_EQ(prefetch_mock::prefetch_locality, 0);  // no temporal locality
}

TEST_F(PrefetchTest, multiple_cachelines_for_indices) {
    // 64 bytes / 2 bytes per term_t = 32 elements per cache line
    // 100 elements = ceil(200 bytes / 64) = 4 cache lines
    std::vector<term_t> indices(100);
    std::vector<float> values(1);  // 4 bytes, 1 cache line

    test_prefetch_vector(indices.data(), values.data(), indices.size());

    // indices: 100 * 2 = 200 bytes -> 4 prefetches (0, 64, 128, 192)
    // values: 100 * 4 = 400 bytes -> 7 prefetches
    size_t expected_indices_prefetches = (100 * sizeof(term_t) + 63) / 64;
    size_t expected_values_prefetches = (100 * sizeof(float) + 63) / 64;
    size_t total = expected_indices_prefetches + expected_values_prefetches;

    ASSERT_EQ(prefetch_mock::prefetched_addresses.size(), total);
}

TEST_F(PrefetchTest, prefetch_addresses_are_cacheline_aligned_offsets) {
    constexpr size_t kCacheLineSize = 64;
    // len=32: indices = 32 * 2 = 64 bytes -> 1 cache line
    //         values = 32 * 4 = 128 bytes -> 2 cache lines
    constexpr size_t len = 32;
    std::vector<term_t> indices(len);
    std::vector<float> values(len);

    test_prefetch_vector(indices.data(), values.data(), len);

    // Should have 3 prefetches total: 1 for indices, 2 for values
    ASSERT_EQ(prefetch_mock::prefetched_addresses.size(), 3);

    const char* indices_base = reinterpret_cast<const char*>(indices.data());
    const char* values_base = reinterpret_cast<const char*>(values.data());

    // Check indices prefetch
    ASSERT_EQ(prefetch_mock::prefetched_addresses[0], indices_base);

    // Check values prefetches
    ASSERT_EQ(prefetch_mock::prefetched_addresses[1], values_base);
    ASSERT_EQ(prefetch_mock::prefetched_addresses[2],
              values_base + kCacheLineSize);
}

TEST_F(PrefetchTest, works_with_different_value_types) {
    // len=64: indices = 64 * 2 = 128 bytes -> 2 cache lines
    //         values = 64 * 1 = 64 bytes -> 1 cache line
    std::vector<term_t> indices(64);
    std::vector<uint8_t> values(64);

    test_prefetch_vector(indices.data(), values.data(), indices.size());

    ASSERT_EQ(prefetch_mock::prefetched_addresses.size(), 3);
}

TEST_F(PrefetchTest, works_with_uint8_values) {
    std::vector<term_t> indices(32);  // 64 bytes -> 1 cache line
    std::vector<uint8_t> values(32);  // 32 bytes -> 1 cache line

    test_prefetch_vector(indices.data(), values.data(), indices.size());

    ASSERT_EQ(prefetch_mock::prefetched_addresses.size(), 2);
}

}  // namespace
