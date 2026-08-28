/**
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 *
 * The OpenSearch Contributors require contributions made to
 * this file be licensed under the Apache-2.0 license or a
 * compatible open source license.
 */

// The dot-product kernels are selected at compile time by distance_simd.h
// (`#if __AVX2__ / __AVX512F__ / ...`), and the main test binary links the
// flag-less library, so distance_test.cpp only ever exercises the scalar
// distance.h. This binary is compiled once per ISA with the matching -m flags
// (see tests/CMakeLists.txt), so `distance_simd.h` here resolves to the SIMD
// kernel, and each dot product is checked against a plain, independent
// reference. Scalar and SIMD kernels share the same nsparse::detail symbol
// names, so they cannot be linked into one binary -- hence the reference is
// computed locally here rather than pulled from distance.h.
//
// ctest gates the launch on CPU support (run_if_isa_supported.cmake), because an
// ISA-specialized binary can execute the ISA in a static initializer before any
// in-process check runs -- so gating has to happen before it starts, not inside
// it. It also fails to build if CMake's requested ISA and the ISA the
// preprocessor selected disagree (a missing -m flag), so it can never silently
// degrade to scalar.

#include <gtest/gtest.h>

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <random>
#include <set>
#include <string_view>
#include <vector>

#include "nsparse/types.h"
#include "nsparse/utils/distance_simd.h"

#ifndef NSPARSE_EXPECT_ISA
#define NSPARSE_EXPECT_ISA "scalar"
#endif

namespace nsparse {
namespace {

// The kernel family this translation unit was compiled for, decided by the same
// macros distance_simd.h dispatches on.
#if defined(__AVX512F__)
constexpr std::string_view kIsa = "avx512";
#elif defined(__AVX2__)
constexpr std::string_view kIsa = "avx2";
#else
constexpr std::string_view kIsa = "scalar";
#endif

// A missing -m flag would leave the preprocessor on the scalar branch while
// CMake believed it built a SIMD binary; catch that at build time.
static_assert(kIsa == std::string_view(NSPARSE_EXPECT_ISA),
              "compiled kernel ISA does not match the one CMake requested -- a "
              "-m compile flag is missing");

constexpr int kDim = 1024;

// Independent references (not distance.h -- see the file header). Float sums in
// double so the tolerance only has to absorb the kernel's own reordering;
// integer sums are exact.
double ref_float(const term_t* indices, const float* weights, size_t len,
                 const float* dense) {
    double sum = 0.0;
    for (size_t i = 0; i < len; ++i) {
        sum += static_cast<double>(weights[i]) *
               static_cast<double>(dense[indices[i]]);
    }
    return sum;
}

template <class T>
int64_t ref_int(const term_t* indices, const T* weights, size_t len,
                const T* dense) {
    int64_t sum = 0;
    for (size_t i = 0; i < len; ++i) {
        sum += static_cast<int64_t>(weights[i]) *
               static_cast<int64_t>(dense[indices[i]]);
    }
    return sum;
}

// Distinct term ids in [0, kDim) with per-length weights and a dense buffer.
// Integer magnitudes are kept modest so the int64 reference sum stays below
// 2^24 and its float cast is exact -- any kernel discrepancy then shows up
// exactly, rather than being hidden by float rounding.
struct Inputs {
    std::vector<term_t> indices;
    std::vector<float> fweights, fdense;
    std::vector<uint16_t> u16weights, u16dense;
    std::vector<uint8_t> u8weights, u8dense;
};

Inputs make_inputs(size_t len, uint32_t seed) {
    std::mt19937 rng(seed);
    std::uniform_int_distribution<int> idx_dist(0, kDim - 1);
    std::uniform_real_distribution<float> f_dist(-4.0F, 4.0F);
    std::uniform_int_distribution<int> small_dist(0, 127);

    Inputs in;
    std::set<int> chosen;  // distinct, like a document's component ids
    while (chosen.size() < len) {
        chosen.insert(idx_dist(rng));
    }
    for (const int t : chosen) {
        in.indices.push_back(static_cast<term_t>(t));
    }
    for (size_t i = 0; i < len; ++i) {
        in.fweights.push_back(f_dist(rng));
        in.u16weights.push_back(static_cast<uint16_t>(small_dist(rng)));
        in.u8weights.push_back(static_cast<uint8_t>(small_dist(rng)));
    }
    in.fdense.resize(kDim);
    in.u16dense.resize(kDim);
    in.u8dense.resize(kDim);
    for (int i = 0; i < kDim; ++i) {
        in.fdense[i] = f_dist(rng);
        in.u16dense[i] = static_cast<uint16_t>(small_dist(rng));
        in.u8dense[i] = static_cast<uint8_t>(small_dist(rng));
    }
    return in;
}

// Lengths straddle the 8- and 16-wide SIMD strides (and AVX-512's 16-wide float
// stride), so the scalar tail of every kernel is exercised.
const std::vector<size_t> kLengths = {0,  1,  2,  3,  7,   8,   9,   15,
                                      16, 17, 31, 32, 33,  63,  64,  65,
                                      127, 128, 129, 255, 256, 257, 500};

// ctest launches this binary only on a CPU that supports the compiled ISA
// (run_if_isa_supported.cmake), so the kernels below always run when reached.
class KernelEquivalence : public ::testing::Test {};

TEST_F(KernelEquivalence, FloatDenseMatchesReference) {
    for (const size_t len : kLengths) {
        const Inputs in = make_inputs(len, /*seed=*/1000 + len);
        const float got = detail::dot_product_float_dense(
            in.indices.data(), in.fweights.data(), len, in.fdense.data());
        const auto want = static_cast<float>(
            ref_float(in.indices.data(), in.fweights.data(), len,
                      in.fdense.data()));
        const float tol = 1e-4F * (std::abs(want) + 1.0F);
        EXPECT_NEAR(got, want, tol) << kIsa << " float, len=" << len;
    }
}

TEST_F(KernelEquivalence, Uint16DenseMatchesReference) {
    for (const size_t len : kLengths) {
        const Inputs in = make_inputs(len, /*seed=*/2000 + len);
        const float got = detail::dot_product_uint16_dense(
            in.indices.data(), in.u16weights.data(), len, in.u16dense.data());
        const auto want = static_cast<float>(
            ref_int<uint16_t>(in.indices.data(), in.u16weights.data(), len,
                              in.u16dense.data()));
        EXPECT_FLOAT_EQ(got, want) << kIsa << " uint16, len=" << len;
    }
}

TEST_F(KernelEquivalence, Uint8DenseMatchesReference) {
    for (const size_t len : kLengths) {
        const Inputs in = make_inputs(len, /*seed=*/3000 + len);
        const float got = detail::dot_product_uint8_dense(
            in.indices.data(), in.u8weights.data(), len, in.u8dense.data());
        const auto want = static_cast<float>(
            ref_int<uint8_t>(in.indices.data(), in.u8weights.data(), len,
                             in.u8dense.data()));
        EXPECT_FLOAT_EQ(got, want) << kIsa << " uint8, len=" << len;
    }
}

// Full-width integer values (the overflow-prone path): the 8-/16-bit codes are
// widened before multiply, so max-valued inputs must not wrap. Kept short so
// the exact reference sum still lands below 2^24.
TEST_F(KernelEquivalence, IntegerMaxValuesDoNotOverflow) {
    constexpr size_t len = 16;
    std::vector<term_t> indices(len);
    for (size_t i = 0; i < len; ++i) {
        indices[i] = static_cast<term_t>(i);
    }
    std::vector<uint8_t> u8w(len, 255), u8d(kDim, 255);
    const float u8_got = detail::dot_product_uint8_dense(
        indices.data(), u8w.data(), len, u8d.data());
    EXPECT_FLOAT_EQ(u8_got, static_cast<float>(len) * 255.0F * 255.0F)
        << kIsa << " uint8 max";

    std::vector<uint16_t> u16w(len, 1000), u16d(kDim, 1000);
    const float u16_got = detail::dot_product_uint16_dense(
        indices.data(), u16w.data(), len, u16d.data());
    EXPECT_FLOAT_EQ(u16_got, static_cast<float>(len) * 1000.0F * 1000.0F)
        << kIsa << " uint16 max";
}

}  // namespace
}  // namespace nsparse
