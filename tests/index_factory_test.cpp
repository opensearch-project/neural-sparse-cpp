/**
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 *
 * The OpenSearch Contributors require contributions made to
 * this file be licensed under the Apache-2.0 license or a
 * compatible open source license.
 */

#include "nsparse/index_factory.h"

#include <gtest/gtest.h>

#include <memory>

#include "nsparse/brutal_index.h"
#include "nsparse/index.h"
#include "nsparse/inverted_index.h"
#include "nsparse/seismic_index.h"
#include "nsparse/seismic_scalar_quantized_index.h"

// Basic factory tests
TEST(IndexFactory, throws_on_null_description) {
    ASSERT_THROW(nsparse::index_factory(10, nullptr), std::invalid_argument);
}

TEST(IndexFactory, throws_on_empty_description) {
    ASSERT_THROW(nsparse::index_factory(10, ""), std::invalid_argument);
}

TEST(IndexFactory, throws_on_unknown_index_type) {
    ASSERT_THROW(nsparse::index_factory(10, "unknown"), std::invalid_argument);
}

// BrutalIndex tests
TEST(IndexFactory, creates_brutal_index) {
    std::unique_ptr<nsparse::Index> index(
        nsparse::index_factory(100, "brutal"));

    ASSERT_NE(index, nullptr);
    ASSERT_EQ(index->get_dimension(), 100);

    auto* brutal = dynamic_cast<nsparse::BrutalIndex*>(index.get());
    ASSERT_NE(brutal, nullptr);
}

TEST(IndexFactory, creates_brutal_index_with_whitespace) {
    std::unique_ptr<nsparse::Index> index(
        nsparse::index_factory(50, "  brutal  "));

    ASSERT_NE(index, nullptr);
    auto* brutal = dynamic_cast<nsparse::BrutalIndex*>(index.get());
    ASSERT_NE(brutal, nullptr);
}

// InvertedIndex tests
TEST(IndexFactory, creates_inverted_index) {
    std::unique_ptr<nsparse::Index> index(
        nsparse::index_factory(100, "inverted"));

    ASSERT_NE(index, nullptr);
    ASSERT_EQ(index->get_dimension(), 100);

    auto* inverted = dynamic_cast<nsparse::InvertedIndex*>(index.get());
    ASSERT_NE(inverted, nullptr);
}

TEST(IndexFactory, creates_inverted_index_with_whitespace) {
    std::unique_ptr<nsparse::Index> index(
        nsparse::index_factory(50, "  inverted  "));

    ASSERT_NE(index, nullptr);
    auto* inverted = dynamic_cast<nsparse::InvertedIndex*>(index.get());
    ASSERT_NE(inverted, nullptr);
}

TEST(IndexFactory, inverted_index_has_correct_id) {
    std::unique_ptr<nsparse::Index> index(
        nsparse::index_factory(100, "inverted"));

    ASSERT_EQ(index->id(), nsparse::InvertedIndex::name);
}

// SeismicIndex tests
TEST(IndexFactory, creates_seismic_index_default_params) {
    std::unique_ptr<nsparse::Index> index(
        nsparse::index_factory(100, "seismic"));

    ASSERT_NE(index, nullptr);
    ASSERT_EQ(index->get_dimension(), 100);

    auto* seismic = dynamic_cast<nsparse::SeismicIndex*>(index.get());
    ASSERT_NE(seismic, nullptr);
}

TEST(IndexFactory, creates_seismic_index_with_params) {
    std::unique_ptr<nsparse::Index> index(
        nsparse::index_factory(100, "seismic,lambda=20|beta=10|alpha=0.8"));

    ASSERT_NE(index, nullptr);
    auto* seismic = dynamic_cast<nsparse::SeismicIndex*>(index.get());
    ASSERT_NE(seismic, nullptr);
}

TEST(IndexFactory, creates_seismic_index_partial_params) {
    std::unique_ptr<nsparse::Index> index(
        nsparse::index_factory(100, "seismic,lambda=15"));

    ASSERT_NE(index, nullptr);
    auto* seismic = dynamic_cast<nsparse::SeismicIndex*>(index.get());
    ASSERT_NE(seismic, nullptr);
}

// SeismicScalarQuantizedIndex tests
TEST(IndexFactory, creates_seismic_sq_index_default_params) {
    std::unique_ptr<nsparse::Index> index(
        nsparse::index_factory(100, "seismic_sq"));

    ASSERT_NE(index, nullptr);
    ASSERT_EQ(index->get_dimension(), 100);

    auto* seismic_sq =
        dynamic_cast<nsparse::SeismicScalarQuantizedIndex*>(index.get());
    ASSERT_NE(seismic_sq, nullptr);
}

TEST(IndexFactory, creates_seismic_sq_index_8bit) {
    std::unique_ptr<nsparse::Index> index(nsparse::index_factory(
        100, "seismic_sq,quantizer=8bit|vmin=0.0|vmax=1.0"));

    ASSERT_NE(index, nullptr);
    auto* seismic_sq =
        dynamic_cast<nsparse::SeismicScalarQuantizedIndex*>(index.get());
    ASSERT_NE(seismic_sq, nullptr);
}

TEST(IndexFactory, creates_seismic_sq_index_16bit) {
    std::unique_ptr<nsparse::Index> index(nsparse::index_factory(
        100, "seismic_sq,quantizer=16bit|vmin=-1.0|vmax=1.0"));

    ASSERT_NE(index, nullptr);
    auto* seismic_sq =
        dynamic_cast<nsparse::SeismicScalarQuantizedIndex*>(index.get());
    ASSERT_NE(seismic_sq, nullptr);
}

TEST(IndexFactory, creates_seismic_sq_index_full_params) {
    std::unique_ptr<nsparse::Index> index(
        nsparse::index_factory(100,
                               "seismic_sq,quantizer=8bit|vmin=0.0|vmax=2.0|"
                               "lambda=20|beta=10|alpha=0.9"));

    ASSERT_NE(index, nullptr);
    auto* seismic_sq =
        dynamic_cast<nsparse::SeismicScalarQuantizedIndex*>(index.get());
    ASSERT_NE(seismic_sq, nullptr);
}

// Parameter parsing edge cases
TEST(IndexFactory, handles_whitespace_in_params) {
    std::unique_ptr<nsparse::Index> index(
        nsparse::index_factory(100, "seismic, lambda = 20 | beta = 10 "));

    ASSERT_NE(index, nullptr);
    auto* seismic = dynamic_cast<nsparse::SeismicIndex*>(index.get());
    ASSERT_NE(seismic, nullptr);
}

TEST(IndexFactory, handles_empty_param_value) {
    // Malformed param: stoi("") throws std::invalid_argument
    ASSERT_ANY_THROW(nsparse::index_factory(100, "seismic,lambda="));
}
