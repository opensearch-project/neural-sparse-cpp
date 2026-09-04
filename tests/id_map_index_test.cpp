/**
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 *
 * The OpenSearch Contributors require contributions made to
 * this file be licensed under the Apache-2.0 license or a
 * compatible open source license.
 */

#include "nsparse/id_map_index.h"

#include <gtest/gtest.h>

#include <cstdint>
#include <filesystem>
#include <random>
#include <set>
#include <string>
#include <system_error>
#include <vector>

#include "nsparse/id_selector.h"
#include "nsparse/index.h"
#include "nsparse/inverted_index.h"
#include "nsparse/io/buffered_io.h"
#include "nsparse/io/index_io.h"
#include "nsparse/seismic_index.h"
#include "nsparse/seismic_scalar_quantized_index.h"
#include "nsparse/types.h"
#include "nsparse/utils/csr_layout.h"
#include "nsparse/utils/scalar_quantizer.h"
#include "tests/csr_interchange_test_util.h"

namespace {

class IDMapIndexTest : public ::testing::Test {
protected:
    void SetUp() override {
        seismic_ = new nsparse::SeismicIndex(
            100, {.lambda = 10, .beta = 2, .alpha = 0.5F});
        idmap_ = new nsparse::IDMapIndex(seismic_);
    }

    void TearDown() override { delete idmap_; }

    nsparse::SeismicIndex* seismic_;
    nsparse::IDMapIndex* idmap_;
};

}  // namespace

TEST_F(IDMapIndexTest, id) {
    EXPECT_EQ(idmap_->id(), nsparse::IDMapIndex::name);
}

TEST_F(IDMapIndexTest, get_vectors_empty) {
    EXPECT_EQ(idmap_->get_vectors(), nullptr);
}

TEST_F(IDMapIndexTest, num_vectors_empty) {
    EXPECT_EQ(idmap_->num_vectors(), 0);
}

TEST_F(IDMapIndexTest, add_with_ids) {
    std::vector<nsparse::idx_t> indptr = {0, 2, 4};
    std::vector<nsparse::term_t> indices = {0, 1, 2, 3};
    std::vector<float> values = {1.0F, 0.5F, 0.8F, 0.3F};
    std::vector<nsparse::idx_t> ids = {100, 200};

    idmap_->add_with_ids(2, indptr.data(), indices.data(), values.data(),
                         ids.data());

    EXPECT_EQ(idmap_->num_vectors(), 2);
}

TEST_F(IDMapIndexTest, add_with_ids_multiple_batches) {
    std::vector<nsparse::idx_t> indptr1 = {0, 2};
    std::vector<nsparse::term_t> indices1 = {0, 1};
    std::vector<float> values1 = {1.0F, 0.5F};
    std::vector<nsparse::idx_t> ids1 = {100};

    idmap_->add_with_ids(1, indptr1.data(), indices1.data(), values1.data(),
                         ids1.data());
    EXPECT_EQ(idmap_->num_vectors(), 1);

    std::vector<nsparse::idx_t> indptr2 = {0, 2};
    std::vector<nsparse::term_t> indices2 = {2, 3};
    std::vector<float> values2 = {0.8F, 0.3F};
    std::vector<nsparse::idx_t> ids2 = {200};

    idmap_->add_with_ids(1, indptr2.data(), indices2.data(), values2.data(),
                         ids2.data());
    EXPECT_EQ(idmap_->num_vectors(), 2);
}

TEST_F(IDMapIndexTest, search_returns_external_ids) {
    // Add vectors with custom external IDs
    std::vector<nsparse::idx_t> indptr = {0, 2, 4, 6};
    std::vector<nsparse::term_t> indices = {0, 1, 0, 1, 0, 1};
    std::vector<float> values = {1.0F, 0.5F, 0.3F, 0.2F, 0.8F, 0.4F};
    std::vector<nsparse::idx_t> ids = {1000, 2000, 3000};

    idmap_->add_with_ids(3, indptr.data(), indices.data(), values.data(),
                         ids.data());
    idmap_->build();

    // Query
    std::vector<nsparse::idx_t> query_indptr = {0, 2};
    std::vector<nsparse::term_t> query_indices = {0, 1};
    std::vector<float> query_values = {1.0F, 1.0F};
    std::vector<nsparse::idx_t> labels(3, -1);
    std::vector<float> distances(3, -1.0F);

    idmap_->search(1, query_indptr.data(), query_indices.data(),
                   query_values.data(), 3, distances.data(), labels.data(),
                   nullptr);

    // Results should be external IDs (1000, 2000, 3000), not internal (0, 1, 2)
    for (const auto& label : labels) {
        EXPECT_TRUE(label == 1000 || label == 2000 || label == 3000 ||
                    label == -1);
    }
}

TEST_F(IDMapIndexTest, search_preserves_negative_ids) {
    // Add one vector
    std::vector<nsparse::idx_t> indptr = {0, 2};
    std::vector<nsparse::term_t> indices = {0, 1};
    std::vector<float> values = {1.0F, 0.5F};
    std::vector<nsparse::idx_t> ids = {1000};

    idmap_->add_with_ids(1, indptr.data(), indices.data(), values.data(),
                         ids.data());
    idmap_->build();

    // Query for k=3 but only 1 result exists
    std::vector<nsparse::idx_t> query_indptr = {0, 2};
    std::vector<nsparse::term_t> query_indices = {0, 1};
    std::vector<float> query_values = {1.0F, 1.0F};
    std::vector<nsparse::idx_t> labels(3, -1);
    std::vector<float> distances(3, -1.0F);

    idmap_->search(1, query_indptr.data(), query_indices.data(),
                   query_values.data(), 3, distances.data(), labels.data(),
                   nullptr);

    // First result should be external ID, rest should be -1 (padding)
    EXPECT_EQ(labels[0], 1000);
    EXPECT_EQ(labels[1], -1);
    EXPECT_EQ(labels[2], -1);
}

TEST_F(IDMapIndexTest, get_vectors_after_add) {
    std::vector<nsparse::idx_t> indptr = {0, 2};
    std::vector<nsparse::term_t> indices = {0, 1};
    std::vector<float> values = {1.0F, 0.5F};
    std::vector<nsparse::idx_t> ids = {100};

    idmap_->add_with_ids(1, indptr.data(), indices.data(), values.data(),
                         ids.data());

    EXPECT_NE(idmap_->get_vectors(), nullptr);
    EXPECT_EQ(idmap_->get_vectors()->num_vectors(), 1);
}

TEST(IDMapIndex, default_constructor) {
    nsparse::IDMapIndex idmap;
    EXPECT_EQ(idmap.get_vectors(), nullptr);
    EXPECT_EQ(idmap.num_vectors(), 0);
}

TEST_F(IDMapIndexTest, search_with_null_search_parameters) {
    std::vector<nsparse::idx_t> indptr = {0, 2, 4};
    std::vector<nsparse::term_t> indices = {0, 1, 0, 1};
    std::vector<float> values = {1.0F, 0.5F, 0.3F, 0.2F};
    std::vector<nsparse::idx_t> ids = {100, 200};

    idmap_->add_with_ids(2, indptr.data(), indices.data(), values.data(),
                         ids.data());
    idmap_->build();

    std::vector<nsparse::idx_t> query_indptr = {0, 2};
    std::vector<nsparse::term_t> query_indices = {0, 1};
    std::vector<float> query_values = {1.0F, 1.0F};
    std::vector<nsparse::idx_t> labels(2, -1);
    std::vector<float> distances(2, -1.0F);

    // Should not crash with nullptr search_parameters
    idmap_->search(1, query_indptr.data(), query_indices.data(),
                   query_values.data(), 2, distances.data(), labels.data(),
                   nullptr);

    for (const auto& label : labels) {
        EXPECT_TRUE(label == 100 || label == 200 || label == -1);
    }
}

TEST_F(IDMapIndexTest, search_with_search_parameters_no_id_selector) {
    std::vector<nsparse::idx_t> indptr = {0, 2, 4};
    std::vector<nsparse::term_t> indices = {0, 1, 0, 1};
    std::vector<float> values = {1.0F, 0.5F, 0.3F, 0.2F};
    std::vector<nsparse::idx_t> ids = {100, 200};

    idmap_->add_with_ids(2, indptr.data(), indices.data(), values.data(),
                         ids.data());
    idmap_->build();

    std::vector<nsparse::idx_t> query_indptr = {0, 2};
    std::vector<nsparse::term_t> query_indices = {0, 1};
    std::vector<float> query_values = {1.0F, 1.0F};
    std::vector<nsparse::idx_t> labels(2, -1);
    std::vector<float> distances(2, -1.0F);

    // SeismicSearchParameters with no IDSelector set (defaults to nullptr)
    nsparse::SeismicSearchParameters params;
    idmap_->search(1, query_indptr.data(), query_indices.data(),
                   query_values.data(), 2, distances.data(), labels.data(),
                   &params);

    // Both vectors should be returned since no filtering
    for (const auto& label : labels) {
        EXPECT_TRUE(label == 100 || label == 200 || label == -1);
    }
}

TEST_F(IDMapIndexTest, search_with_id_selector_filters_by_external_id) {
    std::vector<nsparse::idx_t> indptr = {0, 2, 4, 6};
    std::vector<nsparse::term_t> indices = {0, 1, 0, 1, 0, 1};
    std::vector<float> values = {1.0F, 0.5F, 0.3F, 0.2F, 0.8F, 0.4F};
    std::vector<nsparse::idx_t> ids = {100, 200, 300};

    idmap_->add_with_ids(3, indptr.data(), indices.data(), values.data(),
                         ids.data());
    idmap_->build();

    std::vector<nsparse::idx_t> query_indptr = {0, 2};
    std::vector<nsparse::term_t> query_indices = {0, 1};
    std::vector<float> query_values = {1.0F, 1.0F};
    std::vector<nsparse::idx_t> labels(3, -1);
    std::vector<float> distances(3, -1.0F);

    // SetIDSelector with external IDs — only allow 100 and 300
    std::vector<nsparse::idx_t> allowed_ids = {100, 300};
    nsparse::SetIDSelector selector(allowed_ids.size(), allowed_ids.data());
    nsparse::SeismicSearchParameters params;
    params.set_id_selector(&selector);

    idmap_->search(1, query_indptr.data(), query_indices.data(),
                   query_values.data(), 3, distances.data(), labels.data(),
                   &params);

    // External ID 200 should be filtered out
    for (const auto& label : labels) {
        EXPECT_NE(label, 200);
        EXPECT_TRUE(label == 100 || label == 300 || label == -1);
    }
}

TEST_F(IDMapIndexTest, search_with_id_selector_excludes_all) {
    std::vector<nsparse::idx_t> indptr = {0, 2, 4};
    std::vector<nsparse::term_t> indices = {0, 1, 0, 1};
    std::vector<float> values = {1.0F, 0.5F, 0.3F, 0.2F};
    std::vector<nsparse::idx_t> ids = {100, 200};

    idmap_->add_with_ids(2, indptr.data(), indices.data(), values.data(),
                         ids.data());
    idmap_->build();

    std::vector<nsparse::idx_t> query_indptr = {0, 2};
    std::vector<nsparse::term_t> query_indices = {0, 1};
    std::vector<float> query_values = {1.0F, 1.0F};
    std::vector<nsparse::idx_t> labels(2, -1);
    std::vector<float> distances(2, -1.0F);

    // Selector that matches no existing external IDs
    std::vector<nsparse::idx_t> allowed_ids = {999};
    nsparse::SetIDSelector selector(allowed_ids.size(), allowed_ids.data());
    nsparse::SeismicSearchParameters params;
    params.set_id_selector(&selector);

    idmap_->search(1, query_indptr.data(), query_indices.data(),
                   query_values.data(), 2, distances.data(), labels.data(),
                   &params);

    // All results should be -1 since nothing passes the filter
    for (const auto& label : labels) {
        EXPECT_EQ(label, -1);
    }
}

namespace {

// Minimal Index used to observe delegate destruction. Sets a caller-owned flag
// to true in its destructor so a test can assert the IDMapIndex frees it.
class DestructionTrackingIndex : public nsparse::Index {
public:
    explicit DestructionTrackingIndex(bool* destroyed)
        : nsparse::Index(0), destroyed_(destroyed) {}
    ~DestructionTrackingIndex() override { *destroyed_ = true; }

    std::array<char, 4> id() const override { return {'T', 'E', 'S', 'T'}; }
    void add(nsparse::idx_t, const nsparse::idx_t*, const nsparse::term_t*,
             const float*) override {}

private:
    bool* destroyed_;
};

}  // namespace

// Bug regression: IDMapIndex used to hold its delegate as a raw pointer with a
// defaulted destructor, leaking the delegate (and everything it owned) on
// destruction. The delegate must now be freed when the IDMapIndex is destroyed.
TEST(IDMapIndexOwnership, DeletesDelegateOnDestruction) {
    bool destroyed = false;
    {
        nsparse::IDMapIndex idmap(new DestructionTrackingIndex(&destroyed));
        EXPECT_FALSE(destroyed);
    }
    EXPECT_TRUE(destroyed) << "IDMapIndex must delete its delegate index";
}

// The delegate acquired during deserialization must also be owned/freed. Before
// the fix this leaked the freshly read delegate index.
TEST(IDMapIndexOwnership, DeletesDelegateAcquiredViaReadIndex) {
    // Build and serialize an idmap-wrapped inverted index.
    auto* original = new nsparse::IDMapIndex(new nsparse::InvertedIndex(16));
    std::vector<nsparse::idx_t> indptr = {0, 2, 4};
    std::vector<nsparse::term_t> indices = {0, 1, 2, 3};
    std::vector<float> values = {1.0F, 0.5F, 0.8F, 0.3F};
    std::vector<nsparse::idx_t> ids = {100, 200};
    original->add_with_ids(2, indptr.data(), indices.data(), values.data(),
                           ids.data());
    original->build();

    nsparse::BufferedIOWriter writer;
    nsparse::write_index(original, &writer);
    delete original;  // must not leak its delegate

    // read_index constructs a fresh IDMapIndex whose read_index() allocates a
    // new delegate; deleting the wrapper must free that delegate too. Under
    // ASan/LeakSanitizer this test fails if either delegate leaks.
    nsparse::BufferedIOReader reader(writer.data());
    nsparse::Index* loaded = nsparse::read_index(&reader);
    ASSERT_NE(loaded, nullptr);
    delete loaded;
}

TEST_F(IDMapIndexTest, search_with_not_id_selector) {
    std::vector<nsparse::idx_t> indptr = {0, 2, 4, 6};
    std::vector<nsparse::term_t> indices = {0, 1, 0, 1, 0, 1};
    std::vector<float> values = {1.0F, 0.5F, 0.3F, 0.2F, 0.8F, 0.4F};
    std::vector<nsparse::idx_t> ids = {100, 200, 300};

    idmap_->add_with_ids(3, indptr.data(), indices.data(), values.data(),
                         ids.data());
    idmap_->build();

    std::vector<nsparse::idx_t> query_indptr = {0, 2};
    std::vector<nsparse::term_t> query_indices = {0, 1};
    std::vector<float> query_values = {1.0F, 1.0F};
    std::vector<nsparse::idx_t> labels(3, -1);
    std::vector<float> distances(3, -1.0F);

    // Exclude external ID 200 using NotIDSelector
    std::vector<nsparse::idx_t> excluded_ids = {200};
    nsparse::SetIDSelector inner_selector(excluded_ids.size(),
                                          excluded_ids.data());
    nsparse::NotIDSelector not_selector(&inner_selector);
    nsparse::SeismicSearchParameters params;
    params.set_id_selector(&not_selector);

    idmap_->search(1, query_indptr.data(), query_indices.data(),
                   query_values.data(), 3, distances.data(), labels.data(),
                   &params);

    // 200 should be excluded
    for (const auto& label : labels) {
        EXPECT_NE(label, 200);
        EXPECT_TRUE(label == 100 || label == 300 || label == -1);
    }
}

namespace {

using nsparse::idx_t;
using nsparse::term_t;

// A CSR corpus exposing the fields csr_test::write_interchange_csr needs
// (.n / .indptr / .indices / .values).
struct Corpus {
    idx_t n = 0;
    std::vector<idx_t> indptr;
    std::vector<term_t> indices;
    std::vector<float> values;
};

// Reproducible random corpus: distinct ascending terms per row (CSR
// convention) and values in (0, 1].
Corpus make_corpus(idx_t rows, int dim, unsigned seed) {
    std::mt19937 rng(seed);
    std::uniform_int_distribution<int> nnz_dist(3, 10);
    std::uniform_int_distribution<int> term_dist(0, dim - 1);
    std::uniform_real_distribution<float> val_dist(0.05F, 1.0F);
    Corpus c;
    c.n = rows;
    c.indptr.push_back(0);
    for (idx_t r = 0; r < rows; ++r) {
        std::set<int> terms;
        const int nnz = nnz_dist(rng);
        while (static_cast<int>(terms.size()) < nnz) {
            terms.insert(term_dist(rng));
        }
        for (const int t : terms) {
            c.indices.push_back(static_cast<term_t>(t));
            c.values.push_back(val_dist(rng));
        }
        c.indptr.push_back(static_cast<idx_t>(c.indices.size()));
    }
    return c;
}

// External ids distinct from the internal row indices, to prove the search
// output is translated through the id map rather than echoing internal ids.
std::vector<idx_t> make_external_ids(idx_t rows) {
    std::vector<idx_t> ids(static_cast<size_t>(rows));
    for (idx_t i = 0; i < rows; ++i) {
        ids[static_cast<size_t>(i)] = 1000 + i * 7;
    }
    return ids;
}

// Batch search over the corpus rows as queries; returns per-query {labels,
// scores} for a bit-exact comparison.
std::pair<std::vector<idx_t>, std::vector<float>> search_corpus(
    nsparse::Index& index, const Corpus& queries, int k) {
    std::vector<idx_t> labels(static_cast<size_t>(queries.n) * k,
                              nsparse::detail::INVALID_IDX);
    std::vector<float> distances(static_cast<size_t>(queries.n) * k, -1.0F);
    index.search(queries.n, queries.indptr.data(), queries.indices.data(),
                 queries.values.data(), k, distances.data(), labels.data());
    return {labels, distances};
}

// A temp file removed on destruction (for the id-map file).
class TempIdFile {
public:
    explicit TempIdFile(const std::string& stem)
        : path_((std::filesystem::temp_directory_path() / stem).string()) {
        std::error_code ignored;
        std::filesystem::remove(path_, ignored);
    }
    ~TempIdFile() {
        std::error_code ignored;
        std::filesystem::remove(path_, ignored);
    }
    TempIdFile(const TempIdFile&) = delete;
    TempIdFile& operator=(const TempIdFile&) = delete;
    const std::string& path() const { return path_; }

private:
    std::string path_;
};

constexpr int kDim = 100;
const nsparse::SeismicClusterParameters kClusterParams{
    .lambda = 10, .beta = 2, .alpha = 0.5F, .seed = 42};

}  // namespace

// The single-function file build (delegate read_csr(kMmap) + id map from a
// file) must produce a search result bit-exact to the in-RAM add_with_ids
// build: same corpus, same external ids, same fixed cluster seed. This is the
// mmap-CSR memory-saving path, and its labels must be the external ids.
TEST(IDMapReadCsrAndId, MatchesAddWithIdsBuild) {
    const Corpus corpus = make_corpus(300, kDim, /*seed=*/1);
    const Corpus queries = make_corpus(20, kDim, /*seed=*/2);
    const std::vector<idx_t> ids = make_external_ids(corpus.n);
    constexpr int k = 10;

    // Reference: add_with_ids() then build().
    nsparse::IDMapIndex added(new nsparse::SeismicIndex(kDim, kClusterParams));
    added.add_with_ids(corpus.n, corpus.indptr.data(), corpus.indices.data(),
                       corpus.values.data(), ids.data());
    added.build();
    const auto expected = search_corpus(added, queries, k);

    // Under test: build the delegate from a native CSR borrowed via mmap and
    // read the id map from a file, through the single entry point.
    nsparse::csr_test::TempCsrFiles csr("nsparse_idmap_src");
    nsparse::csr_test::write_interchange_csr(csr.interchange(), corpus, kDim);
    nsparse::csr_layout::convert(csr.interchange(), csr.native());
    TempIdFile idfile("nsparse_idmap_src.ids");
    nsparse::csr_test::write_id_map_file(idfile.path(), ids);

    nsparse::IDMapIndex mapped(new nsparse::SeismicIndex(kDim, kClusterParams));
    mapped.read_csr_and_ids(csr.native().c_str(), idfile.path().c_str(),
                            nsparse::Residency::kMmap);
    ASSERT_EQ(mapped.num_vectors(), static_cast<size_t>(corpus.n));
    mapped.build();
    const auto got = search_corpus(mapped, queries, k);

    EXPECT_EQ(got.first, expected.first) << "external-id labels differ";
    ASSERT_EQ(got.second.size(), expected.second.size());
    for (size_t i = 0; i < got.second.size(); ++i) {
        EXPECT_FLOAT_EQ(got.second[i], expected.second[i]) << "score at " << i;
    }
    // Sanity: the labels are external ids, not internal row indices.
    bool saw_external = false;
    for (const idx_t label : got.first) {
        if (label >= 1000) {
            saw_external = true;
            break;
        }
    }
    EXPECT_TRUE(saw_external) << "labels should be translated to external ids";
}

// read_csr_and_ids over a QUANTIZED delegate: since CR-2 the quantized types
// accept a mmapped (codes) CSR, so the id-map wrapper works over them too --
// this used to throw. The upstream writes codes at the quantizer's width; the
// build is bit-exact to add_with_ids feeding the same corpus (which add()
// quantizes with the same ScalarQuantizer the test wrote).
TEST(IDMapReadCsrAndId, MatchesAddWithIdsBuildQuantized) {
    const Corpus corpus = make_corpus(300, kDim, /*seed=*/1);
    const Corpus queries = make_corpus(20, kDim, /*seed=*/2);
    const std::vector<idx_t> ids = make_external_ids(corpus.n);
    constexpr int k = 10;
    auto delegate = [] {
        return new nsparse::SeismicScalarQuantizedIndex(
            nsparse::QuantizerType::QT_8bit, 0.0F, 1.0F, kClusterParams, kDim);
    };

    // Reference: add_with_ids() feeds floats, which the delegate quantizes.
    nsparse::IDMapIndex added(delegate());
    added.add_with_ids(corpus.n, corpus.indptr.data(), corpus.indices.data(),
                       corpus.values.data(), ids.data());
    added.build();
    const auto expected = search_corpus(added, queries, k);

    // Under test: the same corpus pre-quantized to codes, borrowed via mmap.
    const nsparse::ScalarQuantizer sq(nsparse::QuantizerType::QT_8bit, 0.0F,
                                      1.0F);
    const size_t element_size = sq.bytes_per_value();
    std::vector<uint8_t> codes(corpus.indices.size() * element_size);
    sq.encode(corpus.values.data(), codes.data(), corpus.indices.size());
    nsparse::csr_test::TempCsrFiles csr("nsparse_idmap_sq_src");
    nsparse::csr_test::write_native_codes_csr(csr.native(), corpus.indptr,
                                             corpus.indices, codes, kDim,
                                             element_size);
    TempIdFile idfile("nsparse_idmap_sq_src.ids");
    nsparse::csr_test::write_id_map_file(idfile.path(), ids);

    nsparse::IDMapIndex mapped(delegate());
    mapped.read_csr_and_ids(csr.native().c_str(), idfile.path().c_str(),
                            nsparse::Residency::kMmap);
    ASSERT_EQ(mapped.num_vectors(), static_cast<size_t>(corpus.n));
    mapped.build();
    const auto got = search_corpus(mapped, queries, k);

    EXPECT_EQ(got.first, expected.first) << "external-id labels differ";
    ASSERT_EQ(got.second.size(), expected.second.size());
    for (size_t i = 0; i < got.second.size(); ++i) {
        EXPECT_FLOAT_EQ(got.second[i], expected.second[i]) << "score at " << i;
    }
}

// read_csr_and_ids over an UNQUANTIZED, non-seismic delegate. InvertedIndex
// overrides num_vectors() with a member that only add() and build() maintain,
// so the mapped path -- which populates vectors_ without going through add() --
// left it at 0 and read_csr_and_ids rejected every id map with a count
// mismatch. The build is bit-exact to add_with_ids over the same corpus.
TEST(IDMapReadCsrAndId, MatchesAddWithIdsBuildInverted) {
    const Corpus corpus = make_corpus(300, kDim, /*seed=*/1);
    const Corpus queries = make_corpus(20, kDim, /*seed=*/2);
    const std::vector<idx_t> ids = make_external_ids(corpus.n);
    constexpr int k = 10;

    // Reference: add_with_ids() then build().
    nsparse::IDMapIndex added(new nsparse::InvertedIndex(kDim));
    added.add_with_ids(corpus.n, corpus.indptr.data(), corpus.indices.data(),
                       corpus.values.data(), ids.data());
    added.build();
    const auto expected = search_corpus(added, queries, k);

    // Under test: the same corpus borrowed from a mapped native CSR. Values
    // stay float here, unlike the quantized case -- an inverted index reports
    // the default 4-byte code_element_size and searches over floats.
    nsparse::csr_test::TempCsrFiles csr("nsparse_idmap_inverted_src");
    nsparse::csr_test::write_interchange_csr(csr.interchange(), corpus, kDim);
    nsparse::csr_layout::convert(csr.interchange(), csr.native());
    TempIdFile idfile("nsparse_idmap_inverted_src.ids");
    nsparse::csr_test::write_id_map_file(idfile.path(), ids);

    nsparse::IDMapIndex mapped(new nsparse::InvertedIndex(kDim));
    mapped.read_csr_and_ids(csr.native().c_str(), idfile.path().c_str(),
                            nsparse::Residency::kMmap);
    // The count read_csr_and_ids checks the id file against, and what used to
    // be 0 here.
    ASSERT_EQ(mapped.num_vectors(), static_cast<size_t>(corpus.n));
    mapped.build();
    const auto got = search_corpus(mapped, queries, k);

    EXPECT_EQ(got.first, expected.first) << "external-id labels differ";
    ASSERT_EQ(got.second.size(), expected.second.size());
    for (size_t i = 0; i < got.second.size(); ++i) {
        EXPECT_FLOAT_EQ(got.second[i], expected.second[i]) << "score at " << i;
    }
}

// The id map is row-aligned with the CSR, so a count that disagrees with the
// delegate's vector count is rejected.
TEST(IDMapReadCsrAndId, CountMismatchThrows) {
    const Corpus corpus = make_corpus(50, kDim, /*seed=*/3);
    nsparse::csr_test::TempCsrFiles csr("nsparse_idmap_mismatch");
    nsparse::csr_test::write_interchange_csr(csr.interchange(), corpus, kDim);
    nsparse::csr_layout::convert(csr.interchange(), csr.native());

    // One too few ids for the 50 CSR rows.
    const std::vector<idx_t> short_ids = make_external_ids(corpus.n - 1);
    TempIdFile idfile("nsparse_idmap_mismatch.ids");
    nsparse::csr_test::write_id_map_file(idfile.path(), short_ids);

    nsparse::IDMapIndex mapped(new nsparse::SeismicIndex(kDim, kClusterParams));
    EXPECT_THROW(
        mapped.read_csr_and_ids(csr.native().c_str(), idfile.path().c_str(),
                                nsparse::Residency::kMmap),
        std::invalid_argument);
}

// A file whose byte size does not match its declared count is malformed.
TEST(IDMapReadCsrAndId, MalformedFileThrows) {
    const Corpus corpus = make_corpus(5, kDim, /*seed=*/4);
    nsparse::csr_test::TempCsrFiles csr("nsparse_idmap_malformed");
    nsparse::csr_test::write_interchange_csr(csr.interchange(), corpus, kDim);
    nsparse::csr_layout::convert(csr.interchange(), csr.native());

    // Header claims 5 ids but only 3 follow -> size mismatch.
    TempIdFile idfile("nsparse_idmap_malformed.ids");
    {
        std::ofstream out(idfile.path(), std::ios::binary);
        const int64_t bogus_count = 5;
        out.write(reinterpret_cast<const char*>(&bogus_count),
                  sizeof(bogus_count));
        const std::vector<idx_t> only_three = {1, 2, 3};
        out.write(
            reinterpret_cast<const char*>(only_three.data()),
            static_cast<std::streamsize>(only_three.size() * sizeof(idx_t)));
    }

    nsparse::IDMapIndex mapped(new nsparse::SeismicIndex(kDim, kClusterParams));
    EXPECT_THROW(
        mapped.read_csr_and_ids(csr.native().c_str(), idfile.path().c_str(),
                                nsparse::Residency::kMmap),
        std::invalid_argument);
}

// The reverse map (external -> internal) must be populated by the file build,
// so an id-selector filter over external ids still works.
TEST(IDMapReadCsrAndId, IdSelectorFilterWorksAfterFileBuild) {
    const Corpus corpus = make_corpus(200, kDim, /*seed=*/5);
    const std::vector<idx_t> ids = make_external_ids(corpus.n);

    nsparse::csr_test::TempCsrFiles csr("nsparse_idmap_filter");
    nsparse::csr_test::write_interchange_csr(csr.interchange(), corpus, kDim);
    nsparse::csr_layout::convert(csr.interchange(), csr.native());
    TempIdFile idfile("nsparse_idmap_filter.ids");
    nsparse::csr_test::write_id_map_file(idfile.path(), ids);

    nsparse::IDMapIndex mapped(new nsparse::SeismicIndex(kDim, kClusterParams));
    mapped.read_csr_and_ids(csr.native().c_str(), idfile.path().c_str(),
                            nsparse::Residency::kMmap);
    mapped.build();

    // Allow only the first two external ids (internal rows 0 and 1).
    const std::vector<idx_t> allowed = {ids[0], ids[1]};
    nsparse::SetIDSelector selector(allowed.size(), allowed.data());
    nsparse::SeismicSearchParameters params;
    params.set_id_selector(&selector);

    // Query with the corpus rows themselves, so rows 0 and 1 (the allowed docs)
    // match their own vectors and are guaranteed to score. For an allowed doc
    // to survive the external-id filter and be returned, the reverse map
    // (external -> internal) must be populated by the file build -- an empty
    // reverse map would filter everything out and still pass a "no disallowed
    // leak" check, so this asserts a positive hit too.
    constexpr int k = 5;
    std::vector<idx_t> labels(static_cast<size_t>(corpus.n) * k,
                              nsparse::detail::INVALID_IDX);
    std::vector<float> distances(static_cast<size_t>(corpus.n) * k, -1.0F);
    mapped.search(corpus.n, corpus.indptr.data(), corpus.indices.data(),
                  corpus.values.data(), k, distances.data(), labels.data(),
                  &params);

    bool saw_allowed = false;
    for (const idx_t label : labels) {
        EXPECT_TRUE(label == ids[0] || label == ids[1] ||
                    label == nsparse::detail::INVALID_IDX)
            << "filter must restrict to allowed external ids, got " << label;
        if (label == ids[0] || label == ids[1]) {
            saw_allowed = true;
        }
    }
    EXPECT_TRUE(saw_allowed) << "reverse map must be populated: an allowed doc "
                                "should actually be returned";
}
