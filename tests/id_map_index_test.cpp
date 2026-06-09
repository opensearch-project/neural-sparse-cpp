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

#include <vector>

#include "nsparse/id_selector.h"
#include "nsparse/index.h"
#include "nsparse/inverted_index.h"
#include "nsparse/io/buffered_io.h"
#include "nsparse/io/index_io.h"
#include "nsparse/seismic_index.h"
#include "nsparse/seismic_scalar_quantized_index.h"
#include "nsparse/types.h"

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

TEST(IDMapBuildAndSave, produces_loadable_index_with_sq_delegate) {
    auto* sq = new nsparse::SeismicScalarQuantizedIndex(
        nsparse::QuantizerType::QT_8bit, 0.0F, 1.0F,
        {.lambda = 10, .beta = 2, .alpha = 0.5F}, 5);
    auto* idmap = new nsparse::IDMapIndex(sq);

    std::vector<nsparse::idx_t> indptr = {0, 2, 4, 6};
    std::vector<nsparse::term_t> indices = {0, 1, 2, 3, 0, 4};
    std::vector<float> values = {1.0F, 0.5F, 0.8F, 0.6F, 0.9F, 0.7F};
    std::vector<nsparse::idx_t> ids = {100, 200, 300};

    idmap->add_with_ids(3, indptr.data(), indices.data(), values.data(),
                        ids.data());

    // Streaming build_and_save (prepend IDMapIndex header)
    nsparse::BufferedIOWriter writer;
    auto id_val = nsparse::fourcc(nsparse::IDMapIndex::name);
    int dim = 5;
    writer.write(&id_val, sizeof(uint32_t), 1);
    writer.write(&dim, sizeof(int), 1);
    idmap->build_and_save(&writer);

    nsparse::BufferedIOReader reader(writer.data());
    nsparse::Index* loaded = nsparse::read_index(&reader);

    ASSERT_NE(loaded, nullptr);
    EXPECT_EQ(loaded->get_vectors()->num_vectors(), 3);

    // Search should return external IDs
    std::vector<nsparse::idx_t> query_indptr = {0, 2};
    std::vector<nsparse::term_t> query_indices = {0, 1};
    std::vector<float> query_values = {1.0F, 0.8F};
    std::vector<nsparse::idx_t> labels(3, -1);
    std::vector<float> distances(3, -1.0F);

    nsparse::SeismicSearchParameters params(5, 1000.0F);
    loaded->search(1, query_indptr.data(), query_indices.data(),
                   query_values.data(), 3, distances.data(), labels.data(),
                   &params);

    // Should get external IDs back
    EXPECT_TRUE(labels[0] == 100 || labels[0] == 200 || labels[0] == 300);

    delete loaded;
    delete idmap;
}
