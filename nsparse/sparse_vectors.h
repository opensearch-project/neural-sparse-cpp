/**
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 *
 * The OpenSearch Contributors require contributions made to
 * this file be licensed under the Apache-2.0 license or a
 * compatible open source license.
 */

#ifndef SPARSE_VECTORS_H
#define SPARSE_VECTORS_H

#include <cstdint>
#include <cstdlib>
#include <vector>

#include "nsparse/io/io.h"
#include "nsparse/types.h"

namespace nsparse {

enum ElementSize : uint8_t { U8 = 1, U16 = 2, U32 = 4, U64 = 8 };

struct SparseVectorsConfig {
    size_t element_size;
    size_t dimension;
};

struct SparseVectorsData {
    const idx_t* indptr_data;
    const term_t* indices_data;
    const float* values_data;
};

class SparseVectors : public Serializable {
public:
    SparseVectors() = default;
    explicit SparseVectors(SparseVectorsConfig config);
    ~SparseVectors() = default;

    // copy constructor
    SparseVectors(const SparseVectors& other) = default;
    SparseVectors& operator=(const SparseVectors& other) = default;
    // move constructor
    SparseVectors(SparseVectors&& other) noexcept = default;
    SparseVectors& operator=(SparseVectors&& other) noexcept = default;

    void add_vectors(const std::vector<idx_t>& indptr,
                     const std::vector<term_t>& indices,
                     const std::vector<uint8_t>& weights);

    void add_vectors(const idx_t* indptr, size_t indptr_size,
                     const term_t* indices, size_t indices_size,
                     const uint8_t* weights, size_t weights_size);

    void add_vector(const std::vector<term_t>& indices,
                    const std::vector<uint8_t>& weights);

    void add_vector(const term_t* indices, size_t indices_size,
                    const uint8_t* weights, size_t weights_size);

    size_t num_vectors() const;
    size_t get_dimension() const { return config_.dimension; }
    size_t get_element_size() const { return config_.element_size; }

    std::vector<float> get_dense_vector_float(idx_t vector_idx) const;
    std::vector<uint8_t> get_dense_vector(idx_t vector_idx) const;
    const idx_t* indptr_data() const { return indptr_.data(); }
    const term_t* indices_data() const { return indices_.data(); }
    const float* values_data_float() const {
        return reinterpret_cast<const float*>(values_.data());
    }

    const uint8_t* values_data() const { return values_.data(); }

    template <class T>
    const T* typed_values_data() const {
        return reinterpret_cast<const T*>(values_.data());
    }

    SparseVectorsData get_all_data() const {
        return {.indptr_data = indptr_data(),
                .indices_data = indices_data(),
                .values_data = values_data_float()};
    }

    void serialize(IOWriter* writer) const override;
    void deserialize(IOReader* reader) override;

private:
    std::vector<idx_t> indptr_;
    std::vector<term_t> indices_;
    std::vector<uint8_t> values_;
    SparseVectorsConfig config_;
};

static SparseVectors empty_sparse_vectors = SparseVectors();
}  // namespace nsparse

#endif  // SPARSE_VECTORS_H