/**
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 *
 * The OpenSearch Contributors require contributions made to
 * this file be licensed under the Apache-2.0 license or a
 * compatible open source license.
 */

#ifndef CSR_INTERCHANGE_TEST_UTIL_H
#define CSR_INTERCHANGE_TEST_UTIL_H

#include <array>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <string>
#include <system_error>
#include <vector>

#include "nsparse/types.h"

// Shared helpers for the mmap-CSR build path, used by both the regular and the
// disk-resident index suites: write a corpus as an interchange CSR (the layout
// csr_layout::convert consumes) and manage the interchange + native temp files.
namespace nsparse::csr_test {

// Writes a corpus as an interchange CSR: int64 header {rows, num_cols, nnz},
// int64 indptr[rows + 1], int32 indices[nnz], float values[nnz]. Templated on
// the corpus struct (any type exposing .n / .indptr / .indices / .values), so
// it serves any test corpus. The values are written verbatim, so a convert +
// read_csr(kMmap) build sees the exact same vectors as add().
template <class Corpus>
void write_interchange_csr(const std::string& path, const Corpus& c,
                           int num_cols) {
    std::ofstream out(path, std::ios::binary);
    const std::array<int64_t, 3> header = {
        static_cast<int64_t>(c.n), static_cast<int64_t>(num_cols),
        static_cast<int64_t>(c.indices.size())};
    out.write(reinterpret_cast<const char*>(header.data()),
              header.size() * sizeof(int64_t));
    const std::vector<int64_t> indptr64(c.indptr.begin(), c.indptr.end());
    out.write(reinterpret_cast<const char*>(indptr64.data()),
              static_cast<std::streamsize>(indptr64.size() * sizeof(int64_t)));
    const std::vector<int32_t> indices32(c.indices.begin(), c.indices.end());
    out.write(reinterpret_cast<const char*>(indices32.data()),
              static_cast<std::streamsize>(indices32.size() * sizeof(int32_t)));
    out.write(reinterpret_cast<const char*>(c.values.data()),
              static_cast<std::streamsize>(c.values.size() * sizeof(float)));
}

// Writes the id-map file that IDMapIndex::read_csr_and_ids reads:
// [int64 count][idx_t external_id x count]. Row-aligned with the CSR, so
// external_ids[i] is the external id of CSR row i.
inline void write_id_map_file(const std::string& path,
                              const std::vector<idx_t>& external_ids) {
    std::ofstream out(path, std::ios::binary);
    const int64_t count = static_cast<int64_t>(external_ids.size());
    out.write(reinterpret_cast<const char*>(&count), sizeof(count));
    out.write(
        reinterpret_cast<const char*>(external_ids.data()),
        static_cast<std::streamsize>(external_ids.size() * sizeof(idx_t)));
}

// An interchange CSR temp file and the native path convert writes it to, both
// removed on destruction.
class TempCsrFiles {
public:
    explicit TempCsrFiles(const std::string& stem)
        : interchange_(std::filesystem::temp_directory_path() /
                       (stem + ".csr")),
          native_(std::filesystem::temp_directory_path() / (stem + ".mcsr")) {
        std::error_code ignored;
        std::filesystem::remove(interchange_, ignored);
        std::filesystem::remove(native_, ignored);
    }
    ~TempCsrFiles() {
        std::error_code ignored;
        std::filesystem::remove(interchange_, ignored);
        std::filesystem::remove(native_, ignored);
    }
    TempCsrFiles(const TempCsrFiles&) = delete;
    TempCsrFiles& operator=(const TempCsrFiles&) = delete;
    const std::string& interchange() const { return interchange_str_; }
    const std::string& native() const { return native_str_; }

private:
    std::filesystem::path interchange_;
    std::filesystem::path native_;
    std::string interchange_str_ = interchange_.string();
    std::string native_str_ = native_.string();
};

}  // namespace nsparse::csr_test

#endif  // CSR_INTERCHANGE_TEST_UTIL_H
