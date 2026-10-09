/**
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 *
 * The OpenSearch Contributors require contributions made to
 * this file be licensed under the Apache-2.0 license or a
 * compatible open source license.
 */

// What would recall be if the inline forward index stored only part of each
// document? Reads an existing 8-bit disk_seismic_sq index and runs the real
// block-budget search over a *simulated* narrower layout, so a candidate format
// can be scored for recall before anything is built: an index takes tens of
// minutes to write and 50 GiB to hold, and a grid over it would take a day.
//
//   prune_sim <index.dat> <queries.csr> <truth.txt> <k> <cut> <k_prime> \
//             <T,T,...> <R,R,...>
//
// Each (T, R) pair is one simulated layout:
//   T  keep only the T largest codes of each doc in each block (0 = keep all).
//      This is the lossy part: block scoring sees a truncated vector.
//   R  after block scoring, re-score the R best candidates against their full
//      vectors and re-rank (0 = no re-scoring). Stands in for a deduplicated
//      whole-corpus store, which costs one copy of the corpus rather than the
//      ~13 the inline layout keeps -- so it is read here from the same block,
//      the bytes being identical either way.
// T=0,R=0 reproduces the shipped index and must match its measured recall;
// that is the control this tool is trusted on.
//
// Latency here means nothing (the truncation is recomputed per query from the
// full stored vector, which a real build would do once at write time) -- this
// answers recall only, and reports the kept-nnz fraction so the size the format
// would reach can be read off the same run.

#include <algorithm>
#include <array>
#include <cstdint>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_set>
#include <vector>

#include "absl/container/flat_hash_set.h"
#include "nsparse/cluster/inverted_list_clusters.h"
#include "nsparse/disk_seismic_search.h"
#include "nsparse/inline_forward_index.h"
#include "nsparse/io/inline_forward_index_io.h"
#include "nsparse/io/seismic_invlists_writer.h"
#include "nsparse/types.h"
#include "nsparse/utils/distance_simd.h"
#include "nsparse/utils/mmap_cursor.h"
#include "nsparse/utils/mmap_file.h"
#include "nsparse/utils/ranker.h"
#include "nsparse/utils/scalar_quantizer.h"

using nsparse::InvertedListClusters;
using nsparse::MmapCursor;
using nsparse::MmapFile;
using nsparse::QuantizerType;
using nsparse::ScalarQuantizer;
using nsparse::SeismicInvertedListsWriter;
using nsparse::detail::BlockCandidate;
using nsparse::detail::BlockView;
using nsparse::detail::InlineForwardIndex;

namespace {

struct CSRMatrix {
    int64_t nrow = 0;
    int64_t ncol = 0;
    int64_t nnz = 0;
    std::vector<int64_t> indptr;
    std::vector<nsparse::term_t> indices;
    std::vector<float> data;
};

CSRMatrix read_csr(const std::string& path) {
    std::ifstream file(path, std::ios::binary);
    if (!file.is_open()) {
        throw std::runtime_error("cannot open CSR file: " + path);
    }
    CSRMatrix matrix;
    std::array<int64_t, 3> sizes{};
    file.read(reinterpret_cast<char*>(sizes.data()), sizeof(sizes));
    matrix.nrow = sizes[0];
    matrix.ncol = sizes[1];
    matrix.nnz = sizes[2];
    matrix.indptr.resize(matrix.nrow + 1);
    file.read(reinterpret_cast<char*>(matrix.indptr.data()),
              static_cast<std::streamsize>((matrix.nrow + 1) *
                                           sizeof(int64_t)));
    std::vector<int32_t> indices32(matrix.nnz);
    file.read(reinterpret_cast<char*>(indices32.data()),
              static_cast<std::streamsize>(matrix.nnz * sizeof(int32_t)));
    matrix.indices.assign(indices32.begin(), indices32.end());
    matrix.data.resize(matrix.nnz);
    file.read(reinterpret_cast<char*>(matrix.data.data()),
              static_cast<std::streamsize>(matrix.nnz * sizeof(float)));
    return matrix;
}

std::vector<std::vector<int64_t>> read_truth(const std::string& path) {
    std::ifstream file(path);
    if (!file.is_open()) {
        throw std::runtime_error("cannot open truth file: " + path);
    }
    std::vector<std::vector<int64_t>> truth;
    std::string line;
    while (std::getline(file, line)) {
        if (line.empty()) {
            continue;
        }
        std::vector<int64_t> ids;
        std::stringstream stream(line);
        std::string field;
        while (std::getline(stream, field, ',')) {
            // The truth file holds ids in scientific notation ("7.187155e+06"),
            // so it has to be read as a double: an integer parse would stop at
            // the decimal point and silently yield 7.
            if (!field.empty()) {
                ids.push_back(std::llround(std::stod(field)));
            }
        }
        truth.push_back(std::move(ids));
    }
    return truth;
}

std::vector<uint32_t> parse_list(const std::string& text) {
    std::vector<uint32_t> values;
    std::stringstream stream(text);
    std::string field;
    while (std::getline(stream, field, ',')) {
        if (!field.empty()) {
            values.push_back(
                static_cast<uint32_t>(std::stoul(field)));
        }
    }
    return values;
}

// One candidate doc, carrying where its full vector lives so the re-scoring
// pass can reach it without a second directory lookup. The pointers borrow from
// the mapping, which outlives the query.
struct Candidate {
    float approx = 0.0F;
    nsparse::idx_t doc_id = 0;
    const nsparse::term_t* comps = nullptr;
    const uint8_t* vals = nullptr;
    uint32_t nnz = 0;
};

// How a truncated copy chooses which components to keep.
enum class Criterion {
    // The doc's own largest codes. Ignores which block the copy is in, so every
    // copy of a doc keeps the same components.
    kValue = 0,
    // The doc's code times the block's largest code for that component. A query
    // that selects a block scores highly against the block's summary, so its
    // weight sits on the components the block as a whole is strong in -- which
    // makes this a far better predictor of what the query will actually dot
    // against than the doc's own weight alone. Each copy keeps a different set.
    kValueTimesBlockMax = 1,
};

// Dot the query against only the `keep` components a `kValueTimesBlockMax` copy
// would have stored. `weight` is the block aggregate, indexed by component.
float weighted_truncated_score(const nsparse::term_t* comps,
                               const uint8_t* vals, uint32_t nnz,
                               const uint8_t* dense,
                               const std::vector<uint32_t>& weight,
                               uint32_t keep,
                               std::vector<std::pair<uint64_t, uint32_t>>* scratch,
                               uint64_t* kept_out) {
    scratch->clear();
    scratch->reserve(nnz);
    for (uint32_t i = 0; i < nnz; ++i) {
        scratch->emplace_back(
            static_cast<uint64_t>(vals[i]) * weight[comps[i]], i);
    }
    std::nth_element(scratch->begin(),
                     scratch->begin() + static_cast<ptrdiff_t>(keep) - 1,
                     scratch->end(),
                     [](const auto& a, const auto& b) { return a.first > b.first; });
    uint32_t score = 0;
    for (uint32_t i = 0; i < keep; ++i) {
        const uint32_t idx = (*scratch)[i].second;
        score += static_cast<uint32_t>(dense[comps[idx]]) * vals[idx];
    }
    *kept_out += keep;
    return static_cast<float>(score);
}

// Dot the query against only the `keep` largest codes of a doc, the vector a
// truncating writer would have stored. The cut code comes from a 256-bin
// histogram walked downwards; codes above it are all kept and the bin that
// straddles the limit is taken in array order, which is the tie-break a writer
// doing the same partial selection would land on.
float truncated_score(const nsparse::term_t* comps, const uint8_t* vals,
                      uint32_t nnz, const uint8_t* dense, uint32_t keep,
                      uint64_t* kept_out) {
    std::array<uint32_t, 256> hist{};
    for (uint32_t i = 0; i < nnz; ++i) {
        hist[vals[i]] += 1;
    }
    uint32_t above = 0;
    int cut = 255;
    for (; cut >= 0; --cut) {
        if (above + hist[cut] > keep) {
            break;
        }
        above += hist[cut];
    }
    if (cut < 0) {  // the whole vector fits the budget
        *kept_out += nnz;
        return static_cast<float>(
            nsparse::detail::dot_product_uint8_dense(comps, vals, nnz, dense));
    }
    uint32_t quota = keep - above;
    uint32_t score = 0;
    uint32_t kept = 0;
    const auto cut_code = static_cast<uint8_t>(cut);
    for (uint32_t i = 0; i < nnz; ++i) {
        const uint8_t value = vals[i];
        if (value > cut_code) {
            // fall through to the accumulate below
        } else if (value == cut_code && quota > 0) {
            --quota;
        } else {
            continue;
        }
        score += static_cast<uint32_t>(dense[comps[i]]) * value;
        ++kept;
    }
    *kept_out += kept;
    return static_cast<float>(score);
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 9) {
        std::fprintf(stderr,
                     "usage: %s <index.dat> <queries.csr> <truth.txt> <k> "
                     "<cut> <k_prime> <T,...> <R,...> [criterion]\n",
                     argv[0]);
        return 1;
    }
    const std::string index_path = argv[1];
    const CSRMatrix queries = read_csr(argv[2]);
    const std::vector<std::vector<int64_t>> truth = read_truth(argv[3]);
    const int k = std::stoi(argv[4]);
    const int cut = std::stoi(argv[5]);
    const int k_prime = std::stoi(argv[6]);
    const std::vector<uint32_t> t_list = parse_list(argv[7]);
    const std::vector<uint32_t> r_list = parse_list(argv[8]);
    // 0 = keep the doc's own largest codes; 1 = keep the largest
    // code * block-max products (see Criterion).
    const auto criterion = static_cast<Criterion>(argc > 9 ? std::stoi(argv[9])
                                                           : 0);

    MmapFile file(index_path, MmapFile::AccessPattern::kPointLookup);
    MmapCursor cursor(file.data(), file.size());
    const auto type_id = cursor.read_scalar<uint32_t>();
    cursor.read_scalar<uint32_t>();  // format version
    const auto dimension = cursor.read_scalar<int>();
    std::array<char, 5> fourcc{};
    std::memcpy(fourcc.data(), &type_id, 4);
    if (std::strcmp(fourcc.data(), "DSSQ") != 0) {
        std::fprintf(stderr, "not a disk_seismic_sq index (got %s)\n",
                     fourcc.data());
        return 1;
    }
    const auto quantizer_type = cursor.read_scalar<QuantizerType>();
    const auto vmin = cursor.read_scalar<float>();
    const auto vmax = cursor.read_scalar<float>();
    if (quantizer_type != QuantizerType::QT_8bit) {
        std::fprintf(stderr, "only the 8-bit quantizer is simulated\n");
        return 1;
    }
    const ScalarQuantizer quantizer(quantizer_type, vmin, vmax);

    const auto num_vectors = cursor.read_scalar<uint64_t>();
    SeismicInvertedListsWriter inv_lists;
    inv_lists.mmap_deserialize(&cursor);
    const std::vector<InvertedListClusters> clusters =
        std::move(inv_lists.release());
    InlineForwardIndex fwd;
    fwd.mmap_deserialize(&cursor);
    if (fwd.element_size() != 1) {
        std::fprintf(stderr, "index element_size is %zu, expected 1\n",
                     fwd.element_size());
        return 1;
    }

    // Queries are encoded with the index's own range, which is what
    // DiskSeismicScalarQuantizedIndex::encode_query does absent an override.
    std::vector<uint8_t> query_codes(queries.nnz);
    quantizer.encode(queries.data.data(), query_codes.data(), queries.nnz);

    const int64_t n_queries =
        std::min<int64_t>(queries.nrow, static_cast<int64_t>(truth.size()));
    std::printf(
        "index %s  dim=%d docs=%llu blocks=%llu | queries %lld | k=%d cut=%d "
        "k_prime=%d\n",
        index_path.c_str(), dimension,
        static_cast<unsigned long long>(num_vectors),
        static_cast<unsigned long long>(fwd.num_blocks()),
        static_cast<long long>(n_queries), k, cut, k_prime);
    std::printf("criterion %d (%s)\n", static_cast<int>(criterion),
                criterion == Criterion::kValue ? "code"
                                               : "code * block max");
    std::printf("%6s %6s %10s %10s %10s\n", "T", "R", "recall", "kept_nnz",
                "cands");

    for (const uint32_t keep : t_list) {
        for (const uint32_t rescore : r_list) {
            // Only the candidate pool size can change the result when nothing
            // is truncated, so skip the rescoring sweep for the control.
            if (keep == 0 && rescore != 0 && rescore != r_list.front()) {
                continue;
            }
            double hits = 0;
            uint64_t kept_total = 0;
            uint64_t nnz_total = 0;
            uint64_t cand_total = 0;

#pragma omp parallel
            {
                std::vector<uint8_t> dense(static_cast<size_t>(dimension), 0);
                absl::flat_hash_set<nsparse::idx_t> visited;
                std::vector<BlockCandidate> block_candidates;
                std::vector<float> score_scratch;
                std::vector<Candidate> candidates;
                // Block aggregate, indexed by component, with a per-block stamp
                // so it never has to be cleared.
                std::vector<uint32_t> weight(static_cast<size_t>(dimension), 0);
                std::vector<uint64_t> stamped(static_cast<size_t>(dimension), 0);
                std::vector<std::pair<uint64_t, uint32_t>> sel_scratch;
                uint64_t block_stamp = 0;
                double local_hits = 0;
                uint64_t local_kept = 0;
                uint64_t local_nnz = 0;
                uint64_t local_cands = 0;

#pragma omp for schedule(dynamic, 16)
                for (int64_t q = 0; q < n_queries; ++q) {
                    const int64_t start = queries.indptr[q];
                    const auto len =
                        static_cast<size_t>(queries.indptr[q + 1] - start);
                    const nsparse::term_t* q_idx =
                        queries.indices.data() + start;
                    const uint8_t* q_val = query_codes.data() + start;
                    for (size_t i = 0; i < len; ++i) {
                        dense[q_idx[i]] = q_val[i];
                    }

                    const std::vector<nsparse::term_t> cuts =
                        nsparse::detail::top_cut_tokens(q_idx, q_val, len, cut,
                                                        1);
                    block_candidates.clear();
                    for (const nsparse::term_t term : cuts) {
                        if (term >= clusters.size()) {
                            continue;
                        }
                        const InvertedListClusters& list = clusters[term];
                        const size_t n_clusters = list.cluster_size();
                        if (n_clusters == 0) {
                            continue;
                        }
                        list.score_summaries_transposed(q_idx, q_val, len,
                                                        score_scratch);
                        for (size_t cid = 0; cid < n_clusters; ++cid) {
                            block_candidates.push_back(
                                {score_scratch[cid], term,
                                 static_cast<uint32_t>(cid)});
                        }
                    }
                    const size_t budget =
                        std::min(static_cast<size_t>(k_prime),
                                 block_candidates.size());
                    if (budget < block_candidates.size()) {
                        std::nth_element(
                            block_candidates.begin(),
                            block_candidates.begin() +
                                static_cast<ptrdiff_t>(budget),
                            block_candidates.end(),
                            [](const BlockCandidate& a,
                               const BlockCandidate& b) {
                                return a.score > b.score;
                            });
                        block_candidates.resize(budget);
                    }

                    visited.clear();
                    candidates.clear();
                    for (const BlockCandidate& block : block_candidates) {
                        const BlockView view = fwd.block(block.pl, block.cid);
                        if (view.absent()) {
                            continue;
                        }
                        if (criterion == Criterion::kValueTimesBlockMax &&
                            keep != 0) {
                            ++block_stamp;
                            const uint32_t total = view.offsets[view.n_docs];
                            for (uint32_t j = 0; j < total; ++j) {
                                const nsparse::term_t comp = view.comps[j];
                                const uint8_t value = view.vals[j];
                                if (stamped[comp] != block_stamp) {
                                    stamped[comp] = block_stamp;
                                    weight[comp] = value;
                                } else if (value > weight[comp]) {
                                    weight[comp] = value;
                                }
                            }
                        }
                        for (uint32_t i = 0; i < view.n_docs; ++i) {
                            const auto doc_id =
                                static_cast<nsparse::idx_t>(view.doc_ids[i]);
                            if (!visited.insert(doc_id).second) {
                                continue;
                            }
                            const uint32_t nnz = view.nnz(i);
                            const nsparse::term_t* comps = view.doc_comps(i);
                            const uint8_t* vals = view.doc_vals(i, 1);
                            local_nnz += nnz;
                            float score = 0.0F;
                            if (keep == 0 || nnz <= keep) {
                                local_kept += nnz;
                                score = static_cast<float>(
                                    nsparse::detail::dot_product_uint8_dense(
                                        comps, vals, nnz, dense.data()));
                            } else if (criterion ==
                                       Criterion::kValueTimesBlockMax) {
                                score = weighted_truncated_score(
                                    comps, vals, nnz, dense.data(), weight,
                                    keep, &sel_scratch, &local_kept);
                            } else {
                                score = truncated_score(comps, vals, nnz,
                                                        dense.data(), keep,
                                                        &local_kept);
                            }
                            candidates.push_back(
                                {score, doc_id, comps, vals, nnz});
                        }
                    }
                    local_cands += candidates.size();

                    // Rank on the truncated scores, then re-score the top R
                    // against the full vectors and re-rank those.
                    const size_t pool =
                        rescore == 0
                            ? static_cast<size_t>(k)
                            : std::max(static_cast<size_t>(k),
                                       static_cast<size_t>(rescore));
                    const size_t take = std::min(pool, candidates.size());
                    std::partial_sort(
                        candidates.begin(),
                        candidates.begin() + static_cast<ptrdiff_t>(take),
                        candidates.end(),
                        [](const Candidate& a, const Candidate& b) {
                            return a.approx > b.approx;
                        });
                    candidates.resize(take);
                    if (rescore != 0 && keep != 0) {
                        for (Candidate& candidate : candidates) {
                            candidate.approx = static_cast<float>(
                                nsparse::detail::dot_product_uint8_dense(
                                    candidate.comps, candidate.vals,
                                    candidate.nnz, dense.data()));
                        }
                        const size_t final_take =
                            std::min(static_cast<size_t>(k), candidates.size());
                        std::partial_sort(
                            candidates.begin(),
                            candidates.begin() +
                                static_cast<ptrdiff_t>(final_take),
                            candidates.end(),
                            [](const Candidate& a, const Candidate& b) {
                                return a.approx > b.approx;
                            });
                        candidates.resize(final_take);
                    }

                    const std::vector<int64_t>& want = truth[q];
                    const std::unordered_set<int64_t> want_set(want.begin(),
                                                               want.end());
                    size_t found = 0;
                    for (size_t i = 0;
                         i < candidates.size() && i < static_cast<size_t>(k);
                         ++i) {
                        if (want_set.count(
                                static_cast<int64_t>(candidates[i].doc_id)) !=
                            0) {
                            ++found;
                        }
                    }
                    local_hits += static_cast<double>(found) /
                                  static_cast<double>(k);

                    for (size_t i = 0; i < len; ++i) {
                        dense[q_idx[i]] = 0;
                    }
                }
#pragma omp critical
                {
                    hits += local_hits;
                    kept_total += local_kept;
                    nnz_total += local_nnz;
                    cand_total += local_cands;
                }
            }

            std::printf("%6u %6u %10.6f %9.2f%% %10.1f\n", keep, rescore,
                        hits / static_cast<double>(n_queries),
                        100.0 * static_cast<double>(kept_total) /
                            static_cast<double>(nnz_total),
                        static_cast<double>(cand_total) /
                            static_cast<double>(n_queries));
            std::fflush(stdout);
        }
    }
    return 0;
}
