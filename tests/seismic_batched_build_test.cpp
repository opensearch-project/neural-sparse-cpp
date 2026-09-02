/**
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 *
 * The OpenSearch Contributors require contributions made to
 * this file be licensed under the Apache-2.0 license or a
 * compatible open source license.
 */

#include "nsparse/seismic_batched_build.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <memory>
#include <random>
#include <set>
#include <string>
#include <vector>

#include "nsparse/cluster/inverted_list_clusters.h"
#include "nsparse/disk_seismic_index.h"
#include "nsparse/index_factory.h"
#include "nsparse/io/file_io.h"
#include "nsparse/io/index_io.h"
#include "nsparse/io/seismic_invlists_writer.h"
#include "nsparse/seismic_index.h"
#include "nsparse/seismic_scalar_quantized_index.h"
#include "nsparse/sparse_vectors.h"
#include "nsparse/types.h"
#include "nsparse/utils/csr_layout.h"

namespace nsparse {
namespace {

constexpr int kSeed = 42;
constexpr int kLambda = 64;
constexpr int kBeta = 6;
constexpr float kAlpha = 0.4F;

struct Corpus {
    int dim;
    std::vector<idx_t> indptr;
    std::vector<term_t> indices;
    std::vector<float> values;
    [[nodiscard]] idx_t n() const {
        return static_cast<idx_t>(indptr.size()) - 1;
    }
};

Corpus make_corpus(int n_docs, int dim, unsigned seed) {
    std::mt19937 gen(seed);
    std::uniform_int_distribution<int> nnz_dist(3, 12);
    std::uniform_int_distribution<int> term_dist(0, dim - 1);
    std::uniform_real_distribution<float> val_dist(0.1F, 3.0F);

    Corpus corpus;
    corpus.dim = dim;
    corpus.indptr.push_back(0);
    for (int doc = 0; doc < n_docs; ++doc) {
        // Capped at dim: the loop below draws *distinct* terms, so asking for
        // more than exist would never terminate.
        int nnz = std::min(nnz_dist(gen), dim);
        std::set<int> terms;
        while (static_cast<int>(terms.size()) < nnz) {
            terms.insert(term_dist(gen));
        }
        for (int term : terms) {  // ascending -> a valid CSR row
            corpus.indices.push_back(static_cast<term_t>(term));
            corpus.values.push_back(val_dist(gen));
        }
        corpus.indptr.push_back(static_cast<idx_t>(corpus.indices.size()));
    }
    return corpus;
}

// Removes its directory when it goes out of scope, so a failing EXPECT does not
// leave an index behind.
class TempDir {
public:
    explicit TempDir(const std::string& tag) {
        path_ = (std::filesystem::temp_directory_path() /
                 ("seismic_batched_" + tag + "_" +
                  std::to_string(std::random_device{}())))
                    .string();
        std::filesystem::create_directories(path_);
    }
    ~TempDir() {
        std::error_code ignored;
        std::filesystem::remove_all(path_, ignored);
    }
    TempDir(const TempDir&) = delete;
    TempDir& operator=(const TempDir&) = delete;

    [[nodiscard]] std::string file(const std::string& name) const {
        return path_ + "/" + name;
    }

private:
    std::string path_;
};

SeismicClusterParameters params_for(size_t batch_size,
                                    const std::string& out_path, int seed) {
    SeismicClusterParameters params = {
        .lambda = kLambda, .beta = kBeta, .alpha = kAlpha};
    params.batch_clustering.batch_size = batch_size;
    params.batch_clustering.batch_file_output_path = out_path;
    params.seed = seed;
    return params;
}

std::vector<uint8_t> read_file(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    return {std::istreambuf_iterator<char>(in),
            std::istreambuf_iterator<char>()};
}

// A build streamed straight to `out`, through the index's own build().
std::vector<uint8_t> streamed(const Corpus& corpus, size_t batch_size,
                              const std::string& out, int seed = kSeed) {
    SeismicIndex index(corpus.dim, params_for(batch_size, out, seed));
    index.add(corpus.n(), corpus.indptr.data(), corpus.indices.data(),
              corpus.values.data());
    index.build();
    return read_file(out);
}

// The same corpus built the ordinary way and written with write_index.
std::vector<uint8_t> in_memory(const Corpus& corpus, const std::string& out,
                               size_t batch_size = 1, int seed = kSeed) {
    SeismicIndex index(corpus.dim, params_for(batch_size, "", seed));
    index.add(corpus.n(), corpus.indptr.data(), corpus.indices.data(),
              corpus.values.data());
    index.build();
    write_index(&index, const_cast<char*>(out.c_str()));
    return read_file(out);
}

// Per-term set of all doc ids across that term's clusters, parsed straight from
// the file through the public serialization surface.
std::vector<std::set<idx_t>> per_term_doc_sets(const std::string& path) {
    FileIOReader reader(const_cast<char*>(path.c_str()));
    uint32_t id_val = 0;
    reader.read(&id_val, sizeof(uint32_t), 1);
    uint32_t version = 0;
    reader.read(&version, sizeof(uint32_t), 1);
    int stored_dim = 0;
    reader.read(&stored_dim, sizeof(int), 1);
    EXPECT_EQ(id_val, fourcc(SeismicIndex::name));
    EXPECT_EQ(version, SeismicIndex::kFormatVersion);
    SparseVectors vectors;
    vectors.deserialize(&reader);
    SeismicInvertedListsWriter writer;
    writer.deserialize(&reader);
    std::vector<InvertedListClusters> lists = writer.release();

    std::vector<std::set<idx_t>> out(lists.size());
    for (size_t term = 0; term < lists.size(); ++term) {
        for (size_t cluster = 0; cluster < lists[term].cluster_size();
             ++cluster) {
            for (idx_t doc :
                 lists[term].get_docs(static_cast<idx_t>(cluster))) {
                out[term].insert(doc);
            }
        }
    }
    return out;
}

// The corpus in the interchange CSR layout, converted to native: what a mapped
// read consumes.
std::string write_native_csr(const Corpus& corpus, const std::string& path) {
    std::ofstream out(path, std::ios::binary);
    const std::array<int64_t, 3> sizes = {
        corpus.n(), corpus.dim, static_cast<int64_t>(corpus.indices.size())};
    out.write(reinterpret_cast<const char*>(sizes.data()), sizeof(sizes));
    std::vector<int64_t> indptr64(corpus.indptr.begin(), corpus.indptr.end());
    out.write(reinterpret_cast<const char*>(indptr64.data()),
              static_cast<std::streamsize>(indptr64.size() * sizeof(int64_t)));
    std::vector<int32_t> indices32(corpus.indices.begin(),
                                   corpus.indices.end());
    out.write(reinterpret_cast<const char*>(indices32.data()),
              static_cast<std::streamsize>(indices32.size() * sizeof(int32_t)));
    out.write(
        reinterpret_cast<const char*>(corpus.values.data()),
        static_cast<std::streamsize>(corpus.values.size() * sizeof(float)));
    out.close();
    const std::string native = csr_layout::native_path(path);
    csr_layout::convert(path, native);
    return native;
}

}  // namespace

// The point of the seeding discipline: for a fixed seed a streamed build is not
// merely equivalent to build() + write_index, it is the same file. Each list's
// k-means seed comes from its own global term id, so neither the window a term
// landed in nor the order the threads reached it can leak into the output.
TEST(SeismicBatchedBuild, StreamedBuildIsByteIdenticalToInMemoryBuild) {
    Corpus corpus = make_corpus(/*n_docs=*/2000, /*dim=*/200, /*seed=*/42);
    TempDir dir("identical");
    EXPECT_EQ(in_memory(corpus, dir.file("mem.dat")),
              streamed(corpus, /*batch_size=*/4, dir.file("streamed.dat")));
}

// The window count is a memory knob, not a behaviour knob: at a fixed seed
// every count has to produce the same file.
TEST(SeismicBatchedBuild, StreamedBuildIsIdenticalAcrossBatchCounts) {
    Corpus corpus = make_corpus(/*n_docs=*/2000, /*dim=*/200, /*seed=*/11);
    TempDir dir("counts");
    const auto one = streamed(corpus, 1, dir.file("b1.dat"));
    ASSERT_FALSE(one.empty());
    EXPECT_EQ(one, streamed(corpus, 2, dir.file("b2.dat")));
    EXPECT_EQ(one, streamed(corpus, 10, dir.file("b10.dat")));
    // More windows than terms is clamped to one term each, and 0 means one
    // window rather than none.
    EXPECT_EQ(one, streamed(corpus, 1000, dir.file("b1000.dat")));
    EXPECT_EQ(one, streamed(corpus, 0, dir.file("b0.dat")));
}

// batch_size alone bounds the inverted-list intermediate and leaves the index
// in memory. That is the path every index type gets, including the two disk
// ones, so it must not change what build() produces either.
TEST(SeismicBatchedBuild, BatchSizeAloneDoesNotChangeAnInMemoryBuild) {
    Corpus corpus = make_corpus(/*n_docs=*/1500, /*dim=*/200, /*seed=*/5);
    TempDir dir("inmem_batched");
    const auto unbatched = in_memory(corpus, dir.file("b1.dat"), 1);
    ASSERT_FALSE(unbatched.empty());
    EXPECT_EQ(unbatched, in_memory(corpus, dir.file("b8.dat"), 8));
    EXPECT_EQ(unbatched, in_memory(corpus, dir.file("b64.dat"), 64));
}

// The disk-resident types share the same build, so batch_size has to bound
// their intermediates too without changing what they produce. They have no
// streaming write yet -- their payload interleaves summaries with an inline
// forward index
// -- so this covers the half they do get.
TEST(SeismicBatchedBuild, BatchSizeAloneDoesNotChangeADiskIndex) {
    Corpus corpus = make_corpus(/*n_docs=*/1500, /*dim=*/200, /*seed=*/71);
    TempDir dir("disk");

    auto build_disk = [&corpus](size_t batch_size, const std::string& out) {
        DiskSeismicIndex index(corpus.dim, params_for(batch_size, "", kSeed));
        index.add(corpus.n(), corpus.indptr.data(), corpus.indices.data(),
                  corpus.values.data());
        index.build();
        write_index(&index, const_cast<char*>(out.c_str()));
        return read_file(out);
    };

    const auto unbatched = build_disk(1, dir.file("b1.dat"));
    ASSERT_FALSE(unbatched.empty());
    EXPECT_EQ(unbatched, build_disk(8, dir.file("b8.dat")));
    EXPECT_EQ(unbatched, build_disk(64, dir.file("b64.dat")));
}

// The generalization that matters: a quantizing index streams too, because the
// codes in `vectors_` are already quantized by add() and the shared build only
// needs their width.
TEST(SeismicBatchedBuild, StreamsAQuantizedIndexIdenticallyToo) {
    Corpus corpus = make_corpus(/*n_docs=*/1500, /*dim=*/200, /*seed=*/23);
    TempDir dir("sq");
    const std::string mem_path = dir.file("mem.dat");
    const std::string streamed_path = dir.file("streamed.dat");

    SeismicScalarQuantizedIndex mem(QuantizerType::QT_8bit, 0.0F, 3.0F,
                                    params_for(1, "", kSeed), corpus.dim);
    mem.add(corpus.n(), corpus.indptr.data(), corpus.indices.data(),
            corpus.values.data());
    mem.build();
    write_index(&mem, const_cast<char*>(mem_path.c_str()));

    SeismicScalarQuantizedIndex batched(QuantizerType::QT_8bit, 0.0F, 3.0F,
                                        params_for(4, streamed_path, kSeed),
                                        corpus.dim);
    batched.add(corpus.n(), corpus.indptr.data(), corpus.indices.data(),
                corpus.values.data());
    batched.build();

    EXPECT_EQ(read_file(mem_path), read_file(streamed_path));
    // And it loads as the quantized type it claims to be.
    std::unique_ptr<Index> reloaded(
        read_index(const_cast<char*>(streamed_path.c_str())));
    EXPECT_EQ(reloaded->id(), SeismicScalarQuantizedIndex::name);
    EXPECT_EQ(reloaded->num_vectors(), static_cast<size_t>(corpus.n()));
}

// Same invariant without a seed, where the files legitimately differ: the
// doc-id membership of each term's list still cannot depend on the window
// count, because lambda is computed from the global corpus size.
TEST(SeismicBatchedBuild, PerTermMembershipInvariantAcrossBatches) {
    Corpus corpus = make_corpus(/*n_docs=*/2000, /*dim=*/200, /*seed=*/42);
    TempDir dir("membership");
    const std::string one = dir.file("1.dat");
    const std::string ten = dir.file("10.dat");
    streamed(corpus, 1, one, kRandomSeed);
    streamed(corpus, 10, ten, kRandomSeed);

    auto sets1 = per_term_doc_sets(one);
    auto sets10 = per_term_doc_sets(ten);
    ASSERT_EQ(sets1.size(), static_cast<size_t>(corpus.dim));
    ASSERT_EQ(sets10.size(), sets1.size());
    for (size_t term = 0; term < sets1.size(); ++term) {
        EXPECT_EQ(sets1[term], sets10[term]) << "term " << term;
    }
}

// Regression for the term_t (uint16) window-bound overflow: a dimension at the
// 2^16 boundary must still build every term's list. Before the fix, size_t
// window bounds cast to term_t wrapped mod 65536, so dim=65536 built nothing
// while n_lists claimed 65536 -> a corrupt, unloadable file.
TEST(SeismicBatchedBuild, HandlesDimensionAt65536) {
    const int dim = 65536;  // term ids 0..65535 all fit term_t (uint16)
    Corpus corpus = make_corpus(/*n_docs=*/3000, dim, /*seed=*/5);
    TempDir dir("dim64k");
    const std::string path = dir.file("index.dat");
    streamed(corpus, 1, path);

    // Must load without "unexpected end of index file".
    std::unique_ptr<Index> idx(read_index(const_cast<char*>(path.c_str())));
    EXPECT_EQ(idx->num_vectors(), static_cast<size_t>(corpus.n()));

    auto sets = per_term_doc_sets(path);
    EXPECT_EQ(sets.size(), static_cast<size_t>(dim));
    size_t total_docs = 0;
    for (const auto& docs : sets) {
        total_docs += docs.size();
    }
    EXPECT_GT(total_docs, 0U);
}

// A streamed index must serve correctly through the mapped read path -- the way
// a caller whose corpus did not fit in RAM is going to use it.
TEST(SeismicBatchedBuild, SearchThroughMappedReadMatchesInMemory) {
    Corpus corpus = make_corpus(/*n_docs=*/2000, /*dim=*/200, /*seed=*/7);
    Corpus queries = make_corpus(/*n_docs=*/50, /*dim=*/200, /*seed=*/99);
    const int k = 10;
    TempDir dir("search");
    const std::string streamed_path = dir.file("streamed.dat");

    SeismicIndex mem(corpus.dim, params_for(1, "", kSeed));
    mem.add(corpus.n(), corpus.indptr.data(), corpus.indices.data(),
            corpus.values.data());
    mem.build();
    streamed(corpus, 4, streamed_path);

    std::unique_ptr<Index> disk(read_index(
        const_cast<char*>(streamed_path.c_str()), IndexIoFlag::kUseMmap));

    SeismicSearchParameters search_params(/*cut=*/3, /*heap_factor=*/1.0F);
    const auto n = static_cast<size_t>(queries.n());
    std::vector<float> mem_dist(n * k);
    std::vector<idx_t> mem_lab(n * k);
    std::vector<float> disk_dist(n * k);
    std::vector<idx_t> disk_lab(n * k);
    static_cast<Index&>(mem).search(queries.n(), queries.indptr.data(),
                                    queries.indices.data(),
                                    queries.values.data(), k, mem_dist.data(),
                                    mem_lab.data(), &search_params);
    disk->search(queries.n(), queries.indptr.data(), queries.indices.data(),
                 queries.values.data(), k, disk_dist.data(), disk_lab.data(),
                 &search_params);

    // Identical builds (same seed), so identical results, not merely close.
    EXPECT_EQ(mem_lab, disk_lab);
    EXPECT_EQ(mem_dist, disk_dist);
}

// Residency is SparseVectors' business, not the build's: a corpus borrowed from
// a mapping must produce the same index as one on the heap.
TEST(SeismicBatchedBuild, MappedCorpusMatchesOwnedCorpus) {
    Corpus corpus = make_corpus(/*n_docs=*/1500, /*dim=*/200, /*seed=*/13);
    TempDir dir("mapped");
    const std::string owned_path = dir.file("owned.dat");
    const std::string mapped_path = dir.file("mapped.dat");
    streamed(corpus, 3, owned_path);

    const std::string native = write_native_csr(corpus, dir.file("corpus.csr"));
    SeismicIndex mapped(corpus.dim, params_for(3, mapped_path, kSeed));
    mapped.read_csr(native.c_str(), Residency::kMmap);
    mapped.build();

    EXPECT_EQ(read_file(owned_path), read_file(mapped_path));
}

// Both knobs are reachable through the factory description, which is the only
// way the Python bindings can set them.
TEST(SeismicBatchedBuild, FactoryDescriptionDrivesTheBatchedBuild) {
    Corpus corpus = make_corpus(/*n_docs=*/1000, /*dim=*/128, /*seed=*/31);
    TempDir dir("factory");
    const std::string path = dir.file("index.dat");
    const std::string spec =
        "seismic,lambda=64|beta=6|alpha=0.4|seed=42|"
        "inverted_list_batch_size=8|batch_file_output_path=" +
        path;

    std::unique_ptr<Index> index(index_factory(corpus.dim, spec.c_str()));
    index->add(corpus.n(), corpus.indptr.data(), corpus.indices.data(),
               corpus.values.data());
    index->build();

    ASSERT_TRUE(std::filesystem::exists(path));
    EXPECT_EQ(streamed(corpus, 8, dir.file("direct.dat")), read_file(path));
}

TEST(SeismicBatchedBuild, RejectsInvalidInput) {
    Corpus corpus = make_corpus(/*n_docs=*/50, /*dim=*/16, /*seed=*/1);
    TempDir dir("reject");

    // A term the declared dimension does not cover. The mapped read path does
    // not range-check terms, so this is the build's own guard -- without it the
    // term would be silently dropped from the index.
    SeismicIndex narrow(corpus.dim / 2,
                        params_for(1, dir.file("narrow.dat"), kSeed));
    narrow.add(corpus.n(), corpus.indptr.data(), corpus.indices.data(),
               corpus.values.data());
    EXPECT_THROW(narrow.build(), std::invalid_argument);

    // Streaming an empty corpus would leave a header-only file that read_index
    // cannot parse, so it is refused rather than written.
    SeismicIndex empty(corpus.dim, params_for(4, dir.file("empty.dat"), kSeed));
    EXPECT_THROW(empty.build(), std::invalid_argument);
}

}  // namespace nsparse
