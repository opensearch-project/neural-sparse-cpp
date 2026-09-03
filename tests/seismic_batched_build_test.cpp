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
#include <utility>
#include <vector>

#include "csr_interchange_test_util.h"
#include "nsparse/cluster/inverted_list_clusters.h"
#include "nsparse/disk_seismic_index.h"
#include "nsparse/disk_seismic_scalar_quantized_index.h"
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

// `n` is a field rather than a method so csr_test::write_interchange_csr, which
// is templated on any corpus exposing .n/.indptr/.indices/.values, accepts it.
struct Corpus {
    int dim;
    idx_t n = 0;
    std::vector<idx_t> indptr;
    std::vector<term_t> indices;
    std::vector<float> values;
};

Corpus make_corpus(int n_docs, int dim, unsigned seed) {
    std::mt19937 gen(seed);
    std::uniform_int_distribution<int> nnz_dist(3, 12);
    std::uniform_int_distribution<int> term_dist(0, dim - 1);
    std::uniform_real_distribution<float> val_dist(0.1F, 3.0F);

    Corpus corpus;
    corpus.dim = dim;
    corpus.n = n_docs;
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

    // An existing, empty directory to spill into: what
    // batch_file_output_path takes.
    [[nodiscard]] std::string scratch(
        const std::string& name = "scratch") const {
        const std::string dir = file(name);
        std::filesystem::create_directories(dir);
        return dir;
    }

private:
    std::string path_;
};

SeismicClusterParameters params_for(size_t batch_size,
                                    const std::string& scratch_dir, int seed) {
    SeismicClusterParameters params = {
        .lambda = kLambda, .beta = kBeta, .alpha = kAlpha};
    params.batch_clustering.batch_size = batch_size;
    params.batch_clustering.batch_file_output_path = scratch_dir;
    params.seed = seed;
    return params;
}

std::vector<uint8_t> read_file(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    return {std::istreambuf_iterator<char>(in),
            std::istreambuf_iterator<char>()};
}

// A batched build -- windows spilled to a scratch directory -- then written out
// the ordinary way. build() itself produces no file; write_index is still what
// serializes an index.
std::vector<uint8_t> batched(const Corpus& corpus, size_t batch_size,
                             const TempDir& dir, const std::string& out,
                             int seed = kSeed) {
    SeismicIndex index(corpus.dim, params_for(batch_size, dir.scratch(), seed));
    index.add(corpus.n, corpus.indptr.data(), corpus.indices.data(),
              corpus.values.data());
    index.build();
    write_index(&index, const_cast<char*>(out.c_str()));
    return read_file(out);
}

// The same corpus built whole and written the same way.
std::vector<uint8_t> in_memory(const Corpus& corpus, const std::string& out,
                               size_t batch_size = 1, int seed = kSeed) {
    SeismicIndex index(corpus.dim, params_for(batch_size, "", seed));
    index.add(corpus.n, corpus.indptr.data(), corpus.indices.data(),
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

// The corpus as a native-layout CSR: what a mapped read consumes.
std::string write_native_csr(const Corpus& corpus, const std::string& path) {
    csr_test::write_interchange_csr(path, corpus, corpus.dim);
    const std::string native = csr_layout::native_path(path);
    csr_layout::convert(path, native);
    return native;
}

}  // namespace

// The point of the seeding discipline: for a fixed seed a batched build is not
// merely equivalent to a whole-corpus one, it serializes to the same file. Each
// list's k-means seed comes from its own global term id, so neither the window
// a term landed in nor the order the threads reached it can leak into the
// output.
TEST(SeismicBatchedBuild, BatchedBuildIsByteIdenticalToInMemoryBuild) {
    Corpus corpus = make_corpus(/*n_docs=*/2000, /*dim=*/200, /*seed=*/42);
    TempDir dir("identical");
    EXPECT_EQ(in_memory(corpus, dir.file("mem.dat")),
              batched(corpus, /*batch_size=*/4, dir, dir.file("batched.dat")));
}

// The window count is a memory knob, not a behaviour knob: at a fixed seed
// every count has to produce the same index.
TEST(SeismicBatchedBuild, BatchedBuildIsIdenticalAcrossBatchCounts) {
    Corpus corpus = make_corpus(/*n_docs=*/2000, /*dim=*/200, /*seed=*/11);
    TempDir dir("counts");
    const auto one = in_memory(corpus, dir.file("b1.dat"));
    ASSERT_FALSE(one.empty());
    EXPECT_EQ(one, batched(corpus, 2, dir, dir.file("b2.dat")));
    EXPECT_EQ(one, batched(corpus, 10, dir, dir.file("b10.dat")));
    // More windows than terms is clamped to one term each.
    EXPECT_EQ(one, batched(corpus, 1000, dir, dir.file("b1000.dat")));
}

// The spill is scratch: whatever is left of it goes with the index, and while
// the index lives its lists stay readable from the mapping either way.
//
// When it goes is the platform's business, not the contract's -- unlinked the
// moment it is mapped where that is allowed, removed on release where it is not
// (Windows) -- so this asserts the directory is empty once the index is gone,
// and only that the spill is the sole occupant before then.
TEST(SeismicBatchedBuild, LeavesNothingInTheScratchDirectory) {
    Corpus corpus = make_corpus(/*n_docs=*/1500, /*dim=*/200, /*seed=*/3);
    TempDir dir("scratch");
    const std::string scratch = dir.scratch();

    {
        SeismicIndex index(corpus.dim, params_for(8, scratch, kSeed));
        index.add(corpus.n, corpus.indptr.data(), corpus.indices.data(),
                  corpus.values.data());
        index.build();

        EXPECT_LE(std::distance(std::filesystem::directory_iterator(scratch),
                                std::filesystem::directory_iterator{}),
                  1);
        // Serialized from lists that are still borrowed from the spill.
        write_index(&index, const_cast<char*>(dir.file("out.dat").c_str()));
    }

    EXPECT_TRUE(std::filesystem::is_empty(scratch));
    EXPECT_EQ(read_file(dir.file("out.dat")),
              in_memory(corpus, dir.file("mem.dat")));
}

// A build that throws part-way through spilling must not leave the half-written
// spill behind. The corpus here has a term the declared dimension does not
// cover, which the counting pass rejects after the spill file has been created.
TEST(SeismicBatchedBuild, RemovesAPartialSpillWhenTheBuildThrows) {
    Corpus corpus = make_corpus(/*n_docs=*/50, /*dim=*/32, /*seed=*/1);
    TempDir dir("partial");
    const std::string scratch = dir.scratch();

    {
        SeismicIndex narrow(corpus.dim / 2, params_for(4, scratch, kSeed));
        narrow.add(corpus.n, corpus.indptr.data(), corpus.indices.data(),
                   corpus.values.data());
        EXPECT_THROW(narrow.build(), std::invalid_argument);
        EXPECT_TRUE(std::filesystem::is_empty(scratch))
            << "a failed build left scratch behind";
    }
    EXPECT_TRUE(std::filesystem::is_empty(scratch));
}

// The build deletes its own spill and nothing else. The directory is the
// caller's, so whatever else lives in it -- including a file named like a
// spill, which the build did not create -- has to still be there afterwards.
TEST(SeismicBatchedBuild, LeavesOtherFilesInTheScratchDirectoryAlone) {
    Corpus corpus = make_corpus(/*n_docs=*/1500, /*dim=*/200, /*seed=*/3);
    TempDir dir("scratch_others");
    const std::string scratch = dir.scratch();
    const std::array<std::string, 3> bystanders = {
        scratch + "/keep-me.txt", scratch + "/index.dat",
        scratch + "/nsparse-clustered-lists-999.tmp"};
    for (const std::string& path : bystanders) {
        std::ofstream(path, std::ios::binary) << "not the build's";
    }

    {
        SeismicIndex index(corpus.dim, params_for(8, scratch, kSeed));
        index.add(corpus.n, corpus.indptr.data(), corpus.indices.data(),
                  corpus.values.data());
        index.build();
        // Destroyed here, which is when a spill that could not be unlinked
        // while mapped is removed.
    }

    for (const std::string& path : bystanders) {
        EXPECT_TRUE(std::filesystem::exists(path)) << path;
        EXPECT_EQ(std::filesystem::file_size(path), 15U) << path;
    }
    EXPECT_EQ(std::distance(std::filesystem::directory_iterator(scratch),
                            std::filesystem::directory_iterator{}),
              static_cast<ptrdiff_t>(bystanders.size()));
}

// The two knobs are only useful together. A window count with nowhere to spill
// the windows bounds the fill intermediate while leaving the clustered lists to
// accumulate -- a corpus pass per window for a fraction of the peak -- so it is
// resolved to one window rather than honoured.
TEST(SeismicBatchedBuild, BatchSizeWithoutAScratchDirectoryIsOneWindow) {
    BatchClusteringOption opt;
    opt.batch_size = 64;
    EXPECT_EQ(opt.effective_batch_size(), 1U);

    opt.batch_file_output_path = "/tmp/does-not-need-to-exist";
    EXPECT_EQ(opt.effective_batch_size(), 64U);

    // 0 is not a window count; a build always runs at least one.
    opt.batch_size = 0;
    EXPECT_EQ(opt.effective_batch_size(), 1U);
}

// End to end, the resolution above is invisible: a batch size with no scratch
// directory builds exactly what an unbatched build does.
TEST(SeismicBatchedBuild, BatchSizeAloneDoesNotChangeAnInMemoryBuild) {
    Corpus corpus = make_corpus(/*n_docs=*/1500, /*dim=*/200, /*seed=*/5);
    TempDir dir("inmem_batched");
    const auto unbatched = in_memory(corpus, dir.file("b1.dat"), 1);
    ASSERT_FALSE(unbatched.empty());
    EXPECT_EQ(unbatched, in_memory(corpus, dir.file("b8.dat"), 8));
    EXPECT_EQ(unbatched, in_memory(corpus, dir.file("b64.dat"), 64));
}

// Every type in the family reaches the same build, so each has to spill and
// come back with the index a whole-corpus build would have produced.
// Parametrized over all four: float and quantizing, in-memory and
// disk-resident.
class BatchedBuildEveryType : public testing::TestWithParam<const char*> {};

TEST_P(BatchedBuildEveryType, SpilledBuildIsByteIdenticalToInMemoryBuild) {
    const std::string kind = GetParam();
    Corpus corpus = make_corpus(/*n_docs=*/1500, /*dim=*/200, /*seed=*/71);
    TempDir dir("every_type");
    const std::string base =
        "lambda=64|beta=6|alpha=0.4|seed=42|inverted_list_batch_size=";

    const auto build_and_write = [&](const std::string& spec,
                                     const std::string& out) {
        std::unique_ptr<Index> index(index_factory(corpus.dim, spec.c_str()));
        index->add(corpus.n, corpus.indptr.data(), corpus.indices.data(),
                   corpus.values.data());
        index->build();
        write_index(index.get(), const_cast<char*>(out.c_str()));
        return read_file(out);
    };

    const auto whole =
        build_and_write(kind + "," + base + "1", dir.file("mem.dat"));
    ASSERT_FALSE(whole.empty());
    const std::string scratch = dir.scratch();
    EXPECT_EQ(whole, build_and_write(kind + "," + base +
                                         "8|batch_file_output_path=" + scratch,
                                     dir.file("b8.dat")));
    EXPECT_EQ(whole, build_and_write(kind + "," + base +
                                         "64|batch_file_output_path=" + scratch,
                                     dir.file("b64.dat")));
    EXPECT_TRUE(std::filesystem::is_empty(scratch));
}

INSTANTIATE_TEST_SUITE_P(AllSeismicTypes, BatchedBuildEveryType,
                         testing::Values("seismic", "seismic_sq",
                                         "disk_seismic", "disk_seismic_sq"));

// Windows are cut to equal posting counts, not equal width, so a term heavier
// than a whole window's target has to be handled: it cannot be split, and it
// must still land in exactly one window with every other term. This corpus puts
// most of the postings on one term, and asks for far more windows than that
// allows.
TEST(SeismicBatchedBuild, HandlesATermHeavierThanAWholeWindow) {
    const int dim = 64;
    Corpus corpus = make_corpus(/*n_docs=*/400, dim, /*seed=*/97);
    // Term 7 in every document, on top of what make_corpus drew: one term with
    // an order of magnitude more postings than the rest put together.
    Corpus skewed;
    skewed.dim = dim;
    skewed.n = corpus.n;
    skewed.indptr.push_back(0);
    for (idx_t doc = 0; doc < corpus.n; ++doc) {
        std::vector<std::pair<term_t, float>> row;
        for (idx_t j = corpus.indptr[doc]; j < corpus.indptr[doc + 1]; ++j) {
            if (corpus.indices[j] != 7) {
                row.emplace_back(corpus.indices[j], corpus.values[j]);
            }
        }
        row.emplace_back(static_cast<term_t>(7), 2.5F);
        std::sort(row.begin(), row.end());  // CSR rows must be term-ascending
        for (const auto& [term, value] : row) {
            skewed.indices.push_back(term);
            skewed.values.push_back(value);
        }
        skewed.indptr.push_back(static_cast<idx_t>(skewed.indices.size()));
    }

    TempDir dir("skewed");
    const auto one = in_memory(skewed, dir.file("b1.dat"));
    ASSERT_FALSE(one.empty());
    // Every one of these has to cover all 64 terms exactly once, or the spill
    // would be short and mapping it back would refuse it.
    EXPECT_EQ(one, batched(skewed, 8, dir, dir.file("b8.dat")));
    EXPECT_EQ(one, batched(skewed, 32, dir, dir.file("b32.dat")));
    EXPECT_EQ(one, batched(skewed, 64, dir, dir.file("b64.dat")));
    EXPECT_EQ(one, batched(skewed, 200, dir, dir.file("b200.dat")));
}

// A batched build ends holding its lists, borrowed from the spill, so build()
// leaves an index that serves rather than an empty object. Against an unbatched
// build at the same seed: identical builds, so identical results.
TEST(SeismicBatchedBuild, BatchedBuildIsSearchableAfterBuild) {
    Corpus corpus = make_corpus(/*n_docs=*/2000, /*dim=*/200, /*seed=*/7);
    Corpus queries = make_corpus(/*n_docs=*/50, /*dim=*/200, /*seed=*/99);
    const int k = 10;
    const auto n = static_cast<size_t>(queries.n);
    TempDir dir("searchable");

    const auto search_with = [&](Index& index) {
        std::vector<float> dist(n * k);
        std::vector<idx_t> lab(n * k);
        SeismicSearchParameters params(/*cut=*/3, /*heap_factor=*/1.0F);
        index.search(queries.n, queries.indptr.data(), queries.indices.data(),
                     queries.values.data(), k, dist.data(), lab.data(),
                     &params);
        return std::pair{dist, lab};
    };

    SeismicIndex mem(corpus.dim, params_for(1, "", kSeed));
    mem.add(corpus.n, corpus.indptr.data(), corpus.indices.data(),
            corpus.values.data());
    mem.build();
    const auto [want_dist, want_lab] = search_with(mem);

    SeismicIndex spilled(corpus.dim, params_for(4, dir.scratch(), kSeed));
    spilled.add(corpus.n, corpus.indptr.data(), corpus.indices.data(),
                corpus.values.data());
    spilled.build();

    EXPECT_EQ(spilled.num_vectors(), static_cast<size_t>(corpus.n));
    const auto [got_dist, got_lab] = search_with(spilled);
    EXPECT_EQ(got_lab, want_lab);
    EXPECT_EQ(got_dist, want_dist);
}

// A batched disk build serves too, from the corpus it already holds -- the
// forward index only exists once write_index has laid it out.
TEST(SeismicBatchedBuild, BatchedDiskBuildIsSearchableAfterBuild) {
    Corpus corpus = make_corpus(/*n_docs=*/2000, /*dim=*/200, /*seed=*/17);
    Corpus queries = make_corpus(/*n_docs=*/50, /*dim=*/200, /*seed=*/99);
    const int k = 10;
    const auto n = static_cast<size_t>(queries.n);
    TempDir dir("disk_searchable");

    const auto search_with = [&](Index& index) {
        std::vector<float> dist(n * k);
        std::vector<idx_t> lab(n * k);
        DiskSeismicSearchParameters params(/*cut=*/3, /*k_prime=*/50);
        index.search(queries.n, queries.indptr.data(), queries.indices.data(),
                     queries.values.data(), k, dist.data(), lab.data(),
                     &params);
        return std::pair{dist, lab};
    };

    DiskSeismicIndex mem(corpus.dim, params_for(1, "", kSeed));
    mem.add(corpus.n, corpus.indptr.data(), corpus.indices.data(),
            corpus.values.data());
    mem.build();
    const auto [want_dist, want_lab] = search_with(mem);

    DiskSeismicIndex spilled(corpus.dim, params_for(4, dir.scratch(), kSeed));
    spilled.add(corpus.n, corpus.indptr.data(), corpus.indices.data(),
                corpus.values.data());
    spilled.build();

    EXPECT_EQ(spilled.num_vectors(), static_cast<size_t>(corpus.n));
    const auto [got_dist, got_lab] = search_with(spilled);
    EXPECT_EQ(got_lab, want_lab);
    EXPECT_EQ(got_dist, want_dist);
}

// A corpus borrowed from a mapping is the case two mappings exist for: the
// corpus's, which vectors_ still scores from, and the spill's, which the
// posting lists borrow from. Neither may be given up for the other.
TEST(SeismicBatchedBuild, KeepsTheCorpusMappingWhileBorrowingItsOwnLists) {
    Corpus corpus = make_corpus(/*n_docs=*/1500, /*dim=*/200, /*seed=*/13);
    Corpus queries = make_corpus(/*n_docs=*/40, /*dim=*/200, /*seed=*/77);
    const int k = 10;
    const auto n = static_cast<size_t>(queries.n);
    TempDir dir("mapped_reload");

    const std::string native = write_native_csr(corpus, dir.file("corpus.csr"));
    SeismicIndex index(corpus.dim, params_for(3, dir.scratch(), kSeed));
    index.read_csr(native.c_str(), Residency::kMmap);
    index.build();

    // Still serving: scoring reads the mapped corpus, the lists come from the
    // spill's mapping.
    EXPECT_EQ(index.num_vectors(), static_cast<size_t>(corpus.n));
    std::vector<float> dist(n * k);
    std::vector<idx_t> lab(n * k);
    SeismicSearchParameters params(/*cut=*/3, /*heap_factor=*/1.0F);
    static_cast<Index&>(index).search(
        queries.n, queries.indptr.data(), queries.indices.data(),
        queries.values.data(), k, dist.data(), lab.data(), &params);
    EXPECT_TRUE(
        std::any_of(lab.begin(), lab.end(), [](idx_t id) { return id >= 0; }));
}

// Same invariant without a seed, where the files legitimately differ: the
// doc-id membership of each term's list still cannot depend on the window
// count, because lambda is computed from the global corpus size.
TEST(SeismicBatchedBuild, PerTermMembershipInvariantAcrossBatches) {
    Corpus corpus = make_corpus(/*n_docs=*/2000, /*dim=*/200, /*seed=*/42);
    TempDir dir("membership");
    const std::string one = dir.file("1.dat");
    const std::string ten = dir.file("10.dat");
    in_memory(corpus, one, /*batch_size=*/1, kRandomSeed);
    batched(corpus, 10, dir, ten, kRandomSeed);

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
// while n_lists claimed 65536 -> a corrupt, unloadable spill.
TEST(SeismicBatchedBuild, HandlesDimensionAt65536) {
    const int dim = 65536;  // term ids 0..65535 all fit term_t (uint16)
    Corpus corpus = make_corpus(/*n_docs=*/3000, dim, /*seed=*/5);
    TempDir dir("dim64k");
    const std::string path = dir.file("index.dat");
    // Two windows, so the spilling path is what produces the lists.
    batched(corpus, 2, dir, path);

    // Must load without "unexpected end of index file".
    std::unique_ptr<Index> idx(read_index(const_cast<char*>(path.c_str())));
    EXPECT_EQ(idx->num_vectors(), static_cast<size_t>(corpus.n));

    auto sets = per_term_doc_sets(path);
    EXPECT_EQ(sets.size(), static_cast<size_t>(dim));
    size_t total_docs = 0;
    for (const auto& docs : sets) {
        total_docs += docs.size();
    }
    EXPECT_GT(total_docs, 0U);
}

// A batched build's index must serve correctly through the mapped read path --
// the way a caller whose corpus did not fit in RAM is going to use it.
TEST(SeismicBatchedBuild, SearchThroughMappedReadMatchesInMemory) {
    Corpus corpus = make_corpus(/*n_docs=*/2000, /*dim=*/200, /*seed=*/7);
    Corpus queries = make_corpus(/*n_docs=*/50, /*dim=*/200, /*seed=*/99);
    const int k = 10;
    TempDir dir("search");
    const std::string batched_path = dir.file("batched.dat");

    SeismicIndex mem(corpus.dim, params_for(1, "", kSeed));
    mem.add(corpus.n, corpus.indptr.data(), corpus.indices.data(),
            corpus.values.data());
    mem.build();
    batched(corpus, 4, dir, batched_path);

    std::unique_ptr<Index> disk(read_index(
        const_cast<char*>(batched_path.c_str()), IndexIoFlag::kUseMmap));

    SeismicSearchParameters search_params(/*cut=*/3, /*heap_factor=*/1.0F);
    const auto n = static_cast<size_t>(queries.n);
    std::vector<float> mem_dist(n * k);
    std::vector<idx_t> mem_lab(n * k);
    std::vector<float> disk_dist(n * k);
    std::vector<idx_t> disk_lab(n * k);
    static_cast<Index&>(mem).search(queries.n, queries.indptr.data(),
                                    queries.indices.data(),
                                    queries.values.data(), k, mem_dist.data(),
                                    mem_lab.data(), &search_params);
    disk->search(queries.n, queries.indptr.data(), queries.indices.data(),
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
    batched(corpus, 3, dir, owned_path);

    const std::string native = write_native_csr(corpus, dir.file("corpus.csr"));
    SeismicIndex mapped(corpus.dim, params_for(3, dir.scratch(), kSeed));
    mapped.read_csr(native.c_str(), Residency::kMmap);
    mapped.build();
    write_index(&mapped, const_cast<char*>(mapped_path.c_str()));

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
        dir.scratch("from_factory");

    std::unique_ptr<Index> index(index_factory(corpus.dim, spec.c_str()));
    index->add(corpus.n, corpus.indptr.data(), corpus.indices.data(),
               corpus.values.data());
    index->build();
    write_index(index.get(), const_cast<char*>(path.c_str()));

    EXPECT_EQ(batched(corpus, 8, dir, dir.file("direct.dat")), read_file(path));
}

TEST(SeismicBatchedBuild, RejectsInvalidInput) {
    Corpus corpus = make_corpus(/*n_docs=*/50, /*dim=*/16, /*seed=*/1);
    TempDir dir("reject");

    // A term the declared dimension does not cover. The mapped read path does
    // not range-check terms, so this is the build's own guard -- without it the
    // term would be silently dropped from the index.
    SeismicIndex narrow(corpus.dim / 2, params_for(1, "", kSeed));
    narrow.add(corpus.n, corpus.indptr.data(), corpus.indices.data(),
               corpus.values.data());
    EXPECT_THROW(narrow.build(), std::invalid_argument);

    // Somewhere to spill is the caller's to provide: a path that is not a
    // directory is refused rather than half-written.
    SeismicIndex no_dir(corpus.dim,
                        params_for(4, dir.file("not-a-directory"), kSeed));
    no_dir.add(corpus.n, corpus.indptr.data(), corpus.indices.data(),
               corpus.values.data());
    EXPECT_THROW(no_dir.build(), std::invalid_argument);

    // An empty corpus spills no windows at all, which would map back to no
    // lists, so it is refused rather than silently producing an empty index.
    SeismicIndex empty(corpus.dim, params_for(4, dir.scratch(), kSeed));
    EXPECT_THROW(empty.build(), std::invalid_argument);

    // Same for a disk index, which reaches the same build.
    DiskSeismicIndex empty_disk(corpus.dim,
                                params_for(4, dir.scratch(), kSeed));
    EXPECT_THROW(empty_disk.build(), std::invalid_argument);
}

}  // namespace nsparse
