/**
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 *
 * The OpenSearch Contributors require contributions made to
 * this file be licensed under the Apache-2.0 license or a
 * compatible open source license.
 */

#include "nsparse/utils/mmap_file.h"

#include <gtest/gtest.h>

#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

namespace nsparse {
namespace {

using AccessPattern = MmapFile::AccessPattern;

// The per-type default: DiskSeismic (point lookups of ~k' small inline blocks)
// must suppress readahead with MADV_RANDOM, while the scanned shared forward
// index collapses to huge pages. A regression here silently reinstates the
// ~860x read amplification that makes per_block look slower than shared.
TEST(MmapFileAdvise, DefaultsByAccessPattern) {
    EXPECT_EQ(MmapFile::resolve_advise(AccessPattern::kPointLookup, nullptr),
              "random");
    EXPECT_EQ(MmapFile::resolve_advise(AccessPattern::kScan, nullptr),
              "hugepage");
}

// NSPARSE_MMAP_ADVISE overrides the per-type default, for either pattern.
TEST(MmapFileAdvise, EnvOverridesEitherPattern) {
    for (const char* env : {"hugepage", "random", "normal", "hugetlb"}) {
        EXPECT_EQ(MmapFile::resolve_advise(AccessPattern::kPointLookup, env),
                  env);
        EXPECT_EQ(MmapFile::resolve_advise(AccessPattern::kScan, env), env);
    }
}

// An empty env var (set but "") is treated as unset, not as an override that
// would silently force the "hugepage" fallback and undo DiskSeismic's default.
TEST(MmapFileAdvise, EmptyEnvIsTreatedAsUnset) {
    EXPECT_EQ(MmapFile::resolve_advise(AccessPattern::kPointLookup, ""),
              "random");
    EXPECT_EQ(MmapFile::resolve_advise(AccessPattern::kScan, ""), "hugepage");
}

// The access-pattern hint is advisory: the mapping returns the file's bytes
// unchanged under either pattern.
TEST(MmapFileAdvise, MapsSameBytesUnderEitherPattern) {
    const std::filesystem::path path =
        std::filesystem::temp_directory_path() / "nsparse_mmap_advise_test.bin";
    std::vector<uint8_t> payload(8192);
    for (size_t i = 0; i < payload.size(); ++i) {
        payload[i] = static_cast<uint8_t>(i * 7 + 1);
    }
    {
        std::ofstream out(path, std::ios::binary | std::ios::trunc);
        out.write(reinterpret_cast<const char*>(payload.data()),
                  static_cast<std::streamsize>(payload.size()));
    }

    for (const AccessPattern access :
         {AccessPattern::kPointLookup, AccessPattern::kScan}) {
        MmapFile file(path.string(), access);
        ASSERT_EQ(file.size(), payload.size());
        ASSERT_NE(file.data(), nullptr);
        EXPECT_EQ(std::memcmp(file.data(), payload.data(), payload.size()), 0);
    }

    std::error_code ignored;
    std::filesystem::remove(path, ignored);
}

}  // namespace
}  // namespace nsparse
