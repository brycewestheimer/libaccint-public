// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2026 Bryce M. Westheimer. All rights reserved.

/// @file test_out_of_core_storage.cpp
/// @brief Tests for disk-backed out-of-core three-center storage

#include <libaccint/df/out_of_core_storage.hpp>
#include <libaccint/utils/error_handling.hpp>

#include <gtest/gtest.h>

#include <cmath>
#include <filesystem>
#include <numeric>

namespace libaccint::df {
namespace {

// ============================================================================
// Construction tests
// ============================================================================

TEST(OutOfCoreStorage, Construction) {
    OutOfCoreConfig config;
    config.scratch_dir = "/tmp/libaccint_test_ooc";
    config.delete_on_destroy = true;

    OutOfCoreThreeCenterStorage storage(5, 10, config);
    EXPECT_EQ(storage.n_orb(), 5u);
    EXPECT_EQ(storage.n_aux(), 10u);
    EXPECT_GT(storage.n_blocks(), 0u);
}

TEST(OutOfCoreStorage, BlockGeometry) {
    OutOfCoreConfig config;
    config.scratch_dir = "/tmp/libaccint_test_ooc2";
    config.block_size_aux = 4;
    config.delete_on_destroy = true;

    OutOfCoreThreeCenterStorage storage(3, 10, config);

    // 10 / 4 = 3 blocks (4, 4, 2)
    EXPECT_EQ(storage.n_blocks(), 3u);
    EXPECT_EQ(storage.block_size(0), 4u);
    EXPECT_EQ(storage.block_size(1), 4u);
    EXPECT_EQ(storage.block_size(2), 2u);

    Size total = 0;
    for (Size i = 0; i < storage.n_blocks(); ++i) {
        auto [start, end] = storage.block_range(i);
        EXPECT_EQ(start, total);
        total = end;
    }
    EXPECT_EQ(total, 10u);
}

// ============================================================================
// Write/Read round-trip
// ============================================================================

TEST(OutOfCoreStorage, WriteAndRead) {
    Size n_orb = 3;
    Size n_aux = 6;

    OutOfCoreConfig config;
    config.scratch_dir = "/tmp/libaccint_test_ooc3";
    config.block_size_aux = 3;
    config.delete_on_destroy = true;

    OutOfCoreThreeCenterStorage storage(n_orb, n_aux, config);
    EXPECT_EQ(storage.n_blocks(), 2u);

    // Write block 0
    Size block_elements = n_orb * n_orb * 3;
    std::vector<Real> data(block_elements);
    std::iota(data.begin(), data.end(), 1.0);

    storage.write_block(0, data);
    EXPECT_TRUE(storage.has_block(0));

    // Read back
    auto retrieved = storage.read_block(0);
    EXPECT_EQ(retrieved.size(), data.size());
    for (Size i = 0; i < data.size(); ++i) {
        EXPECT_DOUBLE_EQ(retrieved[i], data[i]);
    }
}

TEST(OutOfCoreStorage, CacheHitRate) {
    Size n_orb = 2;
    Size n_aux = 4;

    OutOfCoreConfig config;
    config.scratch_dir = "/tmp/libaccint_test_ooc4";
    config.block_size_aux = 2;
    config.memory_limit_mb = 100;
    config.delete_on_destroy = true;

    OutOfCoreThreeCenterStorage storage(n_orb, n_aux, config);

    Size block_elements = n_orb * n_orb * 2;
    std::vector<Real> data(block_elements, 1.0);
    storage.write_block(0, data);

    // First read: cache miss (write also caches, so it's a hit)
    storage.read_block(0);

    // Second read: cache hit
    storage.read_block(0);

    // Should have at least one hit
    EXPECT_GT(storage.cache_hits(), 0u);
    EXPECT_GE(storage.cache_hit_rate(), 0.0);
    EXPECT_LE(storage.cache_hit_rate(), 1.0);
}

TEST(OutOfCoreStorage, FlushCacheAndReload) {
    Size n_orb = 2;
    Size n_aux = 4;

    OutOfCoreConfig config;
    config.scratch_dir = "/tmp/libaccint_test_ooc5";
    config.block_size_aux = 2;
    config.delete_on_destroy = true;

    OutOfCoreThreeCenterStorage storage(n_orb, n_aux, config);

    Size block_elements = n_orb * n_orb * 2;
    std::vector<Real> data(block_elements);
    std::iota(data.begin(), data.end(), 100.0);
    storage.write_block(0, data);

    // Flush cache
    storage.flush_cache();
    EXPECT_EQ(storage.cache_usage(), 0u);

    // Read should reload from disk
    auto retrieved = storage.read_block(0);
    EXPECT_EQ(retrieved.size(), data.size());
    for (Size i = 0; i < data.size(); ++i) {
        EXPECT_DOUBLE_EQ(retrieved[i], data[i]);
    }
}

// ============================================================================
// Error handling
// ============================================================================

TEST(OutOfCoreStorage, ReadUnwrittenBlockThrows) {
    OutOfCoreConfig config;
    config.scratch_dir = "/tmp/libaccint_test_ooc6";
    config.delete_on_destroy = true;

    OutOfCoreThreeCenterStorage storage(3, 6, config);
    EXPECT_THROW(storage.read_block(0), libaccint::InvalidStateException);
}

TEST(OutOfCoreStorage, InvalidBlockIndexThrows) {
    OutOfCoreConfig config;
    config.scratch_dir = "/tmp/libaccint_test_ooc7";
    config.delete_on_destroy = true;

    OutOfCoreThreeCenterStorage storage(3, 6, config);
    EXPECT_THROW(storage.block_range(100), libaccint::InvalidArgumentException);
}

TEST(OutOfCoreStorage, WrongDataSizeThrows) {
    OutOfCoreConfig config;
    config.scratch_dir = "/tmp/libaccint_test_ooc8";
    config.block_size_aux = 3;
    config.delete_on_destroy = true;

    OutOfCoreThreeCenterStorage storage(3, 6, config);

    std::vector<Real> wrong_size(5, 1.0);
    EXPECT_THROW(storage.write_block(0, wrong_size),
                 libaccint::InvalidArgumentException);
}

// ============================================================================
// Cleanup test
// ============================================================================

TEST(OutOfCoreStorage, ScratchFilesDeletedOnDestroy) {
    std::string scratch_dir = "/tmp/libaccint_test_ooc_cleanup";

    {
        OutOfCoreConfig config;
        config.scratch_dir = scratch_dir;
        config.block_size_aux = 3;
        config.delete_on_destroy = true;

        OutOfCoreThreeCenterStorage storage(3, 6, config);

        std::vector<Real> data(3 * 3 * 3, 1.0);
        storage.write_block(0, data);
    }

    // After destruction, files should be cleaned up
    if (std::filesystem::exists(scratch_dir)) {
        bool found_block = false;
        for (const auto& entry : std::filesystem::directory_iterator(scratch_dir)) {
            if (entry.path().string().find("libaccint_ooc") != std::string::npos) {
                found_block = true;
            }
        }
        EXPECT_FALSE(found_block);
        std::filesystem::remove_all(scratch_dir);
    }
}

}  // anonymous namespace
}  // namespace libaccint::df
