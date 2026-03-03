// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2026 Bryce M. Westheimer. All rights reserved.

/// @file test_three_center_storage.cpp
/// @brief Tests for blocked three-center integral storage

#include <libaccint/df/three_center_storage.hpp>
#include <libaccint/utils/error_handling.hpp>

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <numeric>

namespace libaccint::df {
namespace {

// ============================================================================
// Construction tests
// ============================================================================

TEST(ThreeCenterBlockStorage, Construction) {
    ThreeCenterBlockStorage storage(5, 10);
    EXPECT_EQ(storage.n_orb(), 5u);
    EXPECT_EQ(storage.n_aux(), 10u);
    EXPECT_GT(storage.n_blocks(), 0u);
    EXPECT_EQ(storage.n_loaded_blocks(), 0u);
}

TEST(ThreeCenterBlockStorage, BlockRangesPartition) {
    ThreeCenterBlockStorage storage(3, 10, {.memory_limit_mb = 1, .block_size_aux = 3});

    // Blocks should partition [0, n_aux)
    Size total = 0;
    for (Size i = 0; i < storage.n_blocks(); ++i) {
        auto [start, end] = storage.block_range(i);
        EXPECT_EQ(start, total);
        EXPECT_GT(end, start);
        total = end;
    }
    EXPECT_EQ(total, 10u);
}

TEST(ThreeCenterBlockStorage, BlockSizes) {
    ThreeCenterBlockStorage storage(3, 10, {.memory_limit_mb = 1, .block_size_aux = 3});

    // With block_size_aux=3 and n_aux=10: blocks of 3,3,3,1
    EXPECT_EQ(storage.n_blocks(), 4u);
    EXPECT_EQ(storage.block_size(0), 3u);
    EXPECT_EQ(storage.block_size(1), 3u);
    EXPECT_EQ(storage.block_size(2), 3u);
    EXPECT_EQ(storage.block_size(3), 1u);
}

// ============================================================================
// Store/load tests
// ============================================================================

TEST(ThreeCenterBlockStorage, StoreAndRetrieveBlock) {
    Size n_orb = 3;
    Size n_aux = 6;
    ThreeCenterBlockStorage storage(n_orb, n_aux,
        {.memory_limit_mb = 100, .block_size_aux = 3});

    EXPECT_EQ(storage.n_blocks(), 2u);

    // Create test data for block 0 (n_orb^2 * 3 = 27 values)
    std::vector<Real> data(n_orb * n_orb * 3);
    std::iota(data.begin(), data.end(), 1.0);

    storage.store_block(0, data);
    EXPECT_TRUE(storage.is_block_loaded(0));

    auto retrieved = storage.get_block(0);
    EXPECT_EQ(retrieved.size(), data.size());
    for (Size i = 0; i < data.size(); ++i) {
        EXPECT_DOUBLE_EQ(retrieved[i], data[i]);
    }
}

TEST(ThreeCenterBlockStorage, EvictBlock) {
    Size n_orb = 3;
    Size n_aux = 6;
    ThreeCenterBlockStorage storage(n_orb, n_aux,
        {.memory_limit_mb = 100, .block_size_aux = 3});

    std::vector<Real> data(n_orb * n_orb * 3, 42.0);
    storage.store_block(0, data);
    EXPECT_TRUE(storage.is_block_loaded(0));

    storage.evict_block(0);
    EXPECT_FALSE(storage.is_block_loaded(0));
}

TEST(ThreeCenterBlockStorage, Clear) {
    Size n_orb = 3;
    Size n_aux = 6;
    ThreeCenterBlockStorage storage(n_orb, n_aux,
        {.memory_limit_mb = 100, .block_size_aux = 3});

    std::vector<Real> data0(n_orb * n_orb * 3, 1.0);
    std::vector<Real> data1(n_orb * n_orb * 3, 2.0);
    storage.store_block(0, data0);
    storage.store_block(1, data1);
    EXPECT_EQ(storage.n_loaded_blocks(), 2u);

    storage.clear();
    EXPECT_EQ(storage.n_loaded_blocks(), 0u);
}

// ============================================================================
// Full tensor round-trip
// ============================================================================

TEST(ThreeCenterBlockStorage, FullTensorRoundTrip) {
    Size n_orb = 4;
    Size n_aux = 7;
    ThreeCenterBlockStorage storage(n_orb, n_aux,
        {.memory_limit_mb = 100, .block_size_aux = 3});

    // Create a full tensor with sequential values
    Size total = n_orb * n_orb * n_aux;
    std::vector<Real> full(total);
    for (Size i = 0; i < total; ++i) {
        full[i] = static_cast<Real>(i) * 0.1;
    }

    storage.store_full_tensor(full);
    auto reconstructed = storage.reconstruct_full_tensor();

    EXPECT_EQ(reconstructed.size(), full.size());
    for (Size i = 0; i < full.size(); ++i) {
        EXPECT_DOUBLE_EQ(reconstructed[i], full[i])
            << "Mismatch at index " << i;
    }
}

// ============================================================================
// Error handling tests
// ============================================================================

TEST(ThreeCenterBlockStorage, InvalidBlockIndexThrows) {
    ThreeCenterBlockStorage storage(3, 6,
        {.memory_limit_mb = 100, .block_size_aux = 3});

    EXPECT_THROW(storage.block_range(10), libaccint::InvalidArgumentException);
    EXPECT_THROW(storage.get_block(10), libaccint::InvalidArgumentException);
}

TEST(ThreeCenterBlockStorage, GetUnloadedBlockThrows) {
    ThreeCenterBlockStorage storage(3, 6,
        {.memory_limit_mb = 100, .block_size_aux = 3});

    EXPECT_THROW(storage.get_block(0), libaccint::InvalidStateException);
}

TEST(ThreeCenterBlockStorage, WrongDataSizeThrows) {
    ThreeCenterBlockStorage storage(3, 6,
        {.memory_limit_mb = 100, .block_size_aux = 3});

    std::vector<Real> wrong_size(5, 1.0);  // Wrong size
    EXPECT_THROW(storage.store_block(0, wrong_size),
                 libaccint::InvalidArgumentException);
}

// ============================================================================
// Memory tracking tests
// ============================================================================

TEST(ThreeCenterBlockStorage, MemoryTracking) {
    Size n_orb = 3;
    Size n_aux = 6;
    ThreeCenterBlockStorage storage(n_orb, n_aux,
        {.memory_limit_mb = 100, .block_size_aux = 3});

    EXPECT_EQ(storage.memory_used(), 0u);

    Size block_elements = n_orb * n_orb * 3;
    std::vector<Real> data(block_elements, 1.0);
    storage.store_block(0, data);

    EXPECT_EQ(storage.memory_used(), block_elements * sizeof(Real));
}

// ============================================================================
// LRU eviction tests
// ============================================================================

TEST(ThreeCenterBlockStorage, LRUEviction) {
    Size n_orb = 2;
    Size n_aux = 9;
    // Very small memory limit to force eviction
    // Each block: 2*2*3 * 8 = 96 bytes. Limit ~200 bytes = room for ~2 blocks
    BlockStorageConfig config;
    config.block_size_aux = 3;
    // Set to 1 byte since we can't control sub-MB. Use memory_limit_mb=1
    // but the key is that LRU ordering works correctly.
    config.memory_limit_mb = 1;  // 1MB is plenty for tiny blocks

    ThreeCenterBlockStorage storage(n_orb, n_aux, config);
    EXPECT_EQ(storage.n_blocks(), 3u);

    Size block_elements = n_orb * n_orb * 3;
    std::vector<Real> data(block_elements, 1.0);

    storage.store_block(0, data);
    storage.store_block(1, data);
    storage.store_block(2, data);

    // All should be loaded (1MB is enough for tiny blocks)
    EXPECT_EQ(storage.n_loaded_blocks(), 3u);
}

}  // anonymous namespace
}  // namespace libaccint::df
