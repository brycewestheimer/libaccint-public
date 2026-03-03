// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2026 Bryce M. Westheimer. All rights reserved.

/// @file test_thread_local_pool.cpp
/// @brief Unit tests for PartitionedPool and ScopedPartitionedPool

#include <libaccint/memory/thread_local_pool.hpp>

#include <gtest/gtest.h>
#include <vector>

#if defined(_OPENMP)
#include <omp.h>
#endif

using namespace libaccint::memory;

// =============================================================================
// PartitionedPool Tests
// =============================================================================

TEST(PartitionedPoolTest, ConstructWithExplicitCount) {
    PartitionedPool pool(4);
    EXPECT_EQ(pool.n_partitions(), 4);
}

TEST(PartitionedPoolTest, ConstructWithAutoDetect) {
    PartitionedPool pool(0);
    EXPECT_GE(pool.n_partitions(), 1);
}

TEST(PartitionedPoolTest, PoolAccessByIndex) {
    PartitionedPool pool(4);

    // Access each partition
    for (int i = 0; i < 4; ++i) {
        auto& p = pool.pool(i);
        // Acquire and release to verify the pool works
        auto buf = p.acquire(64);
        EXPECT_TRUE(static_cast<bool>(buf));
        EXPECT_GE(buf.size(), 64u);
    }
}

TEST(PartitionedPoolTest, AcquireFromPartitions) {
    PartitionedPool pool(2);

    auto buf0 = pool.pool(0).acquire(128);
    auto buf1 = pool.pool(1).acquire(256);

    EXPECT_TRUE(static_cast<bool>(buf0));
    EXPECT_TRUE(static_cast<bool>(buf1));
    EXPECT_GE(buf0.size(), 128u);
    EXPECT_GE(buf1.size(), 256u);

    // Different partitions should give different memory
    EXPECT_NE(buf0.data(), buf1.data());
}

TEST(PartitionedPoolTest, ClearAll) {
    PartitionedPool pool(4);

    // Acquire and release some buffers
    for (int i = 0; i < 4; ++i) {
        auto buf = pool.pool(i).acquire(1024);
        // buf goes out of scope, returning to pool
    }

    // After clearing, pools should be empty
    pool.clear_all();

    auto stats = pool.aggregate_stats();
    EXPECT_EQ(stats.current_pooled, 0u);
}

TEST(PartitionedPoolTest, AggregateStats) {
    PartitionedPool pool(4);

    // Do some allocations
    for (int i = 0; i < 4; ++i) {
        auto buf = pool.pool(i).acquire(64);
        (void)buf;  // Let it go out of scope
    }

    auto stats = pool.aggregate_stats();
    EXPECT_EQ(stats.total_allocations, 4u);
}

#if defined(_OPENMP)
TEST(PartitionedPoolTest, ParallelAccess) {
    const int n_threads = 4;
    PartitionedPool pool(n_threads);

    std::vector<bool> success(n_threads, false);

    #pragma omp parallel num_threads(n_threads)
    {
        auto& my_pool = pool.thread_pool();
        auto buf = my_pool.acquire(512);

        int tid = omp_get_thread_num();
        success[tid] = (buf.data() != nullptr && buf.size() >= 512);
    }

    for (int i = 0; i < n_threads; ++i) {
        EXPECT_TRUE(success[i]) << "Thread " << i << " failed to allocate";
    }
}

TEST(PartitionedPoolTest, ParallelNoContention) {
    // Each thread uses its own partition, so no contention
    const int n_threads = 4;
    const int n_iterations = 100;
    PartitionedPool pool(n_threads);

    #pragma omp parallel num_threads(n_threads)
    {
        auto& my_pool = pool.thread_pool();

        for (int iter = 0; iter < n_iterations; ++iter) {
            auto buf = my_pool.acquire(256);
            EXPECT_TRUE(static_cast<bool>(buf));
            // buf goes out of scope, returning to pool
        }
    }

    auto stats = pool.aggregate_stats();
    EXPECT_EQ(stats.total_allocations,
              static_cast<std::size_t>(n_threads * n_iterations));
    // Most should be pool hits after the first round
    EXPECT_GT(stats.pool_hits, 0u);
}
#endif

// =============================================================================
// ScopedPartitionedPool Tests
// =============================================================================

TEST(ScopedPartitionedPoolTest, BasicUsage) {
    ScopedPartitionedPool scoped(2);

    auto buf = scoped.pool().pool(0).acquire(64);
    EXPECT_TRUE(static_cast<bool>(buf));
}

TEST(ScopedPartitionedPoolTest, CleansUpOnDestruction) {
    // This test primarily verifies no crashes/leaks
    {
        ScopedPartitionedPool scoped(4);
        for (int i = 0; i < 4; ++i) {
            auto buf = scoped.pool().pool(i).acquire(512);
            (void)buf;
        }
    }
    // Pools destroyed here — should not leak
    SUCCEED();
}

