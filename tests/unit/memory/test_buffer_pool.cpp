// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2026 Bryce M. Westheimer. All rights reserved.

/// @file test_buffer_pool.cpp
/// @brief Unit tests for BatchBufferPool
///
/// Covers Task 2.3.2: acquire, release, recycling, clear, stats,
/// pre_warm, default AMClass, and thread-local isolation.

#include <libaccint/memory/buffer_pool.hpp>

#include <gtest/gtest.h>
#include <thread>
#include <vector>

using namespace libaccint::memory;
using namespace libaccint;

// =============================================================================
// Helper: construct AMClass values for testing
// =============================================================================

namespace {

AMClass make_am(int La, int Lb, int Lc, int Ld) {
    return AMClass{La, Lb, Lc, Ld};
}

}  // namespace

// =============================================================================
// BatchBufferPool Tests
// =============================================================================

TEST(BatchBufferPoolTest, AcquireReturnsBuffer) {
    BatchBufferPool pool;
    AMClass am = make_am(0, 0, 0, 0);

    IntegralBuffer buf = pool.acquire(am);
    // An empty IntegralBuffer is valid (default constructed)
    EXPECT_EQ(buf.n_integrals(), 0u);
}

TEST(BatchBufferPoolTest, DefaultAMClassAcquire) {
    BatchBufferPool pool;

    // acquire() with default AMClass{}
    IntegralBuffer buf = pool.acquire();
    EXPECT_EQ(buf.n_integrals(), 0u);

    auto s = pool.stats();
    EXPECT_EQ(s.total_acquires, 1u);
    EXPECT_EQ(s.pool_misses, 1u);
}

TEST(BatchBufferPoolTest, ReleaseAndReacquireRecyclesBuffer) {
    BatchBufferPool pool;
    AMClass am = make_am(1, 0, 1, 0);

    // Acquire → release → acquire should recycle
    IntegralBuffer buf1 = pool.acquire(am);
    pool.release(std::move(buf1), am);

    IntegralBuffer buf2 = pool.acquire(am);

    auto s = pool.stats();
    EXPECT_EQ(s.pool_hits, 1u);
    EXPECT_EQ(s.pool_misses, 1u);
    EXPECT_EQ(s.total_acquires, 2u);
}

TEST(BatchBufferPoolTest, AcquireUnknownClassIsPoolMiss) {
    BatchBufferPool pool;
    AMClass am1 = make_am(0, 0, 0, 0);
    AMClass am2 = make_am(2, 1, 0, 0);

    // Pre-populate with am1
    IntegralBuffer b1 = pool.acquire(am1);
    pool.release(std::move(b1), am1);

    // Acquire am2 (not in pool) should be a miss
    IntegralBuffer b2 = pool.acquire(am2);

    auto s = pool.stats();
    EXPECT_EQ(s.pool_misses, 2u);  // First acquire + am2 acquire
}

TEST(BatchBufferPoolTest, ClearEmptiesAllPools) {
    BatchBufferPool pool;
    AMClass am1 = make_am(0, 0, 0, 0);
    AMClass am2 = make_am(1, 1, 1, 1);

    // Add buffers for two AM classes
    IntegralBuffer b1 = pool.acquire(am1);
    pool.release(std::move(b1), am1);
    IntegralBuffer b2 = pool.acquire(am2);
    pool.release(std::move(b2), am2);

    pool.clear();

    // After clear, acquiring should be misses again
    auto stats_before = pool.stats();
    IntegralBuffer b3 = pool.acquire(am1);
    auto stats_after = pool.stats();

    EXPECT_EQ(stats_after.pool_misses, stats_before.pool_misses + 1);
}

TEST(BatchBufferPoolTest, StatsAccuracy) {
    BatchBufferPool pool;
    AMClass am = make_am(0, 0, 0, 0);

    auto s0 = pool.stats();
    EXPECT_EQ(s0.total_acquires, 0u);
    EXPECT_EQ(s0.pool_hits, 0u);
    EXPECT_EQ(s0.pool_misses, 0u);

    // 3 acquires (all misses)
    auto b1 = pool.acquire(am);
    auto b2 = pool.acquire(am);
    auto b3 = pool.acquire(am);

    auto s1 = pool.stats();
    EXPECT_EQ(s1.total_acquires, 3u);
    EXPECT_EQ(s1.pool_misses, 3u);
    EXPECT_EQ(s1.pool_hits, 0u);

    // Release all, then acquire (should be hits)
    pool.release(std::move(b1), am);
    pool.release(std::move(b2), am);
    pool.release(std::move(b3), am);

    auto b4 = pool.acquire(am);
    auto b5 = pool.acquire(am);

    auto s2 = pool.stats();
    EXPECT_EQ(s2.total_acquires, 5u);
    EXPECT_EQ(s2.pool_hits, 2u);
}

TEST(BatchBufferPoolTest, ResetStats) {
    BatchBufferPool pool;
    AMClass am = make_am(0, 0, 0, 0);

    auto b = pool.acquire(am);
    EXPECT_EQ(pool.stats().total_acquires, 1u);

    pool.reset_stats();

    auto s = pool.stats();
    EXPECT_EQ(s.total_acquires, 0u);
    EXPECT_EQ(s.pool_hits, 0u);
    EXPECT_EQ(s.pool_misses, 0u);
}

TEST(BatchBufferPoolTest, ThreadLocalPoolIsolation) {
    // Different threads should get different pool instances
    BatchBufferPool* ptr_main = &get_thread_local_batch_pool();

    std::atomic<BatchBufferPool*> ptr_other{nullptr};

    std::thread t([&]() {
        ptr_other.store(&get_thread_local_batch_pool());
    });
    t.join();

    EXPECT_NE(ptr_main, ptr_other.load());
}

TEST(BatchBufferPoolTest, MultipleAMClassesIndependent) {
    BatchBufferPool pool;
    AMClass am_ss = make_am(0, 0, 0, 0);
    AMClass am_sp = make_am(0, 1, 0, 0);
    AMClass am_pp = make_am(1, 1, 1, 1);

    // Release buffers for each class
    IntegralBuffer b_ss = pool.acquire(am_ss);
    pool.release(std::move(b_ss), am_ss);

    IntegralBuffer b_sp = pool.acquire(am_sp);
    pool.release(std::move(b_sp), am_sp);

    // Acquire am_ss should hit, am_pp should miss
    auto b_ss2 = pool.acquire(am_ss);
    auto b_pp = pool.acquire(am_pp);

    auto s = pool.stats();
    // am_ss (2 acquires: 1 miss + 1 hit), am_sp (1 miss), am_pp (1 miss)
    EXPECT_EQ(s.total_acquires, 4u);
    EXPECT_EQ(s.pool_hits, 1u);
    EXPECT_EQ(s.pool_misses, 3u);
}
