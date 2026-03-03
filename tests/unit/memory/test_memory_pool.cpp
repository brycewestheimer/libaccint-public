// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2026 Bryce M. Westheimer. All rights reserved.

/// @file test_memory_pool.cpp
/// @brief Unit tests for MemoryPool, GlobalMemoryPool, and PooledBuffer
///
/// Covers Tasks 2.3.1 (MemoryPool), 2.3.3 (GlobalMemoryPool thread-safety),
/// 2.3.4 (PooledBuffer lifecycle), 2.3.6 (memory tracking accuracy).

#include <libaccint/memory/memory_pool.hpp>

#include <gtest/gtest.h>

#include <algorithm>
#include <atomic>
#include <cstdint>
#include <thread>
#include <vector>

using namespace libaccint::memory;

// =============================================================================
// Task 2.3.1: MemoryPool Unit Tests
// =============================================================================

TEST(MemoryPoolTest, AcquireReturnsNonNull) {
    MemoryPool pool;
    for (std::size_t sz : pool_config::SIZE_CLASSES) {
        auto buf = pool.acquire(sz);
        EXPECT_NE(buf.data(), nullptr) << "size=" << sz;
        EXPECT_GE(buf.size(), sz);
    }
}

TEST(MemoryPoolTest, AcquireReturnsAlignedPointer) {
    MemoryPool pool;
    for (std::size_t sz : pool_config::SIZE_CLASSES) {
        auto buf = pool.acquire(sz);
        auto addr = reinterpret_cast<std::uintptr_t>(buf.data());
        EXPECT_EQ(addr % DEFAULT_ALIGNMENT, 0u)
            << "Pointer not " << DEFAULT_ALIGNMENT << "-byte aligned for size=" << sz;
    }
}

TEST(MemoryPoolTest, AcquireZeroBytesReturnsEmpty) {
    MemoryPool pool;
    auto buf = pool.acquire(0);
    EXPECT_EQ(buf.data(), nullptr);
    EXPECT_EQ(buf.size(), 0u);
}

TEST(MemoryPoolTest, SizeClassRouting) {
    MemoryPool pool;

    // 100 bytes should route to 256B class
    auto buf100 = pool.acquire(100);
    EXPECT_EQ(buf100.size(), 256u);

    // 300 bytes should route to 1KB class
    auto buf300 = pool.acquire(300);
    EXPECT_EQ(buf300.size(), 1024u);

    // 5000 bytes should route to 16KB class
    auto buf5000 = pool.acquire(5000);
    EXPECT_EQ(buf5000.size(), 16384u);

    // Exact boundary: 256 bytes → 256B class
    auto bufExact = pool.acquire(256);
    EXPECT_EQ(bufExact.size(), 256u);
}

TEST(MemoryPoolTest, OversizedAllocationBypassesPool) {
    MemoryPool pool;
    constexpr std::size_t oversized = 5 * 1024 * 1024;  // 5 MB > 4 MB max

    auto buf = pool.acquire(oversized);
    EXPECT_NE(buf.data(), nullptr);
    EXPECT_EQ(buf.size(), oversized);

    auto s = pool.stats();
    EXPECT_EQ(s.oversized_allocations, 1u);
}

TEST(MemoryPoolTest, ReleaseAndReacquireReusesMemory) {
    MemoryPool pool;

    void* first_ptr = nullptr;
    {
        auto buf = pool.acquire(128);
        first_ptr = buf.data();
        // buf goes out of scope, returning to pool
    }

    // Second acquire of similar size should reuse the pooled memory
    auto buf2 = pool.acquire(128);
    EXPECT_EQ(buf2.data(), first_ptr);

    auto s = pool.stats();
    EXPECT_EQ(s.pool_hits, 1u);
}

TEST(MemoryPoolTest, ClearFreesPooledMemory) {
    MemoryPool pool;

    void* first_ptr = nullptr;
    {
        auto buf = pool.acquire(512);
        first_ptr = buf.data();
    }

    // Should be a hit before clear
    auto pre = pool.stats();
    EXPECT_EQ(pre.current_pooled, 1u);

    pool.clear();

    auto post = pool.stats();
    EXPECT_EQ(post.current_pooled, 0u);
    EXPECT_EQ(post.current_pooled_bytes, 0u);

    // After clear, next acquire should be a miss (fresh allocation)
    auto buf2 = pool.acquire(512);
    auto s = pool.stats();
    // pool_misses should have incremented
    EXPECT_GT(s.pool_misses, pre.pool_misses);
}

TEST(MemoryPoolTest, StatsAccuracy) {
    MemoryPool pool;

    // 3 normal acquires
    auto b1 = pool.acquire(64);
    auto b2 = pool.acquire(1024);
    auto b3 = pool.acquire(64);

    auto s = pool.stats();
    EXPECT_EQ(s.total_allocations, 3u);
    EXPECT_EQ(s.pool_misses, 3u);
    EXPECT_EQ(s.pool_hits, 0u);
    EXPECT_EQ(s.oversized_allocations, 0u);

    // Invariant: total_allocations == pool_hits + pool_misses + oversized
    EXPECT_EQ(s.total_allocations, s.pool_hits + s.pool_misses + s.oversized_allocations);
}

TEST(MemoryPoolTest, StatsConsistencyAfterMixedOps) {
    MemoryPool pool;

    // Acquire and release to generate pool hits
    {
        auto b = pool.acquire(256);
    }
    auto hit_buf = pool.acquire(256);  // Should reuse

    // One oversized
    auto oversized = pool.acquire(5 * 1024 * 1024);

    auto s = pool.stats();
    EXPECT_EQ(s.total_allocations, 3u);
    EXPECT_EQ(s.pool_hits, 1u);
    EXPECT_EQ(s.pool_misses, 1u);
    EXPECT_EQ(s.oversized_allocations, 1u);
    EXPECT_EQ(s.total_allocations, s.pool_hits + s.pool_misses + s.oversized_allocations);
}

TEST(MemoryPoolTest, FreeListCapacityLimit) {
    MemoryPool pool;
    constexpr std::size_t alloc_size = 256;

    // Acquire and release more buffers than MAX_BUFFERS_PER_CLASS
    std::vector<void*> ptrs;
    for (std::size_t i = 0; i < pool_config::MAX_BUFFERS_PER_CLASS + 4; ++i) {
        auto buf = pool.acquire(alloc_size);
        ptrs.push_back(buf.data());
    }
    // All go out of scope - pool can only keep MAX_BUFFERS_PER_CLASS

    auto s = pool.stats();
    EXPECT_LE(s.current_pooled, pool_config::MAX_BUFFERS_PER_CLASS);
}

TEST(MemoryPoolTest, MultiSizeClassRoundTrip) {
    MemoryPool pool;

    void* ptr64 = nullptr;
    void* ptr4k = nullptr;

    {
        auto b1 = pool.acquire(64);
        auto b2 = pool.acquire(4096);
        ptr64 = b1.data();
        ptr4k = b2.data();
    }

    // Re-acquire different sizes — should hit correct free lists
    auto r1 = pool.acquire(64);
    auto r2 = pool.acquire(4096);

    EXPECT_EQ(r1.data(), ptr64);
    EXPECT_EQ(r2.data(), ptr4k);
}

// =============================================================================
// Task 2.3.4: PooledBuffer Lifecycle Tests
// =============================================================================

TEST(PooledBufferTest, RAIIDestruction) {
    MemoryPool pool;

    {
        auto buf = pool.acquire(256);
        EXPECT_NE(buf.data(), nullptr);
    }
    // After destruction, pool should have the buffer back
    auto s = pool.stats();
    EXPECT_EQ(s.current_pooled, 1u);
}

TEST(PooledBufferTest, MoveConstruction) {
    MemoryPool pool;
    auto original = pool.acquire(512);
    void* original_ptr = original.data();
    std::size_t original_size = original.size();

    PooledBuffer moved(std::move(original));

    EXPECT_EQ(moved.data(), original_ptr);
    EXPECT_EQ(moved.size(), original_size);
    EXPECT_EQ(original.data(), nullptr);
    EXPECT_EQ(original.size(), 0u);
    EXPECT_FALSE(static_cast<bool>(original));
}

TEST(PooledBufferTest, MoveAssignment) {
    MemoryPool pool;
    auto buf1 = pool.acquire(256);
    auto buf2 = pool.acquire(1024);

    void* ptr2 = buf2.data();

    // Move-assign buf2 into buf1 — buf1's old buffer should be released
    buf1 = std::move(buf2);

    EXPECT_EQ(buf1.data(), ptr2);
    EXPECT_EQ(buf2.data(), nullptr);

    // The old buf1 buffer should now be in the pool
    auto s = pool.stats();
    EXPECT_GE(s.current_pooled, 1u);
}

TEST(PooledBufferTest, ManualRelease) {
    MemoryPool pool;
    auto buf = pool.acquire(256);
    void* ptr = buf.data();

    void* released = buf.release();
    EXPECT_EQ(released, ptr);
    EXPECT_EQ(buf.data(), nullptr);
    EXPECT_EQ(buf.size(), 0u);

    // Destructor should be a no-op — verify pool doesn't get the buffer
    auto stats_before = pool.stats();
    // buf goes out of scope at end of test; manually free
    aligned_free(released);
}

TEST(PooledBufferTest, Accessors) {
    MemoryPool pool;
    auto buf = pool.acquire(100);

    EXPECT_NE(buf.data(), nullptr);
    EXPECT_EQ(buf.size(), 256u);  // Rounded up to 256B size class
    EXPECT_TRUE(static_cast<bool>(buf));

    // Typed accessors
    double* dp = buf.as<double>();
    EXPECT_EQ(static_cast<void*>(dp), buf.data());
}

TEST(PooledBufferTest, DefaultConstructed) {
    PooledBuffer buf;
    EXPECT_EQ(buf.data(), nullptr);
    EXPECT_EQ(buf.size(), 0u);
    EXPECT_FALSE(static_cast<bool>(buf));
    // Destructor should be safe on default-constructed buffer
}

TEST(PooledBufferTest, SelfMoveAssignment) {
    MemoryPool pool;
    auto buf = pool.acquire(256);
    void* ptr = buf.data();

    buf = std::move(buf);

    // Self-move should not crash or leak
    EXPECT_EQ(buf.data(), ptr);
}

TEST(PooledBufferTest, MovedFromSizeClassReset) {
    MemoryPool pool;
    auto buf = pool.acquire(256);

    PooledBuffer moved(std::move(buf));

    // Moved-from object should have zeroed all fields
    EXPECT_EQ(buf.data(), nullptr);
    EXPECT_EQ(buf.size(), 0u);
    // Destructor of moved-from buf is safe (ptr_ == nullptr)
}

TEST(PooledBufferTest, GlobalPoolBuffer) {
    auto& gpool = GlobalMemoryPool::instance();

    void* ptr = nullptr;
    {
        auto buf = gpool.acquire(512);
        ptr = buf.data();
        EXPECT_NE(ptr, nullptr);
        // When buf goes out of scope, it should release through
        // GlobalMemoryPool::release() (mutex-protected)
    }

    // Verify the buffer was returned to the pool by acquiring again
    auto buf2 = gpool.acquire(512);
    EXPECT_EQ(buf2.data(), ptr);  // Reused from pool
}

// =============================================================================
// Task 2.3.3: GlobalMemoryPool Unit Tests
// =============================================================================

TEST(GlobalMemoryPoolTest, SingletonIdentity) {
    auto& a = GlobalMemoryPool::instance();
    auto& b = GlobalMemoryPool::instance();
    EXPECT_EQ(&a, &b);
}

TEST(GlobalMemoryPoolTest, AcquireReturnsValidMemory) {
    auto& pool = GlobalMemoryPool::instance();
    auto buf = pool.acquire(256);

    EXPECT_NE(buf.data(), nullptr);
    EXPECT_GE(buf.size(), 256u);

    auto addr = reinterpret_cast<std::uintptr_t>(buf.data());
    EXPECT_EQ(addr % DEFAULT_ALIGNMENT, 0u);
}

TEST(GlobalMemoryPoolTest, AcquireAndRelease) {
    auto& pool = GlobalMemoryPool::instance();

    // A full acquire/release cycle should not leak
    {
        auto buf = pool.acquire(1024);
        EXPECT_NE(buf.data(), nullptr);
    }
    // No crash means success (ASAN would catch leaks)
    SUCCEED();
}

TEST(GlobalMemoryPoolTest, ClearResetsPool) {
    auto& pool = GlobalMemoryPool::instance();

    {
        auto buf = pool.acquire(256);
    }

    pool.clear();

    auto s = pool.stats();
    EXPECT_EQ(s.current_pooled, 0u);
}

TEST(GlobalMemoryPoolTest, StatsAccuracy) {
    auto& pool = GlobalMemoryPool::instance();
    pool.clear();

    // Reset by clearing — note: stats counters are NOT reset by clear()
    auto baseline = pool.stats();

    auto b1 = pool.acquire(128);
    auto b2 = pool.acquire(4096);

    auto s = pool.stats();
    EXPECT_EQ(s.total_allocations, baseline.total_allocations + 2);
}

// --- Stress Tests (Task 2.3.3) ---

TEST(GlobalMemoryPoolTest, ConcurrentAcquireRelease) {
    auto& pool = GlobalMemoryPool::instance();
    pool.clear();

    constexpr int n_threads = 4;
    constexpr int n_iterations = 10000;
    std::atomic<int> success_count{0};

    auto work = [&]() {
        for (int i = 0; i < n_iterations; ++i) {
            std::size_t sz = 64 + (i % 8) * 128;  // Vary sizes 64–960
            auto buf = pool.acquire(sz);
            if (buf.data() != nullptr && buf.size() >= sz) {
                success_count.fetch_add(1, std::memory_order_relaxed);
            }
            // buf released here through mutex-protected path
        }
    };

    std::vector<std::thread> threads;
    threads.reserve(n_threads);
    for (int t = 0; t < n_threads; ++t) {
        threads.emplace_back(work);
    }
    for (auto& t : threads) {
        t.join();
    }

    EXPECT_EQ(success_count.load(), n_threads * n_iterations);
}

TEST(GlobalMemoryPoolTest, ConcurrentPooledBufferDestruction) {
    auto& pool = GlobalMemoryPool::instance();
    pool.clear();

    constexpr int n_threads = 4;
    constexpr int n_iterations = 5000;

    auto work = [&]() {
        for (int i = 0; i < n_iterations; ++i) {
            // Create buffer in this scope; it goes out of scope and
            // the PooledBuffer destructor calls GlobalMemoryPool::release()
            // through the virtual dispatch (mutex-protected)
            auto buf = pool.acquire(256 * ((i % 4) + 1));
            // Write to buffer to ensure it's valid
            auto* p = static_cast<char*>(buf.data());
            p[0] = static_cast<char>(i & 0xFF);
        }
    };

    std::vector<std::thread> threads;
    threads.reserve(n_threads);
    for (int t = 0; t < n_threads; ++t) {
        threads.emplace_back(work);
    }
    for (auto& t : threads) {
        t.join();
    }

    // If we get here without crash/TSAN errors, the mutex protection works
    SUCCEED();
}

TEST(GlobalMemoryPoolTest, MixedOperations) {
    auto& pool = GlobalMemoryPool::instance();
    pool.clear();

    constexpr int n_threads = 4;
    constexpr int n_iterations = 2000;

    auto work = [&](int tid) {
        for (int i = 0; i < n_iterations; ++i) {
            // Mix: acquire/release, stats, clear (occasionally)
            auto buf = pool.acquire(128 + tid * 64);

            if (i % 100 == 0) {
                auto s = pool.stats();
                (void)s;
            }

            if (i % 500 == 0 && tid == 0) {
                pool.clear();
            }
        }
    };

    std::vector<std::thread> threads;
    threads.reserve(n_threads);
    for (int t = 0; t < n_threads; ++t) {
        threads.emplace_back(work, t);
    }
    for (auto& t : threads) {
        t.join();
    }

    SUCCEED();
}

// =============================================================================
// Task 2.3.6: Memory Tracking Accuracy Tests (CPU pool)
// =============================================================================

TEST(MemoryTrackingTest, PoolStatsConsistency) {
    MemoryPool pool;

    // Do a mix of operations
    auto b1 = pool.acquire(64);
    auto b2 = pool.acquire(1024);
    auto b3 = pool.acquire(5 * 1024 * 1024);  // oversized

    {
        auto b4 = pool.acquire(64);
        // b4 released (returns to pool)
    }
    auto b5 = pool.acquire(64);  // should be pool hit

    auto s = pool.stats();
    EXPECT_EQ(s.total_allocations,
              s.pool_hits + s.pool_misses + s.oversized_allocations)
        << "Invariant violated: total != hits + misses + oversized";
}

TEST(MemoryTrackingTest, PoolStatsClearBehavior) {
    MemoryPool pool;

    auto b1 = pool.acquire(256);
    auto s_before = pool.stats();
    EXPECT_EQ(s_before.total_allocations, 1u);

    // clear() should free pooled memory but NOT reset stats counters
    // (this is the documented behavior per code review)
    pool.clear();

    auto s_after = pool.stats();
    EXPECT_EQ(s_after.current_pooled, 0u);
    EXPECT_EQ(s_after.current_pooled_bytes, 0u);
    // Stats counters are preserved after clear
    EXPECT_EQ(s_after.total_allocations, s_before.total_allocations);
    EXPECT_EQ(s_after.pool_misses, s_before.pool_misses);
}

TEST(MemoryTrackingTest, GlobalPoolStatsConsistent) {
    auto& pool = GlobalMemoryPool::instance();
    pool.clear();

    auto baseline = pool.stats();

    auto b1 = pool.acquire(128);
    auto b2 = pool.acquire(256);

    auto s = pool.stats();
    EXPECT_EQ(s.total_allocations, baseline.total_allocations + 2);
    EXPECT_EQ(s.total_allocations,
              s.pool_hits + s.pool_misses + s.oversized_allocations);
}

TEST(MemoryTrackingTest, MultipleAllocationTracking) {
    MemoryPool pool;

    // Allocate 5 buffers of different sizes
    auto b1 = pool.acquire(64);    // -> 256B class
    auto b2 = pool.acquire(500);   // -> 1024B class
    auto b3 = pool.acquire(4096);  // -> 4096B class
    auto b4 = pool.acquire(10000); // -> 16384B class
    auto b5 = pool.acquire(50000); // -> 65536B class

    auto s1 = pool.stats();
    EXPECT_EQ(s1.total_allocations, 5u);
    EXPECT_EQ(s1.pool_misses, 5u);
    EXPECT_EQ(s1.pool_hits, 0u);
    EXPECT_EQ(s1.current_pooled, 0u);  // All still held

    // Release 2 buffers (b1 and b3)
    { auto tmp = std::move(b1); }
    { auto tmp = std::move(b3); }

    auto s2 = pool.stats();
    EXPECT_EQ(s2.current_pooled, 2u);
    // current_pooled_bytes should be sum of the two size classes returned
    EXPECT_EQ(s2.current_pooled_bytes, 256u + 4096u);

    // Re-acquire those size classes — should be pool hits
    auto r1 = pool.acquire(64);    // hit from 256B free list
    auto r3 = pool.acquire(4096);  // hit from 4096B free list

    auto s3 = pool.stats();
    EXPECT_EQ(s3.total_allocations, 7u);
    EXPECT_EQ(s3.pool_hits, 2u);
    EXPECT_EQ(s3.current_pooled, 0u);
    EXPECT_EQ(s3.current_pooled_bytes, 0u);

    // Invariant must still hold
    EXPECT_EQ(s3.total_allocations,
              s3.pool_hits + s3.pool_misses + s3.oversized_allocations);
}
