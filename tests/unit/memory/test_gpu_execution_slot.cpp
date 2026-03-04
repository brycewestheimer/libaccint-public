// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2026 Bryce M. Westheimer. All rights reserved.

/// @file test_gpu_execution_slot.cpp
/// @brief Unit tests for GPU execution slot pool (concurrent GPU access)

#include <gtest/gtest.h>
#include <libaccint/config.hpp>

#if LIBACCINT_USE_CUDA

#include <libaccint/memory/gpu_execution_slot.hpp>
#include <cuda_runtime.h>

#include <thread>
#include <atomic>
#include <vector>
#include <chrono>

namespace libaccint::memory {

// ============================================================================
// Test Fixture
// ============================================================================

class GpuExecutionSlotTest : public ::testing::Test {
protected:
    void SetUp() override {
        int device_count = 0;
        cudaError_t err = cudaGetDeviceCount(&device_count);
        if (err != cudaSuccess || device_count == 0) {
            GTEST_SKIP() << "No CUDA devices available";
        }
    }
};

// ============================================================================
// GpuExecutionSlot Tests
// ============================================================================

TEST_F(GpuExecutionSlotTest, DefaultConstruction) {
    GpuExecutionSlot slot;
    EXPECT_TRUE(slot.stream.valid());
    EXPECT_EQ(slot.d_1e_output, nullptr);
    EXPECT_EQ(slot.d_2e_output, nullptr);
    EXPECT_EQ(slot.d_fused_1e_output, nullptr);
    EXPECT_EQ(slot.d_1e_capacity, 0u);
    EXPECT_EQ(slot.d_2e_capacity, 0u);
    EXPECT_EQ(slot.d_fused_capacity, 0u);
}

TEST_F(GpuExecutionSlotTest, Ensure1eBuffer) {
    GpuExecutionSlot slot;
    slot.ensure_1e_buffer(1024);
    EXPECT_NE(slot.d_1e_output, nullptr);
    EXPECT_GE(slot.d_1e_capacity, 1024u);

    // Second call with smaller size should not reallocate
    double* prev_ptr = slot.d_1e_output;
    size_t prev_cap = slot.d_1e_capacity;
    slot.ensure_1e_buffer(512);
    EXPECT_EQ(slot.d_1e_output, prev_ptr);
    EXPECT_EQ(slot.d_1e_capacity, prev_cap);
}

TEST_F(GpuExecutionSlotTest, Ensure2eBuffer) {
    GpuExecutionSlot slot;
    slot.ensure_2e_buffer(2048);
    EXPECT_NE(slot.d_2e_output, nullptr);
    EXPECT_GE(slot.d_2e_capacity, 2048u);
}

TEST_F(GpuExecutionSlotTest, EnsureFused1eBuffer) {
    GpuExecutionSlot slot;
    slot.ensure_fused_1e_buffer(1000);  // needs 3000 total
    EXPECT_NE(slot.d_fused_1e_output, nullptr);
    EXPECT_GE(slot.d_fused_capacity, 3000u);
}

TEST_F(GpuExecutionSlotTest, MoveConstruction) {
    GpuExecutionSlot original;
    original.ensure_2e_buffer(512);
    double* orig_ptr = original.d_2e_output;

    GpuExecutionSlot moved(std::move(original));
    EXPECT_EQ(moved.d_2e_output, orig_ptr);
    EXPECT_TRUE(moved.stream.valid());
    EXPECT_EQ(original.d_2e_output, nullptr);
}

// ============================================================================
// GpuSlotPool Tests
// ============================================================================

TEST_F(GpuExecutionSlotTest, PoolConstruction) {
    GpuSlotPool pool(3);
    EXPECT_EQ(pool.size(), 3u);
    EXPECT_EQ(pool.available(), 3u);
}

TEST_F(GpuExecutionSlotTest, PoolAcquireRelease) {
    GpuSlotPool pool(2);
    EXPECT_EQ(pool.available(), 2u);

    auto& slot1 = pool.acquire();
    EXPECT_EQ(pool.available(), 1u);
    EXPECT_TRUE(slot1.stream.valid());

    auto& slot2 = pool.acquire();
    EXPECT_EQ(pool.available(), 0u);

    pool.release(slot1);
    EXPECT_EQ(pool.available(), 1u);

    pool.release(slot2);
    EXPECT_EQ(pool.available(), 2u);
}

TEST_F(GpuExecutionSlotTest, ScopedGpuSlot) {
    GpuSlotPool pool(2);
    EXPECT_EQ(pool.available(), 2u);

    {
        ScopedGpuSlot scoped(pool);
        EXPECT_EQ(pool.available(), 1u);
        EXPECT_TRUE(scoped.slot().stream.valid());
        EXPECT_NE(scoped.stream(), nullptr);
    }

    EXPECT_EQ(pool.available(), 2u);
}

TEST_F(GpuExecutionSlotTest, PoolExhaustionAndBlocking) {
    GpuSlotPool pool(1);  // Only 1 slot

    auto& slot = pool.acquire();
    EXPECT_EQ(pool.available(), 0u);

    // Start a thread that tries to acquire (will block)
    std::atomic<bool> acquired{false};
    std::thread blocker([&pool, &acquired] {
        auto& s = pool.acquire();
        acquired = true;
        pool.release(s);
    });

    // Give the thread time to start and block
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
    EXPECT_FALSE(acquired);

    // Release the slot - should unblock the thread
    pool.release(slot);
    blocker.join();
    EXPECT_TRUE(acquired);
}

TEST_F(GpuExecutionSlotTest, ConcurrentSlotAccess) {
    constexpr int N_THREADS = 4;
    constexpr int N_ITERATIONS = 10;
    GpuSlotPool pool(N_THREADS);

    std::atomic<int> completed{0};
    std::vector<std::thread> threads;

    for (int t = 0; t < N_THREADS; ++t) {
        threads.emplace_back([&pool, &completed] {
            for (int i = 0; i < N_ITERATIONS; ++i) {
                ScopedGpuSlot scoped(pool);
                auto& slot = scoped.slot();

                // Each thread independently uses its slot's buffer
                slot.ensure_2e_buffer(256);
                EXPECT_NE(slot.d_2e_output, nullptr);

                // Synchronize the stream to ensure no errors
                slot.stream.synchronize();
            }
            ++completed;
        });
    }

    for (auto& t : threads) {
        t.join();
    }
    EXPECT_EQ(completed, N_THREADS);
    EXPECT_EQ(pool.available(), static_cast<size_t>(N_THREADS));
}

TEST_F(GpuExecutionSlotTest, ConcurrentWithMoreThreadsThanSlots) {
    constexpr int N_THREADS = 8;
    constexpr int N_SLOTS = 2;
    constexpr int N_ITERATIONS = 5;
    GpuSlotPool pool(N_SLOTS);

    std::atomic<int> completed{0};
    std::atomic<int> max_concurrent{0};
    std::atomic<int> current_concurrent{0};
    std::vector<std::thread> threads;

    for (int t = 0; t < N_THREADS; ++t) {
        threads.emplace_back([&] {
            for (int i = 0; i < N_ITERATIONS; ++i) {
                ScopedGpuSlot scoped(pool);

                int c = ++current_concurrent;
                // Track maximum concurrency
                int expected = max_concurrent.load();
                while (c > expected && !max_concurrent.compare_exchange_weak(expected, c)) {}

                // Simulate some GPU work
                scoped.slot().stream.synchronize();

                --current_concurrent;
            }
            ++completed;
        });
    }

    for (auto& t : threads) {
        t.join();
    }
    EXPECT_EQ(completed, N_THREADS);
    // Maximum concurrency should not exceed the number of slots
    EXPECT_LE(max_concurrent.load(), N_SLOTS);
}

TEST_F(GpuExecutionSlotTest, PoolSynchronizeAll) {
    GpuSlotPool pool(3);

    // Acquire and release to ensure streams are valid
    {
        ScopedGpuSlot s1(pool);
        ScopedGpuSlot s2(pool);
        s1.slot().ensure_1e_buffer(128);
        s2.slot().ensure_2e_buffer(128);
    }

    // Should not throw
    EXPECT_NO_THROW(pool.synchronize_all());
}

}  // namespace libaccint::memory

#else  // !LIBACCINT_USE_CUDA

// When CUDA is not available, just have a dummy test
TEST(GpuExecutionSlotTest, SkipWithoutCuda) {
    GTEST_SKIP() << "CUDA not enabled in this build";
}

#endif  // LIBACCINT_USE_CUDA
