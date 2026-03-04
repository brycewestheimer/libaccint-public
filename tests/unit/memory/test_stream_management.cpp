// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2026 Bryce M. Westheimer. All rights reserved.

/// @file test_stream_management.cpp
/// @brief Unit tests for CUDA stream management utilities

#include <gtest/gtest.h>
#include <libaccint/config.hpp>

#if LIBACCINT_USE_CUDA

#include <libaccint/memory/stream_management.hpp>
#include <libaccint/memory/device_memory.hpp>
#include <cuda_runtime.h>

#include <thread>
#include <atomic>
#include <vector>
#include <chrono>

namespace libaccint::memory {

// ============================================================================
// Test Fixture
// ============================================================================

class StreamManagementTest : public ::testing::Test {
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
// StreamHandle Tests
// ============================================================================

TEST_F(StreamManagementTest, StreamHandleConstruction) {
    StreamHandle stream;
    EXPECT_TRUE(stream.valid());
    EXPECT_NE(stream.get(), nullptr);
}

TEST_F(StreamManagementTest, StreamHandleConstructionWithFlags) {
    StreamHandle stream(cudaStreamDefault);
    EXPECT_TRUE(stream.valid());
}

TEST_F(StreamManagementTest, StreamHandleMoveConstruction) {
    StreamHandle original;
    cudaStream_t original_stream = original.get();
    ASSERT_TRUE(original.valid());

    StreamHandle moved(std::move(original));

    EXPECT_EQ(moved.get(), original_stream);
    EXPECT_TRUE(moved.valid());
    EXPECT_FALSE(original.valid());
    EXPECT_EQ(original.get(), nullptr);
}

TEST_F(StreamManagementTest, StreamHandleMoveAssignment) {
    StreamHandle stream1;
    StreamHandle stream2;
    cudaStream_t stream2_ptr = stream2.get();

    stream1 = std::move(stream2);

    EXPECT_EQ(stream1.get(), stream2_ptr);
    EXPECT_TRUE(stream1.valid());
    EXPECT_FALSE(stream2.valid());
}

TEST_F(StreamManagementTest, StreamHandleSelfMoveAssignment) {
    StreamHandle stream;
    cudaStream_t ptr = stream.get();

    stream = std::move(stream);

    EXPECT_EQ(stream.get(), ptr);
    EXPECT_TRUE(stream.valid());
}

TEST_F(StreamManagementTest, StreamHandleSynchronize) {
    StreamHandle stream;
    // Synchronize on an empty stream should succeed
    EXPECT_NO_THROW(stream.synchronize());
}

TEST_F(StreamManagementTest, StreamHandleQuery) {
    StreamHandle stream;
    // Empty stream should be complete
    EXPECT_TRUE(stream.query());
}

TEST_F(StreamManagementTest, StreamHandleImplicitConversion) {
    StreamHandle stream;
    cudaStream_t raw = stream;
    EXPECT_EQ(raw, stream.get());
}

TEST_F(StreamManagementTest, StreamHandleWithDeviceOps) {
    StreamHandle stream;

    constexpr size_t count = 100;
    DeviceBuffer<int> device(count);
    std::vector<int> host(count, 42);

    // Use stream for async operations
    device.upload(host.data(), count, stream.get());
    stream.synchronize();

    std::vector<int> result(count, 0);
    device.download(result.data(), count, stream.get());
    stream.synchronize();

    EXPECT_EQ(host, result);
}

// ============================================================================
// EventHandle Tests
// ============================================================================

TEST_F(StreamManagementTest, EventHandleConstruction) {
    EventHandle event;
    EXPECT_TRUE(event.valid());
    EXPECT_NE(event.get(), nullptr);
}

TEST_F(StreamManagementTest, EventHandleConstructionWithFlags) {
    EventHandle event(cudaEventDisableTiming);
    EXPECT_TRUE(event.valid());
}

TEST_F(StreamManagementTest, EventHandleMoveConstruction) {
    EventHandle original;
    cudaEvent_t original_event = original.get();

    EventHandle moved(std::move(original));

    EXPECT_EQ(moved.get(), original_event);
    EXPECT_TRUE(moved.valid());
    EXPECT_FALSE(original.valid());
}

TEST_F(StreamManagementTest, EventHandleMoveAssignment) {
    EventHandle event1;
    EventHandle event2;
    cudaEvent_t event2_ptr = event2.get();

    event1 = std::move(event2);

    EXPECT_EQ(event1.get(), event2_ptr);
    EXPECT_FALSE(event2.valid());
}

TEST_F(StreamManagementTest, EventHandleRecordAndSynchronize) {
    EventHandle event;
    StreamHandle stream;

    event.record(stream.get());
    EXPECT_NO_THROW(event.synchronize());
}

TEST_F(StreamManagementTest, EventHandleQuery) {
    EventHandle event;
    StreamHandle stream;

    event.record(stream.get());
    stream.synchronize();

    EXPECT_TRUE(event.query());
}

TEST_F(StreamManagementTest, EventHandleImplicitConversion) {
    EventHandle event;
    cudaEvent_t raw = event;
    EXPECT_EQ(raw, event.get());
}

TEST_F(StreamManagementTest, StreamWaitEvent) {
    StreamHandle stream1;
    StreamHandle stream2;
    EventHandle event;

    // Record event in stream1
    event.record(stream1.get());

    // Make stream2 wait for event
    EXPECT_NO_THROW(stream2.wait_event(event.get()));

    // Both streams should complete successfully
    EXPECT_NO_THROW(stream1.synchronize());
    EXPECT_NO_THROW(stream2.synchronize());
}

// ============================================================================
// EventTimer Tests
// ============================================================================

TEST_F(StreamManagementTest, EventTimerConstruction) {
    EXPECT_NO_THROW(EventTimer timer);
}

TEST_F(StreamManagementTest, EventTimerBasicTiming) {
    EventTimer timer;
    StreamHandle stream;

    timer.start(stream.get());

    // Do some work (small memory copy)
    DeviceBuffer<double> buffer(1000);
    std::vector<double> host(1000, 1.0);
    buffer.upload(host.data(), 1000, stream.get());

    timer.stop(stream.get());
    stream.synchronize();

    float elapsed = timer.elapsed_ms();
    EXPECT_GE(elapsed, 0.0f);  // Should be non-negative
}

TEST_F(StreamManagementTest, EventTimerDefaultStream) {
    EventTimer timer;

    timer.start();
    timer.stop();
    DeviceMemoryManager::synchronize();

    float elapsed = timer.elapsed_ms();
    EXPECT_GE(elapsed, 0.0f);
}

TEST_F(StreamManagementTest, EventTimerElapsedWithoutStartThrows) {
    EventTimer timer;
    timer.stop();

    EXPECT_THROW(timer.elapsed_ms(), CudaError);
}

TEST_F(StreamManagementTest, EventTimerElapsedWithoutStopThrows) {
    EventTimer timer;
    timer.start();

    EXPECT_THROW(timer.elapsed_ms(), CudaError);
}

TEST_F(StreamManagementTest, EventTimerReuse) {
    EventTimer timer;
    StreamHandle stream;

    // First timing
    timer.start(stream.get());
    timer.stop(stream.get());
    stream.synchronize();
    float elapsed1 = timer.elapsed_ms();
    EXPECT_GE(elapsed1, 0.0f);

    // Second timing
    timer.start(stream.get());
    timer.stop(stream.get());
    stream.synchronize();
    float elapsed2 = timer.elapsed_ms();
    EXPECT_GE(elapsed2, 0.0f);
}

// ============================================================================
// StreamPool Tests
// ============================================================================

TEST_F(StreamManagementTest, StreamPoolConstruction) {
    StreamPool pool(4);
    EXPECT_EQ(pool.size(), 4);
    EXPECT_EQ(pool.available(), 4);
}

TEST_F(StreamManagementTest, StreamPoolDefaultConstruction) {
    StreamPool pool;  // Default is 4 streams
    EXPECT_EQ(pool.size(), 4);
}

TEST_F(StreamManagementTest, StreamPoolAcquireRelease) {
    StreamPool pool(2);
    EXPECT_EQ(pool.available(), 2);

    StreamHandle& stream = pool.acquire();
    EXPECT_EQ(pool.available(), 1);
    EXPECT_TRUE(stream.valid());

    pool.release(stream);
    EXPECT_EQ(pool.available(), 2);
}

TEST_F(StreamManagementTest, StreamPoolAcquireMultiple) {
    StreamPool pool(3);

    StreamHandle& stream1 = pool.acquire();
    StreamHandle& stream2 = pool.acquire();
    StreamHandle& stream3 = pool.acquire();

    EXPECT_EQ(pool.available(), 0);

    // All streams should be distinct
    EXPECT_NE(stream1.get(), stream2.get());
    EXPECT_NE(stream2.get(), stream3.get());
    EXPECT_NE(stream1.get(), stream3.get());

    pool.release(stream1);
    pool.release(stream2);
    pool.release(stream3);

    EXPECT_EQ(pool.available(), 3);
}

TEST_F(StreamManagementTest, StreamPoolSynchronizeAll) {
    StreamPool pool(2);

    StreamHandle& stream1 = pool.acquire();
    StreamHandle& stream2 = pool.acquire();

    // Do some work on both streams
    DeviceBuffer<int> buffer1(100);
    DeviceBuffer<int> buffer2(100);
    std::vector<int> host(100, 1);
    buffer1.upload(host.data(), 100, stream1.get());
    buffer2.upload(host.data(), 100, stream2.get());

    pool.release(stream1);
    pool.release(stream2);

    // Synchronize all should not throw
    EXPECT_NO_THROW(pool.synchronize_all());
}

TEST_F(StreamManagementTest, StreamPoolConcurrentAccess) {
    StreamPool pool(4);
    std::atomic<int> acquired_count{0};
    std::atomic<int> released_count{0};

    auto worker = [&pool, &acquired_count, &released_count]() {
        for (int i = 0; i < 10; ++i) {
            StreamHandle& stream = pool.acquire();
            acquired_count++;

            // Simulate some work
            std::this_thread::sleep_for(std::chrono::microseconds(100));

            pool.release(stream);
            released_count++;
        }
    };

    std::vector<std::thread> threads;
    for (int i = 0; i < 4; ++i) {
        threads.emplace_back(worker);
    }

    for (auto& t : threads) {
        t.join();
    }

    EXPECT_EQ(acquired_count.load(), 40);
    EXPECT_EQ(released_count.load(), 40);
    EXPECT_EQ(pool.available(), 4);
}

// ============================================================================
// ScopedStream Tests
// ============================================================================

TEST_F(StreamManagementTest, ScopedStreamConstruction) {
    StreamPool pool(2);
    EXPECT_EQ(pool.available(), 2);

    {
        ScopedStream scoped(pool);
        EXPECT_EQ(pool.available(), 1);
        EXPECT_NE(scoped.get(), nullptr);
    }

    EXPECT_EQ(pool.available(), 2);
}

TEST_F(StreamManagementTest, ScopedStreamImplicitConversion) {
    StreamPool pool;
    ScopedStream scoped(pool);

    cudaStream_t raw = scoped;
    EXPECT_EQ(raw, scoped.get());
}

TEST_F(StreamManagementTest, ScopedStreamHandle) {
    StreamPool pool;
    ScopedStream scoped(pool);

    StreamHandle& handle = scoped.handle();
    EXPECT_TRUE(handle.valid());
    EXPECT_EQ(handle.get(), scoped.get());
}

TEST_F(StreamManagementTest, ScopedStreamWithDeviceOps) {
    StreamPool pool;

    constexpr size_t count = 100;
    std::vector<int> result(count, 0);

    {
        ScopedStream scoped(pool);

        DeviceBuffer<int> buffer(count);
        std::vector<int> host(count, 99);

        buffer.upload(host.data(), count, scoped.get());
        buffer.download(result.data(), count, scoped.get());
        scoped.handle().synchronize();
    }

    for (size_t i = 0; i < count; ++i) {
        EXPECT_EQ(result[i], 99);
    }
}

TEST_F(StreamManagementTest, ScopedStreamMultipleInScope) {
    StreamPool pool(4);

    {
        ScopedStream scoped1(pool);
        ScopedStream scoped2(pool);
        ScopedStream scoped3(pool);

        EXPECT_EQ(pool.available(), 1);

        // All streams should be distinct
        EXPECT_NE(scoped1.get(), scoped2.get());
        EXPECT_NE(scoped2.get(), scoped3.get());
    }

    EXPECT_EQ(pool.available(), 4);
}

// ============================================================================
// Integration Tests
// ============================================================================

TEST_F(StreamManagementTest, ConcurrentStreamsWithEvents) {
    StreamPool pool(2);

    constexpr size_t count = 1000;
    DeviceBuffer<double> buffer1(count);
    DeviceBuffer<double> buffer2(count);
    std::vector<double> host(count, 3.14);

    EventHandle event;

    {
        ScopedStream stream1(pool);
        ScopedStream stream2(pool);

        // Upload on stream1
        buffer1.upload(host.data(), count, stream1.get());
        event.record(stream1.get());

        // Make stream2 wait for stream1's upload
        stream2.handle().wait_event(event.get());

        // Copy from buffer1 to buffer2 on stream2
        DeviceMemoryManager::copy_device_to_device(
            buffer2.data(), buffer1.data(), count, stream2.get());

        stream2.handle().synchronize();
    }

    // Verify data in buffer2
    std::vector<double> result(count, 0.0);
    buffer2.download(result.data(), count);
    DeviceMemoryManager::synchronize();

    for (size_t i = 0; i < count; ++i) {
        EXPECT_DOUBLE_EQ(result[i], 3.14);
    }
}

TEST_F(StreamManagementTest, TimedConcurrentOperations) {
    StreamPool pool(2);
    EventTimer timer1, timer2;

    constexpr size_t count = 10000;
    DeviceBuffer<float> buffer1(count);
    DeviceBuffer<float> buffer2(count);
    std::vector<float> host(count, 1.0f);

    {
        ScopedStream stream1(pool);
        ScopedStream stream2(pool);

        timer1.start(stream1.get());
        buffer1.upload(host.data(), count, stream1.get());
        timer1.stop(stream1.get());

        timer2.start(stream2.get());
        buffer2.upload(host.data(), count, stream2.get());
        timer2.stop(stream2.get());

        stream1.handle().synchronize();
        stream2.handle().synchronize();
    }

    float elapsed1 = timer1.elapsed_ms();
    float elapsed2 = timer2.elapsed_ms();

    // Both should have completed in non-negative time
    EXPECT_GE(elapsed1, 0.0f);
    EXPECT_GE(elapsed2, 0.0f);
}

}  // namespace libaccint::memory

#else  // LIBACCINT_USE_CUDA

TEST(StreamManagementTest, CudaNotEnabled) {
    GTEST_SKIP() << "CUDA support not enabled";
}

#endif  // LIBACCINT_USE_CUDA
