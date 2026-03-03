// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2026 Bryce M. Westheimer. All rights reserved.

/// @file test_device_memory.cpp
/// @brief Unit tests for CUDA device memory management

#include <gtest/gtest.h>
#include <libaccint/config.hpp>

#if LIBACCINT_USE_CUDA

#include <libaccint/memory/device_memory.hpp>
#include <cuda_runtime.h>

#include <algorithm>
#include <numeric>
#include <vector>

namespace libaccint::memory {

// ============================================================================
// Test Fixture
// ============================================================================

class DeviceMemoryTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Ensure a CUDA device is available
        int device_count = 0;
        cudaError_t err = cudaGetDeviceCount(&device_count);
        if (err != cudaSuccess || device_count == 0) {
            GTEST_SKIP() << "No CUDA devices available";
        }
    }
};

// ============================================================================
// DeviceMemoryManager Tests
// ============================================================================

TEST_F(DeviceMemoryTest, AllocateAndDeallocateDevice) {
    // Allocate device memory
    constexpr size_t count = 1000;
    double* ptr = DeviceMemoryManager::allocate_device<double>(count);
    ASSERT_NE(ptr, nullptr);

    // Deallocate
    EXPECT_NO_THROW(DeviceMemoryManager::deallocate_device(ptr));
}

TEST_F(DeviceMemoryTest, AllocateZeroElements) {
    // Allocating zero elements should return nullptr
    double* ptr = DeviceMemoryManager::allocate_device<double>(0);
    EXPECT_EQ(ptr, nullptr);
    // Deallocating nullptr should be safe
    EXPECT_NO_THROW(DeviceMemoryManager::deallocate_device(ptr));
}

TEST_F(DeviceMemoryTest, AllocateAndDeallocatePinned) {
    // Allocate pinned memory
    constexpr size_t count = 1000;
    double* ptr = DeviceMemoryManager::allocate_pinned<double>(count);
    ASSERT_NE(ptr, nullptr);

    // Verify we can read/write to pinned memory
    ptr[0] = 42.0;
    ptr[count - 1] = 99.0;
    EXPECT_DOUBLE_EQ(ptr[0], 42.0);
    EXPECT_DOUBLE_EQ(ptr[count - 1], 99.0);

    // Deallocate
    EXPECT_NO_THROW(DeviceMemoryManager::deallocate_pinned(ptr));
}

TEST_F(DeviceMemoryTest, AllocateZeroPinnedElements) {
    float* ptr = DeviceMemoryManager::allocate_pinned<float>(0);
    EXPECT_EQ(ptr, nullptr);
    EXPECT_NO_THROW(DeviceMemoryManager::deallocate_pinned(ptr));
}

TEST_F(DeviceMemoryTest, AsyncCopyHostToDevice) {
    constexpr size_t count = 100;
    std::vector<int> host_data(count);
    std::iota(host_data.begin(), host_data.end(), 0);  // 0, 1, 2, ...

    int* device_ptr = DeviceMemoryManager::allocate_device<int>(count);
    ASSERT_NE(device_ptr, nullptr);

    // Copy to device
    EXPECT_NO_THROW(
        DeviceMemoryManager::copy_to_device(device_ptr, host_data.data(), count));

    // Sync to ensure copy completes
    DeviceMemoryManager::synchronize();

    // Verify by copying back
    std::vector<int> result(count, -1);
    DeviceMemoryManager::copy_to_host(result.data(), device_ptr, count);
    DeviceMemoryManager::synchronize();

    EXPECT_EQ(host_data, result);

    DeviceMemoryManager::deallocate_device(device_ptr);
}

TEST_F(DeviceMemoryTest, AsyncCopyDeviceToHost) {
    constexpr size_t count = 50;

    // Create and initialize device memory via host
    int* device_ptr = DeviceMemoryManager::allocate_device<int>(count);
    ASSERT_NE(device_ptr, nullptr);

    std::vector<int> source(count);
    std::iota(source.begin(), source.end(), 100);  // 100, 101, 102, ...
    DeviceMemoryManager::copy_to_device(device_ptr, source.data(), count);
    DeviceMemoryManager::synchronize();

    // Copy back to host
    std::vector<int> dest(count, 0);
    EXPECT_NO_THROW(
        DeviceMemoryManager::copy_to_host(dest.data(), device_ptr, count));
    DeviceMemoryManager::synchronize();

    EXPECT_EQ(source, dest);

    DeviceMemoryManager::deallocate_device(device_ptr);
}

TEST_F(DeviceMemoryTest, AsyncCopyDeviceToDevice) {
    constexpr size_t count = 75;

    // Create source device buffer
    int* src_ptr = DeviceMemoryManager::allocate_device<int>(count);
    int* dst_ptr = DeviceMemoryManager::allocate_device<int>(count);
    ASSERT_NE(src_ptr, nullptr);
    ASSERT_NE(dst_ptr, nullptr);

    // Initialize source
    std::vector<int> host_data(count);
    std::iota(host_data.begin(), host_data.end(), 200);
    DeviceMemoryManager::copy_to_device(src_ptr, host_data.data(), count);
    DeviceMemoryManager::synchronize();

    // Copy device to device
    EXPECT_NO_THROW(
        DeviceMemoryManager::copy_device_to_device(dst_ptr, src_ptr, count));
    DeviceMemoryManager::synchronize();

    // Verify by copying back
    std::vector<int> result(count, 0);
    DeviceMemoryManager::copy_to_host(result.data(), dst_ptr, count);
    DeviceMemoryManager::synchronize();

    EXPECT_EQ(host_data, result);

    DeviceMemoryManager::deallocate_device(src_ptr);
    DeviceMemoryManager::deallocate_device(dst_ptr);
}

TEST_F(DeviceMemoryTest, CopyZeroElements) {
    int* device_ptr = DeviceMemoryManager::allocate_device<int>(10);
    ASSERT_NE(device_ptr, nullptr);

    int host_val = 42;
    // Copying zero elements should be a no-op
    EXPECT_NO_THROW(DeviceMemoryManager::copy_to_device(device_ptr, &host_val, 0));
    EXPECT_NO_THROW(DeviceMemoryManager::copy_to_host(&host_val, device_ptr, 0));
    EXPECT_NO_THROW(DeviceMemoryManager::copy_device_to_device(device_ptr, device_ptr, 0));

    DeviceMemoryManager::deallocate_device(device_ptr);
}

TEST_F(DeviceMemoryTest, CopyWithStream) {
    constexpr size_t count = 100;

    cudaStream_t stream;
    LIBACCINT_CUDA_CHECK(cudaStreamCreate(&stream));

    int* device_ptr = DeviceMemoryManager::allocate_device<int>(count);
    std::vector<int> host_data(count, 123);

    // Copy with explicit stream
    DeviceMemoryManager::copy_to_device(device_ptr, host_data.data(), count, stream);
    DeviceMemoryManager::synchronize_stream(stream);

    std::vector<int> result(count, 0);
    DeviceMemoryManager::copy_to_host(result.data(), device_ptr, count, stream);
    DeviceMemoryManager::synchronize_stream(stream);

    EXPECT_EQ(host_data, result);

    DeviceMemoryManager::deallocate_device(device_ptr);
    cudaStreamDestroy(stream);
}

// ============================================================================
// DeviceBuffer Tests
// ============================================================================

TEST_F(DeviceMemoryTest, DeviceBufferDefaultConstruction) {
    DeviceBuffer<double> buffer;
    EXPECT_EQ(buffer.data(), nullptr);
    EXPECT_EQ(buffer.size(), 0);
    EXPECT_TRUE(buffer.empty());
}

TEST_F(DeviceMemoryTest, DeviceBufferAllocation) {
    constexpr size_t count = 500;
    DeviceBuffer<float> buffer(count);
    EXPECT_NE(buffer.data(), nullptr);
    EXPECT_EQ(buffer.size(), count);
    EXPECT_FALSE(buffer.empty());
    EXPECT_EQ(buffer.size_bytes(), count * sizeof(float));
}

TEST_F(DeviceMemoryTest, DeviceBufferRAII) {
    // This test verifies that RAII works by scope
    {
        DeviceBuffer<int> buffer(100);
        EXPECT_NE(buffer.data(), nullptr);
        // Buffer should be freed when it goes out of scope
    }
    // If we get here without crash, RAII is working
    SUCCEED();
}

TEST_F(DeviceMemoryTest, DeviceBufferMoveConstruction) {
    constexpr size_t count = 200;
    DeviceBuffer<double> original(count);
    double* original_ptr = original.data();
    ASSERT_NE(original_ptr, nullptr);

    // Move construct
    DeviceBuffer<double> moved(std::move(original));

    // Check moved-to buffer
    EXPECT_EQ(moved.data(), original_ptr);
    EXPECT_EQ(moved.size(), count);

    // Check moved-from buffer is empty
    EXPECT_EQ(original.data(), nullptr);
    EXPECT_EQ(original.size(), 0);
    EXPECT_TRUE(original.empty());
}

TEST_F(DeviceMemoryTest, DeviceBufferMoveAssignment) {
    constexpr size_t count1 = 100;
    constexpr size_t count2 = 200;

    DeviceBuffer<int> buffer1(count1);
    DeviceBuffer<int> buffer2(count2);
    int* ptr2 = buffer2.data();

    // Move assign
    buffer1 = std::move(buffer2);

    EXPECT_EQ(buffer1.data(), ptr2);
    EXPECT_EQ(buffer1.size(), count2);

    EXPECT_EQ(buffer2.data(), nullptr);
    EXPECT_EQ(buffer2.size(), 0);
}

TEST_F(DeviceMemoryTest, DeviceBufferSelfMoveAssignment) {
    constexpr size_t count = 100;
    DeviceBuffer<float> buffer(count);
    float* ptr = buffer.data();

    // Self-move (should be safe)
    buffer = std::move(buffer);

    // Buffer should be unchanged
    EXPECT_EQ(buffer.data(), ptr);
    EXPECT_EQ(buffer.size(), count);
}

TEST_F(DeviceMemoryTest, DeviceBufferUploadDownload) {
    constexpr size_t count = 256;
    std::vector<double> host_data(count);
    for (size_t i = 0; i < count; ++i) {
        host_data[i] = static_cast<double>(i) * 1.5;
    }

    DeviceBuffer<double> buffer(count);

    // Upload
    buffer.upload(host_data.data(), count);
    DeviceMemoryManager::synchronize();

    // Download to new vector
    std::vector<double> result(count, 0.0);
    buffer.download(result.data(), count);
    DeviceMemoryManager::synchronize();

    EXPECT_EQ(host_data, result);
}

TEST_F(DeviceMemoryTest, DeviceBufferRelease) {
    constexpr size_t count = 50;
    DeviceBuffer<int> buffer(count);
    int* ptr = buffer.data();

    int* released = buffer.release();

    EXPECT_EQ(released, ptr);
    EXPECT_EQ(buffer.data(), nullptr);
    EXPECT_EQ(buffer.size(), 0);
    EXPECT_TRUE(buffer.empty());

    // Clean up manually since we released ownership
    DeviceMemoryManager::deallocate_device(released);
}

TEST_F(DeviceMemoryTest, DeviceBufferReset) {
    DeviceBuffer<float> buffer(100);
    EXPECT_FALSE(buffer.empty());

    // Reset to empty
    buffer.reset();
    EXPECT_TRUE(buffer.empty());
    EXPECT_EQ(buffer.data(), nullptr);

    // Reset with new size
    buffer.reset(50);
    EXPECT_FALSE(buffer.empty());
    EXPECT_EQ(buffer.size(), 50);
    EXPECT_NE(buffer.data(), nullptr);
}

// ============================================================================
// PinnedBuffer Tests
// ============================================================================

TEST_F(DeviceMemoryTest, PinnedBufferDefaultConstruction) {
    PinnedBuffer<double> buffer;
    EXPECT_EQ(buffer.data(), nullptr);
    EXPECT_EQ(buffer.size(), 0);
    EXPECT_TRUE(buffer.empty());
}

TEST_F(DeviceMemoryTest, PinnedBufferAllocation) {
    constexpr size_t count = 300;
    PinnedBuffer<int> buffer(count);
    EXPECT_NE(buffer.data(), nullptr);
    EXPECT_EQ(buffer.size(), count);
    EXPECT_FALSE(buffer.empty());
    EXPECT_EQ(buffer.size_bytes(), count * sizeof(int));
}

TEST_F(DeviceMemoryTest, PinnedBufferReadWrite) {
    constexpr size_t count = 100;
    PinnedBuffer<double> buffer(count);

    // Write to pinned memory
    for (size_t i = 0; i < count; ++i) {
        buffer.data()[i] = static_cast<double>(i);
    }

    // Read back
    for (size_t i = 0; i < count; ++i) {
        EXPECT_DOUBLE_EQ(buffer.data()[i], static_cast<double>(i));
    }
}

TEST_F(DeviceMemoryTest, PinnedBufferRAII) {
    {
        PinnedBuffer<float> buffer(200);
        EXPECT_NE(buffer.data(), nullptr);
    }
    SUCCEED();  // No crash means RAII is working
}

TEST_F(DeviceMemoryTest, PinnedBufferMoveConstruction) {
    constexpr size_t count = 150;
    PinnedBuffer<int> original(count);
    int* original_ptr = original.data();
    original.data()[0] = 42;

    PinnedBuffer<int> moved(std::move(original));

    EXPECT_EQ(moved.data(), original_ptr);
    EXPECT_EQ(moved.size(), count);
    EXPECT_EQ(moved.data()[0], 42);

    EXPECT_EQ(original.data(), nullptr);
    EXPECT_EQ(original.size(), 0);
}

TEST_F(DeviceMemoryTest, PinnedBufferMoveAssignment) {
    PinnedBuffer<double> buffer1(100);
    PinnedBuffer<double> buffer2(200);
    double* ptr2 = buffer2.data();
    buffer2.data()[50] = 3.14;

    buffer1 = std::move(buffer2);

    EXPECT_EQ(buffer1.data(), ptr2);
    EXPECT_EQ(buffer1.size(), 200);
    EXPECT_DOUBLE_EQ(buffer1.data()[50], 3.14);

    EXPECT_EQ(buffer2.data(), nullptr);
    EXPECT_EQ(buffer2.size(), 0);
}

TEST_F(DeviceMemoryTest, PinnedBufferUploadToDevice) {
    constexpr size_t count = 100;

    // Create and fill pinned buffer
    PinnedBuffer<int> pinned(count);
    for (size_t i = 0; i < count; ++i) {
        pinned.data()[i] = static_cast<int>(i * 2);
    }

    // Allocate device buffer and upload from pinned
    DeviceBuffer<int> device(count);
    pinned.upload_to_device(device.data(), count);
    DeviceMemoryManager::synchronize();

    // Download and verify
    std::vector<int> result(count, 0);
    device.download(result.data(), count);
    DeviceMemoryManager::synchronize();

    for (size_t i = 0; i < count; ++i) {
        EXPECT_EQ(result[i], static_cast<int>(i * 2));
    }
}

TEST_F(DeviceMemoryTest, PinnedBufferDownloadFromDevice) {
    constexpr size_t count = 80;

    // Create device buffer with data
    std::vector<float> source(count);
    for (size_t i = 0; i < count; ++i) {
        source[i] = static_cast<float>(i) * 0.5f;
    }
    DeviceBuffer<float> device(count);
    device.upload(source.data(), count);
    DeviceMemoryManager::synchronize();

    // Download to pinned buffer
    PinnedBuffer<float> pinned(count);
    pinned.download_from_device(device.data(), count);
    DeviceMemoryManager::synchronize();

    // Verify
    for (size_t i = 0; i < count; ++i) {
        EXPECT_FLOAT_EQ(pinned.data()[i], static_cast<float>(i) * 0.5f);
    }
}

TEST_F(DeviceMemoryTest, PinnedBufferRelease) {
    PinnedBuffer<int> buffer(50);
    int* ptr = buffer.data();
    buffer.data()[10] = 999;

    int* released = buffer.release();

    EXPECT_EQ(released, ptr);
    EXPECT_EQ(released[10], 999);
    EXPECT_EQ(buffer.data(), nullptr);
    EXPECT_TRUE(buffer.empty());

    DeviceMemoryManager::deallocate_pinned(released);
}

TEST_F(DeviceMemoryTest, PinnedBufferReset) {
    PinnedBuffer<double> buffer(100);
    buffer.data()[0] = 42.0;
    EXPECT_FALSE(buffer.empty());

    buffer.reset();
    EXPECT_TRUE(buffer.empty());
    EXPECT_EQ(buffer.data(), nullptr);

    buffer.reset(75);
    EXPECT_FALSE(buffer.empty());
    EXPECT_EQ(buffer.size(), 75);
}

// ============================================================================
// Task 2.3.5: Additional Device Memory Tests
// ============================================================================

TEST_F(DeviceMemoryTest, DeviceBufferZeroSizeConstruction) {
    DeviceBuffer<double> buffer(0);
    EXPECT_EQ(buffer.data(), nullptr);
    EXPECT_EQ(buffer.size(), 0);
    EXPECT_TRUE(buffer.empty());
    // Destructor should be safe
}

TEST_F(DeviceMemoryTest, PinnedBufferZeroSizeConstruction) {
    PinnedBuffer<float> buffer(0);
    EXPECT_EQ(buffer.data(), nullptr);
    EXPECT_EQ(buffer.size(), 0);
    EXPECT_TRUE(buffer.empty());
}

TEST_F(DeviceMemoryTest, PinnedBufferSelfMoveAssignment) {
    PinnedBuffer<int> buffer(100);
    int* ptr = buffer.data();

    buffer = std::move(buffer);

    EXPECT_EQ(buffer.data(), ptr);
    EXPECT_EQ(buffer.size(), 100);
}

TEST_F(DeviceMemoryTest, DeviceBufferSizeBytes) {
    constexpr size_t count = 42;
    DeviceBuffer<double> buffer(count);
    EXPECT_EQ(buffer.size_bytes(), count * sizeof(double));
}

TEST_F(DeviceMemoryTest, PinnedBufferSizeBytes) {
    constexpr size_t count = 37;
    PinnedBuffer<int> buffer(count);
    EXPECT_EQ(buffer.size_bytes(), count * sizeof(int));
}

// ============================================================================
// Round-Trip Tests
// ============================================================================

TEST_F(DeviceMemoryTest, RoundTripDouble) {
    constexpr size_t count = 1000;
    std::vector<double> original(count);
    for (size_t i = 0; i < count; ++i) {
        original[i] = static_cast<double>(i) * 0.001 - 0.5;
    }

    // Upload to device
    DeviceBuffer<double> device(count);
    device.upload(original.data(), count);
    DeviceMemoryManager::synchronize();

    // Download back
    std::vector<double> result(count, 0.0);
    device.download(result.data(), count);
    DeviceMemoryManager::synchronize();

    // Verify exact match
    EXPECT_EQ(original, result);
}

TEST_F(DeviceMemoryTest, RoundTripWithPinned) {
    constexpr size_t count = 512;

    // Create pinned source buffer
    PinnedBuffer<float> pinned_src(count);
    for (size_t i = 0; i < count; ++i) {
        pinned_src.data()[i] = static_cast<float>(count - i);
    }

    // Upload to device
    DeviceBuffer<float> device(count);
    pinned_src.upload_to_device(device.data(), count);
    DeviceMemoryManager::synchronize();

    // Download to pinned dest buffer
    PinnedBuffer<float> pinned_dst(count);
    pinned_dst.download_from_device(device.data(), count);
    DeviceMemoryManager::synchronize();

    // Verify
    for (size_t i = 0; i < count; ++i) {
        EXPECT_FLOAT_EQ(pinned_dst.data()[i], pinned_src.data()[i]);
    }
}

TEST_F(DeviceMemoryTest, RoundTripThroughDeviceCopy) {
    constexpr size_t count = 200;

    std::vector<int> original(count);
    std::iota(original.begin(), original.end(), 1000);

    // Upload to first device buffer
    DeviceBuffer<int> device1(count);
    device1.upload(original.data(), count);
    DeviceMemoryManager::synchronize();

    // Copy to second device buffer
    DeviceBuffer<int> device2(count);
    DeviceMemoryManager::copy_device_to_device(device2.data(), device1.data(), count);
    DeviceMemoryManager::synchronize();

    // Download from second buffer
    std::vector<int> result(count, 0);
    device2.download(result.data(), count);
    DeviceMemoryManager::synchronize();

    EXPECT_EQ(original, result);
}

// ============================================================================
// Error Handling Tests
// ============================================================================

TEST_F(DeviceMemoryTest, CudaErrorMessage) {
    // Test that CudaError includes proper information
    try {
        throw CudaError("test error message", "test_file.cpp", 42);
    } catch (const CudaError& e) {
        std::string what = e.what();
        EXPECT_TRUE(what.find("test error message") != std::string::npos);
        EXPECT_TRUE(what.find("test_file.cpp") != std::string::npos);
        EXPECT_TRUE(what.find("42") != std::string::npos);
        EXPECT_STREQ(e.file(), "test_file.cpp");
        EXPECT_EQ(e.line(), 42);
    }
}

TEST_F(DeviceMemoryTest, ImpossiblyLargeAllocationThrows) {
    // Try to allocate an impossibly large amount of memory
    // Use 1TB (2^40 bytes) which will definitely fail on any current GPU
    constexpr size_t impossible_bytes = static_cast<size_t>(1) << 40;
    constexpr size_t impossible_count = impossible_bytes / sizeof(double);

    EXPECT_THROW(
        DeviceMemoryManager::allocate_device<double>(impossible_count),
        CudaError);
}

TEST_F(DeviceMemoryTest, DeviceBufferImpossibleAllocationThrows) {
    // Use 1TB worth of floats
    constexpr size_t impossible_bytes = static_cast<size_t>(1) << 40;
    constexpr size_t impossible_count = impossible_bytes / sizeof(float);

    EXPECT_THROW(DeviceBuffer<float>(impossible_count), CudaError);
}

TEST_F(DeviceMemoryTest, PinnedBufferImpossibleAllocationThrows) {
    // Use 1TB worth of ints
    constexpr size_t impossible_bytes = static_cast<size_t>(1) << 40;
    constexpr size_t impossible_count = impossible_bytes / sizeof(int);

    EXPECT_THROW(PinnedBuffer<int>(impossible_count), CudaError);
}

// ============================================================================
// Debug Build Statistics Tests (only run in debug builds)
// ============================================================================

#ifndef NDEBUG
TEST_F(DeviceMemoryTest, AllocationTrackingDevice) {
    size_t initial = DeviceMemoryManager::active_device_allocations();

    {
        DeviceBuffer<double> buffer1(100);
        EXPECT_EQ(DeviceMemoryManager::active_device_allocations(), initial + 1);

        DeviceBuffer<int> buffer2(50);
        EXPECT_EQ(DeviceMemoryManager::active_device_allocations(), initial + 2);
    }

    EXPECT_EQ(DeviceMemoryManager::active_device_allocations(), initial);
}

TEST_F(DeviceMemoryTest, AllocationTrackingPinned) {
    size_t initial = DeviceMemoryManager::active_pinned_allocations();

    {
        PinnedBuffer<float> buffer1(100);
        EXPECT_EQ(DeviceMemoryManager::active_pinned_allocations(), initial + 1);

        PinnedBuffer<double> buffer2(50);
        EXPECT_EQ(DeviceMemoryManager::active_pinned_allocations(), initial + 2);
    }

    EXPECT_EQ(DeviceMemoryManager::active_pinned_allocations(), initial);
}
#endif  // NDEBUG

}  // namespace libaccint::memory

#else  // LIBACCINT_USE_CUDA

// Dummy test when CUDA is not enabled
TEST(DeviceMemoryTest, CudaNotEnabled) {
    GTEST_SKIP() << "CUDA support not enabled";
}

#endif  // LIBACCINT_USE_CUDA
