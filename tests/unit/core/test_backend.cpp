// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2026 Bryce M. Westheimer. All rights reserved.

/// @file test_backend.cpp
/// @brief Unit tests for backend types and device abstraction

#include <gtest/gtest.h>
#include <libaccint/core/backend.hpp>

namespace libaccint {

// ============================================================================
// Test BackendType enum
// ============================================================================

TEST(BackendTypeTest, EnumValues) {
    // Verify enum values exist and have correct underlying values
    EXPECT_EQ(static_cast<int>(BackendType::CPU), 0);
    EXPECT_EQ(static_cast<int>(BackendType::CUDA), 1);
}

// ============================================================================
// Test backend_name function
// ============================================================================

TEST(BackendNameTest, CPUBackend) {
    EXPECT_EQ(backend_name(BackendType::CPU), "CPU");
}

TEST(BackendNameTest, CUDABackend) {
    EXPECT_EQ(backend_name(BackendType::CUDA), "CUDA");
}

TEST(BackendNameTest, StringViewProperties) {
    // Verify the returned string_view has correct properties
    auto cpu_name = backend_name(BackendType::CPU);
    EXPECT_EQ(cpu_name.size(), 3);
    EXPECT_TRUE(cpu_name == "CPU");

    auto cuda_name = backend_name(BackendType::CUDA);
    EXPECT_EQ(cuda_name.size(), 4);
    EXPECT_TRUE(cuda_name == "CUDA");
}

// ============================================================================
// Test is_gpu_backend function
// ============================================================================

TEST(IsGPUBackendTest, CPUIsNotGPU) {
    EXPECT_FALSE(is_gpu_backend(BackendType::CPU));
}

TEST(IsGPUBackendTest, CUDAIsGPU) {
    EXPECT_TRUE(is_gpu_backend(BackendType::CUDA));
}

// ============================================================================
// Test is_backend_available function
// ============================================================================

TEST(IsBackendAvailableTest, CPUAlwaysAvailable) {
    // CPU backend should always be available
    EXPECT_TRUE(is_backend_available(BackendType::CPU));
}

TEST(IsBackendAvailableTest, CUDAAvailability) {
    // CUDA availability must be consistent with has_cuda_backend()
    bool cuda_available = is_backend_available(BackendType::CUDA);
    bool has_cuda = has_cuda_backend();
    EXPECT_EQ(cuda_available, has_cuda);

    // If CUDA is not available, it must not be reported as available
    if (!has_cuda) {
        EXPECT_FALSE(cuda_available);
    }
}

// ============================================================================
// Test StreamHandle default construction
// ============================================================================

TEST(StreamHandleTest, DefaultConstruction) {
    // Default-constructed stream should be invalid (no impl)
    StreamHandle stream;
    EXPECT_FALSE(stream.valid());
    EXPECT_EQ(stream.backend(), BackendType::CPU);
}

TEST(StreamHandleTest, BackendQuery) {
    StreamHandle stream;
    EXPECT_EQ(stream.backend(), BackendType::CPU);
}

TEST(StreamHandleTest, CreateCPU) {
    // StreamHandle::create(CPU) produces a valid handle
    StreamHandle stream = StreamHandle::create(BackendType::CPU);
    EXPECT_TRUE(stream.valid());
    EXPECT_EQ(stream.backend(), BackendType::CPU);
}

TEST(StreamHandleTest, SynchronizeCPU) {
    // synchronize() on a CPU handle completes without error
    StreamHandle stream = StreamHandle::create(BackendType::CPU);
    EXPECT_NO_THROW(stream.synchronize());
}

TEST(StreamHandleTest, SynchronizeDefaultHandle) {
    // synchronize() on a default-constructed handle should be safe (no-op)
    StreamHandle stream;
    EXPECT_NO_THROW(stream.synchronize());
}

TEST(StreamHandleTest, CopySemantics) {
    StreamHandle original = StreamHandle::create(BackendType::CPU);
    EXPECT_TRUE(original.valid());

    // Copy the handle
    StreamHandle copy = original;
    EXPECT_TRUE(copy.valid());
    EXPECT_TRUE(original.valid());
    EXPECT_EQ(copy.backend(), BackendType::CPU);
    EXPECT_EQ(original.backend(), BackendType::CPU);
}

TEST(StreamHandleTest, MoveSemantics) {
    StreamHandle original = StreamHandle::create(BackendType::CPU);
    EXPECT_TRUE(original.valid());

    // Move the handle
    StreamHandle moved = std::move(original);
    EXPECT_TRUE(moved.valid());
    EXPECT_EQ(moved.backend(), BackendType::CPU);
    // After move, original's shared_ptr is null
    EXPECT_FALSE(original.valid());
}

TEST(BackendAvailabilityTest, Consistency) {
    // If has_cuda_backend() is false, is_backend_available(CUDA) must be false
    if (!has_cuda_backend()) {
        EXPECT_FALSE(is_backend_available(BackendType::CUDA));
    }
    // GPU backend check should be consistent
    EXPECT_TRUE(is_gpu_backend(BackendType::CUDA));
    EXPECT_FALSE(is_gpu_backend(BackendType::CPU));
}

// ============================================================================
// Test BackendError exception
// ============================================================================

TEST(BackendErrorTest, ConstructionAndMessage) {
    BackendError err(BackendType::CPU, "test error");
    EXPECT_EQ(err.backend(), BackendType::CPU);
    std::string msg = err.what();
    EXPECT_NE(msg.find("CPU"), std::string::npos);
    EXPECT_NE(msg.find("test error"), std::string::npos);
}

TEST(BackendErrorTest, DifferentBackends) {
    BackendError cuda_err(BackendType::CUDA, "cuda failure");
    EXPECT_EQ(cuda_err.backend(), BackendType::CUDA);
    std::string msg = cuda_err.what();
    EXPECT_NE(msg.find("CUDA"), std::string::npos);
}

// ============================================================================
// Test DeviceInfo structure
// ============================================================================

TEST(DeviceInfoTest, DefaultConstruction) {
    DeviceInfo info;
    EXPECT_EQ(info.name, "");
    EXPECT_EQ(info.total_memory, 0);
    EXPECT_EQ(info.available_memory, 0);
    EXPECT_EQ(info.compute_capability, 0);
    EXPECT_EQ(info.multiprocessor_count, 0);
    EXPECT_EQ(info.max_threads_per_block, 0);
    EXPECT_EQ(info.warp_size, 0);
}

TEST(DeviceInfoTest, FieldAssignment) {
    DeviceInfo info;
    info.name = "Test Device";
    info.total_memory = 1024;
    info.available_memory = 512;
    info.compute_capability = 75;
    info.multiprocessor_count = 16;
    info.max_threads_per_block = 1024;
    info.warp_size = 32;

    EXPECT_EQ(info.name, "Test Device");
    EXPECT_EQ(info.total_memory, 1024);
    EXPECT_EQ(info.available_memory, 512);
    EXPECT_EQ(info.compute_capability, 75);
    EXPECT_EQ(info.multiprocessor_count, 16);
    EXPECT_EQ(info.max_threads_per_block, 1024);
    EXPECT_EQ(info.warp_size, 32);
}

}  // namespace libaccint
