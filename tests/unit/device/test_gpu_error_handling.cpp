// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2026 Bryce M. Westheimer. All rights reserved.

/// @file test_gpu_error_handling.cpp
/// @brief Unit tests for GPU error handling
///
/// Task 12.3.10

#include <gtest/gtest.h>
#include <libaccint/config.hpp>

#if !LIBACCINT_USE_CUDA
#define NO_GPU_BACKEND
#endif

#ifndef NO_GPU_BACKEND
#include <libaccint/device/device_manager.hpp>
#include <libaccint/device/multi_device_memory.hpp>
#endif

namespace libaccint::device {
namespace {

// ============================================================================
// GPU Error Handling Tests
// ============================================================================

TEST(GpuErrorHandlingTest, InvalidDeviceThrows) {
#ifdef NO_GPU_BACKEND
    EXPECT_FALSE(LIBACCINT_USE_CUDA);
#else
    auto& mgr = DeviceManager::instance();
    if (!mgr.has_devices()) {
        GTEST_SKIP() << "No GPU devices available";
    }

    // Setting an invalid device ID should throw DeviceError
    const int invalid_id = mgr.device_count() + 100;
    EXPECT_THROW(mgr.set_current_device(invalid_id), DeviceError);
#endif
}

TEST(GpuErrorHandlingTest, MemoryUsageTracked) {
#ifdef NO_GPU_BACKEND
    EXPECT_FALSE(LIBACCINT_USE_CUDA);
#else
    auto& mgr = DeviceManager::instance();
    if (!mgr.has_devices()) {
        GTEST_SKIP() << "No GPU devices available";
    }

    DeviceWorkspace ws(0);
    EXPECT_EQ(ws.allocated_bytes(), 0u);

    constexpr size_t count = 500;
    double* ptr = ws.allocate<double>(count);
    ASSERT_NE(ptr, nullptr);
    EXPECT_EQ(ws.allocated_bytes(), count * sizeof(double));

    ws.deallocate(ptr);
    EXPECT_EQ(ws.allocated_bytes(), 0u);
#endif
}

}  // namespace
}  // namespace libaccint::device
