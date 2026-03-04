// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2026 Bryce M. Westheimer. All rights reserved.

/// @file test_device_host_transfer.cpp
/// @brief Unit tests for device-host data transfer
///
/// Task 12.3.9

#include <gtest/gtest.h>
#include <libaccint/config.hpp>

#if !LIBACCINT_USE_CUDA
#define NO_GPU_BACKEND
#endif

#ifndef NO_GPU_BACKEND
#include <libaccint/device/multi_device_memory.hpp>
#include <libaccint/device/device_manager.hpp>

#include <numeric>
#include <vector>
#endif

namespace libaccint::device {
namespace {

// ============================================================================
// Device-Host Transfer Tests
// ============================================================================

TEST(DeviceHostTransferTest, HostToDeviceAndBack) {
#ifdef NO_GPU_BACKEND
    EXPECT_FALSE(LIBACCINT_USE_CUDA);
#else
    auto& mgr = DeviceManager::instance();
    if (!mgr.has_devices()) {
        GTEST_SKIP() << "No GPU devices available";
    }

    constexpr size_t count = 256;
    std::vector<double> host_src(count);
    std::iota(host_src.begin(), host_src.end(), 1.0);

    // Allocate on device
    DeviceWorkspace ws(0);
    double* d_ptr = ws.allocate<double>(count);
    ASSERT_NE(d_ptr, nullptr);

    // Host → Device
    gpuError_t err = gpuMemcpy(d_ptr, host_src.data(),
                                count * sizeof(double),
                                gpuMemcpyHostToDevice);
    ASSERT_EQ(err, gpuSuccess);

    // Device → Host
    std::vector<double> host_dst(count, 0.0);
    err = gpuMemcpy(host_dst.data(), d_ptr,
                     count * sizeof(double),
                     gpuMemcpyDeviceToHost);
    ASSERT_EQ(err, gpuSuccess);

    // Verify round-trip
    for (size_t i = 0; i < count; ++i) {
        EXPECT_DOUBLE_EQ(host_dst[i], host_src[i])
            << "Mismatch at index " << i;
    }

    ws.deallocate(d_ptr);
#endif
}

TEST(DeviceHostTransferTest, LargeTransfer) {
#ifdef NO_GPU_BACKEND
    EXPECT_FALSE(LIBACCINT_USE_CUDA);
#else
    auto& mgr = DeviceManager::instance();
    if (!mgr.has_devices()) {
        GTEST_SKIP() << "No GPU devices available";
    }

    constexpr size_t count = 1'000'000;
    std::vector<double> host_src(count);
    std::iota(host_src.begin(), host_src.end(), 0.0);

    DeviceWorkspace ws(0);
    double* d_ptr = ws.allocate<double>(count);
    ASSERT_NE(d_ptr, nullptr);

    // Host → Device
    gpuError_t err = gpuMemcpy(d_ptr, host_src.data(),
                                count * sizeof(double),
                                gpuMemcpyHostToDevice);
    ASSERT_EQ(err, gpuSuccess);

    // Device → Host
    std::vector<double> host_dst(count, 0.0);
    err = gpuMemcpy(host_dst.data(), d_ptr,
                     count * sizeof(double),
                     gpuMemcpyDeviceToHost);
    ASSERT_EQ(err, gpuSuccess);

    // Spot-check a few values
    EXPECT_DOUBLE_EQ(host_dst[0], 0.0);
    EXPECT_DOUBLE_EQ(host_dst[count / 2], static_cast<double>(count / 2));
    EXPECT_DOUBLE_EQ(host_dst[count - 1], static_cast<double>(count - 1));

    ws.deallocate(d_ptr);
#endif
}

TEST(DeviceHostTransferTest, ZeroLengthTransferNoError) {
#ifdef NO_GPU_BACKEND
    EXPECT_FALSE(LIBACCINT_USE_CUDA);
#else
    auto& mgr = DeviceManager::instance();
    if (!mgr.has_devices()) {
        GTEST_SKIP() << "No GPU devices available";
    }

    DeviceWorkspace ws(0);

    // Allocate a small buffer so we have a valid device pointer
    double* d_ptr = ws.allocate<double>(1);
    ASSERT_NE(d_ptr, nullptr);

    double host_val = 42.0;

    // Zero-length transfers should succeed without error
    gpuError_t err = gpuMemcpy(d_ptr, &host_val, 0, gpuMemcpyHostToDevice);
    EXPECT_EQ(err, gpuSuccess);

    err = gpuMemcpy(&host_val, d_ptr, 0, gpuMemcpyDeviceToHost);
    EXPECT_EQ(err, gpuSuccess);

    ws.deallocate(d_ptr);
#endif
}

}  // namespace
}  // namespace libaccint::device
