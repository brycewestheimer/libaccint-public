// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2026 Bryce M. Westheimer. All rights reserved.

/// @file test_device_memory_manager.cpp
/// @brief Unit tests for DeviceWorkspace and MultiDeviceMemoryManager
///
/// Tasks 12.3.1, 12.3.2, 12.3.4, 12.3.5

#include <gtest/gtest.h>
#include <libaccint/config.hpp>

#if !LIBACCINT_USE_CUDA
#define NO_GPU_BACKEND
#endif

#ifndef NO_GPU_BACKEND
#include <libaccint/device/multi_device_memory.hpp>
#include <libaccint/device/device_manager.hpp>
#endif

namespace libaccint::device {
namespace {

// ============================================================================
// DeviceWorkspace Tests
// ============================================================================

TEST(DeviceMemoryManagerTest, AllocateReturnsNonNull) {
#ifdef NO_GPU_BACKEND
    EXPECT_FALSE(LIBACCINT_USE_CUDA);
#else
    auto& mgr = DeviceManager::instance();
    if (!mgr.has_devices()) {
        GTEST_SKIP() << "No GPU devices available";
    }

    DeviceWorkspace ws(0);
    double* ptr = ws.allocate<double>(100);
    ASSERT_NE(ptr, nullptr);
    ws.deallocate(ptr);
#endif
}

TEST(DeviceMemoryManagerTest, DeallocateDecrementsBytes) {
#ifdef NO_GPU_BACKEND
    EXPECT_FALSE(LIBACCINT_USE_CUDA);
#else
    auto& mgr = DeviceManager::instance();
    if (!mgr.has_devices()) {
        GTEST_SKIP() << "No GPU devices available";
    }

    DeviceWorkspace ws(0);
    double* ptr = ws.allocate<double>(100);
    ASSERT_NE(ptr, nullptr);
    EXPECT_GT(ws.allocated_bytes(), 0u);

    ws.deallocate(ptr);
    EXPECT_EQ(ws.allocated_bytes(), 0u);
#endif
}

TEST(DeviceMemoryManagerTest, AllocateZeroReturnsNull) {
#ifdef NO_GPU_BACKEND
    EXPECT_FALSE(LIBACCINT_USE_CUDA);
#else
    auto& mgr = DeviceManager::instance();
    if (!mgr.has_devices()) {
        GTEST_SKIP() << "No GPU devices available";
    }

    DeviceWorkspace ws(0);
    double* ptr = ws.allocate<double>(0);
    EXPECT_EQ(ptr, nullptr);
#endif
}

TEST(DeviceMemoryManagerTest, AllocateTracksBytesCorrectly) {
#ifdef NO_GPU_BACKEND
    EXPECT_FALSE(LIBACCINT_USE_CUDA);
#else
    auto& mgr = DeviceManager::instance();
    if (!mgr.has_devices()) {
        GTEST_SKIP() << "No GPU devices available";
    }

    DeviceWorkspace ws(0);
    double* ptr = ws.allocate<double>(100);
    ASSERT_NE(ptr, nullptr);
    EXPECT_EQ(ws.allocated_bytes(), 100 * sizeof(double));
    ws.deallocate(ptr);
#endif
}

TEST(DeviceMemoryManagerTest, MultipleAllocations) {
#ifdef NO_GPU_BACKEND
    EXPECT_FALSE(LIBACCINT_USE_CUDA);
#else
    auto& mgr = DeviceManager::instance();
    if (!mgr.has_devices()) {
        GTEST_SKIP() << "No GPU devices available";
    }

    DeviceWorkspace ws(0);
    double* p1 = ws.allocate<double>(100);
    float* p2 = ws.allocate<float>(200);
    ASSERT_NE(p1, nullptr);
    ASSERT_NE(p2, nullptr);

    const size_t expected = 100 * sizeof(double) + 200 * sizeof(float);
    EXPECT_EQ(ws.allocated_bytes(), expected);

    ws.deallocate(p1);
    ws.deallocate(p2);
    EXPECT_EQ(ws.allocated_bytes(), 0u);
#endif
}

TEST(DeviceMemoryManagerTest, ResetFreesAllMemory) {
#ifdef NO_GPU_BACKEND
    EXPECT_FALSE(LIBACCINT_USE_CUDA);
#else
    auto& mgr = DeviceManager::instance();
    if (!mgr.has_devices()) {
        GTEST_SKIP() << "No GPU devices available";
    }

    DeviceWorkspace ws(0);
    ws.allocate<double>(100);
    ws.allocate<double>(200);
    EXPECT_GT(ws.allocated_bytes(), 0u);

    ws.reset();
    EXPECT_EQ(ws.allocated_bytes(), 0u);
#endif
}

TEST(DeviceMemoryManagerTest, DeallocateNullptrIsNoop) {
#ifdef NO_GPU_BACKEND
    EXPECT_FALSE(LIBACCINT_USE_CUDA);
#else
    auto& mgr = DeviceManager::instance();
    if (!mgr.has_devices()) {
        GTEST_SKIP() << "No GPU devices available";
    }

    DeviceWorkspace ws(0);
    double* ptr = ws.allocate<double>(50);
    const size_t bytes_before = ws.allocated_bytes();

    ws.deallocate<double>(nullptr);
    EXPECT_EQ(ws.allocated_bytes(), bytes_before);

    ws.deallocate(ptr);
#endif
}

TEST(DeviceMemoryManagerTest, DoubleDeallocateIgnored) {
#ifdef NO_GPU_BACKEND
    EXPECT_FALSE(LIBACCINT_USE_CUDA);
#else
    auto& mgr = DeviceManager::instance();
    if (!mgr.has_devices()) {
        GTEST_SKIP() << "No GPU devices available";
    }

    DeviceWorkspace ws(0);
    double* ptr = ws.allocate<double>(100);
    ASSERT_NE(ptr, nullptr);

    ws.deallocate(ptr);
    EXPECT_EQ(ws.allocated_bytes(), 0u);

    // Second deallocation of the same pointer should not crash
    EXPECT_NO_THROW(ws.deallocate(ptr));
    EXPECT_EQ(ws.allocated_bytes(), 0u);
#endif
}

// ============================================================================
// MultiDeviceMemoryManager Tests
// ============================================================================

TEST(DeviceMemoryManagerTest, MultiDeviceManagerCreatesWorkspaces) {
#ifdef NO_GPU_BACKEND
    EXPECT_FALSE(LIBACCINT_USE_CUDA);
#else
    auto& mgr = DeviceManager::instance();
    if (!mgr.has_devices()) {
        GTEST_SKIP() << "No GPU devices available";
    }

    const int n_devices = mgr.device_count();
    std::vector<int> device_ids;
    for (int i = 0; i < n_devices; ++i) {
        device_ids.push_back(i);
    }

    MultiDeviceMemoryManager multi_mgr(device_ids);
    EXPECT_EQ(multi_mgr.device_count(), n_devices);
#endif
}

TEST(DeviceMemoryManagerTest, MultiDeviceAllocateTotalTracking) {
#ifdef NO_GPU_BACKEND
    EXPECT_FALSE(LIBACCINT_USE_CUDA);
#else
    auto& mgr = DeviceManager::instance();
    if (!mgr.has_devices()) {
        GTEST_SKIP() << "No GPU devices available";
    }

    const int n_devices = mgr.device_count();
    std::vector<int> device_ids;
    for (int i = 0; i < n_devices; ++i) {
        device_ids.push_back(i);
    }

    MultiDeviceMemoryManager multi_mgr(device_ids);

    // Allocate on each device
    size_t expected_total = 0;
    std::vector<double*> ptrs;
    for (int i = 0; i < n_devices; ++i) {
        const size_t count = 100 * (i + 1);
        double* ptr = multi_mgr.allocate<double>(i, count);
        ASSERT_NE(ptr, nullptr);
        ptrs.push_back(ptr);
        expected_total += count * sizeof(double);
    }

    EXPECT_EQ(multi_mgr.total_allocated_bytes(), expected_total);

    // Clean up
    for (int i = 0; i < n_devices; ++i) {
        multi_mgr.deallocate(i, ptrs[i]);
    }
#endif
}

}  // namespace
}  // namespace libaccint::device
