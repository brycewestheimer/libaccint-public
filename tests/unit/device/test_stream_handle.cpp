// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2026 Bryce M. Westheimer. All rights reserved.

/// @file test_stream_handle.cpp
/// @brief Unit tests for GPU stream operations
///
/// Task 12.3.3

#include <gtest/gtest.h>
#include <libaccint/config.hpp>

#if !LIBACCINT_USE_CUDA
#define NO_GPU_BACKEND
#endif

#ifndef NO_GPU_BACKEND
#include <libaccint/device/device_manager.hpp>
#include "../../../src/device/common/gpu_compat.hpp"
#endif

namespace libaccint::device {
namespace {

// ============================================================================
// Stream Handle Tests
// ============================================================================

TEST(StreamHandleTest, CreateStreamSucceeds) {
#ifdef NO_GPU_BACKEND
    EXPECT_FALSE(LIBACCINT_USE_CUDA);
#else
    auto& mgr = DeviceManager::instance();
    if (!mgr.has_devices()) {
        GTEST_SKIP() << "No GPU devices available";
    }

    gpuStream_t stream = nullptr;
    gpuError_t err = gpuStreamCreate(&stream);
    ASSERT_EQ(err, gpuSuccess);
    EXPECT_NE(stream, nullptr);

    gpuStreamDestroy(stream);
#endif
}

TEST(StreamHandleTest, SynchronizeStreamDoesNotThrow) {
#ifdef NO_GPU_BACKEND
    EXPECT_FALSE(LIBACCINT_USE_CUDA);
#else
    auto& mgr = DeviceManager::instance();
    if (!mgr.has_devices()) {
        GTEST_SKIP() << "No GPU devices available";
    }

    gpuStream_t stream = nullptr;
    gpuError_t err = gpuStreamCreate(&stream);
    ASSERT_EQ(err, gpuSuccess);

    err = gpuStreamSynchronize(stream);
    EXPECT_EQ(err, gpuSuccess);

    gpuStreamDestroy(stream);
#endif
}

TEST(StreamHandleTest, MultipleStreamsAreDistinct) {
#ifdef NO_GPU_BACKEND
    EXPECT_FALSE(LIBACCINT_USE_CUDA);
#else
    auto& mgr = DeviceManager::instance();
    if (!mgr.has_devices()) {
        GTEST_SKIP() << "No GPU devices available";
    }

    gpuStream_t stream1 = nullptr;
    gpuStream_t stream2 = nullptr;

    ASSERT_EQ(gpuStreamCreate(&stream1), gpuSuccess);
    ASSERT_EQ(gpuStreamCreate(&stream2), gpuSuccess);

    EXPECT_NE(stream1, nullptr);
    EXPECT_NE(stream2, nullptr);
    EXPECT_NE(stream1, stream2);

    gpuStreamDestroy(stream1);
    gpuStreamDestroy(stream2);
#endif
}

}  // namespace
}  // namespace libaccint::device
