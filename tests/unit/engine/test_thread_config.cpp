// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2026 Bryce M. Westheimer. All rights reserved.

/// @file test_thread_config.cpp
/// @brief Unit tests for ThreadConfig, ScopedThreadCount, and CacheLineAligned

#include <libaccint/engine/thread_config.hpp>

#include <gtest/gtest.h>
#include <type_traits>
#include <vector>

using namespace libaccint;
using namespace libaccint::engine;

// =============================================================================
// ThreadConfig Tests
// =============================================================================

TEST(ThreadConfigTest, HardwareThreadsPositive) {
    // hardware_concurrency may return 0, but our wrapper returns >= 1
    EXPECT_GE(ThreadConfig::hardware_threads(), 1);
}

TEST(ThreadConfigTest, RecommendedThreadsPositive) {
    EXPECT_GE(ThreadConfig::recommended_threads(), 1);
}

TEST(ThreadConfigTest, SetAndGetThreadCount) {
    // Save current state
    int original = ThreadConfig::num_threads();

    ThreadConfig::set_num_threads(4);
    EXPECT_EQ(ThreadConfig::num_threads(), 4);
    EXPECT_EQ(ThreadConfig::recommended_threads(), 4);

    ThreadConfig::set_num_threads(1);
    EXPECT_EQ(ThreadConfig::num_threads(), 1);
    EXPECT_EQ(ThreadConfig::recommended_threads(), 1);

    // Restore
    ThreadConfig::set_num_threads(original);
}

TEST(ThreadConfigTest, ResetToAutoDetect) {
    int original = ThreadConfig::num_threads();

    ThreadConfig::set_num_threads(4);
    EXPECT_EQ(ThreadConfig::num_threads(), 4);

    ThreadConfig::reset();
    EXPECT_EQ(ThreadConfig::num_threads(), 0);

    // After reset, recommended should rely on OMP or hardware
    EXPECT_GE(ThreadConfig::recommended_threads(), 1);

    // Restore
    ThreadConfig::set_num_threads(original);
}

TEST(ThreadConfigTest, ZeroMeansAutoDetect) {
    int original = ThreadConfig::num_threads();

    ThreadConfig::set_num_threads(0);
    EXPECT_EQ(ThreadConfig::num_threads(), 0);
    // recommended_threads should still return >= 1
    EXPECT_GE(ThreadConfig::recommended_threads(), 1);

    // Restore
    ThreadConfig::set_num_threads(original);
}

TEST(ThreadConfigTest, EffectiveThreadsPositive) {
    EXPECT_GE(ThreadConfig::effective_threads(), 1);
}

TEST(ThreadConfigTest, OpenMPAvailability) {
#if defined(_OPENMP)
    EXPECT_TRUE(ThreadConfig::openmp_available());
#else
    EXPECT_FALSE(ThreadConfig::openmp_available());
#endif
}

TEST(ThreadConfigTest, ResolveExplicit) {
    EXPECT_EQ(ThreadConfig::resolve(4), 4);
    EXPECT_EQ(ThreadConfig::resolve(1), 1);
    EXPECT_EQ(ThreadConfig::resolve(16), 16);
}

TEST(ThreadConfigTest, ResolveAutoDetect) {
    int resolved = ThreadConfig::resolve(0);
    EXPECT_GE(resolved, 1);
}

// =============================================================================
// ScopedThreadCount Tests
// =============================================================================

TEST(ScopedThreadCountTest, RestoresOnDestruction) {
    int original = ThreadConfig::num_threads();

    {
        ScopedThreadCount guard(8);
        EXPECT_EQ(ThreadConfig::num_threads(), 8);
    }

    EXPECT_EQ(ThreadConfig::num_threads(), original);
}

TEST(ScopedThreadCountTest, NestedScopes) {
    int original = ThreadConfig::num_threads();

    {
        ScopedThreadCount outer(4);
        EXPECT_EQ(ThreadConfig::num_threads(), 4);

        {
            ScopedThreadCount inner(2);
            EXPECT_EQ(ThreadConfig::num_threads(), 2);
        }

        EXPECT_EQ(ThreadConfig::num_threads(), 4);
    }

    EXPECT_EQ(ThreadConfig::num_threads(), original);
}

// =============================================================================
// CacheLineAligned Tests
// =============================================================================

TEST(CacheLineAlignedTest, Alignment) {
    // CacheLineAligned should be aligned to CACHE_LINE_SIZE
    EXPECT_EQ(alignof(CacheLineAligned<double>), CACHE_LINE_SIZE);
    EXPECT_EQ(alignof(CacheLineAligned<int>), CACHE_LINE_SIZE);
}

TEST(CacheLineAlignedTest, SizeAtLeastCacheLine) {
    // Size should be at least CACHE_LINE_SIZE to prevent false sharing
    EXPECT_GE(sizeof(CacheLineAligned<double>), CACHE_LINE_SIZE);
    EXPECT_GE(sizeof(CacheLineAligned<int>), CACHE_LINE_SIZE);
}

TEST(CacheLineAlignedTest, ValueAccess) {
    CacheLineAligned<double> val(3.14);
    EXPECT_DOUBLE_EQ(val.get(), 3.14);

    val.get() = 2.71;
    EXPECT_DOUBLE_EQ(val.get(), 2.71);
}

TEST(CacheLineAlignedTest, DefaultConstruction) {
    CacheLineAligned<double> val;
    EXPECT_DOUBLE_EQ(val.get(), 0.0);
}

TEST(CacheLineAlignedTest, ImplicitConversion) {
    CacheLineAligned<double> val(42.0);
    double d = val;
    EXPECT_DOUBLE_EQ(d, 42.0);
}

TEST(CacheLineAlignedTest, VectorOfAligned) {
    // Verify that arrays of CacheLineAligned objects don't share cache lines
    std::vector<CacheLineAligned<double>> vals(4);
    for (int i = 0; i < 4; ++i) {
        vals[i] = CacheLineAligned<double>(static_cast<double>(i));
    }

    for (int i = 0; i < 4; ++i) {
        EXPECT_DOUBLE_EQ(vals[i].get(), static_cast<double>(i));
    }

    // Verify stride between elements
    if (vals.size() >= 2) {
        auto addr0 = reinterpret_cast<std::uintptr_t>(&vals[0]);
        auto addr1 = reinterpret_cast<std::uintptr_t>(&vals[1]);
        EXPECT_GE(addr1 - addr0, CACHE_LINE_SIZE);
    }
}

TEST(CacheLineAlignedTest, CacheLineSizeConstant) {
    EXPECT_EQ(CACHE_LINE_SIZE, 64u);
}

