// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2026 Bryce M. Westheimer. All rights reserved.

/// @file test_simd.cpp
/// @brief Unit tests for SIMD abstraction layer (Task 1.3.2)

#include <libaccint/utils/simd.hpp>
#include <libaccint/utils/aligned_alloc.hpp>

#include <gtest/gtest.h>
#include <cmath>
#include <cstdint>
#include <numeric>
#include <vector>

using namespace libaccint::simd;
using namespace libaccint::memory;

// ============================================================================
// Helper: compare SIMD result against expected scalar values
// ============================================================================

class SimdTest : public ::testing::Test {
protected:
    // Aligned buffer holding simd_width doubles
    AlignedBuffer<double, 64> buf_a{static_cast<std::size_t>(simd_width)};
    AlignedBuffer<double, 64> buf_b{static_cast<std::size_t>(simd_width)};
    AlignedBuffer<double, 64> buf_c{static_cast<std::size_t>(simd_width)};
    AlignedBuffer<double, 64> buf_out{static_cast<std::size_t>(simd_width)};

    void SetUp() override {
        for (int i = 0; i < simd_width; ++i) {
            buf_a[i] = 1.0 + i * 0.5;   // 1.0, 1.5, 2.0, 2.5 (AVX2)
            buf_b[i] = 3.0 - i * 0.25;  // 3.0, 2.75, 2.5, 2.25
            buf_c[i] = 0.1 * (i + 1);   // 0.1, 0.2, 0.3, 0.4
            buf_out[i] = 0.0;
        }
    }
};

// ============================================================================
// Load/Store
// ============================================================================

TEST_F(SimdTest, LoadStore) {
    SimdDouble v = load(buf_a.data());
    store(buf_out.data(), v);
    for (int i = 0; i < simd_width; ++i) {
        EXPECT_DOUBLE_EQ(buf_out[i], buf_a[i]);
    }
}

TEST_F(SimdTest, LoaduStoreu) {
    // Use unaligned load/store (pointer offset by 1 double won't be aligned)
    // But we test the function itself works on aligned data too
    SimdDouble v = loadu(buf_a.data());
    storeu(buf_out.data(), v);
    for (int i = 0; i < simd_width; ++i) {
        EXPECT_DOUBLE_EQ(buf_out[i], buf_a[i]);
    }
}

TEST_F(SimdTest, Broadcast) {
    SimdDouble v = broadcast(42.0);
    store(buf_out.data(), v);
    for (int i = 0; i < simd_width; ++i) {
        EXPECT_DOUBLE_EQ(buf_out[i], 42.0);
    }
}

TEST_F(SimdTest, Zero) {
    SimdDouble v = zero();
    store(buf_out.data(), v);
    for (int i = 0; i < simd_width; ++i) {
        EXPECT_DOUBLE_EQ(buf_out[i], 0.0);
    }
}

// ============================================================================
// Arithmetic Operations
// ============================================================================

TEST_F(SimdTest, Add) {
    SimdDouble va = load(buf_a.data());
    SimdDouble vb = load(buf_b.data());
    SimdDouble result = add(va, vb);
    store(buf_out.data(), result);
    for (int i = 0; i < simd_width; ++i) {
        EXPECT_DOUBLE_EQ(buf_out[i], buf_a[i] + buf_b[i]);
    }
}

TEST_F(SimdTest, Sub) {
    SimdDouble va = load(buf_a.data());
    SimdDouble vb = load(buf_b.data());
    SimdDouble result = sub(va, vb);
    store(buf_out.data(), result);
    for (int i = 0; i < simd_width; ++i) {
        EXPECT_DOUBLE_EQ(buf_out[i], buf_a[i] - buf_b[i]);
    }
}

TEST_F(SimdTest, Mul) {
    SimdDouble va = load(buf_a.data());
    SimdDouble vb = load(buf_b.data());
    SimdDouble result = mul(va, vb);
    store(buf_out.data(), result);
    for (int i = 0; i < simd_width; ++i) {
        EXPECT_DOUBLE_EQ(buf_out[i], buf_a[i] * buf_b[i]);
    }
}

TEST_F(SimdTest, Div) {
    SimdDouble va = load(buf_a.data());
    SimdDouble vb = load(buf_b.data());
    SimdDouble result = libaccint::simd::div(va, vb);
    store(buf_out.data(), result);
    for (int i = 0; i < simd_width; ++i) {
        EXPECT_NEAR(buf_out[i], buf_a[i] / buf_b[i], 1e-15);
    }
}

TEST_F(SimdTest, FMA) {
    SimdDouble va = load(buf_a.data());
    SimdDouble vb = load(buf_b.data());
    SimdDouble vc = load(buf_c.data());
    SimdDouble result = fma(va, vb, vc);
    store(buf_out.data(), result);
    for (int i = 0; i < simd_width; ++i) {
        EXPECT_NEAR(buf_out[i], buf_a[i] * buf_b[i] + buf_c[i], 1e-14);
    }
}

TEST_F(SimdTest, FMS) {
    SimdDouble va = load(buf_a.data());
    SimdDouble vb = load(buf_b.data());
    SimdDouble vc = load(buf_c.data());
    SimdDouble result = fms(va, vb, vc);
    store(buf_out.data(), result);
    for (int i = 0; i < simd_width; ++i) {
        EXPECT_NEAR(buf_out[i], buf_a[i] * buf_b[i] - buf_c[i], 1e-14);
    }
}

TEST_F(SimdTest, FNMA) {
    SimdDouble va = load(buf_a.data());
    SimdDouble vb = load(buf_b.data());
    SimdDouble vc = load(buf_c.data());
    SimdDouble result = fnma(va, vb, vc);
    store(buf_out.data(), result);
    for (int i = 0; i < simd_width; ++i) {
        EXPECT_NEAR(buf_out[i], buf_c[i] - buf_a[i] * buf_b[i], 1e-14);
    }
}

TEST_F(SimdTest, Neg) {
    SimdDouble va = load(buf_a.data());
    SimdDouble result = neg(va);
    store(buf_out.data(), result);
    for (int i = 0; i < simd_width; ++i) {
        EXPECT_DOUBLE_EQ(buf_out[i], -buf_a[i]);
    }
}

// ============================================================================
// Math Functions
// ============================================================================

TEST_F(SimdTest, Sqrt) {
    SimdDouble va = load(buf_a.data());
    SimdDouble result = libaccint::simd::sqrt(va);
    store(buf_out.data(), result);
    for (int i = 0; i < simd_width; ++i) {
        EXPECT_NEAR(buf_out[i], std::sqrt(buf_a[i]), 1e-15);
    }
}

TEST_F(SimdTest, Rsqrt) {
    SimdDouble va = load(buf_a.data());
    SimdDouble result = rsqrt(va);
    store(buf_out.data(), result);
    for (int i = 0; i < simd_width; ++i) {
        EXPECT_NEAR(buf_out[i], 1.0 / std::sqrt(buf_a[i]), 1e-14);
    }
}

TEST_F(SimdTest, Abs) {
    // Set some negative values
    for (int i = 0; i < simd_width; ++i) {
        buf_a[i] = (i % 2 == 0) ? -(1.0 + i) : (1.0 + i);
    }
    SimdDouble va = load(buf_a.data());
    SimdDouble result = libaccint::simd::abs(va);
    store(buf_out.data(), result);
    for (int i = 0; i < simd_width; ++i) {
        EXPECT_DOUBLE_EQ(buf_out[i], std::abs(buf_a[i]));
    }
}

TEST_F(SimdTest, Min) {
    SimdDouble va = load(buf_a.data());
    SimdDouble vb = load(buf_b.data());
    SimdDouble result = libaccint::simd::min(va, vb);
    store(buf_out.data(), result);
    for (int i = 0; i < simd_width; ++i) {
        EXPECT_DOUBLE_EQ(buf_out[i], std::min(buf_a[i], buf_b[i]));
    }
}

TEST_F(SimdTest, Max) {
    SimdDouble va = load(buf_a.data());
    SimdDouble vb = load(buf_b.data());
    SimdDouble result = libaccint::simd::max(va, vb);
    store(buf_out.data(), result);
    for (int i = 0; i < simd_width; ++i) {
        EXPECT_DOUBLE_EQ(buf_out[i], std::max(buf_a[i], buf_b[i]));
    }
}

// ============================================================================
// Reduction Operations
// ============================================================================

TEST_F(SimdTest, HorizontalSum) {
    SimdDouble va = load(buf_a.data());
    double result = reduce_add(va);
    double expected = 0.0;
    for (int i = 0; i < simd_width; ++i) {
        expected += buf_a[i];
    }
    EXPECT_NEAR(result, expected, 1e-14);
}

TEST_F(SimdTest, Extract) {
    SimdDouble va = load(buf_a.data());
    EXPECT_DOUBLE_EQ(extract<0>(va), buf_a[0]);
    // Only test lane 0 since simd_width varies
}

// ============================================================================
// Exponential Function
// ============================================================================

TEST_F(SimdTest, Exp) {
    // Test exp for moderate values
    for (int i = 0; i < simd_width; ++i) {
        buf_a[i] = -2.0 + i * 1.0;  // -2, -1, 0, 1
    }
    SimdDouble va = load(buf_a.data());
    SimdDouble result = libaccint::simd::exp(va);
    store(buf_out.data(), result);
    for (int i = 0; i < simd_width; ++i) {
        // The SIMD exp uses a polynomial approximation; allow ~1e-8 relative error
        EXPECT_NEAR(buf_out[i], std::exp(buf_a[i]), std::abs(std::exp(buf_a[i])) * 1e-8);
    }
}

// ============================================================================
// Utility Functions
// ============================================================================

TEST_F(SimdTest, IsAligned) {
    EXPECT_TRUE(is_aligned(buf_a.data()));
}

TEST_F(SimdTest, DotProduct) {
    constexpr std::size_t N = 64;
    AlignedBuffer<double, 64> a(N), b(N);
    for (std::size_t i = 0; i < N; ++i) {
        a[i] = static_cast<double>(i + 1);
        b[i] = 1.0 / static_cast<double>(i + 1);
    }
    double result = dot_product(a.data(), b.data(), N);
    // Each term contributes 1.0, so sum = N
    EXPECT_NEAR(result, static_cast<double>(N), 1e-10);
}

TEST_F(SimdTest, AccumulateScaled) {
    constexpr std::size_t N = 32;
    AlignedBuffer<double, 64> result(N, 0.0);
    AlignedBuffer<double, 64> values(N, 1.0);
    accumulate_scaled(result.data(), values.data(), 2.5, N);
    for (std::size_t i = 0; i < N; ++i) {
        EXPECT_DOUBLE_EQ(result[i], 2.5);
    }
    // Accumulate again
    accumulate_scaled(result.data(), values.data(), 1.5, N);
    for (std::size_t i = 0; i < N; ++i) {
        EXPECT_DOUBLE_EQ(result[i], 4.0);
    }
}

// ============================================================================
// SIMD info
// ============================================================================

TEST(SimdInfo, SimdWidth) {
    // SIMD width should be at least 1
    EXPECT_GE(simd_width, 1);
}

TEST(SimdInfo, SimdISAName) {
    // ISA name should be non-empty
    std::string name(simd_isa_name);
    EXPECT_FALSE(name.empty());
}
