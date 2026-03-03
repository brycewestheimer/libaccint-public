// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2026 Bryce M. Westheimer. All rights reserved.

/// @file test_math_constants.cpp
/// @brief Unit tests for mathematical constants and precision utilities

#include <libaccint/utils/constants.hpp>
#include <libaccint/core/precision.hpp>
#include <libaccint/math/boys_function.hpp>
#include <gtest/gtest.h>
#include <cmath>
#include <limits>

using namespace libaccint;

// ============================================================================
// Constants from utils/constants.hpp
// ============================================================================

TEST(MathConstantsTest, PI) {
    EXPECT_NEAR(constants::PI, M_PI, 1e-15);
    EXPECT_NEAR(constants::PI, 3.14159265358979323846, 1e-15);
}

TEST(MathConstantsTest, SQRT_PI) {
    EXPECT_NEAR(constants::SQRT_PI, std::sqrt(M_PI), 1e-15);
    EXPECT_NEAR(constants::SQRT_PI * constants::SQRT_PI, constants::PI, 1e-15);
}

TEST(MathConstantsTest, INV_SQRT_PI) {
    EXPECT_NEAR(constants::INV_SQRT_PI, 1.0 / std::sqrt(M_PI), 1e-15);
    EXPECT_NEAR(constants::INV_SQRT_PI * constants::SQRT_PI, 1.0, 1e-15);
}

TEST(MathConstantsTest, TWO_PI) {
    EXPECT_NEAR(constants::TWO_PI, 2.0 * M_PI, 1e-15);
    EXPECT_NEAR(constants::TWO_PI, 2.0 * constants::PI, 1e-15);
}

TEST(MathConstantsTest, SelfConsistency) {
    // SQRT_PI^2 = PI
    EXPECT_NEAR(constants::SQRT_PI * constants::SQRT_PI, constants::PI, 1e-15);

    // INV_SQRT_PI * SQRT_PI = 1
    EXPECT_NEAR(constants::INV_SQRT_PI * constants::SQRT_PI, 1.0, 1e-15);

    // TWO_PI / 2 = PI
    EXPECT_NEAR(constants::TWO_PI / 2.0, constants::PI, 1e-15);

    // INV_SQRT_PI^2 = 1/PI
    EXPECT_NEAR(constants::INV_SQRT_PI * constants::INV_SQRT_PI,
                1.0 / constants::PI, 1e-15);
}

// ============================================================================
// Constants<double> from core/precision.hpp
// ============================================================================

TEST(PrecisionConstantsDoubleTest, Pi) {
    EXPECT_NEAR(Constants<double>::pi, M_PI, 1e-15);
}

TEST(PrecisionConstantsDoubleTest, SqrtPi) {
    EXPECT_NEAR(Constants<double>::sqrt_pi, std::sqrt(M_PI), 1e-15);
    EXPECT_NEAR(Constants<double>::sqrt_pi * Constants<double>::sqrt_pi,
                Constants<double>::pi, 1e-14);
}

TEST(PrecisionConstantsDoubleTest, OneOverSqrtPi) {
    EXPECT_NEAR(Constants<double>::one_over_sqrt_pi,
                1.0 / std::sqrt(M_PI), 1e-15);
}

TEST(PrecisionConstantsDoubleTest, TwoPi) {
    EXPECT_NEAR(Constants<double>::two_pi,
                2.0 * M_PI, 1e-15);
}

TEST(PrecisionConstantsDoubleTest, SqrtTwo) {
    EXPECT_NEAR(Constants<double>::sqrt_2,
                std::sqrt(2.0), 1e-15);
}

TEST(PrecisionConstantsDoubleTest, OneOverSqrtTwo) {
    EXPECT_NEAR(Constants<double>::one_over_sqrt_2,
                1.0 / std::sqrt(2.0), 1e-15);
    EXPECT_NEAR(Constants<double>::one_over_sqrt_2 * Constants<double>::sqrt_2,
                1.0, 1e-15);
}

TEST(PrecisionConstantsDoubleTest, Ln2) {
    EXPECT_NEAR(Constants<double>::ln_2, std::log(2.0), 1e-15);
}

TEST(PrecisionConstantsDoubleTest, Epsilon) {
    EXPECT_GT(Constants<double>::epsilon, 0.0);
    EXPECT_LT(Constants<double>::epsilon, 1e-10);
}

TEST(PrecisionConstantsDoubleTest, PiSquared) {
    EXPECT_NEAR(Constants<double>::pi_squared,
                Constants<double>::pi * Constants<double>::pi, 1e-14);
}

TEST(PrecisionConstantsDoubleTest, OneOverPi) {
    EXPECT_NEAR(Constants<double>::one_over_pi,
                1.0 / Constants<double>::pi, 1e-15);
}

TEST(PrecisionConstantsDoubleTest, EulersConstant) {
    EXPECT_NEAR(Constants<double>::e, std::exp(1.0), 1e-15);
}

// ============================================================================
// Constants<float> from core/precision.hpp
// ============================================================================

TEST(PrecisionConstantsFloatTest, Pi) {
    EXPECT_NEAR(Constants<float>::pi, static_cast<float>(M_PI), 1e-6f);
}

TEST(PrecisionConstantsFloatTest, SqrtPi) {
    EXPECT_NEAR(Constants<float>::sqrt_pi,
                std::sqrt(static_cast<float>(M_PI)), 1e-6f);
}

TEST(PrecisionConstantsFloatTest, SelfConsistency) {
    float sqrt_pi = Constants<float>::sqrt_pi;
    float pi = Constants<float>::pi;

    EXPECT_NEAR(sqrt_pi * sqrt_pi, pi, 1e-5f);

    float inv_sqrt_pi = Constants<float>::one_over_sqrt_pi;
    EXPECT_NEAR(inv_sqrt_pi * sqrt_pi, 1.0f, 1e-6f);
}

TEST(PrecisionConstantsFloatTest, ReducedPrecision) {
    // Float constants should be less precise than double
    double d_pi = Constants<double>::pi;
    float f_pi = Constants<float>::pi;

    // The float version should be close to double but with float precision
    EXPECT_NEAR(static_cast<double>(f_pi), d_pi, 1e-7);
}

// ============================================================================
// PrecisionTraits Tests
// ============================================================================

TEST(PrecisionTraitsTest, DoubleTraits) {
    EXPECT_GT(PrecisionTraits<double>::integral_threshold, 0.0);
    EXPECT_GT(PrecisionTraits<double>::screening_threshold, 0.0);
}

TEST(PrecisionTraitsTest, FloatTraits) {
    EXPECT_GT(PrecisionTraits<float>::integral_threshold, 0.0f);
    EXPECT_GT(PrecisionTraits<float>::screening_threshold, 0.0f);
}

TEST(PrecisionTraitsTest, FloatMorePermissive) {
    // Float thresholds should be larger (more permissive) than double
    EXPECT_GT(PrecisionTraits<float>::integral_threshold,
              PrecisionTraits<double>::integral_threshold);
}

TEST(PrecisionTraitsTest, BitCounts) {
    EXPECT_EQ(PrecisionTraits<float>::bits, 32);
    EXPECT_EQ(PrecisionTraits<double>::bits, 64);
}

TEST(PrecisionTraitsTest, PrecisionFlags) {
    EXPECT_TRUE(PrecisionTraits<float>::is_single);
    EXPECT_FALSE(PrecisionTraits<float>::is_double);
    EXPECT_FALSE(PrecisionTraits<double>::is_single);
    EXPECT_TRUE(PrecisionTraits<double>::is_double);
}

// ============================================================================
// double_factorial Tests
// ============================================================================

TEST(DoubleFactorialTest, KnownValues) {
    // double_factorial(n) returns (2n-1)!!
    // n=0: (-1)!! = 1 (by convention)
    // n=1: 1!! = 1
    // n=2: 3!! = 3
    // n=3: 5!! = 15
    // n=4: 7!! = 105
    // n=5: 9!! = 945
    // n=6: 11!! = 10395
    EXPECT_DOUBLE_EQ(math::double_factorial(0), 1.0);
    EXPECT_DOUBLE_EQ(math::double_factorial(1), 1.0);
    EXPECT_DOUBLE_EQ(math::double_factorial(2), 3.0);
    EXPECT_DOUBLE_EQ(math::double_factorial(3), 15.0);
    EXPECT_DOUBLE_EQ(math::double_factorial(4), 105.0);
    EXPECT_DOUBLE_EQ(math::double_factorial(5), 945.0);
    EXPECT_DOUBLE_EQ(math::double_factorial(6), 10395.0);
}

TEST(DoubleFactorialTest, LargerValues) {
    EXPECT_DOUBLE_EQ(math::double_factorial(7), 135135.0);
    EXPECT_DOUBLE_EQ(math::double_factorial(8), 2027025.0);
    EXPECT_DOUBLE_EQ(math::double_factorial(9), 34459425.0);
}

TEST(DoubleFactorialTest, Recurrence) {
    // (2n-1)!! = (2n-1) * (2(n-1)-1)!!
    // i.e., double_factorial(n) = (2n-1) * double_factorial(n-1)
    for (int n = 1; n <= 12; ++n) {
        EXPECT_DOUBLE_EQ(math::double_factorial(n),
                         static_cast<double>(2 * n - 1) * math::double_factorial(n - 1))
            << "n=" << n;
    }
}
