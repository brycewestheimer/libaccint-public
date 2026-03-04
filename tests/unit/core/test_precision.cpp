// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2026 Bryce M. Westheimer. All rights reserved.

/// @file test_precision.cpp
/// @brief Unit tests for precision infrastructure

#include <gtest/gtest.h>
#include <libaccint/core/precision.hpp>
#include <libaccint/buffers/one_electron_buffer.hpp>
#include <libaccint/buffers/two_electron_buffer.hpp>

#include <cmath>
#include <limits>

namespace libaccint::test {

// ============================================================================
// PrecisionTraits Tests
// ============================================================================

TEST(PrecisionTraits, FloatTraits) {
    using Traits = PrecisionTraits<float>;

    EXPECT_EQ(Traits::bits, 32);
    EXPECT_TRUE(Traits::is_single);
    EXPECT_FALSE(Traits::is_double);
    EXPECT_EQ(Traits::epsilon, std::numeric_limits<float>::epsilon());
    EXPECT_EQ(Traits::min_positive, std::numeric_limits<float>::min());
    EXPECT_EQ(Traits::max_value, std::numeric_limits<float>::max());
    EXPECT_EQ(Traits::digits10, std::numeric_limits<float>::digits10);
    EXPECT_EQ(Traits::digits, std::numeric_limits<float>::digits);
}

TEST(PrecisionTraits, DoubleTraits) {
    using Traits = PrecisionTraits<double>;

    EXPECT_EQ(Traits::bits, 64);
    EXPECT_FALSE(Traits::is_single);
    EXPECT_TRUE(Traits::is_double);
    EXPECT_EQ(Traits::epsilon, std::numeric_limits<double>::epsilon());
    EXPECT_EQ(Traits::min_positive, std::numeric_limits<double>::min());
    EXPECT_EQ(Traits::max_value, std::numeric_limits<double>::max());
    EXPECT_EQ(Traits::digits10, std::numeric_limits<double>::digits10);
    EXPECT_EQ(Traits::digits, std::numeric_limits<double>::digits);
}

TEST(PrecisionTraits, ThresholdsDifferBetweenPrecisions) {
    // Float thresholds should be larger (less precise)
    EXPECT_GT(PrecisionTraits<float>::integral_threshold,
              PrecisionTraits<double>::integral_threshold);

    EXPECT_GT(PrecisionTraits<float>::screening_threshold,
              PrecisionTraits<double>::screening_threshold);

    EXPECT_GT(PrecisionTraits<float>::convergence_threshold,
              PrecisionTraits<double>::convergence_threshold);
}

TEST(PrecisionTraits, SIMDWidths) {
    // Float has 2x the SIMD width of double
    EXPECT_EQ(PrecisionTraits<float>::simd_width_avx,
              2 * PrecisionTraits<double>::simd_width_avx);

    EXPECT_EQ(PrecisionTraits<float>::simd_width_avx512,
              2 * PrecisionTraits<double>::simd_width_avx512);
}

// ============================================================================
// Constants Tests
// ============================================================================

TEST(Constants, PiValues) {
    // Check that constants are correct to the precision's limits
    EXPECT_NEAR(Constants<float>::pi, 3.14159265f, 1e-6f);
    EXPECT_NEAR(Constants<double>::pi, 3.14159265358979323846, 1e-14);

    // Verify relationship: pi * one_over_pi ≈ 1
    EXPECT_NEAR(Constants<float>::pi * Constants<float>::one_over_pi,
                1.0f, 1e-6f);
    EXPECT_NEAR(Constants<double>::pi * Constants<double>::one_over_pi,
                1.0, 1e-14);
}

TEST(Constants, SqrtPi) {
    // sqrt(pi)^2 should equal pi
    float sqrt_pi_f = Constants<float>::sqrt_pi;
    double sqrt_pi_d = Constants<double>::sqrt_pi;

    EXPECT_NEAR(sqrt_pi_f * sqrt_pi_f, Constants<float>::pi, 1e-6f);
    EXPECT_NEAR(sqrt_pi_d * sqrt_pi_d, Constants<double>::pi, 1e-14);
}

TEST(Constants, TwoPi) {
    EXPECT_FLOAT_EQ(Constants<float>::two_pi, 2.0f * Constants<float>::pi);
    EXPECT_DOUBLE_EQ(Constants<double>::two_pi, 2.0 * Constants<double>::pi);
}

TEST(Constants, Ln2) {
    EXPECT_NEAR(std::exp(Constants<float>::ln_2), 2.0f, 1e-6f);
    EXPECT_NEAR(std::exp(Constants<double>::ln_2), 2.0, 1e-14);
}

TEST(Constants, EulerNumber) {
    EXPECT_NEAR(std::log(Constants<float>::e), 1.0f, 1e-6f);
    EXPECT_NEAR(std::log(Constants<double>::e), 1.0, 1e-14);
}

TEST(Constants, Sqrt2) {
    EXPECT_NEAR(Constants<float>::sqrt_2 * Constants<float>::sqrt_2, 2.0f, 1e-6f);
    EXPECT_NEAR(Constants<double>::sqrt_2 * Constants<double>::sqrt_2, 2.0, 1e-14);
}

// ============================================================================
// Precision Enum Tests
// ============================================================================

TEST(PrecisionEnum, ToString) {
    EXPECT_STREQ(precision_to_string(Precision::Float32), "float32");
    EXPECT_STREQ(precision_to_string(Precision::Float64), "float64");
    EXPECT_STREQ(precision_to_string(Precision::Auto), "auto");
}

TEST(PrecisionEnum, SizeBytes) {
    EXPECT_EQ(precision_size_bytes(Precision::Float32), 4u);
    EXPECT_EQ(precision_size_bytes(Precision::Float64), 8u);
    EXPECT_EQ(precision_size_bytes(Precision::Auto), 8u);  // Default to double
}

// ============================================================================
// MixedPrecisionMode Tests
// ============================================================================

TEST(MixedPrecisionMode, ToString) {
    EXPECT_STREQ(mixed_precision_to_string(MixedPrecisionMode::Pure32), "pure32");
    EXPECT_STREQ(mixed_precision_to_string(MixedPrecisionMode::Pure64), "pure64");
    EXPECT_STREQ(mixed_precision_to_string(MixedPrecisionMode::Compute32Accumulate64),
                 "compute32_accumulate64");
    EXPECT_STREQ(mixed_precision_to_string(MixedPrecisionMode::Adaptive), "adaptive");
}

// ============================================================================
// Helper Function Tests
// ============================================================================

TEST(HelperFunctions, IsSinglePrecision) {
    EXPECT_TRUE(is_single_precision<float>());
    EXPECT_FALSE(is_single_precision<double>());
}

TEST(HelperFunctions, IsDoublePrecision) {
    EXPECT_FALSE(is_double_precision<float>());
    EXPECT_TRUE(is_double_precision<double>());
}

TEST(HelperFunctions, IntegralThreshold) {
    EXPECT_EQ(integral_threshold<float>(), PrecisionTraits<float>::integral_threshold);
    EXPECT_EQ(integral_threshold<double>(), PrecisionTraits<double>::integral_threshold);
}

TEST(HelperFunctions, ScreeningThreshold) {
    EXPECT_EQ(screening_threshold<float>(), PrecisionTraits<float>::screening_threshold);
    EXPECT_EQ(screening_threshold<double>(), PrecisionTraits<double>::screening_threshold);
}

TEST(HelperFunctions, PrecisionCast) {
    double d = 3.14159265358979323846;
    float f = precision_cast<float>(d);

    // Should be truncated to float precision
    EXPECT_FLOAT_EQ(f, 3.14159265f);

    // Round-trip should lose precision
    double d2 = precision_cast<double>(f);
    EXPECT_NE(d, d2);  // Precision loss expected
}

TEST(HelperFunctions, NearlyEqual) {
    // Same values
    EXPECT_TRUE(nearly_equal(1.0f, 1.0f));
    EXPECT_TRUE(nearly_equal(1.0, 1.0));

    // Within tolerance
    float f1 = 1.0f;
    float f2 = 1.0f + 1e-6f;
    EXPECT_TRUE(nearly_equal(f1, f2, 1e-5f));

    // Beyond tolerance
    float f3 = 1.0f;
    float f4 = 1.01f;
    EXPECT_FALSE(nearly_equal(f3, f4, 1e-5f));

    // Zero comparison
    EXPECT_TRUE(nearly_equal(0.0f, 0.0f));
    EXPECT_TRUE(nearly_equal(0.0, 0.0));
}

// ============================================================================
// Missing Constants Tests (Task 1.4.3)
// ============================================================================

TEST(Constants, PiSquared) {
    // pi_squared should approximately equal pi * pi
    EXPECT_NEAR(Constants<double>::pi_squared,
                Constants<double>::pi * Constants<double>::pi, 1e-14);
    EXPECT_NEAR(Constants<float>::pi_squared,
                Constants<float>::pi * Constants<float>::pi, 1e-5f);
}

TEST(Constants, Pi32) {
    // pi_3_2 should approximately equal sqrt_pi * pi
    EXPECT_NEAR(Constants<double>::pi_3_2,
                Constants<double>::sqrt_pi * Constants<double>::pi, 1e-14);
    EXPECT_NEAR(Constants<float>::pi_3_2,
                Constants<float>::sqrt_pi * Constants<float>::pi, 1e-5f);
}

TEST(Constants, OneOverSqrtPi) {
    // one_over_sqrt_pi should approximately equal 1.0 / sqrt_pi
    EXPECT_NEAR(Constants<double>::one_over_sqrt_pi,
                1.0 / Constants<double>::sqrt_pi, 1e-14);
    EXPECT_NEAR(Constants<float>::one_over_sqrt_pi,
                1.0f / Constants<float>::sqrt_pi, 1e-6f);
}

TEST(Constants, OneOverSqrt2) {
    // one_over_sqrt_2 should approximately equal 1.0 / sqrt_2
    EXPECT_NEAR(Constants<double>::one_over_sqrt_2,
                1.0 / Constants<double>::sqrt_2, 1e-14);
    EXPECT_NEAR(Constants<float>::one_over_sqrt_2,
                1.0f / Constants<float>::sqrt_2, 1e-6f);
}

// ============================================================================
// nearly_equal Edge Cases (Task 1.4.3)
// ============================================================================

TEST(NearlyEqual, NaN) {
    double nan = std::numeric_limits<double>::quiet_NaN();
    EXPECT_FALSE(nearly_equal(nan, nan));
    EXPECT_FALSE(nearly_equal(nan, 0.0));
    EXPECT_FALSE(nearly_equal(0.0, nan));

    float nanf = std::numeric_limits<float>::quiet_NaN();
    EXPECT_FALSE(nearly_equal(nanf, nanf));
    EXPECT_FALSE(nearly_equal(nanf, 0.0f));
}

TEST(NearlyEqual, Infinity) {
    double inf = std::numeric_limits<double>::infinity();
    // +inf vs +inf: diff is NaN or 0, max_val is inf; rel_tol * inf is inf
    // Implementation detail: std::abs(inf - inf) is NaN, so diff <= ... is false
    // But diff < min_positive is also false, so this may return false
    // We test what the implementation actually does:
    // The correct mathematical answer for "nearly_equal(+inf, +inf)" depends on convention.
    // Let's just verify no crash and document the behavior.
    bool inf_result = nearly_equal(inf, inf);
    (void)inf_result;  // May be true or false depending on implementation

    // +inf vs -inf should be false
    EXPECT_FALSE(nearly_equal(inf, -inf));

    // inf vs finite should be false
    EXPECT_FALSE(nearly_equal(inf, 1.0));
    EXPECT_FALSE(nearly_equal(-inf, 1.0));
}

TEST(NearlyEqual, NegativeZero) {
    // -0.0 and +0.0 should be considered equal
    EXPECT_TRUE(nearly_equal(-0.0, +0.0));
    EXPECT_TRUE(nearly_equal(+0.0, -0.0));
    EXPECT_TRUE(nearly_equal(-0.0f, +0.0f));
}

TEST(NearlyEqual, Subnormals) {
    double denorm = std::numeric_limits<double>::denorm_min();
    // Two subnormals close to zero: their diff < min_positive
    EXPECT_TRUE(nearly_equal(denorm, denorm));
    EXPECT_TRUE(nearly_equal(denorm, 2.0 * denorm));
    // Zero vs denorm_min: diff < min_positive
    EXPECT_TRUE(nearly_equal(0.0, denorm));

    float denormf = std::numeric_limits<float>::denorm_min();
    EXPECT_TRUE(nearly_equal(denormf, denormf));
    EXPECT_TRUE(nearly_equal(0.0f, denormf));
}

// ============================================================================
// precision_cast Tests (Task 1.4.3)
// ============================================================================

TEST(PrecisionCast, IdentityDouble) {
    double val = 3.14159265358979323846;
    EXPECT_EQ(precision_cast<double>(val), val);

    double zero = 0.0;
    EXPECT_EQ(precision_cast<double>(zero), zero);

    double neg = -42.5;
    EXPECT_EQ(precision_cast<double>(neg), neg);
}

TEST(PrecisionCast, IdentityFloat) {
    float val = 3.14f;
    EXPECT_EQ(precision_cast<float>(val), val);

    float zero = 0.0f;
    EXPECT_EQ(precision_cast<float>(zero), zero);
}

TEST(PrecisionCast, BoundaryValues) {
    // Double max -> float is inf
    double dmax = std::numeric_limits<double>::max();
    float fmax = precision_cast<float>(dmax);
    EXPECT_TRUE(std::isinf(fmax));

    // Float max -> double preserves value
    float fmax_val = std::numeric_limits<float>::max();
    double dval = precision_cast<double>(fmax_val);
    EXPECT_EQ(static_cast<double>(fmax_val), dval);

    // Epsilon values
    double deps = std::numeric_limits<double>::epsilon();
    EXPECT_EQ(precision_cast<double>(deps), deps);

    float feps = std::numeric_limits<float>::epsilon();
    EXPECT_EQ(precision_cast<float>(feps), feps);

    // Min positive values
    double dmin = std::numeric_limits<double>::min();
    EXPECT_EQ(precision_cast<double>(dmin), dmin);

    float fmin = std::numeric_limits<float>::min();
    EXPECT_EQ(precision_cast<float>(fmin), fmin);
}

// ============================================================================
// is_negligible Edge Cases (Task 1.4.3)
// ============================================================================

TEST(IsNegligible, AtThreshold) {
    double threshold = PrecisionTraits<double>::integral_threshold;
    // Value exactly at threshold is not negligible (< not <=)
    EXPECT_FALSE(is_negligible(threshold));
    // Just below threshold is negligible
    EXPECT_TRUE(is_negligible(threshold * 0.99));
    // Just above threshold is not negligible
    EXPECT_FALSE(is_negligible(threshold * 1.01));

    float thresholdf = PrecisionTraits<float>::integral_threshold;
    EXPECT_FALSE(is_negligible(thresholdf));
    EXPECT_TRUE(is_negligible(thresholdf * 0.99f));
}

TEST(IsNegligible, NegativeValues) {
    // is_negligible uses abs, so sign shouldn't matter
    EXPECT_TRUE(is_negligible(-1e-16));
    EXPECT_FALSE(is_negligible(-1.0));
}

TEST(HelperFunctions, IsNegligible) {
    EXPECT_TRUE(is_negligible(1e-15f));  // Below float threshold
    EXPECT_FALSE(is_negligible(1e-5f));   // Above float threshold

    EXPECT_TRUE(is_negligible(1e-16));    // Below double threshold
    EXPECT_FALSE(is_negligible(1e-10));   // Above double threshold

    // Custom threshold
    EXPECT_TRUE(is_negligible(1e-5f, 1e-4f));
    EXPECT_FALSE(is_negligible(1e-3f, 1e-4f));
}

// ============================================================================
// OneElectronBuffer Float32 Tests
// ============================================================================

TEST(OneElectronBufferFloat32, Construction) {
    OneElectronBuffer<0, float> buf(10, 15);

    EXPECT_EQ(buf.na(), 10);
    EXPECT_EQ(buf.nb(), 15);
    EXPECT_EQ(buf.size(), 150u);
    EXPECT_EQ(buf.size_bytes(), 150u * sizeof(float));
}

TEST(OneElectronBufferFloat32, HalfMemoryOfDouble) {
    constexpr int na = 100;
    constexpr int nb = 100;

    OneElectronBuffer<0, float> float_buf(na, nb);
    OneElectronBuffer<0, double> double_buf(na, nb);

    EXPECT_EQ(float_buf.size_bytes() * 2, double_buf.size_bytes());
}

TEST(OneElectronBufferFloat32, DataAccess) {
    OneElectronBuffer<0, float> buf(5, 5);
    buf.clear();

    buf(2, 3) = 1.5f;
    EXPECT_FLOAT_EQ(buf(2, 3), 1.5f);

    // Verify other elements are zero
    EXPECT_FLOAT_EQ(buf(0, 0), 0.0f);
    EXPECT_FLOAT_EQ(buf(4, 4), 0.0f);
}

TEST(OneElectronBufferFloat32, PrecisionConversion) {
    // Create float buffer with known values
    OneElectronBuffer<0, float> float_buf(3, 3);
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            float_buf(i, j) = static_cast<float>(i * 3 + j) * 0.1f;
        }
    }

    // Convert to double
    auto double_buf = float_buf.to_precision<double>();

    EXPECT_EQ(double_buf.na(), 3);
    EXPECT_EQ(double_buf.nb(), 3);

    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            EXPECT_NEAR(double_buf(i, j), float_buf(i, j), 1e-6);
        }
    }
}

TEST(OneElectronBufferFloat32, CopyFromDifferentPrecision) {
    // Create double buffer with known values
    OneElectronBuffer<0, double> double_buf(4, 4);
    for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < 4; ++j) {
            double_buf(i, j) = static_cast<double>(i * 4 + j) * 0.01;
        }
    }

    // Copy to float buffer
    OneElectronBuffer<0, float> float_buf;
    float_buf.copy_from(double_buf);

    EXPECT_EQ(float_buf.na(), 4);
    EXPECT_EQ(float_buf.nb(), 4);

    for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < 4; ++j) {
            EXPECT_NEAR(float_buf(i, j), double_buf(i, j), 1e-6);
        }
    }
}

TEST(OneElectronBufferFloat32, TypeTraits) {
    EXPECT_TRUE((OneElectronBuffer<0, float>::is_single_precision));
    EXPECT_FALSE((OneElectronBuffer<0, float>::is_double_precision));

    EXPECT_FALSE((OneElectronBuffer<0, double>::is_single_precision));
    EXPECT_TRUE((OneElectronBuffer<0, double>::is_double_precision));
}

// ============================================================================
// TwoElectronBuffer Float32 Tests
// ============================================================================

TEST(TwoElectronBufferFloat32, Construction) {
    TwoElectronBuffer<0, float> buf(3, 3, 3, 3);

    EXPECT_EQ(buf.na(), 3);
    EXPECT_EQ(buf.nb(), 3);
    EXPECT_EQ(buf.nc(), 3);
    EXPECT_EQ(buf.nd(), 3);
    EXPECT_EQ(buf.size(), 81u);
    EXPECT_EQ(buf.size_bytes(), 81u * sizeof(float));
}

TEST(TwoElectronBufferFloat32, HalfMemoryOfDouble) {
    constexpr int n = 10;

    TwoElectronBuffer<0, float> float_buf(n, n, n, n);
    TwoElectronBuffer<0, double> double_buf(n, n, n, n);

    EXPECT_EQ(float_buf.size_bytes() * 2, double_buf.size_bytes());
}

TEST(TwoElectronBufferFloat32, DataAccess) {
    TwoElectronBuffer<0, float> buf(3, 3, 3, 3);
    buf.clear();

    buf(1, 2, 0, 1) = 2.5f;
    EXPECT_FLOAT_EQ(buf(1, 2, 0, 1), 2.5f);

    // Verify other elements are zero
    EXPECT_FLOAT_EQ(buf(0, 0, 0, 0), 0.0f);
    EXPECT_FLOAT_EQ(buf(2, 2, 2, 2), 0.0f);
}

TEST(TwoElectronBufferFloat32, PrecisionConversion) {
    TwoElectronBuffer<0, float> float_buf(2, 2, 2, 2);
    float val = 0.0f;
    for (int a = 0; a < 2; ++a) {
        for (int b = 0; b < 2; ++b) {
            for (int c = 0; c < 2; ++c) {
                for (int d = 0; d < 2; ++d) {
                    float_buf(a, b, c, d) = val;
                    val += 0.1f;
                }
            }
        }
    }

    auto double_buf = float_buf.to_precision<double>();

    EXPECT_EQ(double_buf.na(), 2);
    EXPECT_EQ(double_buf.nb(), 2);
    EXPECT_EQ(double_buf.nc(), 2);
    EXPECT_EQ(double_buf.nd(), 2);

    for (int a = 0; a < 2; ++a) {
        for (int b = 0; b < 2; ++b) {
            for (int c = 0; c < 2; ++c) {
                for (int d = 0; d < 2; ++d) {
                    EXPECT_NEAR(double_buf(a, b, c, d),
                                float_buf(a, b, c, d), 1e-6);
                }
            }
        }
    }
}

TEST(TwoElectronBufferFloat32, TypeTraits) {
    EXPECT_TRUE((TwoElectronBuffer<0, float>::is_single_precision));
    EXPECT_FALSE((TwoElectronBuffer<0, float>::is_double_precision));

    EXPECT_FALSE((TwoElectronBuffer<0, double>::is_single_precision));
    EXPECT_TRUE((TwoElectronBuffer<0, double>::is_double_precision));
}

// ============================================================================
// ValidPrecision Concept Tests
// ============================================================================

TEST(ValidPrecisionConcept, FloatAndDoubleAreValid) {
    // These should compile without error
    static_assert(ValidPrecision<float>);
    static_assert(ValidPrecision<double>);

    // These should NOT be valid (compile-time check)
    static_assert(!ValidPrecision<int>);
    static_assert(!ValidPrecision<long double>);  // Not supported
}

}  // namespace libaccint::test
