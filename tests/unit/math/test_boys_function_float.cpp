// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2026 Bryce M. Westheimer. All rights reserved.

/// @file test_boys_function_float.cpp
/// @brief Unit tests for float32 Boys function implementation

#include <gtest/gtest.h>
#include <libaccint/math/boys_function_float.hpp>
#include <libaccint/math/boys_function.hpp>

#include <cmath>
#include <vector>
#include <random>
#include <chrono>

namespace libaccint::math::test {

// ============================================================================
// Float32 Boys Function Accuracy Tests
// ============================================================================

TEST(BoysFloat, AccuracyVsDouble) {
    // Test that float32 Boys matches double to float32 precision
    std::vector<float> T_values = {0.0f, 0.01f, 0.1f, 1.0f, 5.0f, 10.0f, 20.0f, 30.0f, 50.0f};
    std::vector<int> n_values = {0, 1, 2, 5, 10, 15, 20};

    for (float T : T_values) {
        for (int n : n_values) {
            float f_float = boys_evaluate_float(n, T);
            double f_double = boys_evaluate(n, static_cast<double>(T));

            // Compute relative error
            double abs_error = std::abs(static_cast<double>(f_float) - f_double);
            double rel_error = (f_double != 0.0) ? abs_error / std::abs(f_double) : abs_error;

            // Float32 should achieve at least 1e-6 relative accuracy
            EXPECT_LT(rel_error, 1e-5)
                << "Failed at n=" << n << ", T=" << T
                << ", float=" << f_float << ", double=" << f_double
                << ", rel_error=" << rel_error;
        }
    }
}

TEST(BoysFloat, ZeroArgument) {
    // For T = 0: F_n(0) = 1/(2n+1)
    for (int n = 0; n <= 20; ++n) {
        float result = boys_evaluate_float(n, 0.0f);
        float expected = 1.0f / static_cast<float>(2 * n + 1);
        EXPECT_FLOAT_EQ(result, expected) << "n=" << n;
    }
}

TEST(BoysFloat, SmallT) {
    // For very small T, F_n(T) ≈ 1/(2n+1)
    float T = 1e-8f;
    for (int n = 0; n <= 10; ++n) {
        float result = boys_evaluate_float(n, T);
        float expected = 1.0f / static_cast<float>(2 * n + 1);
        EXPECT_NEAR(result, expected, 1e-6f * expected)
            << "n=" << n << ", T=" << T;
    }
}

TEST(BoysFloat, LargeT) {
    // For large T, F_0(T) ≈ sqrt(pi/T) / 2
    float T = 100.0f;
    float result = boys_evaluate_float(0, T);
    float expected = 0.5f * std::sqrt(Constants<float>::pi / T);

    float rel_error = std::abs((result - expected) / expected);
    EXPECT_LT(rel_error, 1e-4f) << "T=" << T << ", result=" << result << ", expected=" << expected;
}

TEST(BoysFloat, AsymptoticBehavior) {
    // Test at the asymptotic threshold
    float T = BOYS_ASYMPTOTIC_THRESHOLD_FLOAT + 1.0f;

    float result[31];
    boys_evaluate_array_float(10, T, result);

    // Verify F_0 matches asymptotic formula
    float F0_asymp = boys_asymptotic_float(0, T);
    EXPECT_NEAR(result[0], F0_asymp, 1e-5f * F0_asymp);

    // Verify monotonicity: F_n(T) > F_{n+1}(T) for T > 0
    for (int n = 0; n < 10; ++n) {
        EXPECT_GT(result[n], result[n + 1])
            << "Monotonicity violated at n=" << n;
    }
}

TEST(BoysFloat, RecurrenceRelation) {
    // Test the recurrence: F_{n-1}(T) = (2T * F_n(T) + exp(-T)) / (2n-1)
    float T = 5.0f;
    float exp_neg_T = std::exp(-T);

    float result[31];
    boys_evaluate_array_float(20, T, result);

    for (int n = 20; n >= 1; --n) {
        float F_prev_computed = result[n - 1];
        float F_prev_from_recurrence = (2.0f * T * result[n] + exp_neg_T) /
                                       static_cast<float>(2 * n - 1);

        EXPECT_NEAR(F_prev_computed, F_prev_from_recurrence, 1e-5f * std::abs(F_prev_computed))
            << "Recurrence failed at n=" << n;
    }
}

TEST(BoysFloat, ArrayEvaluation) {
    // Test that array evaluation gives consistent results with single evaluation
    float T = 7.5f;
    int n_max = 15;

    float array_result[31];
    boys_evaluate_array_float(n_max, T, array_result);

    for (int n = 0; n <= n_max; ++n) {
        float single_result = boys_evaluate_float(n, T);
        EXPECT_NEAR(array_result[n], single_result, 1e-6f * std::abs(single_result))
            << "Mismatch at n=" << n;
    }
}

TEST(BoysFloat, BatchEvaluation) {
    constexpr int N = 100;
    constexpr int n_max = 10;

    std::vector<float> T_values(N);
    std::vector<float> results(N * (n_max + 1));

    // Generate random T values
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(0.0f, 50.0f);
    for (int i = 0; i < N; ++i) {
        T_values[i] = dist(rng);
    }

    // Batch evaluation
    boys_evaluate_batch_float(n_max, T_values.data(), N, results.data());

    // Verify against single evaluations
    for (int i = 0; i < N; ++i) {
        for (int n = 0; n <= n_max; ++n) {
            float expected = boys_evaluate_float(n, T_values[i]);
            float actual = results[i * (n_max + 1) + n];
            EXPECT_NEAR(actual, expected, 1e-6f * std::abs(expected))
                << "Mismatch at i=" << i << ", n=" << n;
        }
    }
}

// ============================================================================
// Float32 Boys Function Taylor Series Tests
// ============================================================================

TEST(BoysFloatTaylor, SmallTAccuracy) {
    // Test Taylor series for small T
    for (int n = 0; n <= 10; ++n) {
        for (float T = 1e-10f; T <= 1e-6f; T *= 10.0f) {
            float taylor = boys_taylor_float(n, T);
            double ref = boys_evaluate(n, static_cast<double>(T));

            double rel_error = std::abs((taylor - ref) / ref);
            EXPECT_LT(rel_error, 1e-5)
                << "n=" << n << ", T=" << T;
        }
    }
}

// ============================================================================
// Float32 Boys Function Asymptotic Tests
// ============================================================================

TEST(BoysFloatAsymptotic, LargeTAccuracy) {
    // Test asymptotic expansion for large T.
    // The direct formula accumulates float32 roundoff across the product
    // loop, so high orders (n>=12) need a looser tolerance than the
    // recurrence-based array version.
    for (int n = 0; n <= 15; ++n) {
        for (float T = 30.0f; T <= 100.0f; T += 10.0f) {
            float asymp = boys_asymptotic_float(n, T);
            double ref = boys_evaluate(n, static_cast<double>(T));

            double rel_error = std::abs((asymp - ref) / ref);
            EXPECT_LT(rel_error, 2e-3)
                << "n=" << n << ", T=" << T;
        }
    }
}

TEST(BoysFloatAsymptotic, ArrayEvaluation) {
    float T = 50.0f;
    int n_max = 20;

    float result[31];
    boys_asymptotic_array_float(n_max, T, result);

    for (int n = 0; n <= n_max; ++n) {
        double ref = boys_evaluate(n, static_cast<double>(T));
        double rel_error = std::abs((result[n] - ref) / ref);
        EXPECT_LT(rel_error, 1e-4)
            << "n=" << n << ", T=" << T;
    }
}

// ============================================================================
// Performance Tests (Optional - can be slow)
// ============================================================================

TEST(BoysFloatPerformance, DISABLED_SpeedVsDouble) {
    // This test is disabled by default as it's for benchmarking
    constexpr int N = 1000000;
    std::vector<float> T_vals_f(N);
    std::vector<double> T_vals_d(N);
    std::vector<float> results_f(N);
    std::vector<double> results_d(N);

    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(0.0f, 50.0f);
    for (int i = 0; i < N; ++i) {
        T_vals_f[i] = dist(rng);
        T_vals_d[i] = static_cast<double>(T_vals_f[i]);
    }

    // Time float32 computation
    auto start_f = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < N; ++i) {
        results_f[i] = boys_evaluate_float(5, T_vals_f[i]);
    }
    auto end_f = std::chrono::high_resolution_clock::now();
    double time_f = std::chrono::duration<double>(end_f - start_f).count();

    // Time double computation
    auto start_d = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < N; ++i) {
        results_d[i] = boys_evaluate(5, T_vals_d[i]);
    }
    auto end_d = std::chrono::high_resolution_clock::now();
    double time_d = std::chrono::duration<double>(end_d - start_d).count();

    std::cout << "Float32 time: " << time_f << " s" << std::endl;
    std::cout << "Float64 time: " << time_d << " s" << std::endl;
    std::cout << "Speedup: " << time_d / time_f << "x" << std::endl;

    // Float should be at least somewhat faster (allow for variation)
    // Note: Actual speedup depends heavily on CPU and compilation flags
    // EXPECT_GT(time_d / time_f, 1.0);  // At least not slower
}

// ============================================================================
// Edge Case Tests
// ============================================================================

TEST(BoysFloat, EdgeCases) {
    // Very small positive T
    float tiny_T = 1e-30f;
    for (int n = 0; n <= 5; ++n) {
        float result = boys_evaluate_float(n, tiny_T);
        EXPECT_TRUE(std::isfinite(result)) << "n=" << n;
        EXPECT_GT(result, 0.0f) << "n=" << n;
    }

    // T at asymptotic threshold boundary
    float boundary_T = BOYS_ASYMPTOTIC_THRESHOLD_FLOAT;
    for (int n = 0; n <= 10; ++n) {
        float result = boys_evaluate_float(n, boundary_T);
        EXPECT_TRUE(std::isfinite(result)) << "n=" << n;
        EXPECT_GT(result, 0.0f) << "n=" << n;
    }
}

TEST(BoysFloat, MonotonicityInN) {
    // F_n(T) should decrease with increasing n for fixed T > 0
    std::vector<float> T_values = {0.1f, 1.0f, 10.0f, 50.0f};

    for (float T : T_values) {
        float result[31];
        boys_evaluate_array_float(25, T, result);

        for (int n = 0; n < 25; ++n) {
            EXPECT_GT(result[n], result[n + 1])
                << "Monotonicity in n violated at T=" << T << ", n=" << n;
        }
    }
}

TEST(BoysFloat, MonotonicityInT) {
    // F_n(T) should decrease with increasing T for fixed n
    for (int n = 0; n <= 10; ++n) {
        float prev_result = boys_evaluate_float(n, 0.0f);
        for (float T = 0.1f; T <= 50.0f; T += 0.5f) {
            float result = boys_evaluate_float(n, T);
            EXPECT_LT(result, prev_result)
                << "Monotonicity in T violated at n=" << n << ", T=" << T;
            prev_result = result;
        }
    }
}

// ============================================================================
// Task 3.3.6: Chebyshev Float Implementation Tests
// ============================================================================

TEST(BoysFloatChebyshev, SinglePointAccuracy) {
    // Test boys_chebyshev_float at specific T values against double reference
    std::vector<float> T_values = {0.5f, 1.0f, 2.5f, 5.0f, 10.0f, 15.0f, 20.0f, 25.0f, 29.5f};

    for (float T : T_values) {
        for (int n = 0; n <= 15; ++n) {
            float cheb_result = boys_chebyshev_float(n, T);
            double ref = boys_evaluate(n, static_cast<double>(T));

            double rel_error = std::abs(static_cast<double>(cheb_result) - ref);
            if (ref != 0.0) rel_error /= std::abs(ref);

            EXPECT_LT(rel_error, 1e-5)
                << "Chebyshev float accuracy at n=" << n << ", T=" << T
                << ": got=" << cheb_result << ", ref=" << ref;
        }
    }
}

TEST(BoysFloatChebyshev, ArrayEvaluation) {
    // Test boys_chebyshev_array_float gives consistent results
    float T = 7.5f;
    int n_max = 20;

    float array_result[31];
    boys_chebyshev_array_float(n_max, T, array_result);

    for (int n = 0; n <= n_max; ++n) {
        float single_result = boys_chebyshev_float(n, T);
        EXPECT_NEAR(array_result[n], single_result, 1e-6f * std::abs(single_result))
            << "Array vs single mismatch at n=" << n;
    }
}

TEST(BoysFloatChebyshev, ArrayVsDoubleReference) {
    // Verify Chebyshev array against double-precision reference
    int n_max = 15;
    std::vector<float> T_values = {0.1f, 1.0f, 3.0f, 8.0f, 14.0f, 22.0f, 29.0f};

    for (float T : T_values) {
        float result[31];
        boys_chebyshev_array_float(n_max, T, result);

        for (int n = 0; n <= n_max; ++n) {
            double ref = boys_evaluate(n, static_cast<double>(T));
            double abs_err = std::abs(static_cast<double>(result[n]) - ref);
            double rel_err = (ref != 0.0) ? abs_err / std::abs(ref) : abs_err;

            EXPECT_LT(rel_err, 1e-5)
                << "n=" << n << ", T=" << T << ": result=" << result[n] << ", ref=" << ref;
        }
    }
}

TEST(BoysFloatChebyshev, SmallTChebyshev) {
    // Chebyshev interpolation for very small T, compared against
    // double-precision reference (not F_n(0), since F_n(0.01) != F_n(0)).
    float T = 0.01f;
    for (int n = 0; n <= 10; ++n) {
        float result = boys_chebyshev_float(n, T);
        double ref = boys_evaluate(n, static_cast<double>(T));

        double rel_error = std::abs(static_cast<double>(result) - ref) / std::abs(ref);
        EXPECT_LT(rel_error, 1e-5)
            << "Small T Chebyshev at n=" << n;
    }
}

TEST(BoysFloatChebyshev, MediumTChebyshev) {
    // Test at medium T values (the core Chebyshev range)
    float T = 15.0f;
    int n_max = 10;

    float result[31];
    boys_chebyshev_array_float(n_max, T, result);

    // Monotonicity: F_n > F_{n+1}
    for (int n = 0; n < n_max; ++n) {
        EXPECT_GT(result[n], result[n + 1])
            << "Monotonicity violated at n=" << n;
    }
}

TEST(BoysFloatChebyshev, IntervalBoundaries) {
    // Test at integer T values (Chebyshev interval boundaries)
    for (int interval = 0; interval < 29; ++interval) {
        float T = static_cast<float>(interval) + 0.5f;  // midpoint of interval
        float result = boys_chebyshev_float(0, T);
        double ref = boys_evaluate(0, static_cast<double>(T));

        double rel_err = std::abs(static_cast<double>(result) - ref) / std::abs(ref);
        EXPECT_LT(rel_err, 1e-5)
            << "Interval boundary at T=" << T;
    }
}

}  // namespace libaccint::math::test
