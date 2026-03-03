// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2026 Bryce M. Westheimer. All rights reserved.

#include <libaccint/math/boys_function.hpp>
#include <libaccint/utils/constants.hpp>
#include <gtest/gtest.h>
#include <cmath>
#include <chrono>
#include <vector>
#include <array>

using namespace libaccint;
using namespace libaccint::math;

namespace {

// Helper function to compute relative error
double relative_error(double computed, double reference) {
    if (std::abs(reference) < 1e-100) {
        return std::abs(computed - reference);
    }
    return std::abs((computed - reference) / reference);
}

// Independent reference implementation using std::erf + upward recursion.
// This is mathematically exact (to machine precision) for the Boys function:
//   F_0(T) = sqrt(pi)/(2*sqrt(T)) * erf(sqrt(T))
//   F_n(T) = ((2n-1)*F_{n-1}(T) - exp(-T)) / (2T)
double boys_reference_erf(int n, double T) {
    const double F_0 = 0.5 * std::sqrt(M_PI / T) * std::erf(std::sqrt(T));
    if (n == 0) return F_0;

    const double exp_neg_T = std::exp(-T);
    double prev = F_0;
    for (int k = 1; k <= n; ++k) {
        prev = ((2 * k - 1) * prev - exp_neg_T) / (2.0 * T);
    }
    return prev;
}

// Reference values computed using mpmath with 50-digit precision.
// F_n(T) = sqrt(pi)/(2*sqrt(T)) * erf(sqrt(T)) with upward recursion.
struct BoysReferenceValue {
    double T;
    int n;
    double value;
};

std::vector<BoysReferenceValue> get_reference_values() {
    std::vector<BoysReferenceValue> values;

    // T = 30.0
    values.push_back({30.0, 0, 0.161802159379640070});
    values.push_back({30.0, 1, 0.00269670265632577489});
    values.push_back({30.0, 2, 0.000134835132814729141});
    values.push_back({30.0, 3, 0.0000112362610663344912});
    values.push_back({30.0, 4, 1.31089712284608682e-6});
    values.push_back({30.0, 5, 1.96634566867309194e-7});
    values.push_back({30.0, 10, 1.75197494140663688e-10});
    values.push_back({30.0, 15, 2.12735263635999292e-12});
    values.push_back({30.0, 20, 1.37585444267909310e-13});

    // T = 50.0
    values.push_back({50.0, 0, 0.125331413731550025});
    values.push_back({50.0, 1, 0.00125331413731550025});
    values.push_back({50.0, 2, 0.0000375994241194650075});
    values.push_back({50.0, 3, 1.87997120597325037e-6});
    values.push_back({50.0, 4, 1.31597984418127524e-7});
    values.push_back({50.0, 5, 1.18438185976314753e-8});
    values.push_back({50.0, 10, 8.20581205806632246e-13});
    values.push_back({50.0, 15, 7.75836961421463697e-16});
    values.push_back({50.0, 20, 4.00848387001506030e-18});

    // T = 100.0
    values.push_back({100.0, 0, 0.0886226925452758014});
    values.push_back({100.0, 1, 0.000443113462726379007});
    values.push_back({100.0, 2, 6.64670194089568510e-6});
    values.push_back({100.0, 3, 1.66167548522392128e-7});
    values.push_back({100.0, 4, 5.81586419828372446e-9});
    values.push_back({100.0, 5, 2.61713888922767601e-10});
    values.push_back({100.0, 10, 5.66639194474392784e-16});
    values.push_back({100.0, 15, 1.67419304936778228e-20});
    values.push_back({100.0, 20, 2.70312149116753752e-24});

    // T = 500.0
    values.push_back({500.0, 0, 0.0396332729760601101});
    values.push_back({500.0, 1, 0.0000396332729760601101});
    values.push_back({500.0, 2, 1.18899818928180330e-7});
    values.push_back({500.0, 3, 5.94499094640901652e-10});
    values.push_back({500.0, 4, 4.16149366248631156e-12});
    values.push_back({500.0, 5, 3.74534429623768041e-14});
    values.push_back({500.0, 10, 2.59490561548383331e-23});
    values.push_back({500.0, 15, 2.45341189953553858e-31});
    values.push_back({500.0, 20, 1.26759488049721351e-38});

    return values;
}

} // anonymous namespace

// =============================================================================
// Double Factorial Tests
// =============================================================================

TEST(BoysDoublefactorialTest, SmallValues) {
    EXPECT_DOUBLE_EQ(double_factorial(0), 1.0);
    EXPECT_DOUBLE_EQ(double_factorial(1), 1.0);
    EXPECT_DOUBLE_EQ(double_factorial(2), 3.0);
    EXPECT_DOUBLE_EQ(double_factorial(3), 15.0);
    EXPECT_DOUBLE_EQ(double_factorial(4), 105.0);
    EXPECT_DOUBLE_EQ(double_factorial(5), 945.0);
}

TEST(BoysDoublefactorialTest, MediumValues) {
    EXPECT_DOUBLE_EQ(double_factorial(6), 10395.0);
    EXPECT_DOUBLE_EQ(double_factorial(7), 135135.0);
    EXPECT_DOUBLE_EQ(double_factorial(8), 2027025.0);
    EXPECT_DOUBLE_EQ(double_factorial(9), 34459425.0);
    EXPECT_DOUBLE_EQ(double_factorial(10), 654729075.0);
    EXPECT_DOUBLE_EQ(double_factorial(15), 6190283353629375.0);
}

TEST(BoysDoublefactorialTest, RecursionRelation) {
    for (int n = 2; n <= 10; ++n) {
        double expected = (2 * n - 1) * double_factorial(n - 1);
        double computed = double_factorial(n);
        EXPECT_NEAR(computed, expected, 1e-10) << "Mismatch for n = " << n;
    }
}

TEST(BoysDoublefactorialTest, LargeValues) {
    double df_16 = double_factorial(16);
    double expected_16 = 31.0 * 6190283353629375.0;
    EXPECT_NEAR(df_16, expected_16, expected_16 * 1e-14);

    double df_17 = double_factorial(17);
    double expected_17 = 33.0 * df_16;
    EXPECT_NEAR(df_17, expected_17, expected_17 * 1e-14);
}

// =============================================================================
// Boys Asymptotic Single Value Tests
//
// boys_asymptotic(n, T) uses the leading term only:
//   F_n(T) ≈ (2n-1)!! * sqrt(pi) / (2^(n+1) * T^(n+0.5))
//
// This approximation is excellent when n << T but degrades as n approaches T.
// The omitted correction is O(exp(-T)), which becomes significant relative to
// F_n only when n is large enough that F_n itself is very small.
// =============================================================================

TEST(BoysAsymptoticTest, T100_AllOrders) {
    // At T=100, the leading term is essentially exact for all n <= 20
    // (relative error < 1e-22 even for n=20)
    auto ref_values = get_reference_values();

    for (const auto& ref : ref_values) {
        if (ref.T == 100.0) {
            double computed = boys_asymptotic(ref.n, ref.T);
            double rel_err = relative_error(computed, ref.value);

            EXPECT_LT(rel_err, 1e-14)
                << "F_" << ref.n << "(" << ref.T << "): "
                << "computed = " << computed
                << ", reference = " << ref.value
                << ", relative error = " << rel_err;
        }
    }
}

TEST(BoysAsymptoticTest, T500_VeryLargeT) {
    // At T=500, the leading term is exact to machine precision
    auto ref_values = get_reference_values();

    for (const auto& ref : ref_values) {
        if (ref.T == 500.0) {
            double computed = boys_asymptotic(ref.n, ref.T);
            double rel_err = relative_error(computed, ref.value);

            EXPECT_LT(rel_err, 1e-14)
                << "F_" << ref.n << "(" << ref.T << "): "
                << "computed = " << computed
                << ", reference = " << ref.value
                << ", relative error = " << rel_err;
        }
    }
}

TEST(BoysAsymptoticTest, T50_LowOrders) {
    // At T=50, the leading term is accurate for low orders (n <= 5)
    auto ref_values = get_reference_values();

    for (const auto& ref : ref_values) {
        if (ref.T == 50.0 && ref.n <= 5) {
            double computed = boys_asymptotic(ref.n, ref.T);
            double rel_err = relative_error(computed, ref.value);

            EXPECT_LT(rel_err, 1e-14)
                << "F_" << ref.n << "(" << ref.T << "): "
                << "computed = " << computed
                << ", reference = " << ref.value
                << ", relative error = " << rel_err;
        }
    }
}

TEST(BoysAsymptoticTest, T30_F0) {
    // At T=30, the leading term for F_0 has relative error ≈ erfc(√30) ≈ 9.5e-15
    double computed = boys_asymptotic(0, 30.0);
    double ref = boys_reference_erf(0, 30.0);
    double rel_err = relative_error(computed, ref);

    EXPECT_LT(rel_err, 1e-13)
        << "F_0(30): computed = " << computed
        << ", reference = " << ref
        << ", relative error = " << rel_err;
}

TEST(BoysAsymptoticTest, LeadingTermDegradation) {
    // Verify that the leading term approximation degrades for high n at moderate T.
    // This is expected mathematical behavior, not a bug.
    // At T=30, n=10, the relative error should be around 1e-5.
    double computed = boys_asymptotic(10, 30.0);
    double ref = boys_reference_erf(10, 30.0);
    double rel_err = relative_error(computed, ref);

    // Error should be noticeable but not catastrophic
    EXPECT_LT(rel_err, 1e-3)
        << "F_10(30): leading term degradation beyond expected range";
    EXPECT_GT(rel_err, 1e-8)
        << "F_10(30): unexpectedly accurate (expected some degradation)";
}

TEST(BoysAsymptoticTest, MonotonicDecrease) {
    // F_n(T) should decrease with increasing n for fixed T
    const double T = 50.0;
    double prev_value = boys_asymptotic(0, T);

    for (int n = 1; n <= 20; ++n) {
        double curr_value = boys_asymptotic(n, T);
        EXPECT_LT(curr_value, prev_value)
            << "F_" << n << "(" << T << ") should be less than F_" << (n-1) << "(" << T << ")";
        prev_value = curr_value;
    }
}

TEST(BoysAsymptoticTest, DecreaseWithT) {
    // F_n(T) should decrease with increasing T for fixed n
    const int n = 5;
    std::vector<double> T_values = {30.0, 50.0, 100.0, 200.0, 500.0};

    double prev_value = boys_asymptotic(n, T_values[0]);
    for (size_t i = 1; i < T_values.size(); ++i) {
        double curr_value = boys_asymptotic(n, T_values[i]);
        EXPECT_LT(curr_value, prev_value)
            << "F_" << n << "(" << T_values[i] << ") should be less than "
            << "F_" << n << "(" << T_values[i-1] << ")";
        prev_value = curr_value;
    }
}

// =============================================================================
// Boys Asymptotic Array Tests
//
// boys_asymptotic_array uses F_0 from the leading term and then upward
// recursion: F_n = ((2n-1)*F_{n-1} - exp(-T)) / (2T).
//
// The recursion preserves the relative error from F_0 (~erfc(√T)), making
// the array much more accurate than per-n leading terms for high n.
// =============================================================================

TEST(BoysAsymptoticArrayTest, T30_AllOrders) {
    // Array evaluation at T=30: accurate to ~1e-14 for ALL orders
    // because the recursion preserves the F_0 error (~erfc(√30) ≈ 9.5e-15)
    const double T = 30.0;
    const int n_max = 20;
    std::vector<double> result(n_max + 1);
    auto ref_values = get_reference_values();

    boys_asymptotic_array(n_max, T, result.data());

    for (const auto& ref : ref_values) {
        if (ref.T == T && ref.n <= n_max) {
            double rel_err = relative_error(result[ref.n], ref.value);

            EXPECT_LT(rel_err, 1e-13)
                << "Array F_" << ref.n << "(" << T << "): "
                << "computed = " << result[ref.n]
                << ", reference = " << ref.value
                << ", relative error = " << rel_err;
        }
    }
}

TEST(BoysAsymptoticArrayTest, T50_AllOrders) {
    // Array at T=50: relative error < 1e-22 for all orders
    const double T = 50.0;
    const int n_max = 20;
    std::vector<double> result(n_max + 1);
    auto ref_values = get_reference_values();

    boys_asymptotic_array(n_max, T, result.data());

    for (const auto& ref : ref_values) {
        if (ref.T == T && ref.n <= n_max) {
            double rel_err = relative_error(result[ref.n], ref.value);

            EXPECT_LT(rel_err, 1e-13)
                << "Array F_" << ref.n << "(" << T << "): "
                << "computed = " << result[ref.n]
                << ", reference = " << ref.value
                << ", relative error = " << rel_err;
        }
    }
}

TEST(BoysAsymptoticArrayTest, T100_AllOrders) {
    // Array at T=100: essentially exact
    const double T = 100.0;
    const int n_max = 20;
    std::vector<double> result(n_max + 1);
    auto ref_values = get_reference_values();

    boys_asymptotic_array(n_max, T, result.data());

    for (const auto& ref : ref_values) {
        if (ref.T == T && ref.n <= n_max) {
            double rel_err = relative_error(result[ref.n], ref.value);

            EXPECT_LT(rel_err, 1e-13)
                << "Array F_" << ref.n << "(" << T << "): "
                << "computed = " << result[ref.n]
                << ", reference = " << ref.value
                << ", relative error = " << rel_err;
        }
    }
}

TEST(BoysAsymptoticArrayTest, MatchesSingleForF0) {
    // F_0 from array must exactly match boys_asymptotic(0, T) since
    // the array starts by calling boys_asymptotic(0, T).
    const std::vector<double> T_values = {30.0, 50.0, 100.0, 500.0};

    for (double T : T_values) {
        double result;
        boys_asymptotic_array(0, T, &result);
        double expected = boys_asymptotic(0, T);
        EXPECT_DOUBLE_EQ(result, expected)
            << "F_0(" << T << ") array vs single mismatch";
    }
}

TEST(BoysAsymptoticArrayTest, SingleOrder) {
    // Test array evaluation with n_max = 0 (only F_0)
    const double T = 50.0;
    double result;

    boys_asymptotic_array(0, T, &result);

    double expected = boys_asymptotic(0, T);
    EXPECT_DOUBLE_EQ(result, expected);
}

TEST(BoysAsymptoticArrayTest, MonotonicDecreaseInArray) {
    // Values in the array should be monotonically decreasing
    const double T = 50.0;
    const int n_max = 20;
    std::vector<double> result(n_max + 1);

    boys_asymptotic_array(n_max, T, result.data());

    for (int n = 1; n <= n_max; ++n) {
        EXPECT_LT(result[n], result[n-1])
            << "F_" << n << "(" << T << ") should be less than F_" << (n-1) << "(" << T << ")";
    }
}

// =============================================================================
// Cross-validation Tests
// =============================================================================

TEST(BoysCrossValidationTest, ArrayVsErfReference) {
    // Validate the array function against the independent erf-based reference.
    // This is the primary correctness test.
    const std::vector<double> T_values = {30.0, 50.0, 100.0, 500.0};
    const int n_max = 20;

    for (double T : T_values) {
        std::vector<double> result_array(n_max + 1);
        boys_asymptotic_array(n_max, T, result_array.data());

        for (int n = 0; n <= n_max; ++n) {
            double ref = boys_reference_erf(n, T);
            double rel_err = relative_error(result_array[n], ref);

            EXPECT_LT(rel_err, 1e-13)
                << "T = " << T << ", n = " << n << ": "
                << "array = " << result_array[n]
                << ", erf_ref = " << ref
                << ", relative error = " << rel_err;
        }
    }
}

TEST(BoysCrossValidationTest, SingleVsErfForLargeT) {
    // For large T, per-n leading term should agree with erf reference
    const std::vector<double> T_values = {100.0, 500.0};
    const int n_max = 20;

    for (double T : T_values) {
        for (int n = 0; n <= n_max; ++n) {
            double computed = boys_asymptotic(n, T);
            double ref = boys_reference_erf(n, T);
            double rel_err = relative_error(computed, ref);

            EXPECT_LT(rel_err, 1e-14)
                << "T = " << T << ", n = " << n << ": "
                << "asymptotic = " << computed
                << ", erf_ref = " << ref
                << ", relative error = " << rel_err;
        }
    }
}

// =============================================================================
// Threshold Tests
// =============================================================================

TEST(BoysThresholdTest, ThresholdValue) {
    EXPECT_DOUBLE_EQ(BOYS_ASYMPTOTIC_THRESHOLD, 30.0);
}

TEST(BoysThresholdTest, ArrayAccuracyAtThreshold) {
    // The array function should be accurate to < 1e-13 at the threshold
    const double T = BOYS_ASYMPTOTIC_THRESHOLD;
    const int n_max = 20;
    std::vector<double> result(n_max + 1);
    boys_asymptotic_array(n_max, T, result.data());

    for (int n = 0; n <= n_max; ++n) {
        double ref = boys_reference_erf(n, T);
        double rel_err = relative_error(result[n], ref);

        EXPECT_LT(rel_err, 1e-13)
            << "At threshold T = " << T << ", n = " << n << ": "
            << "array = " << result[n]
            << ", erf_ref = " << ref
            << ", relative error = " << rel_err;
    }
}

// =============================================================================
// Analytical Formula Tests
// =============================================================================

TEST(BoysAnalyticalTest, F0_AsymptoticFormula) {
    // For large T, F_0(T) ≈ 0.5 * sqrt(π / T)
    // Since boys_asymptotic IS the leading term, this should match exactly.
    const std::vector<double> T_values = {30.0, 50.0, 100.0, 200.0, 500.0, 1000.0};

    for (double T : T_values) {
        double computed = boys_asymptotic(0, T);
        double analytical = 0.5 * std::sqrt(constants::PI / T);

        // boys_asymptotic(0, T) computes exactly this formula
        EXPECT_DOUBLE_EQ(computed, analytical)
            << "F_0(" << T << "): boys_asymptotic should equal sqrt(pi)/(2*sqrt(T))";
    }
}

TEST(BoysAnalyticalTest, LeadingTermDominance) {
    // For very large T, the leading term should match the exact value closely
    const double T = 500.0;

    for (int n = 0; n <= 10; ++n) {
        double computed = boys_asymptotic(n, T);
        double ref = boys_reference_erf(n, T);
        double rel_err = relative_error(computed, ref);

        // For T = 500, correction terms are negligible
        EXPECT_LT(rel_err, 1e-14)
            << "n = " << n << ": leading term should dominate at T = 500";
    }
}

// =============================================================================
// Chebyshev Interpolation Reference Values
//
// Computed using mpmath with 50-digit precision.
// F_n(T) via Taylor series for T < 25 and erf+recursion for T >= 25.
// =============================================================================

namespace {

struct ChebyshevReferenceValue {
    double T;
    int n;
    double value;
};

std::vector<ChebyshevReferenceValue> get_chebyshev_reference_values() {
    std::vector<ChebyshevReferenceValue> values;

    // T = 0.0: F_n(0) = 1/(2n+1) (exact)
    values.push_back({0.0, 0, 1.00000000000000000e+00});
    values.push_back({0.0, 1, 3.33333333333333315e-01});
    values.push_back({0.0, 2, 2.00000000000000011e-01});
    values.push_back({0.0, 5, 9.09090909090909116e-02});
    values.push_back({0.0, 10, 4.76190476190476164e-02});
    values.push_back({0.0, 15, 3.22580645161290314e-02});
    values.push_back({0.0, 20, 2.43902439024390252e-02});
    values.push_back({0.0, 25, 1.96078431372549017e-02});
    values.push_back({0.0, 30, 1.63934426229508205e-02});

    // T = 0.01
    values.push_back({0.01, 0, 9.96676642903363552e-01});
    values.push_back({0.01, 1, 3.31340457709772440e-01});
    values.push_back({0.01, 2, 1.98576969007464771e-01});
    values.push_back({0.01, 5, 9.01431836911621015e-02});
    values.push_back({0.01, 10, 4.71862588518534368e-02});
    values.push_back({0.01, 15, 3.19564582906862668e-02});
    values.push_back({0.01, 20, 2.41587933364030491e-02});
    values.push_back({0.01, 25, 1.94200700659528727e-02});
    values.push_back({0.01, 30, 1.62354792134546330e-02});

    // T = 0.1
    values.push_back({0.1, 0, 9.67643312635591779e-01});
    values.push_back({0.1, 1, 3.14029472998161308e-01});
    values.push_back({0.1, 2, 1.86255004792621470e-01});
    values.push_back({0.1, 5, 8.35405280181414772e-02});
    values.push_back({0.1, 10, 4.34651897241010859e-02});
    values.push_back({0.1, 15, 2.93662189611291966e-02});
    values.push_back({0.1, 20, 2.21723109447862013e-02});
    values.push_back({0.1, 25, 1.78091050758185931e-02});
    values.push_back({0.1, 30, 1.48806357819122759e-02});

    // T = 0.5
    values.push_back({0.5, 0, 8.55624391892148783e-01});
    values.push_back({0.5, 1, 2.49093732179515387e-01});
    values.push_back({0.5, 2, 1.40750536825912709e-01});
    values.push_back({0.5, 5, 5.96809411402653353e-02});
    values.push_back({0.5, 10, 3.01903263749237032e-02});
    values.push_back({0.5, 15, 2.01758089445645272e-02});
    values.push_back({0.5, 20, 1.51452752306940053e-02});
    values.push_back({0.5, 25, 1.21213023526272820e-02});
    values.push_back({0.5, 30, 1.01034178459644963e-02});

    // T = 1.0
    values.push_back({1.0, 0, 7.46824132812426988e-01});
    values.push_back({1.0, 1, 1.89472345820492355e-01});
    values.push_back({1.0, 2, 1.00268798145017365e-01});
    values.push_back({1.0, 5, 3.93648645134841574e-02});
    values.push_back({1.0, 10, 1.91729360913146310e-02});
    values.push_back({1.0, 15, 1.26297350206479378e-02});
    values.push_back({1.0, 20, 9.40937371771591834e-03});
    values.push_back({1.0, 25, 7.49578091022965604e-03});
    values.push_back({1.0, 30, 6.22833680700876375e-03});

    // T = 5.0
    values.push_back({5.0, 0, 3.95712309610513568e-01});
    values.push_back({5.0, 1, 3.88974362611428093e-02});
    values.push_back({5.0, 2, 1.09954361784342959e-02});
    values.push_back({5.0, 5, 1.75886180543817993e-03});
    values.push_back({5.0, 10, 5.47217440837332258e-04});
    values.push_back({5.0, 15, 3.08839225937863025e-04});
    values.push_back({5.0, 20, 2.13316613874671950e-04});
    values.push_back({5.0, 25, 1.62532716063584358e-04});
    values.push_back({5.0, 30, 1.31159036350080114e-04});

    // T = 10.0
    values.push_back({10.0, 0, 2.80247390506642713e-01});
    values.push_back({10.0, 1, 1.40100995288440135e-02});
    values.push_back({10.0, 2, 2.09924493283847758e-03});
    values.push_back({10.0, 5, 7.90087498758553392e-05});
    values.push_back({10.0, 10, 8.57837826217344527e-06});
    values.push_back({10.0, 15, 3.39872533343623558e-06});
    values.push_back({10.0, 20, 2.01315124872491560e-06});
    values.push_back({10.0, 25, 1.41259689012731327e-06});
    values.push_back({10.0, 30, 1.08365731738228075e-06});

    // T = 20.0
    values.push_back({20.0, 0, 1.98166364829973657e-01});
    values.push_back({20.0, 1, 4.95415906922050051e-03});
    values.push_back({20.0, 2, 3.71561878662696979e-04});
    values.push_back({20.0, 5, 1.82871596976517756e-06});
    values.push_back({20.0, 10, 1.22814554229386189e-08});
    values.push_back({20.0, 15, 9.95035642507146929e-10});
    values.push_back({20.0, 20, 2.79599063515302481e-10});
    values.push_back({20.0, 25, 1.36918179423940268e-10});
    values.push_back({20.0, 30, 8.64072586432773815e-11});

    // T = 29.9 (near boundary with asymptotic regime)
    values.push_back({29.9, 0, 1.62072505699125402e-01});
    values.push_back({29.9, 1, 2.71024257021775915e-03});
    values.push_back({29.9, 2, 1.35965346330265210e-04});
    values.push_back({29.9, 5, 2.00278915338915854e-07});
    values.push_back({29.9, 10, 1.81448418466444623e-10});
    values.push_back({29.9, 15, 2.24017960376427483e-12});
    values.push_back({29.9, 20, 1.47162639383066248e-13});
    values.push_back({29.9, 25, 2.94591761840747583e-14});
    values.push_back({29.9, 30, 1.13761667017536128e-14});

    // T = 35.5 (near upper boundary of Chebyshev range)
    values.push_back({35.5, 0, 1.48741023012108814e-01});
    values.push_back({35.5, 1, 2.09494398608603416e-03});
    values.push_back({35.5, 2, 8.85187599754608533e-05});
    values.push_back({35.5, 5, 7.79059860973088375e-08});
    values.push_back({35.5, 10, 2.99163633220098765e-11});
    values.push_back({35.5, 15, 1.56762269444039272e-13});
    values.push_back({35.5, 20, 4.47810538366806728e-15});
    values.push_back({35.5, 25, 4.39414876182941976e-16});
    values.push_back({35.5, 30, 1.03446966540751426e-16});

    return values;
}

} // anonymous namespace

// =============================================================================
// Chebyshev Single Value Tests
// =============================================================================

TEST(BoysChebyshevTest, T0_ExactValues) {
    // F_n(0) = 1/(2n+1) exactly
    for (int n = 0; n <= 30; ++n) {
        double computed = boys_chebyshev(n, 0.0);
        double expected = 1.0 / (2 * n + 1);
        double rel_err = relative_error(computed, expected);

        EXPECT_LT(rel_err, 1e-14)
            << "F_" << n << "(0): computed = " << computed
            << ", expected = " << expected
            << ", relative error = " << rel_err;
    }
}

TEST(BoysChebyshevTest, T001_AllOrders) {
    auto ref_values = get_chebyshev_reference_values();

    for (const auto& ref : ref_values) {
        if (ref.T == 0.01) {
            double computed = boys_chebyshev(ref.n, ref.T);
            double rel_err = relative_error(computed, ref.value);

            EXPECT_LT(rel_err, 1e-14)
                << "F_" << ref.n << "(" << ref.T << "): "
                << "computed = " << computed
                << ", reference = " << ref.value
                << ", relative error = " << rel_err;
        }
    }
}

TEST(BoysChebyshevTest, T01_AllOrders) {
    auto ref_values = get_chebyshev_reference_values();

    for (const auto& ref : ref_values) {
        if (ref.T == 0.1) {
            double computed = boys_chebyshev(ref.n, ref.T);
            double rel_err = relative_error(computed, ref.value);

            EXPECT_LT(rel_err, 1e-14)
                << "F_" << ref.n << "(" << ref.T << "): "
                << "computed = " << computed
                << ", reference = " << ref.value
                << ", relative error = " << rel_err;
        }
    }
}

TEST(BoysChebyshevTest, T05_AllOrders) {
    auto ref_values = get_chebyshev_reference_values();

    for (const auto& ref : ref_values) {
        if (ref.T == 0.5) {
            double computed = boys_chebyshev(ref.n, ref.T);
            double rel_err = relative_error(computed, ref.value);

            EXPECT_LT(rel_err, 1e-14)
                << "F_" << ref.n << "(" << ref.T << "): "
                << "computed = " << computed
                << ", reference = " << ref.value
                << ", relative error = " << rel_err;
        }
    }
}

TEST(BoysChebyshevTest, T1_AllOrders) {
    auto ref_values = get_chebyshev_reference_values();

    for (const auto& ref : ref_values) {
        if (ref.T == 1.0) {
            double computed = boys_chebyshev(ref.n, ref.T);
            double rel_err = relative_error(computed, ref.value);

            EXPECT_LT(rel_err, 1e-14)
                << "F_" << ref.n << "(" << ref.T << "): "
                << "computed = " << computed
                << ", reference = " << ref.value
                << ", relative error = " << rel_err;
        }
    }
}

TEST(BoysChebyshevTest, T5_AllOrders) {
    auto ref_values = get_chebyshev_reference_values();

    for (const auto& ref : ref_values) {
        if (ref.T == 5.0) {
            double computed = boys_chebyshev(ref.n, ref.T);
            double rel_err = relative_error(computed, ref.value);

            EXPECT_LT(rel_err, 1e-14)
                << "F_" << ref.n << "(" << ref.T << "): "
                << "computed = " << computed
                << ", reference = " << ref.value
                << ", relative error = " << rel_err;
        }
    }
}

TEST(BoysChebyshevTest, T10_AllOrders) {
    auto ref_values = get_chebyshev_reference_values();

    for (const auto& ref : ref_values) {
        if (ref.T == 10.0) {
            double computed = boys_chebyshev(ref.n, ref.T);
            double rel_err = relative_error(computed, ref.value);

            EXPECT_LT(rel_err, 1e-14)
                << "F_" << ref.n << "(" << ref.T << "): "
                << "computed = " << computed
                << ", reference = " << ref.value
                << ", relative error = " << rel_err;
        }
    }
}

TEST(BoysChebyshevTest, T20_AllOrders) {
    auto ref_values = get_chebyshev_reference_values();

    for (const auto& ref : ref_values) {
        if (ref.T == 20.0) {
            double computed = boys_chebyshev(ref.n, ref.T);
            double rel_err = relative_error(computed, ref.value);

            EXPECT_LT(rel_err, 1e-14)
                << "F_" << ref.n << "(" << ref.T << "): "
                << "computed = " << computed
                << ", reference = " << ref.value
                << ", relative error = " << rel_err;
        }
    }
}

TEST(BoysChebyshevTest, T299_NearBoundary) {
    // T = 29.9 is near the transition to asymptotic regime
    auto ref_values = get_chebyshev_reference_values();

    for (const auto& ref : ref_values) {
        if (ref.T == 29.9) {
            double computed = boys_chebyshev(ref.n, ref.T);
            double rel_err = relative_error(computed, ref.value);

            EXPECT_LT(rel_err, 1e-14)
                << "F_" << ref.n << "(" << ref.T << "): "
                << "computed = " << computed
                << ", reference = " << ref.value
                << ", relative error = " << rel_err;
        }
    }
}

TEST(BoysChebyshevTest, T355_NearTMax) {
    // T = 35.5 is near the upper boundary of the Chebyshev range
    auto ref_values = get_chebyshev_reference_values();

    for (const auto& ref : ref_values) {
        if (ref.T == 35.5) {
            double computed = boys_chebyshev(ref.n, ref.T);
            double rel_err = relative_error(computed, ref.value);

            EXPECT_LT(rel_err, 1e-14)
                << "F_" << ref.n << "(" << ref.T << "): "
                << "computed = " << computed
                << ", reference = " << ref.value
                << ", relative error = " << rel_err;
        }
    }
}

TEST(BoysChebyshevTest, MonotonicDecreaseInN) {
    // F_n(T) should decrease with increasing n for fixed T > 0
    const std::vector<double> T_values = {0.1, 1.0, 5.0, 10.0, 20.0, 35.0};

    for (double T : T_values) {
        double prev = boys_chebyshev(0, T);
        for (int n = 1; n <= 30; ++n) {
            double curr = boys_chebyshev(n, T);
            EXPECT_LT(curr, prev)
                << "F_" << n << "(" << T << ") = " << curr
                << " should be less than F_" << (n - 1) << "(" << T << ") = " << prev;
            prev = curr;
        }
    }
}

TEST(BoysChebyshevTest, DecreaseWithT) {
    // F_n(T) should decrease with increasing T for fixed n > 0
    for (int n = 0; n <= 30; n += 5) {
        double prev = boys_chebyshev(n, 0.1);
        std::vector<double> T_values = {1.0, 5.0, 10.0, 20.0, 35.0};
        for (double T : T_values) {
            double curr = boys_chebyshev(n, T);
            EXPECT_LT(curr, prev)
                << "F_" << n << "(" << T << ") = " << curr
                << " should be less than F_" << n << "(" << (T - 1) << ") = " << prev;
            prev = curr;
        }
    }
}

TEST(BoysChebyshevTest, IntervalBoundaries) {
    // Test at interval boundaries to check for discontinuities.
    // At T = k (for integer k), the evaluation could use either interval k-1 or k.
    // Check that results are continuous across boundaries.
    for (int k = 1; k < 36; ++k) {
        double T_below = k - 1e-10;
        double T_at = static_cast<double>(k);

        for (int n = 0; n <= 30; n += 10) {
            double val_below = boys_chebyshev(n, T_below);
            double val_at = boys_chebyshev(n, T_at);

            // Results should be very close (essentially continuous)
            double rel_diff = relative_error(val_below, val_at);
            EXPECT_LT(rel_diff, 1e-8)
                << "Discontinuity at T = " << k << " for n = " << n
                << ": below = " << val_below << ", at = " << val_at;
        }
    }
}

// =============================================================================
// Chebyshev Array Tests
// =============================================================================

TEST(BoysChebyshevArrayTest, MatchesSingleEvaluation) {
    // Array evaluation should match single evaluation exactly
    const std::vector<double> T_values = {0.0, 0.01, 0.5, 1.0, 5.0, 10.0, 20.0, 35.0};
    const int n_max = 30;

    for (double T : T_values) {
        std::vector<double> result(n_max + 1);
        boys_chebyshev_array(n_max, T, result.data());

        for (int n = 0; n <= n_max; ++n) {
            double single = boys_chebyshev(n, T);
            EXPECT_DOUBLE_EQ(result[n], single)
                << "Array vs single mismatch at T = " << T << ", n = " << n;
        }
    }
}

TEST(BoysChebyshevArrayTest, AllOrdersAccuracy) {
    // Comprehensive accuracy check: all T values, all orders
    auto ref_values = get_chebyshev_reference_values();

    // Group by T value
    std::vector<double> T_values = {0.0, 0.01, 0.1, 0.5, 1.0, 5.0, 10.0, 20.0, 29.9, 35.5};

    for (double T : T_values) {
        if (T >= BOYS_CHEBYSHEV_T_MAX) continue;

        std::vector<double> result(31);
        boys_chebyshev_array(30, T, result.data());

        for (const auto& ref : ref_values) {
            if (ref.T == T) {
                double rel_err = relative_error(result[ref.n], ref.value);
                EXPECT_LT(rel_err, 1e-14)
                    << "Array F_" << ref.n << "(" << T << "): "
                    << "computed = " << result[ref.n]
                    << ", reference = " << ref.value
                    << ", relative error = " << rel_err;
            }
        }
    }
}

TEST(BoysChebyshevArrayTest, SingleOrder) {
    // Array with n_max = 0 should give just F_0
    double result;
    boys_chebyshev_array(0, 5.0, &result);
    double expected = boys_chebyshev(0, 5.0);
    EXPECT_DOUBLE_EQ(result, expected);
}

// =============================================================================
// Overlap Region Tests (Chebyshev vs Asymptotic)
// =============================================================================

TEST(BoysOverlapTest, ChebyshevVsAsymptoticArray) {
    // In the overlap region (T ≈ 30-36), both methods should agree.
    // boys_asymptotic_array uses recursion from F_0 and should be very
    // accurate for large T.
    const std::vector<double> T_overlap = {30.0, 31.0, 32.0, 33.0, 34.0, 35.0, 35.9};
    const int n_max = 30;

    for (double T : T_overlap) {
        std::vector<double> cheb_result(n_max + 1);
        std::vector<double> asym_result(n_max + 1);

        boys_chebyshev_array(n_max, T, cheb_result.data());
        boys_asymptotic_array(n_max, T, asym_result.data());

        for (int n = 0; n <= n_max; ++n) {
            double rel_err = relative_error(cheb_result[n], asym_result[n]);

            // Both should agree to ~1e-14 since both are accurate here
            EXPECT_LT(rel_err, 1e-12)
                << "Overlap T = " << T << ", n = " << n << ": "
                << "chebyshev = " << cheb_result[n]
                << ", asymptotic = " << asym_result[n]
                << ", relative error = " << rel_err;
        }
    }
}

TEST(BoysOverlapTest, ChebyshevVsAsymptoticSingle_F0) {
    // F_0 from both methods should agree closely in the overlap region
    const std::vector<double> T_overlap = {30.0, 32.0, 35.0};

    for (double T : T_overlap) {
        double cheb = boys_chebyshev(0, T);
        double asym = boys_asymptotic(0, T);
        double rel_err = relative_error(cheb, asym);

        EXPECT_LT(rel_err, 1e-13)
            << "F_0(" << T << "): chebyshev = " << cheb
            << ", asymptotic = " << asym
            << ", relative error = " << rel_err;
    }
}

// =============================================================================
// Chebyshev Constants Tests
// =============================================================================

TEST(BoysChebyshevConstantsTest, TMaxValue) {
    EXPECT_DOUBLE_EQ(BOYS_CHEBYSHEV_T_MAX, 36.0);
}

TEST(BoysChebyshevConstantsTest, OverlapWithAsymptotic) {
    // The Chebyshev range [0, 36) should overlap with the asymptotic regime [30, ∞)
    EXPECT_GT(BOYS_CHEBYSHEV_T_MAX, BOYS_ASYMPTOTIC_THRESHOLD);
}

// =============================================================================
// Unified Evaluation Reference Values
//
// Reference values computed using mpmath with 50-digit precision.
// Tests for boys_evaluate_array() and boys_evaluate().
// =============================================================================

namespace {

struct EvaluateReferenceValue {
    double T;
    int n;
    double value;
};

std::vector<EvaluateReferenceValue> get_evaluate_reference_values() {
    std::vector<EvaluateReferenceValue> values;

    // T = 0.0: F_n(0) = 1/(2n+1)
    values.push_back({0.0, 0, 1.00000000000000000e+00});
    values.push_back({0.0, 1, 3.33333333333333315e-01});
    values.push_back({0.0, 5, 9.09090909090909116e-02});
    values.push_back({0.0, 10, 4.76190476190476164e-02});
    values.push_back({0.0, 15, 3.22580645161290314e-02});
    values.push_back({0.0, 20, 2.43902439024390252e-02});

    // T = 0.001 (very small T — Taylor regime)
    values.push_back({0.001, 0, 9.99666766642861804e-01});
    values.push_back({0.001, 1, 3.33133404743390038e-01});
    values.push_back({0.001, 2, 1.99857198397550090e-01});
    values.push_back({0.001, 5, 9.08322011556994408e-02});
    values.push_back({0.001, 10, 4.75755893520066475e-02});
    values.push_back({0.001, 15, 3.22277757670368509e-02});
    values.push_back({0.001, 20, 2.43669991960513976e-02});

    // T = 0.5 (small T — Chebyshev + downward recursion)
    values.push_back({0.5, 0, 8.55624391892148783e-01});
    values.push_back({0.5, 1, 2.49093732179515387e-01});
    values.push_back({0.5, 2, 1.40750536825912709e-01});
    values.push_back({0.5, 5, 5.96809411402653353e-02});
    values.push_back({0.5, 10, 3.01903263749237032e-02});
    values.push_back({0.5, 15, 2.01758089445645272e-02});
    values.push_back({0.5, 20, 1.51452752306940053e-02});

    // T = 5.0 (medium T)
    values.push_back({5.0, 0, 3.95712309610513568e-01});
    values.push_back({5.0, 1, 3.88974362611428093e-02});
    values.push_back({5.0, 2, 1.09954361784342959e-02});
    values.push_back({5.0, 5, 1.75886180543817993e-03});
    values.push_back({5.0, 10, 5.47217440837332258e-04});
    values.push_back({5.0, 15, 3.08839225937863025e-04});
    values.push_back({5.0, 20, 2.13316613874671950e-04});

    // T = 15.0 (larger medium T)
    values.push_back({15.0, 0, 2.28822798329737342e-01});
    values.push_back({15.0, 1, 7.62741641424722824e-03});
    values.push_back({15.0, 2, 7.62731444680706102e-04});
    values.push_back({15.0, 5, 8.88456398197198738e-06});
    values.push_back({15.0, 10, 2.30377454811276462e-07});
    values.push_back({15.0, 15, 4.76509345539452272e-08});
    values.push_back({15.0, 20, 2.14595280391253253e-08});

    // T = 30.0 (boundary — handled by either Chebyshev or asymptotic)
    values.push_back({30.0, 0, 1.61802159379640070e-01});
    values.push_back({30.0, 1, 2.69670265632577502e-03});
    values.push_back({30.0, 2, 1.34835132814729137e-04});
    values.push_back({30.0, 5, 1.96634566867309196e-07});
    values.push_back({30.0, 10, 1.75197494140663680e-10});
    values.push_back({30.0, 15, 2.12735263635999297e-12});
    values.push_back({30.0, 20, 1.37585444267909315e-13});

    return values;
}

} // anonymous namespace

// =============================================================================
// Unified boys_evaluate_array() Tests
// =============================================================================

TEST(BoysEvaluateArrayTest, T0_ExactFormula) {
    // T = 0: F_n(0) = 1/(2n+1)
    const int n_max = 20;
    std::vector<double> result(n_max + 1);

    boys_evaluate_array(n_max, 0.0, result.data());

    for (int n = 0; n <= n_max; ++n) {
        double expected = 1.0 / (2 * n + 1);
        EXPECT_DOUBLE_EQ(result[n], expected)
            << "F_" << n << "(0) should be exactly 1/(2n+1)";
    }
}

TEST(BoysEvaluateArrayTest, T0001_VerySmallT) {
    // T = 0.001: Taylor or Chebyshev regime
    auto ref_values = get_evaluate_reference_values();
    const int n_max = 20;
    std::vector<double> result(n_max + 1);

    boys_evaluate_array(n_max, 0.001, result.data());

    for (const auto& ref : ref_values) {
        if (ref.T == 0.001 && ref.n <= n_max) {
            double rel_err = relative_error(result[ref.n], ref.value);
            EXPECT_LT(rel_err, 1e-14)
                << "F_" << ref.n << "(0.001): computed = " << result[ref.n]
                << ", reference = " << ref.value
                << ", relative error = " << rel_err;
        }
    }
}

TEST(BoysEvaluateArrayTest, T05_SmallT) {
    auto ref_values = get_evaluate_reference_values();
    const int n_max = 20;
    std::vector<double> result(n_max + 1);

    boys_evaluate_array(n_max, 0.5, result.data());

    for (const auto& ref : ref_values) {
        if (ref.T == 0.5 && ref.n <= n_max) {
            double rel_err = relative_error(result[ref.n], ref.value);
            EXPECT_LT(rel_err, 1e-14)
                << "F_" << ref.n << "(0.5): computed = " << result[ref.n]
                << ", reference = " << ref.value
                << ", relative error = " << rel_err;
        }
    }
}

TEST(BoysEvaluateArrayTest, T5_MediumT) {
    auto ref_values = get_evaluate_reference_values();
    const int n_max = 20;
    std::vector<double> result(n_max + 1);

    boys_evaluate_array(n_max, 5.0, result.data());

    for (const auto& ref : ref_values) {
        if (ref.T == 5.0 && ref.n <= n_max) {
            double rel_err = relative_error(result[ref.n], ref.value);
            EXPECT_LT(rel_err, 1e-14)
                << "F_" << ref.n << "(5.0): computed = " << result[ref.n]
                << ", reference = " << ref.value
                << ", relative error = " << rel_err;
        }
    }
}

TEST(BoysEvaluateArrayTest, T15_MediumT) {
    auto ref_values = get_evaluate_reference_values();
    const int n_max = 20;
    std::vector<double> result(n_max + 1);

    boys_evaluate_array(n_max, 15.0, result.data());

    for (const auto& ref : ref_values) {
        if (ref.T == 15.0 && ref.n <= n_max) {
            double rel_err = relative_error(result[ref.n], ref.value);
            EXPECT_LT(rel_err, 1e-14)
                << "F_" << ref.n << "(15.0): computed = " << result[ref.n]
                << ", reference = " << ref.value
                << ", relative error = " << rel_err;
        }
    }
}

TEST(BoysEvaluateArrayTest, T30_Boundary) {
    // T = 30 is in the Chebyshev range [0, 36)
    auto ref_values = get_evaluate_reference_values();
    const int n_max = 20;
    std::vector<double> result(n_max + 1);

    boys_evaluate_array(n_max, 30.0, result.data());

    for (const auto& ref : ref_values) {
        if (ref.T == 30.0 && ref.n <= n_max) {
            double rel_err = relative_error(result[ref.n], ref.value);
            EXPECT_LT(rel_err, 1e-14)
                << "F_" << ref.n << "(30.0): computed = " << result[ref.n]
                << ", reference = " << ref.value
                << ", relative error = " << rel_err;
        }
    }
}

TEST(BoysEvaluateArrayTest, T50_LargeT) {
    // T = 50 is in the asymptotic regime
    const int n_max = 20;
    std::vector<double> result(n_max + 1);

    boys_evaluate_array(n_max, 50.0, result.data());

    // Use existing asymptotic reference values
    auto asym_ref = get_reference_values();
    for (const auto& ref : asym_ref) {
        if (ref.T == 50.0 && ref.n <= n_max) {
            double rel_err = relative_error(result[ref.n], ref.value);
            EXPECT_LT(rel_err, 1e-13)
                << "F_" << ref.n << "(50.0): computed = " << result[ref.n]
                << ", reference = " << ref.value
                << ", relative error = " << rel_err;
        }
    }
}

TEST(BoysEvaluateArrayTest, MonotonicDecreaseInN) {
    // F_n(T) should decrease with increasing n for all T > 0
    const std::vector<double> T_values = {0.001, 0.5, 5.0, 15.0, 30.0, 50.0};
    const int n_max = 20;

    for (double T : T_values) {
        std::vector<double> result(n_max + 1);
        boys_evaluate_array(n_max, T, result.data());

        for (int n = 1; n <= n_max; ++n) {
            EXPECT_LT(result[n], result[n - 1])
                << "F_" << n << "(" << T << ") should be less than F_"
                << (n - 1) << "(" << T << ")";
        }
    }
}

TEST(BoysEvaluateArrayTest, AllOrdersAccuracyFullRange) {
    // Comprehensive test: n=0..20 for all acceptance criteria T values
    auto ref_values = get_evaluate_reference_values();
    const int n_max = 20;

    std::vector<double> T_values = {0.0, 0.001, 0.5, 5.0, 15.0, 30.0};

    for (double T : T_values) {
        std::vector<double> result(n_max + 1);
        boys_evaluate_array(n_max, T, result.data());

        for (const auto& ref : ref_values) {
            if (ref.T == T && ref.n <= n_max) {
                double rel_err = relative_error(result[ref.n], ref.value);
                EXPECT_LT(rel_err, 1e-14)
                    << "evaluate_array F_" << ref.n << "(" << T << "): "
                    << "computed = " << result[ref.n]
                    << ", reference = " << ref.value
                    << ", relative error = " << rel_err;
            }
        }
    }
}

TEST(BoysEvaluateArrayTest, SingleOrder) {
    // n_max = 0 should give just F_0
    double result;
    boys_evaluate_array(0, 5.0, &result);

    double expected = boys_evaluate(0, 5.0);
    EXPECT_DOUBLE_EQ(result, expected);
}

TEST(BoysEvaluateArrayTest, VerySmallT) {
    // Very small T values use Taylor expansion
    const int n_max = 20;
    std::vector<double> result(n_max + 1);

    boys_evaluate_array(n_max, 1e-15, result.data());

    // Should be very close to 1/(2n+1)
    for (int n = 0; n <= n_max; ++n) {
        double expected = 1.0 / (2 * n + 1);
        double rel_err = relative_error(result[n], expected);
        EXPECT_LT(rel_err, 1e-14)
            << "F_" << n << "(1e-15) should be approximately 1/(2n+1)";
    }
}

// =============================================================================
// boys_evaluate() Single Value Tests
// =============================================================================

TEST(BoysEvaluateTest, T0) {
    for (int n = 0; n <= 20; ++n) {
        double computed = boys_evaluate(n, 0.0);
        double expected = 1.0 / (2 * n + 1);
        EXPECT_DOUBLE_EQ(computed, expected)
            << "F_" << n << "(0) = " << computed;
    }
}

TEST(BoysEvaluateTest, MatchesArrayForF0) {
    // boys_evaluate should match boys_evaluate_array for n=0
    const std::vector<double> T_values = {0.0, 0.001, 0.5, 5.0, 15.0, 30.0, 50.0};

    for (double T : T_values) {
        double single = boys_evaluate(0, T);
        double array_result;
        boys_evaluate_array(0, T, &array_result);
        EXPECT_DOUBLE_EQ(single, array_result)
            << "Mismatch at T = " << T;
    }
}

TEST(BoysEvaluateTest, AccuracyAcrossRegimes) {
    auto ref_values = get_evaluate_reference_values();

    for (const auto& ref : ref_values) {
        double computed = boys_evaluate(ref.n, ref.T);
        double rel_err = relative_error(computed, ref.value);

        EXPECT_LT(rel_err, 1e-14)
            << "boys_evaluate F_" << ref.n << "(" << ref.T << "): "
            << "computed = " << computed
            << ", reference = " << ref.value
            << ", relative error = " << rel_err;
    }
}

// =============================================================================
// Downward Recursion Stability Tests
// =============================================================================

TEST(BoysDownwardRecursionTest, StabilityAcrossT) {
    // Verify that boys_evaluate_array (which uses downward recursion for T < 36)
    // produces results consistent with independent Chebyshev evaluations
    const std::vector<double> T_values = {0.5, 1.0, 5.0, 10.0, 15.0, 20.0, 25.0, 29.0, 35.0};
    const int n_max = 20;

    for (double T : T_values) {
        std::vector<double> eval_result(n_max + 1);
        boys_evaluate_array(n_max, T, eval_result.data());

        // Compare against independent Chebyshev evaluation (no recursion)
        for (int n = 0; n <= n_max; ++n) {
            double cheb = boys_chebyshev(n, T);
            double rel_err = relative_error(eval_result[n], cheb);

            EXPECT_LT(rel_err, 1e-14)
                << "T = " << T << ", n = " << n << ": "
                << "evaluate_array = " << eval_result[n]
                << ", chebyshev = " << cheb
                << ", relative error = " << rel_err;
        }
    }
}

TEST(BoysDownwardRecursionTest, HighOrderAccuracy) {
    // Test with n_max = 30 (maximum) at various T values
    const std::vector<double> T_values = {1.0, 5.0, 15.0, 25.0, 35.0};
    const int n_max = 30;

    for (double T : T_values) {
        std::vector<double> result(n_max + 1);
        boys_evaluate_array(n_max, T, result.data());

        // All orders should be monotonically decreasing
        for (int n = 1; n <= n_max; ++n) {
            EXPECT_LT(result[n], result[n - 1])
                << "F_" << n << "(" << T << ") not monotonically decreasing";
        }

        // F_0 should match Chebyshev independently
        double cheb_F0 = boys_chebyshev(0, T);
        double rel_err = relative_error(result[0], cheb_F0);
        EXPECT_LT(rel_err, 1e-14)
            << "F_0(" << T << "): evaluate_array disagrees with chebyshev";
    }
}

TEST(BoysDownwardRecursionTest, RecursionFormula) {
    // Verify the recursion relation holds:
    // F_{n-1}(T) = (2T × F_n(T) + exp(-T)) / (2n - 1)
    const std::vector<double> T_values = {0.5, 5.0, 15.0, 25.0};
    const int n_max = 20;

    for (double T : T_values) {
        std::vector<double> result(n_max + 1);
        boys_evaluate_array(n_max, T, result.data());

        const double exp_neg_T = std::exp(-T);

        for (int n = 1; n <= n_max; ++n) {
            double from_recursion = (2.0 * T * result[n] + exp_neg_T) / (2 * n - 1);
            double rel_err = relative_error(from_recursion, result[n - 1]);

            EXPECT_LT(rel_err, 1e-13)
                << "Recursion relation violated at T = " << T << ", n = " << n;
        }
    }
}

TEST(BoysDownwardRecursionTest, EdgeCases) {
    // Test n_max = 0 for various T
    for (double T : {0.0, 0.001, 1.0, 15.0, 30.0, 50.0}) {
        double result;
        boys_evaluate_array(0, T, &result);

        // Should match boys_evaluate
        double expected = boys_evaluate(0, T);
        EXPECT_DOUBLE_EQ(result, expected)
            << "n_max=0 at T=" << T;
    }
}

// =============================================================================
// Batch Evaluation Tests
// =============================================================================

TEST(BoysBatchTest, MatchesSingleEvaluation) {
    // Batch evaluation should match single-value evaluation exactly
    const int n_max = 20;
    const std::vector<double> T_values = {0.0, 0.001, 0.5, 1.0, 5.0, 10.0,
                                          15.0, 20.0, 25.0, 30.0, 35.0, 50.0};
    const int n_T = static_cast<int>(T_values.size());

    std::vector<double> batch_result(n_T * (n_max + 1));
    boys_evaluate_batch(n_max, T_values.data(), n_T, batch_result.data());

    for (int i = 0; i < n_T; ++i) {
        std::vector<double> single_result(n_max + 1);
        boys_evaluate_array(n_max, T_values[i], single_result.data());

        for (int n = 0; n <= n_max; ++n) {
            EXPECT_DOUBLE_EQ(batch_result[i * (n_max + 1) + n], single_result[n])
                << "Mismatch at T = " << T_values[i] << ", n = " << n;
        }
    }
}

TEST(BoysBatchTest, MemoryLayout) {
    // Verify the memory layout: result[i * (n_max+1) + n] = F_n(T_i)
    const int n_max = 5;
    const std::vector<double> T_values = {1.0, 5.0, 10.0};
    const int n_T = 3;

    std::vector<double> result(n_T * (n_max + 1));
    boys_evaluate_batch(n_max, T_values.data(), n_T, result.data());

    // Check specific values
    for (int i = 0; i < n_T; ++i) {
        for (int n = 0; n <= n_max; ++n) {
            double expected = boys_evaluate(n, T_values[i]);
            EXPECT_DOUBLE_EQ(result[i * (n_max + 1) + n], expected)
                << "Layout check: T[" << i << "] = " << T_values[i]
                << ", n = " << n;
        }
    }
}

TEST(BoysBatchTest, RandomTValues) {
    // Batch of 100 pseudo-random T values across all regimes
    const int n_max = 20;
    const int n_T = 100;
    std::vector<double> T_values(n_T);

    // Generate deterministic "random" T values spanning all regimes
    for (int i = 0; i < n_T; ++i) {
        // Linear spread from 0 to 70, covering T=0, small T, medium T,
        // Chebyshev regime, and asymptotic regime
        T_values[i] = static_cast<double>(i) * 70.0 / (n_T - 1);
    }

    std::vector<double> batch_result(n_T * (n_max + 1));
    boys_evaluate_batch(n_max, T_values.data(), n_T, batch_result.data());

    for (int i = 0; i < n_T; ++i) {
        std::vector<double> single_result(n_max + 1);
        boys_evaluate_array(n_max, T_values[i], single_result.data());

        for (int n = 0; n <= n_max; ++n) {
            EXPECT_DOUBLE_EQ(batch_result[i * (n_max + 1) + n], single_result[n])
                << "Random T[" << i << "] = " << T_values[i] << ", n = " << n;
        }
    }
}

TEST(BoysBatchTest, SingleTValue) {
    // Batch with n_values = 1 should match array evaluation
    const int n_max = 10;
    double T = 5.0;

    std::vector<double> batch_result(n_max + 1);
    boys_evaluate_batch(n_max, &T, 1, batch_result.data());

    std::vector<double> array_result(n_max + 1);
    boys_evaluate_array(n_max, T, array_result.data());

    for (int n = 0; n <= n_max; ++n) {
        EXPECT_DOUBLE_EQ(batch_result[n], array_result[n]);
    }
}

TEST(BoysBatchTest, EmptyBatch) {
    // n_values = 0 should be a no-op
    const int n_max = 10;
    double dummy_result = -999.0;
    double dummy_T = 1.0;

    // Should not crash or modify anything
    boys_evaluate_batch(n_max, &dummy_T, 0, &dummy_result);
    EXPECT_DOUBLE_EQ(dummy_result, -999.0);
}

TEST(BoysBatchTest, MonotonicDecreaseInBatch) {
    // All entries in batch should have monotonically decreasing F_n
    const int n_max = 20;
    const std::vector<double> T_values = {0.001, 0.5, 5.0, 15.0, 30.0, 50.0};
    const int n_T = static_cast<int>(T_values.size());

    std::vector<double> result(n_T * (n_max + 1));
    boys_evaluate_batch(n_max, T_values.data(), n_T, result.data());

    for (int i = 0; i < n_T; ++i) {
        for (int n = 1; n <= n_max; ++n) {
            EXPECT_LT(result[i * (n_max + 1) + n],
                       result[i * (n_max + 1) + n - 1])
                << "T[" << i << "] = " << T_values[i]
                << ": F_" << n << " should be less than F_" << (n - 1);
        }
    }
}

// =============================================================================
// Comprehensive Validation (Task 0.3.5)
//
// Reference values generated using mpmath with 50-digit precision.
// Tests all n=0..30 at regime boundary T values.
// Reference data also stored in tests/data/boys_reference.json.
// =============================================================================

namespace {

struct ValidationPoint {
    double T;
    std::array<double, 31> values;  // F_0 through F_30
};

std::vector<ValidationPoint> get_validation_data() {
    std::vector<ValidationPoint> data;

    // T = 0.0 (exact: F_n(0) = 1/(2n+1))
    data.push_back({0.0, {1.00000000000000000e+00, 3.33333333333333315e-01, 2.00000000000000011e-01, 1.42857142857142849e-01, 1.11111111111111105e-01, 9.09090909090909116e-02, 7.69230769230769273e-02, 6.66666666666666657e-02, 5.88235294117647051e-02, 5.26315789473684181e-02, 4.76190476190476164e-02, 4.34782608695652162e-02, 4.00000000000000008e-02, 3.70370370370370350e-02, 3.44827586206896547e-02, 3.22580645161290314e-02, 3.03030303030303039e-02, 2.85714285714285705e-02, 2.70270270270270285e-02, 2.56410256410256401e-02, 2.43902439024390252e-02, 2.32558139534883718e-02, 2.22222222222222231e-02, 2.12765957446808505e-02, 2.04081632653061208e-02, 1.96078431372549017e-02, 1.88679245283018861e-02, 1.81818181818181809e-02, 1.75438596491228060e-02, 1.69491525423728813e-02, 1.63934426229508205e-02}});

    // T = 0.01 (Taylor regime)
    data.push_back({0.01, {9.96676642903363552e-01, 3.31340457709772440e-01, 1.98576969007464771e-01, 1.41750564407793211e-01, 1.10205855269221262e-01, 9.01431836911621015e-02, 7.62593426807561120e-02, 6.60810550330711149e-02, 5.82995873449310142e-02, 5.21575557329581677e-02, 4.71862588518534368e-02, 4.30801069877062778e-02, 3.96313484038190653e-02, 3.66938173154331693e-02, 3.41616883763786808e-02, 3.19564582906862668e-02, 3.00186631053063480e-02, 2.83024362970711353e-02, 2.67718324160844574e-02, 2.53982822978434906e-02, 2.41587933364030491e-02, 2.30346521678470757e-02, 2.20104734128113294e-02, 2.10734913670917248e-02, 2.02130252071527974e-02, 1.94200700659528727e-02, 1.86869807214247972e-02, 1.80072243173145935e-02, 1.73751851567254187e-02, 1.67860092090413061e-02, 1.62354792134546330e-02}});

    // T = 5.0 (mid-range Chebyshev)
    data.push_back({5.0, {3.95712309610513568e-01, 3.88974362611428093e-02, 1.09954361784342959e-02, 4.82392338930860091e-03, 2.70295167260747394e-03, 1.75886180543817993e-03, 1.26095328607345116e-03, 9.65444571986939890e-04, 7.74372158071863059e-04, 6.42637968813620521e-04, 5.47217440837332258e-04, 4.75361925849851041e-04, 4.19537729546110634e-04, 3.75049623956729891e-04, 3.38839284774624034e-04, 3.08839225937863025e-04, 2.83606900498828711e-04, 2.62108071737587975e-04, 2.43583551173011137e-04, 2.27464439431594522e-04, 2.13316613874671950e-04, 2.00803416977608307e-04, 1.89659993095169015e-04, 1.79675269019713858e-04, 1.70679064484108371e-04, 1.62532716063584358e-04, 1.55122152015733548e-04, 1.48352705774841082e-04, 1.42145181853079160e-04, 1.36432836654004538e-04, 1.31159036350080114e-04}});

    // T = 15.0 (mid-range)
    data.push_back({15.0, {2.28822798329737342e-01, 7.62741641424722824e-03, 7.62731444680706102e-04, 1.27111710702767616e-04, 2.96492024199623838e-05, 8.88456398197198738e-06, 3.24747671603966789e-06, 1.39704316626712859e-06, 6.88324839116836803e-07, 3.79853998149479981e-07, 2.30377454811276462e-07, 1.51067474351166001e-07, 1.05621652985833073e-07, 7.78213001381333610e-08, 5.98424261075925048e-08, 4.76509345539452272e-08, 3.90425550223492105e-08, 3.27500665078566029e-08, 2.80116669091051783e-08, 2.43509785045021941e-08, 2.14595280391253253e-08, 1.91312776367437501e-08, 1.72247539292718503e-08, 1.56403868771802498e-08, 1.43065287575215300e-08, 1.31705862872243060e-08, 1.21932526715537946e-08, 1.13446690363508428e-08, 1.06018158832490190e-08, 9.94670616144560992e-09, 9.36511143411550820e-09}});

    // T = 30.0 (crossover region)
    data.push_back({30.0, {1.61802159379640070e-01, 2.69670265632577502e-03, 1.34835132814729137e-04, 1.12362610663344919e-05, 1.31089712284608677e-06, 1.96634566867309196e-07, 3.60496690327361878e-08, 7.81076006415567917e-09, 1.95268845643509183e-09, 5.53260169719447884e-10, 1.75197494140663680e-10, 6.13175633454041508e-11, 2.35035063452434524e-11, 9.79156804002329843e-12, 4.40464601418234434e-12, 2.12735263635999297e-12, 1.09757259162452288e-12, 6.02105321565347656e-13, 3.49668500418312724e-13, 2.14069304763152839e-13, 1.37585444267909315e-13, 9.24571164215979953e-14, 6.47013296073385318e-14, 4.69663933773638763e-14, 3.52307376507950044e-14, 2.72121652533425597e-14, 2.15707366372011465e-14, 1.74945468680543156e-14, 1.44770641342430953e-14, 1.21936070993909097e-14, 1.04307764862610329e-14}});

    // T = 100.0 (deep asymptotic regime)
    data.push_back({100.0, {8.86226925452757996e-02, 4.43113462726378994e-04, 6.64670194089568491e-06, 1.66167548522392123e-07, 5.81586419828372463e-09, 2.61713888922767577e-10, 1.43942638907522184e-11, 9.35627152898894133e-13, 7.01720364674170675e-14, 5.96462309973045068e-15, 5.66639194474392747e-16, 5.94971154198112459e-17, 6.84216827327829324e-18, 8.55271034159786655e-19, 1.15461589611571199e-19, 1.67419304936778231e-20, 2.59499922652006243e-21, 4.28174872375810326e-22, 7.49306026657668000e-23, 1.38621614931668581e-23, 2.70312149116753767e-24, 5.54139905689345203e-25, 1.19140079723209211e-25, 2.68065179377220760e-26, 6.29953171536468737e-27, 1.54338527026434841e-27, 3.93563243917408855e-28, 1.04294259638113352e-28, 2.86809214004811690e-29, 8.17406259913713263e-30, 2.41134846674545418e-30}});

    return data;
}

} // anonymous namespace

TEST(BoysValidationTest, AllOrdersAllRegimes) {
    // Comprehensive validation: all n=0..30 at regime boundary T values.
    // Reference data from mpmath with 50-digit precision.
    auto data = get_validation_data();
    const int n_max = 30;

    double worst_error = 0.0;
    double worst_T = 0.0;
    int worst_n = 0;

    for (const auto& point : data) {
        std::vector<double> result(n_max + 1);
        boys_evaluate_array(n_max, point.T, result.data());

        for (int n = 0; n <= n_max; ++n) {
            double rel_err = relative_error(result[n], point.values[n]);

            if (rel_err > worst_error) {
                worst_error = rel_err;
                worst_T = point.T;
                worst_n = n;
            }

            EXPECT_LT(rel_err, 1e-14)
                << "F_" << n << "(" << point.T << "): "
                << "computed = " << result[n]
                << ", reference = " << point.values[n]
                << ", relative error = " << rel_err;
        }
    }

    // Report worst-case error (informational, test still passes)
    std::cout << "  [INFO] Worst-case relative error: " << worst_error
              << " at F_" << worst_n << "(" << worst_T << ")" << std::endl;
}

TEST(BoysValidationTest, BatchMatchesSingle) {
    // Verify batch evaluation matches single evaluation for all validation points
    auto data = get_validation_data();
    const int n_max = 30;
    const int n_T = static_cast<int>(data.size());

    std::vector<double> T_values(n_T);
    for (int i = 0; i < n_T; ++i) {
        T_values[i] = data[i].T;
    }

    std::vector<double> batch_result(n_T * (n_max + 1));
    boys_evaluate_batch(n_max, T_values.data(), n_T, batch_result.data());

    for (int i = 0; i < n_T; ++i) {
        std::vector<double> single_result(n_max + 1);
        boys_evaluate_array(n_max, T_values[i], single_result.data());

        for (int n = 0; n <= n_max; ++n) {
            EXPECT_DOUBLE_EQ(batch_result[i * (n_max + 1) + n], single_result[n])
                << "Batch mismatch at T = " << T_values[i] << ", n = " << n;
        }
    }
}

TEST(BoysValidationTest, RegimeBoundaries) {
    // Verify smooth transitions at regime boundaries
    const int n_max = 30;
    const double epsilon = 1e-10;

    // Boundary: T = 0 (exact formula vs Taylor)
    {
        std::vector<double> r0(n_max + 1), r_eps(n_max + 1);
        boys_evaluate_array(n_max, 0.0, r0.data());
        boys_evaluate_array(n_max, 1e-16, r_eps.data());
        for (int n = 0; n <= n_max; ++n) {
            double diff = relative_error(r0[n], r_eps[n]);
            EXPECT_LT(diff, 1e-14)
                << "Discontinuity at T=0 boundary, n=" << n;
        }
    }

    // Boundary: T near BOYS_CHEBYSHEV_T_MAX (36)
    // Use very tight spacing (1e-10) so natural function variation
    // (dF_n/dT × delta) is negligible (~1e-12) compared to tolerance.
    // Any significant discrepancy must be due to regime switching error.
    {
        std::vector<double> r_below(n_max + 1), r_above(n_max + 1);
        boys_evaluate_array(n_max, 36.0 - 1e-10, r_below.data());
        boys_evaluate_array(n_max, 36.0 + 1e-10, r_above.data());
        for (int n = 0; n <= n_max; ++n) {
            double diff = relative_error(r_below[n], r_above[n]);
            EXPECT_LT(diff, 1e-9)
                << "Discontinuity at T=36 boundary, n=" << n;
        }
    }
}

TEST(BoysValidationTest, PerformanceSingleEvaluation) {
    // Timing test for single evaluation
    const int n_max = 20;
    const int n_iters = 100000;
    std::vector<double> result(n_max + 1);

    // Warm up
    for (int i = 0; i < 1000; ++i) {
        boys_evaluate_array(n_max, 5.0 + i * 0.001, result.data());
    }

    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < n_iters; ++i) {
        // Use varying T to avoid branch prediction artifacts
        double T = static_cast<double>(i % 100) * 0.35;
        boys_evaluate_array(n_max, T, result.data());
    }
    auto end = std::chrono::high_resolution_clock::now();

    double ns_per_eval = std::chrono::duration_cast<std::chrono::nanoseconds>(
                             end - start)
                             .count() /
                         static_cast<double>(n_iters);

    std::cout << "  [PERF] Single evaluation: " << ns_per_eval
              << " ns/eval (n_max=" << n_max << ")" << std::endl;

    // Target: < 100 ns per evaluation (informational, not a hard failure)
    // This may vary by hardware; report but don't fail
    EXPECT_LT(ns_per_eval, 10000.0)
        << "Single evaluation seems unreasonably slow";
}

TEST(BoysValidationTest, PerformanceBatchEvaluation) {
    // Timing test for batch evaluation
    const int n_max = 20;
    const int batch_size = 1000;
    const int n_iters = 1000;

    std::vector<double> T_values(batch_size);
    for (int i = 0; i < batch_size; ++i) {
        T_values[i] = static_cast<double>(i) * 35.0 / (batch_size - 1);
    }

    std::vector<double> result(batch_size * (n_max + 1));

    // Warm up
    for (int i = 0; i < 10; ++i) {
        boys_evaluate_batch(n_max, T_values.data(), batch_size, result.data());
    }

    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < n_iters; ++i) {
        boys_evaluate_batch(n_max, T_values.data(), batch_size, result.data());
    }
    auto end = std::chrono::high_resolution_clock::now();

    double total_evals = static_cast<double>(n_iters) * batch_size;
    double ns_per_eval = std::chrono::duration_cast<std::chrono::nanoseconds>(
                             end - start)
                             .count() /
                         total_evals;

    std::cout << "  [PERF] Batch evaluation: " << ns_per_eval
              << " ns/eval amortized (batch=" << batch_size
              << ", n_max=" << n_max << ")" << std::endl;

    // Target: < 50 ns amortized (informational, not a hard failure)
    EXPECT_LT(ns_per_eval, 10000.0)
        << "Batch evaluation seems unreasonably slow";
}

// ============================================================================
// SIMD Correctness Tests
// ============================================================================

TEST(BoysFunctionTest, SIMD_MatchesScalar_SmallT) {
    // Test SIMD batch evaluation matches scalar for small T values
    constexpr int n_max = 20;
    constexpr int batch_size = 64;

    std::vector<double> T_values(batch_size);
    for (int i = 0; i < batch_size; ++i) {
        T_values[i] = static_cast<double>(i) * 0.5;  // 0 to 31.5
    }

    std::vector<double> result_scalar(batch_size * (n_max + 1));
    std::vector<double> result_simd(batch_size * (n_max + 1));

    // Compute with scalar batch
    boys_evaluate_batch(n_max, T_values.data(), batch_size, result_scalar.data());

    // Compute with SIMD batch
    boys_evaluate_batch_simd(n_max, T_values.data(), batch_size, result_simd.data());

    // Compare results
    for (int i = 0; i < batch_size; ++i) {
        for (int n = 0; n <= n_max; ++n) {
            double scalar_val = result_scalar[i * (n_max + 1) + n];
            double simd_val = result_simd[i * (n_max + 1) + n];
            double rel_err = relative_error(simd_val, scalar_val);

            EXPECT_LT(rel_err, 1e-12)
                << "SIMD mismatch at T=" << T_values[i] << ", n=" << n
                << ": scalar=" << scalar_val << ", simd=" << simd_val
                << ", rel_err=" << rel_err;
        }
    }
}

TEST(BoysFunctionTest, SIMD_MatchesScalar_LargeT) {
    // Test SIMD batch evaluation matches scalar for large T values (asymptotic regime)
    constexpr int n_max = 20;
    constexpr int batch_size = 64;

    std::vector<double> T_values(batch_size);
    for (int i = 0; i < batch_size; ++i) {
        T_values[i] = 30.0 + static_cast<double>(i) * 2.0;  // 30 to 156
    }

    std::vector<double> result_scalar(batch_size * (n_max + 1));
    std::vector<double> result_simd(batch_size * (n_max + 1));

    boys_evaluate_batch(n_max, T_values.data(), batch_size, result_scalar.data());
    boys_evaluate_batch_simd(n_max, T_values.data(), batch_size, result_simd.data());

    for (int i = 0; i < batch_size; ++i) {
        for (int n = 0; n <= n_max; ++n) {
            double scalar_val = result_scalar[i * (n_max + 1) + n];
            double simd_val = result_simd[i * (n_max + 1) + n];
            double rel_err = relative_error(simd_val, scalar_val);

            EXPECT_LT(rel_err, 1e-12)
                << "SIMD mismatch at T=" << T_values[i] << ", n=" << n
                << ": scalar=" << scalar_val << ", simd=" << simd_val;
        }
    }
}

TEST(BoysFunctionTest, SIMD_MatchesScalar_MixedRegimes) {
    // Test SIMD batch with T values spanning multiple Chebyshev intervals
    constexpr int n_max = 15;
    constexpr int batch_size = 128;

    std::vector<double> T_values(batch_size);
    for (int i = 0; i < batch_size; ++i) {
        // Span from 0 to 100, crossing multiple intervals
        T_values[i] = static_cast<double>(i) * 100.0 / (batch_size - 1);
    }

    std::vector<double> result_scalar(batch_size * (n_max + 1));
    std::vector<double> result_simd(batch_size * (n_max + 1));

    boys_evaluate_batch(n_max, T_values.data(), batch_size, result_scalar.data());
    boys_evaluate_batch_simd(n_max, T_values.data(), batch_size, result_simd.data());

    for (int i = 0; i < batch_size; ++i) {
        for (int n = 0; n <= n_max; ++n) {
            double scalar_val = result_scalar[i * (n_max + 1) + n];
            double simd_val = result_simd[i * (n_max + 1) + n];
            double rel_err = relative_error(simd_val, scalar_val);

            EXPECT_LT(rel_err, 1e-12)
                << "SIMD mismatch at T=" << T_values[i] << ", n=" << n
                << ": scalar=" << scalar_val << ", simd=" << simd_val;
        }
    }
}

TEST(BoysFunctionTest, SIMD_MatchesScalar_SameInterval) {
    // Test optimized same-interval path with T values all in one Chebyshev interval
    // Note: The SIMD same-interval optimization uses a different vectorized code path
    // that may have slightly different numerical characteristics due to FMA ordering,
    // but still well within chemical accuracy requirements (1e-10 in integrals)
    constexpr int n_max = 20;
    constexpr int batch_size = 32;

    // All T values in interval [5.0, 6.0) - same Chebyshev interval
    std::vector<double> T_values(batch_size);
    for (int i = 0; i < batch_size; ++i) {
        T_values[i] = 5.0 + static_cast<double>(i) * 0.03;
    }

    std::vector<double> result_scalar(batch_size * (n_max + 1));
    std::vector<double> result_simd(batch_size * (n_max + 1));

    boys_evaluate_batch(n_max, T_values.data(), batch_size, result_scalar.data());
    boys_evaluate_batch_simd(n_max, T_values.data(), batch_size, result_simd.data());

    for (int i = 0; i < batch_size; ++i) {
        for (int n = 0; n <= n_max; ++n) {
            double scalar_val = result_scalar[i * (n_max + 1) + n];
            double simd_val = result_simd[i * (n_max + 1) + n];
            double rel_err = relative_error(simd_val, scalar_val);

            // Allow 1e-8 tolerance for SIMD path (still within chemical accuracy)
            EXPECT_LT(rel_err, 1e-8)
                << "SIMD same-interval mismatch at T=" << T_values[i] << ", n=" << n;
        }
    }
}

TEST(BoysFunctionTest, SIMD_MatchesScalar_HighOrder) {
    // Test high-order Boys functions (n up to 30)
    constexpr int n_max = 30;
    constexpr int batch_size = 16;

    std::vector<double> T_values = {0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 15.0, 20.0,
                                     25.0, 29.0, 30.0, 35.0, 50.0, 75.0, 100.0, 200.0};

    std::vector<double> result_scalar(batch_size * (n_max + 1));
    std::vector<double> result_simd(batch_size * (n_max + 1));

    boys_evaluate_batch(n_max, T_values.data(), batch_size, result_scalar.data());
    boys_evaluate_batch_simd(n_max, T_values.data(), batch_size, result_simd.data());

    for (int i = 0; i < batch_size; ++i) {
        for (int n = 0; n <= n_max; ++n) {
            double scalar_val = result_scalar[i * (n_max + 1) + n];
            double simd_val = result_simd[i * (n_max + 1) + n];
            double rel_err = relative_error(simd_val, scalar_val);

            EXPECT_LT(rel_err, 1e-11)
                << "SIMD high-order mismatch at T=" << T_values[i] << ", n=" << n;
        }
    }
}

TEST(BoysFunctionTest, SIMD_EdgeCases) {
    // Test edge cases: very small T, boundary values
    constexpr int n_max = 10;

    std::vector<double> T_values = {
        1e-15,   // Near zero
        1e-10,
        1e-5,
        0.999,   // Just below interval boundary
        1.0,     // Interval boundary
        1.001,   // Just above interval boundary
        29.99,   // Just below asymptotic threshold
        30.0,    // Asymptotic threshold
        30.01    // Just above asymptotic threshold
    };
    const int batch_size = static_cast<int>(T_values.size());

    std::vector<double> result_scalar(batch_size * (n_max + 1));
    std::vector<double> result_simd(batch_size * (n_max + 1));

    boys_evaluate_batch(n_max, T_values.data(), batch_size, result_scalar.data());
    boys_evaluate_batch_simd(n_max, T_values.data(), batch_size, result_simd.data());

    for (int i = 0; i < batch_size; ++i) {
        for (int n = 0; n <= n_max; ++n) {
            double scalar_val = result_scalar[i * (n_max + 1) + n];
            double simd_val = result_simd[i * (n_max + 1) + n];
            double rel_err = relative_error(simd_val, scalar_val);

            EXPECT_LT(rel_err, 1e-10)
                << "SIMD edge case mismatch at T=" << T_values[i] << ", n=" << n;
        }
    }
}
