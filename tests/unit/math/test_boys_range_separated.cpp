// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2026 Bryce M. Westheimer. All rights reserved.

/// @file test_boys_range_separated.cpp
/// @brief Unit tests for range-separated Boys function

#include <libaccint/math/boys_range_separated.hpp>
#include <libaccint/math/boys_function.hpp>
#include <gtest/gtest.h>
#include <cmath>
#include <vector>

using namespace libaccint;
using namespace libaccint::math;

namespace {

// Helper: relative error
double relative_error(double computed, double reference) {
    if (std::abs(reference) < 1e-100) {
        return std::abs(computed - reference);
    }
    return std::abs((computed - reference) / reference);
}

}  // namespace

// ============================================================================
// Basic Functionality Tests
// ============================================================================

TEST(BoysRangeSeparated, ErfBasicComputation) {
    // Test basic erf-attenuated Boys function computation
    const double T = 1.0;
    const double omega = 0.4;  // CAM-B3LYP typical value

    for (int n = 0; n <= 10; ++n) {
        double result = boys_erf(n, T, omega);
        EXPECT_TRUE(std::isfinite(result)) << "boys_erf(" << n << ", " << T << ", " << omega << ") is not finite";
        EXPECT_GE(result, 0.0) << "boys_erf should be non-negative";
    }
}

TEST(BoysRangeSeparated, ErfcBasicComputation) {
    // Test basic erfc-attenuated Boys function computation
    const double T = 1.0;
    const double omega = 0.4;

    for (int n = 0; n <= 10; ++n) {
        double result = boys_erfc(n, T, omega);
        EXPECT_TRUE(std::isfinite(result)) << "boys_erfc(" << n << ", " << T << ", " << omega << ") is not finite";
        EXPECT_GE(result, 0.0) << "boys_erfc should be non-negative";
    }
}

// ============================================================================
// Limiting Behavior Tests
// ============================================================================

TEST(BoysRangeSeparated, ErfSmallOmegaLimit) {
    // omega -> 0: erf(omega * r) -> 0, so F_n^{erf} -> 0
    // For omega=0.0001: scale = omega^(2n+1) ≈ 10^(-4n-2) which is very small
    const double omega = 0.0001;
    const double T = 1.0;

    for (int n = 0; n <= 10; ++n) {
        double result = boys_erf(n, T, omega);
        // For small omega, F_n^erf ≈ scale * F_n(0) where scale is very small
        // scale = (omega^2/(omega^2+1))^{n+0.5} ≈ omega^(2n+1)
        EXPECT_LT(result, 1e-3)
            << "omega -> 0 should give F_n^{erf} -> 0, n=" << n << ", result=" << result;
    }
}

TEST(BoysRangeSeparated, ErfLargeOmegaLimit) {
    // omega -> infinity: F_n^{erf} -> F_n(T)
    // Using omega > 1000.0 to trigger the limiting case in implementation
    const double omega = 1001.0;
    const double T = 1.0;

    for (int n = 0; n <= 10; ++n) {
        double erf_result = boys_erf(n, T, omega);
        double full_result = boys_evaluate(n, T);
        double rel_error = relative_error(erf_result, full_result);
        EXPECT_LT(rel_error, 1e-12)
            << "omega -> infinity should give F_n^{erf} -> F_n, n=" << n
            << ", erf=" << erf_result << ", full=" << full_result;
    }
}

TEST(BoysRangeSeparated, ErfcSmallOmegaLimit) {
    // omega -> 0: erfc(omega * r) -> 1, so F_n^{erfc} -> F_n(T)
    // Since erfc = full - erf, and erf is small for small omega, erfc ≈ full
    const double omega = 0.0001;
    const double T = 1.0;

    for (int n = 0; n <= 10; ++n) {
        double erfc_result = boys_erfc(n, T, omega);
        double full_result = boys_evaluate(n, T);
        double rel_error = relative_error(erfc_result, full_result);
        EXPECT_LT(rel_error, 1e-3)
            << "omega -> 0 should give F_n^{erfc} -> F_n, n=" << n;
    }
}

TEST(BoysRangeSeparated, ErfcLargeOmegaLimit) {
    // omega -> infinity: F_n^{erfc} -> 0
    // Using omega > 1000.0 to trigger the limiting case in implementation
    const double omega = 1001.0;
    const double T = 1.0;

    for (int n = 0; n <= 10; ++n) {
        double result = boys_erfc(n, T, omega);
        EXPECT_NEAR(result, 0.0, 1e-12)
            << "omega -> infinity should give F_n^{erfc} -> 0, n=" << n;
    }
}

// ============================================================================
// Decomposition Identity Tests (Critical for Quality Gate G3)
// ============================================================================

TEST(BoysRangeSeparated, DecompositionIdentity) {
    // F_n^{erf}(T, omega) + F_n^{erfc}(T, omega) = F_n(T)
    std::vector<double> omegas = {0.1, 0.2, 0.33, 0.4, 0.5, 1.0, 2.0, 5.0, 10.0};
    std::vector<double> T_values = {0.0, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 30.0};

    double max_error = 0.0;

    for (double omega : omegas) {
        for (double T : T_values) {
            for (int n = 0; n <= 15; ++n) {
                double erf_val = boys_erf(n, T, omega);
                double erfc_val = boys_erfc(n, T, omega);
                double full_val = boys_evaluate(n, T);

                double sum = erf_val + erfc_val;
                double abs_error = std::abs(sum - full_val);

                // Use relative error for non-zero values
                double error = (std::abs(full_val) > 1e-100) ?
                    abs_error / std::abs(full_val) : abs_error;

                max_error = std::max(max_error, error);

                EXPECT_LT(error, 1e-12)
                    << "Decomposition identity failed: n=" << n
                    << ", T=" << T << ", omega=" << omega
                    << ", erf=" << erf_val << ", erfc=" << erfc_val
                    << ", sum=" << sum << ", full=" << full_val
                    << ", error=" << error;
            }
        }
    }

    std::cout << "Maximum decomposition error: " << max_error << std::endl;
    EXPECT_LT(max_error, 1e-12) << "Quality Gate G3: Decomposition error must be < 1e-12";
}

// ============================================================================
// Array Function Tests
// ============================================================================

TEST(BoysRangeSeparated, ErfArrayConsistency) {
    // boys_erf_array should match individual boys_erf calls
    const double T = 2.5;
    const double omega = 0.4;
    const int n_max = 10;

    double result_array[n_max + 1];
    boys_erf_array(n_max, T, omega, result_array);

    for (int n = 0; n <= n_max; ++n) {
        double single = boys_erf(n, T, omega);
        double rel_error = relative_error(result_array[n], single);
        EXPECT_LT(rel_error, 1e-14)
            << "Array and single evaluation should match for n=" << n;
    }
}

TEST(BoysRangeSeparated, ErfcArrayConsistency) {
    const double T = 2.5;
    const double omega = 0.4;
    const int n_max = 10;

    double result_array[n_max + 1];
    boys_erfc_array(n_max, T, omega, result_array);

    for (int n = 0; n <= n_max; ++n) {
        double single = boys_erfc(n, T, omega);
        double rel_error = relative_error(result_array[n], single);
        EXPECT_LT(rel_error, 1e-14)
            << "Array and single evaluation should match for n=" << n;
    }
}

// ============================================================================
// Batch Function Tests
// ============================================================================

TEST(BoysRangeSeparated, ErfBatchConsistency) {
    const double omega = 0.4;
    const int n_max = 5;
    const int n_values = 10;

    std::vector<double> T_array(n_values);
    for (int i = 0; i < n_values; ++i) {
        T_array[i] = 0.5 * i;
    }

    std::vector<double> batch_result(n_values * (n_max + 1));
    boys_erf_batch(n_max, T_array.data(), n_values, omega, batch_result.data());

    for (int i = 0; i < n_values; ++i) {
        double T = T_array[i];
        for (int n = 0; n <= n_max; ++n) {
            double single = boys_erf(n, T, omega);
            double batch = batch_result[i * (n_max + 1) + n];
            double rel_error = relative_error(batch, single);
            EXPECT_LT(rel_error, 1e-14)
                << "Batch and single should match for T=" << T << ", n=" << n;
        }
    }
}

TEST(BoysRangeSeparated, ErfcBatchConsistency) {
    const double omega = 0.4;
    const int n_max = 5;
    const int n_values = 10;

    std::vector<double> T_array(n_values);
    for (int i = 0; i < n_values; ++i) {
        T_array[i] = 0.5 * i;
    }

    std::vector<double> batch_result(n_values * (n_max + 1));
    boys_erfc_batch(n_max, T_array.data(), n_values, omega, batch_result.data());

    for (int i = 0; i < n_values; ++i) {
        double T = T_array[i];
        for (int n = 0; n <= n_max; ++n) {
            double single = boys_erfc(n, T, omega);
            double batch = batch_result[i * (n_max + 1) + n];
            double rel_error = relative_error(batch, single);
            EXPECT_LT(rel_error, 1e-14)
                << "Batch and single should match for T=" << T << ", n=" << n;
        }
    }
}

// ============================================================================
// Special Value Tests
// ============================================================================

TEST(BoysRangeSeparated, TZeroHandling) {
    // Test T = 0 case
    const double T = 0.0;
    const double omega = 0.4;

    for (int n = 0; n <= 10; ++n) {
        double erf_val = boys_erf(n, T, omega);
        double erfc_val = boys_erfc(n, T, omega);
        double full_val = boys_evaluate(n, T);

        // F_n(0) = 1/(2n+1)
        double expected_full = 1.0 / (2 * n + 1);
        EXPECT_NEAR(full_val, expected_full, 1e-14)
            << "F_n(0) should equal 1/(2n+1), n=" << n;

        // Sum should still equal full
        double sum = erf_val + erfc_val;
        EXPECT_NEAR(sum, full_val, 1e-12)
            << "Decomposition should hold at T=0, n=" << n;
    }
}

TEST(BoysRangeSeparated, VerySmallT) {
    // Test very small T values (Taylor regime)
    const double T = 1e-10;
    const double omega = 0.4;

    for (int n = 0; n <= 10; ++n) {
        double erf_val = boys_erf(n, T, omega);
        double erfc_val = boys_erfc(n, T, omega);
        double full_val = boys_evaluate(n, T);

        double sum = erf_val + erfc_val;
        double error = std::abs(sum - full_val);
        EXPECT_LT(error, 1e-12)
            << "Decomposition should hold for small T, n=" << n;
    }
}

TEST(BoysRangeSeparated, LargeT) {
    // Test large T values (asymptotic regime)
    const double T = 50.0;
    const double omega = 0.4;

    for (int n = 0; n <= 10; ++n) {
        double erf_val = boys_erf(n, T, omega);
        double erfc_val = boys_erfc(n, T, omega);
        double full_val = boys_evaluate(n, T);

        double sum = erf_val + erfc_val;
        double rel_error = (std::abs(full_val) > 1e-100) ?
            std::abs(sum - full_val) / std::abs(full_val) :
            std::abs(sum - full_val);

        EXPECT_LT(rel_error, 1e-11)
            << "Decomposition should hold for large T, n=" << n;
    }
}

// ============================================================================
// Typical Range-Separated Functional Omega Values
// ============================================================================

TEST(BoysRangeSeparated, CAMB3LYP_Omega) {
    // CAM-B3LYP uses alpha=0.19, beta=0.46, omega=0.33
    const double omega = 0.33;
    const double T = 5.0;

    for (int n = 0; n <= 10; ++n) {
        double erf_val = boys_erf(n, T, omega);
        double erfc_val = boys_erfc(n, T, omega);
        double full_val = boys_evaluate(n, T);

        EXPECT_NEAR(erf_val + erfc_val, full_val, 1e-12)
            << "CAM-B3LYP omega decomposition, n=" << n;
    }
}

TEST(BoysRangeSeparated, wB97X_Omega) {
    // omega-B97X uses omega ~ 0.3
    const double omega = 0.3;
    const double T = 5.0;

    for (int n = 0; n <= 10; ++n) {
        double erf_val = boys_erf(n, T, omega);
        double erfc_val = boys_erfc(n, T, omega);
        double full_val = boys_evaluate(n, T);

        EXPECT_NEAR(erf_val + erfc_val, full_val, 1e-12)
            << "omega-B97X omega decomposition, n=" << n;
    }
}

TEST(BoysRangeSeparated, LCBLYP_Omega) {
    // LC-BLYP uses omega ~ 0.47
    const double omega = 0.47;
    const double T = 5.0;

    for (int n = 0; n <= 10; ++n) {
        double erf_val = boys_erf(n, T, omega);
        double erfc_val = boys_erfc(n, T, omega);
        double full_val = boys_evaluate(n, T);

        EXPECT_NEAR(erf_val + erfc_val, full_val, 1e-12)
            << "LC-BLYP omega decomposition, n=" << n;
    }
}

// ============================================================================
// Numerical Stability Tests
// ============================================================================

TEST(BoysRangeSeparated, NumericalStabilityVerySmallOmega) {
    // Test numerical stability for omega values approaching 0
    std::vector<double> small_omegas = {0.1, 0.05, 0.01, 0.005, 0.001};
    const double T = 1.0;

    for (double omega : small_omegas) {
        for (int n = 0; n <= 10; ++n) {
            double erf_val = boys_erf(n, T, omega);
            double erfc_val = boys_erfc(n, T, omega);

            EXPECT_TRUE(std::isfinite(erf_val))
                << "erf value should be finite for small omega=" << omega;
            EXPECT_TRUE(std::isfinite(erfc_val))
                << "erfc value should be finite for small omega=" << omega;
            EXPECT_GE(erf_val, 0.0) << "erf should be non-negative";
            EXPECT_GE(erfc_val, 0.0) << "erfc should be non-negative";
        }
    }
}

TEST(BoysRangeSeparated, NumericalStabilityLargeOmega) {
    // Test numerical stability for large omega values
    std::vector<double> large_omegas = {10.0, 50.0, 100.0, 500.0, 1000.0};
    const double T = 1.0;

    for (double omega : large_omegas) {
        for (int n = 0; n <= 10; ++n) {
            double erf_val = boys_erf(n, T, omega);
            double erfc_val = boys_erfc(n, T, omega);

            EXPECT_TRUE(std::isfinite(erf_val))
                << "erf value should be finite for large omega=" << omega;
            EXPECT_TRUE(std::isfinite(erfc_val))
                << "erfc value should be finite for large omega=" << omega;
        }
    }
}
