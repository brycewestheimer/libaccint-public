// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2026 Bryce M. Westheimer. All rights reserved.

/// @file test_float32_accuracy.cpp
/// @brief Float32 vs Float64 error analysis (Task 24.4.1)
///
/// Systematic analysis of single-precision integral accuracy across multiple
/// angular momentum combinations and integral types. Computes error statistics
/// (max absolute, RMS, max relative) for overlap, kinetic, nuclear, and ERI.

#include <libaccint/basis/shell.hpp>
#include <libaccint/buffers/one_electron_buffer.hpp>
#include <libaccint/buffers/two_electron_buffer.hpp>
#include <libaccint/core/precision.hpp>
#include <libaccint/core/types.hpp>
#include <libaccint/engine/precision_dispatch.hpp>
#include <libaccint/kernels/overlap_kernel.hpp>
#include <libaccint/kernels/kinetic_kernel.hpp>
#include <libaccint/kernels/nuclear_kernel.hpp>
#include <libaccint/kernels/eri_kernel.hpp>
#include <libaccint/operators/operator_types.hpp>

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <iostream>
#include <vector>

namespace libaccint::test {
namespace {

/// @brief Create a test shell with given AM
Shell make_test_shell(int am, Point3D center) {
    std::vector<Real> exponents;
    std::vector<Real> coefficients;

    switch (am) {
        case 0:
            exponents = {3.42525091, 0.62391373, 0.16885540};
            coefficients = {0.15432897, 0.53532814, 0.44463454};
            break;
        case 1:
            exponents = {2.94124936, 0.68348310, 0.22228990};
            coefficients = {0.15591627, 0.60768372, 0.39195739};
            break;
        case 2:
            exponents = {1.533, 0.5417, 0.2211};
            coefficients = {0.25, 0.50, 0.35};
            break;
        case 3:
            exponents = {1.208, 0.4537, 0.1813};
            coefficients = {0.30, 0.45, 0.35};
            break;
        default:
            exponents = {0.9876, 0.3654, 0.1432};
            coefficients = {0.35, 0.40, 0.35};
            break;
    }

    return Shell(am, center, exponents, coefficients);
}

/// @brief Error statistics for precision comparison
struct ErrorStats {
    double max_absolute{0.0};
    double rms{0.0};
    double max_relative{0.0};
    int n_values{0};
};

/// @brief Compute error statistics between float and double buffers
ErrorStats compute_1e_errors(const OverlapBuffer& double_buf,
                              const OneElectronBuffer<0, float>& float_buf) {
    ErrorStats stats;
    double sum_sq = 0.0;
    int na = double_buf.na();
    int nb = double_buf.nb();
    stats.n_values = na * nb;

    for (int a = 0; a < na; ++a) {
        for (int b = 0; b < nb; ++b) {
            double d_val = double_buf(a, b);
            double f_val = static_cast<double>(float_buf(a, b));
            double abs_err = std::abs(d_val - f_val);
            double rel_err = (std::abs(d_val) > 1e-15)
                ? abs_err / std::abs(d_val) : 0.0;

            stats.max_absolute = std::max(stats.max_absolute, abs_err);
            stats.max_relative = std::max(stats.max_relative, rel_err);
            sum_sq += abs_err * abs_err;
        }
    }

    stats.rms = std::sqrt(sum_sq / stats.n_values);
    return stats;
}

/// @brief Compute error statistics for ERI buffers
ErrorStats compute_2e_errors(const TwoElectronBuffer<0>& double_buf,
                              const TwoElectronBuffer<0, float>& float_buf) {
    ErrorStats stats;
    double sum_sq = 0.0;
    int na = double_buf.na();
    int nb = double_buf.nb();
    int nc = double_buf.nc();
    int nd = double_buf.nd();
    stats.n_values = na * nb * nc * nd;

    for (int a = 0; a < na; ++a) {
        for (int b = 0; b < nb; ++b) {
            for (int c = 0; c < nc; ++c) {
                for (int d = 0; d < nd; ++d) {
                    double d_val = double_buf(a, b, c, d);
                    double f_val = static_cast<double>(float_buf(a, b, c, d));
                    double abs_err = std::abs(d_val - f_val);
                    double rel_err = (std::abs(d_val) > 1e-15)
                        ? abs_err / std::abs(d_val) : 0.0;

                    stats.max_absolute = std::max(stats.max_absolute, abs_err);
                    stats.max_relative = std::max(stats.max_relative, rel_err);
                    sum_sq += abs_err * abs_err;
                }
            }
        }
    }

    stats.rms = std::sqrt(sum_sq / stats.n_values);
    return stats;
}

// ============================================================================
// Overlap Integral Accuracy Tests
// ============================================================================

class Float32OverlapAccuracy : public ::testing::TestWithParam<std::pair<int,int>> {};

TEST_P(Float32OverlapAccuracy, ErrorWithinFloat32Tolerance) {
    auto [la, lb] = GetParam();

    Shell shell_a = make_test_shell(la, {0.0, 0.0, 0.0});
    Shell shell_b = make_test_shell(lb, {1.0, 0.5, -0.3});

    // Compute in double precision
    OverlapBuffer double_buf;
    kernels::compute_overlap(shell_a, shell_b, double_buf);

    // Compute via float dispatch
    OneElectronBuffer<0, float> float_buf;
    engine::compute_overlap_typed<float>(shell_a, shell_b, float_buf);

    auto errors = compute_1e_errors(double_buf, float_buf);

    // Float32 truncation error should be within ~1e-7 relative
    EXPECT_LT(errors.max_relative, 1e-6)
        << "Overlap (" << la << "," << lb << "): max_rel=" << errors.max_relative
        << ", max_abs=" << errors.max_absolute << ", rms=" << errors.rms;
}

INSTANTIATE_TEST_SUITE_P(
    AMCombinations, Float32OverlapAccuracy,
    ::testing::Values(
        std::make_pair(0, 0), std::make_pair(0, 1), std::make_pair(1, 1),
        std::make_pair(0, 2), std::make_pair(1, 2), std::make_pair(2, 2),
        std::make_pair(0, 3), std::make_pair(1, 3), std::make_pair(2, 3)
    )
);

// ============================================================================
// Kinetic Integral Accuracy Tests
// ============================================================================

class Float32KineticAccuracy : public ::testing::TestWithParam<std::pair<int,int>> {};

TEST_P(Float32KineticAccuracy, ErrorWithinFloat32Tolerance) {
    auto [la, lb] = GetParam();

    Shell shell_a = make_test_shell(la, {0.0, 0.0, 0.0});
    Shell shell_b = make_test_shell(lb, {1.2, -0.3, 0.8});

    KineticBuffer double_buf;
    kernels::compute_kinetic(shell_a, shell_b, double_buf);

    OneElectronBuffer<0, float> float_buf;
    engine::compute_kinetic_typed<float>(shell_a, shell_b, float_buf);

    auto errors = compute_1e_errors(double_buf, float_buf);

    EXPECT_LT(errors.max_relative, 1e-6)
        << "Kinetic (" << la << "," << lb << "): max_rel=" << errors.max_relative;
}

INSTANTIATE_TEST_SUITE_P(
    AMCombinations, Float32KineticAccuracy,
    ::testing::Values(
        std::make_pair(0, 0), std::make_pair(0, 1), std::make_pair(1, 1),
        std::make_pair(0, 2), std::make_pair(1, 2), std::make_pair(2, 2)
    )
);

// ============================================================================
// Nuclear Attraction Integral Accuracy Tests
// ============================================================================

class Float32NuclearAccuracy : public ::testing::TestWithParam<std::pair<int,int>> {};

TEST_P(Float32NuclearAccuracy, ErrorWithinFloat32Tolerance) {
    auto [la, lb] = GetParam();

    Shell shell_a = make_test_shell(la, {0.0, 0.0, 0.0});
    Shell shell_b = make_test_shell(lb, {0.0, 0.0, 1.4});

    PointChargeParams charges;
    charges.x = {0.0, 0.0};
    charges.y = {0.0, 0.0};
    charges.z = {0.0, 1.4};
    charges.charge = {1.0, 1.0};

    NuclearBuffer double_buf;
    kernels::compute_nuclear(shell_a, shell_b, charges, double_buf);

    OneElectronBuffer<0, float> float_buf;
    engine::compute_nuclear_typed<float>(shell_a, shell_b, charges, float_buf);

    auto errors = compute_1e_errors(double_buf, float_buf);

    EXPECT_LT(errors.max_relative, 1e-6)
        << "Nuclear (" << la << "," << lb << "): max_rel=" << errors.max_relative;
}

INSTANTIATE_TEST_SUITE_P(
    AMCombinations, Float32NuclearAccuracy,
    ::testing::Values(
        std::make_pair(0, 0), std::make_pair(0, 1), std::make_pair(1, 1),
        std::make_pair(0, 2), std::make_pair(1, 2), std::make_pair(2, 2)
    )
);

// ============================================================================
// ERI Accuracy Tests
// ============================================================================

TEST(Float32ERIAccuracy, SsSsQuartet) {
    Shell sa = make_test_shell(0, {0.0, 0.0, 0.0});
    Shell sb = make_test_shell(0, {1.0, 0.0, 0.0});
    Shell sc = make_test_shell(0, {0.0, 1.0, 0.0});
    Shell sd = make_test_shell(0, {1.0, 1.0, 0.0});

    TwoElectronBuffer<0> double_buf;
    kernels::compute_eri(sa, sb, sc, sd, double_buf);

    TwoElectronBuffer<0, float> float_buf;
    engine::compute_eri_typed<float>(sa, sb, sc, sd, float_buf);

    auto errors = compute_2e_errors(double_buf, float_buf);
    EXPECT_LT(errors.max_relative, 1e-6)
        << "ERI (ss|ss): max_rel=" << errors.max_relative;
}

TEST(Float32ERIAccuracy, SpSpQuartet) {
    Shell sa = make_test_shell(0, {0.0, 0.0, 0.0});
    Shell sb = make_test_shell(1, {1.4, 0.0, 0.0});
    Shell sc = make_test_shell(0, {0.0, 0.0, 0.0});
    Shell sd = make_test_shell(1, {1.4, 0.0, 0.0});

    TwoElectronBuffer<0> double_buf;
    kernels::compute_eri(sa, sb, sc, sd, double_buf);

    TwoElectronBuffer<0, float> float_buf;
    engine::compute_eri_typed<float>(sa, sb, sc, sd, float_buf);

    auto errors = compute_2e_errors(double_buf, float_buf);
    EXPECT_LT(errors.max_relative, 1e-6)
        << "ERI (sp|sp): max_rel=" << errors.max_relative;
}

TEST(Float32ERIAccuracy, PpPpQuartet) {
    Shell sa = make_test_shell(1, {0.0, 0.0, 0.0});
    Shell sb = make_test_shell(1, {1.4, 0.0, 0.0});
    Shell sc = make_test_shell(1, {0.0, 1.4, 0.0});
    Shell sd = make_test_shell(1, {1.4, 1.4, 0.0});

    TwoElectronBuffer<0> double_buf;
    kernels::compute_eri(sa, sb, sc, sd, double_buf);

    TwoElectronBuffer<0, float> float_buf;
    engine::compute_eri_typed<float>(sa, sb, sc, sd, float_buf);

    auto errors = compute_2e_errors(double_buf, float_buf);
    EXPECT_LT(errors.max_relative, 1e-6)
        << "ERI (pp|pp): max_rel=" << errors.max_relative;
}

TEST(Float32ERIAccuracy, DdDdQuartet) {
    Shell sa = make_test_shell(2, {0.0, 0.0, 0.0});
    Shell sb = make_test_shell(2, {1.4, 0.0, 0.0});
    Shell sc = make_test_shell(2, {0.0, 1.4, 0.0});
    Shell sd = make_test_shell(2, {1.4, 1.4, 0.0});

    TwoElectronBuffer<0> double_buf;
    kernels::compute_eri(sa, sb, sc, sd, double_buf);

    TwoElectronBuffer<0, float> float_buf;
    engine::compute_eri_typed<float>(sa, sb, sc, sd, float_buf);

    auto errors = compute_2e_errors(double_buf, float_buf);
    EXPECT_LT(errors.max_relative, 1e-5)
        << "ERI (dd|dd): max_rel=" << errors.max_relative;
}

// ============================================================================
// Summary Statistics Test
// ============================================================================

TEST(Float32AccuracySummary, AllIntegralTypesReport) {
    // Test that all integral types achieve acceptable float32 accuracy
    // and print a summary report

    Shell s0a = make_test_shell(0, {0.0, 0.0, 0.0});
    Shell s0b = make_test_shell(0, {0.0, 0.0, 1.4});
    Shell p0a = make_test_shell(1, {0.0, 0.0, 0.0});
    Shell p0b = make_test_shell(1, {0.0, 0.0, 1.4});

    // Overlap
    {
        OverlapBuffer d_buf;
        kernels::compute_overlap(s0a, p0b, d_buf);
        OneElectronBuffer<0, float> f_buf;
        engine::compute_overlap_typed<float>(s0a, p0b, f_buf);
        auto err = compute_1e_errors(d_buf, f_buf);
        EXPECT_LT(err.max_relative, 1e-6);
    }

    // Kinetic
    {
        KineticBuffer d_buf;
        kernels::compute_kinetic(s0a, p0b, d_buf);
        OneElectronBuffer<0, float> f_buf;
        engine::compute_kinetic_typed<float>(s0a, p0b, f_buf);
        auto err = compute_1e_errors(d_buf, f_buf);
        EXPECT_LT(err.max_relative, 1e-6);
    }

    // Nuclear
    {
        PointChargeParams charges;
        charges.x = {0.0, 0.0};
        charges.y = {0.0, 0.0};
        charges.z = {0.0, 1.4};
        charges.charge = {1.0, 1.0};

        NuclearBuffer d_buf;
        kernels::compute_nuclear(s0a, p0b, charges, d_buf);
        OneElectronBuffer<0, float> f_buf;
        engine::compute_nuclear_typed<float>(s0a, p0b, charges, f_buf);
        auto err = compute_1e_errors(d_buf, f_buf);
        EXPECT_LT(err.max_relative, 1e-6);
    }

    // ERI
    {
        TwoElectronBuffer<0> d_buf;
        kernels::compute_eri(s0a, p0b, s0a, p0b, d_buf);
        TwoElectronBuffer<0, float> f_buf;
        engine::compute_eri_typed<float>(s0a, p0b, s0a, p0b, f_buf);
        auto err = compute_2e_errors(d_buf, f_buf);
        EXPECT_LT(err.max_relative, 1e-6);
    }
}

}  // namespace
}  // namespace libaccint::test
