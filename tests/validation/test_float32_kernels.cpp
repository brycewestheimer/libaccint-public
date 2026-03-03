// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2026 Bryce M. Westheimer. All rights reserved.

/// @file test_float32_kernels.cpp
/// @brief Float32 CPU kernel validation (Tasks 24.2.1, 24.2.2)
///
/// Validates that float32 one-electron and ERI kernels produce results
/// consistent with double-precision reference within float32 tolerance.
/// Tests explicit template instantiation via precision_dispatch.hpp.

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
#include <vector>

namespace libaccint::test {
namespace {

/// @brief Create a test shell for kernel validation
Shell make_kernel_test_shell(int am, Point3D center) {
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
        default:
            exponents = {1.208, 0.4537, 0.1813};
            coefficients = {0.30, 0.45, 0.35};
            break;
    }

    return Shell(am, center, exponents, coefficients);
}

// ============================================================================
// Float32 Overlap Kernel Tests (Task 24.2.1)
// ============================================================================

class Float32OverlapKernel : public ::testing::TestWithParam<std::pair<int,int>> {};

TEST_P(Float32OverlapKernel, ComputeAndCompare) {
    auto [la, lb] = GetParam();

    Shell shell_a = make_kernel_test_shell(la, {0.0, 0.0, 0.0});
    Shell shell_b = make_kernel_test_shell(lb, {0.8, 0.5, -0.3});

    // Double-precision reference
    OverlapBuffer ref_buf;
    kernels::compute_overlap(shell_a, shell_b, ref_buf);

    // Float32 via precision dispatch
    OneElectronBuffer<0, float> float_buf;
    engine::compute_overlap_typed<float>(shell_a, shell_b, float_buf);

    ASSERT_EQ(ref_buf.na(), float_buf.na());
    ASSERT_EQ(ref_buf.nb(), float_buf.nb());

    int na = ref_buf.na();
    int nb = ref_buf.nb();

    double max_rel_err = 0.0;
    for (int a = 0; a < na; ++a) {
        for (int b = 0; b < nb; ++b) {
            double d_val = ref_buf(a, b);
            double f_val = static_cast<double>(float_buf(a, b));

            if (std::abs(d_val) > 1e-15) {
                double rel_err = std::abs(d_val - f_val) / std::abs(d_val);
                max_rel_err = std::max(max_rel_err, rel_err);
            }
        }
    }

    EXPECT_LT(max_rel_err, 1e-6)
        << "Overlap (" << la << "," << lb << ") max relative error: " << max_rel_err;
}

INSTANTIATE_TEST_SUITE_P(
    AMCombinations, Float32OverlapKernel,
    ::testing::Values(
        std::make_pair(0, 0), std::make_pair(0, 1), std::make_pair(1, 0),
        std::make_pair(1, 1), std::make_pair(0, 2), std::make_pair(2, 0),
        std::make_pair(1, 2), std::make_pair(2, 1), std::make_pair(2, 2)
    )
);

// ============================================================================
// Float32 Kinetic Kernel Tests (Task 24.2.1)
// ============================================================================

class Float32KineticKernel : public ::testing::TestWithParam<std::pair<int,int>> {};

TEST_P(Float32KineticKernel, ComputeAndCompare) {
    auto [la, lb] = GetParam();

    Shell shell_a = make_kernel_test_shell(la, {0.0, 0.0, 0.0});
    Shell shell_b = make_kernel_test_shell(lb, {1.2, -0.3, 0.8});

    KineticBuffer ref_buf;
    kernels::compute_kinetic(shell_a, shell_b, ref_buf);

    OneElectronBuffer<0, float> float_buf;
    engine::compute_kinetic_typed<float>(shell_a, shell_b, float_buf);

    ASSERT_EQ(ref_buf.na(), float_buf.na());
    ASSERT_EQ(ref_buf.nb(), float_buf.nb());

    int na = ref_buf.na();
    int nb = ref_buf.nb();

    double max_rel_err = 0.0;
    for (int a = 0; a < na; ++a) {
        for (int b = 0; b < nb; ++b) {
            double d_val = ref_buf(a, b);
            double f_val = static_cast<double>(float_buf(a, b));

            if (std::abs(d_val) > 1e-15) {
                double rel_err = std::abs(d_val - f_val) / std::abs(d_val);
                max_rel_err = std::max(max_rel_err, rel_err);
            }
        }
    }

    EXPECT_LT(max_rel_err, 1e-6)
        << "Kinetic (" << la << "," << lb << ") max relative error: " << max_rel_err;
}

INSTANTIATE_TEST_SUITE_P(
    AMCombinations, Float32KineticKernel,
    ::testing::Values(
        std::make_pair(0, 0), std::make_pair(0, 1), std::make_pair(1, 0),
        std::make_pair(1, 1), std::make_pair(0, 2), std::make_pair(2, 0),
        std::make_pair(1, 2), std::make_pair(2, 1), std::make_pair(2, 2)
    )
);

// ============================================================================
// Float32 Nuclear Kernel Tests (Task 24.2.1)
// ============================================================================

class Float32NuclearKernel : public ::testing::TestWithParam<std::pair<int,int>> {};

TEST_P(Float32NuclearKernel, ComputeAndCompare) {
    auto [la, lb] = GetParam();

    Shell shell_a = make_kernel_test_shell(la, {0.0, 0.0, 0.0});
    Shell shell_b = make_kernel_test_shell(lb, {0.0, 0.0, 1.4});

    PointChargeParams charges;
    charges.x = {0.0, 0.0};
    charges.y = {0.0, 0.0};
    charges.z = {0.0, 1.4};
    charges.charge = {1.0, 1.0};

    NuclearBuffer ref_buf;
    kernels::compute_nuclear(shell_a, shell_b, charges, ref_buf);

    OneElectronBuffer<0, float> float_buf;
    engine::compute_nuclear_typed<float>(shell_a, shell_b, charges, float_buf);

    ASSERT_EQ(ref_buf.na(), float_buf.na());
    ASSERT_EQ(ref_buf.nb(), float_buf.nb());

    int na = ref_buf.na();
    int nb = ref_buf.nb();

    double max_rel_err = 0.0;
    for (int a = 0; a < na; ++a) {
        for (int b = 0; b < nb; ++b) {
            double d_val = ref_buf(a, b);
            double f_val = static_cast<double>(float_buf(a, b));

            if (std::abs(d_val) > 1e-15) {
                double rel_err = std::abs(d_val - f_val) / std::abs(d_val);
                max_rel_err = std::max(max_rel_err, rel_err);
            }
        }
    }

    EXPECT_LT(max_rel_err, 1e-6)
        << "Nuclear (" << la << "," << lb << ") max relative error: " << max_rel_err;
}

INSTANTIATE_TEST_SUITE_P(
    AMCombinations, Float32NuclearKernel,
    ::testing::Values(
        std::make_pair(0, 0), std::make_pair(0, 1), std::make_pair(1, 0),
        std::make_pair(1, 1), std::make_pair(0, 2), std::make_pair(2, 0),
        std::make_pair(1, 2), std::make_pair(2, 1), std::make_pair(2, 2)
    )
);

// ============================================================================
// Float32 ERI Kernel Tests (Task 24.2.2)
// ============================================================================

class Float32ERIKernel : public ::testing::TestWithParam<
    std::tuple<int, int, int, int>> {};

TEST_P(Float32ERIKernel, ComputeAndCompare) {
    auto [la, lb, lc, ld] = GetParam();

    Shell sa = make_kernel_test_shell(la, {0.0, 0.0, 0.0});
    Shell sb = make_kernel_test_shell(lb, {1.0, 0.0, 0.0});
    Shell sc = make_kernel_test_shell(lc, {0.0, 1.0, 0.0});
    Shell sd = make_kernel_test_shell(ld, {1.0, 1.0, 0.0});

    TwoElectronBuffer<0> ref_buf;
    kernels::compute_eri(sa, sb, sc, sd, ref_buf);

    TwoElectronBuffer<0, float> float_buf;
    engine::compute_eri_typed<float>(sa, sb, sc, sd, float_buf);

    ASSERT_EQ(ref_buf.na(), float_buf.na());
    ASSERT_EQ(ref_buf.nb(), float_buf.nb());
    ASSERT_EQ(ref_buf.nc(), float_buf.nc());
    ASSERT_EQ(ref_buf.nd(), float_buf.nd());

    int na = ref_buf.na();
    int nb = ref_buf.nb();
    int nc = ref_buf.nc();
    int nd = ref_buf.nd();

    double max_rel_err = 0.0;
    double max_abs_err = 0.0;

    for (int a = 0; a < na; ++a) {
        for (int b = 0; b < nb; ++b) {
            for (int c = 0; c < nc; ++c) {
                for (int d = 0; d < nd; ++d) {
                    double d_val = ref_buf(a, b, c, d);
                    double f_val = static_cast<double>(float_buf(a, b, c, d));
                    double abs_err = std::abs(d_val - f_val);
                    max_abs_err = std::max(max_abs_err, abs_err);

                    if (std::abs(d_val) > 1e-15) {
                        double rel_err = abs_err / std::abs(d_val);
                        max_rel_err = std::max(max_rel_err, rel_err);
                    }
                }
            }
        }
    }

    // Allow slightly more tolerance for higher AM ERIs
    double tolerance = (la + lb + lc + ld > 4) ? 1e-5 : 1e-6;
    EXPECT_LT(max_rel_err, tolerance)
        << "ERI (" << la << lb << "|" << lc << ld << ") max relative error: " << max_rel_err
        << ", max absolute error: " << max_abs_err;
}

INSTANTIATE_TEST_SUITE_P(
    AMCombinations, Float32ERIKernel,
    ::testing::Values(
        std::make_tuple(0, 0, 0, 0),  // (ss|ss)
        std::make_tuple(0, 0, 0, 1),  // (ss|sp)
        std::make_tuple(0, 1, 0, 1),  // (sp|sp)
        std::make_tuple(1, 1, 0, 0),  // (pp|ss)
        std::make_tuple(1, 1, 1, 1),  // (pp|pp)
        std::make_tuple(0, 0, 2, 2),  // (ss|dd)
        std::make_tuple(0, 2, 0, 2),  // (sd|sd)
        std::make_tuple(1, 1, 2, 2),  // (pp|dd)
        std::make_tuple(2, 2, 2, 2)   // (dd|dd)
    )
);

// ============================================================================
// Buffer Conversion Tests
// ============================================================================

TEST(Float32BufferConversion, OneElectronRoundTrip) {
    Shell sa = make_kernel_test_shell(1, {0.0, 0.0, 0.0});
    Shell sb = make_kernel_test_shell(1, {1.0, 0.5, -0.3});

    OverlapBuffer double_buf;
    kernels::compute_overlap(sa, sb, double_buf);

    // Convert to float
    auto float_buf = double_buf.to_precision<float>();

    // Convert back to double
    auto back_to_double = float_buf.to_precision<double>();

    int na = double_buf.na();
    int nb = double_buf.nb();
    for (int a = 0; a < na; ++a) {
        for (int b = 0; b < nb; ++b) {
            double original = double_buf(a, b);
            double round_tripped = back_to_double(a, b);
            if (std::abs(original) > 1e-15) {
                double rel_err = std::abs(original - round_tripped) / std::abs(original);
                // Round-trip loses float32 precision
                EXPECT_LT(rel_err, 1e-6);
            }
        }
    }
}

TEST(Float32BufferConversion, TwoElectronRoundTrip) {
    Shell sa = make_kernel_test_shell(0, {0.0, 0.0, 0.0});
    Shell sb = make_kernel_test_shell(1, {1.0, 0.0, 0.0});

    TwoElectronBuffer<0> double_buf;
    kernels::compute_eri(sa, sa, sb, sb, double_buf);

    // Convert to float
    auto float_buf = double_buf.to_precision<float>();

    // Convert back to double
    auto back_to_double = float_buf.to_precision<double>();

    int na = double_buf.na(), nb = double_buf.nb();
    int nc = double_buf.nc(), nd = double_buf.nd();
    for (int a = 0; a < na; ++a) {
        for (int b = 0; b < nb; ++b) {
            for (int c = 0; c < nc; ++c) {
                for (int d = 0; d < nd; ++d) {
                    double original = double_buf(a, b, c, d);
                    double round_tripped = back_to_double(a, b, c, d);
                    if (std::abs(original) > 1e-15) {
                        double rel_err = std::abs(original - round_tripped)
                                         / std::abs(original);
                        EXPECT_LT(rel_err, 1e-6);
                    }
                }
            }
        }
    }
}

// ============================================================================
// Precision Dispatch Infrastructure Tests
// ============================================================================

TEST(PrecisionDispatch, DispatchOnPrecisionFloat) {
    auto result = engine::dispatch_on_precision(Precision::Float32,
        []([[maybe_unused]] auto tag) {
            using T = typename decltype(tag)::type;
            return sizeof(T);
        });
    EXPECT_EQ(result, sizeof(float));
}

TEST(PrecisionDispatch, DispatchOnPrecisionDouble) {
    auto result = engine::dispatch_on_precision(Precision::Float64,
        []([[maybe_unused]] auto tag) {
            using T = typename decltype(tag)::type;
            return sizeof(T);
        });
    EXPECT_EQ(result, sizeof(double));
}

TEST(PrecisionDispatch, DispatchOnPrecisionAuto) {
    auto result = engine::dispatch_on_precision(Precision::Auto,
        []([[maybe_unused]] auto tag) {
            using T = typename decltype(tag)::type;
            return sizeof(T);
        });
    EXPECT_EQ(result, sizeof(double));  // Auto defaults to double
}

}  // namespace
}  // namespace libaccint::test
