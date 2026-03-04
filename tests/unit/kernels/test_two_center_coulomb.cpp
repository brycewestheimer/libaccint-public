// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2026 Bryce M. Westheimer. All rights reserved.

/// @file test_two_center_coulomb.cpp
/// @brief Unit tests for two-center Coulomb kernel

#include <gtest/gtest.h>

#include <libaccint/kernels/two_center_coulomb_kernel.hpp>
#include <libaccint/basis/shell.hpp>
#include <libaccint/buffers/two_electron_buffer.hpp>
#include <libaccint/core/types.hpp>

#include <cmath>

namespace libaccint::testing {

// =============================================================================
// Test Utilities
// =============================================================================

/// @brief Create an s-type auxiliary shell
Shell create_aux_s_shell(Point3D center,
                          std::vector<Real> exponents,
                          std::vector<Real> coefficients) {
    return Shell(0, center, exponents, coefficients);
}

/// @brief Create a p-type auxiliary shell
Shell create_aux_p_shell(Point3D center,
                          std::vector<Real> exponents,
                          std::vector<Real> coefficients) {
    return Shell(1, center, exponents, coefficients);
}

/// @brief Compute reference two-center integral for s-s case
///
/// (s|s) = 2 * pi^(5/2) / (zeta^(3/2)) * F_0(T) * K
/// where T = rho * R^2, rho = alpha_P * alpha_Q / (alpha_P + alpha_Q)
Real reference_ss_integral(Real alpha_P, Real alpha_Q,
                            Real coef_P, Real coef_Q,
                            Real R2) {
    const Real zeta = alpha_P + alpha_Q;
    const Real rho = alpha_P * alpha_Q / zeta;
    const Real T = rho * R2;
    const Real K = std::exp(-rho * R2);

    // Boys function F_0(T) = sqrt(pi) / (2 * sqrt(T)) * erf(sqrt(T))
    // For T -> 0: F_0(T) -> 1
    Real F0;
    if (T < 1e-10) {
        F0 = 1.0;
    } else {
        F0 = std::sqrt(M_PI / T) * 0.5 * std::erf(std::sqrt(T));
    }

    return 2.0 * std::pow(M_PI, 2.5) / (zeta * std::sqrt(zeta)) * K * F0 *
           coef_P * coef_Q;
}

// =============================================================================
// Basic Integral Tests
// =============================================================================

TEST(TwoCenterCoulomb, SameCenter_SS) {
    // Two s-shells at the same center
    Shell P = create_aux_s_shell({0.0, 0.0, 0.0}, {1.5}, {1.0});
    Shell Q = create_aux_s_shell({0.0, 0.0, 0.0}, {1.5}, {1.0});

    TwoElectronBuffer<0> buffer;
    kernels::compute_two_center_coulomb(P, Q, buffer);

    Real integral = buffer(0, 0, 0, 0);
    Real reference = reference_ss_integral(1.5, 1.5,
                                            P.coefficients()[0], Q.coefficients()[0], 0.0);

    EXPECT_NEAR(integral, reference, 1e-10);
    EXPECT_GT(integral, 0.0);  // Should be positive (self-overlap)
}

TEST(TwoCenterCoulomb, SeparatedCenters_SS) {
    // Two s-shells separated by 1 Bohr
    Shell P = create_aux_s_shell({0.0, 0.0, 0.0}, {1.5}, {1.0});
    Shell Q = create_aux_s_shell({1.0, 0.0, 0.0}, {1.5}, {1.0});

    TwoElectronBuffer<0> buffer;
    kernels::compute_two_center_coulomb(P, Q, buffer);

    Real integral = buffer(0, 0, 0, 0);
    Real reference = reference_ss_integral(1.5, 1.5,
                                            P.coefficients()[0], Q.coefficients()[0], 1.0);

    EXPECT_NEAR(integral, reference, 1e-10);
    EXPECT_GT(integral, 0.0);
}

TEST(TwoCenterCoulomb, DifferentExponents_SS) {
    Shell P = create_aux_s_shell({0.0, 0.0, 0.0}, {2.0}, {1.0});
    Shell Q = create_aux_s_shell({0.5, 0.0, 0.0}, {0.5}, {1.0});

    TwoElectronBuffer<0> buffer;
    kernels::compute_two_center_coulomb(P, Q, buffer);

    Real integral = buffer(0, 0, 0, 0);
    Real reference = reference_ss_integral(2.0, 0.5,
                                            P.coefficients()[0], Q.coefficients()[0], 0.25);

    EXPECT_NEAR(integral, reference, 1e-10);
}

// =============================================================================
// Symmetry Tests
// =============================================================================

TEST(TwoCenterCoulomb, Symmetry_PQ_QP) {
    Shell P = create_aux_s_shell({0.0, 0.0, 0.0}, {1.5}, {1.0});
    Shell Q = create_aux_s_shell({1.0, 0.5, 0.0}, {2.0}, {1.0});

    TwoElectronBuffer<0> buffer_PQ, buffer_QP;
    kernels::compute_two_center_coulomb(P, Q, buffer_PQ);
    kernels::compute_two_center_coulomb(Q, P, buffer_QP);

    EXPECT_NEAR(buffer_PQ(0, 0, 0, 0), buffer_QP(0, 0, 0, 0), 1e-14);
}

TEST(TwoCenterCoulomb, Symmetry_PP) {
    // p-shell: (px|py) should be zero by symmetry
    Shell P = create_aux_p_shell({0.0, 0.0, 0.0}, {1.0}, {1.0});

    TwoElectronBuffer<0> buffer;
    kernels::compute_two_center_coulomb(P, P, buffer);

    // Diagonal elements (px|px), (py|py), (pz|pz) should be equal
    EXPECT_NEAR(buffer(0, 0, 0, 0), buffer(1, 1, 0, 0), 1e-12);
    EXPECT_NEAR(buffer(0, 0, 0, 0), buffer(2, 2, 0, 0), 1e-12);

    // Off-diagonal (px|py), (px|pz), (py|pz) should be zero
    EXPECT_NEAR(buffer(0, 1, 0, 0), 0.0, 1e-12);
    EXPECT_NEAR(buffer(0, 2, 0, 0), 0.0, 1e-12);
    EXPECT_NEAR(buffer(1, 2, 0, 0), 0.0, 1e-12);
}

// =============================================================================
// Contracted Shell Tests
// =============================================================================

TEST(TwoCenterCoulomb, ContractedShell) {
    // Contracted s-shell (2 primitives)
    Shell P = create_aux_s_shell({0.0, 0.0, 0.0}, {2.0, 0.5}, {0.6, 0.4});
    Shell Q = create_aux_s_shell({1.0, 0.0, 0.0}, {1.5}, {1.0});

    TwoElectronBuffer<0> buffer;
    kernels::compute_two_center_coulomb(P, Q, buffer);

    Real integral = buffer(0, 0, 0, 0);

    // Compute reference by summing primitive contributions with normalized coefficients
    Real reference = 0.0;
    const auto& exps_P = P.exponents();
    const auto& coefs_P = P.coefficients();
    const auto& exps_Q = Q.exponents();
    const auto& coefs_Q = Q.coefficients();
    for (Size i = 0; i < exps_P.size(); ++i) {
        for (Size j = 0; j < exps_Q.size(); ++j) {
            reference += reference_ss_integral(
                exps_P[i], exps_Q[j], coefs_P[i], coefs_Q[j], 1.0);
        }
    }

    EXPECT_NEAR(integral, reference, 1e-10);
}

// =============================================================================
// Metric Matrix Tests
// =============================================================================

TEST(TwoCenterCoulomb, MetricMatrix) {
    std::vector<Shell> shells;
    shells.push_back(create_aux_s_shell({0.0, 0.0, 0.0}, {1.5}, {1.0}));
    shells.push_back(create_aux_s_shell({1.0, 0.0, 0.0}, {1.5}, {1.0}));
    shells.push_back(create_aux_s_shell({0.0, 1.0, 0.0}, {1.5}, {1.0}));

    const Size n_aux = 3;
    std::vector<Real> metric(n_aux * n_aux);

    kernels::compute_two_center_metric(shells, metric.data(), n_aux);

    // Check symmetry
    for (Size i = 0; i < n_aux; ++i) {
        for (Size j = 0; j < n_aux; ++j) {
            EXPECT_NEAR(metric[i * n_aux + j], metric[j * n_aux + i], 1e-14);
        }
    }

    // Check positive definiteness (diagonal elements positive)
    for (Size i = 0; i < n_aux; ++i) {
        EXPECT_GT(metric[i * n_aux + i], 0.0);
    }
}

// =============================================================================
// Block Interface Tests
// =============================================================================

TEST(TwoCenterCoulomb, BlockInterface) {
    Shell P = create_aux_p_shell({0.0, 0.0, 0.0}, {1.0}, {1.0});
    Shell Q = create_aux_s_shell({1.0, 0.0, 0.0}, {1.5}, {1.0});

    const int np = 3;  // p-shell has 3 functions
    const int nq = 1;  // s-shell has 1 function

    std::vector<Real> block(np * nq);
    kernels::compute_two_center_coulomb_block(P, Q, block.data());

    // Compare with buffer interface
    TwoElectronBuffer<0> buffer;
    kernels::compute_two_center_coulomb(P, Q, buffer);

    for (int ip = 0; ip < np; ++ip) {
        for (int iq = 0; iq < nq; ++iq) {
            EXPECT_NEAR(block[ip * nq + iq], buffer(ip, iq, 0, 0), 1e-14);
        }
    }
}

}  // namespace libaccint::testing
