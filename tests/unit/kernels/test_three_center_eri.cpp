// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2026 Bryce M. Westheimer. All rights reserved.

/// @file test_three_center_eri.cpp
/// @brief Unit tests for three-center ERI kernel

#include <gtest/gtest.h>

#include <libaccint/kernels/three_center_eri_kernel.hpp>
#include <libaccint/basis/shell.hpp>
#include <libaccint/buffers/two_electron_buffer.hpp>
#include <libaccint/core/types.hpp>

#include <cmath>

namespace libaccint::testing {

// =============================================================================
// Test Utilities
// =============================================================================

Shell create_s_shell(Point3D center, Real exponent, Real coef = 1.0) {
    return Shell(0, center, {exponent}, {coef});
}

Shell create_p_shell(Point3D center, Real exponent, Real coef = 1.0) {
    return Shell(1, center, {exponent}, {coef});
}

// =============================================================================
// Basic Integral Tests
// =============================================================================

TEST(ThreeCenterERI, SSS_SameCenter) {
    // (ss|s) with all shells at origin
    Shell a = create_s_shell({0.0, 0.0, 0.0}, 1.0);
    Shell b = create_s_shell({0.0, 0.0, 0.0}, 1.0);
    Shell P = create_s_shell({0.0, 0.0, 0.0}, 1.0);

    TwoElectronBuffer<0> buffer;
    kernels::compute_three_center_eri(a, b, P, buffer);

    Real integral = buffer(0, 0, 0, 0);

    // Should be positive and non-zero
    EXPECT_GT(integral, 0.0);
    EXPECT_FALSE(std::isnan(integral));
}

TEST(ThreeCenterERI, SSS_SeparatedCenters) {
    Shell a = create_s_shell({0.0, 0.0, 0.0}, 1.0);
    Shell b = create_s_shell({1.0, 0.0, 0.0}, 1.0);
    Shell P = create_s_shell({0.5, 0.5, 0.0}, 1.5);

    TwoElectronBuffer<0> buffer;
    kernels::compute_three_center_eri(a, b, P, buffer);

    Real integral = buffer(0, 0, 0, 0);

    EXPECT_GT(integral, 0.0);
    EXPECT_FALSE(std::isnan(integral));
}

// =============================================================================
// Symmetry Tests
// =============================================================================

TEST(ThreeCenterERI, Symmetry_ab_ba) {
    // (ab|P) = (ba|P)
    Shell a = create_s_shell({0.0, 0.0, 0.0}, 1.2);
    Shell b = create_s_shell({1.0, 0.0, 0.0}, 0.8);
    Shell P = create_s_shell({0.5, 0.5, 0.0}, 1.5);

    TwoElectronBuffer<0> buffer_ab, buffer_ba;
    kernels::compute_three_center_eri(a, b, P, buffer_ab);
    kernels::compute_three_center_eri(b, a, P, buffer_ba);

    EXPECT_NEAR(buffer_ab(0, 0, 0, 0), buffer_ba(0, 0, 0, 0), 1e-12);
}

TEST(ThreeCenterERI, Symmetry_PP) {
    // For (ss|P) with p-shell P, px, py, pz should have specific symmetry
    Shell a = create_s_shell({0.0, 0.0, 0.0}, 1.0);
    Shell b = create_s_shell({0.0, 0.0, 0.0}, 1.0);
    Shell P = create_p_shell({1.0, 0.0, 0.0}, 1.0);  // Displaced along x

    TwoElectronBuffer<0> buffer;
    kernels::compute_three_center_eri(a, b, P, buffer);

    // (ss|px) should be non-zero (center displaced along x)
    // (ss|py) and (ss|pz) should be zero by symmetry
    EXPECT_NE(buffer(0, 0, 0, 0), 0.0);  // px
    EXPECT_NEAR(buffer(0, 0, 1, 0), 0.0, 1e-12);  // py
    EXPECT_NEAR(buffer(0, 0, 2, 0), 0.0, 1e-12);  // pz
}

// =============================================================================
// Higher Angular Momentum Tests
// =============================================================================

TEST(ThreeCenterERI, PSS) {
    Shell a = create_p_shell({0.0, 0.0, 0.0}, 1.0);
    Shell b = create_s_shell({1.0, 0.0, 0.0}, 1.0);
    Shell P = create_s_shell({0.5, 0.0, 0.0}, 1.5);

    TwoElectronBuffer<0> buffer;
    kernels::compute_three_center_eri(a, b, P, buffer);

    // Should have 3 integrals (px, py, pz with s and s)
    // Check none are NaN
    EXPECT_FALSE(std::isnan(buffer(0, 0, 0, 0)));
    EXPECT_FALSE(std::isnan(buffer(1, 0, 0, 0)));
    EXPECT_FALSE(std::isnan(buffer(2, 0, 0, 0)));
}

TEST(ThreeCenterERI, PPS) {
    Shell a = create_p_shell({0.0, 0.0, 0.0}, 1.0);
    Shell b = create_p_shell({1.0, 0.0, 0.0}, 1.0);
    Shell P = create_s_shell({0.5, 0.0, 0.0}, 1.5);

    TwoElectronBuffer<0> buffer;
    kernels::compute_three_center_eri(a, b, P, buffer);

    // Should have 9 integrals (3 x 3 for pp)
    for (int ia = 0; ia < 3; ++ia) {
        for (int ib = 0; ib < 3; ++ib) {
            EXPECT_FALSE(std::isnan(buffer(ia, ib, 0, 0)));
        }
    }
}

// =============================================================================
// Tensor Interface Tests
// =============================================================================

TEST(ThreeCenterERI, TensorComputation) {
    std::vector<Shell> orbital_shells;
    orbital_shells.push_back(create_s_shell({0.0, 0.0, 0.0}, 1.0));
    orbital_shells.push_back(create_s_shell({1.0, 0.0, 0.0}, 1.0));

    std::vector<Shell> aux_shells;
    aux_shells.push_back(create_s_shell({0.5, 0.0, 0.0}, 1.5));
    aux_shells.push_back(create_s_shell({0.0, 1.0, 0.0}, 1.5));

    const Size n_orb = 2;
    const Size n_aux = 2;

    std::vector<Real> tensor(n_orb * n_orb * n_aux);
    kernels::compute_three_center_tensor(
        orbital_shells, aux_shells,
        tensor.data(), n_orb, n_aux,
        kernels::ThreeCenterStorageFormat::abP);

    // Check symmetry: (ab|P) = (ba|P)
    for (Size P = 0; P < n_aux; ++P) {
        for (Size a = 0; a < n_orb; ++a) {
            for (Size b = 0; b < n_orb; ++b) {
                EXPECT_NEAR(
                    tensor[a * n_orb * n_aux + b * n_aux + P],
                    tensor[b * n_orb * n_aux + a * n_aux + P],
                    1e-12);
            }
        }
    }
}

// =============================================================================
// B Tensor Tests
// =============================================================================

TEST(ThreeCenterERI, BTensorComputation) {
    const Size n_orb = 2;
    const Size n_aux = 2;

    // Create dummy three-center and L_inv matrices
    std::vector<Real> three_center(n_orb * n_orb * n_aux);
    std::vector<Real> L_inv(n_aux * n_aux);
    std::vector<Real> B_tensor(n_orb * n_orb * n_aux);

    // Initialize with simple values
    for (Size i = 0; i < n_orb * n_orb * n_aux; ++i) {
        three_center[i] = static_cast<Real>(i + 1);
    }

    // L_inv = identity for testing
    L_inv[0] = 1.0; L_inv[1] = 0.0;
    L_inv[2] = 0.0; L_inv[3] = 1.0;

    kernels::compute_B_tensor(three_center.data(), L_inv.data(),
                               B_tensor.data(), n_orb, n_aux);

    // With identity L_inv, B should equal three_center
    for (Size i = 0; i < n_orb * n_orb * n_aux; ++i) {
        EXPECT_NEAR(B_tensor[i], three_center[i], 1e-12);
    }
}

// =============================================================================
// Block Interface Tests
// =============================================================================

TEST(ThreeCenterERI, BlockInterface) {
    Shell a = create_p_shell({0.0, 0.0, 0.0}, 1.0);
    Shell b = create_s_shell({1.0, 0.0, 0.0}, 1.0);
    Shell P = create_s_shell({0.5, 0.0, 0.0}, 1.5);

    const int na = 3;  // p-shell
    const int nb = 1;  // s-shell
    const int np = 1;  // s-shell

    std::vector<Real> block(na * nb * np);
    kernels::compute_three_center_eri_block(a, b, P, block.data());

    // Compare with buffer interface
    TwoElectronBuffer<0> buffer;
    kernels::compute_three_center_eri(a, b, P, buffer);

    for (int ia = 0; ia < na; ++ia) {
        for (int ib = 0; ib < nb; ++ib) {
            for (int ip = 0; ip < np; ++ip) {
                EXPECT_NEAR(
                    block[ia * nb * np + ib * np + ip],
                    buffer(ia, ib, ip, 0),
                    1e-14);
            }
        }
    }
}

}  // namespace libaccint::testing
