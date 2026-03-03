// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2026 Bryce M. Westheimer. All rights reserved.

/// @file test_overlap_kernel.cpp
/// @brief Unit tests for overlap integral kernel (Obara-Saika recursion)

#include <libaccint/kernels/overlap_kernel.hpp>
#include <libaccint/basis/shell.hpp>
#include <libaccint/basis/shell_set.hpp>
#include <libaccint/basis/shell_set_pair.hpp>
#include <libaccint/buffers/one_electron_buffer.hpp>
#include <libaccint/math/normalization.hpp>
#include <libaccint/math/cartesian_indices.hpp>
#include <libaccint/utils/constants.hpp>
#include <gtest/gtest.h>
#include <cmath>
#include <functional>

using namespace libaccint;

namespace {

// Tolerances for floating-point comparisons
constexpr Real TIGHT_TOL = 1e-12;
constexpr Real LOOSE_TOL = 1e-10;

// ============================================================================
// Helper: Create STO-3G hydrogen 1s shell at given position
// ============================================================================
Shell make_sto3g_hydrogen(Point3D center) {
    // STO-3G basis for hydrogen: 3 primitives
    std::vector<Real> exponents = {3.425250914, 0.6239137298, 0.168855404};
    std::vector<Real> coefficients = {0.1543289673, 0.5353281423, 0.4446345422};
    return Shell(AngularMomentum::S, center, exponents, coefficients);
}

// ============================================================================
// Helper: Create STO-3G oxygen shells at given position
// ============================================================================
// Oxygen 1s shell
Shell make_sto3g_oxygen_1s(Point3D center) {
    std::vector<Real> exponents = {130.7093214, 23.80886605, 6.443608313};
    std::vector<Real> coefficients = {0.1543289673, 0.5353281423, 0.4446345422};
    return Shell(AngularMomentum::S, center, exponents, coefficients);
}

// Oxygen 2s shell
Shell make_sto3g_oxygen_2s(Point3D center) {
    std::vector<Real> exponents = {5.033151319, 1.169596125, 0.38038896};
    std::vector<Real> coefficients = {-0.09996722919, 0.3995128261, 0.7001154689};
    return Shell(AngularMomentum::S, center, exponents, coefficients);
}

// Oxygen 2p shell
Shell make_sto3g_oxygen_2p(Point3D center) {
    std::vector<Real> exponents = {5.033151319, 1.169596125, 0.38038896};
    std::vector<Real> coefficients = {0.1559162750, 0.6076837186, 0.3919573931};
    return Shell(AngularMomentum::P, center, exponents, coefficients);
}

}  // anonymous namespace

// =============================================================================
// Test 1: s-s overlap at same center should give S = 1.0
// =============================================================================
TEST(OverlapKernelTest, SsSameCenter) {
    Point3D origin(0.0, 0.0, 0.0);
    Shell shell_a(AngularMomentum::S, origin, {1.0}, {1.0});
    Shell shell_b(AngularMomentum::S, origin, {1.0}, {1.0});

    OverlapBuffer buffer;
    kernels::compute_overlap(shell_a, shell_b, buffer);

    EXPECT_EQ(buffer.na(), 1);
    EXPECT_EQ(buffer.nb(), 1);
    EXPECT_NEAR(buffer(0, 0), 1.0, TIGHT_TOL)
        << "Self-overlap of identical s-shells should be 1.0";
}

// =============================================================================
// Test 2: s-s overlap at different centers (analytical verification)
// =============================================================================
TEST(OverlapKernelTest, SsDifferentCentersAnalytical) {
    // Two single-primitive s-shells with alpha=beta=1.0 separated by R=1 bohr
    // Expected: S = exp(-0.5) (see derivation in task description)
    const Real R = 1.0;
    Point3D A(0.0, 0.0, 0.0);
    Point3D B(0.0, 0.0, R);

    Shell shell_a(AngularMomentum::S, A, {1.0}, {1.0});
    Shell shell_b(AngularMomentum::S, B, {1.0}, {1.0});

    OverlapBuffer buffer;
    kernels::compute_overlap(shell_a, shell_b, buffer);

    const Real expected = std::exp(-0.5);
    EXPECT_NEAR(buffer(0, 0), expected, TIGHT_TOL)
        << "s-s overlap at R=1 with alpha=beta=1: expected exp(-0.5)";
}

// =============================================================================
// Test 3: s-s overlap with different exponents at different centers
// =============================================================================
TEST(OverlapKernelTest, SsDifferentExponents) {
    // Two single-primitive s-shells with alpha=2.0, beta=0.5, R=2.0 bohr along x
    const Real alpha = 2.0;
    const Real beta = 0.5;
    const Real R = 2.0;

    Point3D A(0.0, 0.0, 0.0);
    Point3D B(R, 0.0, 0.0);

    Shell shell_a(AngularMomentum::S, A, {alpha}, {1.0});
    Shell shell_b(AngularMomentum::S, B, {beta}, {1.0});

    OverlapBuffer buffer;
    kernels::compute_overlap(shell_a, shell_b, buffer);

    // Analytical: S = N_a * N_b * (pi/zeta)^(3/2) * exp(-mu * R^2)
    // N_a = (2*alpha/pi)^(3/4), N_b = (2*beta/pi)^(3/4)
    // zeta = alpha + beta = 2.5
    // mu = alpha*beta/zeta = 1.0/2.5 = 0.4
    // K_AB = exp(-mu * R^2) = exp(-0.4 * 4) = exp(-1.6)
    const Real zeta = alpha + beta;
    const Real mu = alpha * beta / zeta;
    const Real Na = std::pow(2.0 * alpha / constants::PI, 0.75);
    const Real Nb = std::pow(2.0 * beta / constants::PI, 0.75);
    const Real expected = Na * Nb * std::pow(constants::PI / zeta, 1.5)
                          * std::exp(-mu * R * R);

    EXPECT_NEAR(buffer(0, 0), expected, TIGHT_TOL)
        << "s-s overlap with different exponents at R=2.0";
}

// =============================================================================
// Test 4: s-p overlap at same center
// =============================================================================
TEST(OverlapKernelTest, SpSameCenter) {
    // s and p shells at the same center
    // By symmetry, <s|px> = <s|py> = <s|pz> = 0 at same center
    // because x*exp(-alpha*r^2) is odd in x and s is symmetric
    Point3D origin(0.0, 0.0, 0.0);
    Shell shell_s(AngularMomentum::S, origin, {1.0}, {1.0});
    Shell shell_p(AngularMomentum::P, origin, {1.0}, {1.0});

    OverlapBuffer buffer;
    kernels::compute_overlap(shell_s, shell_p, buffer);

    EXPECT_EQ(buffer.na(), 1);
    EXPECT_EQ(buffer.nb(), 3);

    for (int b = 0; b < 3; ++b) {
        EXPECT_NEAR(buffer(0, b), 0.0, TIGHT_TOL)
            << "s-p overlap at same center should be zero (component " << b << ")";
    }
}

// =============================================================================
// Test 5: s-p overlap at different centers
// =============================================================================
TEST(OverlapKernelTest, SpDifferentCenters) {
    // s at origin, p at (0, 0, R)
    // By symmetry: <s|px> = <s|py> = 0, but <s|pz> != 0
    const Real R = 1.0;
    Point3D A(0.0, 0.0, 0.0);
    Point3D B(0.0, 0.0, R);

    Shell shell_s(AngularMomentum::S, A, {1.0}, {1.0});
    Shell shell_p(AngularMomentum::P, B, {1.0}, {1.0});

    OverlapBuffer buffer;
    kernels::compute_overlap(shell_s, shell_p, buffer);

    EXPECT_EQ(buffer.na(), 1);
    EXPECT_EQ(buffer.nb(), 3);

    // px component: S(s, px) should be 0 (symmetry in x)
    EXPECT_NEAR(buffer(0, 0), 0.0, TIGHT_TOL) << "S(s, px) should be zero";

    // py component: S(s, py) should be 0 (symmetry in y)
    EXPECT_NEAR(buffer(0, 1), 0.0, TIGHT_TOL) << "S(s, py) should be zero";

    // pz component: S(s, pz) should be nonzero
    // Analytical for single primitives alpha=beta=1, separation along z by R=1:
    // I_x(0,0)=1, I_y(0,0)=1, I_z(0,1) = XPB_z * I_z(0,0) = (P_z - B_z) * 1
    // P_z = (alpha*0 + beta*R)/zeta = R/2 = 0.5
    // XPB_z = P_z - B_z = 0.5 - 1.0 = -0.5
    // I_z(0,1) = -0.5
    //
    // S = N_s * N_p * (pi/zeta)^(3/2) * K_AB * I_x(0,0) * I_y(0,0) * I_z(0,1)
    //   * norm_correction_p(0,0,1)
    //
    // N_s = (2/pi)^(3/4) for alpha=1
    // N_p = (2/pi)^(3/4) * (4*1)^(1/2) / sqrt(1!!) = (2/pi)^(3/4) * 2
    // norm_correction for (0,0,1) with L=1: sqrt(1!!/1!!) = 1
    //
    // zeta = 2.0, mu = 0.5, K_AB = exp(-0.5 * 1.0) = exp(-0.5)
    // prefactor = (pi/2)^(3/2) * exp(-0.5)
    //
    // S = (2/pi)^(3/4) * [(2/pi)^(3/4) * 2] * (pi/2)^(3/2) * exp(-0.5) * (-0.5)
    // = (2/pi)^(3/2) * 2 * (pi/2)^(3/2) * exp(-0.5) * (-0.5)
    // = 2 * 1.0 * exp(-0.5) * (-0.5)
    // = -exp(-0.5)
    //
    // Wait, but the Shell constructor normalizes differently for p-shells.
    // Let me just verify it's nonzero and has the right sign.
    // With alpha=beta=1, R=1 along z, the stored coefficient for p-shell
    // includes N_p(1, L=1) which absorbs the (4*alpha)^(1/2) factor.
    //
    // Instead, let's compute this numerically for the stored coefficients.
    // For a single-primitive p-shell with alpha=1:
    //   c_stored = c_raw * primitive_norm(1, 1.0) * contraction_norm
    //   primitive_norm(1, 1.0) = (2/pi)^(3/4) * (4)^(1/2) / sqrt(1!!)
    //                          = (2/pi)^(3/4) * 2
    //   self_overlap = c_prim_norm^2 * (pi/2)^(3/2) * (0.5/2) = c_prim_norm^2 * (pi/2)^(3/2) * 0.25
    //   where c_prim_norm = (2/pi)^(3/4) * 2
    //   c_prim_norm^2 = (2/pi)^(3/2) * 4
    //   self_overlap = (2/pi)^(3/2) * 4 * (pi/2)^(3/2) * 0.25 = 4 * 0.25 = 1.0
    //   So contraction_norm = 1.0, and c_stored = (2/pi)^(3/4) * 2
    //
    // S(s, pz) = c_s * c_p * (pi/2)^(3/2) * exp(-0.5) * (-0.5) * 1.0 (corr)
    //          = (2/pi)^(3/4) * [(2/pi)^(3/4) * 2] * (pi/2)^(3/2) * exp(-0.5) * (-0.5)
    //          = (2/pi)^(3/2) * 2 * (pi/2)^(3/2) * exp(-0.5) * (-0.5)
    //          = 2 * exp(-0.5) * (-0.5)
    //          = -exp(-0.5)
    //          = -0.60653...

    const Real expected_pz = -std::exp(-0.5);
    EXPECT_NEAR(buffer(0, 2), expected_pz, TIGHT_TOL)
        << "S(s, pz) at R=1 with alpha=beta=1";
}

// =============================================================================
// Test 6: p-p overlap at same center (self-overlap matrix)
// =============================================================================
TEST(OverlapKernelTest, PpSameCenter) {
    // Two identical p-shells at the same center
    // Should give identity matrix (all p-components have equal normalization)
    Point3D origin(0.0, 0.0, 0.0);
    Shell shell(AngularMomentum::P, origin, {1.0}, {1.0});

    OverlapBuffer buffer;
    kernels::compute_overlap(shell, shell, buffer);

    EXPECT_EQ(buffer.na(), 3);
    EXPECT_EQ(buffer.nb(), 3);

    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            const Real expected = (i == j) ? 1.0 : 0.0;
            EXPECT_NEAR(buffer(i, j), expected, TIGHT_TOL)
                << "p-p overlap at same center: (" << i << "," << j << ")";
        }
    }
}

// =============================================================================
// Test 7: d-d overlap at same center (verifies normalization correction)
// =============================================================================
TEST(OverlapKernelTest, DdSameCenter) {
    // Two identical d-shells at the same center.
    //
    // IMPORTANT: Cartesian d-functions are NOT all orthogonal!
    // The 6 Cartesian d-functions span a 6D space = 5 spherical d + 1 s-type (r^2).
    // So dxx, dyy, dzz are NOT orthogonal to each other.
    //
    // Canonical ordering: (2,0,0), (1,1,0), (1,0,1), (0,2,0), (0,1,1), (0,0,2)
    // Indices:                 0,       1,       2,       3,       4,       5
    //
    // For the self-overlap matrix of normalized Cartesian d-functions at same center:
    //   S(dxx, dxx) = 1.0    (diagonal)
    //   S(dxy, dxy) = 1.0    (diagonal)
    //   S(dxx, dyy) = 1/3    (both are "squared" components, share s-character)
    //   S(dxx, dxy) = 0.0    (x^2 and xy are orthogonal)
    //   S(dxy, dxz) = 0.0    (xy and xz are orthogonal)
    Point3D origin(0.0, 0.0, 0.0);
    Shell shell(AngularMomentum::D, origin, {1.0}, {1.0});

    OverlapBuffer buffer;
    kernels::compute_overlap(shell, shell, buffer);

    EXPECT_EQ(buffer.na(), 6);
    EXPECT_EQ(buffer.nb(), 6);

    // Diagonal: all self-overlaps should be 1.0
    for (int i = 0; i < 6; ++i) {
        EXPECT_NEAR(buffer(i, i), 1.0, TIGHT_TOL)
            << "d-d self-overlap diagonal (" << i << ")";
    }

    // "Squared" components overlap: S(dxx, dyy) = S(dxx, dzz) = S(dyy, dzz) = 1/3
    // Indices: dxx=0, dyy=3, dzz=5
    EXPECT_NEAR(buffer(0, 3), 1.0 / 3.0, TIGHT_TOL) << "S(dxx, dyy) = 1/3";
    EXPECT_NEAR(buffer(0, 5), 1.0 / 3.0, TIGHT_TOL) << "S(dxx, dzz) = 1/3";
    EXPECT_NEAR(buffer(3, 5), 1.0 / 3.0, TIGHT_TOL) << "S(dyy, dzz) = 1/3";

    // Cross-type overlaps should be zero (squared-type vs mixed-type)
    // dxx(2,0,0) vs dxy(1,1,0): need x^3*y which integrates to 0 (odd in y)
    EXPECT_NEAR(buffer(0, 1), 0.0, TIGHT_TOL) << "S(dxx, dxy) = 0";
    EXPECT_NEAR(buffer(0, 2), 0.0, TIGHT_TOL) << "S(dxx, dxz) = 0";
    EXPECT_NEAR(buffer(0, 4), 0.0, TIGHT_TOL) << "S(dxx, dyz) = 0";

    // Mixed-type vs mixed-type: dxy(1,1,0) vs dxz(1,0,1)
    // integral = <xy exp | xz exp> = <x^2><y><z> -> y and z integrals are odd -> 0
    EXPECT_NEAR(buffer(1, 2), 0.0, TIGHT_TOL) << "S(dxy, dxz) = 0";
    EXPECT_NEAR(buffer(1, 4), 0.0, TIGHT_TOL) << "S(dxy, dyz) = 0";
    EXPECT_NEAR(buffer(2, 4), 0.0, TIGHT_TOL) << "S(dxz, dyz) = 0";

    // Symmetry check
    for (int i = 0; i < 6; ++i) {
        for (int j = i + 1; j < 6; ++j) {
            EXPECT_NEAR(buffer(i, j), buffer(j, i), TIGHT_TOL)
                << "Symmetry: (" << i << "," << j << ")";
        }
    }
}

// =============================================================================
// Test 8: f-f overlap at same center (higher angular momentum)
// =============================================================================
TEST(OverlapKernelTest, FfSameCenter) {
    // Like d-functions, Cartesian f-functions are NOT all orthogonal.
    // The 10 Cartesian f-functions span 10D = 7 spherical f + 3 p-type functions.
    //
    // We verify:
    // 1. All diagonal elements = 1.0
    // 2. Matrix is symmetric
    // 3. Some known zero off-diagonal elements (where parity forbids overlap)
    Point3D origin(0.0, 0.0, 0.0);
    Shell shell(AngularMomentum::F, origin, {1.5}, {1.0});

    OverlapBuffer buffer;
    kernels::compute_overlap(shell, shell, buffer);

    EXPECT_EQ(buffer.na(), 10);
    EXPECT_EQ(buffer.nb(), 10);

    // All diagonal elements should be 1.0
    for (int i = 0; i < 10; ++i) {
        EXPECT_NEAR(buffer(i, i), 1.0, TIGHT_TOL)
            << "f-f self-overlap diagonal (" << i << ")";
    }

    // Matrix should be symmetric
    for (int i = 0; i < 10; ++i) {
        for (int j = i + 1; j < 10; ++j) {
            EXPECT_NEAR(buffer(i, j), buffer(j, i), TIGHT_TOL)
                << "f-f symmetry: (" << i << "," << j << ")";
        }
    }

    // Canonical f ordering: (3,0,0), (2,1,0), (2,0,1), (1,2,0), (1,1,1), (1,0,2),
    //                       (0,3,0), (0,2,1), (0,1,2), (0,0,3)
    //
    // Components with different parity in any direction should have zero overlap.
    // For example, fxxx(3,0,0) and fxxy(2,1,0):
    //   <x^5 * y> has odd parity in y -> overlap = 0
    auto indices = math::generate_cartesian_indices(3);

    for (int i = 0; i < 10; ++i) {
        for (int j = i + 1; j < 10; ++j) {
            // Check if any Cartesian direction has odd sum of angular momenta
            bool any_odd = false;
            for (int d = 0; d < 3; ++d) {
                if ((indices[i][d] + indices[j][d]) % 2 != 0) {
                    any_odd = true;
                    break;
                }
            }
            if (any_odd) {
                EXPECT_NEAR(buffer(i, j), 0.0, TIGHT_TOL)
                    << "f-f off-diagonal with odd parity: (" << i << "," << j << ")";
            }
        }
    }
}

// =============================================================================
// Test 9: Symmetry of overlap matrix S(a,b) = S(b,a)
// =============================================================================
TEST(OverlapKernelTest, Symmetry) {
    // Use two different shells at different centers
    Point3D A(0.0, 0.0, 0.0);
    Point3D B(1.5, 0.7, -0.3);

    Shell shell_a(AngularMomentum::P, A, {2.0, 0.5}, {0.6, 0.4});
    Shell shell_b(AngularMomentum::D, B, {1.5, 0.8}, {0.7, 0.3});

    OverlapBuffer buffer_ab, buffer_ba;
    kernels::compute_overlap(shell_a, shell_b, buffer_ab);
    kernels::compute_overlap(shell_b, shell_a, buffer_ba);

    EXPECT_EQ(buffer_ab.na(), 3);   // p-shell
    EXPECT_EQ(buffer_ab.nb(), 6);   // d-shell
    EXPECT_EQ(buffer_ba.na(), 6);   // d-shell
    EXPECT_EQ(buffer_ba.nb(), 3);   // p-shell

    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 6; ++j) {
            EXPECT_NEAR(buffer_ab(i, j), buffer_ba(j, i), TIGHT_TOL)
                << "S(a,b) should equal S(b,a): (" << i << "," << j << ")";
        }
    }
}

// =============================================================================
// Test 10: STO-3G H2 overlap matrix (contracted shells)
// =============================================================================
TEST(OverlapKernelTest, STO3G_H2) {
    // H2 molecule at R = 1.4 bohr along z-axis
    const Real R = 1.4;
    Point3D H1(0.0, 0.0, 0.0);
    Point3D H2(0.0, 0.0, R);

    Shell shell_H1 = make_sto3g_hydrogen(H1);
    Shell shell_H2 = make_sto3g_hydrogen(H2);

    // Self-overlap should be 1.0
    OverlapBuffer buffer_11;
    kernels::compute_overlap(shell_H1, shell_H1, buffer_11);
    EXPECT_NEAR(buffer_11(0, 0), 1.0, TIGHT_TOL)
        << "STO-3G H1 self-overlap should be 1.0";

    OverlapBuffer buffer_22;
    kernels::compute_overlap(shell_H2, shell_H2, buffer_22);
    EXPECT_NEAR(buffer_22(0, 0), 1.0, TIGHT_TOL)
        << "STO-3G H2 self-overlap should be 1.0";

    // Cross-overlap
    OverlapBuffer buffer_12, buffer_21;
    kernels::compute_overlap(shell_H1, shell_H2, buffer_12);
    kernels::compute_overlap(shell_H2, shell_H1, buffer_21);

    // The cross-overlap should be symmetric
    EXPECT_NEAR(buffer_12(0, 0), buffer_21(0, 0), TIGHT_TOL)
        << "S(H1, H2) should equal S(H2, H1)";

    // The cross-overlap should be between 0 and 1
    const Real S12 = buffer_12(0, 0);
    EXPECT_GT(S12, 0.0) << "H1-H2 overlap should be positive";
    EXPECT_LT(S12, 1.0) << "H1-H2 overlap should be less than 1";

    // Known value: STO-3G H2 at R=1.4 bohr, S12 ~ 0.6593
    // This can be verified against PySCF or libint
    EXPECT_NEAR(S12, 0.6593, 0.005)
        << "STO-3G H2 overlap at R=1.4 bohr should be approximately 0.6593";
}

// =============================================================================
// Test 11: STO-3G H2O overlap matrix properties
// =============================================================================
TEST(OverlapKernelTest, STO3G_H2O_Properties) {
    // H2O geometry (bohr):
    // O at (0, 0, 0)
    // H at (0, 1.43233673, -1.10866041)
    // H at (0, -1.43233673, -1.10866041)
    Point3D O_pos(0.0, 0.0, 0.0);
    Point3D H1_pos(0.0, 1.43233673, -1.10866041);
    Point3D H2_pos(0.0, -1.43233673, -1.10866041);

    // STO-3G basis: O has 1s, 2s, 2p; each H has 1s
    // 7 basis functions total: O_1s, O_2s, O_2px, O_2py, O_2pz, H1_1s, H2_1s
    Shell O_1s = make_sto3g_oxygen_1s(O_pos);
    Shell O_2s = make_sto3g_oxygen_2s(O_pos);
    Shell O_2p = make_sto3g_oxygen_2p(O_pos);
    Shell H1_1s = make_sto3g_hydrogen(H1_pos);
    Shell H2_1s = make_sto3g_hydrogen(H2_pos);

    // Verify diagonal blocks: self-overlap = 1.0
    {
        OverlapBuffer buf;
        kernels::compute_overlap(O_1s, O_1s, buf);
        EXPECT_NEAR(buf(0, 0), 1.0, TIGHT_TOL) << "O_1s self-overlap";
    }
    {
        OverlapBuffer buf;
        kernels::compute_overlap(O_2s, O_2s, buf);
        EXPECT_NEAR(buf(0, 0), 1.0, TIGHT_TOL) << "O_2s self-overlap";
    }
    {
        OverlapBuffer buf;
        kernels::compute_overlap(O_2p, O_2p, buf);
        for (int i = 0; i < 3; ++i) {
            EXPECT_NEAR(buf(i, i), 1.0, TIGHT_TOL) << "O_2p(" << i << ") self-overlap";
            for (int j = 0; j < 3; ++j) {
                if (i != j) {
                    EXPECT_NEAR(buf(i, j), 0.0, TIGHT_TOL)
                        << "O_2p cross-overlap (" << i << "," << j << ")";
                }
            }
        }
    }
    {
        OverlapBuffer buf;
        kernels::compute_overlap(H1_1s, H1_1s, buf);
        EXPECT_NEAR(buf(0, 0), 1.0, TIGHT_TOL) << "H1_1s self-overlap";
    }
    {
        OverlapBuffer buf;
        kernels::compute_overlap(H2_1s, H2_1s, buf);
        EXPECT_NEAR(buf(0, 0), 1.0, TIGHT_TOL) << "H2_1s self-overlap";
    }

    // s-p overlap at same center should be zero
    {
        OverlapBuffer buf;
        kernels::compute_overlap(O_1s, O_2p, buf);
        for (int j = 0; j < 3; ++j) {
            EXPECT_NEAR(buf(0, j), 0.0, TIGHT_TOL)
                << "O_1s - O_2p(" << j << ") overlap at same center";
        }
    }
    {
        OverlapBuffer buf;
        kernels::compute_overlap(O_2s, O_2p, buf);
        for (int j = 0; j < 3; ++j) {
            EXPECT_NEAR(buf(0, j), 0.0, TIGHT_TOL)
                << "O_2s - O_2p(" << j << ") overlap at same center";
        }
    }

    // H1 and H2 should have the same overlap with O_1s (by symmetry)
    {
        OverlapBuffer buf_O1s_H1, buf_O1s_H2;
        kernels::compute_overlap(O_1s, H1_1s, buf_O1s_H1);
        kernels::compute_overlap(O_1s, H2_1s, buf_O1s_H2);
        EXPECT_NEAR(buf_O1s_H1(0, 0), buf_O1s_H2(0, 0), TIGHT_TOL)
            << "O_1s - H1 should equal O_1s - H2 by molecular symmetry";
    }

    // By molecular symmetry (C2v), H1 and H2 have the same overlap with O_2s
    {
        OverlapBuffer buf_O2s_H1, buf_O2s_H2;
        kernels::compute_overlap(O_2s, H1_1s, buf_O2s_H1);
        kernels::compute_overlap(O_2s, H2_1s, buf_O2s_H2);
        EXPECT_NEAR(buf_O2s_H1(0, 0), buf_O2s_H2(0, 0), TIGHT_TOL)
            << "O_2s - H1 should equal O_2s - H2 by molecular symmetry";
    }

    // O_2px - H1/H2 overlap should be zero (H atoms are in yz-plane, so x is symmetric)
    {
        OverlapBuffer buf;
        kernels::compute_overlap(O_2p, H1_1s, buf);
        EXPECT_NEAR(buf(0, 0), 0.0, TIGHT_TOL)
            << "O_2px - H1_1s overlap should be zero (H in yz-plane)";
    }
    {
        OverlapBuffer buf;
        kernels::compute_overlap(O_2p, H2_1s, buf);
        EXPECT_NEAR(buf(0, 0), 0.0, TIGHT_TOL)
            << "O_2px - H2_1s overlap should be zero (H in yz-plane)";
    }

    // O_2py - H1 and O_2py - H2 should be opposite sign (H1.y > 0, H2.y < 0)
    {
        OverlapBuffer buf_H1, buf_H2;
        kernels::compute_overlap(O_2p, H1_1s, buf_H1);
        kernels::compute_overlap(O_2p, H2_1s, buf_H2);
        EXPECT_NEAR(buf_H1(1, 0), -buf_H2(1, 0), TIGHT_TOL)
            << "O_2py overlaps with H1 and H2 should be opposite in sign";
    }

    // O_2pz - H1 and O_2pz - H2 should be equal (same z-component for both H)
    {
        OverlapBuffer buf_H1, buf_H2;
        kernels::compute_overlap(O_2p, H1_1s, buf_H1);
        kernels::compute_overlap(O_2p, H2_1s, buf_H2);
        EXPECT_NEAR(buf_H1(2, 0), buf_H2(2, 0), TIGHT_TOL)
            << "O_2pz overlaps with H1 and H2 should be equal";
    }

    // H1-H2 overlap should be between 0 and 1
    {
        OverlapBuffer buf;
        kernels::compute_overlap(H1_1s, H2_1s, buf);
        EXPECT_GT(buf(0, 0), 0.0) << "H1-H2 overlap should be positive";
        EXPECT_LT(buf(0, 0), 1.0) << "H1-H2 overlap should be less than 1";
    }
}

// =============================================================================
// Test 12: Contracted shell s-s overlap
// =============================================================================
TEST(OverlapKernelTest, ContractedSsOverlap) {
    // Two identical contracted s-shells at the same center
    Point3D center(1.0, 2.0, 3.0);
    Shell shell(AngularMomentum::S, center, {3.0, 1.0, 0.3}, {0.5, 0.3, 0.2});

    OverlapBuffer buffer;
    kernels::compute_overlap(shell, shell, buffer);

    EXPECT_NEAR(buffer(0, 0), 1.0, TIGHT_TOL)
        << "Contracted s-shell self-overlap should be 1.0";
}

// =============================================================================
// Test 13: p-d overlap at different centers
// =============================================================================
TEST(OverlapKernelTest, PdDifferentCenters) {
    Point3D A(0.0, 0.0, 0.0);
    Point3D B(1.0, 0.0, 0.0);

    Shell shell_p(AngularMomentum::P, A, {1.0}, {1.0});
    Shell shell_d(AngularMomentum::D, B, {1.0}, {1.0});

    OverlapBuffer buffer;
    kernels::compute_overlap(shell_p, shell_d, buffer);

    EXPECT_EQ(buffer.na(), 3);   // p-shell: px, py, pz
    EXPECT_EQ(buffer.nb(), 6);   // d-shell: dxx, dxy, dxz, dyy, dyz, dzz

    // Verify symmetry: compute S(d,p) and check transpose
    OverlapBuffer buffer_dp;
    kernels::compute_overlap(shell_d, shell_p, buffer_dp);

    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 6; ++j) {
            EXPECT_NEAR(buffer(i, j), buffer_dp(j, i), TIGHT_TOL)
                << "p-d symmetry check: (" << i << "," << j << ")";
        }
    }

    // With separation along x only:
    // py-dxy should be nonzero (py has ly=1, dxy has ly=1 -> I_y(1,1) != 0)
    // py-dyz should be zero (py has lx=0, dyz has lx=0, but need to check all 3D)
    // Actually let's just verify some are nonzero
    bool has_nonzero = false;
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 6; ++j) {
            if (std::abs(buffer(i, j)) > 1e-8) {
                has_nonzero = true;
            }
        }
    }
    EXPECT_TRUE(has_nonzero) << "p-d overlap at different centers should have nonzero elements";
}

// =============================================================================
// Test 14: Contracted p-p overlap at different centers
// =============================================================================
TEST(OverlapKernelTest, ContractedPpDifferentCenters) {
    // Use a moderate separation where diagonal overlaps remain positive
    Point3D A(0.0, 0.0, 0.0);
    Point3D B(0.0, 0.0, 0.8);

    Shell shell_a(AngularMomentum::P, A, {3.0, 1.0}, {0.6, 0.4});
    Shell shell_b(AngularMomentum::P, B, {3.0, 1.0}, {0.6, 0.4});

    OverlapBuffer buffer;
    kernels::compute_overlap(shell_a, shell_b, buffer);

    EXPECT_EQ(buffer.na(), 3);
    EXPECT_EQ(buffer.nb(), 3);

    // At moderate separation, px-px and py-py (perpendicular to displacement) are positive
    EXPECT_GT(buffer(0, 0), 0.0) << "px-px overlap at moderate R should be positive";
    EXPECT_LT(buffer(0, 0), 1.0) << "px-px overlap should be less than 1";

    // Separation along z: px-px and py-py should be equal by symmetry
    EXPECT_NEAR(buffer(0, 0), buffer(1, 1), TIGHT_TOL)
        << "px-px and py-py should be equal for z-separation";

    // pz-pz should differ from px-px (anisotropic overlap)
    // Note: pz-pz can be negative at certain distances (lobes of opposite sign overlap)
    EXPECT_NE(buffer(2, 2), buffer(0, 0))
        << "pz-pz should differ from px-px for z-separation";

    // pz-pz magnitude should be less than 1
    EXPECT_LT(std::abs(buffer(2, 2)), 1.0) << "pz-pz magnitude should be less than 1";

    // Off-diagonal: px-py, px-pz, py-pz should be zero by symmetry
    EXPECT_NEAR(buffer(0, 1), 0.0, TIGHT_TOL) << "px-py cross-overlap should be zero";
    EXPECT_NEAR(buffer(0, 2), 0.0, TIGHT_TOL) << "px-pz cross-overlap should be zero";
    EXPECT_NEAR(buffer(1, 2), 0.0, TIGHT_TOL) << "py-pz cross-overlap should be zero";

    // Verify symmetry: S(a,b) = S(b,a)^T
    OverlapBuffer buffer_ba;
    kernels::compute_overlap(shell_b, shell_a, buffer_ba);
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            EXPECT_NEAR(buffer(i, j), buffer_ba(j, i), TIGHT_TOL)
                << "p-p symmetry at (" << i << "," << j << ")";
        }
    }
}

// =============================================================================
// Test 15: s-s overlap decay with distance
// =============================================================================
TEST(OverlapKernelTest, SsDecayWithDistance) {
    // Overlap should decrease monotonically with distance
    Point3D A(0.0, 0.0, 0.0);

    Shell shell_a(AngularMomentum::S, A, {1.0}, {1.0});

    Real prev_overlap = 1.0;  // Self-overlap
    for (double R = 0.5; R <= 5.0; R += 0.5) {
        Shell shell_b(AngularMomentum::S, Point3D(R, 0.0, 0.0), {1.0}, {1.0});

        OverlapBuffer buffer;
        kernels::compute_overlap(shell_a, shell_b, buffer);

        EXPECT_GT(buffer(0, 0), 0.0) << "Overlap should be positive at R=" << R;
        EXPECT_LT(buffer(0, 0), prev_overlap)
            << "Overlap should decrease with distance at R=" << R;
        prev_overlap = buffer(0, 0);
    }
}

// =============================================================================
// Test 16: Buffer is properly resized and cleared
// =============================================================================
TEST(OverlapKernelTest, BufferManagement) {
    // Use a buffer that was previously sized differently
    OverlapBuffer buffer;

    // First use with s-s
    {
        Shell shell(AngularMomentum::S, Point3D(0.0, 0.0, 0.0), {1.0}, {1.0});
        kernels::compute_overlap(shell, shell, buffer);
        EXPECT_EQ(buffer.na(), 1);
        EXPECT_EQ(buffer.nb(), 1);
    }

    // Reuse with p-p (should resize properly)
    {
        Shell shell(AngularMomentum::P, Point3D(0.0, 0.0, 0.0), {1.0}, {1.0});
        kernels::compute_overlap(shell, shell, buffer);
        EXPECT_EQ(buffer.na(), 3);
        EXPECT_EQ(buffer.nb(), 3);
    }

    // Reuse with s-d (should resize properly)
    {
        Shell shell_s(AngularMomentum::S, Point3D(0.0, 0.0, 0.0), {1.0}, {1.0});
        Shell shell_d(AngularMomentum::D, Point3D(0.0, 0.0, 0.0), {1.0}, {1.0});
        kernels::compute_overlap(shell_s, shell_d, buffer);
        EXPECT_EQ(buffer.na(), 1);
        EXPECT_EQ(buffer.nb(), 6);
    }
}

// =============================================================================
// Test 17: Numerical stability with widely varying exponents
// =============================================================================
TEST(OverlapKernelTest, NumericalStability) {
    Point3D origin(0.0, 0.0, 0.0);

    // Large exponent difference
    Shell shell_tight(AngularMomentum::S, origin, {100.0}, {1.0});
    Shell shell_diffuse(AngularMomentum::S, origin, {0.01}, {1.0});

    OverlapBuffer buffer;
    kernels::compute_overlap(shell_tight, shell_diffuse, buffer);

    // Should still be a finite, positive value less than 1
    EXPECT_TRUE(std::isfinite(buffer(0, 0))) << "Overlap should be finite";
    EXPECT_GT(buffer(0, 0), 0.0) << "Overlap should be positive";
    EXPECT_LT(buffer(0, 0), 1.0) << "Overlap should be less than 1";
}

// =============================================================================
// Test 18: Analytical s-s overlap with multiple R values
// =============================================================================
TEST(OverlapKernelTest, SsAnalyticalMultipleDistances) {
    // For identical single-primitive s-shells with alpha=1.0, at distance R:
    // S(R) = exp(-R^2 / 4)  [since mu = 0.5 for alpha=beta=1, so S = exp(-0.5 * R^2)]
    // Wait: mu = alpha*beta/zeta = 1*1/2 = 0.5, so S = exp(-0.5 * R^2)

    Point3D A(0.0, 0.0, 0.0);
    const std::vector<Real> distances = {0.0, 0.5, 1.0, 1.5, 2.0, 3.0};

    for (Real R : distances) {
        Shell shell_a(AngularMomentum::S, A, {1.0}, {1.0});
        Shell shell_b(AngularMomentum::S, Point3D(0.0, 0.0, R), {1.0}, {1.0});

        OverlapBuffer buffer;
        kernels::compute_overlap(shell_a, shell_b, buffer);

        const Real expected = std::exp(-0.5 * R * R);
        EXPECT_NEAR(buffer(0, 0), expected, TIGHT_TOL)
            << "s-s overlap at R=" << R << ": expected exp(-0.5*R^2)";
    }
}

// =============================================================================
// Test 19: d-d overlap at different centers (verifies full recursion depth)
// =============================================================================
TEST(OverlapKernelTest, DdDifferentCenters) {
    Point3D A(0.0, 0.0, 0.0);
    Point3D B(1.0, 0.5, -0.3);

    Shell shell_a(AngularMomentum::D, A, {1.5}, {1.0});
    Shell shell_b(AngularMomentum::D, B, {0.8}, {1.0});

    OverlapBuffer buffer_ab, buffer_ba;
    kernels::compute_overlap(shell_a, shell_b, buffer_ab);
    kernels::compute_overlap(shell_b, shell_a, buffer_ba);

    EXPECT_EQ(buffer_ab.na(), 6);
    EXPECT_EQ(buffer_ab.nb(), 6);

    // Check symmetry
    for (int i = 0; i < 6; ++i) {
        for (int j = 0; j < 6; ++j) {
            EXPECT_NEAR(buffer_ab(i, j), buffer_ba(j, i), TIGHT_TOL)
                << "d-d symmetry: (" << i << "," << j << ")";
        }
    }

    // Verify all elements are finite and bounded
    for (int i = 0; i < 6; ++i) {
        for (int j = 0; j < 6; ++j) {
            EXPECT_TRUE(std::isfinite(buffer_ab(i, j)))
                << "d-d element (" << i << "," << j << ") should be finite";
            EXPECT_LT(std::abs(buffer_ab(i, j)), 2.0)
                << "d-d element (" << i << "," << j << ") should be bounded";
        }
    }

    // At least some elements should be nonzero
    bool has_nonzero = false;
    for (int i = 0; i < 6; ++i) {
        for (int j = 0; j < 6; ++j) {
            if (std::abs(buffer_ab(i, j)) > 1e-8) {
                has_nonzero = true;
                break;
            }
        }
        if (has_nonzero) break;
    }
    EXPECT_TRUE(has_nonzero) << "d-d overlap at different centers should have nonzero elements";
}

// =============================================================================
// Test 20: Zero overlap for widely separated shells
// =============================================================================
TEST(OverlapKernelTest, WidelySeparatedShells) {
    // Two s-shells very far apart -> overlap should be essentially zero
    Point3D A(0.0, 0.0, 0.0);
    Point3D B(100.0, 0.0, 0.0);

    Shell shell_a(AngularMomentum::S, A, {1.0}, {1.0});
    Shell shell_b(AngularMomentum::S, B, {1.0}, {1.0});

    OverlapBuffer buffer;
    kernels::compute_overlap(shell_a, shell_b, buffer);

    // exp(-0.5 * 100^2) = exp(-5000) ~ 0
    EXPECT_NEAR(buffer(0, 0), 0.0, 1e-15)
        << "Overlap of widely separated shells should be essentially zero";
}

// =============================================================================
// Test 21: Verify p-shell self-overlap with contracted basis
// =============================================================================
TEST(OverlapKernelTest, ContractedPSelfOverlap) {
    // STO-3G oxygen 2p shell
    Point3D origin(0.0, 0.0, 0.0);
    Shell shell = make_sto3g_oxygen_2p(origin);

    OverlapBuffer buffer;
    kernels::compute_overlap(shell, shell, buffer);

    // All diagonal elements should be 1.0, off-diagonal should be 0.0
    for (int i = 0; i < 3; ++i) {
        EXPECT_NEAR(buffer(i, i), 1.0, TIGHT_TOL)
            << "STO-3G O_2p self-overlap diagonal (" << i << ")";
        for (int j = 0; j < 3; ++j) {
            if (i != j) {
                EXPECT_NEAR(buffer(i, j), 0.0, TIGHT_TOL)
                    << "STO-3G O_2p self-overlap off-diagonal (" << i << "," << j << ")";
            }
        }
    }
}

// =============================================================================
// PrimitivePairData Overload Tests
// =============================================================================

namespace {
ShellSet make_shell_set_from(const std::vector<Shell>& shells) {
    std::vector<std::reference_wrapper<const Shell>> refs;
    refs.reserve(shells.size());
    for (const auto& s : shells) {
        refs.push_back(std::cref(s));
    }
    return ShellSet(refs);
}
}  // anonymous namespace

TEST(OverlapKernelTest, PairDataOverload_SS_MatchesOnTheFly) {
    Shell s1 = make_sto3g_hydrogen(Point3D(0.0, 0.0, 0.0));
    Shell s2 = make_sto3g_hydrogen(Point3D(1.4, 0.0, 0.0));

    std::vector<Shell> shells_a = {s1};
    std::vector<Shell> shells_b = {s2};
    ShellSet set_a = make_shell_set_from(shells_a);
    ShellSet set_b = make_shell_set_from(shells_b);
    ShellSetPair pair(set_a, set_b);

    OverlapBuffer buf_direct, buf_cached;
    kernels::compute_overlap(s1, s2, buf_direct);
    kernels::compute_overlap(s1, s2, pair.pair_data(), 0, 0, buf_cached);

    EXPECT_NEAR(buf_direct(0, 0), buf_cached(0, 0), TIGHT_TOL);
}

TEST(OverlapKernelTest, PairDataOverload_SP_MatchesOnTheFly) {
    Shell s1 = make_sto3g_oxygen_2s(Point3D(0.0, 0.0, 0.0));
    Shell p1 = make_sto3g_oxygen_2p(Point3D(0.0, 0.0, 0.0));

    std::vector<Shell> shells_s = {s1};
    std::vector<Shell> shells_p = {p1};
    ShellSet set_s = make_shell_set_from(shells_s);
    ShellSet set_p = make_shell_set_from(shells_p);
    ShellSetPair pair(set_s, set_p);

    OverlapBuffer buf_direct, buf_cached;
    kernels::compute_overlap(s1, p1, buf_direct);
    kernels::compute_overlap(s1, p1, pair.pair_data(), 0, 0, buf_cached);

    const int na = s1.n_functions();
    const int nb = p1.n_functions();
    for (int a = 0; a < na; ++a) {
        for (int b = 0; b < nb; ++b) {
            EXPECT_NEAR(buf_direct(a, b), buf_cached(a, b), TIGHT_TOL)
                << "Mismatch at (" << a << ", " << b << ")";
        }
    }
}

TEST(OverlapKernelTest, PairDataOverload_PP_MatchesOnTheFly) {
    Shell p1 = make_sto3g_oxygen_2p(Point3D(0.0, 0.0, 0.0));
    Shell p2 = make_sto3g_oxygen_2p(Point3D(1.0, 0.5, 0.0));

    std::vector<Shell> shells_a = {p1};
    std::vector<Shell> shells_b = {p2};
    ShellSet set_a = make_shell_set_from(shells_a);
    ShellSet set_b = make_shell_set_from(shells_b);
    ShellSetPair pair(set_a, set_b);

    OverlapBuffer buf_direct, buf_cached;
    kernels::compute_overlap(p1, p2, buf_direct);
    kernels::compute_overlap(p1, p2, pair.pair_data(), 0, 0, buf_cached);

    for (int a = 0; a < 3; ++a) {
        for (int b = 0; b < 3; ++b) {
            EXPECT_NEAR(buf_direct(a, b), buf_cached(a, b), TIGHT_TOL)
                << "Mismatch at (" << a << ", " << b << ")";
        }
    }
}
