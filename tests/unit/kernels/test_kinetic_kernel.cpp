// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2026 Bryce M. Westheimer. All rights reserved.

/// @file test_kinetic_kernel.cpp
/// @brief Unit tests for kinetic energy integral kernel

#include <libaccint/kernels/kinetic_kernel.hpp>
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
    std::vector<Real> exponents = {3.425250914, 0.6239137298, 0.168855404};
    std::vector<Real> coefficients = {0.1543289673, 0.5353281423, 0.4446345422};
    return Shell(AngularMomentum::S, center, exponents, coefficients);
}

// ============================================================================
// Helper: Create STO-3G oxygen shells at given position
// ============================================================================
Shell make_sto3g_oxygen_1s(Point3D center) {
    std::vector<Real> exponents = {130.7093214, 23.80886605, 6.443608313};
    std::vector<Real> coefficients = {0.1543289673, 0.5353281423, 0.4446345422};
    return Shell(AngularMomentum::S, center, exponents, coefficients);
}

Shell make_sto3g_oxygen_2s(Point3D center) {
    std::vector<Real> exponents = {5.033151319, 1.169596125, 0.38038896};
    std::vector<Real> coefficients = {-0.09996722919, 0.3995128261, 0.7001154689};
    return Shell(AngularMomentum::S, center, exponents, coefficients);
}

Shell make_sto3g_oxygen_2p(Point3D center) {
    std::vector<Real> exponents = {5.033151319, 1.169596125, 0.38038896};
    std::vector<Real> coefficients = {0.1559162750, 0.6076837186, 0.3919573931};
    return Shell(AngularMomentum::P, center, exponents, coefficients);
}

}  // anonymous namespace

// =============================================================================
// Test 1: s-s self kinetic energy (analytical)
// =============================================================================
TEST(KineticKernelTest, SsSelfKineticAnalytical) {
    // For a normalized single-primitive s-type GTO with exponent alpha at
    // the same center, T_ss = -1/2 <s|nabla^2|s> = 3*alpha/2
    //
    // Derivation: The second derivative of exp(-alpha*r^2) in one direction gives
    // (4*alpha^2*x^2 - 2*alpha) * exp(-alpha*r^2). For the s-s case (lx=ly=lz=0),
    // the 1D kinetic integral is T_d(0,0) = -1/2*(-2*alpha) = alpha.
    // In 3D: T = T_x*S_y*S_z + S_x*T_y*S_z + S_x*S_y*T_z = 3*alpha*(overlap factor)
    // For normalized s-GTOs at same center, T = 3*alpha/2.
    const Real alpha = 1.0;
    Point3D origin(0.0, 0.0, 0.0);
    Shell shell(AngularMomentum::S, origin, {alpha}, {1.0});

    KineticBuffer buffer;
    kernels::compute_kinetic(shell, shell, buffer);

    EXPECT_EQ(buffer.na(), 1);
    EXPECT_EQ(buffer.nb(), 1);

    const Real expected = 1.5 * alpha;
    EXPECT_NEAR(buffer(0, 0), expected, TIGHT_TOL)
        << "T_ss self kinetic for alpha=1.0 should be 3/2";
}

// =============================================================================
// Test 2: s-s self kinetic with different exponents
// =============================================================================
TEST(KineticKernelTest, SsSelfKineticDifferentAlpha) {
    // T_ss = 3*alpha/2 for single-primitive normalized s-shells at same center
    for (Real alpha : {0.5, 1.0, 2.0, 5.0, 10.0}) {
        Point3D origin(0.0, 0.0, 0.0);
        Shell shell(AngularMomentum::S, origin, {alpha}, {1.0});

        KineticBuffer buffer;
        kernels::compute_kinetic(shell, shell, buffer);

        const Real expected = 1.5 * alpha;
        EXPECT_NEAR(buffer(0, 0), expected, TIGHT_TOL)
            << "T_ss self kinetic for alpha=" << alpha;
    }
}

// =============================================================================
// Test 3: s-s kinetic at different centers
// =============================================================================
TEST(KineticKernelTest, SsDifferentCenters) {
    // For two identical single-primitive s-shells with exponent alpha at
    // centers A and B separated by R:
    //
    // Using the formula with the 1D kinetic integral formulation:
    //   T = [3*alpha/2 - alpha^2/zeta * R^2] * S
    // where zeta = 2*alpha, mu = alpha/2, and S = exp(-mu*R^2) = exp(-alpha*R^2/4)
    // Wait, actually let's compute it more carefully.
    //
    // For alpha=beta=1, at same exponent, zeta=2, mu=0.5:
    // T = (alpha - 2*alpha^2*(PA^2+PB^2+...)) * S... This is complex.
    // Let's verify properties instead: T should be positive and less than T_self.
    const Real alpha = 1.0;
    const Real R = 1.0;
    Point3D A(0.0, 0.0, 0.0);
    Point3D B(0.0, 0.0, R);

    Shell shell_a(AngularMomentum::S, A, {alpha}, {1.0});
    Shell shell_b(AngularMomentum::S, B, {alpha}, {1.0});

    KineticBuffer buffer;
    kernels::compute_kinetic(shell_a, shell_b, buffer);

    // T should be finite
    EXPECT_TRUE(std::isfinite(buffer(0, 0)))
        << "T(s,s) at different centers should be finite";

    // For normalized s-GTOs with alpha=beta at moderate separation,
    // the kinetic energy integral should be less than the self-kinetic
    const Real T_self = 1.5 * alpha;
    EXPECT_LT(std::abs(buffer(0, 0)), T_self)
        << "T(s,s) at R=1 should be less than self-kinetic";
}

// =============================================================================
// Test 4: s-s kinetic analytical verification at different centers
// =============================================================================
TEST(KineticKernelTest, SsDifferentCentersAnalytical) {
    // For two normalized single-primitive s-shells with exponents alpha, beta
    // at centers A, B separated by R along z:
    //
    // The kinetic energy integral can be computed analytically:
    // T = mu * (3 - 2*mu*R^2) * S
    // where mu = alpha*beta/(alpha+beta), zeta = alpha+beta,
    // and S is the overlap integral.
    //
    // For alpha=beta=1.0, R=1.0 along z:
    //   mu = 0.5, zeta = 2.0
    //   S = exp(-0.5 * 1.0) = exp(-0.5)
    //   T = 0.5 * (3 - 2*0.5*1.0) * exp(-0.5) = 0.5 * 2.0 * exp(-0.5) = exp(-0.5)

    const Real alpha = 1.0;
    const Real beta = 1.0;
    const Real R = 1.0;
    Point3D A(0.0, 0.0, 0.0);
    Point3D B(0.0, 0.0, R);

    Shell shell_a(AngularMomentum::S, A, {alpha}, {1.0});
    Shell shell_b(AngularMomentum::S, B, {beta}, {1.0});

    KineticBuffer buffer;
    kernels::compute_kinetic(shell_a, shell_b, buffer);

    const Real mu = alpha * beta / (alpha + beta);
    const Real R2 = R * R;
    const Real S = std::exp(-mu * R2);
    const Real expected = mu * (3.0 - 2.0 * mu * R2) * S;

    EXPECT_NEAR(buffer(0, 0), expected, TIGHT_TOL)
        << "T(s,s) analytical for alpha=beta=1, R=1";
}

// =============================================================================
// Test 5: s-s kinetic analytical with different exponents
// =============================================================================
TEST(KineticKernelTest, SsDifferentExponentsAnalytical) {
    // T = mu * (3 - 2*mu*R^2) * S_normalized
    // where S_normalized is the overlap between normalized GTOs
    const Real alpha = 2.0;
    const Real beta = 0.5;
    const Real R = 1.5;
    Point3D A(0.0, 0.0, 0.0);
    Point3D B(R, 0.0, 0.0);

    Shell shell_a(AngularMomentum::S, A, {alpha}, {1.0});
    Shell shell_b(AngularMomentum::S, B, {beta}, {1.0});

    KineticBuffer buffer_T;
    kernels::compute_kinetic(shell_a, shell_b, buffer_T);

    // Get the overlap for the same pair
    OverlapBuffer buffer_S;
    kernels::compute_overlap(shell_a, shell_b, buffer_S);

    const Real mu = alpha * beta / (alpha + beta);
    const Real R2 = R * R;
    const Real expected = mu * (3.0 - 2.0 * mu * R2) * buffer_S(0, 0);

    EXPECT_NEAR(buffer_T(0, 0), expected, TIGHT_TOL)
        << "T(s,s) analytical for different exponents at R=1.5";
}

// =============================================================================
// Test 6: Symmetry verification T(a,b) = T(b,a)
// =============================================================================
TEST(KineticKernelTest, Symmetry) {
    Point3D A(0.0, 0.0, 0.0);
    Point3D B(1.5, 0.7, -0.3);

    Shell shell_a(AngularMomentum::P, A, {2.0, 0.5}, {0.6, 0.4});
    Shell shell_b(AngularMomentum::D, B, {1.5, 0.8}, {0.7, 0.3});

    KineticBuffer buffer_ab, buffer_ba;
    kernels::compute_kinetic(shell_a, shell_b, buffer_ab);
    kernels::compute_kinetic(shell_b, shell_a, buffer_ba);

    EXPECT_EQ(buffer_ab.na(), 3);   // p-shell
    EXPECT_EQ(buffer_ab.nb(), 6);   // d-shell
    EXPECT_EQ(buffer_ba.na(), 6);   // d-shell
    EXPECT_EQ(buffer_ba.nb(), 3);   // p-shell

    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 6; ++j) {
            EXPECT_NEAR(buffer_ab(i, j), buffer_ba(j, i), TIGHT_TOL)
                << "T(a,b) should equal T(b,a): (" << i << "," << j << ")";
        }
    }
}

// =============================================================================
// Test 7: Positive diagonal for s-s, p-p, d-d self kinetic
// =============================================================================
TEST(KineticKernelTest, PositiveDiagonal) {
    Point3D origin(0.0, 0.0, 0.0);

    // s-shell
    {
        Shell shell(AngularMomentum::S, origin, {1.5}, {1.0});
        KineticBuffer buffer;
        kernels::compute_kinetic(shell, shell, buffer);
        EXPECT_GT(buffer(0, 0), 0.0) << "s-s self kinetic should be positive";
    }

    // p-shell
    {
        Shell shell(AngularMomentum::P, origin, {1.5}, {1.0});
        KineticBuffer buffer;
        kernels::compute_kinetic(shell, shell, buffer);
        for (int i = 0; i < 3; ++i) {
            EXPECT_GT(buffer(i, i), 0.0)
                << "p-p self kinetic diagonal (" << i << ") should be positive";
        }
    }

    // d-shell
    {
        Shell shell(AngularMomentum::D, origin, {1.5}, {1.0});
        KineticBuffer buffer;
        kernels::compute_kinetic(shell, shell, buffer);
        for (int i = 0; i < 6; ++i) {
            EXPECT_GT(buffer(i, i), 0.0)
                << "d-d self kinetic diagonal (" << i << ") should be positive";
        }
    }
}

// =============================================================================
// Test 8: s-p kinetic at same center
// =============================================================================
TEST(KineticKernelTest, SpSameCenter) {
    // <s|T|px>, <s|T|py>, <s|T|pz> at same center should all be zero.
    // The kinetic energy operator preserves parity, so the integrand has
    // odd parity in the direction of the p-function's angular momentum,
    // which integrates to zero.
    Point3D origin(0.0, 0.0, 0.0);
    Shell shell_s(AngularMomentum::S, origin, {1.0}, {1.0});
    Shell shell_p(AngularMomentum::P, origin, {1.0}, {1.0});

    KineticBuffer buffer;
    kernels::compute_kinetic(shell_s, shell_p, buffer);

    EXPECT_EQ(buffer.na(), 1);
    EXPECT_EQ(buffer.nb(), 3);

    for (int b = 0; b < 3; ++b) {
        EXPECT_NEAR(buffer(0, b), 0.0, TIGHT_TOL)
            << "T(s, p) at same center should be zero (component " << b << ")";
    }
}

// =============================================================================
// Test 9: s-p kinetic at different centers
// =============================================================================
TEST(KineticKernelTest, SpDifferentCenters) {
    // s at origin, p at (0, 0, R)
    // By symmetry: T(s, px) = T(s, py) = 0, but T(s, pz) != 0
    const Real R = 1.0;
    Point3D A(0.0, 0.0, 0.0);
    Point3D B(0.0, 0.0, R);

    Shell shell_s(AngularMomentum::S, A, {1.0}, {1.0});
    Shell shell_p(AngularMomentum::P, B, {1.0}, {1.0});

    KineticBuffer buffer;
    kernels::compute_kinetic(shell_s, shell_p, buffer);

    EXPECT_EQ(buffer.na(), 1);
    EXPECT_EQ(buffer.nb(), 3);

    // T(s, px) and T(s, py) should be zero by symmetry
    EXPECT_NEAR(buffer(0, 0), 0.0, TIGHT_TOL)
        << "T(s, px) should be zero (symmetry in x)";
    EXPECT_NEAR(buffer(0, 1), 0.0, TIGHT_TOL)
        << "T(s, py) should be zero (symmetry in y)";

    // T(s, pz) should be nonzero
    EXPECT_NE(buffer(0, 2), 0.0)
        << "T(s, pz) should be nonzero at different centers along z";
    EXPECT_TRUE(std::isfinite(buffer(0, 2)))
        << "T(s, pz) should be finite";
}

// =============================================================================
// Test 10: p-p self kinetic at same center
// =============================================================================
TEST(KineticKernelTest, PpSelfSameCenter) {
    // For identical p-shells at the same center:
    // - Diagonal elements T(px,px) = T(py,py) = T(pz,pz) should be equal and positive
    // - Off-diagonal elements should be zero by symmetry
    //
    // For a normalized single-primitive p-type GTO with exponent alpha:
    // T(pi, pi) = alpha * (5/2) for each component
    // (the kinetic energy of a p-type Gaussian involves terms from the
    //  s+2 and s components of the second derivative)
    const Real alpha = 1.0;
    Point3D origin(0.0, 0.0, 0.0);
    Shell shell(AngularMomentum::P, origin, {alpha}, {1.0});

    KineticBuffer buffer;
    kernels::compute_kinetic(shell, shell, buffer);

    EXPECT_EQ(buffer.na(), 3);
    EXPECT_EQ(buffer.nb(), 3);

    // Diagonal elements should be equal (spherical symmetry at same center)
    EXPECT_NEAR(buffer(0, 0), buffer(1, 1), TIGHT_TOL)
        << "T(px,px) should equal T(py,py)";
    EXPECT_NEAR(buffer(0, 0), buffer(2, 2), TIGHT_TOL)
        << "T(px,px) should equal T(pz,pz)";

    // All diagonal elements should be positive
    for (int i = 0; i < 3; ++i) {
        EXPECT_GT(buffer(i, i), 0.0)
            << "T(pi,pi) should be positive for component " << i;
    }

    // Off-diagonal elements should be zero
    EXPECT_NEAR(buffer(0, 1), 0.0, TIGHT_TOL) << "T(px,py) should be zero";
    EXPECT_NEAR(buffer(0, 2), 0.0, TIGHT_TOL) << "T(px,pz) should be zero";
    EXPECT_NEAR(buffer(1, 2), 0.0, TIGHT_TOL) << "T(py,pz) should be zero";

    // For a single-primitive normalized p-type GTO with exponent alpha at same center:
    // T(pi, pi) = 5*alpha/2
    const Real expected = 2.5 * alpha;
    EXPECT_NEAR(buffer(0, 0), expected, TIGHT_TOL)
        << "T(px,px) for normalized p-shell with alpha=1.0";
}

// =============================================================================
// Test 11: d-d self kinetic at same center
// =============================================================================
TEST(KineticKernelTest, DdSelfSameCenter) {
    // For identical d-shells at the same center:
    // - All diagonal elements should be positive
    // - Matrix should be symmetric
    // - "Pure" components (dxy, dxz, dyz) should have equal diagonal elements
    // - "Squared" components (dxx, dyy, dzz) should have equal diagonal elements
    const Real alpha = 1.5;
    Point3D origin(0.0, 0.0, 0.0);
    Shell shell(AngularMomentum::D, origin, {alpha}, {1.0});

    KineticBuffer buffer;
    kernels::compute_kinetic(shell, shell, buffer);

    EXPECT_EQ(buffer.na(), 6);
    EXPECT_EQ(buffer.nb(), 6);

    // All diagonal elements should be positive
    for (int i = 0; i < 6; ++i) {
        EXPECT_GT(buffer(i, i), 0.0)
            << "d-d self kinetic diagonal (" << i << ") should be positive";
    }

    // Matrix should be symmetric
    for (int i = 0; i < 6; ++i) {
        for (int j = i + 1; j < 6; ++j) {
            EXPECT_NEAR(buffer(i, j), buffer(j, i), TIGHT_TOL)
                << "d-d kinetic symmetry: (" << i << "," << j << ")";
        }
    }

    // Canonical d ordering: (2,0,0)=0, (1,1,0)=1, (1,0,1)=2, (0,2,0)=3, (0,1,1)=4, (0,0,2)=5
    // Squared components dxx=0, dyy=3, dzz=5 should have equal kinetic energy
    EXPECT_NEAR(buffer(0, 0), buffer(3, 3), TIGHT_TOL)
        << "T(dxx,dxx) should equal T(dyy,dyy)";
    EXPECT_NEAR(buffer(0, 0), buffer(5, 5), TIGHT_TOL)
        << "T(dxx,dxx) should equal T(dzz,dzz)";

    // Mixed components dxy=1, dxz=2, dyz=4 should have equal kinetic energy
    EXPECT_NEAR(buffer(1, 1), buffer(2, 2), TIGHT_TOL)
        << "T(dxy,dxy) should equal T(dxz,dxz)";
    EXPECT_NEAR(buffer(1, 1), buffer(4, 4), TIGHT_TOL)
        << "T(dxy,dxy) should equal T(dyz,dyz)";
}

// =============================================================================
// Test 12: Buffer reuse (resize and clear)
// =============================================================================
TEST(KineticKernelTest, BufferReuse) {
    KineticBuffer buffer;

    // First use: s-s
    {
        Point3D origin(0.0, 0.0, 0.0);
        Shell shell(AngularMomentum::S, origin, {1.0}, {1.0});
        kernels::compute_kinetic(shell, shell, buffer);
        EXPECT_EQ(buffer.na(), 1);
        EXPECT_EQ(buffer.nb(), 1);
        EXPECT_NEAR(buffer(0, 0), 1.5, TIGHT_TOL);
    }

    // Reuse: p-p (should resize properly)
    {
        Point3D origin(0.0, 0.0, 0.0);
        Shell shell(AngularMomentum::P, origin, {1.0}, {1.0});
        kernels::compute_kinetic(shell, shell, buffer);
        EXPECT_EQ(buffer.na(), 3);
        EXPECT_EQ(buffer.nb(), 3);
    }

    // Reuse: s-d (should resize properly)
    {
        Point3D origin(0.0, 0.0, 0.0);
        Shell shell_s(AngularMomentum::S, origin, {1.0}, {1.0});
        Shell shell_d(AngularMomentum::D, origin, {1.0}, {1.0});
        kernels::compute_kinetic(shell_s, shell_d, buffer);
        EXPECT_EQ(buffer.na(), 1);
        EXPECT_EQ(buffer.nb(), 6);
    }
}

// =============================================================================
// Test 13: STO-3G H2 kinetic integrals
// =============================================================================
TEST(KineticKernelTest, STO3G_H2) {
    // H2 molecule at R = 1.4 bohr along z-axis
    const Real R = 1.4;
    Point3D H1(0.0, 0.0, 0.0);
    Point3D H2(0.0, 0.0, R);

    Shell shell_H1 = make_sto3g_hydrogen(H1);
    Shell shell_H2 = make_sto3g_hydrogen(H2);

    // Self-kinetic (diagonal blocks)
    KineticBuffer buffer_11, buffer_22;
    kernels::compute_kinetic(shell_H1, shell_H1, buffer_11);
    kernels::compute_kinetic(shell_H2, shell_H2, buffer_22);

    // Both self-kinetics should be the same (same basis)
    EXPECT_NEAR(buffer_11(0, 0), buffer_22(0, 0), TIGHT_TOL)
        << "STO-3G H1 and H2 self-kinetic should be equal";

    // Self-kinetic should be positive
    EXPECT_GT(buffer_11(0, 0), 0.0)
        << "STO-3G H self-kinetic should be positive";

    // Known reference value: STO-3G H self-kinetic ~ 0.7600
    EXPECT_NEAR(buffer_11(0, 0), 0.7600, 0.005)
        << "STO-3G H self-kinetic should be approximately 0.7600";

    // Cross-kinetic should be symmetric
    KineticBuffer buffer_12, buffer_21;
    kernels::compute_kinetic(shell_H1, shell_H2, buffer_12);
    kernels::compute_kinetic(shell_H2, shell_H1, buffer_21);

    EXPECT_NEAR(buffer_12(0, 0), buffer_21(0, 0), TIGHT_TOL)
        << "T(H1, H2) should equal T(H2, H1)";
}

// =============================================================================
// Test 14: STO-3G H2O kinetic matrix symmetries
// =============================================================================
TEST(KineticKernelTest, STO3G_H2O_Properties) {
    // H2O geometry (bohr)
    Point3D O_pos(0.0, 0.0, 0.0);
    Point3D H1_pos(0.0, 1.43233673, -1.10866041);
    Point3D H2_pos(0.0, -1.43233673, -1.10866041);

    Shell O_1s = make_sto3g_oxygen_1s(O_pos);
    Shell O_2s = make_sto3g_oxygen_2s(O_pos);
    Shell O_2p = make_sto3g_oxygen_2p(O_pos);
    Shell H1_1s = make_sto3g_hydrogen(H1_pos);
    Shell H2_1s = make_sto3g_hydrogen(H2_pos);

    // Self-kinetic should be positive
    {
        KineticBuffer buf;
        kernels::compute_kinetic(O_1s, O_1s, buf);
        EXPECT_GT(buf(0, 0), 0.0) << "O_1s self-kinetic should be positive";
    }
    {
        KineticBuffer buf;
        kernels::compute_kinetic(O_2s, O_2s, buf);
        EXPECT_GT(buf(0, 0), 0.0) << "O_2s self-kinetic should be positive";
    }
    {
        KineticBuffer buf;
        kernels::compute_kinetic(O_2p, O_2p, buf);
        for (int i = 0; i < 3; ++i) {
            EXPECT_GT(buf(i, i), 0.0) << "O_2p(" << i << ") self-kinetic should be positive";
        }
        // All p-component self-kinetics should be equal
        EXPECT_NEAR(buf(0, 0), buf(1, 1), TIGHT_TOL)
            << "O_2px and O_2py self-kinetic should be equal";
        EXPECT_NEAR(buf(0, 0), buf(2, 2), TIGHT_TOL)
            << "O_2px and O_2pz self-kinetic should be equal";
        // Off-diagonal should be zero
        for (int i = 0; i < 3; ++i) {
            for (int j = 0; j < 3; ++j) {
                if (i != j) {
                    EXPECT_NEAR(buf(i, j), 0.0, TIGHT_TOL)
                        << "O_2p cross-kinetic (" << i << "," << j << ") should be zero";
                }
            }
        }
    }

    // T(O_1s, O_2p) = 0 at same center (s-p symmetry)
    {
        KineticBuffer buf;
        kernels::compute_kinetic(O_1s, O_2p, buf);
        for (int j = 0; j < 3; ++j) {
            EXPECT_NEAR(buf(0, j), 0.0, TIGHT_TOL)
                << "T(O_1s, O_2p_" << j << ") should be zero at same center";
        }
    }

    // T(O_2s, O_2p) = 0 at same center (s-p symmetry)
    {
        KineticBuffer buf;
        kernels::compute_kinetic(O_2s, O_2p, buf);
        for (int j = 0; j < 3; ++j) {
            EXPECT_NEAR(buf(0, j), 0.0, TIGHT_TOL)
                << "T(O_2s, O_2p_" << j << ") should be zero at same center";
        }
    }

    // By molecular symmetry, H1 and H2 should have the same kinetic with O_1s
    {
        KineticBuffer buf_H1, buf_H2;
        kernels::compute_kinetic(O_1s, H1_1s, buf_H1);
        kernels::compute_kinetic(O_1s, H2_1s, buf_H2);
        EXPECT_NEAR(buf_H1(0, 0), buf_H2(0, 0), TIGHT_TOL)
            << "T(O_1s, H1) should equal T(O_1s, H2)";
    }

    // T(O_2px, H) should be zero (H atoms in yz-plane)
    {
        KineticBuffer buf;
        kernels::compute_kinetic(O_2p, H1_1s, buf);
        EXPECT_NEAR(buf(0, 0), 0.0, TIGHT_TOL)
            << "T(O_2px, H1) should be zero";
    }

    // T(O_2py, H1) and T(O_2py, H2) should have opposite signs
    {
        KineticBuffer buf_H1, buf_H2;
        kernels::compute_kinetic(O_2p, H1_1s, buf_H1);
        kernels::compute_kinetic(O_2p, H2_1s, buf_H2);
        EXPECT_NEAR(buf_H1(1, 0), -buf_H2(1, 0), TIGHT_TOL)
            << "T(O_2py, H1) and T(O_2py, H2) should be opposite in sign";
    }

    // T(O_2pz, H1) and T(O_2pz, H2) should be equal
    {
        KineticBuffer buf_H1, buf_H2;
        kernels::compute_kinetic(O_2p, H1_1s, buf_H1);
        kernels::compute_kinetic(O_2p, H2_1s, buf_H2);
        EXPECT_NEAR(buf_H1(2, 0), buf_H2(2, 0), TIGHT_TOL)
            << "T(O_2pz, H1) and T(O_2pz, H2) should be equal";
    }
}

// =============================================================================
// Test 15: Contracted s-s self kinetic
// =============================================================================
TEST(KineticKernelTest, ContractedSsSelfKinetic) {
    // Contracted s-shell at same center
    // Self-kinetic should be positive
    Point3D center(1.0, 2.0, 3.0);
    Shell shell(AngularMomentum::S, center, {3.0, 1.0, 0.3}, {0.5, 0.3, 0.2});

    KineticBuffer buffer;
    kernels::compute_kinetic(shell, shell, buffer);

    EXPECT_GT(buffer(0, 0), 0.0)
        << "Contracted s-shell self-kinetic should be positive";
    EXPECT_TRUE(std::isfinite(buffer(0, 0)))
        << "Contracted s-shell self-kinetic should be finite";
}

// =============================================================================
// Test 16: f-f self kinetic at same center
// =============================================================================
TEST(KineticKernelTest, FfSelfSameCenter) {
    // f-shell self-kinetic at same center
    // - All diagonal elements should be positive
    // - Matrix should be symmetric
    // - Components related by permutation of x,y,z should have equal kinetic energy
    const Real alpha = 1.5;
    Point3D origin(0.0, 0.0, 0.0);
    Shell shell(AngularMomentum::F, origin, {alpha}, {1.0});

    KineticBuffer buffer;
    kernels::compute_kinetic(shell, shell, buffer);

    EXPECT_EQ(buffer.na(), 10);
    EXPECT_EQ(buffer.nb(), 10);

    // All diagonal elements should be positive
    for (int i = 0; i < 10; ++i) {
        EXPECT_GT(buffer(i, i), 0.0)
            << "f-f self kinetic diagonal (" << i << ") should be positive";
    }

    // Matrix should be symmetric
    for (int i = 0; i < 10; ++i) {
        for (int j = i + 1; j < 10; ++j) {
            EXPECT_NEAR(buffer(i, j), buffer(j, i), TIGHT_TOL)
                << "f-f kinetic symmetry: (" << i << "," << j << ")";
        }
    }

    // Canonical f ordering: (3,0,0)=0, (2,1,0)=1, (2,0,1)=2, (1,2,0)=3,
    //                       (1,1,1)=4, (1,0,2)=5, (0,3,0)=6, (0,2,1)=7,
    //                       (0,1,2)=8, (0,0,3)=9
    // fxxx=0, fyyy=6, fzzz=9 should all have the same kinetic energy
    EXPECT_NEAR(buffer(0, 0), buffer(6, 6), TIGHT_TOL)
        << "T(fxxx,fxxx) should equal T(fyyy,fyyy)";
    EXPECT_NEAR(buffer(0, 0), buffer(9, 9), TIGHT_TOL)
        << "T(fxxx,fxxx) should equal T(fzzz,fzzz)";

    // Components with same pattern: fxxy=1, fxxz=2, fxyy=3, fxzz=5, fyyz=7, fyzz=8
    // By permutation symmetry: T(fxxy,fxxy) = T(fxxz,fxxz) = T(fxyy,fxyy) etc.
    EXPECT_NEAR(buffer(1, 1), buffer(2, 2), TIGHT_TOL)
        << "T(fxxy,fxxy) should equal T(fxxz,fxxz)";
    EXPECT_NEAR(buffer(1, 1), buffer(3, 3), TIGHT_TOL)
        << "T(fxxy,fxxy) should equal T(fxyy,fxyy)";
    EXPECT_NEAR(buffer(1, 1), buffer(5, 5), TIGHT_TOL)
        << "T(fxxy,fxxy) should equal T(fxzz,fxzz)";
    EXPECT_NEAR(buffer(1, 1), buffer(7, 7), TIGHT_TOL)
        << "T(fxxy,fxxy) should equal T(fyyz,fyyz)";
    EXPECT_NEAR(buffer(1, 1), buffer(8, 8), TIGHT_TOL)
        << "T(fxxy,fxxy) should equal T(fyzz,fyzz)";
}

// =============================================================================
// Test 17: Kinetic energy behavior with distance for s-s
// =============================================================================
TEST(KineticKernelTest, SsBehaviorWithDistance) {
    // For s-s kinetic with alpha=beta=1:
    //   T = mu*(3 - 2*mu*R^2) * S = 0.5*(3 - R^2) * exp(-0.5*R^2)
    //
    // This function:
    //   - Is positive for R < sqrt(3) ~ 1.732
    //   - Crosses zero at R = sqrt(3)
    //   - Becomes negative for R > sqrt(3)
    //   - Decays to zero at large R
    //
    // We verify these properties and the analytical formula.
    Point3D A(0.0, 0.0, 0.0);
    Shell shell_a(AngularMomentum::S, A, {1.0}, {1.0});

    const Real R_zero = std::sqrt(3.0);

    for (double R = 0.5; R <= 5.0; R += 0.5) {
        Shell shell_b(AngularMomentum::S, Point3D(R, 0.0, 0.0), {1.0}, {1.0});

        KineticBuffer buffer;
        kernels::compute_kinetic(shell_a, shell_b, buffer);

        EXPECT_TRUE(std::isfinite(buffer(0, 0)))
            << "T should be finite at R=" << R;

        // Verify against analytical formula: T = 0.5*(3 - R^2) * exp(-0.5*R^2)
        const Real expected = 0.5 * (3.0 - R * R) * std::exp(-0.5 * R * R);
        EXPECT_NEAR(buffer(0, 0), expected, TIGHT_TOL)
            << "T should match analytical formula at R=" << R;

        // Verify sign behavior
        if (R < R_zero - 0.1) {
            EXPECT_GT(buffer(0, 0), 0.0) << "T should be positive at R=" << R;
        } else if (R > R_zero + 0.1) {
            EXPECT_LT(buffer(0, 0), 0.0) << "T should be negative at R=" << R;
        }
    }

    // At very large R, T should be essentially zero
    Shell shell_far(AngularMomentum::S, Point3D(10.0, 0.0, 0.0), {1.0}, {1.0});
    KineticBuffer buffer_far;
    kernels::compute_kinetic(shell_a, shell_far, buffer_far);
    EXPECT_NEAR(buffer_far(0, 0), 0.0, 1e-15)
        << "T should decay to zero at large distance";
}

// =============================================================================
// Test 18: Numerical stability with widely varying exponents
// =============================================================================
TEST(KineticKernelTest, NumericalStability) {
    Point3D origin(0.0, 0.0, 0.0);

    Shell shell_tight(AngularMomentum::S, origin, {100.0}, {1.0});
    Shell shell_diffuse(AngularMomentum::S, origin, {0.01}, {1.0});

    KineticBuffer buffer;
    kernels::compute_kinetic(shell_tight, shell_diffuse, buffer);

    EXPECT_TRUE(std::isfinite(buffer(0, 0)))
        << "Kinetic integral should be finite for widely varying exponents";
}

// =============================================================================
// Test 19: Zero kinetic for widely separated shells
// =============================================================================
TEST(KineticKernelTest, WidelySeparatedShells) {
    Point3D A(0.0, 0.0, 0.0);
    Point3D B(100.0, 0.0, 0.0);

    Shell shell_a(AngularMomentum::S, A, {1.0}, {1.0});
    Shell shell_b(AngularMomentum::S, B, {1.0}, {1.0});

    KineticBuffer buffer;
    kernels::compute_kinetic(shell_a, shell_b, buffer);

    EXPECT_NEAR(buffer(0, 0), 0.0, 1e-15)
        << "Kinetic integral of widely separated shells should be essentially zero";
}

// =============================================================================
// Test 20: Contracted p-p kinetic at different centers
// =============================================================================
TEST(KineticKernelTest, ContractedPpDifferentCenters) {
    Point3D A(0.0, 0.0, 0.0);
    Point3D B(0.0, 0.0, 0.8);

    Shell shell_a(AngularMomentum::P, A, {3.0, 1.0}, {0.6, 0.4});
    Shell shell_b(AngularMomentum::P, B, {3.0, 1.0}, {0.6, 0.4});

    KineticBuffer buffer;
    kernels::compute_kinetic(shell_a, shell_b, buffer);

    EXPECT_EQ(buffer.na(), 3);
    EXPECT_EQ(buffer.nb(), 3);

    // Separation along z: px-px and py-py kinetic should be equal by symmetry
    EXPECT_NEAR(buffer(0, 0), buffer(1, 1), TIGHT_TOL)
        << "T(px,px) and T(py,py) should be equal for z-separation";

    // Off-diagonal: px-py, px-pz, py-pz should be zero by symmetry
    EXPECT_NEAR(buffer(0, 1), 0.0, TIGHT_TOL) << "T(px,py) should be zero";
    EXPECT_NEAR(buffer(0, 2), 0.0, TIGHT_TOL) << "T(px,pz) should be zero";
    EXPECT_NEAR(buffer(1, 2), 0.0, TIGHT_TOL) << "T(py,pz) should be zero";

    // Verify symmetry T(a,b) = T(b,a)^T
    KineticBuffer buffer_ba;
    kernels::compute_kinetic(shell_b, shell_a, buffer_ba);
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            EXPECT_NEAR(buffer(i, j), buffer_ba(j, i), TIGHT_TOL)
                << "p-p kinetic symmetry at (" << i << "," << j << ")";
        }
    }
}

// =============================================================================
// Test 21: Kinetic integral relates to overlap via T = mu*(3-2*mu*R^2)*S for s-s
// =============================================================================
TEST(KineticKernelTest, SsKineticOverlapRelation) {
    // For normalized single-primitive s-shells with exponents alpha and beta:
    // T = mu * (3 - 2*mu*R^2) * S
    // where mu = alpha*beta/(alpha+beta)
    // This holds for any alpha, beta, and R.
    struct TestCase {
        Real alpha;
        Real beta;
        Real R;
    };

    std::vector<TestCase> cases = {
        {1.0, 1.0, 0.0},
        {1.0, 1.0, 1.0},
        {1.0, 1.0, 2.0},
        {2.0, 0.5, 1.0},
        {0.3, 3.0, 0.5},
        {5.0, 5.0, 0.1},
    };

    for (const auto& tc : cases) {
        Point3D A(0.0, 0.0, 0.0);
        Point3D B(tc.R, 0.0, 0.0);

        Shell shell_a(AngularMomentum::S, A, {tc.alpha}, {1.0});
        Shell shell_b(AngularMomentum::S, B, {tc.beta}, {1.0});

        KineticBuffer buffer_T;
        kernels::compute_kinetic(shell_a, shell_b, buffer_T);

        OverlapBuffer buffer_S;
        kernels::compute_overlap(shell_a, shell_b, buffer_S);

        const Real mu = tc.alpha * tc.beta / (tc.alpha + tc.beta);
        const Real R2 = tc.R * tc.R;
        const Real expected_T = mu * (3.0 - 2.0 * mu * R2) * buffer_S(0, 0);

        EXPECT_NEAR(buffer_T(0, 0), expected_T, TIGHT_TOL)
            << "T = mu*(3-2*mu*R^2)*S for alpha=" << tc.alpha
            << ", beta=" << tc.beta << ", R=" << tc.R;
    }
}

// =============================================================================
// Test 22: p-p self kinetic analytical value
// =============================================================================
TEST(KineticKernelTest, PpSelfKineticAnalytical) {
    // For a single-primitive normalized p-type GTO with exponent alpha:
    // T(pi, pi) = 5*alpha/2
    // This is because the second derivative of x*exp(-alpha*r^2) gives contributions
    // from all three directions, and the total works out to 5*alpha/2 for each
    // diagonal element.
    for (Real alpha : {0.5, 1.0, 2.0, 5.0}) {
        Point3D origin(0.0, 0.0, 0.0);
        Shell shell(AngularMomentum::P, origin, {alpha}, {1.0});

        KineticBuffer buffer;
        kernels::compute_kinetic(shell, shell, buffer);

        const Real expected = 2.5 * alpha;
        for (int i = 0; i < 3; ++i) {
            EXPECT_NEAR(buffer(i, i), expected, TIGHT_TOL)
                << "T(pi,pi) for alpha=" << alpha << ", component " << i;
        }
    }
}

// =============================================================================
// PrimitivePairData Overload Tests
// =============================================================================

namespace {
ShellSet make_shell_set_from_kinetic(const std::vector<Shell>& shells) {
    std::vector<std::reference_wrapper<const Shell>> refs;
    refs.reserve(shells.size());
    for (const auto& s : shells) {
        refs.push_back(std::cref(s));
    }
    return ShellSet(refs);
}
}  // anonymous namespace

TEST(KineticKernelTest, PairDataOverload_SS_MatchesOnTheFly) {
    Shell s1 = make_sto3g_hydrogen(Point3D(0.0, 0.0, 0.0));
    Shell s2 = make_sto3g_hydrogen(Point3D(1.4, 0.0, 0.0));

    std::vector<Shell> shells_a = {s1};
    std::vector<Shell> shells_b = {s2};
    ShellSet set_a = make_shell_set_from_kinetic(shells_a);
    ShellSet set_b = make_shell_set_from_kinetic(shells_b);
    ShellSetPair pair(set_a, set_b);

    KineticBuffer buf_direct, buf_cached;
    kernels::compute_kinetic(s1, s2, buf_direct);
    kernels::compute_kinetic(s1, s2, pair.pair_data(), 0, 0, buf_cached);

    EXPECT_NEAR(buf_direct(0, 0), buf_cached(0, 0), TIGHT_TOL);
}

TEST(KineticKernelTest, PairDataOverload_SP_MatchesOnTheFly) {
    Shell s1 = make_sto3g_oxygen_2s(Point3D(0.0, 0.0, 0.0));
    Shell p1 = make_sto3g_oxygen_2p(Point3D(1.0, 0.0, 0.0));

    std::vector<Shell> shells_s = {s1};
    std::vector<Shell> shells_p = {p1};
    ShellSet set_s = make_shell_set_from_kinetic(shells_s);
    ShellSet set_p = make_shell_set_from_kinetic(shells_p);
    ShellSetPair pair(set_s, set_p);

    KineticBuffer buf_direct, buf_cached;
    kernels::compute_kinetic(s1, p1, buf_direct);
    kernels::compute_kinetic(s1, p1, pair.pair_data(), 0, 0, buf_cached);

    const int na = s1.n_functions();
    const int nb = p1.n_functions();
    for (int a = 0; a < na; ++a) {
        for (int b = 0; b < nb; ++b) {
            EXPECT_NEAR(buf_direct(a, b), buf_cached(a, b), TIGHT_TOL)
                << "Mismatch at (" << a << ", " << b << ")";
        }
    }
}
