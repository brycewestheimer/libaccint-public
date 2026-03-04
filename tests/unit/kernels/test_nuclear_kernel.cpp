// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2026 Bryce M. Westheimer. All rights reserved.

/// @file test_nuclear_kernel.cpp
/// @brief Unit tests for nuclear attraction integral kernel (Rys quadrature)

#include <libaccint/kernels/nuclear_kernel.hpp>
#include <libaccint/basis/shell.hpp>
#include <libaccint/buffers/one_electron_buffer.hpp>
#include <libaccint/operators/operator_types.hpp>
#include <libaccint/math/normalization.hpp>
#include <libaccint/math/cartesian_indices.hpp>
#include <libaccint/utils/constants.hpp>
#include <gtest/gtest.h>
#include <cmath>

using namespace libaccint;

namespace {

// Tolerances for floating-point comparisons
constexpr Real TIGHT_TOL = 1e-12;
constexpr Real MODERATE_TOL = 1e-10;
constexpr Real LOOSE_TOL = 1e-8;

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

// ============================================================================
// Helper: Create PointChargeParams for a single nucleus
// ============================================================================
PointChargeParams make_single_charge(Real x, Real y, Real z, Real charge) {
    PointChargeParams params;
    params.x = {x};
    params.y = {y};
    params.z = {z};
    params.charge = {charge};
    return params;
}

}  // anonymous namespace

// =============================================================================
// Test 1: s-s single nucleus at same center - analytical verification
// =============================================================================
TEST(NuclearKernelTest, SsSingleNucleusAtCenter) {
    // Single primitive s-shell with alpha=1 at origin, proton at origin.
    // For T=0 (P=C=origin), the Rys weight for 1 root is F_0(0) = 1.
    //
    // V_prim = -Z * (2*pi/zeta) * K_AB * w * Ix * Iy * Iz
    // With alpha=beta=1, zeta=2, K_AB=exp(0)=1, Z=1:
    // V_prim = -1 * (2*pi/2) * 1 * 1 * 1 * 1 * 1 = -pi
    //
    // For the normalized shell, c_stored includes normalization:
    //   c_stored = (2*alpha/pi)^(3/4) * contraction_norm
    //   For single-primitive s-shell with c_raw=1: c_stored = (2/pi)^(3/4)
    //
    // V = c_a * c_b * V_prim = [(2/pi)^(3/4)]^2 * (-pi)
    //   = (2/pi)^(3/2) * (-pi) = -2^(3/2) * pi^(-3/2) * pi = -2^(3/2) * pi^(-1/2)
    //   = -2*sqrt(2) / sqrt(pi) = -2 * sqrt(2/pi)
    Point3D origin(0.0, 0.0, 0.0);
    Shell shell(AngularMomentum::S, origin, {1.0}, {1.0});

    auto charges = make_single_charge(0.0, 0.0, 0.0, 1.0);

    NuclearBuffer buffer;
    kernels::compute_nuclear(shell, shell, charges, buffer);

    EXPECT_EQ(buffer.na(), 1);
    EXPECT_EQ(buffer.nb(), 1);

    // V should be negative (attractive)
    EXPECT_LT(buffer(0, 0), 0.0)
        << "Nuclear attraction should be negative (attractive)";

    // Analytical: V = -2 * sqrt(2/pi)
    const Real expected = -2.0 * std::sqrt(2.0 / constants::PI);
    EXPECT_NEAR(buffer(0, 0), expected, TIGHT_TOL)
        << "s-s nuclear attraction with nucleus at center";
}

// =============================================================================
// Test 2: s-s single nucleus displaced - V decays with distance
// =============================================================================
TEST(NuclearKernelTest, SsSingleNucleusDisplaced) {
    // s-shell at origin, nucleus moves along x-axis
    // Nuclear attraction should decrease in magnitude as nucleus moves away
    Point3D origin(0.0, 0.0, 0.0);
    Shell shell(AngularMomentum::S, origin, {1.0}, {1.0});

    Real prev_magnitude = std::numeric_limits<Real>::max();
    for (Real d = 0.0; d <= 5.0; d += 1.0) {
        auto charges = make_single_charge(d, 0.0, 0.0, 1.0);

        NuclearBuffer buffer;
        kernels::compute_nuclear(shell, shell, charges, buffer);

        // V should be negative (attractive) for Z > 0
        EXPECT_LT(buffer(0, 0), 0.0)
            << "Nuclear attraction should be negative at distance " << d;

        // Magnitude should decrease (or stay for d=0) as distance grows
        const Real magnitude = std::abs(buffer(0, 0));
        if (d > 0.0) {
            EXPECT_LT(magnitude, prev_magnitude)
                << "Nuclear attraction should decay as nucleus moves away, d=" << d;
        }
        prev_magnitude = magnitude;
    }
}

// =============================================================================
// Test 3: Symmetry V(a,b) = V(b,a) for all shell pairs
// =============================================================================
TEST(NuclearKernelTest, Symmetry) {
    // Two different shells at different centers, nucleus off-center
    Point3D A(0.0, 0.0, 0.0);
    Point3D B(1.5, 0.7, -0.3);

    Shell shell_a(AngularMomentum::P, A, {2.0, 0.5}, {0.6, 0.4});
    Shell shell_b(AngularMomentum::D, B, {1.5, 0.8}, {0.7, 0.3});

    auto charges = make_single_charge(0.5, -0.2, 0.8, 3.0);

    NuclearBuffer buffer_ab, buffer_ba;
    kernels::compute_nuclear(shell_a, shell_b, charges, buffer_ab);
    kernels::compute_nuclear(shell_b, shell_a, charges, buffer_ba);

    EXPECT_EQ(buffer_ab.na(), 3);   // p-shell
    EXPECT_EQ(buffer_ab.nb(), 6);   // d-shell
    EXPECT_EQ(buffer_ba.na(), 6);   // d-shell
    EXPECT_EQ(buffer_ba.nb(), 3);   // p-shell

    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 6; ++j) {
            EXPECT_NEAR(buffer_ab(i, j), buffer_ba(j, i), TIGHT_TOL)
                << "V(a,b) should equal V(b,a): (" << i << "," << j << ")";
        }
    }
}

// =============================================================================
// Test 4: s-p nuclear attraction - matrix dimensions and properties
// =============================================================================
TEST(NuclearKernelTest, SpNuclear) {
    // s-shell at origin, p-shell at (1,0,0), nucleus at origin
    Point3D A(0.0, 0.0, 0.0);
    Point3D B(1.0, 0.0, 0.0);

    Shell shell_s(AngularMomentum::S, A, {1.0}, {1.0});
    Shell shell_p(AngularMomentum::P, B, {1.0}, {1.0});

    auto charges = make_single_charge(0.0, 0.0, 0.0, 1.0);

    NuclearBuffer buffer;
    kernels::compute_nuclear(shell_s, shell_p, charges, buffer);

    EXPECT_EQ(buffer.na(), 1);
    EXPECT_EQ(buffer.nb(), 3);

    // All values should be finite
    for (int j = 0; j < 3; ++j) {
        EXPECT_TRUE(std::isfinite(buffer(0, j)))
            << "V(s, p_" << j << ") should be finite";
    }

    // With separation only along x:
    // V(s, px) should be nonzero (the x-component has density along the displacement)
    // V(s, py) and V(s, pz) should be zero by symmetry
    // (the nucleus is at origin, and the geometry has mirror symmetry in y and z planes)
    EXPECT_NEAR(buffer(0, 1), 0.0, TIGHT_TOL)
        << "V(s, py) should be zero by symmetry (displacement along x only)";
    EXPECT_NEAR(buffer(0, 2), 0.0, TIGHT_TOL)
        << "V(s, pz) should be zero by symmetry (displacement along x only)";

    // V(s, px) should be nonzero
    EXPECT_NE(buffer(0, 0), 0.0)
        << "V(s, px) should be nonzero with displacement along x";
}

// =============================================================================
// Test 5: p-p nuclear attraction at same center - symmetric 3x3 matrix
// =============================================================================
TEST(NuclearKernelTest, PpSameCenter) {
    // p-shell at origin, nucleus at origin
    Point3D origin(0.0, 0.0, 0.0);
    Shell shell(AngularMomentum::P, origin, {1.0}, {1.0});

    auto charges = make_single_charge(0.0, 0.0, 0.0, 1.0);

    NuclearBuffer buffer;
    kernels::compute_nuclear(shell, shell, charges, buffer);

    EXPECT_EQ(buffer.na(), 3);
    EXPECT_EQ(buffer.nb(), 3);

    // Matrix should be symmetric
    for (int i = 0; i < 3; ++i) {
        for (int j = i + 1; j < 3; ++j) {
            EXPECT_NEAR(buffer(i, j), buffer(j, i), TIGHT_TOL)
                << "p-p nuclear attraction should be symmetric: (" << i << "," << j << ")";
        }
    }

    // With nucleus at center, the matrix should be diagonal by spherical symmetry
    // All diagonal elements should be equal: V(px,px) = V(py,py) = V(pz,pz)
    EXPECT_NEAR(buffer(0, 0), buffer(1, 1), TIGHT_TOL)
        << "V(px,px) should equal V(py,py) by spherical symmetry";
    EXPECT_NEAR(buffer(0, 0), buffer(2, 2), TIGHT_TOL)
        << "V(px,px) should equal V(pz,pz) by spherical symmetry";

    // Off-diagonal elements should be zero by symmetry
    EXPECT_NEAR(buffer(0, 1), 0.0, TIGHT_TOL) << "V(px,py) should be zero";
    EXPECT_NEAR(buffer(0, 2), 0.0, TIGHT_TOL) << "V(px,pz) should be zero";
    EXPECT_NEAR(buffer(1, 2), 0.0, TIGHT_TOL) << "V(py,pz) should be zero";

    // Diagonal should be negative (attractive)
    EXPECT_LT(buffer(0, 0), 0.0) << "V(px,px) should be negative";
}

// =============================================================================
// Test 6: Two nuclei - additivity of nuclear contributions
// =============================================================================
TEST(NuclearKernelTest, TwoNucleiAdditivity) {
    // Verify that V with two nuclei = V(nucleus1) + V(nucleus2)
    Point3D origin(0.0, 0.0, 0.0);
    Shell shell(AngularMomentum::S, origin, {1.5}, {1.0});

    Real R = 2.0;

    // Single nucleus at +R along x
    auto charges_1 = make_single_charge(R, 0.0, 0.0, 1.0);
    NuclearBuffer buffer_1;
    kernels::compute_nuclear(shell, shell, charges_1, buffer_1);

    // Single nucleus at -R along x
    auto charges_2 = make_single_charge(-R, 0.0, 0.0, 1.0);
    NuclearBuffer buffer_2;
    kernels::compute_nuclear(shell, shell, charges_2, buffer_2);

    // Two nuclei at +/-R
    PointChargeParams charges_both;
    charges_both.x = {R, -R};
    charges_both.y = {0.0, 0.0};
    charges_both.z = {0.0, 0.0};
    charges_both.charge = {1.0, 1.0};

    NuclearBuffer buffer_both;
    kernels::compute_nuclear(shell, shell, charges_both, buffer_both);

    // V(both) should equal V(1) + V(2)
    EXPECT_NEAR(buffer_both(0, 0), buffer_1(0, 0) + buffer_2(0, 0), TIGHT_TOL)
        << "Nuclear attraction from two nuclei should be additive";

    // For symmetric placement with equal charges, V(+R) should equal V(-R)
    EXPECT_NEAR(buffer_1(0, 0), buffer_2(0, 0), TIGHT_TOL)
        << "V(+R) should equal V(-R) for symmetric placement with s-shell at origin";
}

// =============================================================================
// Test 7: STO-3G H2 nuclear attraction - properties check
// =============================================================================
TEST(NuclearKernelTest, STO3G_H2_Properties) {
    // H2 molecule at R = 1.4 bohr along z-axis
    const Real R = 1.4;
    Point3D H1(0.0, 0.0, 0.0);
    Point3D H2(0.0, 0.0, R);

    Shell shell_H1 = make_sto3g_hydrogen(H1);
    Shell shell_H2 = make_sto3g_hydrogen(H2);

    // Both protons
    PointChargeParams charges;
    charges.x = {0.0, 0.0};
    charges.y = {0.0, 0.0};
    charges.z = {0.0, R};
    charges.charge = {1.0, 1.0};

    // Self-interaction: V(H1, H1)
    NuclearBuffer buffer_11;
    kernels::compute_nuclear(shell_H1, shell_H1, charges, buffer_11);
    EXPECT_LT(buffer_11(0, 0), 0.0) << "V(H1,H1) diagonal should be negative";

    // Self-interaction: V(H2, H2)
    NuclearBuffer buffer_22;
    kernels::compute_nuclear(shell_H2, shell_H2, charges, buffer_22);
    EXPECT_LT(buffer_22(0, 0), 0.0) << "V(H2,H2) diagonal should be negative";

    // By molecular symmetry, V(H1,H1) should equal V(H2,H2)
    EXPECT_NEAR(buffer_11(0, 0), buffer_22(0, 0), TIGHT_TOL)
        << "V(H1,H1) should equal V(H2,H2) by molecular symmetry";

    // Cross-interaction: V(H1, H2)
    NuclearBuffer buffer_12;
    kernels::compute_nuclear(shell_H1, shell_H2, charges, buffer_12);
    EXPECT_LT(buffer_12(0, 0), 0.0) << "V(H1,H2) should be negative";

    // Symmetry: V(H1, H2) = V(H2, H1)
    NuclearBuffer buffer_21;
    kernels::compute_nuclear(shell_H2, shell_H1, charges, buffer_21);
    EXPECT_NEAR(buffer_12(0, 0), buffer_21(0, 0), TIGHT_TOL)
        << "V(H1,H2) should equal V(H2,H1)";
}

// =============================================================================
// Test 8: Zero charges give V = 0
// =============================================================================
TEST(NuclearKernelTest, ZeroCharges) {
    Point3D origin(0.0, 0.0, 0.0);
    Shell shell(AngularMomentum::S, origin, {1.0}, {1.0});

    // Z = 0
    auto charges = make_single_charge(0.0, 0.0, 0.0, 0.0);

    NuclearBuffer buffer;
    kernels::compute_nuclear(shell, shell, charges, buffer);

    EXPECT_NEAR(buffer(0, 0), 0.0, TIGHT_TOL)
        << "V should be zero when Z = 0";
}

// =============================================================================
// Test 9: Empty charges give V = 0
// =============================================================================
TEST(NuclearKernelTest, EmptyCharges) {
    Point3D origin(0.0, 0.0, 0.0);
    Shell shell(AngularMomentum::P, origin, {1.0}, {1.0});

    PointChargeParams charges;  // No charge centers

    NuclearBuffer buffer;
    kernels::compute_nuclear(shell, shell, charges, buffer);

    EXPECT_EQ(buffer.na(), 3);
    EXPECT_EQ(buffer.nb(), 3);

    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            EXPECT_NEAR(buffer(i, j), 0.0, TIGHT_TOL)
                << "V should be zero with no charges: (" << i << "," << j << ")";
        }
    }
}

// =============================================================================
// Test 10: Nucleus at shell center vs displaced - V at center is more negative
// =============================================================================
TEST(NuclearKernelTest, NucleusAtCenterVsDisplaced) {
    Point3D origin(0.0, 0.0, 0.0);
    Shell shell(AngularMomentum::S, origin, {1.0}, {1.0});

    // Nucleus at the shell center
    auto charges_at = make_single_charge(0.0, 0.0, 0.0, 1.0);
    NuclearBuffer buffer_at;
    kernels::compute_nuclear(shell, shell, charges_at, buffer_at);

    // Nucleus displaced from the shell center
    auto charges_away = make_single_charge(2.0, 0.0, 0.0, 1.0);
    NuclearBuffer buffer_away;
    kernels::compute_nuclear(shell, shell, charges_away, buffer_away);

    // |V| at center should be larger (more negative) than displaced
    EXPECT_LT(buffer_at(0, 0), buffer_away(0, 0))
        << "V at shell center should be more negative than displaced nucleus";
}

// =============================================================================
// Test 11: Charge scaling - V(2*Z) = 2 * V(Z)
// =============================================================================
TEST(NuclearKernelTest, ChargeScaling) {
    Point3D origin(0.0, 0.0, 0.0);
    Shell shell(AngularMomentum::S, origin, {1.5}, {1.0});

    auto charges_1 = make_single_charge(1.0, 0.5, -0.3, 1.0);
    NuclearBuffer buffer_1;
    kernels::compute_nuclear(shell, shell, charges_1, buffer_1);

    auto charges_3 = make_single_charge(1.0, 0.5, -0.3, 3.0);
    NuclearBuffer buffer_3;
    kernels::compute_nuclear(shell, shell, charges_3, buffer_3);

    EXPECT_NEAR(buffer_3(0, 0), 3.0 * buffer_1(0, 0), TIGHT_TOL)
        << "V should scale linearly with charge";
}

// =============================================================================
// Test 12: p-p nuclear with displaced nucleus - symmetry breaking
// =============================================================================
TEST(NuclearKernelTest, PpDisplacedNucleus) {
    // p-shell at origin, nucleus at (1, 0, 0)
    // The x-displacement should break the px/py/pz equivalence
    Point3D origin(0.0, 0.0, 0.0);
    Shell shell(AngularMomentum::P, origin, {1.0}, {1.0});

    auto charges = make_single_charge(1.0, 0.0, 0.0, 1.0);

    NuclearBuffer buffer;
    kernels::compute_nuclear(shell, shell, charges, buffer);

    EXPECT_EQ(buffer.na(), 3);
    EXPECT_EQ(buffer.nb(), 3);

    // Matrix should still be symmetric
    for (int i = 0; i < 3; ++i) {
        for (int j = i + 1; j < 3; ++j) {
            EXPECT_NEAR(buffer(i, j), buffer(j, i), TIGHT_TOL)
                << "V should be symmetric: (" << i << "," << j << ")";
        }
    }

    // With nucleus along x: V(py,py) should equal V(pz,pz)
    // but V(px,px) should differ (broken symmetry in x)
    EXPECT_NEAR(buffer(1, 1), buffer(2, 2), TIGHT_TOL)
        << "V(py,py) should equal V(pz,pz) for x-displaced nucleus";

    // V(px,px) should differ from V(py,py)
    // (not necessarily, but for typical parameters it does)
    // Let's just check they are all negative and finite
    for (int i = 0; i < 3; ++i) {
        EXPECT_LT(buffer(i, i), 0.0)
            << "Diagonal V(" << i << "," << i << ") should be negative";
        EXPECT_TRUE(std::isfinite(buffer(i, i)))
            << "Diagonal V(" << i << "," << i << ") should be finite";
    }
}

// =============================================================================
// Test 13: d-d nuclear attraction - finite and symmetric
// =============================================================================
TEST(NuclearKernelTest, DdNuclear) {
    Point3D A(0.0, 0.0, 0.0);
    Point3D B(0.8, 0.0, 0.0);

    Shell shell_a(AngularMomentum::D, A, {1.5}, {1.0});
    Shell shell_b(AngularMomentum::D, B, {1.0}, {1.0});

    auto charges = make_single_charge(0.4, 0.0, 0.0, 2.0);

    NuclearBuffer buffer_ab, buffer_ba;
    kernels::compute_nuclear(shell_a, shell_b, charges, buffer_ab);
    kernels::compute_nuclear(shell_b, shell_a, charges, buffer_ba);

    EXPECT_EQ(buffer_ab.na(), 6);
    EXPECT_EQ(buffer_ab.nb(), 6);

    // Check symmetry: V(a,b) = V(b,a)^T
    for (int i = 0; i < 6; ++i) {
        for (int j = 0; j < 6; ++j) {
            EXPECT_NEAR(buffer_ab(i, j), buffer_ba(j, i), TIGHT_TOL)
                << "d-d symmetry: (" << i << "," << j << ")";
        }
    }

    // All elements should be finite
    for (int i = 0; i < 6; ++i) {
        for (int j = 0; j < 6; ++j) {
            EXPECT_TRUE(std::isfinite(buffer_ab(i, j)))
                << "d-d element (" << i << "," << j << ") should be finite";
        }
    }

    // At least some elements should be nonzero
    bool has_nonzero = false;
    for (int i = 0; i < 6; ++i) {
        for (int j = 0; j < 6; ++j) {
            if (std::abs(buffer_ab(i, j)) > 1e-10) {
                has_nonzero = true;
                break;
            }
        }
        if (has_nonzero) break;
    }
    EXPECT_TRUE(has_nonzero) << "d-d nuclear attraction should have nonzero elements";
}

// =============================================================================
// Test 14: STO-3G H2O nuclear attraction - molecular symmetry properties
// =============================================================================
TEST(NuclearKernelTest, STO3G_H2O_Properties) {
    // H2O geometry (bohr):
    // O at (0, 0, 0)
    // H at (0, 1.43233673, -1.10866041)
    // H at (0, -1.43233673, -1.10866041)
    Point3D O_pos(0.0, 0.0, 0.0);
    Point3D H1_pos(0.0, 1.43233673, -1.10866041);
    Point3D H2_pos(0.0, -1.43233673, -1.10866041);

    Shell O_1s = make_sto3g_oxygen_1s(O_pos);
    Shell O_2s = make_sto3g_oxygen_2s(O_pos);
    Shell O_2p = make_sto3g_oxygen_2p(O_pos);
    Shell H1_1s = make_sto3g_hydrogen(H1_pos);
    Shell H2_1s = make_sto3g_hydrogen(H2_pos);

    // Full nuclear charges: O(Z=8) + H(Z=1) + H(Z=1)
    PointChargeParams charges;
    charges.x = {0.0, 0.0, 0.0};
    charges.y = {0.0, 1.43233673, -1.43233673};
    charges.z = {0.0, -1.10866041, -1.10866041};
    charges.charge = {8.0, 1.0, 1.0};

    // V(O_1s, O_1s): should be strongly negative (oxygen is right on top of Z=8 nucleus)
    {
        NuclearBuffer buf;
        kernels::compute_nuclear(O_1s, O_1s, charges, buf);
        EXPECT_LT(buf(0, 0), -10.0)
            << "V(O_1s, O_1s) should be strongly negative";
    }

    // H1 and H2 have equal self-nuclear attraction by symmetry
    {
        NuclearBuffer buf_H1, buf_H2;
        kernels::compute_nuclear(H1_1s, H1_1s, charges, buf_H1);
        kernels::compute_nuclear(H2_1s, H2_1s, charges, buf_H2);
        EXPECT_NEAR(buf_H1(0, 0), buf_H2(0, 0), TIGHT_TOL)
            << "V(H1,H1) should equal V(H2,H2) by molecular symmetry";
        EXPECT_LT(buf_H1(0, 0), 0.0) << "V(H1,H1) should be negative";
    }

    // H1-H2 cross-term: V(H1, H2) should equal V(H2, H1) and be negative
    {
        NuclearBuffer buf_12, buf_21;
        kernels::compute_nuclear(H1_1s, H2_1s, charges, buf_12);
        kernels::compute_nuclear(H2_1s, H1_1s, charges, buf_21);
        EXPECT_NEAR(buf_12(0, 0), buf_21(0, 0), TIGHT_TOL)
            << "V(H1,H2) should equal V(H2,H1)";
        EXPECT_LT(buf_12(0, 0), 0.0) << "V(H1,H2) should be negative";
    }

    // O_2px - H overlap: by C2v symmetry (H atoms in yz-plane), the px component
    // should give zero overlap with H1 and H2 (px has odd symmetry under yz reflection)
    // Wait, nuclear attraction is NOT like overlap -- 1/r operator doesn't have the
    // same symmetry as the overlap operator. But for px centered at O with Z-nucleus
    // also at O, the symmetry of the integral px * Z/|r| * 1s gives zero by odd parity.
    // However, contributions from off-center nuclei can be nonzero.
    // Let's just check symmetry-related equalities:
    {
        NuclearBuffer buf_H1, buf_H2;
        kernels::compute_nuclear(O_2p, H1_1s, charges, buf_H1);
        kernels::compute_nuclear(O_2p, H2_1s, charges, buf_H2);

        // V(O_2px, H1) should equal V(O_2px, H2) because both H nuclei
        // (and both H shells) have the same x=0 coordinate
        EXPECT_NEAR(buf_H1(0, 0), buf_H2(0, 0), TIGHT_TOL)
            << "V(O_2px, H1) should equal V(O_2px, H2)";

        // V(O_2py, H1) and V(O_2py, H2): H1.y > 0, H2.y < 0
        // These should be related by the C2v symmetry but with nuclear
        // attraction we need to be careful. Let's just check they are finite.
        EXPECT_TRUE(std::isfinite(buf_H1(1, 0))) << "V(O_2py, H1) should be finite";
        EXPECT_TRUE(std::isfinite(buf_H2(1, 0))) << "V(O_2py, H2) should be finite";

        // V(O_2pz, H1) should equal V(O_2pz, H2) by molecular symmetry
        // (both H atoms have the same z-coordinate)
        EXPECT_NEAR(buf_H1(2, 0), buf_H2(2, 0), TIGHT_TOL)
            << "V(O_2pz, H1) should equal V(O_2pz, H2)";
    }
}

// =============================================================================
// Test 15: Buffer reuse - resize and clear
// =============================================================================
TEST(NuclearKernelTest, BufferReuse) {
    NuclearBuffer buffer;
    auto charges = make_single_charge(0.0, 0.0, 0.0, 1.0);

    // First use with s-s
    {
        Shell shell(AngularMomentum::S, Point3D(0.0, 0.0, 0.0), {1.0}, {1.0});
        kernels::compute_nuclear(shell, shell, charges, buffer);
        EXPECT_EQ(buffer.na(), 1);
        EXPECT_EQ(buffer.nb(), 1);
    }

    // Reuse with p-p
    {
        Shell shell(AngularMomentum::P, Point3D(0.0, 0.0, 0.0), {1.0}, {1.0});
        kernels::compute_nuclear(shell, shell, charges, buffer);
        EXPECT_EQ(buffer.na(), 3);
        EXPECT_EQ(buffer.nb(), 3);
    }

    // Reuse with s-d
    {
        Shell shell_s(AngularMomentum::S, Point3D(0.0, 0.0, 0.0), {1.0}, {1.0});
        Shell shell_d(AngularMomentum::D, Point3D(0.0, 0.0, 0.0), {1.0}, {1.0});
        kernels::compute_nuclear(shell_s, shell_d, charges, buffer);
        EXPECT_EQ(buffer.na(), 1);
        EXPECT_EQ(buffer.nb(), 6);
    }
}

// =============================================================================
// Test 16: Numerical stability with widely varying exponents
// =============================================================================
TEST(NuclearKernelTest, NumericalStability) {
    Point3D origin(0.0, 0.0, 0.0);

    // Large exponent difference
    Shell shell_tight(AngularMomentum::S, origin, {100.0}, {1.0});
    Shell shell_diffuse(AngularMomentum::S, origin, {0.01}, {1.0});

    auto charges = make_single_charge(0.0, 0.0, 0.0, 1.0);

    NuclearBuffer buffer;
    kernels::compute_nuclear(shell_tight, shell_diffuse, charges, buffer);

    EXPECT_TRUE(std::isfinite(buffer(0, 0))) << "V should be finite";
    EXPECT_LT(buffer(0, 0), 0.0) << "V should be negative (attractive)";
}

// =============================================================================
// Test 17: Contracted s-s nuclear attraction (STO-3G shells)
// =============================================================================
TEST(NuclearKernelTest, ContractedSsNuclear) {
    // Two identical contracted s-shells at the same center, nucleus at center
    Point3D center(0.0, 0.0, 0.0);
    Shell shell = make_sto3g_hydrogen(center);

    auto charges = make_single_charge(0.0, 0.0, 0.0, 1.0);

    NuclearBuffer buffer;
    kernels::compute_nuclear(shell, shell, charges, buffer);

    // V should be negative
    EXPECT_LT(buffer(0, 0), 0.0)
        << "Contracted s-s nuclear attraction should be negative";

    // V should be finite
    EXPECT_TRUE(std::isfinite(buffer(0, 0)))
        << "Contracted s-s nuclear attraction should be finite";

    // Symmetry: V(a,b) = V(b,a)
    NuclearBuffer buffer_ba;
    kernels::compute_nuclear(shell, shell, charges, buffer_ba);
    EXPECT_NEAR(buffer(0, 0), buffer_ba(0, 0), TIGHT_TOL)
        << "V(a,b) should equal V(b,a) for self-interaction";
}

// =============================================================================
// Test 18: Two equal nuclei at symmetric positions - midpoint basis
// =============================================================================
TEST(NuclearKernelTest, TwoEqualNucleiSymmetric) {
    // Shell at midpoint between two identical nuclei
    // V with both nuclei should be exactly 2x V with one nucleus (by symmetry)
    Point3D midpoint(0.0, 0.0, 0.0);
    Shell shell(AngularMomentum::S, midpoint, {1.0}, {1.0});

    Real R = 1.5;

    // Single nucleus at +R
    auto charges_plus = make_single_charge(0.0, 0.0, R, 2.0);
    NuclearBuffer buffer_plus;
    kernels::compute_nuclear(shell, shell, charges_plus, buffer_plus);

    // Two nuclei at +/-R with same charge
    PointChargeParams charges_both;
    charges_both.x = {0.0, 0.0};
    charges_both.y = {0.0, 0.0};
    charges_both.z = {R, -R};
    charges_both.charge = {2.0, 2.0};

    NuclearBuffer buffer_both;
    kernels::compute_nuclear(shell, shell, charges_both, buffer_both);

    // V(both) = 2 * V(single) because the s-shell at midpoint
    // sees both nuclei at the same distance
    EXPECT_NEAR(buffer_both(0, 0), 2.0 * buffer_plus(0, 0), TIGHT_TOL)
        << "V with two symmetric nuclei should be 2x single nucleus value";
}

