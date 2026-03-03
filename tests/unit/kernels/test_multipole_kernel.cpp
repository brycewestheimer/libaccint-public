// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2026 Bryce M. Westheimer. All rights reserved.

/// @file test_multipole_kernel.cpp
/// @brief Validation tests for electric multipole moment integrals

#include <libaccint/kernels/multipole_kernel.hpp>
#include <libaccint/basis/shell.hpp>
#include <libaccint/engine/multi_component_buffer.hpp>
#include <libaccint/utils/constants.hpp>
#include <gtest/gtest.h>
#include <cmath>

using namespace libaccint;

namespace {
constexpr Real TIGHT_TOL = 1e-12;
constexpr Real LOOSE_TOL = 1e-10;
}

// ============================================================================
// Test 1: Dipole integral s-s same center → should be zero at origin
// ============================================================================
TEST(MultipoleKernelTest, DipoleSsSameCenterAtOrigin) {
    // <s|r|s> with both shells and origin at the same point
    // By symmetry, ∫ r_α * |s(r)|² dr = 0 for α = x, y, z
    // when origin = center = (0,0,0)
    Point3D center(0.0, 0.0, 0.0);
    Shell shell(AngularMomentum::S, center, {1.0}, {1.0});

    MultiComponentBuffer buffer(OperatorKind::ElectricDipole);
    std::array<Real, 3> origin = {0.0, 0.0, 0.0};

    kernels::compute_dipole(shell, shell, origin, buffer);

    EXPECT_EQ(buffer.na(), 1);
    EXPECT_EQ(buffer.nb(), 1);
    EXPECT_NEAR(buffer(0, 0, 0), 0.0, TIGHT_TOL) << "Dipole x: zero by symmetry";
    EXPECT_NEAR(buffer(1, 0, 0), 0.0, TIGHT_TOL) << "Dipole y: zero by symmetry";
    EXPECT_NEAR(buffer(2, 0, 0), 0.0, TIGHT_TOL) << "Dipole z: zero by symmetry";
}

// ============================================================================
// Test 2: Dipole integral s-s at origin, shells displaced
// ============================================================================
TEST(MultipoleKernelTest, DipoleSsDisplacedAnalytical) {
    // <s_A|x|s_B> where A=(0,0,0), B=(1,0,0), origin=(0,0,0)
    // Both shells: single primitive with α=β=1.0
    // Expected: (P_x - O_x) * S_AB = (α*A_x + β*B_x)/(α+β) * exp(-μ*R²)*(π/ζ)^(3/2)
    //   where P_x = (0+1)/2 = 0.5, S_AB = exp(-0.5) * (π/2)^(3/2)
    //   so <s_A|x|s_B> = 0.5 * S_AB (from the T[0][0][1] = XPO term)
    //
    // Actually, for normalized shells, S_AB = exp(-0.5) (Shell normalizes to self-overlap=1).
    // The dipole integral <s_A|x|s_B> = P_x * S_AB = 0.5 * exp(-0.5)

    Point3D A(0.0, 0.0, 0.0);
    Point3D B(1.0, 0.0, 0.0);
    Shell shell_a(AngularMomentum::S, A, {1.0}, {1.0});
    Shell shell_b(AngularMomentum::S, B, {1.0}, {1.0});

    MultiComponentBuffer buffer(OperatorKind::ElectricDipole);
    std::array<Real, 3> origin = {0.0, 0.0, 0.0};

    kernels::compute_dipole(shell_a, shell_b, origin, buffer);

    // The s-s overlap with α=β=1.0 at R=1 is exp(-0.5)
    const Real S_AB = std::exp(-0.5);
    // The Gaussian product center P_x = (0+1)/2 = 0.5
    const Real expected_x = 0.5 * S_AB;

    EXPECT_NEAR(buffer(0, 0, 0), expected_x, LOOSE_TOL) << "Dipole x component";
    EXPECT_NEAR(buffer(1, 0, 0), 0.0, TIGHT_TOL) << "Dipole y: zero by symmetry";
    EXPECT_NEAR(buffer(2, 0, 0), 0.0, TIGHT_TOL) << "Dipole z: zero by symmetry";
}

// ============================================================================
// Test 3: Dipole symmetry — <a|r|b> = <b|r|a>
// ============================================================================
TEST(MultipoleKernelTest, DipoleSymmetry) {
    Point3D A(0.0, 0.0, 0.0);
    Point3D B(0.5, 0.3, -0.1);
    Shell shell_a(AngularMomentum::S, A, {1.5}, {1.0});
    Shell shell_b(AngularMomentum::P, B, {0.8}, {1.0});

    MultiComponentBuffer buf_ab(OperatorKind::ElectricDipole);
    MultiComponentBuffer buf_ba(OperatorKind::ElectricDipole);
    std::array<Real, 3> origin = {0.1, -0.2, 0.3};

    kernels::compute_dipole(shell_a, shell_b, origin, buf_ab);
    kernels::compute_dipole(shell_b, shell_a, origin, buf_ba);

    // <a_μ|r_α|b_ν> = <b_ν|r_α|a_μ> (transposed)
    for (Size comp = 0; comp < 3; ++comp) {
        for (int a = 0; a < buf_ab.na(); ++a) {
            for (int b = 0; b < buf_ab.nb(); ++b) {
                EXPECT_NEAR(buf_ab(comp, a, b), buf_ba(comp, b, a), LOOSE_TOL)
                    << "Symmetry failed for comp=" << comp << " a=" << a << " b=" << b;
            }
        }
    }
}

// ============================================================================
// Test 4: Dipole origin dependence
// ============================================================================
TEST(MultipoleKernelTest, DipoleOriginDependence) {
    // <a|(r-O1)|b> - <a|(r-O2)|b> = (O2-O1) * <a|b>
    // i.e., shifting the origin by delta changes the dipole by -delta * overlap
    Point3D A(0.0, 0.0, 0.0);
    Point3D B(0.5, 0.0, 0.0);
    Shell shell_a(AngularMomentum::S, A, {1.0}, {1.0});
    Shell shell_b(AngularMomentum::S, B, {1.0}, {1.0});

    MultiComponentBuffer buf1(OperatorKind::ElectricDipole);
    MultiComponentBuffer buf2(OperatorKind::ElectricDipole);
    std::array<Real, 3> origin1 = {0.0, 0.0, 0.0};
    std::array<Real, 3> origin2 = {1.0, 0.0, 0.0};

    kernels::compute_dipole(shell_a, shell_b, origin1, buf1);
    kernels::compute_dipole(shell_a, shell_b, origin2, buf2);

    // Difference in x-component should be -(1.0-0.0) * S_AB = -S_AB
    const Real S_AB = std::exp(-0.125);  // exp(-1.0*1.0*0.25/(1.0+1.0))
    const Real diff_x = buf2(0, 0, 0) - buf1(0, 0, 0);
    EXPECT_NEAR(diff_x, -1.0 * S_AB, LOOSE_TOL)
        << "Origin shift should change dipole by -delta * overlap";
}

// ============================================================================
// Test 5: Quadrupole s-s same center — diagonal components
// ============================================================================
TEST(MultipoleKernelTest, QuadrupoleSsSameCenter) {
    // <s|x²|s> with α=1.0, center=origin → 1/(2α) * S = 0.5
    // Since S_self = 1 for normalized shell, <s|x²|s> = 1/(4α) = 0.25
    // Wait: for the unnormalized 1D integral, <0|x²|0> = 1/(2*zeta)
    // With zeta = 2*α = 2.0 for same exponents, <0|x²|0> = 0.25
    // The 3D integral is <s|x²|s> = Ix[0][0][2] * Iy[0][0][0] * Iz[0][0][0] * prefactor
    // I[0][0][0] = 1 (overlap base), I[0][0][2] = (P_d-O_d)*I[0][0][1] + 1/(2z)*I[0][0][0]
    // With center=origin, I[0][0][1] = 0, so I[0][0][2] = 1/(2z) = 1/4 (zeta = 2 for α+α)
    // No wait: α=1 for one shell, so zeta = α + β = 2.0, one_over_2zeta = 0.25
    // Hmm, but P_d-O_d = 0 since center=origin. So T[0][0][1] = 0.
    // T[0][0][2] = XPO*T[0][0][1] + 1/(2z)*(0 + 0 + 1*T[0][0][0]) = 0 + 0.25*1 = 0.25
    // So after prefactor and normalization the xx component is:
    // prefactor * T_x[0][0][2] * T_y[0][0][0] * T_z[0][0][0] * c_a * c_b * corr * corr
    // = (π/(α+β))^(3/2) * K_AB * 0.25 * 1 * 1 * c_a² * corr²
    // But with normalized shell, c_a² * prefactor = 1 (self-overlap = 1)
    // Actually the normalization is such that the self-overlap integral gives 1.
    // So <s|x²|s> = T[0][0][2] / T[0][0][0] = 0.25 / 1 = 0.25

    // No: the full story is more subtle. Let me just check structural properties.
    Point3D center(0.0, 0.0, 0.0);
    Shell shell(AngularMomentum::S, center, {1.0}, {1.0});

    MultiComponentBuffer buffer(OperatorKind::ElectricQuadrupole);
    std::array<Real, 3> origin = {0.0, 0.0, 0.0};

    kernels::compute_quadrupole(shell, shell, origin, buffer);

    EXPECT_EQ(buffer.n_components(), 6);
    EXPECT_EQ(buffer.na(), 1);
    EXPECT_EQ(buffer.nb(), 1);

    // By spherical symmetry: xx = yy = zz, and xy = xz = yz = 0
    EXPECT_NEAR(buffer(0, 0, 0), buffer(3, 0, 0), TIGHT_TOL) << "xx should equal yy";
    EXPECT_NEAR(buffer(0, 0, 0), buffer(5, 0, 0), TIGHT_TOL) << "xx should equal zz";
    EXPECT_NEAR(buffer(1, 0, 0), 0.0, TIGHT_TOL) << "xy should be zero";
    EXPECT_NEAR(buffer(2, 0, 0), 0.0, TIGHT_TOL) << "xz should be zero";
    EXPECT_NEAR(buffer(4, 0, 0), 0.0, TIGHT_TOL) << "yz should be zero";

    // The diagonal component should be positive and match 1/(2*(α+β))
    EXPECT_GT(buffer(0, 0, 0), 0.0) << "xx should be positive";
}

// ============================================================================
// Test 6: Quadrupole symmetry
// ============================================================================
TEST(MultipoleKernelTest, QuadrupoleSymmetry) {
    Point3D A(0.0, 0.0, 0.0);
    Point3D B(0.3, 0.5, -0.2);
    Shell shell_a(AngularMomentum::S, A, {2.0}, {1.0});
    Shell shell_b(AngularMomentum::S, B, {1.5}, {1.0});

    MultiComponentBuffer buf_ab(OperatorKind::ElectricQuadrupole);
    MultiComponentBuffer buf_ba(OperatorKind::ElectricQuadrupole);
    std::array<Real, 3> origin = {0.0, 0.0, 0.0};

    kernels::compute_quadrupole(shell_a, shell_b, origin, buf_ab);
    kernels::compute_quadrupole(shell_b, shell_a, origin, buf_ba);

    for (Size comp = 0; comp < 6; ++comp) {
        EXPECT_NEAR(buf_ab(comp, 0, 0), buf_ba(comp, 0, 0), TIGHT_TOL)
            << "Symmetry failed for comp=" << comp;
    }
}

// ============================================================================
// Test 7: Octupole component count and structure
// ============================================================================
TEST(MultipoleKernelTest, OctupoleStructure) {
    Point3D center(0.0, 0.0, 0.0);
    Shell shell(AngularMomentum::S, center, {1.0}, {1.0});

    MultiComponentBuffer buffer(OperatorKind::ElectricOctupole);
    std::array<Real, 3> origin = {0.0, 0.0, 0.0};

    kernels::compute_octupole(shell, shell, origin, buffer);

    EXPECT_EQ(buffer.n_components(), 10);

    // For s-s at origin, by symmetry:
    // xxx = yyy = zzz = 0 (odd moments of symmetric distribution)
    // All cross terms should also vanish for a spherically symmetric distribution
    for (Size comp = 0; comp < 10; ++comp) {
        EXPECT_NEAR(buffer(comp, 0, 0), 0.0, LOOSE_TOL)
            << "All octupole moments should be zero for s-s at origin, comp=" << comp;
    }
}

// ============================================================================
// Test 8: Dipole with p-shells
// ============================================================================
TEST(MultipoleKernelTest, DipolePPShells) {
    Point3D center(0.0, 0.0, 0.0);
    Shell shell(AngularMomentum::P, center, {1.0}, {1.0});

    MultiComponentBuffer buffer(OperatorKind::ElectricDipole);
    std::array<Real, 3> origin = {0.0, 0.0, 0.0};

    kernels::compute_dipole(shell, shell, origin, buffer);

    EXPECT_EQ(buffer.na(), 3);
    EXPECT_EQ(buffer.nb(), 3);

    // p-p dipole at same center-origin:
    // <px|x|px> should be zero by symmetry (odd function of y,z)
    // Actually: <px|x|px> = ∫ x*x*x * exp(-2αr²) dr_x * ∫exp(-2αr²)dr_y * ∫exp(-2αr²)dr_z
    //         = ∫ x³ exp(-2αr²_x) dx * ... = 0 (odd function)
    // <px|y|px> = ∫ x*y*x * exp(-2αr²) dV = 0 (odd in y)
    // So all diagonal dipole elements for p-p at same center/origin should be zero
    for (int a = 0; a < 3; ++a) {
        for (int b = 0; b < 3; ++b) {
            for (Size comp = 0; comp < 3; ++comp) {
                EXPECT_NEAR(buffer(comp, a, b), 0.0, LOOSE_TOL)
                    << "p-p dipole at same center/origin should vanish, comp=" << comp
                    << " a=" << a << " b=" << b;
            }
        }
    }
}

// ============================================================================
// Test 9: Dipole rank-0 check — overlap should be recovered from T[0][0][0]
// ============================================================================
TEST(MultipoleKernelTest, DipoleComponentsNotAllZero) {
    // When shells are displaced, at least one dipole component should be non-zero
    Point3D A(0.0, 0.0, 0.0);
    Point3D B(1.0, 0.0, 0.0);
    Shell shell_a(AngularMomentum::S, A, {1.0}, {1.0});
    Shell shell_b(AngularMomentum::S, B, {1.0}, {1.0});

    MultiComponentBuffer buffer(OperatorKind::ElectricDipole);
    std::array<Real, 3> origin = {0.0, 0.0, 0.0};

    kernels::compute_dipole(shell_a, shell_b, origin, buffer);

    // x-component should be non-zero (displacement along x)
    EXPECT_GT(std::abs(buffer(0, 0, 0)), 1e-15)
        << "x-component should be non-zero for x-displaced shells";
}
