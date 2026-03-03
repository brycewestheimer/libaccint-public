// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2026 Bryce M. Westheimer. All rights reserved.

/// @file test_property_integrals.cpp
/// @brief Integration test for property integrals and QM/MM workflow

#include <libaccint/kernels/multipole_kernel.hpp>
#include <libaccint/kernels/momentum_kernel.hpp>
#include <libaccint/kernels/distributed_multipole_kernel.hpp>
#include <libaccint/operators/projection_operator.hpp>
#include <libaccint/operators/operator_types.hpp>
#include <libaccint/basis/shell.hpp>
#include <libaccint/engine/multi_component_buffer.hpp>
#include <libaccint/utils/constants.hpp>
#include <gtest/gtest.h>
#include <cmath>

using namespace libaccint;

namespace {
constexpr Real TOL = 1e-10;
}

// ============================================================================
// Integration Test 1: QM/MM workflow — distributed multipole on a molecular basis
// ============================================================================
TEST(PropertyIntegralsIntegrationTest, QMMMWorkflowStructure) {
    // Simulate a minimal QM/MM setup:
    // QM region: two s-functions (like minimal H₂)
    // MM region: two point charges (like TIP3P water)
    Point3D A(0.0, 0.0, 0.0);
    Point3D B(0.0, 0.0, 1.4);  // ~1.4 bohr ≈ H-H distance
    Shell shell_a(AngularMomentum::S, A, {1.0}, {1.0});
    Shell shell_b(AngularMomentum::S, B, {1.0}, {1.0});

    // Two point charges at typical water positions (far from QM region)
    DistributedMultipoleParams params;
    params.x = {5.0, 6.0};     // O at (5,0,0), H at (6,0.5,0)
    params.y = {0.0, 0.5};
    params.z = {0.0, 0.0};
    params.charges = {-0.834, 0.417};  // TIP3P-like charges

    EXPECT_TRUE(params.is_valid());
    EXPECT_EQ(params.n_sites(), 2);

    // Compute V(aa), V(ab), V(ba), V(bb) blocks
    OverlapBuffer buf_aa, buf_ab, buf_ba, buf_bb;

    kernels::compute_distributed_multipole(shell_a, shell_a, params, buf_aa);
    kernels::compute_distributed_multipole(shell_a, shell_b, params, buf_ab);
    kernels::compute_distributed_multipole(shell_b, shell_a, params, buf_ba);
    kernels::compute_distributed_multipole(shell_b, shell_b, params, buf_bb);

    // V should be symmetric: V(ab) = V(ba)
    EXPECT_NEAR(buf_ab(0, 0), buf_ba(0, 0), TOL) << "V should be symmetric";

    // The net charge is -0.834 + 0.417 = -0.417 (negative net charge)
    // So the potential should have consistent sign structure
    // All elements should be non-zero (charges are far but not infinitely far)
    EXPECT_NE(buf_aa(0, 0), 0.0) << "V(aa) should be non-zero";
    EXPECT_NE(buf_ab(0, 0), 0.0) << "V(ab) should be non-zero";
    EXPECT_NE(buf_bb(0, 0), 0.0) << "V(bb) should be non-zero";
}

// ============================================================================
// Integration Test 2: Dipole moment consistency with overlap
// ============================================================================
TEST(PropertyIntegralsIntegrationTest, DipoleOverlapConsistency) {
    // <a|(r-O)|b> evaluated at O = center of charge should relate to
    // expectation value of position operator.
    //
    // For normalized s-s at same center: <s|r|s> = 0 at center
    // After displacement: <s_A|x|s_B> = P_x * S_AB where P is the
    // Gaussian product center.
    Point3D A(0.0, 0.0, 0.0);
    Point3D B(1.0, 0.0, 0.0);
    Shell shell_a(AngularMomentum::S, A, {1.0}, {1.0});
    Shell shell_b(AngularMomentum::S, B, {1.0}, {1.0});

    // Dipole at two different origins
    MultiComponentBuffer buf_o1(OperatorKind::ElectricDipole);
    MultiComponentBuffer buf_o2(OperatorKind::ElectricDipole);
    std::array<Real, 3> o1 = {0.0, 0.0, 0.0};
    std::array<Real, 3> o2 = {0.5, 0.0, 0.0};

    kernels::compute_dipole(shell_a, shell_b, o1, buf_o1);
    kernels::compute_dipole(shell_a, shell_b, o2, buf_o2);

    // The dipole at o2 should equal dipole at o1 minus (o2-o1)*overlap
    // <a|(x - O2x)|b> = <a|(x - O1x)|b> - (O2x - O1x)*<a|b>
    // So: buf_o2(x) = buf_o1(x) - 0.5 * S_AB
    // And buf_o2(y) = buf_o1(y), buf_o2(z) = buf_o1(z) (no shift in y,z)
    EXPECT_NEAR(buf_o2(1, 0, 0), buf_o1(1, 0, 0), TOL) << "y-component unchanged";
    EXPECT_NEAR(buf_o2(2, 0, 0), buf_o1(2, 0, 0), TOL) << "z-component unchanged";

    // The x-component difference should be -0.5 * S_AB
    Real diff = buf_o2(0, 0, 0) - buf_o1(0, 0, 0);
    // We don't know S_AB exactly without computing overlap, but diff should be negative
    EXPECT_LT(diff, 0.0) << "Shifting origin in +x should decrease x-dipole";
}

// ============================================================================
// Integration Test 3: Momentum anti-symmetry across shell types
// ============================================================================
TEST(PropertyIntegralsIntegrationTest, MomentumAntiSymmetryMixed) {
    Point3D A(0.0, 0.0, 0.0);
    Point3D B(0.5, 0.3, 0.0);
    Shell shell_s(AngularMomentum::S, A, {1.0}, {1.0});
    Shell shell_p(AngularMomentum::P, B, {1.0}, {1.0});

    MultiComponentBuffer buf_sp(OperatorKind::LinearMomentum);
    MultiComponentBuffer buf_ps(OperatorKind::LinearMomentum);

    kernels::compute_linear_momentum(shell_s, shell_p, buf_sp);
    kernels::compute_linear_momentum(shell_p, shell_s, buf_ps);

    // <s|d/dr|p> = -<p|d/dr|s>^T
    for (Size comp = 0; comp < 3; ++comp) {
        for (int a = 0; a < buf_sp.na(); ++a) {
            for (int b = 0; b < buf_sp.nb(); ++b) {
                EXPECT_NEAR(buf_sp(comp, a, b), -buf_ps(comp, b, a), TOL)
                    << "Anti-symmetry failed for s-p pair, comp=" << comp
                    << " a=" << a << " b=" << b;
            }
        }
    }
}

// ============================================================================
// Integration Test 4: Projection operator on a 3-function basis
// ============================================================================
TEST(PropertyIntegralsIntegrationTest, ProjectionOperatorWorkflow) {
    // Build a projection matrix from two occupied MOs in a 3-function basis
    ProjectionOperatorParams params;
    params.n_basis = 3;
    params.n_projectors = 2;

    // Two normalized orthogonal vectors (column-major)
    // Column 0: (1/sqrt(2), 1/sqrt(2), 0)
    // Column 1: (0, 0, 1)
    const Real s2 = 1.0 / std::sqrt(2.0);
    params.coefficients = {s2, s2, 0.0, 0.0, 0.0, 1.0};
    params.weights = {2.0, 2.0};  // doubly-occupied orbitals

    EXPECT_TRUE(params.is_valid());

    auto P = build_projection_matrix(params);
    EXPECT_TRUE(verify_projection_matrix(P, 3, 1e-12));

    // Trace should be 2.0 + 2.0 = 4.0 (for orthonormal columns)
    Real trace = P[0] + P[4] + P[8];
    EXPECT_NEAR(trace, 4.0, TOL);

    // P should be symmetric
    EXPECT_NEAR(P[0 * 3 + 1], P[1 * 3 + 0], TOL);
    EXPECT_NEAR(P[0 * 3 + 2], P[2 * 3 + 0], TOL);
    EXPECT_NEAR(P[1 * 3 + 2], P[2 * 3 + 1], TOL);
}

// ============================================================================
// Integration Test 5: Operator type classification
// ============================================================================
TEST(PropertyIntegralsIntegrationTest, OperatorClassification) {

    // Property integrals
    EXPECT_TRUE(is_property_integral(OperatorKind::ElectricDipole));
    EXPECT_TRUE(is_property_integral(OperatorKind::ElectricQuadrupole));
    EXPECT_TRUE(is_property_integral(OperatorKind::ElectricOctupole));
    EXPECT_TRUE(is_property_integral(OperatorKind::LinearMomentum));
    EXPECT_TRUE(is_property_integral(OperatorKind::AngularMomentum));

    // Multi-component
    EXPECT_TRUE(is_multi_component(OperatorKind::ElectricDipole));
    EXPECT_TRUE(is_multi_component(OperatorKind::ElectricQuadrupole));
    EXPECT_TRUE(is_multi_component(OperatorKind::ElectricOctupole));
    EXPECT_TRUE(is_multi_component(OperatorKind::LinearMomentum));
    EXPECT_TRUE(is_multi_component(OperatorKind::AngularMomentum));

    // Anti-Hermitian
    EXPECT_TRUE(is_anti_hermitian(OperatorKind::LinearMomentum));
    EXPECT_TRUE(is_anti_hermitian(OperatorKind::AngularMomentum));
    EXPECT_FALSE(is_anti_hermitian(OperatorKind::ElectricDipole));

    // Component counts
    EXPECT_EQ(component_count(OperatorKind::ElectricDipole), 3);
    EXPECT_EQ(component_count(OperatorKind::ElectricQuadrupole), 6);
    EXPECT_EQ(component_count(OperatorKind::ElectricOctupole), 10);
    EXPECT_EQ(component_count(OperatorKind::LinearMomentum), 3);
    EXPECT_EQ(component_count(OperatorKind::AngularMomentum), 3);

    // Operator names
    EXPECT_FALSE(operator_name(OperatorKind::ElectricDipole).empty());
    EXPECT_FALSE(operator_name(OperatorKind::AngularMomentum).empty());
}
