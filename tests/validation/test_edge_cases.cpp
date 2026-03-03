// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2026 Bryce M. Westheimer. All rights reserved.

/// @file test_edge_cases.cpp
/// @brief Tests for edge case handling (Tasks 25.2.1, 25.2.3)
///
/// Validates behavior with:
/// - Empty/degenerate inputs
/// - Numerical edge cases (very diffuse/tight functions, near-linear deps)
/// - Boundary conditions

#include <libaccint/basis/basis_set.hpp>
#include <libaccint/basis/shell.hpp>
#include <libaccint/core/types.hpp>
#include <libaccint/data/builtin_basis.hpp>
#include <libaccint/engine/engine.hpp>
#include <libaccint/operators/one_electron_operator.hpp>
#include <libaccint/utils/error_handling.hpp>
#include <libaccint/utils/input_validation.hpp>

#include <gtest/gtest.h>
#include <cmath>
#include <limits>
#include <vector>

using namespace libaccint;

// ============================================================================
// Empty/Degenerate Input Tests (Task 25.2.1)
// ============================================================================

TEST(EdgeCases, EmptyBasisSetValidation) {
    BasisSet empty;
    EXPECT_EQ(empty.n_shells(), 0u);
    EXPECT_EQ(empty.n_basis_functions(), 0u);

    auto result = validation::validate_basis_set(empty);
    EXPECT_FALSE(result);
}

TEST(EdgeCases, SingleShellBasis) {
    // Minimal basis: single s-function on hydrogen
    Shell s(0, {0.0, 0.0, 0.0}, {1.0}, {1.0});
    std::vector<Shell> shells = {s};
    BasisSet basis(std::move(shells));

    EXPECT_EQ(basis.n_shells(), 1u);
    EXPECT_EQ(basis.n_basis_functions(), 1u);
    EXPECT_EQ(basis.max_angular_momentum(), 0);
}

TEST(EdgeCases, SingleAtomBasis) {
    std::vector<data::Atom> atoms = {{1, {0.0, 0.0, 0.0}}};
    auto basis = data::create_sto3g(atoms);

    EXPECT_GE(basis.n_shells(), 1u);
    EXPECT_GE(basis.n_basis_functions(), 1u);

    Engine engine(basis);
    std::vector<Real> S;
    engine.compute_overlap_matrix(S);

    // Single H atom: overlap matrix should have diagonal ~1.0
    Size nbf = basis.n_basis_functions();
    EXPECT_EQ(S.size(), nbf * nbf);
    for (Size i = 0; i < nbf; ++i) {
        EXPECT_NEAR(S[i * nbf + i], 1.0, 1e-10);
    }
}

TEST(EdgeCases, CoincidentAtoms) {
    // Two hydrogen atoms at the same position
    std::vector<data::Atom> atoms = {
        {1, {0.0, 0.0, 0.0}},
        {1, {0.0, 0.0, 0.0}}
    };
    auto basis = data::create_sto3g(atoms);
    Engine engine(basis);

    std::vector<Real> S;
    engine.compute_overlap_matrix(S);

    // Should compute without error, overlap between coincident shells = 1
    Size nbf = basis.n_basis_functions();
    EXPECT_EQ(S.size(), nbf * nbf);
    // Both diagonals should be 1.0
    for (Size i = 0; i < nbf; ++i) {
        EXPECT_NEAR(S[i * nbf + i], 1.0, 1e-10);
    }
}

TEST(EdgeCases, VeryDistantAtoms) {
    // Atoms very far apart — overlap should be ~zero
    std::vector<data::Atom> atoms = {
        {1, {0.0, 0.0, 0.0}},
        {1, {1000.0, 0.0, 0.0}}
    };
    auto basis = data::create_sto3g(atoms);
    Engine engine(basis);

    std::vector<Real> S;
    engine.compute_overlap_matrix(S);

    Size nbf = basis.n_basis_functions();
    // Off-diagonal blocks (between atoms) should be ~0
    EXPECT_NEAR(S[0 * nbf + 1], 0.0, 1e-10);
    EXPECT_NEAR(S[1 * nbf + 0], 0.0, 1e-10);
}

// ============================================================================
// Numerical Edge Cases (Task 25.2.3)
// ============================================================================

TEST(EdgeCases, VeryDiffuseFunction) {
    // Very diffuse exponent (small value)
    Shell s(0, {0.0, 0.0, 0.0}, {0.001}, {1.0});
    std::vector<Shell> shells = {s};
    BasisSet basis(std::move(shells));

    Engine engine(basis);
    std::vector<Real> S;
    engine.compute_overlap_matrix(S);

    EXPECT_EQ(S.size(), 1u);
    EXPECT_NEAR(S[0], 1.0, 1e-8);  // Self-overlap should be 1
}

TEST(EdgeCases, VeryTightFunction) {
    // Very tight exponent (large value)
    Shell s(0, {0.0, 0.0, 0.0}, {100000.0}, {1.0});
    std::vector<Shell> shells = {s};
    BasisSet basis(std::move(shells));

    Engine engine(basis);
    std::vector<Real> S;
    engine.compute_overlap_matrix(S);

    EXPECT_EQ(S.size(), 1u);
    EXPECT_NEAR(S[0], 1.0, 1e-8);
}

TEST(EdgeCases, MixedDiffuseTight) {
    // Mix of very diffuse and tight functions
    Shell s1(0, {0.0, 0.0, 0.0}, {0.01}, {1.0});
    Shell s2(0, {0.0, 0.0, 0.0}, {10000.0}, {1.0});
    std::vector<Shell> shells = {s1, s2};
    BasisSet basis(std::move(shells));

    Engine engine(basis);
    std::vector<Real> S;
    engine.compute_overlap_matrix(S);

    Size nbf = basis.n_basis_functions();
    EXPECT_EQ(nbf, 2u);
    // Diagonal should be ~1
    EXPECT_NEAR(S[0], 1.0, 1e-8);
    EXPECT_NEAR(S[3], 1.0, 1e-8);
    // Off-diagonal should be small (very different exponents)
    EXPECT_LT(std::abs(S[1]), 0.1);
}

TEST(EdgeCases, NearLinearDependence) {
    // Two very similar shells — near-linear dependence
    Shell s1(0, {0.0, 0.0, 0.0}, {1.0}, {1.0});
    Shell s2(0, {0.0, 0.0, 0.0}, {1.001}, {1.0});
    std::vector<Shell> shells = {s1, s2};
    BasisSet basis(std::move(shells));

    Engine engine(basis);
    std::vector<Real> S;
    engine.compute_overlap_matrix(S);

    Size nbf = basis.n_basis_functions();
    EXPECT_EQ(nbf, 2u);
    // Off-diagonal overlap should be very close to 1 (near-linear dep)
    EXPECT_GT(std::abs(S[1]), 0.99);
}

TEST(EdgeCases, HighAngularMomentumSelfOverlap) {
    // D-shell self-overlap
    Shell s(2, {0.0, 0.0, 0.0}, {1.0}, {1.0});
    std::vector<Shell> shells = {s};
    BasisSet basis(std::move(shells));

    Engine engine(basis);
    std::vector<Real> S;
    engine.compute_overlap_matrix(S);

    Size nbf = basis.n_basis_functions();
    EXPECT_EQ(nbf, static_cast<Size>(n_cartesian(2)));
    // Diagonal elements should be 1
    for (Size i = 0; i < nbf; ++i) {
        EXPECT_NEAR(S[i * nbf + i], 1.0, 1e-8);
    }
}

TEST(EdgeCases, KineticEnergyFinite) {
    // Ensure kinetic energy integrals are finite for diffuse functions
    std::vector<data::Atom> atoms = {{1, {0.0, 0.0, 0.0}}};
    auto basis = data::create_sto3g(atoms);
    Engine engine(basis);

    std::vector<Real> T;
    engine.compute_kinetic_matrix(T);

    for (Size i = 0; i < T.size(); ++i) {
        EXPECT_TRUE(std::isfinite(T[i]))
            << "Non-finite kinetic integral at index " << i;
    }
}

TEST(EdgeCases, OverlapMatrixSymmetry) {
    // Multi-atom system — verify symmetry
    std::vector<data::Atom> atoms = {
        {1, {0.0, 0.0, 0.0}},
        {6, {2.0, 0.0, 0.0}},
        {8, {0.0, 2.0, 0.0}}
    };
    auto basis = data::create_sto3g(atoms);
    Engine engine(basis);

    std::vector<Real> S;
    engine.compute_overlap_matrix(S);

    Size nbf = basis.n_basis_functions();
    for (Size i = 0; i < nbf; ++i) {
        for (Size j = i + 1; j < nbf; ++j) {
            EXPECT_NEAR(S[i * nbf + j], S[j * nbf + i], 1e-14)
                << "Asymmetry at (" << i << "," << j << ")";
        }
    }
}

TEST(EdgeCases, AllMaxAngularMomentumSShells) {
    // Create basis with multiple s-shells (AM=0)
    std::vector<Shell> shells;
    for (int i = 0; i < 5; ++i) {
        shells.emplace_back(0, Point3D{static_cast<Real>(i), 0.0, 0.0},
                            std::vector<Real>{1.0 + 0.1 * i},
                            std::vector<Real>{1.0});
    }
    BasisSet basis(std::move(shells));
    EXPECT_EQ(basis.max_angular_momentum(), 0);
    EXPECT_EQ(basis.n_basis_functions(), 5u);
}

TEST(EdgeCases, ShellOnOrigin) {
    // Shell exactly at origin
    Shell s(0, {0.0, 0.0, 0.0}, {1.0}, {1.0});
    EXPECT_NEAR(s.center().x, 0.0, 1e-15);
    EXPECT_NEAR(s.center().y, 0.0, 1e-15);
    EXPECT_NEAR(s.center().z, 0.0, 1e-15);
}

TEST(EdgeCases, ShellWithManyCenters) {
    // Many shells to test memory handling
    std::vector<Shell> shells;
    for (int i = 0; i < 100; ++i) {
        shells.emplace_back(
            i % 3, // S, P, D
            Point3D{static_cast<Real>(i) * 0.1, 0.0, 0.0},
            std::vector<Real>{1.0},
            std::vector<Real>{1.0});
    }
    BasisSet basis(std::move(shells));
    EXPECT_EQ(basis.n_shells(), 100u);
}
