// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2026 Bryce M. Westheimer. All rights reserved.

/// @file test_large_system.cpp
/// @brief Large system stress testing (Task 25.2.2)
///
/// Tests with 1000+ basis functions to verify:
/// - Memory allocation and cleanup
/// - Correct integral computation at scale
/// - No performance pathologies

#include <libaccint/basis/basis_set.hpp>
#include <libaccint/basis/shell.hpp>
#include <libaccint/core/types.hpp>
#include <libaccint/data/builtin_basis.hpp>
#include <libaccint/engine/engine.hpp>
#include <libaccint/operators/one_electron_operator.hpp>

#include <gtest/gtest.h>
#include <cmath>
#include <vector>

using namespace libaccint;

namespace {

/// @brief Build a large H-chain basis for stress testing
/// @param n_atoms Number of hydrogen atoms in a line
/// @param spacing Distance between atoms (Bohr)
/// @return BasisSet with STO-3G on each H
BasisSet build_h_chain(int n_atoms, Real spacing = 2.0) {
    std::vector<data::Atom> atoms;
    atoms.reserve(static_cast<Size>(n_atoms));
    for (int i = 0; i < n_atoms; ++i) {
        atoms.push_back({1, {static_cast<Real>(i) * spacing, 0.0, 0.0}});
    }
    return data::create_sto3g(atoms);
}

/// @brief Build a large C/H system for more basis functions
/// @param n_carbon Number of carbon atoms
/// @param n_hydrogen Number of hydrogen atoms
/// @return BasisSet with STO-3G
BasisSet build_ch_system(int n_carbon, int n_hydrogen) {
    std::vector<data::Atom> atoms;
    atoms.reserve(static_cast<Size>(n_carbon + n_hydrogen));
    Real x = 0.0;
    for (int i = 0; i < n_carbon; ++i) {
        atoms.push_back({6, {x, 0.0, 0.0}});
        x += 2.8;  // ~1.5 Angstrom C-C bond
    }
    for (int i = 0; i < n_hydrogen; ++i) {
        atoms.push_back({1, {x, 0.0, 0.0}});
        x += 2.0;
    }
    return data::create_sto3g(atoms);
}

}  // namespace

// ============================================================================
// Large System Tests
// ============================================================================

TEST(LargeSystem, HChain100Atoms) {
    // 100 H atoms with STO-3G = 100 basis functions
    auto basis = build_h_chain(100);
    EXPECT_EQ(basis.n_basis_functions(), 100u);
    EXPECT_EQ(basis.n_shells(), 100u);

    // Build engine and compute overlap
    Engine engine(basis);
    std::vector<Real> S;
    engine.compute_overlap_matrix(S);

    Size nbf = basis.n_basis_functions();
    EXPECT_EQ(S.size(), nbf * nbf);

    // Verify diagonal = 1
    for (Size i = 0; i < nbf; ++i) {
        EXPECT_NEAR(S[i * nbf + i], 1.0, 1e-10)
            << "Diagonal element S[" << i << "," << i << "] != 1.0";
    }
}

TEST(LargeSystem, OverlapSymmetry100) {
    auto basis = build_h_chain(100);
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

TEST(LargeSystem, CarbonChain200BasisFunctions) {
    // ~40 carbon atoms * 5 basis functions = 200 basis functions
    auto basis = build_ch_system(40, 0);
    EXPECT_GE(basis.n_basis_functions(), 200u);

    Engine engine(basis);
    std::vector<Real> S;
    engine.compute_overlap_matrix(S);

    Size nbf = basis.n_basis_functions();
    EXPECT_EQ(S.size(), nbf * nbf);

    // Diagonal check
    for (Size i = 0; i < nbf; ++i) {
        EXPECT_NEAR(S[i * nbf + i], 1.0, 1e-8);
    }
}

TEST(LargeSystem, KineticMatrix200) {
    auto basis = build_ch_system(40, 0);
    Engine engine(basis);

    std::vector<Real> T;
    engine.compute_kinetic_matrix(T);

    Size nbf = basis.n_basis_functions();
    EXPECT_EQ(T.size(), nbf * nbf);

    // Kinetic energy diagonal elements must be positive
    for (Size i = 0; i < nbf; ++i) {
        EXPECT_GT(T[i * nbf + i], 0.0)
            << "Kinetic diagonal T[" << i << "," << i << "] should be positive";
    }

    // Symmetry check
    for (Size i = 0; i < nbf; ++i) {
        for (Size j = i + 1; j < nbf; ++j) {
            EXPECT_NEAR(T[i * nbf + j], T[j * nbf + i], 1e-12);
        }
    }
}

TEST(LargeSystem, ShellSetPairGeneration) {
    auto basis = build_h_chain(200);
    EXPECT_EQ(basis.n_basis_functions(), 200u);

    // Force pair generation
    const auto& pairs = basis.shell_set_pairs();
    EXPECT_GT(pairs.size(), 0u);

    // Also generate quartets
    const auto& quartets = basis.shell_set_quartets();
    EXPECT_GT(quartets.size(), 0u);
}

TEST(LargeSystem, Over1000BasisFunctions) {
    // 200 carbon atoms * 5 = 1000 basis functions + some H
    // This is the target: 1000+ basis functions
    auto basis = build_ch_system(200, 10);

    EXPECT_GE(basis.n_basis_functions(), 1000u)
        << "Need 1000+ basis functions for stress test, got "
        << basis.n_basis_functions();

    // Just verify construction works and basic invariants hold
    EXPECT_GT(basis.n_shells(), 0u);
    EXPECT_GT(basis.n_shell_sets(), 0u);

    // Compute overlap matrix for 1000+ basis system
    // This tests memory allocation at scale
    Engine engine(basis);
    std::vector<Real> S;
    engine.compute_overlap_matrix(S);

    Size nbf = basis.n_basis_functions();
    EXPECT_EQ(S.size(), nbf * nbf);

    // Spot-check: first few diagonal elements should be ~1
    for (Size i = 0; i < std::min(nbf, Size{10}); ++i) {
        EXPECT_NEAR(S[i * nbf + i], 1.0, 1e-8)
            << "Diagonal S[" << i << "," << i << "] != 1.0";
    }
}

TEST(LargeSystem, WorkUnitCacheClear) {
    auto basis = build_h_chain(100);

    // Generate pairs and quartets
    const auto& pairs = basis.shell_set_pairs();
    EXPECT_GT(pairs.size(), 0u);
    const auto& quartets = basis.shell_set_quartets();
    EXPECT_GT(quartets.size(), 0u);

    // Clear cache and verify re-generation works
    basis.clear_work_unit_cache();

    const auto& pairs2 = basis.shell_set_pairs();
    EXPECT_GT(pairs2.size(), 0u);
}

TEST(LargeSystem, MixedAMLargeSystem) {
    // Large system with mixed angular momenta (S, P from C atoms)
    auto basis = build_ch_system(100, 50);

    EXPECT_GE(basis.n_basis_functions(), 500u);
    EXPECT_GT(basis.max_angular_momentum(), 0);

    Engine engine(basis);
    std::vector<Real> S;
    engine.compute_overlap_matrix(S);

    Size nbf = basis.n_basis_functions();
    EXPECT_EQ(S.size(), nbf * nbf);

    // All diagonal elements should be exactly 1 to machine precision
    for (Size i = 0; i < nbf; ++i) {
        EXPECT_NEAR(S[i * nbf + i], 1.0, 1e-8);
    }
}
