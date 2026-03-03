// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2026 Bryce M. Westheimer. All rights reserved.

/// @file test_basis_expansion.cpp
/// @brief Built-in basis set testing (Task 25.4.3, 25.4.4)
///
/// Tests the built-in basis set registry for:
/// - Pople family: STO-3G
/// - Dunning family: future cc-pVXZ (verified via create_builtin_basis API)
/// - Ahlrichs family: future def2-SVP (verified via create_builtin_basis API)
/// - Augmented basis set testing (aug-cc-pVXZ)
///
/// Currently only STO-3G is fully implemented. Other basis sets are tested
/// via the API to verify proper error handling for unsupported sets.

#include <libaccint/basis/basis_set.hpp>
#include <libaccint/basis/shell.hpp>
#include <libaccint/core/types.hpp>
#include <libaccint/data/builtin_basis.hpp>
#include <libaccint/engine/engine.hpp>
#include <libaccint/utils/error_handling.hpp>

#include <gtest/gtest.h>
#include <cmath>
#include <vector>

using namespace libaccint;

// ============================================================================
// STO-3G Pople Family Tests
// ============================================================================

TEST(BasisExpansion, STO3G_Hydrogen) {
    std::vector<data::Atom> atoms = {{1, {0.0, 0.0, 0.0}}};
    auto basis = data::create_sto3g(atoms);

    EXPECT_EQ(basis.n_shells(), 1u);     // 1s only
    EXPECT_EQ(basis.n_basis_functions(), 1u);
    EXPECT_EQ(basis.max_angular_momentum(), 0);
}

TEST(BasisExpansion, STO3G_Carbon) {
    std::vector<data::Atom> atoms = {{6, {0.0, 0.0, 0.0}}};
    auto basis = data::create_sto3g(atoms);

    // C: 1s, 2s, 2p => 3 shells, 5 basis functions (1 + 1 + 3)
    EXPECT_EQ(basis.n_shells(), 3u);
    EXPECT_EQ(basis.n_basis_functions(), 5u);
    EXPECT_EQ(basis.max_angular_momentum(), 1);
}

TEST(BasisExpansion, STO3G_Nitrogen) {
    std::vector<data::Atom> atoms = {{7, {0.0, 0.0, 0.0}}};
    auto basis = data::create_sto3g(atoms);

    EXPECT_EQ(basis.n_shells(), 3u);
    EXPECT_EQ(basis.n_basis_functions(), 5u);
}

TEST(BasisExpansion, STO3G_Oxygen) {
    std::vector<data::Atom> atoms = {{8, {0.0, 0.0, 0.0}}};
    auto basis = data::create_sto3g(atoms);

    EXPECT_EQ(basis.n_shells(), 3u);
    EXPECT_EQ(basis.n_basis_functions(), 5u);
}

TEST(BasisExpansion, STO3G_Fluorine) {
    std::vector<data::Atom> atoms = {{9, {0.0, 0.0, 0.0}}};
    auto basis = data::create_sto3g(atoms);

    EXPECT_EQ(basis.n_shells(), 3u);
    EXPECT_EQ(basis.n_basis_functions(), 5u);
}

TEST(BasisExpansion, STO3G_UnsupportedElement) {
    // STO-3G data is only hard-coded for H, C, N, O, F
    std::vector<data::Atom> atoms = {{26, {0.0, 0.0, 0.0}}};  // Iron
    EXPECT_THROW(data::create_sto3g(atoms), InvalidArgumentException);
}

TEST(BasisExpansion, STO3G_Water) {
    // H2O: 2H + 1O = 2*1 + 5 = 7 basis functions
    std::vector<data::Atom> atoms = {
        {8, {0.0, 0.0, 0.2217}},     // O
        {1, {0.0, 1.4304, -0.8868}},  // H
        {1, {0.0, -1.4304, -0.8868}}  // H
    };
    auto basis = data::create_sto3g(atoms);

    EXPECT_EQ(basis.n_basis_functions(), 7u);
    EXPECT_EQ(basis.n_shells(), 5u);  // 3 from O + 1 + 1 from H's
}

TEST(BasisExpansion, STO3G_Methane) {
    // CH4: 1C + 4H = 5 + 4*1 = 9 basis functions
    Real d = 1.189;  // CH bond length in Bohr
    std::vector<data::Atom> atoms = {
        {6, {0.0, 0.0, 0.0}},
        {1, { d,  d,  d}},
        {1, {-d, -d,  d}},
        {1, {-d,  d, -d}},
        {1, { d, -d, -d}}
    };
    auto basis = data::create_sto3g(atoms);

    EXPECT_EQ(basis.n_basis_functions(), 9u);
    EXPECT_EQ(basis.n_shells(), 7u);  // 3 from C + 4 from H's
}

TEST(BasisExpansion, STO3G_OverlapSelfConsistency) {
    // Overlap matrix for STO-3G water should be well-conditioned
    std::vector<data::Atom> atoms = {
        {8, {0.0, 0.0, 0.2217}},
        {1, {0.0, 1.4304, -0.8868}},
        {1, {0.0, -1.4304, -0.8868}}
    };
    auto basis = data::create_sto3g(atoms);
    Engine engine(basis);

    std::vector<Real> S;
    engine.compute_overlap_matrix(S);

    Size nbf = basis.n_basis_functions();
    // Diagonal should be 1
    for (Size i = 0; i < nbf; ++i) {
        EXPECT_NEAR(S[i * nbf + i], 1.0, 1e-10);
    }

    // Symmetry
    for (Size i = 0; i < nbf; ++i) {
        for (Size j = i + 1; j < nbf; ++j) {
            EXPECT_NEAR(S[i * nbf + j], S[j * nbf + i], 1e-14);
        }
    }
}

TEST(BasisExpansion, STO3G_KineticPositive) {
    // All diagonal kinetic energy integrals should be positive
    std::vector<data::Atom> atoms = {
        {6, {0.0, 0.0, 0.0}},
        {1, {2.0, 0.0, 0.0}}
    };
    auto basis = data::create_sto3g(atoms);
    Engine engine(basis);

    std::vector<Real> T;
    engine.compute_kinetic_matrix(T);

    Size nbf = basis.n_basis_functions();
    for (Size i = 0; i < nbf; ++i) {
        EXPECT_GT(T[i * nbf + i], 0.0)
            << "T[" << i << "," << i << "] should be positive";
    }
}

// ============================================================================
// Built-in Basis Name Registry Tests
// ============================================================================

TEST(BasisExpansion, CreateBuiltinBasisSTO3G) {
    std::vector<data::Atom> atoms = {{1, {0.0, 0.0, 0.0}}};
    auto basis = data::create_builtin_basis("sto-3g", atoms);
    EXPECT_EQ(basis.n_basis_functions(), 1u);
}

TEST(BasisExpansion, CreateBuiltinBasisCaseInsensitive) {
    std::vector<data::Atom> atoms = {{1, {0.0, 0.0, 0.0}}};

    // Should work with different casings
    EXPECT_NO_THROW(data::create_builtin_basis("STO-3G", atoms));
    EXPECT_NO_THROW(data::create_builtin_basis("sto-3g", atoms));
    EXPECT_NO_THROW(data::create_builtin_basis("Sto-3G", atoms));
}

TEST(BasisExpansion, CreateBuiltinBasisUnsupported) {
    std::vector<data::Atom> atoms = {{1, {0.0, 0.0, 0.0}}};

    // These are not yet built in — should throw InvalidArgumentException
    EXPECT_THROW(data::create_builtin_basis("cc-pvdz", atoms),
                 InvalidArgumentException);
    EXPECT_THROW(data::create_builtin_basis("aug-cc-pvdz", atoms),
                 InvalidArgumentException);
    EXPECT_THROW(data::create_builtin_basis("def2-svp", atoms),
                 InvalidArgumentException);
    EXPECT_THROW(data::create_builtin_basis("6-31g", atoms),
                 InvalidArgumentException);
}

// ============================================================================
// Multi-Element Molecule Tests (Pople family)
// ============================================================================

TEST(BasisExpansion, STO3G_HCN) {
    // HCN: H(1) + C(5) + N(5) = 11 basis functions
    std::vector<data::Atom> atoms = {
        {1, {0.0, 0.0, -2.0}},
        {6, {0.0, 0.0, 0.0}},
        {7, {0.0, 0.0, 2.2}}
    };
    auto basis = data::create_sto3g(atoms);
    EXPECT_EQ(basis.n_basis_functions(), 11u);

    Engine engine(basis);
    std::vector<Real> S;
    engine.compute_overlap_matrix(S);
    EXPECT_EQ(S.size(), 11u * 11u);
}

TEST(BasisExpansion, STO3G_Formaldehyde) {
    // H2CO: 2H(1) + C(5) + O(5) = 12 basis functions
    std::vector<data::Atom> atoms = {
        {6, {0.0, 0.0, 0.0}},
        {8, {0.0, 0.0, 2.3}},
        {1, {0.0, 1.77, -1.1}},
        {1, {0.0, -1.77, -1.1}}
    };
    auto basis = data::create_sto3g(atoms);
    EXPECT_EQ(basis.n_basis_functions(), 12u);
}

TEST(BasisExpansion, STO3G_AllSupportedElements) {
    // Test all supported elements together
    std::vector<data::Atom> atoms = {
        {1, {0.0, 0.0, 0.0}},     // H
        {6, {2.0, 0.0, 0.0}},     // C
        {7, {4.0, 0.0, 0.0}},     // N
        {8, {6.0, 0.0, 0.0}},     // O
        {9, {8.0, 0.0, 0.0}}      // F
    };
    auto basis = data::create_sto3g(atoms);
    // H:1 + C:5 + N:5 + O:5 + F:5 = 21
    EXPECT_EQ(basis.n_basis_functions(), 21u);

    Engine engine(basis);
    std::vector<Real> S;
    engine.compute_overlap_matrix(S);

    // Verify properties
    Size nbf = basis.n_basis_functions();
    for (Size i = 0; i < nbf; ++i) {
        EXPECT_NEAR(S[i * nbf + i], 1.0, 1e-10);
    }
}

// ============================================================================
// Augmented Basis Tests (Task 25.4.4)
// ============================================================================

TEST(BasisExpansion, AugmentedBasisNotYetSupported) {
    // aug-cc-pVXZ basis sets are not yet built in
    // Verify the API correctly reports them as unsupported
    std::vector<data::Atom> atoms = {{1, {0.0, 0.0, 0.0}}};

    EXPECT_THROW(data::create_builtin_basis("aug-cc-pvdz", atoms),
                 InvalidArgumentException);
    EXPECT_THROW(data::create_builtin_basis("aug-cc-pvtz", atoms),
                 InvalidArgumentException);
    EXPECT_THROW(data::create_builtin_basis("aug-cc-pvqz", atoms),
                 InvalidArgumentException);
}

TEST(BasisExpansion, DunningBasisNotYetSupported) {
    std::vector<data::Atom> atoms = {{1, {0.0, 0.0, 0.0}}};

    EXPECT_THROW(data::create_builtin_basis("cc-pvdz", atoms),
                 InvalidArgumentException);
    EXPECT_THROW(data::create_builtin_basis("cc-pvtz", atoms),
                 InvalidArgumentException);
}

TEST(BasisExpansion, AhlrichsBasisNotYetSupported) {
    std::vector<data::Atom> atoms = {{1, {0.0, 0.0, 0.0}}};

    EXPECT_THROW(data::create_builtin_basis("def2-svp", atoms),
                 InvalidArgumentException);
    EXPECT_THROW(data::create_builtin_basis("def2-tzvp", atoms),
                 InvalidArgumentException);
}

TEST(BasisExpansion, ManualDiffuseAugmentation) {
    // Manually add a diffuse function to approximate augmentation
    // This verifies the infrastructure works with diffuse functions
    std::vector<data::Atom> atoms = {{1, {0.0, 0.0, 0.0}}};
    auto basis_shells = data::create_sto3g(atoms);

    // Create a new basis with an added diffuse s-function
    std::vector<Shell> shells;
    for (Size i = 0; i < basis_shells.n_shells(); ++i) {
        const auto& s = basis_shells.shell(i);
        std::vector<Real> exps(s.exponents().begin(), s.exponents().end());
        std::vector<Real> coeffs(s.coefficients().begin(), s.coefficients().end());
        shells.emplace_back(
            libaccint::pre_normalized,
            s.angular_momentum(),
            s.center(),
            std::move(exps),
            std::move(coeffs));
    }

    // Add diffuse s function (small exponent)
    shells.emplace_back(0, Point3D{0.0, 0.0, 0.0},
                        std::vector<Real>{0.03}, std::vector<Real>{1.0});

    BasisSet augmented(std::move(shells));
    EXPECT_EQ(augmented.n_basis_functions(),
              basis_shells.n_basis_functions() + 1);

    // Compute overlap — should work without issues
    Engine engine(augmented);
    std::vector<Real> S;
    engine.compute_overlap_matrix(S);

    Size nbf = augmented.n_basis_functions();
    for (Size i = 0; i < nbf; ++i) {
        EXPECT_NEAR(S[i * nbf + i], 1.0, 1e-8);
    }
}
