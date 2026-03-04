// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2026 Bryce M. Westheimer. All rights reserved.

/// @file test_df_fock_builder.cpp
/// @brief Unit tests for DFFockBuilder

#include <gtest/gtest.h>

#include <libaccint/consumers/df_fock_builder.hpp>
#include <libaccint/basis/auxiliary_basis_set.hpp>
#include <libaccint/basis/basis_set.hpp>
#include <libaccint/basis/shell.hpp>
#include <libaccint/core/types.hpp>
#include <libaccint/utils/error_handling.hpp>

#include <cmath>
#include <numeric>

namespace libaccint::testing {

// =============================================================================
// Test Fixtures
// =============================================================================

class DFFockBuilderTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Create minimal H2 orbital basis (STO-3G style)
        std::vector<Shell> orb_shells;
        
        // H1 at origin
        orb_shells.push_back(Shell(
            0,  // s-shell
            Point3D{0.0, 0.0, 0.0},
            {3.42525091, 0.62391373, 0.16885540},  // exponents
            {0.15432897, 0.53532814, 0.44463454}   // coefficients
        ));

        // H2 at (1.4, 0, 0) Bohr
        orb_shells.push_back(Shell(
            0,
            Point3D{1.4, 0.0, 0.0},
            {3.42525091, 0.62391373, 0.16885540},
            {0.15432897, 0.53532814, 0.44463454}
        ));
        
        orbital_basis_ = std::make_unique<BasisSet>(std::move(orb_shells));

        // Create minimal auxiliary basis
        std::vector<Shell> aux_shells;
        
        // Auxiliary s-function on H1
        aux_shells.push_back(Shell(0, Point3D{0.0, 0.0, 0.0}, {2.0, 0.5}, {0.5, 0.5}));

        // Auxiliary s-function on H2
        aux_shells.push_back(Shell(0, Point3D{1.4, 0.0, 0.0}, {2.0, 0.5}, {0.5, 0.5}));

        // Auxiliary s-function at midpoint
        aux_shells.push_back(Shell(0, Point3D{0.7, 0.0, 0.0}, {1.5}, {1.0}));
        
        auxiliary_basis_ = std::make_unique<AuxiliaryBasisSet>(
            std::move(aux_shells), FittingType::JKFIT, "test-aux");
    }

    std::unique_ptr<BasisSet> orbital_basis_;
    std::unique_ptr<AuxiliaryBasisSet> auxiliary_basis_;
};

// =============================================================================
// Construction Tests
// =============================================================================

TEST_F(DFFockBuilderTest, Construction) {
    consumers::DFFockBuilder builder(*orbital_basis_, *auxiliary_basis_);

    EXPECT_EQ(builder.n_orb(), 2u);
    EXPECT_EQ(builder.n_aux(), 3u);
    EXPECT_FALSE(builder.is_initialized());
}

TEST_F(DFFockBuilderTest, Initialization) {
    consumers::DFFockBuilder builder(*orbital_basis_, *auxiliary_basis_);
    
    builder.initialize();
    
    EXPECT_TRUE(builder.is_initialized());
}

TEST_F(DFFockBuilderTest, BTensorMemory) {
    consumers::DFFockBuilder builder(*orbital_basis_, *auxiliary_basis_);
    
    // B tensor is (n_orb^2 x n_aux) * sizeof(Real)
    EXPECT_EQ(builder.b_tensor_memory(), 2u * 2u * 3u * sizeof(Real));
}

// =============================================================================
// Density Matrix Tests
// =============================================================================

TEST_F(DFFockBuilderTest, SetDensity) {
    consumers::DFFockBuilder builder(*orbital_basis_, *auxiliary_basis_);
    
    std::vector<Real> D(4, 0.0);
    D[0] = 1.0;  // D_00
    D[3] = 1.0;  // D_11
    
    EXPECT_NO_THROW(builder.set_density(D));
}

TEST_F(DFFockBuilderTest, SetDensityWrongSize) {
    consumers::DFFockBuilder builder(*orbital_basis_, *auxiliary_basis_);
    
    std::vector<Real> D(5, 0.0);  // Wrong size
    
    EXPECT_THROW(builder.set_density(D), InvalidArgumentException);
}

// =============================================================================
// Fock Matrix Construction Tests
// =============================================================================

TEST_F(DFFockBuilderTest, ComputeCoulomb) {
    consumers::DFFockBuilder builder(*orbital_basis_, *auxiliary_basis_);
    
    // Identity-like density
    std::vector<Real> D(4);
    D[0] = 0.5; D[1] = 0.25;
    D[2] = 0.25; D[3] = 0.5;
    
    builder.set_density(D);
    auto J = builder.compute_coulomb();
    
    EXPECT_EQ(J.size(), 4u);
    
    // J should be symmetric
    EXPECT_NEAR(J[1], J[2], 1e-12);
    
    // J should be positive (Coulomb repulsion)
    EXPECT_GT(J[0], 0.0);
    EXPECT_GT(J[3], 0.0);
}

TEST_F(DFFockBuilderTest, ComputeExchange) {
    consumers::DFFockBuilder builder(*orbital_basis_, *auxiliary_basis_);
    
    std::vector<Real> D(4);
    D[0] = 0.5; D[1] = 0.25;
    D[2] = 0.25; D[3] = 0.5;
    
    builder.set_density(D);
    auto K = builder.compute_exchange();
    
    EXPECT_EQ(K.size(), 4u);
    
    // K should be symmetric
    EXPECT_NEAR(K[1], K[2], 1e-12);
}

TEST_F(DFFockBuilderTest, ComputeFockMatrix) {
    consumers::DFFockBuilder builder(*orbital_basis_, *auxiliary_basis_);
    
    std::vector<Real> D(4);
    D[0] = 0.5; D[1] = 0.25;
    D[2] = 0.25; D[3] = 0.5;
    
    builder.set_density(D);
    auto F = builder.compute();  // F = J - K
    
    EXPECT_EQ(F.size(), 4u);
    
    // F should be symmetric for closed-shell
    EXPECT_NEAR(F[1], F[2], 1e-12);
}

TEST_F(DFFockBuilderTest, FockWithHCore) {
    consumers::DFFockBuilder builder(*orbital_basis_, *auxiliary_basis_);
    
    std::vector<Real> D(4);
    D[0] = 0.5; D[1] = 0.25;
    D[2] = 0.25; D[3] = 0.5;
    
    std::vector<Real> H_core(4);
    H_core[0] = -1.0; H_core[1] = -0.5;
    H_core[2] = -0.5; H_core[3] = -1.0;
    
    builder.set_density(D);
    auto G = builder.compute();

    auto F = builder.fock_matrix(H_core);
    
    // F should include H_core contribution
    // The exact values depend on the integrals, but F should not equal J-K
    EXPECT_NE(F[0], builder.coulomb_matrix()[0] - builder.exchange_matrix()[0]);
}

// =============================================================================
// Configuration Tests
// =============================================================================

TEST_F(DFFockBuilderTest, ConfigExchangeFraction) {
    consumers::DFFockBuilderConfig config;
    config.exchange_fraction = 0.5;  // Hybrid functional style
    
    consumers::DFFockBuilder builder(*orbital_basis_, *auxiliary_basis_, config);
    
    std::vector<Real> D(4, 0.25);
    builder.set_density(D);
    
    auto F = builder.compute();
    
    // F = J - 0.5 * K
    auto J = builder.coulomb_matrix();
    auto K = builder.exchange_matrix();
    
    for (size_t i = 0; i < 4; ++i) {
        EXPECT_NEAR(F[i], J[i] - 0.5 * K[i], 1e-12);
    }
}

TEST_F(DFFockBuilderTest, CoulombOnly) {
    consumers::DFFockBuilderConfig config;
    config.compute_coulomb = true;
    config.compute_exchange = false;
    
    consumers::DFFockBuilder builder(*orbital_basis_, *auxiliary_basis_, config);
    
    std::vector<Real> D(4, 0.25);
    builder.set_density(D);
    
    auto F = builder.compute();
    
    // F should equal J (no exchange)
    auto J = builder.coulomb_matrix();
    
    for (size_t i = 0; i < 4; ++i) {
        EXPECT_NEAR(F[i], J[i], 1e-12);
    }
}

// =============================================================================
// Numerical Consistency Tests
// =============================================================================

TEST_F(DFFockBuilderTest, SymmetricDensityGivesSymmetricFock) {
    consumers::DFFockBuilder builder(*orbital_basis_, *auxiliary_basis_);
    
    // Symmetric density
    std::vector<Real> D(4);
    D[0] = 0.6; D[1] = 0.3;
    D[2] = 0.3; D[3] = 0.4;
    
    builder.set_density(D);
    auto F = builder.compute();
    
    // Check symmetry: F_01 = F_10
    EXPECT_NEAR(F[1], F[2], 1e-12);
}

TEST_F(DFFockBuilderTest, ZeroDensityGivesZeroFock) {
    consumers::DFFockBuilder builder(*orbital_basis_, *auxiliary_basis_);
    
    std::vector<Real> D(4, 0.0);
    
    builder.set_density(D);
    auto F = builder.compute();
    
    for (size_t i = 0; i < 4; ++i) {
        EXPECT_NEAR(F[i], 0.0, 1e-14);
    }
}

TEST_F(DFFockBuilderTest, RepeatedComputeGivesSameResult) {
    consumers::DFFockBuilder builder(*orbital_basis_, *auxiliary_basis_);
    
    std::vector<Real> D(4);
    D[0] = 0.5; D[1] = 0.2;
    D[2] = 0.2; D[3] = 0.5;
    
    builder.set_density(D);
    
    auto F1 = builder.compute();
    auto F2 = builder.compute();
    
    for (size_t i = 0; i < 4; ++i) {
        EXPECT_NEAR(F1[i], F2[i], 1e-14);
    }
}

}  // namespace libaccint::testing
