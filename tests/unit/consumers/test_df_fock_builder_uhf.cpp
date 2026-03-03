// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2026 Bryce M. Westheimer. All rights reserved.

/// @file test_df_fock_builder_uhf.cpp
/// @brief Tests for DFFockBuilder UHF exchange: separate alpha/beta spin channels

#include <libaccint/consumers/df_fock_builder.hpp>
#include <libaccint/basis/auxiliary_basis_set.hpp>
#include <libaccint/basis/basis_set.hpp>
#include <libaccint/basis/shell.hpp>
#include <libaccint/core/types.hpp>

#include <gtest/gtest.h>
#include <cmath>
#include <memory>
#include <numeric>
#include <vector>

using namespace libaccint;
using namespace libaccint::consumers;

namespace {

constexpr Real TOL = 1e-10;

/// Create a minimal H2 orbital basis (STO-3G style, 2 basis functions)
std::unique_ptr<BasisSet> make_h2_orbital() {
    std::vector<Shell> shells;

    Shell s0(0, Point3D{0.0, 0.0, 0.0},
             {3.42525091, 0.62391373, 0.16885540},
             {0.15432897, 0.53532814, 0.44463454});
    s0.set_atom_index(0);
    shells.push_back(std::move(s0));

    Shell s1(0, Point3D{1.4, 0.0, 0.0},
             {3.42525091, 0.62391373, 0.16885540},
             {0.15432897, 0.53532814, 0.44463454});
    s1.set_atom_index(1);
    shells.push_back(std::move(s1));

    return std::make_unique<BasisSet>(std::move(shells));
}

/// Create a minimal auxiliary basis (3 s-type functions)
std::unique_ptr<AuxiliaryBasisSet> make_h2_auxiliary() {
    std::vector<Shell> shells;

    Shell a0(0, Point3D{0.0, 0.0, 0.0}, {2.0, 0.5}, {0.5, 0.5});
    a0.set_atom_index(0);
    shells.push_back(std::move(a0));

    Shell a1(0, Point3D{1.4, 0.0, 0.0}, {2.0, 0.5}, {0.5, 0.5});
    a1.set_atom_index(1);
    shells.push_back(std::move(a1));

    Shell a2(0, Point3D{0.7, 0.0, 0.0}, {1.5}, {1.0});
    a2.set_atom_index(0);
    shells.push_back(std::move(a2));

    return std::make_unique<AuxiliaryBasisSet>(
        std::move(shells), FittingType::JKFIT, "test-aux");
}

}  // anonymous namespace

// =============================================================================
// UHF Mode Tests (10.3.5)
// =============================================================================

TEST(DFFockBuilderUHF, SetDensityUnrestrictedActivatesUHF) {
    auto orb = make_h2_orbital();
    auto aux = make_h2_auxiliary();
    DFFockBuilder builder(*orb, *aux);

    EXPECT_FALSE(builder.is_uhf());

    // Set UHF densities
    const Size n = builder.n_orb();
    std::vector<Real> Da(n * n, 0.0), Db(n * n, 0.0);
    Da[0] = 0.5;
    Db[n * n - 1] = 0.5;

    builder.set_density_unrestricted(Da, Db);
    EXPECT_TRUE(builder.is_uhf());
}

TEST(DFFockBuilderUHF, SetDensityResetsUHFMode) {
    auto orb = make_h2_orbital();
    auto aux = make_h2_auxiliary();
    DFFockBuilder builder(*orb, *aux);
    const Size n = builder.n_orb();

    // Set UHF first
    std::vector<Real> Da(n * n, 0.25), Db(n * n, 0.25);
    builder.set_density_unrestricted(Da, Db);
    EXPECT_TRUE(builder.is_uhf());

    // Set RHF → should deactivate UHF
    std::vector<Real> D(n * n, 0.5);
    builder.set_density(D);
    EXPECT_FALSE(builder.is_uhf());
}

TEST(DFFockBuilderUHF, UHFExchangeMatricesDifferForDifferentDensities) {
    auto orb = make_h2_orbital();
    auto aux = make_h2_auxiliary();
    DFFockBuilder builder(*orb, *aux);
    const Size n = builder.n_orb();

    // Alpha electrons localized on atom 0, beta on atom 1
    std::vector<Real> Da(n * n, 0.0), Db(n * n, 0.0);
    Da[0] = 0.8;  Da[1] = 0.1;  Da[2] = 0.1;  Da[3] = 0.2;
    Db[0] = 0.2;  Db[1] = 0.1;  Db[2] = 0.1;  Db[3] = 0.8;

    builder.set_density_unrestricted(Da, Db);
    auto K = builder.compute_exchange();

    // K_alpha and K_beta should exist and be different
    auto Ka = builder.exchange_matrix_alpha();
    auto Kb = builder.exchange_matrix_beta();

    ASSERT_EQ(Ka.size(), n * n);
    ASSERT_EQ(Kb.size(), n * n);

    bool differs = false;
    for (Size i = 0; i < n * n; ++i) {
        if (std::abs(Ka[i] - Kb[i]) > TOL) {
            differs = true;
            break;
        }
    }
    EXPECT_TRUE(differs) << "K_alpha and K_beta should differ for asymmetric densities";
}

TEST(DFFockBuilderUHF, UHFExchangeMatricesEqualForEqualDensities) {
    auto orb = make_h2_orbital();
    auto aux = make_h2_auxiliary();
    DFFockBuilder builder(*orb, *aux);
    const Size n = builder.n_orb();

    // Same density for alpha and beta → should give same K
    std::vector<Real> D(n * n, 0.0);
    D[0] = 0.3;  D[1] = 0.1;  D[2] = 0.1;  D[3] = 0.3;

    builder.set_density_unrestricted(D, D);
    auto K = builder.compute_exchange();

    auto Ka = builder.exchange_matrix_alpha();
    auto Kb = builder.exchange_matrix_beta();

    ASSERT_EQ(Ka.size(), n * n);
    ASSERT_EQ(Kb.size(), n * n);

    for (Size i = 0; i < n * n; ++i) {
        EXPECT_NEAR(Ka[i], Kb[i], TOL)
            << "K_alpha and K_beta should be equal for equal densities at " << i;
    }
}

TEST(DFFockBuilderUHF, UHFCoulombSameAsRHFWithTotalDensity) {
    auto orb = make_h2_orbital();
    auto aux = make_h2_auxiliary();
    const Size n = 2;  // H2 has 2 basis functions

    // Total density
    std::vector<Real> D_total(n * n, 0.0);
    D_total[0] = 0.5;  D_total[1] = 0.1;
    D_total[2] = 0.1;  D_total[3] = 0.5;

    // Split into alpha + beta = total
    std::vector<Real> Da(n * n, 0.0), Db(n * n, 0.0);
    Da[0] = 0.3;  Da[1] = 0.05; Da[2] = 0.05; Da[3] = 0.2;
    Db[0] = 0.2;  Db[1] = 0.05; Db[2] = 0.05; Db[3] = 0.3;

    // RHF with total density
    DFFockBuilder builder_rhf(*orb, *aux);
    builder_rhf.set_density(D_total);
    auto J_rhf = builder_rhf.compute_coulomb();

    // UHF with split density
    DFFockBuilder builder_uhf(*orb, *aux);
    builder_uhf.set_density_unrestricted(Da, Db);
    auto J_uhf = builder_uhf.compute_coulomb();

    // Coulomb matrices should be the same (J depends on total density)
    ASSERT_EQ(J_rhf.size(), J_uhf.size());
    for (Size i = 0; i < J_rhf.size(); ++i) {
        EXPECT_NEAR(J_rhf[i], J_uhf[i], TOL)
            << "UHF and RHF Coulomb differ at element " << i;
    }
}
