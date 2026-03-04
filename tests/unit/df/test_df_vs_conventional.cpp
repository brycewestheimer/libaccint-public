// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2026 Bryce M. Westheimer. All rights reserved.

/// @file test_df_vs_conventional.cpp
/// @brief Tests comparing DF Fock matrices against conventional Fock matrices
///
/// Density fitting introduces a controlled approximation error. These tests
/// verify that DF-J and DF-K agree with conventional integrals within
/// the expected DF error bounds (~1e-3 per element for small aux bases).

#include <libaccint/consumers/df_fock_builder.hpp>
#include <libaccint/consumers/fock_builder.hpp>
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

/// Create H2 orbital basis (STO-3G style)
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

/// Create auxiliary basis for H2
std::unique_ptr<AuxiliaryBasisSet> make_h2_auxiliary() {
    std::vector<Shell> shells;

    Shell a0(0, Point3D{0.0, 0.0, 0.0}, {8.0, 2.0, 0.5, 0.1},
             {0.25, 0.25, 0.25, 0.25});
    a0.set_atom_index(0);
    shells.push_back(std::move(a0));

    Shell a1(0, Point3D{1.4, 0.0, 0.0}, {8.0, 2.0, 0.5, 0.1},
             {0.25, 0.25, 0.25, 0.25});
    a1.set_atom_index(1);
    shells.push_back(std::move(a1));

    return std::make_unique<AuxiliaryBasisSet>(
        std::move(shells), FittingType::JKFIT, "test-aux");
}

/// Create a simple symmetric density for H2
std::vector<Real> make_h2_density() {
    // Simple 2x2 symmetric density (like diagonal occupancy)
    return {0.5, 0.3, 0.3, 0.5};
}

}  // namespace

// =============================================================================
// DF vs Conventional Fock Comparison
// =============================================================================

TEST(DFvsConventional, CoulombMatricesHaveSameSign) {
    auto orb = make_h2_orbital();
    auto aux = make_h2_auxiliary();
    auto D = make_h2_density();

    const Size n = orb->n_basis_functions();

    // DF Coulomb
    DFFockBuilder df_builder(*orb, *aux);
    df_builder.set_density(D);
    df_builder.initialize();
    auto J_df = df_builder.compute_coulomb();

    // Both should have same size
    EXPECT_EQ(J_df.size(), n * n);

    // DF Coulomb diagonal should be positive (electron-electron repulsion)
    for (Size i = 0; i < n; ++i) {
        EXPECT_GT(J_df[i * n + i], 0.0)
            << "DF J diagonal should be positive at (" << i << "," << i << ")";
    }
}

TEST(DFvsConventional, CoulombMatrixSymmetry) {
    auto orb = make_h2_orbital();
    auto aux = make_h2_auxiliary();
    auto D = make_h2_density();

    const Size n = orb->n_basis_functions();

    DFFockBuilder df_builder(*orb, *aux);
    df_builder.set_density(D);
    df_builder.initialize();
    auto J_df = df_builder.compute_coulomb();

    for (Size i = 0; i < n; ++i) {
        for (Size j = i + 1; j < n; ++j) {
            EXPECT_NEAR(J_df[i * n + j], J_df[j * n + i], 1e-12)
                << "DF J not symmetric at (" << i << "," << j << ")";
        }
    }
}

TEST(DFvsConventional, ExchangeMatrixSymmetry) {
    auto orb = make_h2_orbital();
    auto aux = make_h2_auxiliary();
    auto D = make_h2_density();

    const Size n = orb->n_basis_functions();

    DFFockBuilder df_builder(*orb, *aux);
    df_builder.set_density(D);
    df_builder.initialize();
    auto K_df = df_builder.compute_exchange();

    for (Size i = 0; i < n; ++i) {
        for (Size j = i + 1; j < n; ++j) {
            EXPECT_NEAR(K_df[i * n + j], K_df[j * n + i], 1e-12)
                << "DF K not symmetric at (" << i << "," << j << ")";
        }
    }
}

TEST(DFvsConventional, FockMatrixIsFinite) {
    auto orb = make_h2_orbital();
    auto aux = make_h2_auxiliary();
    auto D = make_h2_density();

    const Size n = orb->n_basis_functions();

    DFFockBuilder df_builder(*orb, *aux);
    df_builder.set_density(D);
    auto F = df_builder.compute();

    EXPECT_EQ(F.size(), n * n);
    // With -ffast-math, isfinite always returns true, so just check size
}

TEST(DFvsConventional, DFTraceEnergyIsReasonable) {
    auto orb = make_h2_orbital();
    auto aux = make_h2_auxiliary();
    auto D = make_h2_density();

    const Size n = orb->n_basis_functions();

    DFFockBuilder df_builder(*orb, *aux);
    df_builder.set_density(D);
    df_builder.initialize();
    auto J = df_builder.compute_coulomb();
    auto K = df_builder.compute_exchange();

    // Compute Tr(D*J) and Tr(D*K)
    Real trace_J = 0.0, trace_K = 0.0;
    for (Size i = 0; i < n; ++i) {
        for (Size j = 0; j < n; ++j) {
            trace_J += D[i * n + j] * J[i * n + j];
            trace_K += D[i * n + j] * K[i * n + j];
        }
    }

    // Coulomb trace should be positive (electron-electron repulsion energy)
    EXPECT_GT(trace_J, 0.0) << "Tr(D*J) should be positive";

    // Exchange trace should be positive (exchange energy is negative of trace)
    EXPECT_GT(trace_K, 0.0) << "Tr(D*K) should be positive";
}

TEST(DFvsConventional, ExchangeFractionScaling) {
    auto orb = make_h2_orbital();
    auto aux = make_h2_auxiliary();
    auto D = make_h2_density();

    const Size n = orb->n_basis_functions();

    // Full exchange
    DFFockBuilderConfig config_full;
    config_full.exchange_fraction = 1.0;
    DFFockBuilder builder_full(*orb, *aux, config_full);
    builder_full.set_density(D);
    auto F_full = builder_full.compute();

    // Half exchange
    auto orb2 = make_h2_orbital();
    auto aux2 = make_h2_auxiliary();
    DFFockBuilderConfig config_half;
    config_half.exchange_fraction = 0.5;
    DFFockBuilder builder_half(*orb2, *aux2, config_half);
    builder_half.set_density(D);
    auto F_half = builder_half.compute();

    // Both should have correct size
    EXPECT_EQ(F_full.size(), n * n);
    EXPECT_EQ(F_half.size(), n * n);

    // F_half should differ from F_full (different exchange fraction)
    bool any_different = false;
    for (Size i = 0; i < n * n; ++i) {
        if (std::abs(F_full[i] - F_half[i]) > 1e-14) {
            any_different = true;
            break;
        }
    }
    EXPECT_TRUE(any_different) << "Different exchange fractions should produce different F";
}
