// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2026 Bryce M. Westheimer. All rights reserved.

/// @file test_df_fock_builder_uhf_rhf.cpp
/// @brief Tests verifying UHF/RHF equivalence and spin-channel behavior
///
/// Key invariants:
///   - RHF-UHF equivalence: D_α = D_β = D/2 ⟹ K_α = K_β = K/2
///   - J is spin-independent (uses total density)
///   - K_α and K_β are each symmetric
///   - Open-shell: F_α ≠ F_β when D_α ≠ D_β

#include <libaccint/consumers/df_fock_builder.hpp>
#include <libaccint/basis/auxiliary_basis_set.hpp>
#include <libaccint/basis/basis_set.hpp>
#include <libaccint/basis/shell.hpp>
#include <libaccint/core/types.hpp>

#include <gtest/gtest.h>
#include <cmath>
#include <memory>
#include <vector>

using namespace libaccint;
using namespace libaccint::consumers;

namespace {

constexpr Real TOL = 1e-10;

/// Create H2 orbital basis (STO-3G, 2 BFs)
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

/// Auxiliary basis (3 s-type functions, same as existing UHF tests)
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

/// Symmetric density for H2
std::vector<Real> make_density() {
    return {0.5, 0.3, 0.3, 0.5};
}

}  // namespace

// =============================================================================
// RHF-UHF Equivalence
// =============================================================================

TEST(DFUhfRhfEquivalence, ClosedShellRhfUhfGiveSameJ) {
    // D_α = D_β = D/2 ⟹ D_total same ⟹ J should be identical
    auto D = make_density();
    const Size n2 = D.size();

    // RHF
    auto orb1 = make_h2_orbital();
    auto aux1 = make_h2_auxiliary();
    DFFockBuilder rhf_builder(*orb1, *aux1);
    rhf_builder.set_density(D);
    rhf_builder.initialize();
    auto J_rhf = rhf_builder.compute_coulomb();

    // UHF with D_α = D_β = D/2
    auto orb2 = make_h2_orbital();
    auto aux2 = make_h2_auxiliary();
    DFFockBuilder uhf_builder(*orb2, *aux2);
    std::vector<Real> D_half(n2);
    for (Size i = 0; i < n2; ++i) D_half[i] = D[i] * 0.5;
    uhf_builder.set_density_unrestricted(D_half, D_half);
    uhf_builder.initialize();
    auto J_uhf = uhf_builder.compute_coulomb();

    for (Size i = 0; i < n2; ++i) {
        EXPECT_NEAR(J_rhf[i], J_uhf[i], TOL)
            << "Closed-shell RHF/UHF J differ at index " << i;
    }
}

TEST(DFUhfRhfEquivalence, ClosedShellRhfUhfExchangeEquivalence) {
    // D_α = D_β = D/2 ⟹ K_α = K_β, and K_α + K_β = K_rhf
    auto D = make_density();
    const Size n2 = D.size();

    // RHF
    auto orb1 = make_h2_orbital();
    auto aux1 = make_h2_auxiliary();
    DFFockBuilder rhf_builder(*orb1, *aux1);
    rhf_builder.set_density(D);
    rhf_builder.initialize();
    auto K_rhf = rhf_builder.compute_exchange();

    // UHF with D_α = D_β = D/2
    auto orb2 = make_h2_orbital();
    auto aux2 = make_h2_auxiliary();
    DFFockBuilder uhf_builder(*orb2, *aux2);
    std::vector<Real> D_half(n2);
    for (Size i = 0; i < n2; ++i) D_half[i] = D[i] * 0.5;
    uhf_builder.set_density_unrestricted(D_half, D_half);
    uhf_builder.initialize();
    [[maybe_unused]] auto K_uhf_eq = uhf_builder.compute_exchange();

    auto K_alpha = uhf_builder.exchange_matrix_alpha();
    auto K_beta = uhf_builder.exchange_matrix_beta();

    // K_α should equal K_β (identical spin densities)
    for (Size i = 0; i < n2; ++i) {
        EXPECT_NEAR(K_alpha[i], K_beta[i], TOL)
            << "K_alpha != K_beta for closed-shell at index " << i;
    }

    // K_α + K_β should equal K_rhf
    for (Size i = 0; i < n2; ++i) {
        EXPECT_NEAR(K_alpha[i] + K_beta[i], K_rhf[i], TOL)
            << "K_alpha + K_beta != K_rhf at index " << i;
    }
}

// =============================================================================
// Spin-channel isolation
// =============================================================================

TEST(DFUhfRhfEquivalence, KAlphaOnlyDependsOnDAlpha) {
    const Size n = 2;
    const Size n2 = n * n;

    // Case 1: D_α = {0.5, 0.3, 0.3, 0.5}, D_β = {0.1, 0.0, 0.0, 0.1}
    auto orb1 = make_h2_orbital();
    auto aux1 = make_h2_auxiliary();
    DFFockBuilder builder1(*orb1, *aux1);
    std::vector<Real> Da = {0.5, 0.3, 0.3, 0.5};
    std::vector<Real> Db1 = {0.1, 0.0, 0.0, 0.1};
    builder1.set_density_unrestricted(Da, Db1);
    builder1.initialize();
    [[maybe_unused]] auto K1_discard = builder1.compute_exchange();
    auto Ka_1 = std::vector<Real>(builder1.exchange_matrix_alpha().begin(),
                                   builder1.exchange_matrix_alpha().end());

    // Case 2: same D_α, different D_β
    auto orb2 = make_h2_orbital();
    auto aux2 = make_h2_auxiliary();
    DFFockBuilder builder2(*orb2, *aux2);
    std::vector<Real> Db2 = {0.9, 0.2, 0.2, 0.3};
    builder2.set_density_unrestricted(Da, Db2);
    builder2.initialize();
    [[maybe_unused]] auto K2_discard = builder2.compute_exchange();
    auto Ka_2 = std::vector<Real>(builder2.exchange_matrix_alpha().begin(),
                                   builder2.exchange_matrix_alpha().end());

    // K_α should be the same in both cases (same D_α)
    for (Size i = 0; i < n2; ++i) {
        EXPECT_NEAR(Ka_1[i], Ka_2[i], TOL)
            << "K_alpha changed when only D_beta changed, index " << i;
    }
}

// =============================================================================
// J is spin-independent (depends only on total density)
// =============================================================================

TEST(DFUhfRhfEquivalence, JIsSpinIndependent) {
    const Size n = 2;
    const Size n2 = n * n;

    // Two different spin decompositions with same total D
    std::vector<Real> Da1 = {0.3, 0.2, 0.2, 0.3};
    std::vector<Real> Db1 = {0.2, 0.1, 0.1, 0.2};

    std::vector<Real> Da2 = {0.1, 0.05, 0.05, 0.1};
    std::vector<Real> Db2 = {0.4, 0.25, 0.25, 0.4};

    // Total D should be the same in both cases
    for (Size i = 0; i < n2; ++i) {
        EXPECT_NEAR(Da1[i] + Db1[i], Da2[i] + Db2[i], 1e-15)
            << "Test setup error: total densities differ";
    }

    auto orb1 = make_h2_orbital();
    auto aux1 = make_h2_auxiliary();
    DFFockBuilder builder1(*orb1, *aux1);
    builder1.set_density_unrestricted(Da1, Db1);
    builder1.initialize();
    auto J1 = builder1.compute_coulomb();

    auto orb2 = make_h2_orbital();
    auto aux2 = make_h2_auxiliary();
    DFFockBuilder builder2(*orb2, *aux2);
    builder2.set_density_unrestricted(Da2, Db2);
    builder2.initialize();
    auto J2 = builder2.compute_coulomb();

    for (Size i = 0; i < n2; ++i) {
        EXPECT_NEAR(J1[i], J2[i], TOL)
            << "J matrices differ for same total density at index " << i;
    }
}

// =============================================================================
// Spin-exchange matrices are symmetric
// =============================================================================

TEST(DFUhfRhfEquivalence, KAlphaIsSymmetric) {
    const Size n = 2;
    auto orb = make_h2_orbital();
    auto aux = make_h2_auxiliary();

    DFFockBuilder builder(*orb, *aux);
    std::vector<Real> Da = {0.5, 0.3, 0.3, 0.5};
    std::vector<Real> Db = {0.2, 0.1, 0.1, 0.2};
    builder.set_density_unrestricted(Da, Db);
    builder.initialize();
    [[maybe_unused]] auto K_sym_discard = builder.compute_exchange();

    auto Ka = builder.exchange_matrix_alpha();
    auto Kb = builder.exchange_matrix_beta();

    for (Size i = 0; i < n; ++i) {
        for (Size j = i + 1; j < n; ++j) {
            EXPECT_NEAR(Ka[i * n + j], Ka[j * n + i], TOL)
                << "K_alpha not symmetric at (" << i << "," << j << ")";
            EXPECT_NEAR(Kb[i * n + j], Kb[j * n + i], TOL)
                << "K_beta not symmetric at (" << i << "," << j << ")";
        }
    }
}

// =============================================================================
// Open-shell: F_α ≠ F_β
// =============================================================================

TEST(DFUhfRhfEquivalence, OpenShellFalphaDiffersFbeta) {
    const Size n = 2;
    const Size n2 = n * n;

    auto orb = make_h2_orbital();
    auto aux = make_h2_auxiliary();

    DFFockBuilder builder(*orb, *aux);

    // Distinctly different spin densities
    std::vector<Real> Da = {0.7, 0.2, 0.2, 0.3};
    std::vector<Real> Db = {0.1, 0.05, 0.05, 0.1};
    builder.set_density_unrestricted(Da, Db);
    builder.initialize();

    [[maybe_unused]] auto J_os = builder.compute_coulomb();
    [[maybe_unused]] auto K_os = builder.compute_exchange();

    auto J = builder.coulomb_matrix();
    auto Ka = builder.exchange_matrix_alpha();
    auto Kb = builder.exchange_matrix_beta();

    // F_α = J - K_α, F_β = J - K_β
    // Since Da ≠ Db, Ka ≠ Kb, so F_α ≠ F_β
    bool any_different = false;
    for (Size i = 0; i < n2; ++i) {
        Real Fa_i = J[i] - Ka[i];
        Real Fb_i = J[i] - Kb[i];
        if (std::abs(Fa_i - Fb_i) > 1e-14) {
            any_different = true;
            break;
        }
    }
    EXPECT_TRUE(any_different) << "Open-shell F_alpha should differ from F_beta";
}

// =============================================================================
// UHF total K is sum of spin K
// =============================================================================

TEST(DFUhfRhfEquivalence, TotalKIsSumOfSpinK) {
    const Size n = 2;
    const Size n2 = n * n;

    auto orb = make_h2_orbital();
    auto aux = make_h2_auxiliary();

    DFFockBuilder builder(*orb, *aux);
    std::vector<Real> Da = {0.4, 0.2, 0.2, 0.3};
    std::vector<Real> Db = {0.3, 0.1, 0.1, 0.4};
    builder.set_density_unrestricted(Da, Db);
    builder.initialize();

    auto K_total = builder.compute_exchange();
    auto Ka = builder.exchange_matrix_alpha();
    auto Kb = builder.exchange_matrix_beta();

    for (Size i = 0; i < n2; ++i) {
        EXPECT_NEAR(K_total[i], Ka[i] + Kb[i], TOL)
            << "K_total != K_alpha + K_beta at index " << i;
    }
}
