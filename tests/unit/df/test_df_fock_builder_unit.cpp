// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2026 Bryce M. Westheimer. All rights reserved.

/// @file test_df_fock_builder_unit.cpp
/// @brief Unit tests for DFFockBuilder: factory functions, lifecycle,
///        individual J/K computation, compute_accumulate, fock_matrix,
///        exchange fraction configuration, and set_density → compute cycle.

#include <libaccint/consumers/df_fock_builder.hpp>
#include <libaccint/basis/auxiliary_basis_set.hpp>
#include <libaccint/basis/basis_set.hpp>
#include <libaccint/basis/shell.hpp>
#include <libaccint/core/types.hpp>
#include <libaccint/utils/error_handling.hpp>

#include <gtest/gtest.h>
#include <cmath>
#include <memory>
#include <numeric>
#include <vector>

using namespace libaccint;
using namespace libaccint::consumers;

namespace {

constexpr Real TOL = 1e-10;

/// Create H2 orbital basis (STO-3G style, 2 basis functions)
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

/// Create an auxiliary basis (4 s-type functions per atom)
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

/// Create a simple symmetric density for H2 (2x2)
std::vector<Real> make_h2_density() {
    return {0.5, 0.3, 0.3, 0.5};
}

}  // namespace

// =============================================================================
// Construction & Initialization
// =============================================================================

TEST(DFFockBuilderUnit, NonOwningConstructorSetsBasicFields) {
    auto orb = make_h2_orbital();
    auto aux = make_h2_auxiliary();

    DFFockBuilder builder(*orb, *aux);

    EXPECT_EQ(builder.n_orb(), orb->n_basis_functions());
    EXPECT_EQ(builder.n_aux(), aux->n_functions());
    EXPECT_FALSE(builder.is_initialized());
    EXPECT_FALSE(builder.is_uhf());
}

TEST(DFFockBuilderUnit, OwningConstructorTakesOwnership) {
    auto orb = make_h2_orbital();
    auto aux = make_h2_auxiliary();
    Size n_aux_expected = aux->n_functions();

    DFFockBuilder builder(*orb, std::move(aux));

    EXPECT_EQ(builder.n_aux(), n_aux_expected);
    EXPECT_FALSE(builder.is_initialized());
    // aux is now moved-from — the builder owns it
}

TEST(DFFockBuilderUnit, InitializeOnce) {
    auto orb = make_h2_orbital();
    auto aux = make_h2_auxiliary();

    DFFockBuilder builder(*orb, *aux);
    builder.set_density(make_h2_density());

    EXPECT_FALSE(builder.is_initialized());
    builder.initialize();
    EXPECT_TRUE(builder.is_initialized());

    // Calling initialize again should be a no-op (no crash)
    builder.initialize();
    EXPECT_TRUE(builder.is_initialized());
}

TEST(DFFockBuilderUnit, SetDensitySizeMismatchThrows) {
    auto orb = make_h2_orbital();
    auto aux = make_h2_auxiliary();

    DFFockBuilder builder(*orb, *aux);

    // Wrong size density
    std::vector<Real> bad_D(5, 0.0);
    EXPECT_THROW(builder.set_density(bad_D), InvalidArgumentException);
}

// =============================================================================
// Compute Coulomb and Exchange Independently
// =============================================================================

TEST(DFFockBuilderUnit, ComputeCoulombReturnsCorrectSize) {
    auto orb = make_h2_orbital();
    auto aux = make_h2_auxiliary();
    const Size n = orb->n_basis_functions();

    DFFockBuilder builder(*orb, *aux);
    builder.set_density(make_h2_density());
    auto J = builder.compute_coulomb();

    EXPECT_EQ(J.size(), n * n);
}

TEST(DFFockBuilderUnit, ComputeExchangeReturnsCorrectSize) {
    auto orb = make_h2_orbital();
    auto aux = make_h2_auxiliary();
    const Size n = orb->n_basis_functions();

    DFFockBuilder builder(*orb, *aux);
    builder.set_density(make_h2_density());
    auto K = builder.compute_exchange();

    EXPECT_EQ(K.size(), n * n);
}

TEST(DFFockBuilderUnit, ComputeAutoInitializes) {
    auto orb = make_h2_orbital();
    auto aux = make_h2_auxiliary();

    DFFockBuilder builder(*orb, *aux);
    builder.set_density(make_h2_density());

    EXPECT_FALSE(builder.is_initialized());
    auto F = builder.compute();
    EXPECT_TRUE(builder.is_initialized());
    EXPECT_EQ(F.size(), orb->n_basis_functions() * orb->n_basis_functions());
}

// =============================================================================
// Fock = J - fraction * K
// =============================================================================

TEST(DFFockBuilderUnit, FockIsJMinusK) {
    auto orb = make_h2_orbital();
    auto aux = make_h2_auxiliary();
    const Size n = orb->n_basis_functions();
    auto D = make_h2_density();

    DFFockBuilder builder(*orb, *aux);
    builder.set_density(D);
    builder.initialize();

    auto J = builder.compute_coulomb();
    auto K = builder.compute_exchange();
    auto F = builder.compute();

    // F = J - 1.0 * K (default exchange_fraction = 1.0)
    for (Size i = 0; i < n * n; ++i) {
        EXPECT_NEAR(F[i], J[i] - K[i], TOL)
            << "F[" << i << "] != J - K";
    }
}

TEST(DFFockBuilderUnit, ExchangeFractionAffectsFock) {
    auto orb = make_h2_orbital();
    auto aux = make_h2_auxiliary();
    const Size n = orb->n_basis_functions();
    auto D = make_h2_density();

    DFFockBuilderConfig config;
    config.exchange_fraction = 0.5;
    DFFockBuilder builder(*orb, *aux, config);
    builder.set_density(D);
    builder.initialize();

    auto J = builder.compute_coulomb();
    auto K = builder.compute_exchange();
    auto F = builder.compute();

    // F = J - 0.5 * K
    for (Size i = 0; i < n * n; ++i) {
        EXPECT_NEAR(F[i], J[i] - 0.5 * K[i], TOL)
            << "F[" << i << "] != J - 0.5*K";
    }
}

// =============================================================================
// compute_accumulate
// =============================================================================

TEST(DFFockBuilderUnit, ComputeAccumulateAddsToBuffer) {
    auto orb = make_h2_orbital();
    auto aux = make_h2_auxiliary();
    const Size n = orb->n_basis_functions();
    auto D = make_h2_density();

    DFFockBuilder builder(*orb, *aux);
    builder.set_density(D);

    auto F_ref = builder.compute();

    // Reset and use compute_accumulate onto a zero buffer
    auto orb2 = make_h2_orbital();
    auto aux2 = make_h2_auxiliary();
    DFFockBuilder builder2(*orb2, *aux2);
    builder2.set_density(D);

    std::vector<Real> F_accum(n * n, 0.0);
    builder2.compute_accumulate(F_accum);

    for (Size i = 0; i < n * n; ++i) {
        EXPECT_NEAR(F_accum[i], F_ref[i], TOL)
            << "compute_accumulate result differs at index " << i;
    }
}

TEST(DFFockBuilderUnit, ComputeAccumulateAccumulatesOntoExisting) {
    auto orb = make_h2_orbital();
    auto aux = make_h2_auxiliary();
    const Size n = orb->n_basis_functions();
    auto D = make_h2_density();

    DFFockBuilder builder(*orb, *aux);
    builder.set_density(D);

    auto F_single = builder.compute();

    // Accumulate twice
    auto orb2 = make_h2_orbital();
    auto aux2 = make_h2_auxiliary();
    DFFockBuilder builder2(*orb2, *aux2);
    builder2.set_density(D);
    builder2.initialize();

    std::vector<Real> F_double(n * n, 0.0);
    builder2.compute_accumulate(F_double);
    builder2.compute_accumulate(F_double);

    for (Size i = 0; i < n * n; ++i) {
        EXPECT_NEAR(F_double[i], 2.0 * F_single[i], TOL)
            << "Double accumulate should be 2x single at index " << i;
    }
}

// =============================================================================
// fock_matrix with H_core
// =============================================================================

TEST(DFFockBuilderUnit, FockMatrixWithHCore) {
    auto orb = make_h2_orbital();
    auto aux = make_h2_auxiliary();
    const Size n = orb->n_basis_functions();
    auto D = make_h2_density();

    DFFockBuilder builder(*orb, *aux);
    builder.set_density(D);
    builder.initialize();
    [[maybe_unused]] auto J_tmp = builder.compute_coulomb();
    [[maybe_unused]] auto K_tmp = builder.compute_exchange();

    // Create a simple H_core
    std::vector<Real> H_core(n * n, 0.0);
    H_core[0] = -1.5;
    H_core[n * n - 1] = -1.5;

    auto F = builder.fock_matrix(H_core);
    auto F_no_h = builder.fock_matrix();

    // F_with_H = H + F_no_H
    for (Size i = 0; i < n * n; ++i) {
        EXPECT_NEAR(F[i], F_no_h[i] + H_core[i], TOL);
    }
}

// =============================================================================
// Set density → repeated compute cycle
// =============================================================================

TEST(DFFockBuilderUnit, RepeatedComputeWithDifferentDensity) {
    auto orb = make_h2_orbital();
    auto aux = make_h2_auxiliary();
    const Size n = orb->n_basis_functions();

    DFFockBuilder builder(*orb, *aux);

    // First density
    auto D1 = make_h2_density();
    builder.set_density(D1);
    auto F1 = builder.compute();

    // Second different density
    std::vector<Real> D2 = {0.3, 0.1, 0.1, 0.7};
    builder.set_density(D2);
    auto F2 = builder.compute();

    // F1 and F2 should differ (different densities)
    bool any_different = false;
    for (Size i = 0; i < n * n; ++i) {
        if (std::abs(F1[i] - F2[i]) > 1e-14) {
            any_different = true;
            break;
        }
    }
    EXPECT_TRUE(any_different) << "Different densities should yield different Fock matrices";
}

// =============================================================================
// Factory Function Tests
// =============================================================================

TEST(DFFockBuilderUnit, FactoryWithoutAtomDataThrowsEmpty) {
    auto orb = make_h2_orbital();

    // Without auxiliary name and atom data, should throw
    EXPECT_THROW(make_df_fock_builder(*orb, ""), InvalidArgumentException);
}

TEST(DFFockBuilderUnit, FactoryWithExplicitAuxNameCreatesBuilder) {
    auto orb = make_h2_orbital();

    // With an explicit name, factory creates a minimal aux basis
    auto builder = make_df_fock_builder(*orb, "test-fallback");
    EXPECT_NE(builder, nullptr);
    EXPECT_EQ(builder->n_orb(), orb->n_basis_functions());
    EXPECT_GT(builder->n_aux(), 0u);
}

// =============================================================================
// Configuration access
// =============================================================================

TEST(DFFockBuilderUnit, ConfigAccessReturnsSetValues) {
    auto orb = make_h2_orbital();
    auto aux = make_h2_auxiliary();

    DFFockBuilderConfig config;
    config.exchange_fraction = 0.75;
    config.compute_coulomb = true;
    config.compute_exchange = false;

    DFFockBuilder builder(*orb, *aux, config);

    EXPECT_NEAR(builder.config().exchange_fraction, 0.75, 1e-15);
    EXPECT_TRUE(builder.config().compute_coulomb);
    EXPECT_FALSE(builder.config().compute_exchange);
}

TEST(DFFockBuilderUnit, ComputeWithExchangeDisabled) {
    auto orb = make_h2_orbital();
    auto aux = make_h2_auxiliary();
    const Size n = orb->n_basis_functions();
    auto D = make_h2_density();

    DFFockBuilderConfig config;
    config.compute_exchange = false;

    DFFockBuilder builder(*orb, *aux, config);
    builder.set_density(D);
    builder.initialize();

    auto J = builder.compute_coulomb();
    auto F = builder.compute();

    // With exchange disabled, F should equal J
    for (Size i = 0; i < n * n; ++i) {
        EXPECT_NEAR(F[i], J[i], TOL)
            << "F should equal J when exchange is disabled";
    }
}
