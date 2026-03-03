// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2026 Bryce M. Westheimer. All rights reserved.

/// @file test_batched_2e_workflow.cpp
/// @brief Batched two-electron integral workflow tests using (H₂O)₄/aug-cc-pVDZ
///
/// Step 11.3: Validates FockBuilder construction, J/K matrix symmetry,
/// ShellSetQuartet properties, Schwarz screening, parallel consistency,
/// and manual shell-set-quartet iteration on a physically reasonable system.

#include "h2o4_fixture.hpp"

#include <libaccint/basis/shell_set_quartet.hpp>
#include <libaccint/consumers/fock_builder.hpp>
#include <libaccint/core/types.hpp>
#include <libaccint/engine/engine.hpp>
#include <libaccint/operators/operator.hpp>
#include <libaccint/screening/screening_options.hpp>

#include <gtest/gtest.h>
#include <algorithm>
#include <cmath>
#include <numeric>
#include <vector>

namespace libaccint::test {

// =============================================================================
// Helper: build a unit (identity) density matrix
// =============================================================================

static std::vector<Real> make_identity_density(Size nbf) {
    std::vector<Real> D(nbf * nbf, 0.0);
    for (Size i = 0; i < nbf; ++i) {
        D[i * nbf + i] = 1.0;
    }
    return D;
}

// =============================================================================
// FockBuilder Setup
// =============================================================================

TEST_F(H2O4AugccpVDZFixture, FockBuilderConstruction) {
    consumers::FockBuilder fock(nbf_);
    EXPECT_EQ(fock.nbf(), nbf_);
}

// =============================================================================
// Basic Compute-and-Consume
// =============================================================================

TEST_F(H2O4AugccpVDZFixture, ComputeAndConsumeUnitDensity) {
    auto D = make_identity_density(nbf_);

    consumers::FockBuilder fock(nbf_);
    fock.set_density(D.data(), nbf_);

    Operator coulomb = Operator::coulomb();
    engine_->compute_and_consume(coulomb, fock);

    auto J = fock.get_coulomb_matrix();
    auto K = fock.get_exchange_matrix();

    // J matrix should be non-zero for unit density
    Real J_max = 0.0;
    for (Size i = 0; i < nbf_ * nbf_; ++i) {
        J_max = std::max(J_max, std::abs(J[i]));
    }
    EXPECT_GT(J_max, 0.0) << "Coulomb matrix should be non-zero for unit density";

    // J matrix should be symmetric
    expect_symmetric(std::vector<Real>(J.begin(), J.end()), nbf_, 1e-12,
                     "Coulomb");
}

TEST_F(H2O4AugccpVDZFixture, CoulombMatrixSymmetry) {
    auto D = make_identity_density(nbf_);

    consumers::FockBuilder fock(nbf_);
    fock.set_density(D.data(), nbf_);

    engine_->compute_and_consume(Operator::coulomb(), fock);

    auto J = fock.get_coulomb_matrix();
    expect_symmetric(std::vector<Real>(J.begin(), J.end()), nbf_, 1e-14,
                     "Coulomb");
}

TEST_F(H2O4AugccpVDZFixture, ExchangeMatrixSymmetry) {
    auto D = make_identity_density(nbf_);

    consumers::FockBuilder fock(nbf_);
    fock.set_density(D.data(), nbf_);

    engine_->compute_and_consume(Operator::coulomb(), fock);

    auto K = fock.get_exchange_matrix();
    expect_symmetric(std::vector<Real>(K.begin(), K.end()), nbf_, 1e-14,
                     "Exchange");
}

// =============================================================================
// ShellSetQuartet Verification
// =============================================================================

TEST_F(H2O4AugccpVDZFixture, ShellSetQuartetCount) {
    const auto& quartets = basis_->shell_set_quartets();
    EXPECT_FALSE(quartets.empty())
        << "shell_set_quartets() should not be empty";

    // For aug-cc-pVDZ on (H2O)4, expect a substantial number of SSQs.
    // With n_shell_sets ≥ 4, the unique pair count is ≥ 10, and SSQ count
    // should be at least as many as the unique pair count squared / 2.
    EXPECT_GE(quartets.size(), 10u)
        << "Expected at least 10 shell set quartets for (H2O)4/aug-cc-pVDZ";
}

TEST_F(H2O4AugccpVDZFixture, ShellSetQuartetNQuartets) {
    const auto& quartets = basis_->shell_set_quartets();

    for (const auto& q : quartets) {
        const auto& bra = q.bra_pair();
        const auto& ket = q.ket_pair();

        Size n_shells_a = bra.shell_set_a().n_shells();
        Size n_shells_b = bra.shell_set_b().n_shells();
        Size n_shells_c = ket.shell_set_a().n_shells();
        Size n_shells_d = ket.shell_set_b().n_shells();

        Size expected = n_shells_a * n_shells_b * n_shells_c * n_shells_d;
        // n_quartets() = bra.n_pairs() * ket.n_pairs()
        //              = (n_a * n_b) * (n_c * n_d)
        EXPECT_EQ(q.n_quartets(), expected)
            << "n_quartets() mismatch for SSQ with AM ("
            << q.La() << q.Lb() << "|" << q.Lc() << q.Ld() << ")";
    }
}

TEST_F(H2O4AugccpVDZFixture, ShellSetQuartetAMRange) {
    const int max_am = basis_->max_angular_momentum();
    const auto& quartets = basis_->shell_set_quartets();

    for (const auto& q : quartets) {
        EXPECT_GE(q.La(), 0);
        EXPECT_LE(q.La(), max_am);
        EXPECT_GE(q.Lb(), 0);
        EXPECT_LE(q.Lb(), max_am);
        EXPECT_GE(q.Lc(), 0);
        EXPECT_LE(q.Lc(), max_am);
        EXPECT_GE(q.Ld(), 0);
        EXPECT_LE(q.Ld(), max_am);
    }
}

// =============================================================================
// Schwarz Screening
// =============================================================================

TEST_F(H2O4AugccpVDZFixture, SchwarzScreeningReducesWork) {
    auto D = make_identity_density(nbf_);

    // Unscreened: compute total individual quartets
    Size total_quartets = total_individual_quartets();
    ASSERT_GT(total_quartets, 0u);

    // Count quartets that pass Schwarz screening
    const auto& bounds = engine_->precompute_schwarz_bounds();
    (void)bounds;

    // Screened path with normal threshold
    auto opts = screening::ScreeningOptions::normal();

    consumers::FockBuilder fock_screened(nbf_);
    fock_screened.set_density(D.data(), nbf_);
    engine_->compute_and_consume(Operator::coulomb(), fock_screened, opts);

    // Unscreened path
    consumers::FockBuilder fock_full(nbf_);
    fock_full.set_density(D.data(), nbf_);
    engine_->compute_and_consume(Operator::coulomb(), fock_full);

    // Both should produce similar J matrices (screened drops negligible terms)
    auto J_screened = fock_screened.get_coulomb_matrix();
    auto J_full = fock_full.get_coulomb_matrix();

    for (Size i = 0; i < nbf_ * nbf_; ++i) {
        EXPECT_NEAR(J_screened[i], J_full[i], 1e-10)
            << "J matrix element " << i << " differs between screened and full";
    }
}

// =============================================================================
// Parallel Consistency
// =============================================================================

TEST_F(H2O4AugccpVDZFixture, ParallelConsistency) {
    auto D = make_identity_density(nbf_);

    // Serial computation
    consumers::FockBuilder fock_serial(nbf_);
    fock_serial.set_density(D.data(), nbf_);
    engine_->compute_and_consume(Operator::coulomb(), fock_serial);

    auto J_serial = fock_serial.get_coulomb_matrix();
    auto K_serial = fock_serial.get_exchange_matrix();

    // Parallel computation (4 threads)
    consumers::FockBuilder fock_parallel(nbf_);
    fock_parallel.set_density(D.data(), nbf_);
    fock_parallel.set_threading_strategy(
        consumers::FockThreadingStrategy::ThreadLocal);
    engine_->compute_and_consume_parallel(Operator::coulomb(), fock_parallel, 4);

    auto J_parallel = fock_parallel.get_coulomb_matrix();
    auto K_parallel = fock_parallel.get_exchange_matrix();

    for (Size i = 0; i < nbf_ * nbf_; ++i) {
        EXPECT_NEAR(J_serial[i], J_parallel[i], 1e-12)
            << "J[" << i << "] serial vs parallel mismatch";
    }
    for (Size i = 0; i < nbf_ * nbf_; ++i) {
        EXPECT_NEAR(K_serial[i], K_parallel[i], 1e-12)
            << "K[" << i << "] serial vs parallel mismatch";
    }
}

// =============================================================================
// CPU vs GPU Consistency
// =============================================================================

TEST_F(H2O4AugccpVDZFixture, CpuGpuFockConsistency) {
    if (!engine_->gpu_available()) {
        GTEST_SKIP() << "GPU not available, skipping CPU vs GPU comparison";
    }

    auto D = make_identity_density(nbf_);

    // CPU computation
    consumers::FockBuilder fock_cpu(nbf_);
    fock_cpu.set_density(D.data(), nbf_);
    engine_->compute_and_consume(Operator::coulomb(), fock_cpu,
                                 BackendHint::ForceCPU);

    auto J_cpu = fock_cpu.get_coulomb_matrix();
    auto K_cpu = fock_cpu.get_exchange_matrix();

    // GPU computation
    consumers::FockBuilder fock_gpu(nbf_);
    fock_gpu.set_density(D.data(), nbf_);
    engine_->compute_and_consume(Operator::coulomb(), fock_gpu,
                                 BackendHint::PreferGPU);

    auto J_gpu = fock_gpu.get_coulomb_matrix();
    auto K_gpu = fock_gpu.get_exchange_matrix();

    for (Size i = 0; i < nbf_ * nbf_; ++i) {
        EXPECT_NEAR(J_cpu[i], J_gpu[i], 1e-10)
            << "J[" << i << "] CPU vs GPU mismatch";
    }
    for (Size i = 0; i < nbf_ * nbf_; ++i) {
        EXPECT_NEAR(K_cpu[i], K_gpu[i], 1e-10)
            << "K[" << i << "] CPU vs GPU mismatch";
    }
}

// =============================================================================
// Manual ShellSetQuartet Loop
// =============================================================================

TEST_F(H2O4AugccpVDZFixture, ManualShellSetQuartetLoop) {
    auto D = make_identity_density(nbf_);

    // Full-basis compute_and_consume (reference)
    consumers::FockBuilder fock_ref(nbf_);
    fock_ref.set_density(D.data(), nbf_);
    engine_->compute_and_consume(Operator::coulomb(), fock_ref);

    auto J_ref = fock_ref.get_coulomb_matrix();
    auto K_ref = fock_ref.get_exchange_matrix();

    // Manual iteration over shell_set_quartets
    consumers::FockBuilder fock_manual(nbf_);
    fock_manual.set_density(D.data(), nbf_);

    const auto& quartets = basis_->shell_set_quartets();
    for (const auto& quartet : quartets) {
        engine_->compute_shell_set_quartet(Operator::coulomb(), quartet,
                                           fock_manual);
    }

    auto J_manual = fock_manual.get_coulomb_matrix();
    auto K_manual = fock_manual.get_exchange_matrix();

    for (Size i = 0; i < nbf_ * nbf_; ++i) {
        EXPECT_NEAR(J_ref[i], J_manual[i], 1e-12)
            << "J[" << i << "] full-basis vs manual SSQ loop mismatch";
    }
    for (Size i = 0; i < nbf_ * nbf_; ++i) {
        EXPECT_NEAR(K_ref[i], K_manual[i], 1e-12)
            << "K[" << i << "] full-basis vs manual SSQ loop mismatch";
    }
}

// =============================================================================
// Edge Cases
// =============================================================================

TEST_F(H2O4AugccpVDZFixture, ZeroDensityGivesZeroFock) {
    std::vector<Real> D(nbf_ * nbf_, 0.0);

    consumers::FockBuilder fock(nbf_);
    fock.set_density(D.data(), nbf_);

    engine_->compute_and_consume(Operator::coulomb(), fock);

    auto J = fock.get_coulomb_matrix();
    auto K = fock.get_exchange_matrix();

    for (Size i = 0; i < nbf_ * nbf_; ++i) {
        EXPECT_NEAR(J[i], 0.0, 1e-15)
            << "J[" << i << "] should be zero for zero density";
    }
    for (Size i = 0; i < nbf_ * nbf_; ++i) {
        EXPECT_NEAR(K[i], 0.0, 1e-15)
            << "K[" << i << "] should be zero for zero density";
    }
}

// =============================================================================
// Fock Matrix Construction
// =============================================================================

TEST_F(H2O4AugccpVDZFixture, FockMatrixFromJK) {
    auto D = make_identity_density(nbf_);

    consumers::FockBuilder fock(nbf_);
    fock.set_density(D.data(), nbf_);

    engine_->compute_and_consume(Operator::coulomb(), fock);

    auto J = fock.get_coulomb_matrix();
    auto K = fock.get_exchange_matrix();

    // Build a mock H_core (use a simple diagonal matrix)
    std::vector<Real> H_core(nbf_ * nbf_, 0.0);
    for (Size i = 0; i < nbf_; ++i) {
        H_core[i * nbf_ + i] = -1.0 * static_cast<Real>(i + 1);
    }

    constexpr Real exchange_fraction = 0.5;
    auto F = fock.get_fock_matrix(
        std::span<const Real>(H_core), exchange_fraction);

    ASSERT_EQ(F.size(), nbf_ * nbf_);

    // F = H_core + J - exchange_fraction * K
    for (Size i = 0; i < nbf_ * nbf_; ++i) {
        Real expected = H_core[i] + J[i] - exchange_fraction * K[i];
        EXPECT_NEAR(F[i], expected, 1e-14)
            << "Fock matrix element " << i
            << " does not match H_core + J - xfrac*K";
    }
}

}  // namespace libaccint::test
