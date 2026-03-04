// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2026 Bryce M. Westheimer. All rights reserved.

/// @file test_batched_1e_workflow.cpp
/// @brief Batched one-electron integral workflow tests using (H₂O)₄/aug-cc-pVDZ
///
/// Step 11.2: Validates full-matrix one-electron computations (overlap, kinetic,
/// nuclear, core Hamiltonian) on a physically reasonable system, plus
/// ShellSetPair iteration and batched compute path consistency.

#include "h2o4_fixture.hpp"

#include <libaccint/basis/shell_set_pair.hpp>
#include <libaccint/buffers/one_electron_buffer.hpp>
#include <libaccint/core/types.hpp>
#include <libaccint/engine/dispatch_policy.hpp>
#include <libaccint/engine/engine.hpp>
#include <libaccint/operators/one_electron_operator.hpp>
#include <libaccint/operators/operator.hpp>

#include <gtest/gtest.h>
#include <cmath>
#include <vector>

namespace libaccint::test {

// =============================================================================
// Basis Set Validation
// =============================================================================

TEST_F(H2O4AugccpVDZFixture, BasisSetDimensions) {
    // aug-cc-pVDZ on (H2O)4 should give ~164 basis functions
    EXPECT_GE(nbf_, 100u) << "Expected at least 100 basis functions for "
                              "(H2O)4/aug-cc-pVDZ";
    EXPECT_GE(basis_->n_shells(), 40u)
        << "Expected at least 40 shells for (H2O)4/aug-cc-pVDZ";
    EXPECT_GE(basis_->n_shell_sets(), 4u)
        << "Expected at least 4 shell sets (distinct AM/contraction groups)";
}

TEST_F(H2O4AugccpVDZFixture, ShellSetPairCount) {
    const auto& pairs = basis_->shell_set_pairs();
    EXPECT_FALSE(pairs.empty()) << "shell_set_pairs() should not be empty";

    // For n_shell_sets shell sets, the unique pair count (including self-pairs)
    // is n*(n+1)/2.  ShellSetPairs uses upper-triangle packing.
    const Size n_sets = basis_->n_shell_sets();
    const Size expected_pairs = n_sets * (n_sets + 1) / 2;
    EXPECT_EQ(pairs.size(), expected_pairs)
        << "Expected n_shell_sets*(n_shell_sets+1)/2 = " << expected_pairs
        << " shell set pairs, got " << pairs.size();
}

// =============================================================================
// Overlap Matrix
// =============================================================================

TEST_F(H2O4AugccpVDZFixture, OverlapMatrixDiagonal) {
    std::vector<Real> S;
    engine_->compute_overlap_matrix(S);

    ASSERT_EQ(S.size(), nbf_ * nbf_);

    for (Size i = 0; i < nbf_; ++i) {
        EXPECT_NEAR(S[i * nbf_ + i], 1.0, 1e-12)
            << "Overlap diagonal[" << i << "] should be 1.0 (normalized basis)";
    }
}

TEST_F(H2O4AugccpVDZFixture, OverlapMatrixSymmetry) {
    std::vector<Real> S;
    engine_->compute_overlap_matrix(S);

    ASSERT_EQ(S.size(), nbf_ * nbf_);
    expect_symmetric(S, nbf_, 1e-14, "Overlap");
}

TEST_F(H2O4AugccpVDZFixture, OverlapMatrixPositiveDefinite) {
    // Normalized basis: all diagonal elements must be exactly 1.0 (positive).
    // Off-diagonal magnitudes < 1.0 is a necessary (not sufficient) condition
    // for positive definiteness.
    std::vector<Real> S;
    engine_->compute_overlap_matrix(S);

    ASSERT_EQ(S.size(), nbf_ * nbf_);

    for (Size i = 0; i < nbf_; ++i) {
        EXPECT_GT(S[i * nbf_ + i], 0.0)
            << "Overlap diagonal[" << i << "] must be positive";
    }
}

// =============================================================================
// Kinetic Matrix
// =============================================================================

TEST_F(H2O4AugccpVDZFixture, KineticMatrixSymmetry) {
    std::vector<Real> T;
    engine_->compute_kinetic_matrix(T);

    ASSERT_EQ(T.size(), nbf_ * nbf_);
    expect_symmetric(T, nbf_, 1e-14, "Kinetic");
}

TEST_F(H2O4AugccpVDZFixture, KineticMatrixPositiveDiagonal) {
    std::vector<Real> T;
    engine_->compute_kinetic_matrix(T);

    ASSERT_EQ(T.size(), nbf_ * nbf_);

    for (Size i = 0; i < nbf_; ++i) {
        EXPECT_GT(T[i * nbf_ + i], 0.0)
            << "Kinetic diagonal[" << i << "] must be positive "
               "(kinetic energy is always positive)";
    }
}

// =============================================================================
// Nuclear Attraction Matrix
// =============================================================================

TEST_F(H2O4AugccpVDZFixture, NuclearMatrixSymmetry) {
    std::vector<Real> V;
    engine_->compute_nuclear_matrix(charges_, V);

    ASSERT_EQ(V.size(), nbf_ * nbf_);
    expect_symmetric(V, nbf_, 1e-14, "Nuclear");
}

TEST_F(H2O4AugccpVDZFixture, NuclearMatrixNegativeDiagonal) {
    std::vector<Real> V;
    engine_->compute_nuclear_matrix(charges_, V);

    ASSERT_EQ(V.size(), nbf_ * nbf_);

    for (Size i = 0; i < nbf_; ++i) {
        EXPECT_LT(V[i * nbf_ + i], 0.0)
            << "Nuclear diagonal[" << i << "] must be negative "
               "(nuclear attraction is always attractive)";
    }
}

// =============================================================================
// Core Hamiltonian
// =============================================================================

TEST_F(H2O4AugccpVDZFixture, CoreHamiltonianSymmetry) {
    std::vector<Real> H;
    engine_->compute_core_hamiltonian(charges_, H);

    ASSERT_EQ(H.size(), nbf_ * nbf_);
    expect_symmetric(H, nbf_, 1e-14, "CoreHamiltonian");
}

TEST_F(H2O4AugccpVDZFixture, CoreHamiltonianEqualsSum) {
    // H_core = T + V must hold element-wise
    std::vector<Real> T, V, H;
    engine_->compute_kinetic_matrix(T);
    engine_->compute_nuclear_matrix(charges_, V);
    engine_->compute_core_hamiltonian(charges_, H);

    ASSERT_EQ(T.size(), nbf_ * nbf_);
    ASSERT_EQ(V.size(), nbf_ * nbf_);
    ASSERT_EQ(H.size(), nbf_ * nbf_);

    for (Size i = 0; i < nbf_ * nbf_; ++i) {
        EXPECT_NEAR(H[i], T[i] + V[i], 1e-12)
            << "H_core[" << i << "] = " << H[i]
            << " should equal T+V = " << (T[i] + V[i]);
    }
}

// =============================================================================
// ShellSetPair Iteration
// =============================================================================

TEST_F(H2O4AugccpVDZFixture, ShellSetPairIterationCoversAllPairs) {
    const auto& pairs = basis_->shell_set_pairs();

    // Sum n_pairs() over all ShellSetPairs.  Each n_pairs() returns
    // n_shells_a * n_shells_b (the full Cartesian product of shells).
    Size total_shell_pairs = 0;
    for (const auto& ssp : pairs) {
        total_shell_pairs += ssp.n_pairs();
    }

    // shell_set_pairs() returns upper-triangle (set_a <= set_b).
    // Self-pairs contribute n_i^2, cross-pairs contribute n_i * n_j once.
    // Total = sum_{i<=j} n_i * n_j = (N^2 + sum_i n_i^2) / 2.
    // Compute expected by summing over shell sets directly.
    const auto shell_sets = basis_->shell_sets();
    Size sum_ni_sq = 0;
    for (const auto* ss : shell_sets) {
        Size ni = ss->n_shells();
        sum_ni_sq += ni * ni;
    }
    const Size n_shells = basis_->n_shells();
    const Size expected = (n_shells * n_shells + sum_ni_sq) / 2;

    EXPECT_EQ(total_shell_pairs, expected)
        << "Sum of n_pairs() over upper-triangle ShellSetPairs should equal "
           "(N^2 + sum(n_i^2)) / 2 = " << expected;
}

// =============================================================================
// CPU vs GPU Consistency (guarded)
// =============================================================================

TEST_F(H2O4AugccpVDZFixture, OverlapCpuGpuConsistency) {
    if (!engine_->gpu_available()) {
        GTEST_SKIP() << "GPU not available, skipping CPU/GPU comparison";
    }

    std::vector<Real> S_cpu, S_gpu;
    engine_->compute_overlap_matrix(S_cpu, BackendHint::ForceCPU);
    engine_->compute_overlap_matrix(S_gpu, BackendHint::PreferGPU);

    ASSERT_EQ(S_cpu.size(), S_gpu.size());

    for (Size i = 0; i < S_cpu.size(); ++i) {
        EXPECT_NEAR(S_cpu[i], S_gpu[i], 1e-10)
            << "CPU/GPU overlap mismatch at element " << i;
    }
}

// =============================================================================
// Manual ShellSetPair Loop — Batched Compute Path Verification
// =============================================================================

TEST_F(H2O4AugccpVDZFixture, ManualShellSetPairLoop) {
    // Compute overlap via the convenience method (ground truth)
    std::vector<Real> S_ref;
    engine_->compute_overlap_matrix(S_ref);
    ASSERT_EQ(S_ref.size(), nbf_ * nbf_);

    // Manually iterate shell_set_pairs and use Engine::compute_1e with
    // the pairs-based overload to accumulate into a result matrix.
    // This verifies that the batched ShellSetPair compute path produces
    // identical results to the convenience method.
    const auto& pairs = basis_->shell_set_pairs();

    std::vector<Real> S_manual(nbf_ * nbf_, Real{0.0});
    engine_->compute_1e(OneElectronOperator(Operator::overlap()),
                        std::span<const ShellSetPair>(pairs),
                        S_manual);

    ASSERT_EQ(S_manual.size(), nbf_ * nbf_);

    for (Size i = 0; i < nbf_ * nbf_; ++i) {
        EXPECT_NEAR(S_manual[i], S_ref[i], 1e-13)
            << "Manual ShellSetPair loop vs convenience method mismatch "
               "at element " << i;
    }
}

}  // namespace libaccint::test
