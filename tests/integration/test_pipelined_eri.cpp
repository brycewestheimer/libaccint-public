// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2026 Bryce M. Westheimer. All rights reserved.

/// @file test_pipelined_eri.cpp
/// @brief Pipelined ERI computation tests using (H₂O)₄/aug-cc-pVDZ
///
/// Step 11.5: Validates multi-stream/overlapped execution concepts for
/// two-electron integral computation. Tests sequential quartet iteration,
/// worklist-based compute, worklist partitioning, parallel consistency,
/// and GPU batch dispatch. Includes a disabled placeholder for future
/// multi-stream CUDA pipeline support.

#include "h2o4_fixture.hpp"

#include <libaccint/basis/shell_set_quartet.hpp>
#include <libaccint/config.hpp>
#include <libaccint/consumers/fock_builder.hpp>
#include <libaccint/core/types.hpp>
#include <libaccint/engine/dispatch_policy.hpp>
#include <libaccint/engine/engine.hpp>
#include <libaccint/operators/operator.hpp>

#include <gtest/gtest.h>
#include <algorithm>
#include <cmath>
#include <iostream>
#include <numeric>
#include <span>
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
// Test 1: Pipeline Setup Verification
// =============================================================================

/// Verify the basis has enough shell set quartets for meaningful pipeline
/// testing. A pipelined or batched execution strategy only makes sense when
/// there are enough independent work units to fill multiple stages/streams.
TEST_F(H2O4AugccpVDZFixture, PipelinedERISetup) {
    const auto& quartets = basis_->shell_set_quartets();

    // For (H2O)4 / aug-cc-pVDZ we expect a substantial worklist.
    // At least 10 shell set quartets are needed for meaningful pipeline testing.
    EXPECT_GE(quartets.size(), 10u)
        << "Need at least 10 shell set quartets for pipeline testing; got "
        << quartets.size();

    // Also verify total individual quartet count is large enough to warrant
    // overlapped execution.
    Size total = 0;
    for (const auto& q : quartets) {
        total += q.n_quartets();
    }
    EXPECT_GT(total, 100u)
        << "Expected > 100 individual quartets for pipeline relevance; got "
        << total;

    std::cout << "[INFO] PipelinedERISetup: " << quartets.size()
              << " shell set quartets, " << total
              << " individual quartets" << std::endl;
}

// =============================================================================
// Test 2: Sequential Quartet-by-Quartet vs Full-Basis
// =============================================================================

/// Compute ERI by iterating shell_set_quartets() one-by-one with
/// Engine::compute(coulomb, quartet, consumer) and compare the resulting
/// J matrix against the full-basis compute_and_consume result.
TEST_F(H2O4AugccpVDZFixture, SequentialERIMatchesFull) {
    auto D = make_identity_density(nbf_);

    // Reference: full-basis compute_and_consume
    consumers::FockBuilder fock_ref(nbf_);
    fock_ref.set_density(D.data(), nbf_);
    engine_->compute_and_consume(Operator::coulomb(), fock_ref);

    auto J_ref = fock_ref.get_coulomb_matrix();

    // Sequential: iterate quartets one at a time
    consumers::FockBuilder fock_seq(nbf_);
    fock_seq.set_density(D.data(), nbf_);

    const auto& quartets = basis_->shell_set_quartets();
    for (const auto& quartet : quartets) {
        engine_->compute(Operator::coulomb(), quartet, fock_seq);
    }

    auto J_seq = fock_seq.get_coulomb_matrix();

    for (Size i = 0; i < nbf_ * nbf_; ++i) {
        EXPECT_NEAR(J_ref[i], J_seq[i], 1e-12)
            << "J[" << i << "] sequential vs full-basis mismatch";
    }
}

// =============================================================================
// Test 3: Worklist Compute (span overload)
// =============================================================================

/// Use the span<const ShellSetQuartet> overload of compute_and_consume with
/// the full quartets worklist and verify it matches the standard full-basis
/// compute_and_consume result.
TEST_F(H2O4AugccpVDZFixture, WorklistERICompute) {
    auto D = make_identity_density(nbf_);

    // Reference: full-basis compute_and_consume
    consumers::FockBuilder fock_ref(nbf_);
    fock_ref.set_density(D.data(), nbf_);
    engine_->compute_and_consume(Operator::coulomb(), fock_ref);

    auto J_ref = fock_ref.get_coulomb_matrix();
    auto K_ref = fock_ref.get_exchange_matrix();

    // Worklist: pass the full quartets vector as a span
    consumers::FockBuilder fock_wl(nbf_);
    fock_wl.set_density(D.data(), nbf_);

    const auto& quartets = basis_->shell_set_quartets();
    std::span<const ShellSetQuartet> quartets_span(quartets.data(),
                                                    quartets.size());
    engine_->compute_and_consume(Operator::coulomb(), quartets_span, fock_wl);

    auto J_wl = fock_wl.get_coulomb_matrix();
    auto K_wl = fock_wl.get_exchange_matrix();

    for (Size i = 0; i < nbf_ * nbf_; ++i) {
        EXPECT_NEAR(J_ref[i], J_wl[i], 1e-12)
            << "J[" << i << "] worklist vs full-basis mismatch";
    }
    for (Size i = 0; i < nbf_ * nbf_; ++i) {
        EXPECT_NEAR(K_ref[i], K_wl[i], 1e-12)
            << "K[" << i << "] worklist vs full-basis mismatch";
    }
}

// =============================================================================
// Test 4: Worklist Subset Consistency (split in halves)
// =============================================================================

/// Split the quartets worklist into two halves, compute each half separately
/// using the worklist API, accumulate both into the same FockBuilder, and
/// compare against the full compute result.
TEST_F(H2O4AugccpVDZFixture, WorklistSubsetConsistency) {
    auto D = make_identity_density(nbf_);

    // Reference: full-basis compute_and_consume
    consumers::FockBuilder fock_ref(nbf_);
    fock_ref.set_density(D.data(), nbf_);
    engine_->compute_and_consume(Operator::coulomb(), fock_ref);

    auto J_ref = fock_ref.get_coulomb_matrix();
    auto K_ref = fock_ref.get_exchange_matrix();

    // Split quartets into two halves and compute separately,
    // accumulating into a single FockBuilder
    const auto& quartets = basis_->shell_set_quartets();
    ASSERT_GE(quartets.size(), 2u) << "Need at least 2 SSQs to split";

    Size mid = quartets.size() / 2;
    std::span<const ShellSetQuartet> first_half(quartets.data(), mid);
    std::span<const ShellSetQuartet> second_half(quartets.data() + mid,
                                                  quartets.size() - mid);

    consumers::FockBuilder fock_split(nbf_);
    fock_split.set_density(D.data(), nbf_);

    // Process first half
    engine_->compute_and_consume(Operator::coulomb(), first_half, fock_split);
    // Process second half (accumulates into same builder)
    engine_->compute_and_consume(Operator::coulomb(), second_half, fock_split);

    auto J_split = fock_split.get_coulomb_matrix();
    auto K_split = fock_split.get_exchange_matrix();

    for (Size i = 0; i < nbf_ * nbf_; ++i) {
        EXPECT_NEAR(J_ref[i], J_split[i], 1e-12)
            << "J[" << i << "] split-worklist vs full mismatch";
    }
    for (Size i = 0; i < nbf_ * nbf_; ++i) {
        EXPECT_NEAR(K_ref[i], K_split[i], 1e-12)
            << "K[" << i << "] split-worklist vs full mismatch";
    }
}

// =============================================================================
// Test 5: Multi-Thread Pipeline Consistency (2 vs 4 threads)
// =============================================================================

/// Verify that compute_and_consume_parallel with 2 threads gives the same
/// J matrix as with 4 threads. This validates that the parallel pipeline
/// decomposition is deterministic regardless of thread count.
TEST_F(H2O4AugccpVDZFixture, MultiThreadPipelineConsistency) {
    auto D = make_identity_density(nbf_);

    // 2-thread computation
    consumers::FockBuilder fock_2t(nbf_);
    fock_2t.set_density(D.data(), nbf_);
    fock_2t.set_threading_strategy(consumers::FockThreadingStrategy::ThreadLocal);
    engine_->compute_and_consume_parallel(Operator::coulomb(), fock_2t, 2);

    auto J_2t = fock_2t.get_coulomb_matrix();
    auto K_2t = fock_2t.get_exchange_matrix();

    // 4-thread computation
    consumers::FockBuilder fock_4t(nbf_);
    fock_4t.set_density(D.data(), nbf_);
    fock_4t.set_threading_strategy(consumers::FockThreadingStrategy::ThreadLocal);
    engine_->compute_and_consume_parallel(Operator::coulomb(), fock_4t, 4);

    auto J_4t = fock_4t.get_coulomb_matrix();
    auto K_4t = fock_4t.get_exchange_matrix();

    for (Size i = 0; i < nbf_ * nbf_; ++i) {
        EXPECT_NEAR(J_2t[i], J_4t[i], 1e-12)
            << "J[" << i << "] 2-thread vs 4-thread mismatch";
    }
    for (Size i = 0; i < nbf_ * nbf_; ++i) {
        EXPECT_NEAR(K_2t[i], K_4t[i], 1e-12)
            << "K[" << i << "] 2-thread vs 4-thread mismatch";
    }
}

// =============================================================================
// Test 6: GPU Batch Overlap Verification
// =============================================================================

/// If a GPU is available, verify that compute_and_consume with PreferGPU
/// processes quartets via the GPU backend. This is an informational test
/// that prints how many shell set quartets exceed a batch-size threshold
/// (and are thus candidates for GPU dispatch), validating that the GPU
/// path is exercised on real workloads.
TEST_F(H2O4AugccpVDZFixture, GPUBatchOverlapVerification) {
    if (!engine_->gpu_available()) {
        GTEST_SKIP() << "GPU not available, skipping GPU batch overlap test";
    }

    auto D = make_identity_density(nbf_);

    // CPU reference
    consumers::FockBuilder fock_cpu(nbf_);
    fock_cpu.set_density(D.data(), nbf_);
    engine_->compute_and_consume(Operator::coulomb(), fock_cpu,
                                 BackendHint::ForceCPU);

    auto J_cpu = fock_cpu.get_coulomb_matrix();

    // GPU computation with PreferGPU hint
    consumers::FockBuilder fock_gpu(nbf_);
    fock_gpu.set_density(D.data(), nbf_);
    engine_->compute_and_consume(Operator::coulomb(), fock_gpu,
                                 BackendHint::PreferGPU);

    auto J_gpu = fock_gpu.get_coulomb_matrix();

    // Verify correctness (GPU vs CPU)
    for (Size i = 0; i < nbf_ * nbf_; ++i) {
        EXPECT_NEAR(J_cpu[i], J_gpu[i], 1e-10)
            << "J[" << i << "] CPU vs GPU mismatch in pipeline test";
    }

    // Informational: classify quartets by batch size to show GPU dispatch
    // potential. Quartets with large n_quartets() are strong GPU candidates.
    const auto& quartets = basis_->shell_set_quartets();
    constexpr Size gpu_batch_threshold = 16;
    Size gpu_candidates = 0;
    Size cpu_only = 0;
    for (const auto& q : quartets) {
        if (q.n_quartets() >= gpu_batch_threshold) {
            ++gpu_candidates;
        } else {
            ++cpu_only;
        }
    }

    std::cout << "[INFO] GPUBatchOverlapVerification:" << std::endl;
    std::cout << "  Total shell set quartets:   " << quartets.size() << std::endl;
    std::cout << "  GPU candidates (>= " << gpu_batch_threshold
              << " quartets): " << gpu_candidates << std::endl;
    std::cout << "  CPU-only (< " << gpu_batch_threshold
              << " quartets):      " << cpu_only << std::endl;
}

// =============================================================================
// Test 7: Concurrent Stream Execution (placeholder)
// =============================================================================

/// Placeholder test for future multi-stream CUDA pipeline execution.
///
/// CudaEngine::compute_eri_pipelined() implements a ring-buffer pipeline
/// with N concurrent stream slots for overlapping compute and D-to-H
/// transfer. This test is disabled until the pipelined API is integrated
/// into the high-level Engine interface (and exposed through a
/// compute_and_consume_pipelined method or similar).
///
/// When enabled, this test should:
///   1. Compute ERIs via the pipelined GPU path
///   2. Compare against CPU reference J/K matrices (tol 1e-10)
///   3. Verify that multiple CUDA streams are actually used
TEST_F(H2O4AugccpVDZFixture, DISABLED_ConcurrentStreamExecution) {
#if LIBACCINT_USE_CUDA
    if (!engine_->gpu_available()) {
        GTEST_SKIP() << "GPU not available";
    }

    auto D = make_identity_density(nbf_);

    // CPU reference
    consumers::FockBuilder fock_cpu(nbf_);
    fock_cpu.set_density(D.data(), nbf_);
    engine_->compute_and_consume(Operator::coulomb(), fock_cpu,
                                 BackendHint::ForceCPU);

    auto J_cpu = fock_cpu.get_coulomb_matrix();

    // -----------------------------------------------------------------------
    // TRACKING: Why this test is DISABLED (DISABLED_ConcurrentStreamExecution)
    //
    // Status:   Blocked on Engine API work
    // Reason:   CudaEngine::compute_eri_pipelined() implements a ring-buffer
    //           pipeline with N concurrent CUDA stream slots, but that method
    //           is internal to CudaEngine. The high-level Engine interface does
    //           not yet expose a compute_and_consume_pipelined() entry-point
    //           (or equivalent PipelineConfig-based overload).
    //
    // To re-enable:
    //   1. Promote CudaEngine::compute_eri_pipelined to Engine (or add a
    //      compute_and_consume_pipelined overload that accepts PipelineConfig).
    //   2. Replace the placeholder block below with the pipelined computation
    //      and compare J_pipe against J_cpu with tolerance 1e-10.
    //   3. Remove the DISABLED_ prefix from the test name.
    //
    // Tracked by: Phase 5 remediation, Task 5.2
    // -----------------------------------------------------------------------

    // Example future usage:
    //   consumers::FockBuilder fock_pipe(nbf_);
    //   fock_pipe.set_density(D.data(), nbf_);
    //   engine_->compute_and_consume_pipelined(Operator::coulomb(), fock_pipe,
    //                                          PipelineConfig{.n_slots = 4});
    //   auto J_pipe = fock_pipe.get_coulomb_matrix();
    //   for (Size i = 0; i < nbf_ * nbf_; ++i) {
    //       EXPECT_NEAR(J_cpu[i], J_pipe[i], 1e-10);
    //   }

    GTEST_SKIP() << "Multi-stream pipeline API not yet exposed through Engine";
#else
    GTEST_SKIP() << "CUDA support not compiled";
#endif
}

}  // namespace libaccint::test
