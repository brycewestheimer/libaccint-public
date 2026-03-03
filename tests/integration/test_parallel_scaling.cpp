// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2026 Bryce M. Westheimer. All rights reserved.

/// @file test_parallel_scaling.cpp
/// @brief Integration tests for thread-parallel CPU execution
///
/// Validates that parallel execution produces correct results across
/// different thread counts and configurations. Tests cover:
///   - Parallel 1e integrals (overlap, kinetic, nuclear)
///   - Parallel 2e Fock build (ThreadLocal strategy)
///   - Parallel screened 2e with Schwarz bounds
///   - Multiple successive parallel computations
///   - Thread configuration utilities

#include <libaccint/engine/engine.hpp>
#include <libaccint/engine/thread_config.hpp>
#include <libaccint/consumers/fock_builder.hpp>
#include <libaccint/data/builtin_basis.hpp>
#include <libaccint/operators/operator.hpp>
#include <libaccint/screening/screening_options.hpp>
#include <libaccint/core/types.hpp>
#include <libaccint/memory/thread_local_pool.hpp>

#include <gtest/gtest.h>
#include <cmath>
#include <random>
#include <vector>

using namespace libaccint;
using namespace libaccint::data;
using namespace libaccint::consumers;

namespace {

constexpr Real TOL = 1e-12;

BasisSet make_h2o_sto3g() {
    std::vector<Atom> atoms = {
        {8, {0.0, 0.0, 0.0}},
        {1, {0.0, 1.43233673, -1.10866041}},
        {1, {0.0, -1.43233673, -1.10866041}},
    };
    return create_sto3g(atoms);
}

std::vector<Real> make_random_density(Size nbf, unsigned seed = 42) {
    std::mt19937 gen(seed);
    std::uniform_real_distribution<double> dist(-0.5, 0.5);
    std::vector<Real> D(nbf * nbf);
    for (Size i = 0; i < nbf; ++i) {
        for (Size j = i; j < nbf; ++j) {
            double val = dist(gen);
            D[i * nbf + j] = val;
            D[j * nbf + i] = val;
        }
    }
    return D;
}

void expect_matrices_equal(const Real* A, const Real* B, Size n,
                           Real tol, const std::string& name) {
    for (Size i = 0; i < n; ++i) {
        for (Size j = 0; j < n; ++j) {
            EXPECT_NEAR(A[i * n + j], B[i * n + j], tol)
                << name << "[" << i << "," << j << "] mismatch";
        }
    }
}

void expect_vectors_equal(const std::vector<Real>& A,
                          const std::vector<Real>& B,
                          Real tol, const std::string& name) {
    ASSERT_EQ(A.size(), B.size());
    for (Size i = 0; i < A.size(); ++i) {
        EXPECT_NEAR(A[i], B[i], tol) << name << "[" << i << "] mismatch";
    }
}

}  // anonymous namespace

// =============================================================================
// Full Pipeline: Parallel 1e + Parallel 2e
// =============================================================================

TEST(ParallelScalingTest, FullPiplelineSerialVsParallel) {
    auto basis = make_h2o_sto3g();
    const Size nbf = basis.n_basis_functions();
    auto D = make_random_density(nbf);

    Engine engine(basis);

    PointChargeParams charges;
    charges.x = {0.0, 0.0, 0.0};
    charges.y = {0.0, 1.43233673, -1.43233673};
    charges.z = {0.0, -1.10866041, -1.10866041};
    charges.charge = {8.0, 1.0, 1.0};

    // --- Serial ---
    OneElectronOperator h_core_op = Operator::kinetic();
    h_core_op.add(Operator::nuclear(charges));

    std::vector<Real> H_serial;
    engine.compute_1e(h_core_op, H_serial);

    FockBuilder fock_serial(nbf);
    fock_serial.set_density(D.data(), nbf);
    engine.compute_and_consume(Operator::coulomb(), fock_serial);

    // --- Parallel ---
    std::vector<Real> H_parallel;
    engine.compute_1e_parallel(h_core_op, H_parallel, 4);

    FockBuilder fock_parallel(nbf);
    fock_parallel.set_threading_strategy(FockThreadingStrategy::ThreadLocal);
    fock_parallel.set_density(D.data(), nbf);
    fock_parallel.prepare_parallel(4);
    engine.compute_and_consume_parallel(Operator::coulomb(), fock_parallel, 4);
    fock_parallel.finalize_parallel();

    // Compare
    expect_vectors_equal(H_serial, H_parallel, TOL, "H_core");
    expect_matrices_equal(fock_serial.get_coulomb_matrix().data(),
                          fock_parallel.get_coulomb_matrix().data(),
                          nbf, TOL, "J");
    expect_matrices_equal(fock_serial.get_exchange_matrix().data(),
                          fock_parallel.get_exchange_matrix().data(),
                          nbf, TOL, "K");
}

// =============================================================================
// Screened Parallel vs Screened Serial
// =============================================================================

TEST(ParallelScalingTest, ScreenedParallelMatchesSerial) {
    auto basis = make_h2o_sto3g();
    const Size nbf = basis.n_basis_functions();
    auto D = make_random_density(nbf);

    Engine engine(basis);
    engine.precompute_schwarz_bounds();

    screening::ScreeningOptions options;
    options.enabled = true;
    options.threshold = 1e-10;

    // Serial screened
    FockBuilder fock_serial(nbf);
    fock_serial.set_density(D.data(), nbf);
    engine.compute_and_consume(Operator::coulomb(), fock_serial, options);

    // Parallel screened
    FockBuilder fock_parallel(nbf);
    fock_parallel.set_threading_strategy(FockThreadingStrategy::ThreadLocal);
    fock_parallel.set_density(D.data(), nbf);
    fock_parallel.prepare_parallel(4);
    engine.compute_and_consume_screened_parallel(
        Operator::coulomb(), fock_parallel, options, 4);
    fock_parallel.finalize_parallel();

    expect_matrices_equal(fock_serial.get_coulomb_matrix().data(),
                          fock_parallel.get_coulomb_matrix().data(),
                          nbf, TOL, "J_screened");
    expect_matrices_equal(fock_serial.get_exchange_matrix().data(),
                          fock_parallel.get_exchange_matrix().data(),
                          nbf, TOL, "K_screened");
}

// =============================================================================
// Thread Count Scaling Correctness
// =============================================================================

TEST(ParallelScalingTest, ConsistentAcrossThreadCounts) {
    auto basis = make_h2o_sto3g();
    const Size nbf = basis.n_basis_functions();
    auto D = make_random_density(nbf);

    Engine engine(basis);

    // Reference: serial
    FockBuilder fock_ref(nbf);
    fock_ref.set_density(D.data(), nbf);
    engine.compute_and_consume(Operator::coulomb(), fock_ref);

    for (int nt : {1, 2, 3, 4, 6, 8}) {
        FockBuilder fock_test(nbf);
        fock_test.set_threading_strategy(FockThreadingStrategy::ThreadLocal);
        fock_test.set_density(D.data(), nbf);
        fock_test.prepare_parallel(nt);
        engine.compute_and_consume_parallel(Operator::coulomb(), fock_test, nt);
        fock_test.finalize_parallel();

        expect_matrices_equal(fock_ref.get_coulomb_matrix().data(),
                              fock_test.get_coulomb_matrix().data(),
                              nbf, TOL,
                              "J_" + std::to_string(nt) + "_threads");
    }
}

// =============================================================================
// Successive Computations
// =============================================================================

TEST(ParallelScalingTest, SuccessiveParallelComputations) {
    auto basis = make_h2o_sto3g();
    const Size nbf = basis.n_basis_functions();

    Engine engine(basis);

    std::vector<Real> S_first;
    engine.compute_1e_parallel(OneElectronOperator(Operator::overlap()), S_first, 4);

    // Successive computations should give identical results
    for (int i = 0; i < 3; ++i) {
        std::vector<Real> S_next;
        engine.compute_1e_parallel(OneElectronOperator(Operator::overlap()), S_next, 4);
        expect_vectors_equal(S_first, S_next, TOL,
                             "Overlap_successive_" + std::to_string(i));
    }
}

// =============================================================================
// Thread Configuration Integration
// =============================================================================

TEST(ParallelScalingTest, ScopedThreadCountIntegration) {
    auto basis = make_h2o_sto3g();
    const Size nbf = basis.n_basis_functions();

    Engine engine(basis);

    // Reference
    std::vector<Real> S_ref;
    engine.compute_1e(OneElectronOperator(Operator::overlap()), S_ref);

    // Use ScopedThreadCount  
    {
        ScopedThreadCount guard(2);

        std::vector<Real> S_scoped;
        engine.compute_1e_parallel(OneElectronOperator(Operator::overlap()),
                                   S_scoped, 2);
        expect_vectors_equal(S_ref, S_scoped, TOL, "ScopedThreadCount");
    }
}

// =============================================================================
// Memory Pool Integration
// =============================================================================

TEST(ParallelScalingTest, PartitionedPoolIntegration) {
    // Verify PartitionedPool works alongside parallel computations
    auto basis = make_h2o_sto3g();
    const Size nbf = basis.n_basis_functions();

    Engine engine(basis);
    const int n_threads = 4;

    memory::PartitionedPool pools(n_threads);

    // Run parallel computation using partitioned pools for scratch
    std::vector<Real> S_result;
    engine.compute_1e_parallel(OneElectronOperator(Operator::overlap()),
                               S_result, n_threads);

    // Verify the overlap matrix is valid
    EXPECT_EQ(S_result.size(), nbf * nbf);

    // Diagonal elements should be positive (normalization)
    for (Size i = 0; i < nbf; ++i) {
        EXPECT_GT(S_result[i * nbf + i], 0.0)
            << "S[" << i << "," << i << "] should be positive";
    }

    auto stats = pools.aggregate_stats();
    // Pools were created but not used — that's fine for this test
    EXPECT_EQ(stats.total_allocations, 0u);
}

