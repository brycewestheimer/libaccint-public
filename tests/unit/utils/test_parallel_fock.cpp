// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2026 Bryce M. Westheimer. All rights reserved.

/// @file test_parallel_fock.cpp
/// @brief Unit tests for OpenMP parallel Fock matrix construction correctness

#include <libaccint/engine/engine.hpp>
#include <libaccint/consumers/fock_builder.hpp>
#include <libaccint/data/builtin_basis.hpp>
#include <libaccint/operators/operator.hpp>
#include <libaccint/core/types.hpp>

#include <gtest/gtest.h>
#include <cmath>
#include <vector>
#include <random>

using namespace libaccint;
using namespace libaccint::data;
using namespace libaccint::consumers;

namespace {

constexpr Real FOCK_TOL = 1e-12;

/// H2O geometry in Bohr
BasisSet make_h2o_sto3g() {
    std::vector<Atom> atoms = {
        {8, {0.0, 0.0, 0.0}},
        {1, {0.0, 1.43233673, -1.10866041}},
        {1, {0.0, -1.43233673, -1.10866041}},
    };
    return create_sto3g(atoms);
}

/// Generate a random symmetric density matrix
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

/// Check if two matrices are equal within tolerance
void expect_matrices_equal(const Real* A, const Real* B, Size n,
                           Real tol, const std::string& name) {
    for (Size i = 0; i < n; ++i) {
        for (Size j = 0; j < n; ++j) {
            EXPECT_NEAR(A[i * n + j], B[i * n + j], tol)
                << name << "[" << i << "," << j << "] mismatch";
        }
    }
}

/// Check matrix symmetry
void expect_matrix_symmetric(const Real* A, Size n, Real tol) {
    for (Size i = 0; i < n; ++i) {
        for (Size j = i + 1; j < n; ++j) {
            EXPECT_NEAR(A[i * n + j], A[j * n + i], tol)
                << "Symmetry violation at [" << i << "," << j << "]";
        }
    }
}

}  // anonymous namespace

// =============================================================================
// OpenMP Correctness Tests
// =============================================================================

TEST(ParallelFockTest, AtomicMatchesSequential) {
    auto basis = make_h2o_sto3g();
    const Size nbf = basis.n_basis_functions();
    auto D = make_random_density(nbf);

    Engine engine(basis);

    // Sequential computation
    FockBuilder fock_seq(nbf);
    fock_seq.set_threading_strategy(FockThreadingStrategy::Sequential);
    fock_seq.set_density(D.data(), nbf);
    engine.compute_and_consume(Operator::coulomb(), fock_seq);

    // Parallel with Atomic strategy
    FockBuilder fock_atomic(nbf);
    fock_atomic.set_threading_strategy(FockThreadingStrategy::Atomic);
    fock_atomic.set_density(D.data(), nbf);
    engine.compute_and_consume_parallel(Operator::coulomb(), fock_atomic, 4);

    // Compare J matrices
    expect_matrices_equal(fock_seq.get_coulomb_matrix().data(),
                          fock_atomic.get_coulomb_matrix().data(),
                          nbf, FOCK_TOL, "J_atomic");

    // Compare K matrices
    expect_matrices_equal(fock_seq.get_exchange_matrix().data(),
                          fock_atomic.get_exchange_matrix().data(),
                          nbf, FOCK_TOL, "K_atomic");
}

TEST(ParallelFockTest, ThreadLocalMatchesSequential) {
    auto basis = make_h2o_sto3g();
    const Size nbf = basis.n_basis_functions();
    auto D = make_random_density(nbf);

    Engine engine(basis);

    // Sequential computation
    FockBuilder fock_seq(nbf);
    fock_seq.set_threading_strategy(FockThreadingStrategy::Sequential);
    fock_seq.set_density(D.data(), nbf);
    engine.compute_and_consume(Operator::coulomb(), fock_seq);

    // Parallel with ThreadLocal strategy
    FockBuilder fock_tl(nbf);
    fock_tl.set_threading_strategy(FockThreadingStrategy::ThreadLocal);
    fock_tl.set_density(D.data(), nbf);
    fock_tl.prepare_parallel(4);
    engine.compute_and_consume_parallel(Operator::coulomb(), fock_tl, 4);
    fock_tl.finalize_parallel();

    // Compare J matrices
    expect_matrices_equal(fock_seq.get_coulomb_matrix().data(),
                          fock_tl.get_coulomb_matrix().data(),
                          nbf, FOCK_TOL, "J_threadlocal");

    // Compare K matrices
    expect_matrices_equal(fock_seq.get_exchange_matrix().data(),
                          fock_tl.get_exchange_matrix().data(),
                          nbf, FOCK_TOL, "K_threadlocal");
}

TEST(ParallelFockTest, AtomicMatchesThreadLocal) {
    auto basis = make_h2o_sto3g();
    const Size nbf = basis.n_basis_functions();
    auto D = make_random_density(nbf);

    Engine engine(basis);

    // Atomic strategy
    FockBuilder fock_atomic(nbf);
    fock_atomic.set_threading_strategy(FockThreadingStrategy::Atomic);
    fock_atomic.set_density(D.data(), nbf);
    engine.compute_and_consume_parallel(Operator::coulomb(), fock_atomic, 4);

    // ThreadLocal strategy
    FockBuilder fock_tl(nbf);
    fock_tl.set_threading_strategy(FockThreadingStrategy::ThreadLocal);
    fock_tl.set_density(D.data(), nbf);
    fock_tl.prepare_parallel(4);
    engine.compute_and_consume_parallel(Operator::coulomb(), fock_tl, 4);
    fock_tl.finalize_parallel();

    // Compare J matrices
    expect_matrices_equal(fock_atomic.get_coulomb_matrix().data(),
                          fock_tl.get_coulomb_matrix().data(),
                          nbf, FOCK_TOL, "J_atomic_vs_tl");

    // Compare K matrices
    expect_matrices_equal(fock_atomic.get_exchange_matrix().data(),
                          fock_tl.get_exchange_matrix().data(),
                          nbf, FOCK_TOL, "K_atomic_vs_tl");
}

TEST(ParallelFockTest, SymmetryPreserved_Atomic) {
    auto basis = make_h2o_sto3g();
    const Size nbf = basis.n_basis_functions();
    auto D = make_random_density(nbf);

    Engine engine(basis);

    FockBuilder fock(nbf);
    fock.set_threading_strategy(FockThreadingStrategy::Atomic);
    fock.set_density(D.data(), nbf);
    engine.compute_and_consume_parallel(Operator::coulomb(), fock, 8);

    expect_matrix_symmetric(fock.get_coulomb_matrix().data(), nbf, FOCK_TOL);
    expect_matrix_symmetric(fock.get_exchange_matrix().data(), nbf, FOCK_TOL);
}

TEST(ParallelFockTest, SymmetryPreserved_ThreadLocal) {
    auto basis = make_h2o_sto3g();
    const Size nbf = basis.n_basis_functions();
    auto D = make_random_density(nbf);

    Engine engine(basis);

    FockBuilder fock(nbf);
    fock.set_threading_strategy(FockThreadingStrategy::ThreadLocal);
    fock.set_density(D.data(), nbf);
    fock.prepare_parallel(8);
    engine.compute_and_consume_parallel(Operator::coulomb(), fock, 8);
    fock.finalize_parallel();

    expect_matrix_symmetric(fock.get_coulomb_matrix().data(), nbf, FOCK_TOL);
    expect_matrix_symmetric(fock.get_exchange_matrix().data(), nbf, FOCK_TOL);
}

TEST(ParallelFockTest, VaryingThreadCounts) {
    auto basis = make_h2o_sto3g();
    const Size nbf = basis.n_basis_functions();
    auto D = make_random_density(nbf);

    Engine engine(basis);

    // Reference: single-threaded
    FockBuilder fock_ref(nbf);
    fock_ref.set_density(D.data(), nbf);
    engine.compute_and_consume_parallel(Operator::coulomb(), fock_ref, 1);

    // Test with various thread counts
    for (int n_threads : {2, 3, 4, 7, 8, 16}) {
        FockBuilder fock_test(nbf);
        fock_test.set_threading_strategy(FockThreadingStrategy::ThreadLocal);
        fock_test.set_density(D.data(), nbf);
        fock_test.prepare_parallel(n_threads);
        engine.compute_and_consume_parallel(Operator::coulomb(), fock_test, n_threads);
        fock_test.finalize_parallel();

        expect_matrices_equal(fock_ref.get_coulomb_matrix().data(),
                              fock_test.get_coulomb_matrix().data(),
                              nbf, FOCK_TOL, "J_" + std::to_string(n_threads) + "_threads");
    }
}

TEST(ParallelFockTest, RepeatedComputations) {
    // Verify that repeated parallel computations give identical results
    auto basis = make_h2o_sto3g();
    const Size nbf = basis.n_basis_functions();
    auto D = make_random_density(nbf);

    Engine engine(basis);

    std::vector<Real> J_prev(nbf * nbf);

    for (int iter = 0; iter < 5; ++iter) {
        FockBuilder fock(nbf);
        fock.set_threading_strategy(FockThreadingStrategy::Atomic);
        fock.set_density(D.data(), nbf);
        engine.compute_and_consume_parallel(Operator::coulomb(), fock, 4);

        if (iter > 0) {
            expect_matrices_equal(J_prev.data(),
                                  fock.get_coulomb_matrix().data(),
                                  nbf, FOCK_TOL, "J_iter_" + std::to_string(iter));
        }

        std::copy(fock.get_coulomb_matrix().begin(),
                  fock.get_coulomb_matrix().end(),
                  J_prev.begin());
    }
}

TEST(ParallelFockTest, DifferentDensities) {
    auto basis = make_h2o_sto3g();
    const Size nbf = basis.n_basis_functions();

    Engine engine(basis);

    // Test with different random densities
    for (unsigned seed : {1, 42, 123, 999}) {
        auto D = make_random_density(nbf, seed);

        // Sequential
        FockBuilder fock_seq(nbf);
        fock_seq.set_density(D.data(), nbf);
        engine.compute_and_consume(Operator::coulomb(), fock_seq);

        // Parallel
        FockBuilder fock_par(nbf);
        fock_par.set_threading_strategy(FockThreadingStrategy::ThreadLocal);
        fock_par.set_density(D.data(), nbf);
        fock_par.prepare_parallel(4);
        engine.compute_and_consume_parallel(Operator::coulomb(), fock_par, 4);
        fock_par.finalize_parallel();

        expect_matrices_equal(fock_seq.get_coulomb_matrix().data(),
                              fock_par.get_coulomb_matrix().data(),
                              nbf, FOCK_TOL, "J_seed_" + std::to_string(seed));
    }
}
