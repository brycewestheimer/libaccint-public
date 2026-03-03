// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2026 Bryce M. Westheimer. All rights reserved.

/// @file test_parallel_1e.cpp
/// @brief Tests for parallel one-electron integral computation
///
/// Validates that OpenMP-parallelized compute_1e_parallel produces results
/// matching sequential compute_1e to full floating-point precision.

#include <libaccint/engine/engine.hpp>
#include <libaccint/engine/thread_config.hpp>
#include <libaccint/data/builtin_basis.hpp>
#include <libaccint/operators/operator.hpp>
#include <libaccint/core/types.hpp>

#include <gtest/gtest.h>
#include <cmath>
#include <vector>

using namespace libaccint;
using namespace libaccint::data;

namespace {

constexpr Real TOL = 1e-12;

/// H2O geometry in Bohr
BasisSet make_h2o_sto3g() {
    std::vector<Atom> atoms = {
        {8, {0.0, 0.0, 0.0}},
        {1, {0.0, 1.43233673, -1.10866041}},
        {1, {0.0, -1.43233673, -1.10866041}},
    };
    return create_sto3g(atoms);
}

/// Compare two matrices element-wise
void expect_matrices_equal(const std::vector<Real>& A,
                           const std::vector<Real>& B,
                           Size nbf, Real tol,
                           const std::string& name) {
    ASSERT_EQ(A.size(), B.size());
    for (Size i = 0; i < nbf; ++i) {
        for (Size j = 0; j < nbf; ++j) {
            EXPECT_NEAR(A[i * nbf + j], B[i * nbf + j], tol)
                << name << "[" << i << "," << j << "] mismatch";
        }
    }
}

/// Check matrix symmetry
void expect_matrix_symmetric(const std::vector<Real>& A,
                             Size nbf, Real tol) {
    for (Size i = 0; i < nbf; ++i) {
        for (Size j = i + 1; j < nbf; ++j) {
            EXPECT_NEAR(A[i * nbf + j], A[j * nbf + i], tol)
                << "Asymmetry at [" << i << "," << j << "]";
        }
    }
}

}  // anonymous namespace

// =============================================================================
// Parallel 1e vs Serial Comparison
// =============================================================================

TEST(Parallel1eTest, OverlapMatchesSerial) {
    auto basis = make_h2o_sto3g();
    const Size nbf = basis.n_basis_functions();

    Engine engine(basis);
    OneElectronOperator overlap_op(Operator::overlap());

    // Serial computation
    std::vector<Real> S_serial;
    engine.compute_1e(overlap_op, S_serial);

    // Parallel computation
    std::vector<Real> S_parallel;
    engine.compute_1e_parallel(overlap_op, S_parallel, 4);

    expect_matrices_equal(S_serial, S_parallel, nbf, TOL, "Overlap");
}

TEST(Parallel1eTest, KineticMatchesSerial) {
    auto basis = make_h2o_sto3g();
    const Size nbf = basis.n_basis_functions();

    Engine engine(basis);
    OneElectronOperator kinetic_op(Operator::kinetic());

    std::vector<Real> T_serial;
    engine.compute_1e(kinetic_op, T_serial);

    std::vector<Real> T_parallel;
    engine.compute_1e_parallel(kinetic_op, T_parallel, 4);

    expect_matrices_equal(T_serial, T_parallel, nbf, TOL, "Kinetic");
}

TEST(Parallel1eTest, NuclearMatchesSerial) {
    auto basis = make_h2o_sto3g();
    const Size nbf = basis.n_basis_functions();

    PointChargeParams charges;
    charges.x = {0.0, 0.0, 0.0};
    charges.y = {0.0, 1.43233673, -1.43233673};
    charges.z = {0.0, -1.10866041, -1.10866041};
    charges.charge = {8.0, 1.0, 1.0};

    Engine engine(basis);
    OneElectronOperator nuclear_op(Operator::nuclear(charges));

    std::vector<Real> V_serial;
    engine.compute_1e(nuclear_op, V_serial);

    std::vector<Real> V_parallel;
    engine.compute_1e_parallel(nuclear_op, V_parallel, 4);

    expect_matrices_equal(V_serial, V_parallel, nbf, TOL, "Nuclear");
}

TEST(Parallel1eTest, CoreHamiltonianMatchesSerial) {
    auto basis = make_h2o_sto3g();
    const Size nbf = basis.n_basis_functions();

    PointChargeParams charges;
    charges.x = {0.0, 0.0, 0.0};
    charges.y = {0.0, 1.43233673, -1.43233673};
    charges.z = {0.0, -1.10866041, -1.10866041};
    charges.charge = {8.0, 1.0, 1.0};

    Engine engine(basis);

    // Composed operator: T + V
    OneElectronOperator h_core = Operator::kinetic();
    h_core.add(Operator::nuclear(charges));

    std::vector<Real> H_serial;
    engine.compute_1e(h_core, H_serial);

    std::vector<Real> H_parallel;
    engine.compute_1e_parallel(h_core, H_parallel, 4);

    expect_matrices_equal(H_serial, H_parallel, nbf, TOL, "H_core");
}

TEST(Parallel1eTest, SymmetryPreservedParallel) {
    auto basis = make_h2o_sto3g();
    const Size nbf = basis.n_basis_functions();

    Engine engine(basis);
    OneElectronOperator overlap_op(Operator::overlap());

    std::vector<Real> S_parallel;
    engine.compute_1e_parallel(overlap_op, S_parallel, 8);

    expect_matrix_symmetric(S_parallel, nbf, TOL);
}

TEST(Parallel1eTest, PrimaryApiMatchesExplicitParallelWhenThreadsConfigured) {
    auto basis = make_h2o_sto3g();
    const Size nbf = basis.n_basis_functions();

    Engine engine(basis);
    OneElectronOperator overlap_op(Operator::overlap());

    std::vector<Real> S_primary;
    {
        ScopedThreadCount thread_guard(4);
        engine.compute_1e(overlap_op, S_primary);
    }

    std::vector<Real> S_parallel;
    engine.compute_1e_parallel(overlap_op, S_parallel, 4);

    expect_matrices_equal(S_primary, S_parallel, nbf, TOL, "PrimaryApiParallel");
}

TEST(Parallel1eTest, VaryingThreadCounts) {
    auto basis = make_h2o_sto3g();
    const Size nbf = basis.n_basis_functions();

    Engine engine(basis);
    OneElectronOperator overlap_op(Operator::overlap());

    // Reference: serial
    std::vector<Real> S_ref;
    engine.compute_1e(overlap_op, S_ref);

    // Test with various thread counts
    for (int n_threads : {1, 2, 3, 4, 7, 8}) {
        std::vector<Real> S_test;
        engine.compute_1e_parallel(overlap_op, S_test, n_threads);

        expect_matrices_equal(S_ref, S_test, nbf, TOL,
                              "Overlap_" + std::to_string(n_threads) + "_threads");
    }
}

TEST(Parallel1eTest, RepeatedComputations) {
    auto basis = make_h2o_sto3g();
    const Size nbf = basis.n_basis_functions();

    Engine engine(basis);
    OneElectronOperator overlap_op(Operator::overlap());

    std::vector<Real> S_prev;
    engine.compute_1e_parallel(overlap_op, S_prev, 4);

    for (int iter = 0; iter < 5; ++iter) {
        std::vector<Real> S_test;
        engine.compute_1e_parallel(overlap_op, S_test, 4);

        expect_matrices_equal(S_prev, S_test, nbf, TOL,
                              "Overlap_iter_" + std::to_string(iter));
    }
}

TEST(Parallel1eTest, EmptyBasisSet) {
    auto basis = BasisSet(std::vector<Shell>{});

    Engine engine(basis);
    OneElectronOperator overlap_op(Operator::overlap());

    std::vector<Real> S_parallel;
    engine.compute_1e_parallel(overlap_op, S_parallel, 4);

    EXPECT_TRUE(S_parallel.empty());
}

TEST(Parallel1eTest, SingleShell) {
    // Single s-function
    Shell shell(0, Point3D{0.0, 0.0, 0.0},
                {1.0}, {1.0});
    shell.set_atom_index(0);

    auto basis = BasisSet(std::vector<Shell>{shell});
    const Size nbf = basis.n_basis_functions();

    Engine engine(basis);
    OneElectronOperator overlap_op(Operator::overlap());

    std::vector<Real> S_serial;
    engine.compute_1e(overlap_op, S_serial);

    std::vector<Real> S_parallel;
    engine.compute_1e_parallel(overlap_op, S_parallel, 4);

    expect_matrices_equal(S_serial, S_parallel, nbf, TOL, "SingleShell");
}
