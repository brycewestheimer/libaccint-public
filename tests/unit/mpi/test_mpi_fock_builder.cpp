// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2026 Bryce M. Westheimer. All rights reserved.

/// @file test_mpi_fock_builder.cpp
/// @brief Unit tests for MPIFockBuilder stub behavior (Task 13.3.2)
///
/// Tests the MPIFockBuilder stub in non-MPI builds: construction, accessors,
/// matrix sizes, reset, and that reduction operations are safe no-ops.

#include <gtest/gtest.h>

#include <libaccint/basis/basis_set.hpp>
#include <libaccint/basis/shell.hpp>
#include <libaccint/mpi/mpi_fock_builder.hpp>
#include <libaccint/mpi/mpi_guard.hpp>
#include <libaccint/core/types.hpp>

#include <cmath>
#include <vector>

namespace {

#if LIBACCINT_USE_MPI
libaccint::mpi::MPIGuard& mpi_guard() {
    static libaccint::mpi::MPIGuard guard;
    return guard;
}

MPI_Comm test_comm() {
    (void)mpi_guard();
    return MPI_COMM_WORLD;
}
#else
void* test_comm() {
    return nullptr;
}
#endif

libaccint::mpi::MPIFockBuilder make_builder(libaccint::Size nbf) {
    return libaccint::mpi::MPIFockBuilder(test_comm(), nbf);
}

libaccint::BasisSet make_single_s_basis() {
    using namespace libaccint;
    std::vector<Shell> shells;
    shells.emplace_back(0, Point3D{0.0, 0.0, 0.0}, std::vector<Real>{1.0},
                        std::vector<Real>{1.0});
    return BasisSet(std::move(shells));
}

libaccint::ShellSetQuartet make_single_s_quartet(const libaccint::BasisSet& basis) {
    const auto& pairs = basis.shell_set_pairs();
    return libaccint::ShellSetQuartet(pairs.front(), pairs.front());
}

}  // namespace

// ============================================================================
// Construction
// ============================================================================

TEST(MPIFockBuilderTest, StubConstruction) {
    constexpr libaccint::Size nbf = 10;
    EXPECT_NO_THROW({
        auto builder = make_builder(nbf);
        (void)builder;
    });
}

// ============================================================================
// Accessors
// ============================================================================

TEST(MPIFockBuilderTest, StubAccessors) {
    constexpr libaccint::Size nbf = 8;
    auto builder = make_builder(nbf);

    EXPECT_EQ(builder.nbf(), nbf);
    EXPECT_GE(builder.rank(), 0);
    EXPECT_GE(builder.size(), 1);
    EXPECT_EQ(builder.is_root(), builder.rank() == 0);
}

// ============================================================================
// Matrix sizes
// ============================================================================

TEST(MPIFockBuilderTest, StubMatrixSizes) {
    constexpr libaccint::Size nbf = 5;
    auto builder = make_builder(nbf);

    auto J = builder.get_coulomb_matrix();
    auto K = builder.get_exchange_matrix();

    EXPECT_EQ(J.size(), nbf * nbf);
    EXPECT_EQ(K.size(), nbf * nbf);
}

// ============================================================================
// Reset clears matrices
// ============================================================================

TEST(MPIFockBuilderTest, StubReset) {
    constexpr libaccint::Size nbf = 4;
    auto builder = make_builder(nbf);

    builder.reset();

    auto J = builder.get_coulomb_matrix();
    auto K = builder.get_exchange_matrix();

    // After reset, all elements should be zero
    for (libaccint::Size i = 0; i < J.size(); ++i) {
        EXPECT_DOUBLE_EQ(J[i], 0.0);
    }
    for (libaccint::Size i = 0; i < K.size(); ++i) {
        EXPECT_DOUBLE_EQ(K[i], 0.0);
    }
}

// ============================================================================
// allreduce / reduce_to_root are no-ops (must not crash)
// ============================================================================

TEST(MPIFockBuilderTest, StubAllreduceNoOp) {
    constexpr libaccint::Size nbf = 6;
    auto builder = make_builder(nbf);
    EXPECT_NO_THROW(builder.allreduce());
}

TEST(MPIFockBuilderTest, StubReduceToRoot) {
    constexpr libaccint::Size nbf = 6;
    auto builder = make_builder(nbf);
    EXPECT_NO_THROW(builder.reduce_to_root());
}

// ============================================================================
// set_density + allreduce + get matrices (full no-op workflow)
// ============================================================================

TEST(MPIFockBuilderTest, SetDensityAndReduce) {
    constexpr libaccint::Size nbf = 3;
    auto builder = make_builder(nbf);

    // Create a simple identity-like density matrix
    std::vector<libaccint::Real> D(nbf * nbf, 0.0);
    for (libaccint::Size i = 0; i < nbf; ++i) {
        D[i * nbf + i] = 1.0;
    }

    builder.set_density(D.data(), nbf);
    builder.allreduce();

    // Matrices should still be zero since no integrals were accumulated
    auto J = builder.get_coulomb_matrix();
    auto K = builder.get_exchange_matrix();

    for (libaccint::Size i = 0; i < J.size(); ++i) {
        EXPECT_DOUBLE_EQ(J[i], 0.0);
    }
    for (libaccint::Size i = 0; i < K.size(); ++i) {
        EXPECT_DOUBLE_EQ(K[i], 0.0);
    }
}

TEST(MPIFockBuilderTest, ThreadingStrategyPassthrough) {
    constexpr libaccint::Size nbf = 3;
    auto builder = make_builder(nbf);

    builder.set_threading_strategy(
        libaccint::consumers::FockThreadingStrategy::ThreadLocal);
    EXPECT_EQ(builder.threading_strategy(),
              libaccint::consumers::FockThreadingStrategy::ThreadLocal);

    EXPECT_NO_THROW(builder.prepare_parallel(2));
    EXPECT_NO_THROW(builder.finalize_parallel());
}

TEST(MPIFockBuilderTest, HostFlatAccumulateProducesLocalContribution) {
    auto basis = make_single_s_basis();
    auto quartet = make_single_s_quartet(basis);

    auto builder = make_builder(basis.n_basis_functions());
    std::vector<libaccint::Real> D(basis.n_basis_functions() * basis.n_basis_functions(), 1.0);
    builder.set_density(D.data(), basis.n_basis_functions());

    const double eri_value = 2.0;
    builder.accumulate(&eri_value, quartet);

    auto J = builder.get_coulomb_matrix();
    auto K = builder.get_exchange_matrix();
    ASSERT_EQ(J.size(), 1u);
    ASSERT_EQ(K.size(), 1u);
    EXPECT_NEAR(J[0], 2.0, 1e-12);
    EXPECT_NEAR(K[0], 2.0, 1e-12);
}

// ============================================================================
// Multi-rank tests (require world_size > 1, i.e. mpirun -np 2+)
// ============================================================================

TEST(MPIFockBuilderTest, MultiRankAllreduceAccumulates) {
    constexpr libaccint::Size nbf = 4;
    auto builder = make_builder(nbf);

    if (builder.size() <= 1) {
        GTEST_SKIP() << "Requires world_size > 1 (run with mpirun -np 2)";
    }

    // In a multi-rank build, allreduce should sum J/K across ranks.
    // With zero-initialized matrices and no integral accumulation,
    // the result should still be zero after reduction.
    builder.allreduce();

    auto J = builder.get_coulomb_matrix();
    auto K = builder.get_exchange_matrix();

    for (libaccint::Size i = 0; i < J.size(); ++i) {
        EXPECT_DOUBLE_EQ(J[i], 0.0) << "J[" << i << "] non-zero after multi-rank allreduce";
    }
    for (libaccint::Size i = 0; i < K.size(); ++i) {
        EXPECT_DOUBLE_EQ(K[i], 0.0) << "K[" << i << "] non-zero after multi-rank allreduce";
    }
}

TEST(MPIFockBuilderTest, MultiRankRankConsistency) {
    constexpr libaccint::Size nbf = 4;
    auto builder = make_builder(nbf);

    if (builder.size() <= 1) {
        GTEST_SKIP() << "Requires world_size > 1 (run with mpirun -np 2)";
    }

    // Verify rank is in valid range and exactly one rank is root
    EXPECT_GE(builder.rank(), 0);
    EXPECT_LT(builder.rank(), builder.size());
    if (builder.rank() == 0) {
        EXPECT_TRUE(builder.is_root());
    } else {
        EXPECT_FALSE(builder.is_root());
    }
}
