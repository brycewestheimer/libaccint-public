// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2026 Bryce M. Westheimer. All rights reserved.

/// @file test_mpi_integration_fock.cpp
/// @brief Integration-level tests for MPI Fock pipeline using stubs (Task 13.3.6)
///
/// Exercises the full MPI Fock workflow with stubs: creating an MPIEngine
/// and MPIFockBuilder together, verifying zero-initialized matrices, and
/// running the no-op compute path.

#include <gtest/gtest.h>

#include <libaccint/mpi/mpi_engine.hpp>
#include <libaccint/mpi/mpi_fock_builder.hpp>
#include <libaccint/mpi/mpi_guard.hpp>
#include <libaccint/basis/basis_set.hpp>
#include <libaccint/basis/shell.hpp>
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

libaccint::mpi::MPIEngine make_engine(const libaccint::BasisSet& basis,
                                      libaccint::mpi::MPIEngineConfig config = {}) {
#if LIBACCINT_USE_MPI
    (void)mpi_guard();
#endif
    return libaccint::mpi::MPIEngine(basis, config);
}

libaccint::mpi::MPIFockBuilder make_builder(libaccint::Size nbf) {
    return libaccint::mpi::MPIFockBuilder(test_comm(), nbf);
}

/// @brief Create a minimal basis set for integration tests
libaccint::BasisSet make_test_basis() {
    using namespace libaccint;
    // s-type shell at origin
    Shell s_shell(0, Point3D{0.0, 0.0, 0.0}, {1.0}, {1.0});
    // p-type shell at displaced center
    Shell p_shell(1, Point3D{1.0, 0.0, 0.0}, {0.5}, {1.0});

    std::vector<Shell> shells = {s_shell, p_shell};
    return BasisSet(shells);
}

}  // namespace

// ============================================================================
// Stub engine + builder coexistence
// ============================================================================

TEST(MPIIntegrationFock, StubEngineAndBuilder) {
    auto basis = make_test_basis();
    const auto nbf = basis.n_basis_functions();

    libaccint::mpi::MPIEngineConfig config;
    auto engine = make_engine(basis, config);
    auto builder = make_builder(nbf);

    // Both should report consistent rank/size
    EXPECT_EQ(engine.rank(), builder.rank());
    EXPECT_EQ(engine.size(), builder.size());
    EXPECT_EQ(engine.is_root(), builder.is_root());
}

// ============================================================================
// J/K matrices zero-initialized from stub
// ============================================================================

TEST(MPIIntegrationFock, StubBuilderZeroInit) {
    auto basis = make_test_basis();
    const auto nbf = basis.n_basis_functions();

    auto builder = make_builder(nbf);

    auto J = builder.get_coulomb_matrix();
    auto K = builder.get_exchange_matrix();

    ASSERT_EQ(J.size(), nbf * nbf);
    ASSERT_EQ(K.size(), nbf * nbf);

    for (libaccint::Size i = 0; i < J.size(); ++i) {
        EXPECT_DOUBLE_EQ(J[i], 0.0) << "J[" << i << "] not zero";
    }
    for (libaccint::Size i = 0; i < K.size(); ++i) {
        EXPECT_DOUBLE_EQ(K[i], 0.0) << "K[" << i << "] not zero";
    }
}

// ============================================================================
// Stub compute_all_eri is a no-op — J/K remain zero
// ============================================================================

TEST(MPIIntegrationFock, StubComputeERI) {
    auto basis = make_test_basis();
    const auto nbf = basis.n_basis_functions();

    auto engine = make_engine(basis);
    auto builder = make_builder(nbf);

    // Set a density matrix
    std::vector<libaccint::Real> D(nbf * nbf, 0.0);
    for (libaccint::Size i = 0; i < nbf; ++i) {
        D[i * nbf + i] = 1.0;
    }
    builder.set_density(D.data(), nbf);

    // compute_all_eri is a no-op in the stub engine
    engine.compute_all_eri(builder);

    auto J = builder.get_coulomb_matrix();
    auto K = builder.get_exchange_matrix();
#if LIBACCINT_USE_MPI
    bool j_nonzero = false;
    bool k_nonzero = false;
    for (libaccint::Size i = 0; i < J.size(); ++i) {
        j_nonzero = j_nonzero || std::abs(J[i]) > 1e-15;
        k_nonzero = k_nonzero || std::abs(K[i]) > 1e-15;
    }
    EXPECT_TRUE(j_nonzero);
    EXPECT_TRUE(k_nonzero);
#else
    for (libaccint::Size i = 0; i < J.size(); ++i) {
        EXPECT_DOUBLE_EQ(J[i], 0.0);
    }
    for (libaccint::Size i = 0; i < K.size(); ++i) {
        EXPECT_DOUBLE_EQ(K[i], 0.0);
    }
#endif
}

// ============================================================================
// Full stub workflow: engine → density → compute → reduce → read
// ============================================================================

TEST(MPIIntegrationFock, StubFullWorkflow) {
    auto basis = make_test_basis();
    const auto nbf = basis.n_basis_functions();

    // 1. Create engine and builder
    libaccint::mpi::MPIEngineConfig config;
    config.collect_stats = true;
    auto engine = make_engine(basis, config);
    auto builder = make_builder(nbf);

    // 2. Set density
    std::vector<libaccint::Real> D(nbf * nbf, 0.0);
    for (libaccint::Size i = 0; i < nbf; ++i) {
        D[i * nbf + i] = 2.0;
    }
    builder.set_density(D.data(), nbf);

    // 3. Compute
    engine.compute_all_eri(builder);

    // 4. Reduce
    builder.allreduce();

    // 5. Read results
    auto J = builder.get_coulomb_matrix();
    auto K = builder.get_exchange_matrix();

    ASSERT_EQ(J.size(), nbf * nbf);
    ASSERT_EQ(K.size(), nbf * nbf);
#if LIBACCINT_USE_MPI
    bool j_nonzero = false;
    bool k_nonzero = false;
    for (libaccint::Size i = 0; i < nbf * nbf; ++i) {
        j_nonzero = j_nonzero || std::abs(J[i]) > 1e-15;
        k_nonzero = k_nonzero || std::abs(K[i]) > 1e-15;
    }
    EXPECT_TRUE(j_nonzero);
    EXPECT_TRUE(k_nonzero);
#else
    for (libaccint::Size i = 0; i < nbf * nbf; ++i) {
        EXPECT_DOUBLE_EQ(J[i], 0.0);
        EXPECT_DOUBLE_EQ(K[i], 0.0);
    }
#endif
}

// ============================================================================
// Multi-rank tests (require world_size > 1, i.e. mpirun -np 2+)
// ============================================================================

TEST(MPIIntegrationFock, MultiRankFullWorkflow) {
    auto basis = make_test_basis();
    const auto nbf = basis.n_basis_functions();

    libaccint::mpi::MPIEngineConfig config;
    config.collect_stats = true;
    auto engine = make_engine(basis, config);

    if (engine.size() <= 1) {
        GTEST_SKIP() << "Requires world_size > 1 (run with mpirun -np 2)";
    }

    auto builder = make_builder(nbf);

    // Set density
    std::vector<libaccint::Real> D(nbf * nbf, 0.0);
    for (libaccint::Size i = 0; i < nbf; ++i) {
        D[i * nbf + i] = 1.0;
    }
    builder.set_density(D.data(), nbf);

    // Compute (distributed)
    engine.compute_all_eri(builder);

    // Reduce across ranks
    builder.allreduce();

    // All ranks should have consistent J/K dimensions
    auto J = builder.get_coulomb_matrix();
    auto K = builder.get_exchange_matrix();
    EXPECT_EQ(J.size(), nbf * nbf);
    EXPECT_EQ(K.size(), nbf * nbf);

    // Engine and builder should report consistent rank/size
    EXPECT_EQ(engine.rank(), builder.rank());
    EXPECT_EQ(engine.size(), builder.size());
}

// ============================================================================
// Reset after use — J/K go back to zero
// ============================================================================

TEST(MPIIntegrationFock, FockBuilderResetAfterUse) {
    auto basis = make_test_basis();
    const auto nbf = basis.n_basis_functions();

    auto builder = make_builder(nbf);

    // Set density and run the no-op reduction path
    std::vector<libaccint::Real> D(nbf * nbf, 1.0);
    builder.set_density(D.data(), nbf);
    builder.allreduce();

    // Reset
    builder.reset();

    auto J = builder.get_coulomb_matrix();
    auto K = builder.get_exchange_matrix();

    for (libaccint::Size i = 0; i < J.size(); ++i) {
        EXPECT_DOUBLE_EQ(J[i], 0.0);
    }
    for (libaccint::Size i = 0; i < K.size(); ++i) {
        EXPECT_DOUBLE_EQ(K[i], 0.0);
    }
}
