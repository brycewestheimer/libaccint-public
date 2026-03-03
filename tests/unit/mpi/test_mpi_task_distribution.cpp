// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2026 Bryce M. Westheimer. All rights reserved.

/// @file test_mpi_task_distribution.cpp
/// @brief Unit tests for MPI task distribution / partitioning logic (Task 13.3.1)
///
/// Tests MPIEngineConfig defaults, GPUMapping enum values, MPIStats
/// efficiency calculation, and MPIEngine stub behavior in non-MPI builds.

#include <gtest/gtest.h>

#include <libaccint/mpi/mpi_engine.hpp>
#include <libaccint/mpi/mpi_guard.hpp>
#include <libaccint/basis/basis_set.hpp>
#include <libaccint/basis/shell.hpp>
#include <libaccint/core/types.hpp>

#include <string>
#include <vector>

namespace {

#if LIBACCINT_USE_MPI
libaccint::mpi::MPIGuard& mpi_guard() {
    static libaccint::mpi::MPIGuard guard;
    return guard;
}
#endif

libaccint::mpi::MPIEngine make_engine(const libaccint::BasisSet& basis,
                                      libaccint::mpi::MPIEngineConfig config = {}) {
#if LIBACCINT_USE_MPI
    (void)mpi_guard();
#endif
    return libaccint::mpi::MPIEngine(basis, config);
}

/// @brief Create a minimal basis set for engine construction tests
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
// MPIEngineConfig defaults
// ============================================================================

TEST(MPITaskDistribution, ConfigDefaults) {
    libaccint::mpi::MPIEngineConfig config;
#if LIBACCINT_USE_MPI
    EXPECT_NE(config.comm, MPI_COMM_NULL);
#else
    EXPECT_EQ(config.comm, nullptr);
#endif
    EXPECT_EQ(config.gpu_mapping, libaccint::mpi::GPUMapping::RoundRobin);
    EXPECT_TRUE(config.local_gpu_ids.empty());
    EXPECT_EQ(config.gpus_per_rank, 0);
    EXPECT_FALSE(config.collect_stats);
}

// ============================================================================
// GPUMapping enum values
// ============================================================================

TEST(MPITaskDistribution, GPUMappingValues) {
    // Verify all enum values are distinct and exist
    auto rr = libaccint::mpi::GPUMapping::RoundRobin;
    auto pk = libaccint::mpi::GPUMapping::Packed;
    auto ex = libaccint::mpi::GPUMapping::Exclusive;
    auto ud = libaccint::mpi::GPUMapping::UserDefined;

    EXPECT_NE(static_cast<int>(rr), static_cast<int>(pk));
    EXPECT_NE(static_cast<int>(pk), static_cast<int>(ex));
    EXPECT_NE(static_cast<int>(ex), static_cast<int>(ud));
    EXPECT_NE(static_cast<int>(rr), static_cast<int>(ud));
}

// ============================================================================
// MPIStats efficiency calculation
// ============================================================================

TEST(MPITaskDistribution, StatsEfficiencyZeroRanks) {
    libaccint::mpi::MPIStats stats;
    stats.total_ranks = 0;
    stats.total_quartets = 100;
    stats.local_quartets = 50;
    EXPECT_DOUBLE_EQ(stats.work_distribution_efficiency(), 0.0);
}

TEST(MPITaskDistribution, StatsEfficiencyIdeal) {
    // 4 ranks, 100 quartets, this rank gets exactly 25 (ideal share)
    libaccint::mpi::MPIStats stats;
    stats.total_ranks = 4;
    stats.total_quartets = 100;
    stats.local_quartets = 25;
    EXPECT_DOUBLE_EQ(stats.work_distribution_efficiency(), 1.0);
}

TEST(MPITaskDistribution, StatsEfficiencyUnbalanced) {
    // 4 ranks, 100 quartets, this rank only got 10 (under-loaded)
    libaccint::mpi::MPIStats stats;
    stats.total_ranks = 4;
    stats.total_quartets = 100;
    stats.local_quartets = 10;
    // ideal_per_rank = 100/4 = 25 → efficiency = 10/25 = 0.4
    EXPECT_DOUBLE_EQ(stats.work_distribution_efficiency(), 0.4);
}

TEST(MPITaskDistribution, StatsEfficiencyZeroQuartets) {
    libaccint::mpi::MPIStats stats;
    stats.total_ranks = 4;
    stats.total_quartets = 0;
    stats.local_quartets = 0;
    EXPECT_DOUBLE_EQ(stats.work_distribution_efficiency(), 0.0);
}

// ============================================================================
// MPIEngine stub construction and accessors
// ============================================================================

TEST(MPITaskDistribution, StubEngineConstruction) {
    auto basis = make_test_basis();
    libaccint::mpi::MPIEngineConfig config;
    EXPECT_NO_THROW({
        auto engine = make_engine(basis, config);
        (void)engine;
    });
}

TEST(MPITaskDistribution, StubRankSize) {
    auto basis = make_test_basis();
    auto engine = make_engine(basis);
#if LIBACCINT_USE_MPI
    // In a real MPI build, rank/size reflect the communicator
    EXPECT_GE(engine.rank(), 0);
    EXPECT_GE(engine.size(), 1);
#else
    EXPECT_EQ(engine.rank(), 0);
    EXPECT_EQ(engine.size(), 1);
#endif
}

// ============================================================================
// Multi-rank tests (require world_size > 1, i.e. mpirun -np 2+)
// ============================================================================

TEST(MPITaskDistribution, MultiRankWorkDistribution) {
    auto basis = make_test_basis();
    auto engine = make_engine(basis);

    if (engine.size() <= 1) {
        GTEST_SKIP() << "Requires world_size > 1 (run with mpirun -np 2)";
    }

    // Each rank should have a valid rank in [0, size)
    EXPECT_GE(engine.rank(), 0);
    EXPECT_LT(engine.rank(), engine.size());
    // Root rank is 0
    if (engine.rank() == 0) {
        EXPECT_TRUE(engine.is_root());
    } else {
        EXPECT_FALSE(engine.is_root());
    }
}

TEST(MPITaskDistribution, StubIsRoot) {
    auto basis = make_test_basis();
    auto engine = make_engine(basis);
    EXPECT_EQ(engine.is_root(), engine.rank() == 0);
}

TEST(MPITaskDistribution, StubSummary) {
    auto basis = make_test_basis();
    auto engine = make_engine(basis);
    std::string summary = engine.summary();
    EXPECT_FALSE(summary.empty());
#if LIBACCINT_USE_MPI
    EXPECT_NE(summary.find("MPIEngine: rank"), std::string::npos);
#else
    EXPECT_NE(summary.find("stub"), std::string::npos);
#endif
}
