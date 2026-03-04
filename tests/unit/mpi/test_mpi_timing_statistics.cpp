// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2026 Bryce M. Westheimer. All rights reserved.

/// @file test_mpi_timing_statistics.cpp
/// @brief Unit tests for MPIStats timing and efficiency (Task 13.3.3)
///
/// Tests MPIStats default values, field mutation, work_distribution_efficiency()
/// edge cases, and the collect_stats configuration flag.

#include <gtest/gtest.h>

#include <libaccint/mpi/mpi_engine.hpp>
#include <libaccint/core/types.hpp>

// ============================================================================
// Default values
// ============================================================================

TEST(MPITimingStatistics, DefaultTimingValues) {
    libaccint::mpi::MPIStats stats;

    EXPECT_DOUBLE_EQ(stats.total_time_ms, 0.0);
    EXPECT_DOUBLE_EQ(stats.compute_time_ms, 0.0);
    EXPECT_DOUBLE_EQ(stats.communication_time_ms, 0.0);
    EXPECT_DOUBLE_EQ(stats.reduction_time_ms, 0.0);
    EXPECT_EQ(stats.total_ranks, 0);
    EXPECT_EQ(stats.local_quartets, 0u);
    EXPECT_EQ(stats.total_quartets, 0u);
}

// ============================================================================
// Set and read fields
// ============================================================================

TEST(MPITimingStatistics, SetAndReadFields) {
    libaccint::mpi::MPIStats stats;

    stats.total_time_ms = 100.5;
    stats.compute_time_ms = 80.0;
    stats.communication_time_ms = 15.0;
    stats.reduction_time_ms = 5.5;
    stats.total_ranks = 16;
    stats.local_quartets = 250;
    stats.total_quartets = 4000;

    EXPECT_DOUBLE_EQ(stats.total_time_ms, 100.5);
    EXPECT_DOUBLE_EQ(stats.compute_time_ms, 80.0);
    EXPECT_DOUBLE_EQ(stats.communication_time_ms, 15.0);
    EXPECT_DOUBLE_EQ(stats.reduction_time_ms, 5.5);
    EXPECT_EQ(stats.total_ranks, 16);
    EXPECT_EQ(stats.local_quartets, 250u);
    EXPECT_EQ(stats.total_quartets, 4000u);
}

// ============================================================================
// Efficiency calculations
// ============================================================================

TEST(MPITimingStatistics, EfficiencyPerfectBalance) {
    libaccint::mpi::MPIStats stats;
    stats.total_ranks = 8;
    stats.total_quartets = 800;
    stats.local_quartets = 100;  // 800/8 = 100 → perfect balance
    EXPECT_DOUBLE_EQ(stats.work_distribution_efficiency(), 1.0);
}

TEST(MPITimingStatistics, EfficiencyImbalance) {
    libaccint::mpi::MPIStats stats;
    stats.total_ranks = 4;
    stats.total_quartets = 200;
    stats.local_quartets = 75;  // ideal = 50 → efficiency = 75/50 = 1.5
    EXPECT_DOUBLE_EQ(stats.work_distribution_efficiency(), 1.5);
}

TEST(MPITimingStatistics, EfficiencyEdgeCases) {
    // Both zero → 0.0
    {
        libaccint::mpi::MPIStats stats;
        stats.total_ranks = 0;
        stats.total_quartets = 0;
        stats.local_quartets = 0;
        EXPECT_DOUBLE_EQ(stats.work_distribution_efficiency(), 0.0);
    }

    // total_ranks > 0 but total_quartets = 0 → 0.0
    {
        libaccint::mpi::MPIStats stats;
        stats.total_ranks = 4;
        stats.total_quartets = 0;
        stats.local_quartets = 0;
        EXPECT_DOUBLE_EQ(stats.work_distribution_efficiency(), 0.0);
    }

    // total_quartets < total_ranks → ideal_per_rank = 0 → returns 1.0
    {
        libaccint::mpi::MPIStats stats;
        stats.total_ranks = 10;
        stats.total_quartets = 3;     // 3/10 = 0 (integer division)
        stats.local_quartets = 1;
        EXPECT_DOUBLE_EQ(stats.work_distribution_efficiency(), 1.0);
    }

    // Single rank, perfect
    {
        libaccint::mpi::MPIStats stats;
        stats.total_ranks = 1;
        stats.total_quartets = 50;
        stats.local_quartets = 50;
        EXPECT_DOUBLE_EQ(stats.work_distribution_efficiency(), 1.0);
    }
}

// ============================================================================
// collect_stats flag in config
// ============================================================================

TEST(MPITimingStatistics, CollectStatsFlag) {
    // Default is false
    libaccint::mpi::MPIEngineConfig config;
    EXPECT_FALSE(config.collect_stats);

    // Can be toggled
    config.collect_stats = true;
    EXPECT_TRUE(config.collect_stats);

    config.collect_stats = false;
    EXPECT_FALSE(config.collect_stats);
}
