// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2026 Bryce M. Westheimer. All rights reserved.

/// @file test_mpi_error_handling.cpp
/// @brief Unit tests for MPIGuard stub behavior and error detection (Task 13.3.5)
///
/// Tests the MPIGuard stub that is active in non-MPI builds. The stub
/// reports MPI as not-initialized / already-finalized and provides safe
/// default values for rank, size, etc.

#include <gtest/gtest.h>

#include <libaccint/mpi/mpi_guard.hpp>

#include <string>

// ============================================================================
// MPIGuard stub construction
// ============================================================================

TEST(MPIErrorHandling, StubConstruction) {
    EXPECT_NO_THROW(libaccint::mpi::MPIGuard guard);
}

// ============================================================================
// Static query stubs
// ============================================================================

TEST(MPIErrorHandling, StubNotInitialized) {
    EXPECT_FALSE(libaccint::mpi::MPIGuard::is_initialized());
}

TEST(MPIErrorHandling, StubIsFinalized) {
    EXPECT_TRUE(libaccint::mpi::MPIGuard::is_finalized());
}

TEST(MPIErrorHandling, StubRank) {
    EXPECT_EQ(libaccint::mpi::MPIGuard::rank(), 0);
}

TEST(MPIErrorHandling, StubSize) {
    EXPECT_EQ(libaccint::mpi::MPIGuard::size(), 1);
}

TEST(MPIErrorHandling, StubIsRoot) {
    EXPECT_TRUE(libaccint::mpi::MPIGuard::is_root());
}

TEST(MPIErrorHandling, StubProcessorName) {
    std::string name = libaccint::mpi::MPIGuard::processor_name();
    EXPECT_EQ(name, "localhost");
}

TEST(MPIErrorHandling, StubBarrier) {
    EXPECT_NO_THROW(libaccint::mpi::MPIGuard::barrier());
}

// ============================================================================
// Instance-level stubs
// ============================================================================

TEST(MPIErrorHandling, StubThreadLevel) {
    libaccint::mpi::MPIGuard guard;
    EXPECT_EQ(guard.thread_level(), 0);
}

TEST(MPIErrorHandling, StubOwnsFinalize) {
    libaccint::mpi::MPIGuard guard;
    EXPECT_FALSE(guard.owns_finalize());
}

TEST(MPIErrorHandling, StubThreadSupport) {
    libaccint::mpi::MPIGuard guard;
    // Stub always reports thread support as satisfied
    EXPECT_TRUE(guard.thread_support_satisfied());
}
