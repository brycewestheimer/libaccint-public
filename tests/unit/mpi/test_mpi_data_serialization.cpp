// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2026 Bryce M. Westheimer. All rights reserved.

/// @file test_mpi_data_serialization.cpp
/// @brief Unit tests for safe_mpi_count and data conversion (Task 13.3.4)
///
/// Tests the safe_mpi_count() utility which validates that a size_t value
/// fits within an int for MPI function calls.

#include <gtest/gtest.h>

#include <libaccint/mpi/mpi_utils.hpp>

#include <climits>
#include <cstddef>
#include <limits>
#include <stdexcept>

// ============================================================================
// safe_mpi_count tests
// ============================================================================

TEST(MPIDataSerialization, CountZero) {
    int result = libaccint::mpi::safe_mpi_count(0);
    EXPECT_EQ(result, 0);
}

TEST(MPIDataSerialization, CountSmall) {
    int result = libaccint::mpi::safe_mpi_count(100);
    EXPECT_EQ(result, 100);
}

TEST(MPIDataSerialization, CountIntMax) {
    const auto int_max = static_cast<std::size_t>(std::numeric_limits<int>::max());
    int result = libaccint::mpi::safe_mpi_count(int_max);
    EXPECT_EQ(result, std::numeric_limits<int>::max());
}

TEST(MPIDataSerialization, CountOverflowThrows) {
    const auto one_past = static_cast<std::size_t>(std::numeric_limits<int>::max()) + 1;
    EXPECT_THROW(libaccint::mpi::safe_mpi_count(one_past), std::overflow_error);
}

TEST(MPIDataSerialization, CountSizeMaxThrows) {
    EXPECT_THROW(
        libaccint::mpi::safe_mpi_count(std::numeric_limits<std::size_t>::max()),
        std::overflow_error);
}
