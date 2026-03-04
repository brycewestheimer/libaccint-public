// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2026 Bryce M. Westheimer. All rights reserved.

/// @file test_debug_checks.cpp
/// @brief Tests for debug mode validation checks (Task 25.3.4)

#include <libaccint/utils/debug_checks.hpp>

#include <gtest/gtest.h>
#include <cmath>
#include <limits>
#include <vector>

using namespace libaccint;
using namespace libaccint::debug;

// ============================================================================
// Debug Mode Tests (active in debug builds or when LIBACCINT_DEBUG_CHECKS)
// ============================================================================

TEST(DebugChecks, IsDebugMode) {
    // In test builds (debug), this should be true
    // Test the function exists and returns a bool
    [[maybe_unused]] bool mode = is_debug_mode();
    // Can't assert specific value since it depends on NDEBUG
}

TEST(DebugChecks, CheckFiniteValid) {
    std::vector<Real> data = {1.0, 2.0, -3.5, 0.0, 1e-15};
    // Should not throw for valid data
    if (is_debug_mode()) {
        EXPECT_NO_THROW(check_finite(data, "test_data"));
    }
}

TEST(DebugChecks, CheckFiniteNaN) {
    std::vector<Real> data = {1.0, std::numeric_limits<Real>::quiet_NaN(), 3.0};
    if (is_debug_mode()) {
        EXPECT_THROW(check_finite(data, "test_data"), NumericalException);
    }
}

TEST(DebugChecks, CheckFiniteInf) {
    std::vector<Real> data = {1.0, std::numeric_limits<Real>::infinity()};
    if (is_debug_mode()) {
        EXPECT_THROW(check_finite(data, "test_data"), NumericalException);
    }
}

TEST(DebugChecks, CheckFiniteNegInf) {
    std::vector<Real> data = {-std::numeric_limits<Real>::infinity()};
    if (is_debug_mode()) {
        EXPECT_THROW(check_finite(data, "test_data"), NumericalException);
    }
}

TEST(DebugChecks, CheckFiniteEmpty) {
    std::vector<Real> data;
    // Empty data should be fine
    EXPECT_NO_THROW(check_finite(data, "empty"));
}

TEST(DebugChecks, CheckSymmetricValid) {
    // 3x3 symmetric matrix
    std::vector<Real> mat = {
        1.0, 2.0, 3.0,
        2.0, 4.0, 5.0,
        3.0, 5.0, 6.0
    };
    if (is_debug_mode()) {
        EXPECT_NO_THROW(check_symmetric(mat, 3, 1e-12, "test_matrix"));
    }
}

TEST(DebugChecks, CheckSymmetricInvalid) {
    // 3x3 asymmetric matrix
    std::vector<Real> mat = {
        1.0, 2.0, 3.0,
        2.1, 4.0, 5.0,  // mat[1][0] = 2.1 != mat[0][1] = 2.0
        3.0, 5.0, 6.0
    };
    if (is_debug_mode()) {
        EXPECT_THROW(check_symmetric(mat, 3, 1e-3, "test_matrix"),
                     NumericalException);
    }
}

TEST(DebugChecks, CheckSymmetricWrongSize) {
    std::vector<Real> mat = {1.0, 2.0, 3.0};
    if (is_debug_mode()) {
        EXPECT_THROW(check_symmetric(mat, 2, 1e-12, "test_matrix"),
                     InvalidArgumentException);
    }
}

TEST(DebugChecks, CheckPositiveExponents) {
    std::vector<Real> exp = {1.0, 0.5, 10.0};
    if (is_debug_mode()) {
        EXPECT_NO_THROW(check_positive_exponents(exp, "exponents"));
    }
}

TEST(DebugChecks, CheckNonPositiveExponents) {
    std::vector<Real> exp = {1.0, -0.5};
    if (is_debug_mode()) {
        EXPECT_THROW(check_positive_exponents(exp, "exponents"),
                     InvalidArgumentException);
    }
}

TEST(DebugChecks, CheckPositiveDiagonalValid) {
    std::vector<Real> mat = {
        1.0, 0.0,
        0.0, 2.0
    };
    if (is_debug_mode()) {
        EXPECT_NO_THROW(check_positive_diagonal(mat, 2, "test"));
    }
}

TEST(DebugChecks, CheckPositiveDiagonalNegative) {
    std::vector<Real> mat = {
        -1.0, 0.0,
        0.0, 2.0
    };
    if (is_debug_mode()) {
        EXPECT_THROW(check_positive_diagonal(mat, 2, "test"),
                     NumericalException);
    }
}

TEST(DebugChecks, CheckAngularMomentumValid) {
    if (is_debug_mode()) {
        for (int am = 0; am <= MAX_ANGULAR_MOMENTUM; ++am) {
            EXPECT_NO_THROW(check_angular_momentum(am, "test"));
        }
    }
}

TEST(DebugChecks, CheckAngularMomentumInvalid) {
    if (is_debug_mode()) {
        EXPECT_THROW(check_angular_momentum(-1, "test"),
                     InvalidArgumentException);
        EXPECT_THROW(check_angular_momentum(MAX_ANGULAR_MOMENTUM + 1, "test"),
                     InvalidArgumentException);
    }
}

TEST(DebugChecks, CheckBufferSizeValid) {
    if (is_debug_mode()) {
        EXPECT_NO_THROW(check_buffer_size(100, 50, "test"));
        EXPECT_NO_THROW(check_buffer_size(50, 50, "test"));
    }
}

TEST(DebugChecks, CheckBufferSizeTooSmall) {
    if (is_debug_mode()) {
        EXPECT_THROW(check_buffer_size(10, 50, "test"),
                     InvalidArgumentException);
    }
}

TEST(DebugChecks, CheckShellIndicesValid) {
    if (is_debug_mode()) {
        EXPECT_NO_THROW(check_shell_indices(0, 0, "test"));
        EXPECT_NO_THROW(check_shell_indices(5, 15, "test"));
    }
}

TEST(DebugChecks, CheckShellIndicesUnassigned) {
    if (is_debug_mode()) {
        EXPECT_THROW(check_shell_indices(-1, 0, "test"),
                     InvalidStateException);
        EXPECT_THROW(check_shell_indices(0, -1, "test"),
                     InvalidStateException);
    }
}

TEST(DebugChecks, DebugAssertMacro) {
    // LIBACCINT_DEBUG_ASSERT should not throw for true condition
    EXPECT_NO_THROW(LIBACCINT_DEBUG_ASSERT(true, "should pass"));

    // In debug mode, false condition should throw
    if (is_debug_mode()) {
        EXPECT_THROW(LIBACCINT_DEBUG_ASSERT(false, "should fail"),
                     InvalidStateException);
    }
}
