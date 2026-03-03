// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2026 Bryce M. Westheimer. All rights reserved.

#pragma once

/// @file debug_checks.hpp
/// @brief Debug mode extra validation checks for LibAccInt
///
/// Provides compile-time and runtime debug checks that are active only
/// in debug builds (NDEBUG not defined) or when explicitly enabled.
/// These checks verify invariants that are too expensive for release mode.

#include <libaccint/core/types.hpp>
#include <libaccint/utils/error_handling.hpp>

#include <cmath>
#include <span>
#include <string>
#include <vector>

namespace libaccint::debug {

/// @brief Check if debug mode is active
///
/// Returns true in debug builds (NDEBUG undefined) or when
/// LIBACCINT_DEBUG_CHECKS is defined at compile time.
[[nodiscard]] constexpr bool is_debug_mode() noexcept {
#if defined(NDEBUG) && !defined(LIBACCINT_DEBUG_CHECKS)
    return false;
#else
    return true;
#endif
}

/// @brief Validate that a vector contains no NaN or Inf values
/// @param data The data to check
/// @param label A label for the error message
/// @throws NumericalException if NaN/Inf found (debug mode only)
inline void check_finite(std::span<const Real> data,
                         const std::string& label) {
    if (!is_debug_mode()) return;
    for (Size i = 0; i < data.size(); ++i) {
        if (!std::isfinite(data[i])) {
            throw NumericalException(
                label + ": non-finite value at index " +
                std::to_string(i) + " (value=" +
                std::to_string(data[i]) + ")");
        }
    }
}

/// @brief Validate matrix symmetry: |M[i,j] - M[j,i]| < tol
/// @param matrix Row-major square matrix data
/// @param n Matrix dimension
/// @param tol Symmetry tolerance
/// @param label A label for the error message
/// @throws NumericalException if asymmetry found (debug mode only)
inline void check_symmetric(std::span<const Real> matrix, Size n,
                            Real tol, const std::string& label) {
    if (!is_debug_mode()) return;
    if (matrix.size() != n * n) {
        throw InvalidArgumentException(
            label + ": expected " + std::to_string(n * n) +
            " elements, got " + std::to_string(matrix.size()));
    }
    for (Size i = 0; i < n; ++i) {
        for (Size j = i + 1; j < n; ++j) {
            Real diff = std::abs(matrix[i * n + j] - matrix[j * n + i]);
            if (diff > tol) {
                throw NumericalException(
                    label + ": asymmetry at (" + std::to_string(i) + "," +
                    std::to_string(j) + "): diff=" + std::to_string(diff));
            }
        }
    }
}

/// @brief Validate that exponents are all positive
/// @param exponents The exponents to check
/// @param label A label for the error message
/// @throws InvalidArgumentException if any exponent is non-positive
inline void check_positive_exponents(std::span<const Real> exponents,
                                     const std::string& label) {
    if (!is_debug_mode()) return;
    for (Size i = 0; i < exponents.size(); ++i) {
        if (exponents[i] <= 0.0) {
            throw InvalidArgumentException(
                label + ": non-positive exponent at index " +
                std::to_string(i) + " (value=" +
                std::to_string(exponents[i]) + ")");
        }
    }
}

/// @brief Validate that a matrix is positive semi-definite (via diagonal check)
/// @note This is a necessary but not sufficient condition
/// @param matrix Row-major square matrix
/// @param n Matrix dimension
/// @param label Label for error reporting
/// @throws NumericalException if negative diagonal found
inline void check_positive_diagonal(std::span<const Real> matrix, Size n,
                                    const std::string& label) {
    if (!is_debug_mode()) return;
    if (matrix.size() != n * n) return;
    for (Size i = 0; i < n; ++i) {
        if (matrix[i * n + i] < -1e-10) {
            throw NumericalException(
                label + ": negative diagonal at index " +
                std::to_string(i) + " (value=" +
                std::to_string(matrix[i * n + i]) + ")");
        }
    }
}

/// @brief Validate angular momentum is within supported range
/// @param am Angular momentum value
/// @param label Label for error reporting
/// @throws InvalidArgumentException if AM is out of range
inline void check_angular_momentum(int am, const std::string& label) {
    if (!is_debug_mode()) return;
    if (am < 0 || am > MAX_ANGULAR_MOMENTUM) {
        throw InvalidArgumentException(
            label + ": angular momentum " + std::to_string(am) +
            " out of supported range [0, " +
            std::to_string(MAX_ANGULAR_MOMENTUM) + "]");
    }
}

/// @brief Validate buffer size is sufficient
/// @param actual Actual buffer size
/// @param required Required buffer size
/// @param label Label for error reporting
/// @throws InvalidArgumentException if buffer too small
inline void check_buffer_size(Size actual, Size required,
                              const std::string& label) {
    if (!is_debug_mode()) return;
    if (actual < required) {
        throw InvalidArgumentException(
            label + ": buffer too small, need " +
            std::to_string(required) + " elements, got " +
            std::to_string(actual));
    }
}

/// @brief Validate that shell indices are consistently assigned
/// @param shell_index The shell index to check
/// @param function_index The function index to check
/// @param label Label for error reporting
/// @throws InvalidStateException if indices are inconsistent
inline void check_shell_indices(Index shell_index, Index function_index,
                                const std::string& label) {
    if (!is_debug_mode()) return;
    if (shell_index < 0) {
        throw InvalidStateException(
            label + ": shell_index not assigned (value=" +
            std::to_string(shell_index) + ")");
    }
    if (function_index < 0) {
        throw InvalidStateException(
            label + ": function_index not assigned (value=" +
            std::to_string(function_index) + ")");
    }
}

}  // namespace libaccint::debug

// ============================================================================
// Debug-Only Assertion Macro
// ============================================================================

/// @brief Assert only in debug mode (zero-cost in release)
#if defined(NDEBUG) && !defined(LIBACCINT_DEBUG_CHECKS)
#define LIBACCINT_DEBUG_ASSERT(cond, msg) ((void)0)
#else
#define LIBACCINT_DEBUG_ASSERT(cond, msg) LIBACCINT_ASSERT(cond, msg)
#endif
