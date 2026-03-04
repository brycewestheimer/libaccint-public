// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2026 Bryce M. Westheimer. All rights reserved.

#pragma once

/// @file input_validation.hpp
/// @brief Public API input validation utilities for LibAccInt
///
/// Provides reusable validation functions for all public API boundaries.
/// Each function throws InvalidArgumentException with descriptive messages
/// on validation failure.

#include <libaccint/basis/basis_set.hpp>
#include <libaccint/basis/shell.hpp>
#include <libaccint/core/types.hpp>
#include <libaccint/utils/error_handling.hpp>

#include <cmath>
#include <span>
#include <string>
#include <vector>

namespace libaccint::validation {

/// @brief Result of a validation check
struct ValidationResult {
    bool valid{true};
    std::string message;

    explicit operator bool() const noexcept { return valid; }

    static ValidationResult ok() { return {true, ""}; }
    static ValidationResult fail(const std::string& msg) { return {false, msg}; }
};

// ============================================================================
// Shell Validation
// ============================================================================

/// @brief Validate shell construction parameters
/// @throws InvalidArgumentException on validation failure
void validate_shell_params(int am,
                           std::span<const Real> exponents,
                           std::span<const Real> coefficients);

/// @brief Validate that a shell is in a valid state
[[nodiscard]] ValidationResult validate_shell(const Shell& shell);

// ============================================================================
// BasisSet Validation
// ============================================================================

/// @brief Validate that a basis set is non-empty and properly indexed
[[nodiscard]] ValidationResult validate_basis_set(const BasisSet& basis);

/// @brief Validate that a matrix matches the basis set dimensions
/// @throws InvalidArgumentException on size mismatch
void validate_matrix_size(const BasisSet& basis,
                          Size matrix_size,
                          const std::string& matrix_name = "matrix");

/// @brief Validate that a density matrix is sane (finite, correct size)
void validate_density_matrix(const BasisSet& basis,
                             std::span<const Real> density);

// ============================================================================
// Index Validation
// ============================================================================

/// @brief Validate shell index is in bounds
/// @throws InvalidArgumentException if out of bounds
void validate_shell_index(const BasisSet& basis, Size index);

/// @brief Validate atom index is in bounds
/// @throws InvalidArgumentException if out of bounds
void validate_atom_index(Size atom_index, Size n_atoms);

// ============================================================================
// Numerical Validation
// ============================================================================

/// @brief Validate that all values are finite (no NaN/Inf)
/// @throws NumericalException if NaN or Inf found
void validate_finite(std::span<const Real> data,
                     const std::string& label = "data");

/// @brief Validate that all exponents are positive
/// @throws InvalidArgumentException if any non-positive
void validate_positive_exponents(std::span<const Real> exponents);

/// @brief Validate a screening threshold is reasonable
/// @throws InvalidArgumentException if threshold is invalid
void validate_screening_threshold(Real threshold);

// ============================================================================
// Operator Validation
// ============================================================================

/// @brief Validate nuclear charges and positions
/// @throws InvalidArgumentException on validation failure
void validate_nuclear_data(std::span<const Real> charges,
                           std::span<const Point3D> positions);

/// @brief Validate range-separation parameter omega
/// @throws InvalidArgumentException if omega is not positive
void validate_omega(Real omega);

}  // namespace libaccint::validation
