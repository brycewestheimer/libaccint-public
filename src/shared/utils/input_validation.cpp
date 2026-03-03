// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2026 Bryce M. Westheimer. All rights reserved.

/// @file input_validation.cpp
/// @brief Implementation of public API input validation utilities

#include <libaccint/utils/input_validation.hpp>

#include <cmath>
#include <cstring>
#include <sstream>

namespace libaccint::validation {

namespace {

/// @brief Robust finite check that works even with -ffast-math
/// Uses bit-level inspection to avoid compiler optimization
[[nodiscard]] bool robust_isfinite(Real x) noexcept {
    // IEEE 754: exponent bits all-1 means Inf or NaN
    static_assert(sizeof(Real) == sizeof(uint64_t));
    uint64_t bits;
    std::memcpy(&bits, &x, sizeof(bits));
    uint64_t exponent = (bits >> 52) & 0x7FF;
    return exponent != 0x7FF;
}

}  // namespace

// ============================================================================
// Shell Validation
// ============================================================================

void validate_shell_params(int am,
                           std::span<const Real> exponents,
                           std::span<const Real> coefficients) {
    if (am < 0 || am > MAX_ANGULAR_MOMENTUM) {
        throw InvalidArgumentException(
            "Angular momentum " + std::to_string(am) +
            " out of range [0, " + std::to_string(MAX_ANGULAR_MOMENTUM) + "]");
    }
    if (exponents.empty()) {
        throw InvalidArgumentException("Exponents vector must not be empty");
    }
    if (coefficients.empty()) {
        throw InvalidArgumentException("Coefficients vector must not be empty");
    }
    if (exponents.size() != coefficients.size()) {
        throw InvalidArgumentException(
            "Exponents size (" + std::to_string(exponents.size()) +
            ") != coefficients size (" + std::to_string(coefficients.size()) + ")");
    }
    for (Size i = 0; i < exponents.size(); ++i) {
        if (exponents[i] <= 0.0) {
            throw InvalidArgumentException(
                "Exponent[" + std::to_string(i) + "] = " +
                std::to_string(exponents[i]) + " must be positive");
        }
        if (!robust_isfinite(exponents[i])) {
            throw InvalidArgumentException(
                "Exponent[" + std::to_string(i) + "] is not finite");
        }
        if (!robust_isfinite(coefficients[i])) {
            throw InvalidArgumentException(
                "Coefficient[" + std::to_string(i) + "] is not finite");
        }
    }
}

ValidationResult validate_shell(const Shell& shell) {
    if (!shell.valid()) {
        return ValidationResult::fail("Shell has no primitives");
    }
    if (shell.angular_momentum() < 0 ||
        shell.angular_momentum() > MAX_ANGULAR_MOMENTUM) {
        return ValidationResult::fail(
            "Angular momentum " + std::to_string(shell.angular_momentum()) +
            " out of range");
    }
    if (shell.n_primitives() == 0) {
        return ValidationResult::fail("Shell has zero primitives");
    }
    return ValidationResult::ok();
}

// ============================================================================
// BasisSet Validation
// ============================================================================

ValidationResult validate_basis_set(const BasisSet& basis) {
    if (basis.n_shells() == 0) {
        return ValidationResult::fail("BasisSet has no shells");
    }
    if (basis.n_basis_functions() == 0) {
        return ValidationResult::fail("BasisSet has no basis functions");
    }
    // Verify all shells have valid indices
    for (Size i = 0; i < basis.n_shells(); ++i) {
        const auto& shell = basis.shell(i);
        if (shell.shell_index() < 0) {
            return ValidationResult::fail(
                "Shell " + std::to_string(i) + " has unassigned shell_index");
        }
        if (shell.function_index() < 0) {
            return ValidationResult::fail(
                "Shell " + std::to_string(i) + " has unassigned function_index");
        }
    }
    return ValidationResult::ok();
}

void validate_matrix_size(const BasisSet& basis,
                          Size matrix_size,
                          const std::string& matrix_name) {
    Size nbf = basis.n_basis_functions();
    Size expected = nbf * nbf;
    if (matrix_size != expected) {
        throw InvalidArgumentException(
            matrix_name + " size mismatch: expected " +
            std::to_string(expected) + " (" + std::to_string(nbf) +
            " x " + std::to_string(nbf) + "), got " +
            std::to_string(matrix_size));
    }
}

void validate_density_matrix(const BasisSet& basis,
                             std::span<const Real> density) {
    Size nbf = basis.n_basis_functions();
    validate_matrix_size(basis, density.size(), "density matrix");
    // Check for finite values in a reasonable subset
    Size check_count = std::min(density.size(), Size{1000});
    for (Size i = 0; i < check_count; ++i) {
        if (!robust_isfinite(density[i])) {
            throw NumericalException(
                "Density matrix contains non-finite value at index " +
                std::to_string(i));
        }
    }
    (void)nbf;
}

// ============================================================================
// Index Validation
// ============================================================================

void validate_shell_index(const BasisSet& basis, Size index) {
    if (index >= basis.n_shells()) {
        throw InvalidArgumentException(
            "Shell index " + std::to_string(index) +
            " out of range [0, " + std::to_string(basis.n_shells()) + ")");
    }
}

void validate_atom_index(Size atom_index, Size n_atoms) {
    if (atom_index >= n_atoms) {
        throw InvalidArgumentException(
            "Atom index " + std::to_string(atom_index) +
            " out of range [0, " + std::to_string(n_atoms) + ")");
    }
}

// ============================================================================
// Numerical Validation
// ============================================================================

void validate_finite(std::span<const Real> data, const std::string& label) {
    for (Size i = 0; i < data.size(); ++i) {
        if (!robust_isfinite(data[i])) {
            throw NumericalException(
                label + " contains non-finite value at index " +
                std::to_string(i) + " (value=" +
                std::to_string(data[i]) + ")");
        }
    }
}

void validate_positive_exponents(std::span<const Real> exponents) {
    for (Size i = 0; i < exponents.size(); ++i) {
        if (exponents[i] <= 0.0) {
            throw InvalidArgumentException(
                "Exponent[" + std::to_string(i) + "] = " +
                std::to_string(exponents[i]) + " must be positive");
        }
    }
}

void validate_screening_threshold(Real threshold) {
    if (threshold < 0.0) {
        throw InvalidArgumentException(
            "Screening threshold must be non-negative, got " +
            std::to_string(threshold));
    }
    if (!robust_isfinite(threshold)) {
        throw InvalidArgumentException(
            "Screening threshold must be finite");
    }
    if (threshold > 1.0) {
        throw InvalidArgumentException(
            "Screening threshold " + std::to_string(threshold) +
            " is unusually large (> 1.0). Did you mean a smaller value?");
    }
}

// ============================================================================
// Operator Validation
// ============================================================================

void validate_nuclear_data(std::span<const Real> charges,
                           std::span<const Point3D> positions) {
    if (charges.size() != positions.size()) {
        throw InvalidArgumentException(
            "Nuclear charges size (" + std::to_string(charges.size()) +
            ") != positions size (" + std::to_string(positions.size()) + ")");
    }
    if (charges.empty()) {
        throw InvalidArgumentException("Nuclear charges vector must not be empty");
    }
    for (Size i = 0; i < charges.size(); ++i) {
        if (!robust_isfinite(charges[i])) {
            throw InvalidArgumentException(
                "Nuclear charge[" + std::to_string(i) + "] is not finite");
        }
    }
}

void validate_omega(Real omega) {
    if (omega <= 0.0) {
        throw InvalidArgumentException(
            "Range-separation parameter omega must be positive, got " +
            std::to_string(omega));
    }
    if (!robust_isfinite(omega)) {
        throw InvalidArgumentException(
            "Range-separation parameter omega must be finite");
    }
}

}  // namespace libaccint::validation
