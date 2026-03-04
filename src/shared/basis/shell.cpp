// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2026 Bryce M. Westheimer. All rights reserved.

/// @file shell.cpp
/// @brief Shell class implementation

#include <libaccint/basis/shell.hpp>
#include <libaccint/math/normalization.hpp>
#include <libaccint/utils/constants.hpp>
#include <libaccint/utils/error_handling.hpp>

#include <cmath>
#include <sstream>

namespace libaccint {

namespace {

/**
 * @brief Compute primitive normalization factor
 *
 * For a primitive Gaussian: exp(-α|r-A|²) with angular momentum l,
 * the normalization factor is:
 *   N = (2α/π)^(3/4) * (4α)^(l/2) / sqrt((2l-1)!!)
 *
 * This ensures the self-overlap of the (l,0,0) Cartesian component equals 1.
 *
 * @param l Angular momentum
 * @param exponent Primitive exponent α
 * @return Primitive normalization factor
 */
double primitive_normalization(int l, Real exponent) {
    // N = (2α/π)^(3/4) * (4α)^(l/2) / sqrt((2l-1)!!)

    // Compute (2α/π)^(3/4)
    const Real two_alpha_over_pi = 2.0 * exponent / constants::PI;
    const Real term1 = std::pow(two_alpha_over_pi, 0.75);

    // Compute (4α)^(l/2)
    const Real four_alpha = 4.0 * exponent;
    const Real term2 = std::pow(four_alpha, l * 0.5);

    // Compute sqrt((2l-1)!!)
    const Real dfact = static_cast<Real>(math::double_factorial_odd(l));
    const Real term3 = std::sqrt(dfact);

    return term1 * term2 / term3;
}

/**
 * @brief Normalize contraction coefficients
 *
 * This function applies both primitive and contraction normalization:
 * 1. Multiply each coefficient by its primitive normalization factor
 * 2. Compute self-overlap of the contracted shell (l,0,0 component)
 * 3. Scale all coefficients to make self-overlap = 1
 *
 * This matches the PySCF normalization convention.
 *
 * @param am Angular momentum
 * @param exponents Primitive exponents
 * @param coefficients Un-normalized contraction coefficients
 * @return Fully normalized contraction coefficients
 */
std::vector<Real> normalize_coefficients(
    int am,
    const std::vector<Real>& exponents,
    const std::vector<Real>& coefficients)
{
    const Size n_prim = exponents.size();
    std::vector<Real> normalized = coefficients;

    // Step 1: Apply primitive normalization to each coefficient
    for (Size i = 0; i < n_prim; ++i) {
        normalized[i] *= primitive_normalization(am, exponents[i]);
    }

    // Step 2: Compute self-overlap of the contracted shell
    // For the (l,0,0) component, the overlap integral between two primitives is:
    //   <exp(-α_i*r²) x^l | exp(-α_j*r²) x^l> = (π/p)^(3/2) * (1/2p)^l * (2l-1)!!
    // where p = α_i + α_j
    Real self_overlap = 0.0;
    for (Size i = 0; i < n_prim; ++i) {
        for (Size j = 0; j < n_prim; ++j) {
            const Real p = exponents[i] + exponents[j];

            // Compute (π/p)^(3/2)
            Real overlap = std::pow(constants::PI / p, 1.5);

            // Apply angular momentum factor: (1/2p)^l * (2l-1)!!
            // We can compute this as: product of (1/2p) for l times, times (2l-1)!!
            // Or more efficiently: (0.5/p)^l * (2l-1)!!
            // But to match v0 exactly, we use the loop form:
            for (int k = 0; k < am; ++k) {
                overlap *= 0.5 / p;
            }

            // Note: The (2l-1)!! factor is already included in the primitive
            // normalization, so we don't need it here in the overlap calculation.
            // The form above directly computes the overlap between normalized primitives.

            self_overlap += normalized[i] * normalized[j] * overlap;
        }
    }

    // Step 3: Scale coefficients to make self-overlap = 1
    const Real scale = 1.0 / std::sqrt(self_overlap);
    for (auto& c : normalized) {
        c *= scale;
    }

    return normalized;
}

/**
 * @brief Validate shell parameters
 *
 * @throws InvalidArgumentException if any validation fails
 */
void validate_shell_params(
    int am,
    const std::vector<Real>& exponents,
    const std::vector<Real>& coefficients)
{
    // Check angular momentum range
    if (am < 0 || am > MAX_ANGULAR_MOMENTUM) {
        std::ostringstream oss;
        oss << "Shell: angular momentum " << am
            << " is outside valid range [0, " << MAX_ANGULAR_MOMENTUM << "]";
        throw InvalidArgumentException(oss.str());
    }

    // Check that we have at least one primitive
    if (exponents.empty()) {
        throw InvalidArgumentException("Shell: must have at least one primitive Gaussian");
    }

    // Check that exponents and coefficients have the same size
    if (exponents.size() != coefficients.size()) {
        std::ostringstream oss;
        oss << "Shell: exponents size (" << exponents.size()
            << ") does not match coefficients size (" << coefficients.size() << ")";
        throw InvalidArgumentException(oss.str());
    }

    // Check that all exponents are positive
    for (Size i = 0; i < exponents.size(); ++i) {
        if (exponents[i] <= 0.0) {
            std::ostringstream oss;
            oss << "Shell: exponent[" << i << "] = " << exponents[i]
                << " is not positive";
            throw InvalidArgumentException(oss.str());
        }
    }
}

}  // anonymous namespace

// =============================================================================
// Constructors
// =============================================================================

Shell::Shell(AngularMomentum am,
             Point3D center,
             std::vector<Real> exponents,
             std::vector<Real> coefficients)
    : am_(to_int(am))
    , center_(center)
    , exponents_(std::move(exponents))
    , coefficients_()  // Will be set after normalization
{
    validate_shell_params(am_, exponents_, coefficients);
    coefficients_ = normalize_coefficients(am_, exponents_, coefficients);
}

Shell::Shell(int am,
             Point3D center,
             std::vector<Real> exponents,
             std::vector<Real> coefficients)
    : am_(am)
    , center_(center)
    , exponents_(std::move(exponents))
    , coefficients_()  // Will be set after normalization
{
    validate_shell_params(am_, exponents_, coefficients);
    coefficients_ = normalize_coefficients(am_, exponents_, coefficients);
}

Shell::Shell(PreNormalizedTag /* tag */,
             AngularMomentum am,
             Point3D center,
             std::vector<Real> exponents,
             std::vector<Real> coefficients)
    : am_(to_int(am))
    , center_(center)
    , exponents_(std::move(exponents))
    , coefficients_(std::move(coefficients))
{
    validate_shell_params(am_, exponents_, coefficients_);
}

Shell::Shell(PreNormalizedTag /* tag */,
             int am,
             Point3D center,
             std::vector<Real> exponents,
             std::vector<Real> coefficients)
    : am_(am)
    , center_(center)
    , exponents_(std::move(exponents))
    , coefficients_(std::move(coefficients))
{
    validate_shell_params(am_, exponents_, coefficients_);
}

// =============================================================================
// Accessors
// =============================================================================

Real Shell::exponent(Size i) const {
    if (i >= exponents_.size()) {
        std::ostringstream oss;
        oss << "Shell::exponent: index " << i
            << " is out of bounds (size = " << exponents_.size() << ")";
        throw InvalidArgumentException(oss.str());
    }
    return exponents_[i];
}

Real Shell::coefficient(Size i) const {
    if (i >= coefficients_.size()) {
        std::ostringstream oss;
        oss << "Shell::coefficient: index " << i
            << " is out of bounds (size = " << coefficients_.size() << ")";
        throw InvalidArgumentException(oss.str());
    }
    return coefficients_[i];
}

}  // namespace libaccint
