// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2026 Bryce M. Westheimer. All rights reserved.

#pragma once

/// @file operator_types.hpp
/// @brief Operator classification, parameter structs, and type definitions

#include <libaccint/core/types.hpp>
#include <array>
#include <string_view>
#include <variant>
#include <vector>

namespace libaccint {

// ============================================================================
// Operator Kind Enumeration
// ============================================================================

/// Enumeration of supported integral operators
enum class OperatorKind {
    // One-electron operators
    Overlap,                ///< Overlap operator (no parameters)
    Kinetic,                ///< Kinetic energy operator (no parameters)
    Nuclear,                ///< Nuclear attraction (requires PointChargeParams)
    PointCharge,            ///< Point charge interaction (requires PointChargeParams)
    DistributedMultipole,   ///< Distributed multipole operator (requires DistributedMultipoleParams)
    ProjectionOperator,     ///< Projection operator (requires ProjectionOperatorParams)

    // Two-electron operators
    Coulomb,                ///< Coulomb operator 1/r₁₂ (no parameters)
    ErfCoulomb,             ///< Short-range erf(ω*r₁₂)/r₁₂ (requires RangeSeparatedParams)
    ErfcCoulomb,            ///< Long-range erfc(ω*r₁₂)/r₁₂ (requires RangeSeparatedParams)

    // Property integral operators (Phase 17)
    ElectricDipole,         ///< Electric dipole moment <μ|r|ν> — 3-component vector
    ElectricQuadrupole,     ///< Electric quadrupole moment <μ|rr|ν> — 6-component symmetric tensor
    ElectricOctupole,       ///< Electric octupole moment <μ|rrr|ν> — 10-component symmetric tensor
    LinearMomentum,         ///< Linear momentum <μ|-i∇|ν> — 3-component anti-Hermitian vector
    AngularMomentum,        ///< Angular momentum <μ|r×(-i∇)|ν> — 3-component anti-Hermitian vector
};

// ============================================================================
// Operator Parameter Structs
// ============================================================================

/// Structure-of-Arrays layout for nuclear positions and charges
/// Used by Nuclear and PointCharge operators
struct PointChargeParams {
    std::vector<Real> x;      ///< X-coordinates of charge centers
    std::vector<Real> y;      ///< Y-coordinates of charge centers
    std::vector<Real> z;      ///< Z-coordinates of charge centers
    std::vector<Real> charge; ///< Charges (atomic numbers for Nuclear, arbitrary for PointCharge)

    /// Number of charge centers
    [[nodiscard]] Size n_centers() const noexcept {
        return charge.size();
    }
};

/// Range-separated Coulomb parameters
/// Used by ErfCoulomb and ErfcCoulomb operators
struct RangeSeparatedParams {
    Real omega{0.0}; ///< Range-separation parameter ω (in atomic units)
};

/// Distributed multipole operator parameters
/// Structure-of-Arrays layout for external multipole sites
struct DistributedMultipoleParams {
    // Site positions
    std::vector<Real> x;      ///< X-coordinates of multipole sites
    std::vector<Real> y;      ///< Y-coordinates of multipole sites
    std::vector<Real> z;      ///< Z-coordinates of multipole sites

    // Rank 0: charges
    std::vector<Real> charges; ///< Site charges (q)

    // Rank 1: dipoles (empty if max_rank < 1)
    std::vector<Real> dipole_x;  ///< Dipole x-components
    std::vector<Real> dipole_y;  ///< Dipole y-components
    std::vector<Real> dipole_z;  ///< Dipole z-components

    // Rank 2: quadrupoles (empty if max_rank < 2)
    std::vector<Real> quad_xx;   ///< Quadrupole xx-components
    std::vector<Real> quad_xy;   ///< Quadrupole xy-components
    std::vector<Real> quad_xz;   ///< Quadrupole xz-components
    std::vector<Real> quad_yy;   ///< Quadrupole yy-components
    std::vector<Real> quad_yz;   ///< Quadrupole yz-components
    std::vector<Real> quad_zz;   ///< Quadrupole zz-components

    /// Number of multipole sites
    [[nodiscard]] Size n_sites() const noexcept { return charges.size(); }

    /// Maximum multipole rank present (0=charge, 1=dipole, 2=quadrupole)
    [[nodiscard]] int max_rank() const noexcept {
        if (!quad_xx.empty()) return 2;
        if (!dipole_x.empty()) return 1;
        return 0;
    }

    /// Validate internal consistency
    [[nodiscard]] bool is_valid() const noexcept {
        Size n = n_sites();
        if (x.size() != n || y.size() != n || z.size() != n) return false;
        if (!dipole_x.empty() && (dipole_x.size() != n || dipole_y.size() != n || dipole_z.size() != n)) return false;
        if (!quad_xx.empty() && (quad_xx.size() != n || quad_xy.size() != n || quad_xz.size() != n ||
            quad_yy.size() != n || quad_yz.size() != n || quad_zz.size() != n)) return false;
        return true;
    }
};

/// Projection operator parameters
/// Stores projector coefficient matrix and weights: P = C * diag(w) * C^T
struct ProjectionOperatorParams {
    std::vector<Real> coefficients; ///< Flattened coefficient matrix (n_basis × n_projectors, column-major)
    std::vector<Real> weights;      ///< Weight for each projector function
    Size n_basis{0};                ///< Number of basis functions
    Size n_projectors{0};           ///< Number of projector functions

    /// Access coefficient C(mu, k)
    [[nodiscard]] Real coefficient(Size mu, Size k) const noexcept {
        return coefficients[k * n_basis + mu]; // column-major
    }

    /// Validate internal consistency
    [[nodiscard]] bool is_valid() const noexcept {
        return coefficients.size() == n_basis * n_projectors &&
               weights.size() == n_projectors;
    }
};

/// Parameters for origin-dependent property integrals (dipole, quadrupole, etc.)
struct OriginParams {
    std::array<Real, 3> origin{0.0, 0.0, 0.0}; ///< Gauge/expansion origin
};

// ============================================================================
// Operator Parameter Variant
// ============================================================================

/// Type-safe variant holding operator-specific parameters
/// - std::monostate: parameter-free operators (Overlap, Kinetic, Coulomb)
/// - PointChargeParams: Nuclear, PointCharge
/// - RangeSeparatedParams: ErfCoulomb, ErfcCoulomb
/// - DistributedMultipoleParams: DistributedMultipole
/// - ProjectionOperatorParams: ProjectionOperator
/// - OriginParams: ElectricDipole, ElectricQuadrupole, ElectricOctupole, AngularMomentum
using OperatorParams = std::variant<
    std::monostate,              // Parameter-free operators
    PointChargeParams,           // Nuclear, PointCharge
    RangeSeparatedParams,        // ErfCoulomb, ErfcCoulomb
    DistributedMultipoleParams,  // DistributedMultipole
    ProjectionOperatorParams,    // ProjectionOperator
    OriginParams                 // Property integrals with origin
>;

// ============================================================================
// Classification Functions
// ============================================================================

/// Check if operator is a one-electron operator
[[nodiscard]] constexpr bool is_one_electron(OperatorKind kind) noexcept {
    switch (kind) {
        case OperatorKind::Overlap:
        case OperatorKind::Kinetic:
        case OperatorKind::Nuclear:
        case OperatorKind::PointCharge:
        case OperatorKind::DistributedMultipole:
        case OperatorKind::ProjectionOperator:
        case OperatorKind::ElectricDipole:
        case OperatorKind::ElectricQuadrupole:
        case OperatorKind::ElectricOctupole:
        case OperatorKind::LinearMomentum:
        case OperatorKind::AngularMomentum:
            return true;
        case OperatorKind::Coulomb:
        case OperatorKind::ErfCoulomb:
        case OperatorKind::ErfcCoulomb:
            return false;
    }
    return false;  // Unreachable, but silences compiler warnings
}

/// Check if operator is a two-electron operator
[[nodiscard]] constexpr bool is_two_electron(OperatorKind kind) noexcept {
    switch (kind) {
        case OperatorKind::Coulomb:
        case OperatorKind::ErfCoulomb:
        case OperatorKind::ErfcCoulomb:
            return true;
        case OperatorKind::Overlap:
        case OperatorKind::Kinetic:
        case OperatorKind::Nuclear:
        case OperatorKind::PointCharge:
        case OperatorKind::DistributedMultipole:
        case OperatorKind::ProjectionOperator:
        case OperatorKind::ElectricDipole:
        case OperatorKind::ElectricQuadrupole:
        case OperatorKind::ElectricOctupole:
        case OperatorKind::LinearMomentum:
        case OperatorKind::AngularMomentum:
            return false;
    }
    return false;  // Unreachable, but silences compiler warnings
}

/// Check if operator is a multi-component property integral
[[nodiscard]] constexpr bool is_multi_component(OperatorKind kind) noexcept {
    switch (kind) {
        case OperatorKind::ElectricDipole:
        case OperatorKind::ElectricQuadrupole:
        case OperatorKind::ElectricOctupole:
        case OperatorKind::LinearMomentum:
        case OperatorKind::AngularMomentum:
            return true;
        default:
            return false;
    }
}

/// Check if operator produces anti-Hermitian (anti-symmetric) matrices
[[nodiscard]] constexpr bool is_anti_hermitian(OperatorKind kind) noexcept {
    switch (kind) {
        case OperatorKind::LinearMomentum:
        case OperatorKind::AngularMomentum:
            return true;
        default:
            return false;
    }
}

/// Check if operator is a property integral (not part of the Hamiltonian)
[[nodiscard]] constexpr bool is_property_integral(OperatorKind kind) noexcept {
    switch (kind) {
        case OperatorKind::ElectricDipole:
        case OperatorKind::ElectricQuadrupole:
        case OperatorKind::ElectricOctupole:
        case OperatorKind::LinearMomentum:
        case OperatorKind::AngularMomentum:
            return true;
        default:
            return false;
    }
}

/// Return the number of integral components for a given operator kind
[[nodiscard]] constexpr int component_count(OperatorKind kind) noexcept {
    switch (kind) {
        case OperatorKind::ElectricDipole:      return 3;
        case OperatorKind::ElectricQuadrupole:   return 6;
        case OperatorKind::ElectricOctupole:     return 10;
        case OperatorKind::LinearMomentum:       return 3;
        case OperatorKind::AngularMomentum:      return 3;
        default:                                 return 1;
    }
}

/// Return a human-readable name for the operator kind
[[nodiscard]] constexpr std::string_view operator_name(OperatorKind kind) noexcept {
    switch (kind) {
        case OperatorKind::Overlap:              return "Overlap";
        case OperatorKind::Kinetic:              return "Kinetic";
        case OperatorKind::Nuclear:              return "Nuclear";
        case OperatorKind::PointCharge:          return "PointCharge";
        case OperatorKind::DistributedMultipole: return "DistributedMultipole";
        case OperatorKind::ProjectionOperator:   return "ProjectionOperator";
        case OperatorKind::Coulomb:              return "Coulomb";
        case OperatorKind::ErfCoulomb:           return "ErfCoulomb";
        case OperatorKind::ErfcCoulomb:          return "ErfcCoulomb";
        case OperatorKind::ElectricDipole:       return "ElectricDipole";
        case OperatorKind::ElectricQuadrupole:   return "ElectricQuadrupole";
        case OperatorKind::ElectricOctupole:     return "ElectricOctupole";
        case OperatorKind::LinearMomentum:       return "LinearMomentum";
        case OperatorKind::AngularMomentum:      return "AngularMomentum";
    }
    return "";
}

/// Convert OperatorKind to string representation
[[nodiscard]] constexpr std::string_view to_string(OperatorKind kind) noexcept {
    return operator_name(kind);
}

}  // namespace libaccint
