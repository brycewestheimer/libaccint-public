// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2026 Bryce M. Westheimer. All rights reserved.

#pragma once

/// @file shell.hpp
/// @brief Shell class representing a contracted Gaussian basis function

#include <libaccint/core/types.hpp>
#include <vector>
#include <span>

namespace libaccint {

/// @brief Tag type to indicate that coefficients are already normalized
struct PreNormalizedTag {};

/// @brief Constant to indicate pre-normalized coefficients
inline constexpr PreNormalizedTag pre_normalized{};

/**
 * @brief Contracted Gaussian shell
 *
 * A Shell represents a contracted Gaussian basis function centered at a point
 * in space with a given angular momentum. It consists of:
 *   - A center position (Point3D)
 *   - An angular momentum quantum number (l)
 *   - A set of primitive Gaussian exponents (α_i)
 *   - A set of contraction coefficients (c_i)
 *
 * The basis function is defined as:
 *   φ(r) = Σ_i c_i * (x-A_x)^i (y-A_y)^j (z-A_z)^k * exp(-α_i |r-A|²)
 *
 * where i+j+k = l (angular momentum).
 *
 * Normalization:
 * By default, the constructor normalizes the contraction coefficients to ensure
 * the self-overlap of the shell equals 1. This includes both primitive
 * normalization and contraction normalization. To use pre-normalized
 * coefficients, use the constructor with PreNormalizedTag.
 *
 * Tracking Indices:
 * Shells maintain three indices for their position within a BasisSet:
 *   - atom_index: Index of the atom this shell is centered on
 *   - shell_index: Index of this shell within the basis set
 *   - function_index: Starting basis function index (offset) for this shell
 */
class Shell {
public:
    /// @brief Default constructor (creates an invalid shell)
    Shell() = default;

    /**
     * @brief Construct a shell with automatic normalization
     *
     * Normalizes coefficients to ensure self-overlap equals 1.
     * The normalization includes:
     *   1. Primitive normalization: N_prim = (2α/π)^(3/4) * (4α)^(L/2) / sqrt((2L-1)!!)
     *   2. Contraction normalization: scale to make self-overlap = 1
     *
     * @param am Angular momentum as enum
     * @param center Center position
     * @param exponents Primitive exponents (must be positive)
     * @param coefficients Contraction coefficients (un-normalized)
     * @throws InvalidArgumentException if validation fails
     */
    Shell(AngularMomentum am,
          Point3D center,
          std::vector<Real> exponents,
          std::vector<Real> coefficients);

    /**
     * @brief Construct a shell with automatic normalization (int AM)
     *
     * @param am Angular momentum as integer [0, MAX_ANGULAR_MOMENTUM]
     * @param center Center position
     * @param exponents Primitive exponents (must be positive)
     * @param coefficients Contraction coefficients (un-normalized)
     * @throws InvalidArgumentException if validation fails
     */
    Shell(int am,
          Point3D center,
          std::vector<Real> exponents,
          std::vector<Real> coefficients);

    /**
     * @brief Construct a shell with pre-normalized coefficients
     *
     * Use this constructor when coefficients are already fully normalized
     * (both primitive and contraction normalization applied).
     *
     * @param tag PreNormalizedTag indicating coefficients are already normalized
     * @param am Angular momentum as enum
     * @param center Center position
     * @param exponents Primitive exponents (must be positive)
     * @param coefficients Pre-normalized contraction coefficients
     * @throws InvalidArgumentException if validation fails
     */
    Shell(PreNormalizedTag tag,
          AngularMomentum am,
          Point3D center,
          std::vector<Real> exponents,
          std::vector<Real> coefficients);

    /**
     * @brief Construct a shell with pre-normalized coefficients (int AM)
     *
     * @param tag PreNormalizedTag indicating coefficients are already normalized
     * @param am Angular momentum as integer [0, MAX_ANGULAR_MOMENTUM]
     * @param center Center position
     * @param exponents Primitive exponents (must be positive)
     * @param coefficients Pre-normalized contraction coefficients
     * @throws InvalidArgumentException if validation fails
     */
    Shell(PreNormalizedTag tag,
          int am,
          Point3D center,
          std::vector<Real> exponents,
          std::vector<Real> coefficients);

    // =========================================================================
    // Accessors
    // =========================================================================

    /// @brief Get angular momentum as integer
    [[nodiscard]] int angular_momentum() const noexcept { return am_; }

    /// @brief Get angular momentum as enum
    [[nodiscard]] AngularMomentum angular_momentum_enum() const noexcept {
        return static_cast<AngularMomentum>(am_);
    }

    /// @brief Get shell center position
    [[nodiscard]] const Point3D& center() const noexcept { return center_; }

    /// @brief Get number of primitive Gaussians
    [[nodiscard]] Size n_primitives() const noexcept { return exponents_.size(); }

    /// @brief Get number of Cartesian basis functions
    [[nodiscard]] int n_functions() const noexcept { return n_cartesian(am_); }

    /// @brief Get all exponents as span
    [[nodiscard]] std::span<const Real> exponents() const noexcept {
        return std::span<const Real>(exponents_);
    }

    /// @brief Get all coefficients as span
    [[nodiscard]] std::span<const Real> coefficients() const noexcept {
        return std::span<const Real>(coefficients_);
    }

    /// @brief Get single exponent by index
    /// @throws InvalidArgumentException if index out of bounds
    [[nodiscard]] Real exponent(Size i) const;

    /// @brief Get single coefficient by index
    /// @throws InvalidArgumentException if index out of bounds
    [[nodiscard]] Real coefficient(Size i) const;

    /// @brief Check if shell is valid (has primitives)
    [[nodiscard]] bool valid() const noexcept { return !exponents_.empty(); }

    // =========================================================================
    // Tracking Indices
    // =========================================================================

    /// @brief Get atom index (-1 if not set)
    [[nodiscard]] Index atom_index() const noexcept { return atom_index_; }

    /// @brief Set atom index
    void set_atom_index(Index idx) noexcept { atom_index_ = idx; }

    /// @brief Get shell index (-1 if not set)
    [[nodiscard]] Index shell_index() const noexcept { return shell_index_; }

    /// @brief Set shell index
    void set_shell_index(Index idx) noexcept { shell_index_ = idx; }

    /// @brief Get basis function index/offset (-1 if not set)
    [[nodiscard]] Index function_index() const noexcept { return function_index_; }

    /// @brief Set basis function index/offset
    void set_function_index(Index idx) noexcept { function_index_ = idx; }

private:
    /// Angular momentum quantum number
    int am_{0};

    /// Center position in 3D space
    Point3D center_{};

    /// Primitive exponents (α_i > 0)
    std::vector<Real> exponents_;

    /// Normalized contraction coefficients
    std::vector<Real> coefficients_;

    /// Index of atom this shell is centered on (-1 if not set)
    Index atom_index_{-1};

    /// Index of this shell in the basis set (-1 if not set)
    Index shell_index_{-1};

    /// Starting basis function index for this shell (-1 if not set)
    Index function_index_{-1};
};

}  // namespace libaccint
