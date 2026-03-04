// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2026 Bryce M. Westheimer. All rights reserved.

#pragma once

/// @file auxiliary_basis_set.hpp
/// @brief AuxiliaryBasisSet class for density fitting / RI calculations
///
/// Manages auxiliary basis sets used for density fitting approximations.
/// Auxiliary functions are used to expand orbital products:
///   phi_a(r) * phi_b(r) ≈ sum_P C_ab^P * chi_P(r)
/// reducing the scaling of two-electron integrals from O(N^4) to O(N^3).

#include <libaccint/basis/shell.hpp>
#include <libaccint/core/types.hpp>

#include <cstddef>
#include <functional>
#include <memory>
#include <mutex>
#include <optional>
#include <span>
#include <vector>

namespace libaccint {

// Forward declaration
class BasisSet;

// =============================================================================
// FittingType
// =============================================================================

/// @brief Type of fitting for auxiliary basis set
enum class FittingType {
    RI,        ///< General resolution-of-identity
    JKFIT,     ///< Coulomb + exchange fitting (default for HF/DFT)
    JFIT,      ///< Coulomb-only fitting
    MP2FIT,    ///< MP2 correlation fitting
    RIFIT,     ///< Alias for general RI
    Unknown    ///< Unknown or unspecified type
};

/// @brief Convert FittingType to string representation
[[nodiscard]] constexpr const char* fitting_type_to_string(FittingType type) noexcept {
    switch (type) {
        case FittingType::RI:     return "RI";
        case FittingType::JKFIT:  return "JKFIT";
        case FittingType::JFIT:   return "JFIT";
        case FittingType::MP2FIT: return "MP2FIT";
        case FittingType::RIFIT:  return "RIFIT";
        case FittingType::Unknown: return "Unknown";
    }
    return "Unknown";
}

// =============================================================================
// AuxiliaryBasisSetSoA
// =============================================================================

/// @brief Structure-of-Arrays data layout for auxiliary basis sets
///
/// Optimized layout for GPU transfer and vectorized computation.
/// All primitive data is stored contiguously for each shell.
struct AuxiliaryBasisSetSoA {
    Size n_shells{0};           ///< Number of auxiliary shells
    Size n_primitives{0};       ///< Total number of primitives
    Size n_functions{0};        ///< Total number of auxiliary functions

    // Shell centers (indexed by shell) [n_shells]
    std::vector<Real> center_x;
    std::vector<Real> center_y;
    std::vector<Real> center_z;

    // Primitive data: flat arrays [n_primitives]
    std::vector<Real> exponents;
    std::vector<Real> coefficients;

    // Per-shell indexing [n_shells]
    std::vector<int> angular_momenta;         ///< AM per shell
    std::vector<Size> primitive_offsets;      ///< Offset into exponent array
    std::vector<Size> n_primitives_per_shell; ///< Primitives per shell
    std::vector<Size> function_offsets;       ///< Basis function offset per shell
    std::vector<Size> n_functions_per_shell;  ///< Functions per shell
};

// =============================================================================
// AuxiliaryBasisSet
// =============================================================================

/// @brief A collection of auxiliary basis functions for density fitting
///
/// AuxiliaryBasisSet manages auxiliary basis shells used for RI/DF calculations.
/// Key differences from orbital BasisSet:
///   - May have higher angular momentum than orbital basis
///   - Shells are not associated with atomic orbitals
///   - Contains fitting type metadata (JKFIT, RI, etc.)
///   - Can be paired with an orbital basis for validation
///
/// The class provides:
///   - Shell access and indexing
///   - Function-to-shell and shell-to-function mappings
///   - SoA data for GPU compatibility
///   - Optional association with orbital basis
///
/// @note SoA data is lazily computed and cached on first access.
class AuxiliaryBasisSet {
public:
    /// @brief Default constructor (creates empty auxiliary basis)
    AuxiliaryBasisSet() = default;

    /// @brief Non-copyable due to std::once_flag member
    AuxiliaryBasisSet(const AuxiliaryBasisSet&) = delete;
    AuxiliaryBasisSet& operator=(const AuxiliaryBasisSet&) = delete;

    /// @brief Move constructor
    AuxiliaryBasisSet(AuxiliaryBasisSet&& other) noexcept;

    /// @brief Move assignment
    AuxiliaryBasisSet& operator=(AuxiliaryBasisSet&& other) noexcept;

    /// @brief Construct from a vector of shells
    ///
    /// @param shells Auxiliary basis shells
    /// @param type Fitting type (RI, JKFIT, etc.)
    /// @param name Optional basis set name (e.g., "cc-pVDZ-RI")
    explicit AuxiliaryBasisSet(std::vector<Shell> shells,
                                FittingType type = FittingType::JKFIT,
                                std::string name = "");

    /// @brief Construct from shell span (copies shells)
    explicit AuxiliaryBasisSet(std::span<const Shell> shells,
                                FittingType type = FittingType::JKFIT,
                                std::string name = "");

    // =========================================================================
    // Shell Access
    // =========================================================================

    /// @brief Get the number of auxiliary shells
    [[nodiscard]] Size n_shells() const noexcept { return shells_.size(); }

    /// @brief Get the total number of auxiliary basis functions
    [[nodiscard]] Size n_functions() const noexcept { return n_functions_; }

    /// @brief Get the total number of primitives across all shells
    [[nodiscard]] Size n_primitives() const noexcept { return n_primitives_; }

    /// @brief Get the maximum angular momentum across all shells
    [[nodiscard]] int max_angular_momentum() const noexcept { return max_am_; }

    /// @brief Access a shell by index
    /// @throws InvalidArgumentException if index is out of bounds
    [[nodiscard]] const Shell& shell(Size i) const;

    /// @brief Access all shells as a span
    [[nodiscard]] std::span<const Shell> shells() const noexcept { return shells_; }

    /// @brief Check if the auxiliary basis is empty
    [[nodiscard]] bool empty() const noexcept { return shells_.empty(); }

    // =========================================================================
    // Indexing
    // =========================================================================

    /// @brief Get the basis function offset for a given shell
    /// @param shell_idx Index of the shell
    /// @return Starting basis function index for this shell
    /// @throws InvalidArgumentException if shell_idx is out of bounds
    [[nodiscard]] Size shell_to_function(Size shell_idx) const;

    /// @brief Get the shell containing a given basis function
    /// @param func_idx Index of the basis function
    /// @return Shell index containing this function
    /// @throws InvalidArgumentException if func_idx is out of bounds
    [[nodiscard]] Size function_to_shell(Size func_idx) const;

    /// @brief Get function offsets for all shells
    [[nodiscard]] std::span<const Size> function_offsets() const noexcept {
        return shell_to_func_;
    }

    // =========================================================================
    // Fitting Type and Metadata
    // =========================================================================

    /// @brief Get the fitting type
    [[nodiscard]] FittingType fitting_type() const noexcept { return fitting_type_; }

    /// @brief Set the fitting type
    void set_fitting_type(FittingType type) noexcept { fitting_type_ = type; }

    /// @brief Get the basis set name
    [[nodiscard]] const std::string& name() const noexcept { return name_; }

    /// @brief Set the basis set name
    void set_name(const std::string& name) { name_ = name; }

    // =========================================================================
    // Orbital Basis Pairing
    // =========================================================================

    /// @brief Associate this auxiliary basis with an orbital basis
    ///
    /// The association is optional but useful for validation and
    /// ensuring geometric consistency.
    ///
    /// @param orbital Reference to orbital basis set
    void set_orbital_basis(const BasisSet& orbital);

    /// @brief Check if an orbital basis is associated
    [[nodiscard]] bool has_orbital_basis() const noexcept {
        return orbital_basis_ != nullptr;
    }

    /// @brief Get the associated orbital basis
    /// @throws InvalidStateException if no orbital basis is set
    [[nodiscard]] const BasisSet& orbital_basis() const;

    /// @brief Clear the orbital basis association
    void clear_orbital_basis() noexcept { orbital_basis_ = nullptr; }

    // =========================================================================
    // SoA Data Access
    // =========================================================================

    /// @brief Get Structure-of-Arrays data for GPU transfer
    ///
    /// Lazily constructs and caches the SoA representation.
    /// Thread-safe via std::call_once.
    ///
    /// @return Const reference to cached SoA data
    [[nodiscard]] const AuxiliaryBasisSetSoA& soa_data() const;

    /// @brief Invalidate cached SoA data
    ///
    /// Call this if shells are modified after construction.
    void invalidate_soa_cache() noexcept { soa_valid_ = false; }

private:
    std::vector<Shell> shells_;                  ///< Auxiliary basis shells
    std::vector<Size> shell_to_func_;            ///< Function offset per shell
    std::vector<Size> func_to_shell_;            ///< Shell index per function
    Size n_functions_{0};                        ///< Total basis functions
    Size n_primitives_{0};                       ///< Total primitives
    int max_am_{0};                              ///< Maximum angular momentum
    FittingType fitting_type_{FittingType::JKFIT};
    std::string name_;

    const BasisSet* orbital_basis_{nullptr};     ///< Optional associated orbital basis

    mutable std::once_flag soa_init_flag_;       ///< For thread-safe lazy init
    mutable AuxiliaryBasisSetSoA soa_data_;      ///< Cached SoA data
    mutable bool soa_valid_{false};              ///< SoA cache validity

    /// @brief Build function indexing arrays
    void build_indices();

    /// @brief Construct SoA data (called via call_once)
    void build_soa_data() const;
};

// =============================================================================
// Factory Functions
// =============================================================================

/// @brief Load a built-in auxiliary basis set for given atoms
///
/// @param basis_name Name of auxiliary basis (e.g., "def2-JKFIT", "cc-pVDZ-RI")
/// @param atoms Atom symbols or atomic numbers
/// @param centers Atomic centers (x, y, z for each atom)
/// @return AuxiliaryBasisSet with shells for all atoms
/// @throws InvalidArgumentException if basis is not available for an atom
[[nodiscard]] AuxiliaryBasisSet load_auxiliary_basis(
    const std::string& basis_name,
    std::span<const int> atomic_numbers,
    std::span<const std::array<Real, 3>> centers);

/// @brief List available auxiliary basis sets
/// @return Vector of available basis set names
[[nodiscard]] std::vector<std::string> available_auxiliary_bases();

/// @brief Check if an auxiliary basis is available for all given atoms
[[nodiscard]] bool is_auxiliary_basis_available(
    const std::string& basis_name,
    std::span<const int> atomic_numbers);

}  // namespace libaccint
