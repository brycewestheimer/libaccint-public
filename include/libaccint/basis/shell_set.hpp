// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2026 Bryce M. Westheimer. All rights reserved.

#pragma once

/// @file shell_set.hpp
/// @brief ShellSet class for batched integral computation
///
/// ShellSet groups shells with identical angular momentum and primitive count
/// for efficient GPU-parallel and vectorized CPU computation.

#include <libaccint/basis/shell.hpp>
#include <libaccint/core/types.hpp>

#include <cstddef>
#include <functional>
#include <mutex>
#include <span>
#include <vector>

namespace libaccint {

// =============================================================================
// ShellSetKey
// =============================================================================

/// @brief Key for organizing ShellSets in hash maps
///
/// Identifies a ShellSet by its angular momentum and number of primitives.
/// Provides operator== and a hash specialization for use in std::unordered_map.
struct ShellSetKey {
    int angular_momentum{0};
    int n_primitives{0};

    constexpr ShellSetKey() = default;
    constexpr ShellSetKey(int am, int n_prim)
        : angular_momentum(am), n_primitives(n_prim) {}

    [[nodiscard]] constexpr bool operator==(const ShellSetKey& other) const noexcept = default;

    /// @brief Comparison for std::map ordering
    [[nodiscard]] constexpr bool operator<(const ShellSetKey& other) const noexcept {
        if (angular_momentum != other.angular_momentum)
            return angular_momentum < other.angular_momentum;
        return n_primitives < other.n_primitives;
    }
};

}  // namespace libaccint

/// @brief Hash specialization for ShellSetKey
template<>
struct std::hash<libaccint::ShellSetKey> {
    [[nodiscard]] std::size_t operator()(const libaccint::ShellSetKey& key) const noexcept {
        // Combine hashes using a standard technique (boost-style)
        std::size_t h = std::hash<int>{}(key.angular_momentum);
        h ^= std::hash<int>{}(key.n_primitives) + 0x9e3779b9 + (h << 6) + (h >> 2);
        return h;
    }
};

namespace libaccint {

// =============================================================================
// ShellSetDataSoA
// =============================================================================

/// @brief Structure-of-Arrays data layout for a set of shells
///
/// Stores shell data in SoA layout for efficient GPU access and vectorized
/// CPU computation. All shells in the set share the same angular momentum
/// and primitive count, so primitive data is laid out contiguously:
///   - exponents[i * K + k] is the k-th exponent of the i-th shell
///   - coefficients[i * K + k] is the k-th coefficient of the i-th shell
///
/// Center coordinates are stored as separate x, y, z arrays indexed by
/// shell within the set.
struct ShellSetDataSoA {
    // Shell centers (indexed by shell within set) [n_shells]
    std::vector<Real> center_x;
    std::vector<Real> center_y;
    std::vector<Real> center_z;

    // Primitive data: flat arrays [n_shells * K], contiguous per shell
    std::vector<Real> exponents;
    std::vector<Real> coefficients;

    // Indexing arrays (indexed by shell within set) [n_shells]
    std::vector<Index> shell_indices;       ///< Original shell indices in basis
    std::vector<Index> atom_indices;        ///< Atom each shell belongs to
    std::vector<Index> function_offsets;    ///< Basis function offset for each shell

    /// @brief Number of shells in this set
    [[nodiscard]] Size n_shells() const noexcept { return center_x.size(); }

    /// @brief Total number of primitives across all shells
    [[nodiscard]] Size n_total_primitives() const noexcept { return exponents.size(); }
};

// =============================================================================
// ShellSet
// =============================================================================

/// @brief A set of shells with identical angular momentum and contraction degree
///
/// ShellSet groups shells that share the same angular momentum (L) and number
/// of primitives (K). This uniformity enables efficient batched computation
/// on GPUs where uniform work across threads is critical.
///
/// Invariant: all shells in a ShellSet have the same L and the same K.
///
/// SoA data is lazily constructed on first access via soa_data(), with
/// thread-safe initialization guaranteed by std::call_once.
///
/// @note ShellSets are typically constructed automatically when building
/// a BasisSet, not by users directly.
class ShellSet {
public:
    /// @brief Default constructor (creates an empty ShellSet)
    ShellSet() = default;

    /// @brief Construct a ShellSet from a vector of shell references
    ///
    /// Validates that all shells have the same angular momentum and primitive
    /// count. Stores copies of the shells internally.
    ///
    /// @param shells Vector of references to shells to include
    /// @throws InvalidArgumentException if shells is empty
    /// @throws InvalidArgumentException if shells have mismatched L or K
    explicit ShellSet(std::span<const std::reference_wrapper<const Shell>> shells);

    /// @brief Construct a ShellSet with specified properties (no shells yet)
    ///
    /// Creates an empty ShellSet that only accepts shells with matching
    /// angular momentum and primitive count via add_shell().
    ///
    /// @param am Angular momentum for all shells in this set
    /// @param n_primitives Number of primitives (contraction degree)
    /// @throws InvalidArgumentException if am is out of range [0, MAX_ANGULAR_MOMENTUM]
    /// @throws InvalidArgumentException if n_primitives < 1
    ShellSet(int am, int n_primitives);

    /// @brief Add a shell to this set
    ///
    /// The shell must have the same angular momentum and primitive count
    /// as the ShellSet. Adding a shell invalidates any previously cached
    /// SoA data.
    ///
    /// @param shell Shell to add (must have matching AM and n_primitives)
    /// @throws InvalidArgumentException if shell properties don't match
    void add_shell(const Shell& shell);

    // =========================================================================
    // Accessors
    // =========================================================================

    /// @brief Get angular momentum (same for all shells in set)
    [[nodiscard]] int angular_momentum() const noexcept { return am_; }

    /// @brief Get angular momentum as enum
    [[nodiscard]] AngularMomentum angular_momentum_enum() const noexcept {
        return static_cast<AngularMomentum>(am_);
    }

    /// @brief Get number of primitives per shell (same for all)
    [[nodiscard]] int n_primitives_per_shell() const noexcept { return n_primitives_; }

    /// @brief Get number of shells in this set
    [[nodiscard]] Size n_shells() const noexcept { return shells_.size(); }

    /// @brief Get number of Cartesian basis functions per shell
    [[nodiscard]] int n_functions_per_shell() const noexcept {
        return n_cartesian(am_);
    }

    /// @brief Total number of basis functions across all shells
    [[nodiscard]] Size n_total_functions() const noexcept {
        return n_shells() * static_cast<Size>(n_functions_per_shell());
    }

    /// @brief Access individual shell by index
    /// @throws InvalidArgumentException if index out of bounds
    [[nodiscard]] const Shell& shell(Size i) const;

    /// @brief Access all shells as a span
    [[nodiscard]] std::span<const Shell> shells() const noexcept {
        return shells_;
    }

    /// @brief Get the ShellSetKey identifying this set
    [[nodiscard]] ShellSetKey key() const noexcept {
        return ShellSetKey{am_, n_primitives_};
    }

    /// @brief Check if the ShellSet contains any shells
    [[nodiscard]] bool empty() const noexcept { return shells_.empty(); }

    // =========================================================================
    // SoA Data Access
    // =========================================================================

    /// @brief Get SoA data for efficient computation (lazy initialization)
    ///
    /// Thread-safe via std::call_once. Returns a reference to the cached
    /// SoA data, constructing it on first access.
    ///
    /// @return Const reference to the SoA data
    /// @throws InvalidStateException if ShellSet is empty
    [[nodiscard]] const ShellSetDataSoA& soa_data() const;

    /// @brief Check if SoA data has been built
    [[nodiscard]] bool soa_ready() const noexcept { return soa_initialized_; }

private:
    int am_{0};
    int n_primitives_{0};
    std::vector<Shell> shells_;

    // Lazy-initialized SoA data with thread-safe init
    mutable ShellSetDataSoA soa_data_;
    mutable std::once_flag soa_init_flag_;
    mutable bool soa_initialized_{false};

    /// @brief Build SoA data from stored shells
    void build_soa_data() const;
};

}  // namespace libaccint
