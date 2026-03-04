// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2026 Bryce M. Westheimer. All rights reserved.

#pragma once

/// @file shell_set_triple.hpp
/// @brief ShellSetTriple class for three-center integral computation
///
/// Groups shells into triples (orbital a, orbital b, auxiliary P) for efficient
/// batched three-center integral computation in density fitting.

#include <libaccint/basis/shell_set.hpp>
#include <libaccint/basis/shell_set_pair.hpp>
#include <libaccint/core/types.hpp>

#include <cstddef>
#include <functional>
#include <span>
#include <vector>

namespace libaccint {

// Forward declarations
class AuxiliaryBasisSet;
class BasisSet;

// =============================================================================
// ShellSetTripleKey
// =============================================================================

/// @brief Key for organizing ShellSetTriples by angular momenta and primitives
struct ShellSetTripleKey {
    int am_a{0};         ///< Angular momentum of orbital shell a
    int am_b{0};         ///< Angular momentum of orbital shell b
    int am_P{0};         ///< Angular momentum of auxiliary shell P
    int n_prim_a{0};     ///< Number of primitives in shell a
    int n_prim_b{0};     ///< Number of primitives in shell b
    int n_prim_P{0};     ///< Number of primitives in shell P

    constexpr ShellSetTripleKey() = default;
    constexpr ShellSetTripleKey(int a, int b, int p, int na, int nb, int np)
        : am_a(a), am_b(b), am_P(p), n_prim_a(na), n_prim_b(nb), n_prim_P(np) {}

    [[nodiscard]] constexpr bool operator==(const ShellSetTripleKey& other) const noexcept = default;

    [[nodiscard]] constexpr bool operator<(const ShellSetTripleKey& other) const noexcept {
        if (am_a != other.am_a) return am_a < other.am_a;
        if (am_b != other.am_b) return am_b < other.am_b;
        if (am_P != other.am_P) return am_P < other.am_P;
        if (n_prim_a != other.n_prim_a) return n_prim_a < other.n_prim_a;
        if (n_prim_b != other.n_prim_b) return n_prim_b < other.n_prim_b;
        return n_prim_P < other.n_prim_P;
    }
};

}  // namespace libaccint

/// @brief Hash specialization for ShellSetTripleKey
template<>
struct std::hash<libaccint::ShellSetTripleKey> {
    [[nodiscard]] std::size_t operator()(const libaccint::ShellSetTripleKey& key) const noexcept {
        std::size_t h = std::hash<int>{}(key.am_a);
        h ^= std::hash<int>{}(key.am_b) + 0x9e3779b9 + (h << 6) + (h >> 2);
        h ^= std::hash<int>{}(key.am_P) + 0x9e3779b9 + (h << 6) + (h >> 2);
        h ^= std::hash<int>{}(key.n_prim_a) + 0x9e3779b9 + (h << 6) + (h >> 2);
        h ^= std::hash<int>{}(key.n_prim_b) + 0x9e3779b9 + (h << 6) + (h >> 2);
        h ^= std::hash<int>{}(key.n_prim_P) + 0x9e3779b9 + (h << 6) + (h >> 2);
        return h;
    }
};

namespace libaccint {

// =============================================================================
// ShellTriple
// =============================================================================

/// @brief A single triple of shells for three-center integral computation
///
/// Represents (a, b | P) where a, b are orbital shells and P is an auxiliary shell.
struct ShellTriple {
    Size shell_a_idx{0};  ///< Index of orbital shell a in basis set
    Size shell_b_idx{0};  ///< Index of orbital shell b in basis set
    Size shell_P_idx{0};  ///< Index of auxiliary shell P in auxiliary basis

    const Shell* shell_a{nullptr};  ///< Pointer to orbital shell a
    const Shell* shell_b{nullptr};  ///< Pointer to orbital shell b
    const Shell* shell_P{nullptr};  ///< Pointer to auxiliary shell P

    /// @brief Check validity of the triple
    [[nodiscard]] bool is_valid() const noexcept {
        return shell_a != nullptr && shell_b != nullptr && shell_P != nullptr;
    }
};

// =============================================================================
// ShellSetTriple
// =============================================================================

/// @brief A batch of shell triples with uniform angular momenta and primitives
///
/// Groups (ab|P) shell combinations that share the same angular momentum
/// signature (L_a, L_b, L_P) and primitive counts (K_a, K_b, K_P).
/// This uniformity enables efficient GPU batching and vectorized CPU computation.
///
/// The class supports:
///   - Iteration over shell triples
///   - Symmetry handling for a <-> b
///   - Direct indexing for parallel work distribution
class ShellSetTriple {
public:
    /// @brief Default constructor (empty triple batch)
    ShellSetTriple() = default;

    /// @brief Construct from orbital pair and auxiliary shell sets
    ///
    /// Creates all (a, b, P) combinations where:
    ///   - a comes from shell_set_a
    ///   - b comes from shell_set_b
    ///   - P comes from shell_set_P
    ///
    /// @param shell_set_a First orbital ShellSet
    /// @param shell_set_b Second orbital ShellSet
    /// @param shell_set_P Auxiliary ShellSet
    /// @param symmetric If true, only include a <= b combinations
    ShellSetTriple(const ShellSet& shell_set_a,
                   const ShellSet& shell_set_b,
                   const ShellSet& shell_set_P,
                   bool symmetric = true);

    // =========================================================================
    // Properties
    // =========================================================================

    /// @brief Get the key identifying this triple batch
    [[nodiscard]] ShellSetTripleKey key() const noexcept { return key_; }

    /// @brief Number of triples in this batch
    [[nodiscard]] Size size() const noexcept { return triples_.size(); }

    /// @brief Check if batch is empty
    [[nodiscard]] bool empty() const noexcept { return triples_.empty(); }

    /// @brief Angular momentum of shell a
    [[nodiscard]] int am_a() const noexcept { return key_.am_a; }

    /// @brief Angular momentum of shell b
    [[nodiscard]] int am_b() const noexcept { return key_.am_b; }

    /// @brief Angular momentum of shell P
    [[nodiscard]] int am_P() const noexcept { return key_.am_P; }

    /// @brief Total angular momentum La + Lb + LP
    [[nodiscard]] int total_am() const noexcept {
        return key_.am_a + key_.am_b + key_.am_P;
    }

    /// @brief Number of primitives in shell a
    [[nodiscard]] int n_primitives_a() const noexcept { return key_.n_prim_a; }

    /// @brief Number of primitives in shell b
    [[nodiscard]] int n_primitives_b() const noexcept { return key_.n_prim_b; }

    /// @brief Number of primitives in shell P
    [[nodiscard]] int n_primitives_P() const noexcept { return key_.n_prim_P; }

    /// @brief Number of Cartesian functions in shell a
    [[nodiscard]] int n_functions_a() const noexcept;

    /// @brief Number of Cartesian functions in shell b
    [[nodiscard]] int n_functions_b() const noexcept;

    /// @brief Number of Cartesian functions in shell P
    [[nodiscard]] int n_functions_P() const noexcept;

    /// @brief Check if this batch uses symmetric orbital pairs
    [[nodiscard]] bool is_symmetric() const noexcept { return symmetric_; }

    // =========================================================================
    // Triple Access
    // =========================================================================

    /// @brief Access a triple by index
    [[nodiscard]] const ShellTriple& operator[](Size idx) const { return triples_[idx]; }

    /// @brief Access a triple by index with bounds checking
    [[nodiscard]] const ShellTriple& at(Size idx) const;

    /// @brief Get all triples as a span
    [[nodiscard]] std::span<const ShellTriple> triples() const noexcept { return triples_; }

    // =========================================================================
    // Iterators
    // =========================================================================

    using iterator = std::vector<ShellTriple>::const_iterator;
    using const_iterator = std::vector<ShellTriple>::const_iterator;

    [[nodiscard]] const_iterator begin() const noexcept { return triples_.begin(); }
    [[nodiscard]] const_iterator end() const noexcept { return triples_.end(); }
    [[nodiscard]] const_iterator cbegin() const noexcept { return triples_.cbegin(); }
    [[nodiscard]] const_iterator cend() const noexcept { return triples_.cend(); }

private:
    // Factory needs access to rebind auxiliary pointers to stable
    // AuxiliaryBasisSet-owned shells.
    friend std::vector<ShellSetTriple> generate_shell_set_triples(
        const BasisSet& orbital,
        const AuxiliaryBasisSet& auxiliary,
        bool symmetric);

    ShellSetTripleKey key_;
    std::vector<ShellTriple> triples_;
    bool symmetric_{true};
};

// =============================================================================
// Factory Functions
// =============================================================================

/// @brief Generate all ShellSetTriples from orbital and auxiliary basis sets
///
/// Creates batches grouped by (L_a, L_b, L_P, K_a, K_b, K_P) for efficient
/// batched computation of three-center integrals.
///
/// @param orbital Orbital basis set
/// @param auxiliary Auxiliary basis set
/// @param symmetric If true, only include a <= b orbital pairs
/// @return Vector of ShellSetTriple batches
[[nodiscard]] std::vector<ShellSetTriple> generate_shell_set_triples(
    const BasisSet& orbital,
    const AuxiliaryBasisSet& auxiliary,
    bool symmetric = true);

/// @brief Estimate computational cost for a ShellSetTriple
///
/// @param triple The shell triple batch
/// @return Estimated FLOP count
[[nodiscard]] Size estimate_triple_cost(const ShellSetTriple& triple);

}  // namespace libaccint
