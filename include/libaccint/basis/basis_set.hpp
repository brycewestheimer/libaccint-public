// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2026 Bryce M. Westheimer. All rights reserved.

#pragma once

/// @file basis_set.hpp
/// @brief BasisSet class that organizes shells into ShellSets and provides
///        iteration infrastructure for integral computation.

#include <libaccint/basis/shell.hpp>
#include <libaccint/basis/shell_set.hpp>
#include <libaccint/basis/shell_set_pair.hpp>
#include <libaccint/basis/shell_set_quartet.hpp>
#include <libaccint/core/types.hpp>

#include <atomic>
#include <memory>
#include <mutex>
#include <span>
#include <unordered_map>
#include <vector>

namespace libaccint {

/**
 * @brief A collection of Shell objects organized into ShellSets for integral computation
 *
 * BasisSet is the primary container for a molecular basis set. On construction
 * from a vector of Shell objects, it:
 *   1. Assigns sequential shell indices and basis function offsets
 *   2. Groups shells into ShellSets by (angular_momentum, n_primitives)
 *   3. Computes summary statistics (n_basis_functions, max_am, max_primitives)
 *
 * The BasisSet owns its Shell objects and ShellSets. It provides:
 *   - Shell access by index and as a span
 *   - ShellSet lookup by key (am, K) or by angular momentum
 *   - Generation of ShellSetPairs and ShellSetQuartets for integral computation
 *   - Atom-based queries for shells
 *
 * ShellSetPairs are lazily cached internally so that ShellSetQuartets
 * (which store pointers to pairs) have stable references.
 *
 * @note ShellSets are stored via unique_ptr because ShellSet contains
 *       std::once_flag which is neither copyable nor movable.
 */
class BasisSet {
public:
    /// @brief Default constructor (creates an empty BasisSet)
    BasisSet() = default;

    /// @brief BasisSet is non-copyable (contains unique_ptr<ShellSet>)
    BasisSet(const BasisSet&) = delete;
    BasisSet& operator=(const BasisSet&) = delete;

    /// @brief BasisSet is movable (explicit due to std::atomic members)
    BasisSet(BasisSet&& other) noexcept
        : shells_(std::move(other.shells_)),
          shell_sets_(std::move(other.shell_sets_)),
          shell_set_index_(std::move(other.shell_set_index_)),
          pairs_(std::move(other.pairs_)),
          pairs_generated_(other.pairs_generated_.load(std::memory_order_relaxed)),
          quartets_(std::move(other.quartets_)),
          quartets_generated_(other.quartets_generated_.load(std::memory_order_relaxed)),
          cache_mutex_(std::move(other.cache_mutex_)),
          n_basis_functions_(other.n_basis_functions_),
          n_basis_functions_sph_(other.n_basis_functions_sph_),
          max_am_(other.max_am_),
          max_primitives_(other.max_primitives_),
          is_spherical_(other.is_spherical_) {}

    BasisSet& operator=(BasisSet&& other) noexcept {
        if (this != &other) {
            shells_ = std::move(other.shells_);
            shell_sets_ = std::move(other.shell_sets_);
            shell_set_index_ = std::move(other.shell_set_index_);
            pairs_ = std::move(other.pairs_);
            pairs_generated_.store(other.pairs_generated_.load(std::memory_order_relaxed),
                                   std::memory_order_relaxed);
            quartets_ = std::move(other.quartets_);
            quartets_generated_.store(other.quartets_generated_.load(std::memory_order_relaxed),
                                      std::memory_order_relaxed);
            cache_mutex_ = std::move(other.cache_mutex_);
            n_basis_functions_ = other.n_basis_functions_;
            n_basis_functions_sph_ = other.n_basis_functions_sph_;
            max_am_ = other.max_am_;
            max_primitives_ = other.max_primitives_;
            is_spherical_ = other.is_spherical_;
        }
        return *this;
    }

    /**
     * @brief Construct a BasisSet from a vector of Shell objects
     *
     * Moves the shells into internal storage, assigns shell_index and
     * function_index on each shell, computes summary statistics, and
     * groups shells into ShellSets by (angular_momentum, n_primitives).
     *
     * @param shells Vector of Shell objects to include in the basis set
     */
    explicit BasisSet(std::vector<Shell> shells);

    // =========================================================================
    // Shell Access
    // =========================================================================

    /// @brief Get the total number of shells in the basis set
    [[nodiscard]] Size n_shells() const noexcept { return shells_.size(); }

    /// @brief Get the total number of basis functions (sum of n_functions for all shells)
    [[nodiscard]] Size n_basis_functions() const noexcept { return n_basis_functions_; }

    /// @brief Get the maximum angular momentum across all shells
    [[nodiscard]] int max_angular_momentum() const noexcept { return max_am_; }

    /// @brief Get the maximum number of primitives across all shells
    [[nodiscard]] int max_n_primitives() const noexcept { return max_primitives_; }

    /// @brief Access a shell by index
    /// @throws InvalidArgumentException if index is out of bounds
    [[nodiscard]] const Shell& shell(Size i) const;

    /// @brief Access all shells as a span
    [[nodiscard]] std::span<const Shell> shells() const noexcept { return shells_; }

    // =========================================================================
    // ShellSet Access
    // =========================================================================

    /// @brief Get all ShellSets as const pointers
    [[nodiscard]] std::vector<const ShellSet*> shell_sets() const;

    /// @brief Get the number of ShellSets
    [[nodiscard]] Size n_shell_sets() const noexcept { return shell_sets_.size(); }

    /// @brief Look up a ShellSet by angular momentum and primitive count
    /// @return Pointer to the ShellSet, or nullptr if no matching set exists
    [[nodiscard]] const ShellSet* shell_set(int am, int n_primitives) const;

    /// @brief Get all ShellSets with a given angular momentum
    /// @return Vector of pointers to matching ShellSets
    [[nodiscard]] std::vector<const ShellSet*> shell_sets_with_am(int am) const;

    // =========================================================================
    // Pair/Quartet Generation
    // =========================================================================

    /**
     * @brief Access the unique ShellSetPairs worklist (canonical accessor)
     *
     * Returns the upper-triangle set of ShellSetPairs, i.e. pairs (i, j)
     * where i <= j over ShellSets.
     *
     * **Caching behaviour:** The first call computes and caches all pairs;
     * every subsequent call returns the cached vector in O(1) with no
     * additional allocation.
     *
     * **Ordering:** Pairs are enumerated in upper-triangle order over
     * ShellSets: for every pair the bra ShellSet index i is ≤ the ket
     * ShellSet index j.
     *
     * **Thread-safety:** The cache generation is protected by a mutex,
     * so concurrent calls from multiple threads are safe.
     *
     * **Lifetime:** The returned reference (and the ShellSetPair objects
     * within it) remain valid for the lifetime of the BasisSet.
     *
     * @return Const reference to the vector of ShellSetPairs
     */
    [[nodiscard]] const std::vector<ShellSetPair>& shell_set_pairs() const {
        return generate_shell_set_pairs_impl();
    }

    /**
     * @brief Access the unique ShellSetQuartets worklist (canonical accessor)
     *
     * Returns the upper-triangle set of ShellSetQuartets, i.e. quartets
     * (p, q) where p <= q over the cached ShellSetPairs.
     *
     * **Caching behaviour:** The first call computes and caches all quartets;
     * every subsequent call returns the cached vector in O(1) with no
     * additional allocation.
     *
     * **Ordering:** Quartets are enumerated in upper-triangle order over
     * ShellSetPairs: for every quartet the bra pair index p is ≤ the ket
     * pair index q.
     *
     * **Thread-safety:** The cache generation is protected by a mutex,
     * so concurrent calls from multiple threads are safe.
     *
     * **Lifetime:** The returned reference (and the ShellSetQuartet objects
     * within it) remain valid for the lifetime of the BasisSet.
     *
     * @return Const reference to the vector of ShellSetQuartets
     */
    [[nodiscard]] const std::vector<ShellSetQuartet>& shell_set_quartets() const {
        return generate_shell_set_quartets_impl();
    }

    /**
     * @brief Clear all cached ShellSetPair and ShellSetQuartet data
     *
     * Releases memory held by the pair and quartet caches. Subsequent calls
     * to shell_set_pairs() or shell_set_quartets() will re-generate the caches.
     * Useful for releasing memory after computation is complete for large basis sets.
     *
     * @warning Invalidates all pointers and references obtained from previous
     *          shell_set_pairs()/shell_set_quartets() calls. Must not be called
     *          while any thread holds references returned by shell_set_pairs()
     *          or shell_set_quartets(). Accessing invalidated references is
     *          undefined behavior (use-after-free).
     */
    void clear_work_unit_cache() const {
        std::lock_guard<std::mutex> lock(*cache_mutex_);
        quartets_.clear();
        quartets_.shrink_to_fit();
        quartets_generated_.store(false, std::memory_order_release);
        pairs_.clear();
        pairs_.shrink_to_fit();
        pairs_generated_.store(false, std::memory_order_release);
    }

    // =========================================================================
    // Atom Queries
    // =========================================================================

    /// @brief Get all shells centered on a given atom
    /// @param atom_idx The atom index to query
    /// @return Vector of const pointers to shells on that atom
    [[nodiscard]] std::vector<const Shell*> shells_on_atom(Index atom_idx) const;

    // =========================================================================
    // Spherical/Cartesian Configuration
    // =========================================================================

    /// @brief Check if basis set uses spherical harmonics
    /// @return true if spherical, false if Cartesian (default)
    [[nodiscard]] bool is_spherical() const noexcept { return is_spherical_; }

    /// @brief Set spherical/Cartesian mode
    /// @param spherical true for spherical harmonics, false for Cartesian
    void set_spherical(bool spherical) noexcept { is_spherical_ = spherical; }

    /// @brief Get number of basis functions respecting spherical flag
    ///
    /// When spherical=true, returns sum of (2l+1) for each shell.
    /// When spherical=false (default), returns sum of (l+1)(l+2)/2 for each shell.
    [[nodiscard]] Size n_basis_functions_spherical() const noexcept;

private:
    /// Owned Shell objects
    std::vector<Shell> shells_;

    /// ShellSets grouped by (am, K), owned via unique_ptr because ShellSet
    /// contains std::once_flag which is non-copyable and non-movable
    std::vector<std::unique_ptr<ShellSet>> shell_sets_;

    /// Map from ShellSetKey to index in shell_sets_
    std::unordered_map<ShellSetKey, Size> shell_set_index_;

    /// Cached ShellSetPairs for stable references from quartets
    mutable std::vector<ShellSetPair> pairs_;
    mutable std::atomic<bool> pairs_generated_{false};

    /// Cached ShellSetQuartets referencing cached pairs_
    mutable std::vector<ShellSetQuartet> quartets_;
    mutable std::atomic<bool> quartets_generated_{false};

    /// Mutex for thread-safe lazy cache generation
    mutable std::unique_ptr<std::mutex> cache_mutex_{std::make_unique<std::mutex>()};

    /// Summary statistics
    Size n_basis_functions_{0};
    Size n_basis_functions_sph_{0};  ///< Spherical function count
    int max_am_{0};
    int max_primitives_{0};

    /// Spherical/Cartesian flag (default: Cartesian for backward compatibility)
    bool is_spherical_{false};

    /// @brief Assign shell_index and function_index to each shell
    void assign_indices();

    /// @brief Group shells into ShellSets by (am, n_primitives)
    void organize_into_shell_sets();

    /// @brief Internal pair generation (no deprecation warning)
    const std::vector<ShellSetPair>& generate_shell_set_pairs_impl() const;

    /// @brief Internal quartet generation (no deprecation warning)
    const std::vector<ShellSetQuartet>& generate_shell_set_quartets_impl() const;
};

}  // namespace libaccint
