// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2026 Bryce M. Westheimer. All rights reserved.

#pragma once

#include <libaccint/basis/primitive_pair_data.hpp>
#include <libaccint/basis/shell_set.hpp>

#include <atomic>
#include <memory>

namespace libaccint {

/**
 * @brief Pairs two ShellSets for one-electron and two-electron integrals.
 *
 * ShellSetPair represents the combination of two ShellSets A and B,
 * used for computing integrals of the form (a|O|b) where O is an operator.
 *
 * This class supports Schwarz screening bounds which are computed lazily
 * and cached for subsequent access. The class remains copyable; copied
 * instances share the same cached Schwarz bound via shared_ptr.
 */
class ShellSetPair {
public:
    /**
     * @brief Constructs a ShellSetPair from two ShellSets.
     * @param set_a First ShellSet (bra side)
     * @param set_b Second ShellSet (ket side)
     */
    ShellSetPair(const ShellSet& set_a, const ShellSet& set_b)
        : set_a_(&set_a)
        , set_b_(&set_b)
        , schwarz_cache_(std::make_shared<SchwarzCache>()) {}

    /**
     * @brief Returns a reference to the first ShellSet (A).
     */
    [[nodiscard]] const ShellSet& shell_set_a() const noexcept { return *set_a_; }

    /**
     * @brief Returns a reference to the second ShellSet (B).
     */
    [[nodiscard]] const ShellSet& shell_set_b() const noexcept { return *set_b_; }

    /**
     * @brief Returns the angular momentum of ShellSet A.
     */
    [[nodiscard]] int La() const noexcept { return set_a_->angular_momentum(); }

    /**
     * @brief Returns the angular momentum of ShellSet B.
     */
    [[nodiscard]] int Lb() const noexcept { return set_b_->angular_momentum(); }

    /**
     * @brief Returns the total angular momentum (La + Lb).
     */
    [[nodiscard]] int L_total() const noexcept { return La() + Lb(); }

    /**
     * @brief Returns the total number of shell pairs.
     * @return n_shells_a * n_shells_b
     */
    [[nodiscard]] Size n_pairs() const noexcept {
        return set_a_->n_shells() * set_b_->n_shells();
    }

    /**
     * @brief Returns Schwarz screening bound for this pair.
     *
     * Computes Q_ab = sqrt(max (ab|ab)) where the max is taken over all
     * shell pairs in this ShellSetPair. The bound is computed lazily on
     * first access and cached for subsequent calls.
     *
     * @return Schwarz bound Q_ab for this shell set pair
     * @note Thread-safe via std::call_once
     */
    [[nodiscard]] Real schwarz_bound() const;

    /**
     * @brief Force eager computation of Schwarz bound.
     *
     * Useful for precomputing all bounds before parallel computation
     * to avoid contention on lazy initialization.
     */
    void precompute_schwarz_bound() const;

    /**
     * @brief Check if Schwarz bound has been computed.
     * @return true if bound has been computed, false otherwise
     */
    [[nodiscard]] bool schwarz_computed() const noexcept {
        return schwarz_cache_->computed.load(std::memory_order_acquire);
    }

    /**
     * @brief Returns pre-computed Gaussian product data for all primitive pairs.
     *
     * Computes and caches the Gaussian product (P, zeta, K_AB, etc.) for every
     * primitive pair across the two ShellSets. This enables O(1) lookup during
     * integral computation instead of recomputing these quantities each time.
     *
     * @return Const reference to the cached PrimitivePairData
     * @note Thread-safe via std::call_once
     */
    [[nodiscard]] const PrimitivePairData& pair_data() const;

    /**
     * @brief Force eager computation of primitive pair data.
     *
     * Useful for precomputing all pair data before parallel computation
     * to avoid contention on lazy initialization.
     */
    void precompute_pair_data() const;

    /**
     * @brief Check if primitive pair data has been computed.
     * @return true if pair data has been computed, false otherwise
     */
    [[nodiscard]] bool pair_data_ready() const noexcept {
        return pair_data_cache_->computed.load(std::memory_order_acquire);
    }

private:
    const ShellSet* set_a_;  ///< Pointer to first ShellSet
    const ShellSet* set_b_;  ///< Pointer to second ShellSet

    /// @brief Cache for Schwarz bound with thread-safe lazy initialization
    struct SchwarzCache {
        std::atomic<bool> computed{false};
        std::atomic<bool> computing{false};
        Real bound{0.0};
    };
    std::shared_ptr<SchwarzCache> schwarz_cache_;

    /// @brief Internal implementation of Schwarz bound computation
    void compute_schwarz_bound_impl() const;

    /// @brief Cache for primitive pair data with thread-safe lazy initialization
    struct PairDataCache {
        std::atomic<bool> computed{false};
        std::atomic<bool> computing{false};
        PrimitivePairData data;
    };
    std::shared_ptr<PairDataCache> pair_data_cache_{std::make_shared<PairDataCache>()};

    /// @brief Build primitive pair data from the two ShellSets
    void build_pair_data_impl() const;
};

}  // namespace libaccint
