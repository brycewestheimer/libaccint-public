// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2026 Bryce M. Westheimer. All rights reserved.

#pragma once

/// @file schwarz_bounds.hpp
/// @brief Efficient storage and lookup for Schwarz screening bounds

#include <libaccint/basis/basis_set.hpp>
#include <libaccint/core/types.hpp>

#include <vector>
#include <cmath>

namespace libaccint::screening {

/// @brief Efficient storage and O(1) lookup for Schwarz bounds
///
/// Stores Schwarz bounds Q[i][j] = sqrt(max |(ij|ij)|) for all shell pairs
/// in a packed triangular format to minimize memory usage.
///
/// Features:
///   - O(1) lookup by shell pair indices
///   - Symmetric: Q(i,j) = Q(j,i)
///   - Memory-efficient packed storage: O(n^2/2) instead of O(n^2)
///   - Thread-safe concurrent reads
///
/// Usage:
/// @code
///   BasisSet basis(shells);
///   SchwarzBounds bounds(basis);
///
///   Real Q_ij = bounds(i, j);  // O(1) lookup
///   Real Q_kl = bounds(k, l);
///
///   if (Q_ij * Q_kl < threshold) {
///       // Skip this quartet
///   }
/// @endcode
class SchwarzBounds {
public:
    /// @brief Construct empty Schwarz bounds
    SchwarzBounds() = default;

    /// @brief Construct and compute Schwarz bounds for a basis set
    /// @param basis The basis set to compute bounds for
    explicit SchwarzBounds(const BasisSet& basis);

    /// @brief Look up Schwarz bound for shell pair (i, j)
    /// @param i First shell index
    /// @param j Second shell index
    /// @return Q_ij = sqrt(max |(ij|ij)|)
    /// @throws std::out_of_range if indices are out of bounds (debug mode)
    [[nodiscard]] Real operator()(Size i, Size j) const noexcept {
        // Exploit symmetry: swap if i > j
        if (i > j) std::swap(i, j);
        return bounds_[packed_index(i, j)];
    }

    /// @brief Get the number of shells
    [[nodiscard]] Size n_shells() const noexcept { return n_shells_; }

    /// @brief Get the storage size (number of bound values stored)
    [[nodiscard]] Size storage_size() const noexcept { return bounds_.size(); }

    /// @brief Check if a quartet passes Schwarz screening
    /// @param i First bra shell
    /// @param j Second bra shell
    /// @param k First ket shell
    /// @param l Second ket shell
    /// @param threshold Screening threshold
    /// @return true if quartet should be computed, false if it should be skipped
    [[nodiscard]] bool passes_screening(Size i, Size j, Size k, Size l,
                                         Real threshold) const noexcept {
        return (*this)(i, j) * (*this)(k, l) >= threshold;
    }

    /// @brief Get the Schwarz bound product for a quartet
    /// @param i First bra shell
    /// @param j Second bra shell
    /// @param k First ket shell
    /// @param l Second ket shell
    /// @return Q_ij * Q_kl
    [[nodiscard]] Real quartet_bound(Size i, Size j, Size k, Size l) const noexcept {
        return (*this)(i, j) * (*this)(k, l);
    }

    /// @brief Get the maximum Schwarz bound (useful for normalization)
    [[nodiscard]] Real max_bound() const noexcept { return max_bound_; }

    /// @brief Check if bounds have been computed
    [[nodiscard]] bool is_initialized() const noexcept { return n_shells_ > 0; }

    // =========================================================================
    // Batch Screening Utilities
    // =========================================================================

    /// @brief Count quartets that pass screening at a given threshold
    /// @param threshold Screening threshold
    /// @return Number of unique quartets (with 8-fold symmetry) that pass
    [[nodiscard]] Size count_passing_quartets(Real threshold) const;

    /// @brief Estimate quartet reduction factor at a given threshold
    /// @param threshold Screening threshold
    /// @return Fraction of quartets that pass (0.0 to 1.0)
    [[nodiscard]] Real estimate_pass_fraction(Real threshold) const;

private:
    /// @brief Compute packed linear index for symmetric matrix (i <= j)
    [[nodiscard]] Size packed_index(Size i, Size j) const noexcept {
        // Lower triangular packing: idx = j*(j+1)/2 + i (for i <= j)
        return j * (j + 1) / 2 + i;
    }

    /// @brief Compute diagonal ERI and store Schwarz bound
    void compute_bounds(const BasisSet& basis);

    Size n_shells_{0};              ///< Number of shells
    std::vector<Real> bounds_;      ///< Packed triangular storage
    Real max_bound_{0.0};           ///< Maximum bound value
};

}  // namespace libaccint::screening

