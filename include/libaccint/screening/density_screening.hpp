// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2026 Bryce M. Westheimer. All rights reserved.

#pragma once

/// @file density_screening.hpp
/// @brief Density-weighted Schwarz screening for enhanced efficiency

#include <libaccint/basis/basis_set.hpp>
#include <libaccint/core/types.hpp>
#include <libaccint/screening/schwarz_bounds.hpp>

#include <algorithm>
#include <cmath>
#include <span>
#include <vector>

namespace libaccint::screening {

/// @brief Density-weighted Schwarz screening
///
/// Provides tighter screening bounds by incorporating density matrix elements:
///   D_max * Q_ij * Q_kl < threshold
///
/// where D_max = max(|D_ac|, |D_ad|, |D_bc|, |D_bd|) over all basis functions
/// in the shell quartet.
///
/// This is particularly effective during SCF iterations where the density
/// matrix concentrates in certain regions, allowing more quartets to be skipped.
///
/// Usage:
/// @code
///   SchwarzBounds schwarz(basis);
///   DensityScreening density_screen(basis);
///
///   density_screen.update_density(D.data(), basis.n_basis_functions());
///
///   if (!density_screen.passes_screening(i, j, k, l, schwarz, threshold)) {
///       // Skip this quartet
///   }
/// @endcode
class DensityScreening {
public:
    /// @brief Construct empty density screening (must call update_density before use)
    DensityScreening() = default;

    /// @brief Construct and initialize with a basis set
    /// @param basis The basis set (needed for shell->AO mapping)
    explicit DensityScreening(const BasisSet& basis);

    /// @brief Update the density matrix and recompute shell-level D_max values
    /// @param D Pointer to row-major density matrix (nbf x nbf)
    /// @param nbf Number of basis functions
    void update_density(const Real* D, Size nbf);

    /// @brief Update the density matrix from a span
    /// @param D Density matrix as a span (must be nbf^2 elements)
    /// @param nbf Number of basis functions
    void update_density(std::span<const Real> D, Size nbf) {
        update_density(D.data(), nbf);
    }

    /// @brief Get the maximum density element for a shell pair block
    /// @param i First shell index
    /// @param j Second shell index
    /// @return max |D_ab| over all AO pairs (a in shell i, b in shell j)
    [[nodiscard]] Real shell_pair_d_max(Size i, Size j) const noexcept {
        if (i > j) std::swap(i, j);
        return d_max_shell_pairs_[packed_index(i, j)];
    }

    /// @brief Check if a quartet passes density-weighted screening
    /// @param i First bra shell
    /// @param j Second bra shell
    /// @param k First ket shell
    /// @param l Second ket shell
    /// @param schwarz Schwarz bounds storage
    /// @param threshold Screening threshold
    /// @return true if quartet should be computed, false if it should be skipped
    [[nodiscard]] bool passes_screening(Size i, Size j, Size k, Size l,
                                         const SchwarzBounds& schwarz,
                                         Real threshold) const noexcept {
        // Density-weighted bound: D_max * Q_ij * Q_kl
        // D_max = max over all exchange-type density elements
        Real d_max = quartet_d_max(i, j, k, l);
        Real bound = d_max * schwarz.quartet_bound(i, j, k, l);
        return bound >= threshold;
    }

    /// @brief Get the maximum density element for a quartet's exchange contributions
    /// @param i First bra shell
    /// @param j Second bra shell
    /// @param k First ket shell
    /// @param l Second ket shell
    /// @return max(D_max(i,k), D_max(i,l), D_max(j,k), D_max(j,l))
    [[nodiscard]] Real quartet_d_max(Size i, Size j, Size k, Size l) const noexcept {
        // For exchange contributions, need max over (mu,lambda) pairs
        // where mu is from (i,j) and lambda is from (k,l)
        Real d_max = shell_pair_d_max(i, k);
        d_max = std::max(d_max, shell_pair_d_max(i, l));
        d_max = std::max(d_max, shell_pair_d_max(j, k));
        d_max = std::max(d_max, shell_pair_d_max(j, l));
        return d_max;
    }

    /// @brief Check if density has been set
    [[nodiscard]] bool is_initialized() const noexcept { return n_shells_ > 0 && has_density_; }

    /// @brief Get the maximum density element across all shell pairs
    [[nodiscard]] Real max_d_max() const noexcept { return max_d_max_; }

private:
    /// @brief Compute packed linear index for symmetric matrix (i <= j)
    [[nodiscard]] Size packed_index(Size i, Size j) const noexcept {
        if (i > j) std::swap(i, j);
        return j * (j + 1) / 2 + i;
    }

    /// @brief Compute D_max for each shell pair from the density matrix
    void compute_shell_pair_d_max(const Real* D, Size nbf);

    Size n_shells_{0};                     ///< Number of shells
    std::vector<Index> shell_first_ao_;    ///< First AO index for each shell
    std::vector<int> shell_n_ao_;          ///< Number of AOs in each shell
    std::vector<Real> d_max_shell_pairs_;  ///< Shell-pair D_max (packed triangular)
    Real max_d_max_{0.0};                  ///< Global maximum D_max
    bool has_density_{false};              ///< Whether density has been set
};

}  // namespace libaccint::screening

