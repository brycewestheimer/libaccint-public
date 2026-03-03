// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2026 Bryce M. Westheimer. All rights reserved.

#pragma once

/// @file screened_quartet_iterator.hpp
/// @brief Iterator yielding shell quartets that pass Schwarz screening

#include <libaccint/basis/basis_set.hpp>
#include <libaccint/basis/shell_set_pair.hpp>
#include <libaccint/basis/shell_set_quartet.hpp>
#include <libaccint/core/types.hpp>
#include <libaccint/screening/screening_options.hpp>
#include <libaccint/screening/schwarz_bounds.hpp>
#include <libaccint/screening/density_screening.hpp>

#include <vector>
#include <optional>

namespace libaccint::screening {

/// @brief Represents a shell quartet that passed screening
struct ScreenedQuartet {
    Size shell_i;       ///< Index of first bra shell
    Size shell_j;       ///< Index of second bra shell
    Size shell_k;       ///< Index of first ket shell
    Size shell_l;       ///< Index of second ket shell
    Real schwarz_bound; ///< Product of Schwarz bounds Q_ab * Q_cd
};

/// @brief Iterator that yields only shell quartets passing Schwarz screening
///
/// This class generates unique shell quartets respecting 8-fold permutational
/// symmetry and filters them using Schwarz screening (Q_ab * Q_cd > threshold).
///
/// Usage:
/// @code
///   BasisSet basis(...);
///   ScreenedQuartetIterator iter(basis, 1e-12);
///
///   while (auto batch = iter.next_batch(1000)) {
///       for (const auto& quartet : *batch) {
///           // Process quartet
///       }
///   }
///   std::cout << "Screening efficiency: " << iter.statistics().efficiency() << "\n";
/// @endcode
class ScreenedQuartetIterator {
public:
    /// @brief Construct a screened quartet iterator
    /// @param basis The basis set to iterate over
    /// @param threshold Schwarz screening threshold (default: 1e-12)
    /// @throws InvalidArgumentException if threshold is negative
    explicit ScreenedQuartetIterator(const BasisSet& basis, Real threshold = 1e-12);

    /// @brief Construct a screened quartet iterator with options
    /// @param basis The basis set to iterate over
    /// @param options Screening options
    explicit ScreenedQuartetIterator(const BasisSet& basis, const ScreeningOptions& options);

    /// @brief Construct with pre-computed Schwarz bounds
    /// @param basis The basis set to iterate over
    /// @param bounds Pre-computed Schwarz bounds
    /// @param threshold Schwarz screening threshold
    ScreenedQuartetIterator(const BasisSet& basis, const SchwarzBounds& bounds, Real threshold);

    /// @brief Get the next batch of screened quartets
    /// @param max_size Maximum number of quartets to return
    /// @return Vector of screened quartets, or nullopt if exhausted
    [[nodiscard]] std::optional<std::vector<ScreenedQuartet>> next_batch(Size max_size);

    /// @brief Reset the iterator to the beginning
    void reset() noexcept;

    /// @brief Check if there are more quartets to iterate
    [[nodiscard]] bool has_more() const noexcept;

    /// @brief Get the screening statistics
    [[nodiscard]] const ScreeningStatistics& statistics() const noexcept { return stats_; }

    /// @brief Get the screening threshold
    [[nodiscard]] Real threshold() const noexcept { return threshold_; }

    /// @brief Get total unique quartet count (without screening)
    [[nodiscard]] Size total_unique_quartets() const noexcept;

    /// @brief Enable density-weighted screening
    /// @param density_screen Pointer to density screening object
    void set_density_screening(const DensityScreening* density_screen) noexcept {
        density_screen_ = density_screen;
    }

private:
    /// @brief Initialize or refresh Schwarz bounds for all shell pairs
    void initialize_schwarz_bounds();

    /// @brief Check if the quartet passes Schwarz screening
    /// @param i, j, k, l Shell indices
    /// @return Product of Schwarz bounds if passed, nullopt if screened out
    [[nodiscard]] std::optional<Real> check_quartet(Size i, Size j, Size k, Size l) const;

    /// @brief Advance to the next valid quartet position
    void advance_to_next_valid();

    const BasisSet* basis_;
    Real threshold_;

    // Schwarz bounds Q[i][j] = Q_ij for shell pair (i,j)
    std::vector<std::vector<Real>> schwarz_bounds_;

    // Optional external Schwarz bounds (for efficiency)
    const SchwarzBounds* external_bounds_{nullptr};

    // Optional density-weighted screening
    const DensityScreening* density_screen_{nullptr};

    // Current position in the iteration
    Size current_i_{0};
    Size current_j_{0};
    Size current_k_{0};
    Size current_l_{0};
    bool exhausted_{false};

    // Statistics
    ScreeningStatistics stats_;
};

}  // namespace libaccint::screening
