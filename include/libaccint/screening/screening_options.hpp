// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2026 Bryce M. Westheimer. All rights reserved.

#pragma once

/// @file screening_options.hpp
/// @brief Configuration options for Schwarz screening

#include <libaccint/core/types.hpp>
#include <iostream>
#include <string>

namespace libaccint::screening {

/// @brief Preset levels for screening thresholds
enum class ScreeningPreset {
    None,    ///< No screening (threshold = infinity, all quartets computed)
    Loose,   ///< 1e-10, faster but less accurate
    Normal,  ///< 1e-12, good balance (default)
    Tight,   ///< 1e-14, maximum accuracy
    Custom   ///< User-specified threshold
};

/// @brief Configuration options for Schwarz screening
///
/// Controls how screening is applied during integral computation.
/// The Schwarz inequality states: |(ab|cd)| <= sqrt((ab|ab)) * sqrt((cd|cd)) = Q_ab * Q_cd
/// Quartets where Q_ab * Q_cd < threshold are skipped.
///
/// Usage:
/// @code
///   ScreeningOptions opts = ScreeningOptions::normal();
///   opts.density_weighted = true;
///   engine.compute_and_consume(fock, D, opts);
/// @endcode
struct ScreeningOptions {
    /// @brief Schwarz screening threshold
    ///
    /// Quartets where Q_ab * Q_cd < threshold are skipped.
    /// Lower values = tighter screening = more accuracy but less speedup.
    Real threshold = 1e-12;

    /// @brief Whether screening is enabled
    ///
    /// When false, all quartets are computed (equivalent to threshold = 0).
    bool enabled = true;

    /// @brief Whether to use density-weighted screening
    ///
    /// When true, uses D_max * Q_ab * Q_cd < threshold where D_max is the
    /// maximum density matrix element over all AO pairs in the quartet.
    /// This provides tighter bounds during SCF iteration.
    bool density_weighted = false;

    /// @brief Whether to exploit 8-fold permutation symmetry
    ///
    /// When true, the screened integral path iterates only canonical
    /// shell quartets (i<=j, k<=l, ij<=kl), reducing the number of computed
    /// quartets by up to 8x. Requires a symmetry-aware consumer such as
    /// FockBuilder (which must have an accumulate_symmetric method).
    bool use_permutation_symmetry = false;

    /// @brief Whether to collect screening statistics
    ///
    /// When true, Engine will track computed/skipped quartet counts.
    bool enable_statistics = false;

    /// @brief Verbosity level for screening diagnostics
    ///
    /// 0 = silent, 1 = summary, 2 = detailed
    int verbosity = 0;

    // =========================================================================
    // Factory Methods
    // =========================================================================

    /// @brief Create options with no screening (compute all quartets)
    [[nodiscard]] static ScreeningOptions none() noexcept {
        return ScreeningOptions{.threshold = 0.0, .enabled = false};
    }

    /// @brief Create options with loose screening (1e-10)
    [[nodiscard]] static ScreeningOptions loose() noexcept {
        return ScreeningOptions{.threshold = 1e-10, .enabled = true};
    }

    /// @brief Create options with normal screening (1e-12, default)
    [[nodiscard]] static ScreeningOptions normal() noexcept {
        return ScreeningOptions{.threshold = 1e-12, .enabled = true};
    }

    /// @brief Create options with tight screening (1e-14)
    [[nodiscard]] static ScreeningOptions tight() noexcept {
        return ScreeningOptions{.threshold = 1e-14, .enabled = true};
    }

    /// @brief Create options from a preset
    [[nodiscard]] static ScreeningOptions from_preset(ScreeningPreset preset) noexcept {
        switch (preset) {
            case ScreeningPreset::None:   return none();
            case ScreeningPreset::Loose:  return loose();
            case ScreeningPreset::Normal: return normal();
            case ScreeningPreset::Tight:  return tight();
            case ScreeningPreset::Custom: return normal();  // Default to normal
        }
        return normal();
    }

    // =========================================================================
    // Validation
    // =========================================================================

    /// @brief Validate the screening options and warn about extreme values
    ///
    /// Prints warnings to stderr if threshold is outside recommended range.
    void validate() const {
        if (!enabled) return;

        if (threshold < 0.0) {
            std::cerr << "Warning: Screening threshold is negative (" << threshold
                      << "), will be treated as 0 (no screening)\n";
        }

        if (threshold > 1e-8) {
            std::cerr << "Warning: Screening threshold " << threshold
                      << " is very loose (> 1e-8), results may be inaccurate\n";
        }

        if (threshold > 0.0 && threshold < 1e-16) {
            std::cerr << "Warning: Screening threshold " << threshold
                      << " is very tight (< 1e-16), minimal screening benefit\n";
        }
    }

    /// @brief Get the effective threshold
    ///
    /// Returns 0 if screening is disabled, otherwise the configured threshold.
    [[nodiscard]] Real effective_threshold() const noexcept {
        if (!enabled) return 0.0;
        return (threshold < 0.0) ? 0.0 : threshold;
    }

    /// @brief Check if a Schwarz bound product passes screening
    ///
    /// @param schwarz_product Q_ab * Q_cd
    /// @return true if the quartet should be computed, false if it should be skipped
    [[nodiscard]] bool passes_screening(Real schwarz_product) const noexcept {
        if (!enabled) return true;
        return schwarz_product >= effective_threshold();
    }

    /// @brief Get the preset name as a string
    [[nodiscard]] std::string preset_name() const {
        if (!enabled) return "None";
        if (threshold == 1e-10) return "Loose";
        if (threshold == 1e-12) return "Normal";
        if (threshold == 1e-14) return "Tight";
        return "Custom";
    }
};

/// @brief Statistics about screening effectiveness
struct ScreeningStatistics {
    Size total_quartets{0};     ///< Total number of unique quartets considered
    Size computed_quartets{0};  ///< Number of quartets computed
    Size skipped_quartets{0};   ///< Number of quartets screened out

    /// @brief Compute the screening efficiency ratio
    /// @return Fraction of quartets that were skipped (0.0 to 1.0)
    [[nodiscard]] Real efficiency() const noexcept {
        if (total_quartets == 0) return 0.0;
        return static_cast<Real>(skipped_quartets) / static_cast<Real>(total_quartets);
    }

    /// @brief Compute the percentage of quartets skipped
    [[nodiscard]] Real skip_percentage() const noexcept {
        return efficiency() * 100.0;
    }

    /// @brief Reset all counters to zero
    void reset() noexcept {
        total_quartets = 0;
        computed_quartets = 0;
        skipped_quartets = 0;
    }

    /// @brief Merge statistics from another source
    void merge(const ScreeningStatistics& other) noexcept {
        total_quartets += other.total_quartets;
        computed_quartets += other.computed_quartets;
        skipped_quartets += other.skipped_quartets;
    }
};

}  // namespace libaccint::screening
