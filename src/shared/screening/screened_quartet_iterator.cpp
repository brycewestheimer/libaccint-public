// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2026 Bryce M. Westheimer. All rights reserved.

/// @file screened_quartet_iterator.cpp
/// @brief Implementation of Schwarz-screened shell quartet iterator

#include <libaccint/screening/screened_quartet_iterator.hpp>
#include <libaccint/kernels/eri_kernel.hpp>
#include <libaccint/buffers/two_electron_buffer.hpp>
#include <libaccint/utils/error_handling.hpp>

#include <cmath>
#include <algorithm>

namespace libaccint::screening {

ScreenedQuartetIterator::ScreenedQuartetIterator(const BasisSet& basis, Real threshold)
    : basis_(&basis)
    , threshold_(threshold) {
    if (threshold < 0.0) {
        throw InvalidArgumentException(
            "ScreenedQuartetIterator: threshold must be non-negative, got " +
            std::to_string(threshold));
    }
    initialize_schwarz_bounds();
}

ScreenedQuartetIterator::ScreenedQuartetIterator(const BasisSet& basis,
                                                   const ScreeningOptions& options)
    : basis_(&basis)
    , threshold_(options.effective_threshold()) {
    initialize_schwarz_bounds();
}

ScreenedQuartetIterator::ScreenedQuartetIterator(const BasisSet& basis,
                                                   const SchwarzBounds& bounds,
                                                   Real threshold)
    : basis_(&basis)
    , threshold_(threshold)
    , external_bounds_(&bounds) {
    if (threshold < 0.0) {
        throw InvalidArgumentException(
            "ScreenedQuartetIterator: threshold must be non-negative, got " +
            std::to_string(threshold));
    }
    // Use external bounds, no need to initialize internal storage
}

void ScreenedQuartetIterator::initialize_schwarz_bounds() {
    const Size n_shells = basis_->n_shells();

    // Initialize the 2D Schwarz bounds array
    schwarz_bounds_.resize(n_shells);
    for (Size i = 0; i < n_shells; ++i) {
        schwarz_bounds_[i].resize(n_shells);
    }

    // Compute Q[i][j] = sqrt(max |(ij|ij)|) for each unique shell pair
    TwoElectronBuffer<0> buffer;

    for (Size i = 0; i < n_shells; ++i) {
        const auto& shell_i = basis_->shell(i);

        for (Size j = i; j < n_shells; ++j) {
            const auto& shell_j = basis_->shell(j);

            // Compute diagonal ERI (ij|ij)
            kernels::compute_eri(shell_i, shell_j, shell_i, shell_j, buffer);

            // Find maximum absolute value
            Real max_eri = 0.0;
            const int ni = shell_i.n_functions();
            const int nj = shell_j.n_functions();

            for (int a = 0; a < ni; ++a) {
                for (int b = 0; b < nj; ++b) {
                    Real eri_val = std::abs(buffer(a, b, a, b));
                    max_eri = std::max(max_eri, eri_val);
                }
            }

            Real Q_ij = std::sqrt(max_eri);
            schwarz_bounds_[i][j] = Q_ij;
            schwarz_bounds_[j][i] = Q_ij;  // Symmetric
        }
    }
}

std::optional<Real> ScreenedQuartetIterator::check_quartet(Size i, Size j, Size k, Size l) const {
    Real Q_ij, Q_kl;

    if (external_bounds_) {
        Q_ij = (*external_bounds_)(i, j);
        Q_kl = (*external_bounds_)(k, l);
    } else {
        Q_ij = schwarz_bounds_[i][j];
        Q_kl = schwarz_bounds_[k][l];
    }

    Real bound = Q_ij * Q_kl;

    // Apply density-weighted screening if available
    if (density_screen_ && density_screen_->is_initialized()) {
        Real d_max = density_screen_->quartet_d_max(i, j, k, l);
        Real effective_bound = d_max * bound;
        if (effective_bound >= threshold_) {
            return bound;
        }
        return std::nullopt;
    }

    // Standard Schwarz screening
    if (bound >= threshold_) {
        return bound;
    }
    return std::nullopt;
}

Size ScreenedQuartetIterator::total_unique_quartets() const noexcept {
    // With 8-fold symmetry, unique quartets satisfy: i <= j, k <= l, (i,j) <= (k,l)
    // where (i,j) <= (k,l) means i < k, or (i == k and j <= l)
    const Size n = basis_->n_shells();

    // Number of unique pairs
    Size n_pairs = n * (n + 1) / 2;

    // Number of unique quartets from pairs
    return n_pairs * (n_pairs + 1) / 2;
}

void ScreenedQuartetIterator::reset() noexcept {
    current_i_ = 0;
    current_j_ = 0;
    current_k_ = 0;
    current_l_ = 0;
    exhausted_ = false;
    stats_ = ScreeningStatistics{};
}

bool ScreenedQuartetIterator::has_more() const noexcept {
    return !exhausted_;
}

void ScreenedQuartetIterator::advance_to_next_valid() {
    const Size n = basis_->n_shells();

    // Move to next position
    // Ordering: i <= j, k <= l, (i,j) <= (k,l)
    // We iterate as: l++, then k++, then j++, then i++

    ++current_l_;

    // If l > n-1, increment k and reset l
    while (current_l_ >= n) {
        ++current_k_;
        current_l_ = current_k_;  // k <= l

        // If k exceeded valid range for this (i,j), move to next (i,j)
        // (i,j) <= (k,l) means we need k >= i, or (k == i and l >= j)
        while (current_k_ >= n || (current_k_ == current_i_ && current_l_ < current_j_)) {
            ++current_j_;

            // If j > n-1, increment i and reset j
            while (current_j_ >= n) {
                ++current_i_;

                if (current_i_ >= n) {
                    exhausted_ = true;
                    return;
                }

                current_j_ = current_i_;  // i <= j
            }

            // Reset k,l for new (i,j)
            current_k_ = current_i_;
            current_l_ = current_j_;
        }
    }

    // Ensure (i,j) <= (k,l) constraint
    // If k == i, need l >= j
    if (current_k_ == current_i_ && current_l_ < current_j_) {
        current_l_ = current_j_;
        if (current_l_ >= n) {
            advance_to_next_valid();  // Recurse to handle overflow
        }
    }
}

std::optional<std::vector<ScreenedQuartet>> ScreenedQuartetIterator::next_batch(Size max_size) {
    if (exhausted_ || max_size == 0) {
        return std::nullopt;
    }

    std::vector<ScreenedQuartet> batch;
    batch.reserve(max_size);

    const Size n = basis_->n_shells();

    while (batch.size() < max_size && !exhausted_) {
        // Count this quartet
        ++stats_.total_quartets;

        // Check if it passes screening
        if (auto bound = check_quartet(current_i_, current_j_, current_k_, current_l_)) {
            batch.push_back(ScreenedQuartet{
                current_i_, current_j_, current_k_, current_l_, *bound
            });
            ++stats_.computed_quartets;
        } else {
            ++stats_.skipped_quartets;
        }

        // Advance to next quartet
        advance_to_next_valid();
    }

    if (batch.empty()) {
        return std::nullopt;
    }

    return batch;
}

}  // namespace libaccint::screening
