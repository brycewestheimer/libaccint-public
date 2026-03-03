// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2026 Bryce M. Westheimer. All rights reserved.

/// @file density_screening.cpp
/// @brief Implementation of density-weighted Schwarz screening

#include <libaccint/screening/density_screening.hpp>
#include <libaccint/utils/error_handling.hpp>

#include <algorithm>
#include <cmath>

namespace libaccint::screening {

DensityScreening::DensityScreening(const BasisSet& basis) {
    n_shells_ = basis.n_shells();

    // Build shell->AO mapping
    shell_first_ao_.resize(n_shells_);
    shell_n_ao_.resize(n_shells_);

    for (Size i = 0; i < n_shells_; ++i) {
        const auto& shell = basis.shell(i);
        shell_first_ao_[i] = static_cast<Index>(shell.function_index());
        shell_n_ao_[i] = shell.n_functions();
    }

    // Allocate packed triangular storage for D_max
    const Size n_pairs = n_shells_ * (n_shells_ + 1) / 2;
    d_max_shell_pairs_.resize(n_pairs, 0.0);
}

void DensityScreening::update_density(const Real* D, Size nbf) {
    LIBACCINT_ASSERT(D != nullptr, "Density matrix pointer must not be null");
    LIBACCINT_ASSERT(nbf > 0, "Number of basis functions must be positive");

    if (n_shells_ == 0) return;

    compute_shell_pair_d_max(D, nbf);
    has_density_ = true;
}

void DensityScreening::compute_shell_pair_d_max(const Real* D, Size nbf) {
    max_d_max_ = 0.0;

    // Compute max |D_ab| for each shell pair (i, j)
    for (Size i = 0; i < n_shells_; ++i) {
        const Index first_i = shell_first_ao_[i];
        const int n_i = shell_n_ao_[i];

        for (Size j = i; j < n_shells_; ++j) {
            const Index first_j = shell_first_ao_[j];
            const int n_j = shell_n_ao_[j];

            Real d_max = 0.0;

            // Find max |D_ab| over all AO pairs in this shell block
            for (int a = 0; a < n_i; ++a) {
                for (int b = 0; b < n_j; ++b) {
                    Index ao_a = first_i + static_cast<Index>(a);
                    Index ao_b = first_j + static_cast<Index>(b);

                    // D is stored row-major: D[ao_a, ao_b] = D[ao_a * nbf + ao_b]
                    Real d_val = std::abs(D[ao_a * static_cast<Index>(nbf) + ao_b]);
                    d_max = std::max(d_max, d_val);

                    // Also check transposed element for off-diagonal shell pairs
                    if (i != j) {
                        Real d_val_t = std::abs(D[ao_b * static_cast<Index>(nbf) + ao_a]);
                        d_max = std::max(d_max, d_val_t);
                    }
                }
            }

            d_max_shell_pairs_[packed_index(i, j)] = d_max;
            max_d_max_ = std::max(max_d_max_, d_max);
        }
    }
}

}  // namespace libaccint::screening

