// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2026 Bryce M. Westheimer. All rights reserved.

/// @file schwarz_bounds.cpp
/// @brief Implementation of SchwarzBounds storage and computation

#include <libaccint/screening/schwarz_bounds.hpp>
#include <libaccint/kernels/eri_kernel.hpp>
#include <libaccint/buffers/two_electron_buffer.hpp>

#include <algorithm>
#include <cmath>

namespace libaccint::screening {

SchwarzBounds::SchwarzBounds(const BasisSet& basis) {
    compute_bounds(basis);
}

void SchwarzBounds::compute_bounds(const BasisSet& basis) {
    n_shells_ = basis.n_shells();

    // Allocate packed triangular storage: n*(n+1)/2 elements
    const Size n_pairs = n_shells_ * (n_shells_ + 1) / 2;
    bounds_.resize(n_pairs);

    max_bound_ = 0.0;

    TwoElectronBuffer<0> buffer;

    // Compute Q[i][j] = sqrt(max |(ij|ij)|) for each unique shell pair
    for (Size i = 0; i < n_shells_; ++i) {
        const auto& shell_i = basis.shell(i);

        for (Size j = i; j < n_shells_; ++j) {
            const auto& shell_j = basis.shell(j);

            // Compute diagonal ERI (ij|ij)
            kernels::compute_eri(shell_i, shell_j, shell_i, shell_j, buffer);

            // Find maximum absolute value over all function indices
            Real max_eri = 0.0;
            const int ni = shell_i.n_functions();
            const int nj = shell_j.n_functions();

            for (int a = 0; a < ni; ++a) {
                for (int b = 0; b < nj; ++b) {
                    // Diagonal element: (a,b|a,b)
                    Real eri_val = std::abs(buffer(a, b, a, b));
                    max_eri = std::max(max_eri, eri_val);
                }
            }

            Real Q_ij = std::sqrt(max_eri);
            bounds_[packed_index(i, j)] = Q_ij;
            max_bound_ = std::max(max_bound_, Q_ij);
        }
    }
}

Size SchwarzBounds::count_passing_quartets(Real threshold) const {
    if (n_shells_ == 0 || threshold <= 0.0) {
        // All quartets pass if no threshold
        Size n_pairs = n_shells_ * (n_shells_ + 1) / 2;
        return n_pairs * (n_pairs + 1) / 2;
    }

    Size count = 0;

    // Iterate over unique quartets with 8-fold symmetry: (ij) <= (kl)
    for (Size i = 0; i < n_shells_; ++i) {
        for (Size j = i; j < n_shells_; ++j) {
            Real Q_ij = (*this)(i, j);

            // Early exit if Q_ij alone can't pass
            if (Q_ij * max_bound_ < threshold) continue;

            for (Size k = i; k < n_shells_; ++k) {
                // Start l based on symmetry constraint
                Size l_start = (k == i) ? j : k;

                for (Size l = l_start; l < n_shells_; ++l) {
                    Real Q_kl = (*this)(k, l);

                    if (Q_ij * Q_kl >= threshold) {
                        ++count;
                    }
                }
            }
        }
    }

    return count;
}

Real SchwarzBounds::estimate_pass_fraction(Real threshold) const {
    if (n_shells_ == 0) return 1.0;

    // Total unique quartets (8-fold symmetry)
    Size n_pairs = n_shells_ * (n_shells_ + 1) / 2;
    Size total = n_pairs * (n_pairs + 1) / 2;

    if (total == 0) return 1.0;

    Size passing = count_passing_quartets(threshold);
    return static_cast<Real>(passing) / static_cast<Real>(total);
}

}  // namespace libaccint::screening

