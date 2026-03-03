// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2026 Bryce M. Westheimer. All rights reserved.

#include <libaccint/math/normalization.hpp>
#include <libaccint/utils/constants.hpp>
#include <cmath>

namespace libaccint::math {

// ============================================================================
// Implementation of normalization_factor functions
// ============================================================================

double normalization_factor(Real alpha, int i, int j, int k) noexcept {
    // L = i + j + k (total angular momentum)
    const int L = i + j + k;

    // Get double factorials for each component
    const Real df_i = static_cast<Real>(double_factorial_odd(i));
    const Real df_j = static_cast<Real>(double_factorial_odd(j));
    const Real df_k = static_cast<Real>(double_factorial_odd(k));

    // Compute normalization factor:
    // N_ijk(α) = sqrt( (2α/π)^(3/2) * (4α)^L / (df_i * df_j * df_k) )

    // Compute (2α/π)^(3/2)
    const Real two_alpha_over_pi = 2.0 * alpha / constants::PI;
    const Real term1 = std::pow(two_alpha_over_pi, 1.5);

    // Compute (4α)^L
    const Real four_alpha = 4.0 * alpha;
    const Real term2 = std::pow(four_alpha, static_cast<Real>(L));

    // Compute denominator
    const Real denominator = df_i * df_j * df_k;

    // Combine: sqrt(term1 * term2 / denominator)
    const Real result = std::sqrt(term1 * term2 / denominator);

    return result;
}

double normalization_factor(Real alpha, int L) noexcept {
    // Convenience function: equivalent to normalization_factor(alpha, L, 0, 0)
    return normalization_factor(alpha, L, 0, 0);
}

}  // namespace libaccint::math
