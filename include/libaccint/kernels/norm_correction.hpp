// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2026 Bryce M. Westheimer. All rights reserved.

#pragma once

/// @file norm_correction.hpp
/// @brief Shared normalization correction factor for Cartesian Gaussian components
///
/// This header provides the norm_correction function used by all CPU host
/// kernels to adjust for the Shell normalization convention. The Shell class
/// normalizes its contraction coefficients for the (L,0,0) component; for
/// other Cartesian components (lx, ly, lz) we apply:
///   correction = 1 / sqrt((2lx-1)!! * (2ly-1)!! * (2lz-1)!!)

#include <libaccint/core/types.hpp>
#include <libaccint/math/normalization.hpp>

#include <cmath>

namespace libaccint {

/// @brief Compute normalization correction factor for a Cartesian Gaussian component
///
/// @param lx x-component of angular momentum
/// @param ly y-component of angular momentum
/// @param lz z-component of angular momentum
/// @return Normalization correction factor for this component
inline Real norm_correction(int lx, int ly, int lz) noexcept {
    const Real denominator = static_cast<Real>(
        math::double_factorial_odd(lx) *
        math::double_factorial_odd(ly) *
        math::double_factorial_odd(lz));
    return 1.0 / std::sqrt(denominator);
}

}  // namespace libaccint
