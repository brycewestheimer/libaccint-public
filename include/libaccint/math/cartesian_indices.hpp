// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2026 Bryce M. Westheimer. All rights reserved.

#pragma once

/// @file cartesian_indices.hpp
/// @brief Utilities for generating and indexing Cartesian basis function tuples

#include <array>
#include <vector>
#include <libaccint/core/types.hpp>

namespace libaccint::math {

/// Generate all (lx, ly, lz) tuples for a given angular momentum in canonical order.
///
/// Canonical ordering follows the libint2 convention: lx descending, then ly descending.
/// For example:
/// - L=0 (S): (0,0,0)
/// - L=1 (P): (1,0,0), (0,1,0), (0,0,1)
/// - L=2 (D): (2,0,0), (1,1,0), (1,0,1), (0,2,0), (0,1,1), (0,0,2)
/// - L=3 (F): (3,0,0), (2,1,0), (2,0,1), (1,2,0), (1,1,1), (1,0,2), (0,3,0), (0,2,1), (0,1,2), (0,0,3)
///
/// @param L Angular momentum quantum number (L = lx + ly + lz)
/// @return Vector of (lx, ly, lz) tuples in canonical order
inline std::vector<std::array<int, 3>> generate_cartesian_indices(int L) {
    std::vector<std::array<int, 3>> indices;
    indices.reserve(n_cartesian(L));
    for (int lx = L; lx >= 0; --lx) {
        for (int ly = L - lx; ly >= 0; --ly) {
            int lz = L - lx - ly;
            indices.push_back({lx, ly, lz});
        }
    }
    return indices;
}

/// Compute the linear index of a Cartesian basis function within its shell.
///
/// Given (lx, ly, lz) components, returns the index in the canonical ordering
/// for the corresponding shell with angular momentum L = lx + ly + lz.
/// This function uses the libint2 convention: lx descending, then ly descending.
///
/// @param lx x-component of angular momentum
/// @param ly y-component of angular momentum
/// @param lz z-component of angular momentum (typically lz = L - lx - ly, not used in computation)
/// @return Linear index within the shell (0, 1, 2, ...)
constexpr int cartesian_index(int lx, int ly, int lz) noexcept {
    int L = lx + ly + lz;
    // Count all tuples with higher lx than the current lx
    int index = 0;
    for (int x = L; x > lx; --x) {
        // For each lx value > current lx, count how many (ly, lz) pairs exist
        // For a given lx, ly ranges from (L - lx) down to 0, giving (L - lx + 1) pairs
        index += L - x + 1;
    }
    // Within the same lx, count tuples with higher ly
    // For current lx, we want the index of ly within [0, L - lx]
    // In canonical order (ly descending), the index is (L - lx) - ly
    index += (L - lx) - ly;
    return index;
}

}  // namespace libaccint::math
