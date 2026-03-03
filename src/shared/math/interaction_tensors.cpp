// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2026 Bryce M. Westheimer. All rights reserved.

/// @file interaction_tensors.cpp
/// @brief Implementation of solid harmonic interaction tensors

#include <libaccint/math/interaction_tensors.hpp>

#include <cmath>
#include <stdexcept>

namespace libaccint::math {

Real regular_solid_harmonic(int l, int m, Real x, Real y, Real z) {
    switch (l) {
        case 0: // R_00 = 1
            return 1.0;
        case 1: // R_1m: x, y, z
            switch (m) {
                case 0: return x;
                case 1: return y;
                case 2: return z;
                default: return 0.0;
            }
        case 2: // R_2m: xx, xy, xz, yy, yz, zz (Cartesian)
            switch (m) {
                case 0: return x * x;
                case 1: return x * y;
                case 2: return x * z;
                case 3: return y * y;
                case 4: return y * z;
                case 5: return z * z;
                default: return 0.0;
            }
        default:
            return 0.0;
    }
}

Real irregular_solid_harmonic(int l, int m, Real x, Real y, Real z) {
    Real r2 = x * x + y * y + z * z;
    if (r2 < 1e-30) return 0.0;

    Real r = std::sqrt(r2);
    Real r_pow = std::pow(r, 2 * l + 1);

    return regular_solid_harmonic(l, m, x, y, z) / r_pow;
}

std::vector<Real> interaction_tensor(int rank, const std::array<Real, 3>& R) {
    Real Rx = R[0], Ry = R[1], Rz = R[2];
    Real r2 = Rx * Rx + Ry * Ry + Rz * Rz;

    if (r2 < 1e-30) {
        return std::vector<Real>(n_tensor_components(rank), 0.0);
    }

    Real r = std::sqrt(r2);

    switch (rank) {
        case 0: {
            // T^(0) = 1/R
            return {1.0 / r};
        }
        case 1: {
            // T^(1)_i = -R_i / R^3
            Real inv_r3 = 1.0 / (r * r2);
            return {-Rx * inv_r3, -Ry * inv_r3, -Rz * inv_r3};
        }
        case 2: {
            // T^(2)_{ij} = (3*R_i*R_j - R^2 * δ_ij) / R^5
            Real inv_r5 = 1.0 / (r * r2 * r2);
            return {
                (3.0 * Rx * Rx - r2) * inv_r5,  // xx
                3.0 * Rx * Ry * inv_r5,          // xy
                3.0 * Rx * Rz * inv_r5,          // xz
                (3.0 * Ry * Ry - r2) * inv_r5,   // yy
                3.0 * Ry * Rz * inv_r5,          // yz
                (3.0 * Rz * Rz - r2) * inv_r5,   // zz
            };
        }
        default:
            throw std::invalid_argument(
                "interaction_tensor: rank " + std::to_string(rank) +
                " is not yet implemented (only ranks 0, 1, 2 are supported)");
    }
}

}  // namespace libaccint::math
