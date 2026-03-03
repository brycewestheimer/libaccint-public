// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2026 Bryce M. Westheimer. All rights reserved.

#pragma once

/// @file spherical_transform.hpp
/// @brief Cartesian-to-spherical harmonic transformation matrices and utilities

#include <libaccint/core/types.hpp>
#include <array>
#include <span>
#include <cmath>

namespace libaccint::math {

/// Maximum angular momentum for precomputed transformation matrices (stable G-only support)
constexpr int MAX_L_TRANSFORM = 4;

/**
 * @brief Get the Cartesian-to-spherical transformation matrix for angular momentum L
 *
 * Returns a pointer to the transformation matrix C[n_spherical x n_cartesian].
 * The matrix transforms Cartesian Gaussian integrals to spherical (pure harmonic)
 * Gaussian integrals using real solid harmonics.
 *
 * Convention:
 * - Real solid harmonics (tesseral harmonics)
 * - Condon-Shortley phase convention
 * - Spherical ordering: m = 0, 1, -1, 2, -2, ... (like PySCF)
 * - Cartesian ordering: standard (xx, xy, xz, yy, yz, zz for d-functions)
 *
 * @param L Angular momentum (0 <= L <= MAX_L_TRANSFORM)
 * @return Pointer to transformation matrix in row-major order
 * @throws std::out_of_range if L is outside stable spherical support (0..MAX_L_TRANSFORM)
 */
[[nodiscard]] const double* get_cart_to_sph_matrix(int L);

/**
 * @brief Get dimensions of transformation matrix
 * @param L Angular momentum
 * @return Pair of (n_spherical, n_cartesian)
 */
[[nodiscard]] constexpr std::pair<int, int> cart_to_sph_dimensions(int L) noexcept {
    return {n_spherical(L), n_cartesian(L)};
}

/**
 * @brief Transform a 1D vector from Cartesian to spherical
 *
 * Performs: spherical[i] = sum_j C[i,j] * cartesian[j]
 *
 * @param L Angular momentum
 * @param cartesian Input Cartesian integrals [n_cartesian]
 * @param spherical Output spherical integrals [n_spherical]
 */
void transform_1d(int L, const double* cartesian, double* spherical);

/**
 * @brief Transform a 2D matrix from Cartesian to spherical (two-index)
 *
 * Performs: S_sph = C_a^T * S_cart * C_b
 *
 * @param La Angular momentum of first index (rows)
 * @param Lb Angular momentum of second index (columns)
 * @param cartesian Input Cartesian matrix [n_cart_a x n_cart_b] row-major
 * @param spherical Output spherical matrix [n_sph_a x n_sph_b] row-major
 * @param work Working buffer [max(n_cart_a * n_sph_b, n_sph_a * n_cart_b)]
 */
void transform_2d(int La, int Lb,
                  const double* cartesian, double* spherical,
                  double* work);

/**
 * @brief Transform a 4D tensor from Cartesian to spherical (four-index)
 *
 * Performs: (ab|cd)_sph = C_a^T C_b^T (ab|cd)_cart C_c C_d
 *
 * Transformation is done one index at a time to minimize memory usage.
 *
 * @param La, Lb, Lc, Ld Angular momenta for each index
 * @param cartesian Input Cartesian tensor [n_cart_a * n_cart_b * n_cart_c * n_cart_d]
 * @param spherical Output spherical tensor [n_sph_a * n_sph_b * n_sph_c * n_sph_d]
 * @param work Working buffer (size depends on angular momenta)
 */
void transform_4d(int La, int Lb, int Lc, int Ld,
                  const double* cartesian, double* spherical,
                  double* work);

/**
 * @brief Compute required work buffer size for 2D transformation
 * @param La, Lb Angular momenta
 * @return Required buffer size in doubles
 */
[[nodiscard]] constexpr int work_size_2d(int La, int Lb) noexcept {
    return std::max(n_cartesian(La) * n_spherical(Lb),
                    n_spherical(La) * n_cartesian(Lb));
}

/**
 * @brief Compute required work buffer size for 4D transformation
 * @param La, Lb, Lc, Ld Angular momenta
 * @return Required buffer size in doubles
 */
[[nodiscard]] constexpr int work_size_4d(int La, int Lb, int Lc, int Ld) noexcept {
    // Need space for largest intermediate tensor
    int n_a = n_cartesian(La), n_b = n_cartesian(Lb);
    int n_c = n_cartesian(Lc), n_d = n_cartesian(Ld);
    int s_a = n_spherical(La), s_b = n_spherical(Lb);
    int s_c = n_spherical(Lc), s_d = n_spherical(Ld);

    // After transforming d: n_a * n_b * n_c * s_d
    // After transforming c: n_a * n_b * s_c * s_d
    // After transforming b: n_a * s_b * s_c * s_d
    // After transforming a: s_a * s_b * s_c * s_d (final)
    int max_size = n_a * n_b * n_c * s_d;
    max_size = std::max(max_size, n_a * n_b * s_c * s_d);
    max_size = std::max(max_size, n_a * s_b * s_c * s_d);
    // Need two buffers for ping-pong
    return 2 * max_size;
}

/**
 * @brief SphericalTransformer class for efficient repeated transformations
 *
 * Pre-allocates working memory and caches transformation matrix pointers
 * for a specific angular momentum configuration.
 */
class SphericalTransformer {
public:
    /// @brief Construct transformer for 1-electron integrals (2-index)
    SphericalTransformer(int max_am);

    /// @brief Transform 2-index integrals from Cartesian to spherical
    void transform_1e(int La, int Lb,
                      const double* cartesian, double* spherical);

    /// @brief Transform 4-index integrals from Cartesian to spherical
    void transform_2e(int La, int Lb, int Lc, int Ld,
                      const double* cartesian, double* spherical);

private:
    int max_am_;
    std::vector<double> work_buffer_;
};

// ============================================================================
// Precomputed transformation matrices (up to G-functions, L=4)
// ============================================================================

namespace detail {

/// S-type: trivial (1x1 identity)
constexpr std::array<double, 1> C_S = {1.0};

/// P-type: identity with PySCF ordering (px, py, pz -> py, pz, px)
constexpr std::array<double, 9> C_P = {
    // m=0 (pz) <- z
    0.0, 0.0, 1.0,
    // m=1 (px) <- x
    1.0, 0.0, 0.0,
    // m=-1 (py) <- y
    0.0, 1.0, 0.0
};

/// D-type transformation (5 spherical from 6 Cartesian)
/// Cartesian order: xx, xy, xz, yy, yz, zz
/// Spherical order: d0, d1, d-1, d2, d-2
extern const std::array<double, 30> C_D;

/// F-type transformation (7 spherical from 10 Cartesian)
extern const std::array<double, 70> C_F;

/// G-type transformation (9 spherical from 15 Cartesian)
extern const std::array<double, 135> C_G;

} // namespace detail

} // namespace libaccint::math
