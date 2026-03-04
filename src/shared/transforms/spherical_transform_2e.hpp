// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2026 Bryce M. Westheimer. All rights reserved.

#pragma once

/// @file spherical_transform_2e.hpp
/// @brief Spherical transformation utilities for two-electron integrals

#include <libaccint/basis/basis_set.hpp>
#include <libaccint/math/spherical_transform.hpp>
#include <vector>

namespace libaccint::transforms {

/**
 * @brief Transform a two-electron integral block from Cartesian to spherical
 *
 * Performs the four-index transformation:
 *   (ab|cd)_sph = C_a^T C_b^T (ab|cd)_cart C_c C_d
 *
 * @param La, Lb, Lc, Ld Angular momenta of the four shells
 * @param cartesian Input Cartesian integrals [n_cart(a) * n_cart(b) * n_cart(c) * n_cart(d)]
 * @param spherical Output spherical integrals [n_sph(a) * n_sph(b) * n_sph(c) * n_sph(d)]
 * @param work Working buffer of size >= work_size_4d(La, Lb, Lc, Ld)
 */
void transform_2e_block(int La, int Lb, int Lc, int Ld,
                        const double* cartesian, double* spherical,
                        double* work);

/**
 * @brief SphericalTransform2E class for efficient batch transformations
 *
 * Pre-allocates working memory for transforming ERI quartets.
 */
class SphericalTransform2E {
public:
    /**
     * @brief Construct transformer for a given maximum angular momentum
     * @param max_am Maximum angular momentum to support
     */
    explicit SphericalTransform2E(int max_am);

    /**
     * @brief Transform a shell quartet of integrals from Cartesian to spherical
     *
     * @param La, Lb, Lc, Ld Angular momenta of the quartet
     * @param cartesian Input Cartesian integrals
     * @param spherical Output spherical integrals
     */
    void transform(int La, int Lb, int Lc, int Ld,
                   const double* cartesian, double* spherical);

    /**
     * @brief Get number of spherical integrals for a quartet
     */
    [[nodiscard]] static int n_spherical_quartet(int La, int Lb, int Lc, int Ld) {
        return n_spherical(La) * n_spherical(Lb) * n_spherical(Lc) * n_spherical(Ld);
    }

    /**
     * @brief Get number of Cartesian integrals for a quartet
     */
    [[nodiscard]] static int n_cartesian_quartet(int La, int Lb, int Lc, int Ld) {
        return n_cartesian(La) * n_cartesian(Lb) * n_cartesian(Lc) * n_cartesian(Ld);
    }

private:
    int max_am_;
    std::vector<double> work_;
    math::SphericalTransformer transformer_;
};

/**
 * @brief Transform batched ERI tensor from Cartesian to spherical
 *
 * For applications where the full (or partial) ERI tensor is available,
 * this function transforms all integrals between specified shells.
 *
 * @param basis The basis set
 * @param cartesian_tensor Input Cartesian ERI tensor
 * @param spherical_tensor Output spherical ERI tensor
 * @param shells_a, shells_b, shells_c, shells_d Shell ranges to transform
 */
void transform_2e_tensor(const BasisSet& basis,
                         const std::vector<double>& cartesian_tensor,
                         std::vector<double>& spherical_tensor);

} // namespace libaccint::transforms
