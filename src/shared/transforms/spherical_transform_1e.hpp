// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2026 Bryce M. Westheimer. All rights reserved.

#pragma once

/// @file spherical_transform_1e.hpp
/// @brief Spherical transformation utilities for one-electron integrals

#include <libaccint/basis/basis_set.hpp>
#include <libaccint/math/spherical_transform.hpp>
#include <vector>

namespace libaccint::transforms {

/**
 * @brief Transform a one-electron integral matrix from Cartesian to spherical basis
 *
 * For each unique shell pair (i,j), transforms the Cartesian integral block
 * to spherical using:
 *   S_sph(i,j) = C_i^T * S_cart(i,j) * C_j
 *
 * where C_i is the Cartesian-to-spherical transformation matrix for shell i.
 *
 * @param basis The basis set (used to get shell info)
 * @param cartesian Input Cartesian integrals [n_cart x n_cart] row-major
 * @param spherical Output spherical integrals [n_sph x n_sph] row-major
 */
void transform_1e_to_spherical(const BasisSet& basis,
                               const std::vector<double>& cartesian,
                               std::vector<double>& spherical);

/**
 * @brief Transform a one-electron integral shell pair block from Cartesian to spherical
 *
 * @param La Angular momentum of shell a
 * @param Lb Angular momentum of shell b
 * @param cartesian Input block [n_cart_a x n_cart_b]
 * @param spherical Output block [n_sph_a x n_sph_b]
 * @param work Working buffer of size >= work_size_2d(La, Lb)
 */
void transform_1e_block(int La, int Lb,
                        const double* cartesian, double* spherical,
                        double* work);

/**
 * @brief SphericalTransform1E class for efficient batch transformations
 *
 * Pre-allocates working memory for transforming entire integral matrices.
 */
class SphericalTransform1E {
public:
    /**
     * @brief Construct transformer for a given basis set
     * @param basis The basis set to transform integrals for
     */
    explicit SphericalTransform1E(const BasisSet& basis);

    /**
     * @brief Transform full integral matrix from Cartesian to spherical
     * @param cartesian Input matrix [n_cart x n_cart]
     * @param spherical Output matrix [n_sph x n_sph]
     */
    void transform(const std::vector<double>& cartesian,
                   std::vector<double>& spherical);

    /// @brief Get the number of Cartesian basis functions
    [[nodiscard]] Size n_cartesian() const noexcept { return n_cart_; }

    /// @brief Get the number of spherical basis functions
    [[nodiscard]] Size n_spherical() const noexcept { return n_sph_; }

private:
    const BasisSet* basis_;
    Size n_cart_;
    Size n_sph_;
    std::vector<double> work_;
    math::SphericalTransformer transformer_;
};

} // namespace libaccint::transforms
