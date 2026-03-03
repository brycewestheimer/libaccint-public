// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2026 Bryce M. Westheimer. All rights reserved.

#pragma once

/// @file gaussian_product.hpp
/// @brief Gaussian product theorem utilities for integral computation

#include <libaccint/core/types.hpp>

namespace libaccint::math {

/**
 * @brief Result of Gaussian product computation
 *
 * For two Gaussian functions centered at A and B with exponents alpha and beta:
 *   exp(-alpha|r-A|²) * exp(-beta|r-B|²) = K_AB * exp(-zeta|r-P|²)
 *
 * where:
 *   zeta = alpha + beta (combined exponent)
 *   mu = alpha*beta / (alpha + beta) (reduced exponent)
 *   P = (alpha*A + beta*B) / (alpha + beta) (product center)
 *   K_AB = exp(-mu * |A-B|²) (prefactor/overlap integral)
 */
struct GaussianProduct {
    Point3D P;     ///< Product center coordinates
    Real zeta;     ///< Combined exponent zeta = alpha + beta
    Real mu;       ///< Reduced exponent mu = alpha*beta / (alpha + beta)
    Real K_AB;     ///< Prefactor K_AB = exp(-mu * |A-B|²)

    constexpr GaussianProduct() = default;
    constexpr GaussianProduct(Point3D P_, Real zeta_, Real mu_, Real K_AB_)
        : P(P_), zeta(zeta_), mu(mu_), K_AB(K_AB_) {}
};

/**
 * @brief Compute Gaussian product for two primitive Gaussians
 *
 * Computes the product of two unnormalized Gaussian primitives:
 *   alpha * exp(-alpha * |r - A|²) * beta * exp(-beta * |r - B|²)
 *
 * @param alpha Exponent of first Gaussian
 * @param A Center of first Gaussian
 * @param beta Exponent of second Gaussian
 * @param B Center of second Gaussian
 * @return GaussianProduct containing product center, exponents, and prefactor
 */
[[nodiscard]] GaussianProduct compute_gaussian_product(
    Real alpha, const Point3D& A,
    Real beta, const Point3D& B) noexcept;

/**
 * @brief Batch Gaussian product computation for Structure-of-Arrays (SoA) layout
 *
 * Efficiently computes products for multiple primitive pairs using SoA memory layout.
 * This is the preferred interface for GPU kernels and vectorized CPU code.
 *
 * All pointer arrays must have length >= n_products.
 *
 * @param n_products Number of Gaussian product pairs to compute
 * @param alphas Array of first Gaussian exponents
 * @param A_x Array of first Gaussian center x-coordinates
 * @param A_y Array of first Gaussian center y-coordinates
 * @param A_z Array of first Gaussian center z-coordinates
 * @param betas Array of second Gaussian exponents
 * @param B_x Array of second Gaussian center x-coordinates
 * @param B_y Array of second Gaussian center y-coordinates
 * @param B_z Array of second Gaussian center z-coordinates
 * @param[out] zetas Array to store combined exponents zeta = alpha + beta
 * @param[out] mus Array to store reduced exponents mu = alpha*beta/(alpha+beta)
 * @param[out] P_x Array to store product center x-coordinates
 * @param[out] P_y Array to store product center y-coordinates
 * @param[out] P_z Array to store product center z-coordinates
 * @param[out] K_AB Array to store prefactors
 */
void compute_gaussian_products_batch(
    Size n_products,
    const Real* alphas, const Real* A_x, const Real* A_y, const Real* A_z,
    const Real* betas, const Real* B_x, const Real* B_y, const Real* B_z,
    Real* zetas, Real* mus,
    Real* P_x, Real* P_y, Real* P_z,
    Real* K_AB) noexcept;

}  // namespace libaccint::math
