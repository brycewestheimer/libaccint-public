// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2026 Bryce M. Westheimer. All rights reserved.

#include <libaccint/math/gaussian_product.hpp>
#include <cmath>

#ifdef __AVX2__
#include <immintrin.h>
#endif

namespace libaccint::math {

GaussianProduct compute_gaussian_product(
    Real alpha, const Point3D& A,
    Real beta, const Point3D& B) noexcept {

    // Combined exponent: zeta = alpha + beta
    const Real zeta = alpha + beta;
    const Real inv_zeta = Real{1.0} / zeta;

    // Reduced exponent: mu = alpha * beta / (alpha + beta)
    const Real mu = alpha * beta * inv_zeta;

    // Product center: P = (alpha*A + beta*B) / (alpha + beta)
    const Point3D P{
        (alpha * A.x + beta * B.x) * inv_zeta,
        (alpha * A.y + beta * B.y) * inv_zeta,
        (alpha * A.z + beta * B.z) * inv_zeta
    };

    // Prefactor: K_AB = exp(-mu * |A-B|²)
    const Real AB_squared = A.distance_squared(B);
    const Real K_AB = std::exp(-mu * AB_squared);

    return GaussianProduct{P, zeta, mu, K_AB};
}

void compute_gaussian_products_batch(
    Size n_products,
    const Real* alphas, const Real* A_x, const Real* A_y, const Real* A_z,
    const Real* betas, const Real* B_x, const Real* B_y, const Real* B_z,
    Real* zetas, Real* mus,
    Real* P_x, Real* P_y, Real* P_z,
    Real* K_AB) noexcept {

#ifdef __AVX2__
    // ========================================================================
    // AVX2 SIMD path: process 4 doubles at a time
    // ========================================================================
    Size i = 0;
    const __m256d one = _mm256_set1_pd(1.0);

    for (; i + 4 <= n_products; i += 4) {
        // Load exponents
        __m256d alpha_v = _mm256_loadu_pd(alphas + i);
        __m256d beta_v  = _mm256_loadu_pd(betas + i);

        // Combined exponent: zeta = alpha + beta
        __m256d zeta_v = _mm256_add_pd(alpha_v, beta_v);
        _mm256_storeu_pd(zetas + i, zeta_v);

        // inv_zeta = 1.0 / zeta
        __m256d inv_zeta_v = _mm256_div_pd(one, zeta_v);

        // Reduced exponent: mu = alpha * beta / zeta
        __m256d mu_v = _mm256_mul_pd(_mm256_mul_pd(alpha_v, beta_v), inv_zeta_v);
        _mm256_storeu_pd(mus + i, mu_v);

        // Load center coordinates
        __m256d ax_v = _mm256_loadu_pd(A_x + i);
        __m256d ay_v = _mm256_loadu_pd(A_y + i);
        __m256d az_v = _mm256_loadu_pd(A_z + i);
        __m256d bx_v = _mm256_loadu_pd(B_x + i);
        __m256d by_v = _mm256_loadu_pd(B_y + i);
        __m256d bz_v = _mm256_loadu_pd(B_z + i);

        // Product centers: P = (alpha*A + beta*B) * inv_zeta
        // Use FMA: alpha*A + beta*B = fmadd(alpha, A, beta*B)
        __m256d px_v = _mm256_mul_pd(
            _mm256_fmadd_pd(alpha_v, ax_v, _mm256_mul_pd(beta_v, bx_v)),
            inv_zeta_v);
        __m256d py_v = _mm256_mul_pd(
            _mm256_fmadd_pd(alpha_v, ay_v, _mm256_mul_pd(beta_v, by_v)),
            inv_zeta_v);
        __m256d pz_v = _mm256_mul_pd(
            _mm256_fmadd_pd(alpha_v, az_v, _mm256_mul_pd(beta_v, bz_v)),
            inv_zeta_v);

        _mm256_storeu_pd(P_x + i, px_v);
        _mm256_storeu_pd(P_y + i, py_v);
        _mm256_storeu_pd(P_z + i, pz_v);

        // K_AB = exp(-mu * |A-B|²)
        __m256d dx_v = _mm256_sub_pd(ax_v, bx_v);
        __m256d dy_v = _mm256_sub_pd(ay_v, by_v);
        __m256d dz_v = _mm256_sub_pd(az_v, bz_v);

        // |A-B|² = dx² + dy² + dz² (using FMA)
        __m256d ab_sq_v = _mm256_fmadd_pd(dx_v, dx_v,
            _mm256_fmadd_pd(dy_v, dy_v, _mm256_mul_pd(dz_v, dz_v)));

        // -mu * |A-B|²
        __m256d neg_mu_ab = _mm256_sub_pd(_mm256_setzero_pd(),
            _mm256_mul_pd(mu_v, ab_sq_v));

        // exp has no AVX2 intrinsic — extract, compute scalar, reload
        alignas(32) double neg_mu_ab_arr[4];
        alignas(32) double kab_arr[4];
        _mm256_store_pd(neg_mu_ab_arr, neg_mu_ab);

        kab_arr[0] = std::exp(neg_mu_ab_arr[0]);
        kab_arr[1] = std::exp(neg_mu_ab_arr[1]);
        kab_arr[2] = std::exp(neg_mu_ab_arr[2]);
        kab_arr[3] = std::exp(neg_mu_ab_arr[3]);

        _mm256_storeu_pd(K_AB + i, _mm256_load_pd(kab_arr));
    }

    // Scalar remainder for non-multiple-of-4 tail
    for (; i < n_products; ++i) {
        const Real alpha = alphas[i];
        const Real beta = betas[i];
        const Real zeta = alpha + beta;
        const Real inv_zeta = Real{1.0} / zeta;

        zetas[i] = zeta;
        mus[i] = alpha * beta * inv_zeta;

        P_x[i] = (alpha * A_x[i] + beta * B_x[i]) * inv_zeta;
        P_y[i] = (alpha * A_y[i] + beta * B_y[i]) * inv_zeta;
        P_z[i] = (alpha * A_z[i] + beta * B_z[i]) * inv_zeta;

        const Real dx = A_x[i] - B_x[i];
        const Real dy = A_y[i] - B_y[i];
        const Real dz = A_z[i] - B_z[i];
        const Real AB_squared = dx*dx + dy*dy + dz*dz;
        K_AB[i] = std::exp(-mus[i] * AB_squared);
    }

#else
    // ========================================================================
    // Scalar fallback (no AVX2)
    // ========================================================================
    for (Size i = 0; i < n_products; ++i) {
        const Real alpha = alphas[i];
        const Real beta = betas[i];

        // Combined exponent
        const Real zeta = alpha + beta;
        const Real inv_zeta = Real{1.0} / zeta;
        zetas[i] = zeta;

        // Reduced exponent
        mus[i] = alpha * beta * inv_zeta;

        // Product centers
        P_x[i] = (alpha * A_x[i] + beta * B_x[i]) * inv_zeta;
        P_y[i] = (alpha * A_y[i] + beta * B_y[i]) * inv_zeta;
        P_z[i] = (alpha * A_z[i] + beta * B_z[i]) * inv_zeta;

        // Prefactor: K_AB = exp(-mu * |A-B|²)
        const Real dx = A_x[i] - B_x[i];
        const Real dy = A_y[i] - B_y[i];
        const Real dz = A_z[i] - B_z[i];
        const Real AB_squared = dx*dx + dy*dy + dz*dz;
        K_AB[i] = std::exp(-mus[i] * AB_squared);
    }
#endif  // __AVX2__
}

}  // namespace libaccint::math
