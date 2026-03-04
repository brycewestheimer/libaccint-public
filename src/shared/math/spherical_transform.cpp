// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2026 Bryce M. Westheimer. All rights reserved.

/// @file spherical_transform.cpp
/// @brief Implementation of Cartesian-to-spherical transformation matrices

#include <libaccint/math/spherical_transform.hpp>
#include <stdexcept>
#include <cstring>

namespace libaccint::math {

namespace detail {

// ============================================================================
// Transformation Matrix Data
// ============================================================================

// D-type transformation matrix (5 x 6)
// Cartesian order: xx, xy, xz, yy, yz, zz (indices 0-5)
// Spherical order: d0 (z²), d1 (xz), d-1 (yz), d2 (x²-y²), d-2 (xy)
//
// Real solid harmonics (following PySCF convention):
// d0  = (2z² - x² - y²) / 2 = -0.5*xx - 0.5*yy + zz
// d1  = sqrt(3) * xz
// d-1 = sqrt(3) * yz
// d2  = sqrt(3)/2 * (x² - y²) = sqrt(3)/2 * xx - sqrt(3)/2 * yy
// d-2 = sqrt(3) * xy
//
// Note: These coefficients are for normalized Cartesian Gaussians
// transforming to normalized spherical Gaussians.

const std::array<double, 30> C_D = {
    // d0 (m=0): -0.5*xx + 0*xy + 0*xz - 0.5*yy + 0*yz + 1.0*zz
    -0.5, 0.0, 0.0, -0.5, 0.0, 1.0,
    // d1 (m=1): 0*xx + 0*xy + sqrt(3)*xz + 0*yy + 0*yz + 0*zz
    0.0, 0.0, 1.7320508075688772, 0.0, 0.0, 0.0,
    // d-1 (m=-1): 0*xx + 0*xy + 0*xz + 0*yy + sqrt(3)*yz + 0*zz
    0.0, 0.0, 0.0, 0.0, 1.7320508075688772, 0.0,
    // d2 (m=2): sqrt(3)/2*xx + 0*xy + 0*xz - sqrt(3)/2*yy + 0*yz + 0*zz
    0.8660254037844386, 0.0, 0.0, -0.8660254037844386, 0.0, 0.0,
    // d-2 (m=-2): 0*xx + sqrt(3)*xy + 0*xz + 0*yy + 0*yz + 0*zz
    0.0, 1.7320508075688772, 0.0, 0.0, 0.0, 0.0
};

// F-type transformation matrix (7 x 10)
// Cartesian order: xxx, xxy, xxz, xyy, xyz, xzz, yyy, yyz, yzz, zzz
// Spherical order: f0, f1, f-1, f2, f-2, f3, f-3
const std::array<double, 70> C_F = {
    // f0 (m=0): xxz, yyz, zzz
    0.0, 0.0, -1.5000000000000000e+00, 0.0, 0.0, 0.0, 0.0, -1.5000000000000000e+00, 0.0, 1.0000000000000000e+00,
    // f1 (m=1): xxx, xyy, xzz
    -6.1237243569579447e-01, 0.0, 0.0, -6.1237243569579447e-01, 0.0, 2.4494897427831779e+00, 0.0, 0.0, 0.0, 0.0,
    // f-1 (m=-1): xxy, yyy, yzz
    0.0, -6.1237243569579447e-01, 0.0, 0.0, 0.0, 0.0, -6.1237243569579447e-01, 0.0, 2.4494897427831779e+00, 0.0,
    // f2 (m=2): xxz, yyz
    0.0, 0.0, 1.9364916731037085e+00, 0.0, 0.0, 0.0, 0.0, -1.9364916731037085e+00, 0.0, 0.0,
    // f-2 (m=-2): xyz
    0.0, 0.0, 0.0, 0.0, 3.8729833462074170e+00, 0.0, 0.0, 0.0, 0.0, 0.0,
    // f3 (m=3): xxx, xyy
    7.9056941504209488e-01, 0.0, 0.0, -2.3717082451262845e+00, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    // f-3 (m=-3): xxy, yyy
    0.0, 2.3717082451262845e+00, 0.0, 0.0, 0.0, 0.0, -7.9056941504209488e-01, 0.0, 0.0, 0.0
};

// G-type transformation matrix (9 x 15)
// Cartesian order: xxxx, xxxy, xxxz, xxyy, xxyz, xxzz, xyyy, xyyz, xyzz, xzzz,
//                  yyyy, yyyz, yyzz, yzzz, zzzz
// Following PySCF convention for spherical ordering
const std::array<double, 135> C_G = {
    // g0 (m=0): xxxx, xxyy, xxzz, yyyy, yyzz, zzzz
    3.7500000000000000e-01, 0.0, 0.0, 7.5000000000000000e-01, 0.0, -3.0000000000000000e+00, 0.0, 0.0, 0.0, 0.0,
    3.7500000000000000e-01, 0.0, -3.0000000000000000e+00, 0.0, 1.0000000000000000e+00,
    // g1 (m=1): xxxz, xyyz, xzzz
    0.0, 0.0, -2.3717082451262845e+00, 0.0, 0.0, 0.0, 0.0, -2.3717082451262845e+00, 0.0, 3.1622776601683795e+00,
    0.0, 0.0, 0.0, 0.0, 0.0,
    // g-1 (m=-1): xxyz, yyyz, yzzz
    0.0, 0.0, 0.0, 0.0, -2.3717082451262845e+00, 0.0, 0.0, 0.0, 0.0, 0.0,
    0.0, -2.3717082451262845e+00, 0.0, 3.1622776601683795e+00, 0.0,
    // g2 (m=2): xxxx, xxzz, yyyy, yyzz
    -5.5901699437494745e-01, 0.0, 0.0, 0.0, 0.0, 3.3541019662496847e+00, 0.0, 0.0, 0.0, 0.0,
    5.5901699437494745e-01, 0.0, -3.3541019662496847e+00, 0.0, 0.0,
    // g-2 (m=-2): xxxy, xyyy, xyzz
    0.0, -1.1180339887498949e+00, 0.0, 0.0, 0.0, 0.0, -1.1180339887498949e+00, 0.0, 6.7082039324993694e+00, 0.0,
    0.0, 0.0, 0.0, 0.0, 0.0,
    // g3 (m=3): xxxz, xyyz
    0.0, 0.0, 2.0916500663351889e+00, 0.0, 0.0, 0.0, 0.0, -6.2749501990055663e+00, 0.0, 0.0,
    0.0, 0.0, 0.0, 0.0, 0.0,
    // g-3 (m=-3): xxyz, yyyz
    0.0, 0.0, 0.0, 0.0, 6.2749501990055663e+00, 0.0, 0.0, 0.0, 0.0, 0.0,
    0.0, -2.0916500663351889e+00, 0.0, 0.0, 0.0,
    // g4 (m=4): xxxx, xxyy, yyyy
    7.3950997288745202e-01, 0.0, 0.0, -4.4370598373247123e+00, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    7.3950997288745202e-01, 0.0, 0.0, 0.0, 0.0,
    // g-4 (m=-4): xxxy, xyyy
    0.0, 2.9580398915498081e+00, 0.0, 0.0, 0.0, 0.0, -2.9580398915498081e+00, 0.0, 0.0, 0.0,
    0.0, 0.0, 0.0, 0.0, 0.0
};

} // namespace detail

// ============================================================================
// Public API Implementation
// ============================================================================

const double* get_cart_to_sph_matrix(int L) {
    if (L < 0 || L > MAX_L_TRANSFORM) {
        throw std::out_of_range(
            "Unsupported spherical transform angular momentum L=" + std::to_string(L) +
            ". Stable contract supports only L=0.." + std::to_string(MAX_L_TRANSFORM) +
            " (S..G) in this cycle.");
    }

    switch (L) {
        case 0: return detail::C_S.data();
        case 1: return detail::C_P.data();
        case 2: return detail::C_D.data();
        case 3: return detail::C_F.data();
        case 4: return detail::C_G.data();
        default: return nullptr;  // Should never reach here
    }
}

void transform_1d(int L, const double* cartesian, double* spherical) {
    const int n_sph = n_spherical(L);
    const int n_cart = n_cartesian(L);
    const double* C = get_cart_to_sph_matrix(L);

    // Matrix-vector multiply: spherical[i] = sum_j C[i*n_cart + j] * cartesian[j]
    for (int i = 0; i < n_sph; ++i) {
        double sum = 0.0;
        for (int j = 0; j < n_cart; ++j) {
            sum += C[i * n_cart + j] * cartesian[j];
        }
        spherical[i] = sum;
    }
}

void transform_2d(int La, int Lb,
                  const double* cartesian, double* spherical,
                  double* work) {
    const int n_cart_a = n_cartesian(La);
    const int n_cart_b = n_cartesian(Lb);
    const int n_sph_a = n_spherical(La);
    const int n_sph_b = n_spherical(Lb);

    const double* C_a = get_cart_to_sph_matrix(La);
    const double* C_b = get_cart_to_sph_matrix(Lb);

    // Step 1: Transform second index (columns)
    // work[i, k] = sum_j cartesian[i, j] * C_b[k, j]^T = sum_j cartesian[i, j] * C_b[k * n_cart_b + j]
    // But C is stored as C[sph, cart], so C_b[k, j] = C_b[k * n_cart_b + j]
    // Result: work[n_cart_a x n_sph_b]
    for (int i = 0; i < n_cart_a; ++i) {
        for (int k = 0; k < n_sph_b; ++k) {
            double sum = 0.0;
            for (int j = 0; j < n_cart_b; ++j) {
                sum += cartesian[i * n_cart_b + j] * C_b[k * n_cart_b + j];
            }
            work[i * n_sph_b + k] = sum;
        }
    }

    // Step 2: Transform first index (rows)
    // spherical[m, k] = sum_i C_a[m, i] * work[i, k]
    // Result: spherical[n_sph_a x n_sph_b]
    for (int m = 0; m < n_sph_a; ++m) {
        for (int k = 0; k < n_sph_b; ++k) {
            double sum = 0.0;
            for (int i = 0; i < n_cart_a; ++i) {
                sum += C_a[m * n_cart_a + i] * work[i * n_sph_b + k];
            }
            spherical[m * n_sph_b + k] = sum;
        }
    }
}

void transform_4d(int La, int Lb, int Lc, int Ld,
                  const double* cartesian, double* spherical,
                  double* work) {
    const int na = n_cartesian(La), nb = n_cartesian(Lb);
    const int nc = n_cartesian(Lc), nd = n_cartesian(Ld);
    const int sa = n_spherical(La), sb = n_spherical(Lb);
    const int sc = n_spherical(Lc), sd = n_spherical(Ld);

    const double* Ca = get_cart_to_sph_matrix(La);
    const double* Cb = get_cart_to_sph_matrix(Lb);
    const double* Cc = get_cart_to_sph_matrix(Lc);
    const double* Cd = get_cart_to_sph_matrix(Ld);

    // Use ping-pong buffers
    double* buf1 = work;
    double* buf2 = work + na * nb * nc * sd;

    // Step 1: Transform index d
    // cartesian[a,b,c,d] -> buf1[a,b,c,sd]
    for (int a = 0; a < na; ++a) {
        for (int b = 0; b < nb; ++b) {
            for (int c = 0; c < nc; ++c) {
                for (int sd_idx = 0; sd_idx < sd; ++sd_idx) {
                    double sum = 0.0;
                    for (int d = 0; d < nd; ++d) {
                        int cart_idx = ((a * nb + b) * nc + c) * nd + d;
                        sum += cartesian[cart_idx] * Cd[sd_idx * nd + d];
                    }
                    int idx = ((a * nb + b) * nc + c) * sd + sd_idx;
                    buf1[idx] = sum;
                }
            }
        }
    }

    // Step 2: Transform index c
    // buf1[a,b,c,sd] -> buf2[a,b,sc,sd]
    for (int a = 0; a < na; ++a) {
        for (int b = 0; b < nb; ++b) {
            for (int sc_idx = 0; sc_idx < sc; ++sc_idx) {
                for (int sd_idx = 0; sd_idx < sd; ++sd_idx) {
                    double sum = 0.0;
                    for (int c = 0; c < nc; ++c) {
                        int idx1 = ((a * nb + b) * nc + c) * sd + sd_idx;
                        sum += buf1[idx1] * Cc[sc_idx * nc + c];
                    }
                    int idx2 = ((a * nb + b) * sc + sc_idx) * sd + sd_idx;
                    buf2[idx2] = sum;
                }
            }
        }
    }

    // Step 3: Transform index b
    // buf2[a,b,sc,sd] -> buf1[a,sb,sc,sd]
    for (int a = 0; a < na; ++a) {
        for (int sb_idx = 0; sb_idx < sb; ++sb_idx) {
            for (int sc_idx = 0; sc_idx < sc; ++sc_idx) {
                for (int sd_idx = 0; sd_idx < sd; ++sd_idx) {
                    double sum = 0.0;
                    for (int b = 0; b < nb; ++b) {
                        int idx2 = ((a * nb + b) * sc + sc_idx) * sd + sd_idx;
                        sum += buf2[idx2] * Cb[sb_idx * nb + b];
                    }
                    int idx1 = ((a * sb + sb_idx) * sc + sc_idx) * sd + sd_idx;
                    buf1[idx1] = sum;
                }
            }
        }
    }

    // Step 4: Transform index a
    // buf1[a,sb,sc,sd] -> spherical[sa,sb,sc,sd]
    for (int sa_idx = 0; sa_idx < sa; ++sa_idx) {
        for (int sb_idx = 0; sb_idx < sb; ++sb_idx) {
            for (int sc_idx = 0; sc_idx < sc; ++sc_idx) {
                for (int sd_idx = 0; sd_idx < sd; ++sd_idx) {
                    double sum = 0.0;
                    for (int a = 0; a < na; ++a) {
                        int idx1 = ((a * sb + sb_idx) * sc + sc_idx) * sd + sd_idx;
                        sum += buf1[idx1] * Ca[sa_idx * na + a];
                    }
                    int sph_idx = ((sa_idx * sb + sb_idx) * sc + sc_idx) * sd + sd_idx;
                    spherical[sph_idx] = sum;
                }
            }
        }
    }
}

// ============================================================================
// SphericalTransformer Implementation
// ============================================================================

SphericalTransformer::SphericalTransformer(int max_am)
    : max_am_(max_am) {
    if (max_am_ < 0 || max_am_ > MAX_L_TRANSFORM) {
        throw std::invalid_argument(
            "SphericalTransformer max_am=" + std::to_string(max_am_) +
            " is outside stable spherical transform support [0, " +
            std::to_string(MAX_L_TRANSFORM) + "] (S..G).");
    }
    // Allocate work buffer for worst-case 4D transformation
    int max_cart = n_cartesian(max_am);
    int max_sph = n_spherical(max_am);
    // Need space for ping-pong buffers
    work_buffer_.resize(2 * max_cart * max_cart * max_cart * max_sph);
}

void SphericalTransformer::transform_1e(int La, int Lb,
                                         const double* cartesian, double* spherical) {
    transform_2d(La, Lb, cartesian, spherical, work_buffer_.data());
}

void SphericalTransformer::transform_2e(int La, int Lb, int Lc, int Ld,
                                         const double* cartesian, double* spherical) {
    transform_4d(La, Lb, Lc, Ld, cartesian, spherical, work_buffer_.data());
}

} // namespace libaccint::math
