// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2026 Bryce M. Westheimer. All rights reserved.

/// @file test_spherical_transform.cpp
/// @brief Unit tests for Cartesian-to-spherical transformation matrices

#include <libaccint/math/spherical_transform.hpp>
#include <libaccint/core/types.hpp>
#include <gtest/gtest.h>
#include <cmath>
#include <vector>

using namespace libaccint;
using namespace libaccint::math;

class SphericalTransformTest : public ::testing::Test {
protected:
    static constexpr double TOL = 1e-12;
};

// ============================================================================
// Matrix Dimension Tests
// ============================================================================

TEST_F(SphericalTransformTest, MatrixDimensions) {
    for (int L = 0; L <= 4; ++L) {
        auto [n_sph, n_cart] = cart_to_sph_dimensions(L);
        EXPECT_EQ(n_sph, n_spherical(L)) << "L=" << L;
        EXPECT_EQ(n_cart, n_cartesian(L)) << "L=" << L;
    }
}

TEST_F(SphericalTransformTest, MatrixPointerNotNull) {
    for (int L = 0; L <= MAX_L_TRANSFORM; ++L) {
        const double* C = get_cart_to_sph_matrix(L);
        EXPECT_NE(C, nullptr) << "L=" << L;
    }
}

TEST_F(SphericalTransformTest, OutOfRangeThrows) {
    EXPECT_THROW(get_cart_to_sph_matrix(MAX_L_TRANSFORM + 1), std::out_of_range);
    EXPECT_THROW(get_cart_to_sph_matrix(-1), std::out_of_range);
}

// ============================================================================
// S-type (L=0) Tests
// ============================================================================

TEST_F(SphericalTransformTest, STypeIsIdentity) {
    const double* C = get_cart_to_sph_matrix(0);
    EXPECT_DOUBLE_EQ(C[0], 1.0);
}

TEST_F(SphericalTransformTest, STypeTransform) {
    double cart = 2.5;
    double sph = 0.0;
    transform_1d(0, &cart, &sph);
    EXPECT_DOUBLE_EQ(sph, 2.5);
}

// ============================================================================
// P-type (L=1) Tests
// ============================================================================

TEST_F(SphericalTransformTest, PTypeDimensions) {
    auto [n_sph, n_cart] = cart_to_sph_dimensions(1);
    EXPECT_EQ(n_sph, 3);
    EXPECT_EQ(n_cart, 3);
}

TEST_F(SphericalTransformTest, PTypeTransform) {
    // Cartesian p-functions: x, y, z
    double cart[3] = {1.0, 2.0, 3.0};
    double sph[3] = {0.0, 0.0, 0.0};

    transform_1d(1, cart, sph);

    // PySCF ordering: pz, px, py -> m=0, m=1, m=-1
    // Based on our matrix: m=0 <- z, m=1 <- x, m=-1 <- y
    EXPECT_DOUBLE_EQ(sph[0], 3.0);  // pz (m=0)
    EXPECT_DOUBLE_EQ(sph[1], 1.0);  // px (m=1)
    EXPECT_DOUBLE_EQ(sph[2], 2.0);  // py (m=-1)
}

// ============================================================================
// D-type (L=2) Tests
// ============================================================================

TEST_F(SphericalTransformTest, DTypeDimensions) {
    auto [n_sph, n_cart] = cart_to_sph_dimensions(2);
    EXPECT_EQ(n_sph, 5);
    EXPECT_EQ(n_cart, 6);
}

TEST_F(SphericalTransformTest, DTypeMatrixRowSumsPositive) {
    const double* C = get_cart_to_sph_matrix(2);
    const int n_sph = n_spherical(2);
    const int n_cart = n_cartesian(2);

    // Each row should have at least one non-zero element
    for (int i = 0; i < n_sph; ++i) {
        double row_sum_sq = 0.0;
        for (int j = 0; j < n_cart; ++j) {
            row_sum_sq += C[i * n_cart + j] * C[i * n_cart + j];
        }
        EXPECT_GT(row_sum_sq, 0.0) << "Row " << i << " is all zeros";
    }
}

TEST_F(SphericalTransformTest, DTypeSpecificTransformation) {
    // Test d_z2 function: should be -0.5*xx - 0.5*yy + 1.0*zz
    // Cartesian order: xx(0), xy(1), xz(2), yy(3), yz(4), zz(5)
    double cart[6] = {1.0, 0.0, 0.0, 1.0, 0.0, 1.0};  // xx=yy=zz=1
    double sph[5] = {0.0};

    transform_1d(2, cart, sph);

    // d0 = -0.5*1 - 0.5*1 + 1.0*1 = 0.0
    EXPECT_NEAR(sph[0], 0.0, TOL);
}

TEST_F(SphericalTransformTest, DTypeXZTransformation) {
    // Test d_xz function: sqrt(3) * xz
    double cart[6] = {0.0, 0.0, 1.0, 0.0, 0.0, 0.0};  // xz = 1
    double sph[5] = {0.0};

    transform_1d(2, cart, sph);

    // d1 (m=1) = sqrt(3) * xz = 1.732...
    EXPECT_NEAR(sph[1], std::sqrt(3.0), TOL);
}

// ============================================================================
// F-type (L=3) Tests
// ============================================================================

TEST_F(SphericalTransformTest, FTypeDimensions) {
    auto [n_sph, n_cart] = cart_to_sph_dimensions(3);
    EXPECT_EQ(n_sph, 7);
    EXPECT_EQ(n_cart, 10);
}

// ============================================================================
// G-type (L=4) Tests
// ============================================================================

TEST_F(SphericalTransformTest, GTypeDimensions) {
    auto [n_sph, n_cart] = cart_to_sph_dimensions(4);
    EXPECT_EQ(n_sph, 9);
    EXPECT_EQ(n_cart, 15);
}

// ============================================================================
// 2D Transformation Tests
// ============================================================================

TEST_F(SphericalTransformTest, Transform2DSSType) {
    // (ss|ss) should remain unchanged
    double cart = 1.5;
    double sph = 0.0;
    double work[4];

    transform_2d(0, 0, &cart, &sph, work);
    EXPECT_DOUBLE_EQ(sph, 1.5);
}

TEST_F(SphericalTransformTest, Transform2DPPType) {
    // (pp|..) block: 3x3 Cartesian -> 3x3 spherical
    double cart[9] = {
        1.0, 0.0, 0.0,
        0.0, 1.0, 0.0,
        0.0, 0.0, 1.0
    };
    double sph[9] = {0.0};
    double work[9];

    transform_2d(1, 1, cart, sph, work);

    // Identity in Cartesian should give identity-like result in spherical
    // (with possible reordering)
    for (int i = 0; i < 3; ++i) {
        EXPECT_NEAR(sph[i * 3 + i], 1.0, TOL);
    }
}

TEST_F(SphericalTransformTest, Transform2DSDType) {
    // (sd|..) block: 1x6 Cartesian -> 1x5 spherical
    double cart[6] = {1.0, 0.0, 0.0, 1.0, 0.0, 1.0};  // s overlaps with xx, yy, zz
    double sph[5] = {0.0};
    double work[6];

    transform_2d(0, 2, cart, sph, work);

    // First spherical function (d0 = -0.5xx - 0.5yy + zz)
    // With input xx=yy=zz=1: d0 = -0.5 - 0.5 + 1 = 0
    EXPECT_NEAR(sph[0], 0.0, TOL);
}

// ============================================================================
// 4D Transformation Tests
// ============================================================================

TEST_F(SphericalTransformTest, Transform4DSSSSType) {
    // (ss|ss) should remain unchanged
    double cart = 2.5;
    double sph = 0.0;
    double work[16];

    transform_4d(0, 0, 0, 0, &cart, &sph, work);
    EXPECT_DOUBLE_EQ(sph, 2.5);
}

TEST_F(SphericalTransformTest, Transform4DSPSPType) {
    // (sp|sp) block: 3x3 Cartesian -> 3x3 spherical for each bra/ket pair
    const int size = 1 * 3 * 1 * 3;  // 9 elements
    double cart[size];
    double sph[size];
    double work[2 * size];

    // Initialize to identity-like pattern
    for (int i = 0; i < size; ++i) cart[i] = 0.0;
    cart[0] = 1.0;  // (s,px|s,px)
    cart[4] = 1.0;  // (s,py|s,py)
    cart[8] = 1.0;  // (s,pz|s,pz)

    transform_4d(0, 1, 0, 1, cart, sph, work);

    // Should preserve diagonal-like structure
    double trace = sph[0] + sph[4] + sph[8];
    EXPECT_NEAR(trace, 3.0, TOL);
}

// ============================================================================
// SphericalTransformer Class Tests
// ============================================================================

TEST_F(SphericalTransformTest, TransformerConstruction) {
    EXPECT_NO_THROW(SphericalTransformer(4));
}

TEST_F(SphericalTransformTest, TransformerTransform1e) {
    SphericalTransformer transformer(2);

    double cart[9] = {  // (pp) block identity
        1.0, 0.0, 0.0,
        0.0, 1.0, 0.0,
        0.0, 0.0, 1.0
    };
    double sph[9] = {0.0};

    transformer.transform_1e(1, 1, cart, sph);

    // Trace should be preserved
    double trace = sph[0] + sph[4] + sph[8];
    EXPECT_NEAR(trace, 3.0, TOL);
}

TEST_F(SphericalTransformTest, TransformerTransform2e) {
    SphericalTransformer transformer(2);

    double cart = 1.0;
    double sph = 0.0;

    transformer.transform_2e(0, 0, 0, 0, &cart, &sph);
    EXPECT_DOUBLE_EQ(sph, 1.0);
}

// ============================================================================
// Work Buffer Size Tests
// ============================================================================

TEST_F(SphericalTransformTest, WorkSize2D) {
    // For (dd) block: max(6*5, 5*6) = 30
    EXPECT_EQ(work_size_2d(2, 2), 30);

    // For (sp) block: max(1*3, 3*3) = 9
    EXPECT_EQ(work_size_2d(0, 1), 3);
}

TEST_F(SphericalTransformTest, WorkSize4D) {
    // For (ssss): minimal buffer needed
    int size = work_size_4d(0, 0, 0, 0);
    EXPECT_GT(size, 0);

    // For (pppp): larger buffer
    int size_pppp = work_size_4d(1, 1, 1, 1);
    EXPECT_GT(size_pppp, size);
}

// ============================================================================
// Numerical Stability Tests
// ============================================================================

TEST_F(SphericalTransformTest, LargeValueStability) {
    double cart[6] = {1e10, 1e10, 1e10, 1e10, 1e10, 1e10};
    double sph[5] = {0.0};

    EXPECT_NO_THROW(transform_1d(2, cart, sph));

    for (int i = 0; i < 5; ++i) {
        EXPECT_TRUE(std::isfinite(sph[i])) << "i=" << i;
    }
}

TEST_F(SphericalTransformTest, SmallValueStability) {
    double cart[6] = {1e-15, 1e-15, 1e-15, 1e-15, 1e-15, 1e-15};
    double sph[5] = {0.0};

    EXPECT_NO_THROW(transform_1d(2, cart, sph));

    for (int i = 0; i < 5; ++i) {
        EXPECT_TRUE(std::isfinite(sph[i])) << "i=" << i;
    }
}

// ============================================================================
// Task 3.3.3: Known Coefficient Value Tests
// ============================================================================

TEST_F(SphericalTransformTest, DTypeKnownCoefficients) {
    const double* C = get_cart_to_sph_matrix(2);
    const int n_cart = n_cartesian(2);  // 6

    // d0 (m=0): -0.5*xx + 0*xy + 0*xz - 0.5*yy + 0*yz + 1.0*zz
    EXPECT_DOUBLE_EQ(C[0 * n_cart + 0], -0.5);       // xx
    EXPECT_DOUBLE_EQ(C[0 * n_cart + 3], -0.5);       // yy
    EXPECT_DOUBLE_EQ(C[0 * n_cart + 5],  1.0);       // zz
    EXPECT_DOUBLE_EQ(C[0 * n_cart + 1],  0.0);       // xy
    EXPECT_DOUBLE_EQ(C[0 * n_cart + 2],  0.0);       // xz
    EXPECT_DOUBLE_EQ(C[0 * n_cart + 4],  0.0);       // yz

    // d1 (m=1): sqrt(3)*xz
    EXPECT_NEAR(C[1 * n_cart + 2], std::sqrt(3.0), TOL);

    // d-1 (m=-1): sqrt(3)*yz
    EXPECT_NEAR(C[2 * n_cart + 4], std::sqrt(3.0), TOL);

    // d2 (m=2): sqrt(3)/2 * (xx - yy)
    EXPECT_NEAR(C[3 * n_cart + 0], std::sqrt(3.0) / 2.0, TOL);
    EXPECT_NEAR(C[3 * n_cart + 3], -std::sqrt(3.0) / 2.0, TOL);

    // d-2 (m=-2): sqrt(3)*xy
    EXPECT_NEAR(C[4 * n_cart + 1], std::sqrt(3.0), TOL);
}

TEST_F(SphericalTransformTest, FTypeKnownCoefficients) {
    const double* C = get_cart_to_sph_matrix(3);
    const int n_cart = n_cartesian(3);  // 10

    // f0 (m=0): 0*xxx + 0*xxy + coeff*xxz + ... + 1.0*zzz
    EXPECT_DOUBLE_EQ(C[0 * n_cart + 9], 1.0);  // zzz coefficient

    // f-2 (m=-2): sqrt(15) * xyz
    EXPECT_NEAR(C[4 * n_cart + 4], std::sqrt(15.0), TOL);  // xyz coefficient
}

TEST_F(SphericalTransformTest, STypeCoefficient) {
    const double* C = get_cart_to_sph_matrix(0);
    EXPECT_DOUBLE_EQ(C[0], 1.0);
}

TEST_F(SphericalTransformTest, PTypeCoefficients) {
    const double* C = get_cart_to_sph_matrix(1);
    const int n_cart = 3;

    // PySCF convention: p0=z, p1=x, p-1=y
    // Row 0 (pz): 0*x + 0*y + 1*z
    EXPECT_DOUBLE_EQ(C[0 * n_cart + 2], 1.0);
    EXPECT_DOUBLE_EQ(C[0 * n_cart + 0], 0.0);
    EXPECT_DOUBLE_EQ(C[0 * n_cart + 1], 0.0);

    // Row 1 (px): 1*x + 0*y + 0*z
    EXPECT_DOUBLE_EQ(C[1 * n_cart + 0], 1.0);
    EXPECT_DOUBLE_EQ(C[1 * n_cart + 1], 0.0);
    EXPECT_DOUBLE_EQ(C[1 * n_cart + 2], 0.0);

    // Row 2 (py): 0*x + 1*y + 0*z
    EXPECT_DOUBLE_EQ(C[2 * n_cart + 1], 1.0);
    EXPECT_DOUBLE_EQ(C[2 * n_cart + 0], 0.0);
    EXPECT_DOUBLE_EQ(C[2 * n_cart + 2], 0.0);
}

TEST_F(SphericalTransformTest, GTypeNonZeroCoefficients) {
    const double* C = get_cart_to_sph_matrix(4);
    const int n_sph = n_spherical(4);  // 9
    const int n_cart = n_cartesian(4); // 15

    // Each row should have at least one non-zero element
    for (int i = 0; i < n_sph; ++i) {
        double row_norm_sq = 0.0;
        for (int j = 0; j < n_cart; ++j) {
            row_norm_sq += C[i * n_cart + j] * C[i * n_cart + j];
        }
        EXPECT_GT(row_norm_sq, 0.0) << "Row " << i << " of G matrix has no nonzero entries";
    }
}

TEST_F(SphericalTransformTest, Transform1dFType) {
    // Test D-type transform_1d with a unit xz Cartesian input
    // Cart: xxx=0, xxy=0, xxz=0, xyy=0, xyz=1, xzz=0, yyy=0, yyz=0, yzz=0, zzz=0
    double cart[10] = {0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0};
    double sph[7] = {0.0};

    transform_1d(3, cart, sph);

    // Only f-2 (index 4) should be nonzero: sqrt(15) * xyz
    EXPECT_NEAR(sph[4], std::sqrt(15.0), TOL);

    // All others should be zero for pure xyz input
    EXPECT_NEAR(sph[0], 0.0, TOL);
    EXPECT_NEAR(sph[1], 0.0, TOL);
    EXPECT_NEAR(sph[2], 0.0, TOL);
    EXPECT_NEAR(sph[3], 0.0, TOL);
    EXPECT_NEAR(sph[5], 0.0, TOL);
    EXPECT_NEAR(sph[6], 0.0, TOL);
}

// ============================================================================
// Task 3.3.4: Orthogonality Tests (C * C^T is diagonal)
// ============================================================================

TEST_F(SphericalTransformTest, OrthogonalitySType) {
    // C * C^T for S-type: [1.0] * [1.0]^T = [1.0] (identity)
    const double* C = get_cart_to_sph_matrix(0);
    EXPECT_DOUBLE_EQ(C[0] * C[0], 1.0);
}

TEST_F(SphericalTransformTest, OrthogonalityPType) {
    // C * C^T for P-type should be identity (permutation matrix)
    const double* C = get_cart_to_sph_matrix(1);
    const int n_sph = 3;
    const int n_cart = 3;

    for (int i = 0; i < n_sph; ++i) {
        for (int j = 0; j < n_sph; ++j) {
            double dot = 0.0;
            for (int k = 0; k < n_cart; ++k) {
                dot += C[i * n_cart + k] * C[j * n_cart + k];
            }
            double expected = (i == j) ? 1.0 : 0.0;
            EXPECT_NEAR(dot, expected, TOL)
                << "C_P * C_P^T [" << i << "][" << j << "]";
        }
    }
}

TEST_F(SphericalTransformTest, OrthogonalityDType) {
    // For D-type: rows of C should be orthogonal (off-diagonal of C*C^T = 0)
    const double* C = get_cart_to_sph_matrix(2);
    const int n_sph = 5;
    const int n_cart = 6;

    for (int i = 0; i < n_sph; ++i) {
        for (int j = i + 1; j < n_sph; ++j) {
            double dot = 0.0;
            for (int k = 0; k < n_cart; ++k) {
                dot += C[i * n_cart + k] * C[j * n_cart + k];
            }
            EXPECT_NEAR(dot, 0.0, TOL)
                << "Rows " << i << " and " << j << " of C_D are not orthogonal";
        }
    }

    // Diagonal of C*C^T should be positive
    for (int i = 0; i < n_sph; ++i) {
        double norm_sq = 0.0;
        for (int k = 0; k < n_cart; ++k) {
            norm_sq += C[i * n_cart + k] * C[i * n_cart + k];
        }
        EXPECT_GT(norm_sq, 0.0) << "Row " << i << " has zero norm";
    }
}

// For l >= 3, Cartesian monomials are NOT orthogonal on the unit sphere,
// so C * C^T != I.  The correct property is C * S * C^T = (4pi/(2l+1)) * I,
// where S is the Cartesian overlap matrix on the unit sphere.
//
// Unit sphere integral: int x^a y^b z^c dOmega = 0 if any exponent is odd,
// otherwise = 4*pi * (a-1)!! * (b-1)!! * (c-1)!! / (a+b+c+1)!!
// with (-1)!! = 1 by convention.

namespace {

double double_factorial(int n) {
    // (-1)!! = 1, 0!! = 1, 1!! = 1, 3!! = 3, 5!! = 15, ...
    if (n <= 0) return 1.0;
    double result = 1.0;
    for (int i = n; i >= 2; i -= 2) result *= i;
    return result;
}

// Overlap of x^{lx1+lx2} y^{ly1+ly2} z^{lz1+lz2} on the unit sphere.
double cart_overlap_sphere(int lx1, int ly1, int lz1,
                           int lx2, int ly2, int lz2) {
    int a = lx1 + lx2, b = ly1 + ly2, c = lz1 + lz2;
    if (a % 2 != 0 || b % 2 != 0 || c % 2 != 0) return 0.0;
    return 4.0 * M_PI * double_factorial(a - 1) * double_factorial(b - 1)
                      * double_factorial(c - 1) / double_factorial(a + b + c + 1);
}

struct CartIndex { int lx, ly, lz; };

std::vector<CartIndex> cart_indices(int l) {
    std::vector<CartIndex> idx;
    for (int lx = l; lx >= 0; --lx)
        for (int ly = l - lx; ly >= 0; --ly)
            idx.push_back({lx, ly, l - lx - ly});
    return idx;
}

// Compute C * S_cart * C^T and verify = (4*pi/(2l+1)) * I.
void check_overlap_orthogonality(const double* C, int l, double tol) {
    auto idx = cart_indices(l);
    int n_cart = static_cast<int>(idx.size());
    int n_sph = 2 * l + 1;
    double expected_diag = 4.0 * M_PI / (2 * l + 1);

    // Build S_cart
    std::vector<double> S(n_cart * n_cart);
    for (int a = 0; a < n_cart; ++a)
        for (int b = 0; b < n_cart; ++b)
            S[a * n_cart + b] = cart_overlap_sphere(
                idx[a].lx, idx[a].ly, idx[a].lz,
                idx[b].lx, idx[b].ly, idx[b].lz);

    // Compute C * S * C^T
    for (int i = 0; i < n_sph; ++i) {
        for (int j = i; j < n_sph; ++j) {
            double val = 0.0;
            for (int a = 0; a < n_cart; ++a)
                for (int b = 0; b < n_cart; ++b)
                    val += C[i * n_cart + a] * S[a * n_cart + b] * C[j * n_cart + b];
            if (i == j) {
                EXPECT_NEAR(val, expected_diag, tol)
                    << "(C*S*C^T)[" << i << "][" << i
                    << "] should be 4*pi/(2*" << l << "+1) = " << expected_diag;
            } else {
                EXPECT_NEAR(val, 0.0, tol)
                    << "(C*S*C^T)[" << i << "][" << j << "] should be 0";
            }
        }
    }
}

}  // namespace

TEST_F(SphericalTransformTest, OrthogonalityFType) {
    const double* C = get_cart_to_sph_matrix(3);
    check_overlap_orthogonality(C, 3, TOL);
}

TEST_F(SphericalTransformTest, OrthogonalityGType) {
    const double* C = get_cart_to_sph_matrix(4);
    check_overlap_orthogonality(C, 4, TOL);
}

TEST_F(SphericalTransformTest, OutOfRangeHAndIThrow) {
    EXPECT_THROW(get_cart_to_sph_matrix(5), std::out_of_range);
    EXPECT_THROW(get_cart_to_sph_matrix(6), std::out_of_range);
}
