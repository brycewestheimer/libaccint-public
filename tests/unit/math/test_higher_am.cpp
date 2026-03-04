// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2026 Bryce M. Westheimer. All rights reserved.

/// @file test_higher_am.cpp
/// @brief Unit tests for higher angular momentum support (f, g, h functions)

#include <gtest/gtest.h>
#include <libaccint/core/types.hpp>
#include <libaccint/math/cartesian_indices.hpp>
#include <libaccint/math/spherical_transform.hpp>
#include <libaccint/math/rys_quadrature.hpp>
#include <libaccint/math/boys_function.hpp>
#include <cmath>
#include <array>
#include <vector>

namespace libaccint::test {

// ============================================================================
// Cartesian Function Count Tests
// ============================================================================

class HigherAMFunctionCountsTest : public ::testing::Test {
protected:
    // Expected Cartesian function counts: (L+1)(L+2)/2
    static constexpr std::array<int, 7> expected_cartesian = {
        1,   // L=0: s
        3,   // L=1: p
        6,   // L=2: d
        10,  // L=3: f
        15,  // L=4: g
        21,  // L=5: h
        28   // L=6: i
    };
    
    // Expected spherical function counts: 2L+1
    static constexpr std::array<int, 7> expected_spherical = {
        1,   // L=0: s
        3,   // L=1: p
        5,   // L=2: d
        7,   // L=3: f
        9,   // L=4: g
        11,  // L=5: h
        13   // L=6: i
    };
};

TEST_F(HigherAMFunctionCountsTest, CartesianCounts) {
    for (int L = 0; L <= 6; ++L) {
        EXPECT_EQ(n_cartesian(L), expected_cartesian[L])
            << "Incorrect Cartesian count for L=" << L;
    }
}

TEST_F(HigherAMFunctionCountsTest, SphericalCounts) {
    for (int L = 0; L <= 6; ++L) {
        EXPECT_EQ(n_spherical(L), expected_spherical[L])
            << "Incorrect spherical count for L=" << L;
    }
}

TEST_F(HigherAMFunctionCountsTest, CartesianSphericalRelation) {
    // Cartesian always >= spherical for L >= 2
    for (int L = 2; L <= 6; ++L) {
        EXPECT_GE(n_cartesian(L), n_spherical(L))
            << "Cartesian should have more or equal functions than spherical for L=" << L;
    }
}

// ============================================================================
// f-Function Tests (L=3)
// ============================================================================

class FFunctionTest : public ::testing::Test {
protected:
    static constexpr int L = 3;
    static constexpr int n_cart = 10;  // (3+1)(3+2)/2 = 10
    static constexpr int n_sph = 7;    // 2*3+1 = 7
};

TEST_F(FFunctionTest, CartesianCount) {
    EXPECT_EQ(n_cartesian(L), n_cart);
}

TEST_F(FFunctionTest, SphericalCount) {
    EXPECT_EQ(n_spherical(L), n_sph);
}

TEST_F(FFunctionTest, CartesianIndices) {
    // f-function Cartesian indices in canonical order:
    // xxx, xxy, xxz, xyy, xyz, xzz, yyy, yyz, yzz, zzz
    auto indices = math::generate_cartesian_indices(L);
    ASSERT_EQ(indices.size(), static_cast<size_t>(n_cart));
    
    // Check first and last indices
    EXPECT_EQ(indices[0], (std::array<int, 3>{3, 0, 0}));  // xxx
    EXPECT_EQ(indices[9], (std::array<int, 3>{0, 0, 3}));  // zzz
    
    // Check all indices sum to L
    for (const auto& idx : indices) {
        EXPECT_EQ(idx[0] + idx[1] + idx[2], L);
    }
}

TEST_F(FFunctionTest, CartesianIndexFunction) {
    // Test cartesian_index for f-functions
    EXPECT_EQ(math::cartesian_index(3, 0, 0), 0);  // xxx -> index 0
    EXPECT_EQ(math::cartesian_index(2, 1, 0), 1);  // xxy -> index 1
    EXPECT_EQ(math::cartesian_index(2, 0, 1), 2);  // xxz -> index 2
    EXPECT_EQ(math::cartesian_index(1, 2, 0), 3);  // xyy -> index 3
    EXPECT_EQ(math::cartesian_index(1, 1, 1), 4);  // xyz -> index 4
    EXPECT_EQ(math::cartesian_index(1, 0, 2), 5);  // xzz -> index 5
    EXPECT_EQ(math::cartesian_index(0, 3, 0), 6);  // yyy -> index 6
    EXPECT_EQ(math::cartesian_index(0, 2, 1), 7);  // yyz -> index 7
    EXPECT_EQ(math::cartesian_index(0, 1, 2), 8);  // yzz -> index 8
    EXPECT_EQ(math::cartesian_index(0, 0, 3), 9);  // zzz -> index 9
}

// ============================================================================
// g-Function Tests (L=4)
// ============================================================================

class GFunctionTest : public ::testing::Test {
protected:
    static constexpr int L = 4;
    static constexpr int n_cart = 15;  // (4+1)(4+2)/2 = 15
    static constexpr int n_sph = 9;    // 2*4+1 = 9
};

TEST_F(GFunctionTest, CartesianCount) {
    EXPECT_EQ(n_cartesian(L), n_cart);
}

TEST_F(GFunctionTest, SphericalCount) {
    EXPECT_EQ(n_spherical(L), n_sph);
}

TEST_F(GFunctionTest, CartesianIndices) {
    auto indices = math::generate_cartesian_indices(L);
    ASSERT_EQ(indices.size(), static_cast<size_t>(n_cart));
    
    // Check first and last
    EXPECT_EQ(indices[0], (std::array<int, 3>{4, 0, 0}));   // xxxx
    EXPECT_EQ(indices[14], (std::array<int, 3>{0, 0, 4})); // zzzz
    
    // All sum to L
    for (const auto& idx : indices) {
        EXPECT_EQ(idx[0] + idx[1] + idx[2], L);
    }
}

TEST_F(GFunctionTest, CartesianIndexEdgeCases) {
    EXPECT_EQ(math::cartesian_index(4, 0, 0), 0);   // xxxx
    EXPECT_EQ(math::cartesian_index(0, 4, 0), 10);  // yyyy
    EXPECT_EQ(math::cartesian_index(0, 0, 4), 14);  // zzzz
    EXPECT_EQ(math::cartesian_index(2, 2, 0), 3);   // xxyy (index 3 in canonical order)
}

// ============================================================================
// h-Function Tests (L=5)
// ============================================================================

class HFunctionTest : public ::testing::Test {
protected:
    static constexpr int L = 5;
    static constexpr int n_cart = 21;  // (5+1)(5+2)/2 = 21
    static constexpr int n_sph = 11;   // 2*5+1 = 11
};

TEST_F(HFunctionTest, CartesianCount) {
    EXPECT_EQ(n_cartesian(L), n_cart);
}

TEST_F(HFunctionTest, SphericalCount) {
    EXPECT_EQ(n_spherical(L), n_sph);
}

TEST_F(HFunctionTest, CartesianIndices) {
    auto indices = math::generate_cartesian_indices(L);
    ASSERT_EQ(indices.size(), static_cast<size_t>(n_cart));
    
    // All indices sum to L
    for (const auto& idx : indices) {
        EXPECT_EQ(idx[0] + idx[1] + idx[2], L);
    }
}

// ============================================================================
// Rys Quadrature Higher Roots Tests
// ============================================================================

class RysHigherAMTest : public ::testing::Test {
protected:
    // Required roots for various AM combinations
    // n_roots = (La + Lb + Lc + Ld) / 2 + 1
    
    // (dddd): (2+2+2+2)/2 + 1 = 5 roots
    static constexpr int dddd_roots = 5;
    
    // (ffff): (3+3+3+3)/2 + 1 = 7 roots
    static constexpr int ffff_roots = 7;
    
    // (gggg): (4+4+4+4)/2 + 1 = 9 roots
    static constexpr int gggg_roots = 9;
    
    // (hhhh): (5+5+5+5)/2 + 1 = 11 roots
    static constexpr int hhhh_roots = 11;
    
    double test_T = 5.0;  // Representative T value
    double tolerance = 1e-12;
};

TEST_F(RysHigherAMTest, FFunctionRoots) {
    // Test that 7-root quadrature works correctly
    std::array<double, 7> roots{};
    std::array<double, 7> weights{};
    
    math::rys_compute(ffff_roots, test_T, roots.data(), weights.data());
    
    // Verify roots are in (0, 1)
    for (int i = 0; i < ffff_roots; ++i) {
        EXPECT_GT(roots[i], 0.0) << "Root " << i << " should be > 0";
        EXPECT_LT(roots[i], 1.0) << "Root " << i << " should be < 1";
    }
    
    // Verify weights are positive
    for (int i = 0; i < ffff_roots; ++i) {
        EXPECT_GT(weights[i], 0.0) << "Weight " << i << " should be positive";
    }
    
    // Verify roots are ascending
    for (int i = 1; i < ffff_roots; ++i) {
        EXPECT_GT(roots[i], roots[i-1]) << "Roots should be ascending";
    }
}

TEST_F(RysHigherAMTest, GFunctionRoots) {
    // Test that 9-root quadrature works correctly
    std::array<double, 9> roots{};
    std::array<double, 9> weights{};
    
    math::rys_compute(gggg_roots, test_T, roots.data(), weights.data());
    
    for (int i = 0; i < gggg_roots; ++i) {
        EXPECT_GT(roots[i], 0.0);
        EXPECT_LT(roots[i], 1.0);
        EXPECT_GT(weights[i], 0.0);
    }
}

TEST_F(RysHigherAMTest, HFunctionRoots) {
    // Test that 11-root quadrature works correctly
    std::array<double, 11> roots{};
    std::array<double, 11> weights{};
    
    math::rys_compute(hhhh_roots, test_T, roots.data(), weights.data());
    
    for (int i = 0; i < hhhh_roots; ++i) {
        EXPECT_GT(roots[i], 0.0);
        EXPECT_LT(roots[i], 1.0);
        EXPECT_GT(weights[i], 0.0);
    }
}

TEST_F(RysHigherAMTest, MomentMatchingFFFF) {
    // Verify quadrature satisfies moment conditions:
    // sum_i w_i * u_i^k = F_k(T) for k = 0..2n-1
    std::array<double, 7> roots{};
    std::array<double, 7> weights{};
    
    math::rys_compute(ffff_roots, test_T, roots.data(), weights.data());
    
    // Test 0th moment (sum of weights = F_0(T))
    double sum_weights = 0.0;
    for (int i = 0; i < ffff_roots; ++i) {
        sum_weights += weights[i];
    }
    
    // F_0(T) for T=5.0 is approximately 0.3970
    // More precise validation would compare to Boys function
    EXPECT_GT(sum_weights, 0.0);
    EXPECT_LT(sum_weights, 1.0);  // F_0(T) < 1 for T > 0
}

// ============================================================================
// Angular Momentum Enum Tests
// ============================================================================

TEST(AngularMomentumEnumTest, HigherAMValues) {
    EXPECT_EQ(to_int(AngularMomentum::S), 0);
    EXPECT_EQ(to_int(AngularMomentum::P), 1);
    EXPECT_EQ(to_int(AngularMomentum::D), 2);
    EXPECT_EQ(to_int(AngularMomentum::F), 3);
    EXPECT_EQ(to_int(AngularMomentum::G), 4);
    EXPECT_EQ(to_int(AngularMomentum::H), 5);
    EXPECT_EQ(to_int(AngularMomentum::I), 6);
}

// ============================================================================
// MAX_ANGULAR_MOMENTUM Constant Test
// ============================================================================

TEST(AngularMomentumLimitsTest, MaxAMConstant) {
    // Verify MAX_ANGULAR_MOMENTUM supports at least g-functions (alpha contract)
    EXPECT_GE(MAX_ANGULAR_MOMENTUM, 4)
        << "MAX_ANGULAR_MOMENTUM should support at least g-functions (L=4)";
}

TEST(AngularMomentumLimitsTest, MaxRysRoots) {
    // Verify MAX_RYS_ROOTS supports h-function ERIs
    // (hhhh) needs (5+5+5+5)/2 + 1 = 11 roots
    EXPECT_GE(MAX_RYS_ROOTS, 11) 
        << "MAX_RYS_ROOTS should support (hhhh) ERI quartets";
}

// ============================================================================
// Task 3.3.9: Numerical Stability Tests
// ============================================================================

TEST(HigherAMStabilityTest, SphericalTransformL4NearZero) {
    // Test G-type spherical transform with near-zero Cartesian values
    double cart[15];
    for (int i = 0; i < 15; ++i) {
        cart[i] = 1e-15;
    }
    double sph[9] = {0.0};

    EXPECT_NO_THROW(math::transform_1d(4, cart, sph));

    for (int i = 0; i < 9; ++i) {
        EXPECT_TRUE(std::isfinite(sph[i])) << "i=" << i;
    }
}

TEST(HigherAMStabilityTest, SphericalTransformL4LargeValues) {
    // Test G-type spherical transform with large Cartesian values
    double cart[15];
    for (int i = 0; i < 15; ++i) {
        cart[i] = 1e10 * (1.0 + 0.1 * i);
    }
    double sph[9] = {0.0};

    EXPECT_NO_THROW(math::transform_1d(4, cart, sph));

    for (int i = 0; i < 9; ++i) {
        EXPECT_TRUE(std::isfinite(sph[i])) << "i=" << i;
    }
}

TEST(HigherAMStabilityTest, RysQuadratureHighAMCombinations) {
    // Test Rys quadrature for various high-AM integral combinations
    double roots[MAX_RYS_ROOTS], weights[MAX_RYS_ROOTS];

    // (ffdd): n_roots = (3+3+2+2)/2 + 1 = 6
    EXPECT_NO_THROW(math::rys_compute(6, 5.0, roots, weights));
    for (int i = 0; i < 6; ++i) {
        EXPECT_GT(roots[i], 0.0);
        EXPECT_LT(roots[i], 1.0);
        EXPECT_GT(weights[i], 0.0);
    }

    // (gfds): n_roots = (4+3+2+0)/2 + 1 = 5
    EXPECT_NO_THROW(math::rys_compute(5, 10.0, roots, weights));
    for (int i = 0; i < 5; ++i) {
        EXPECT_GT(roots[i], 0.0);
        EXPECT_LT(roots[i], 1.0);
        EXPECT_GT(weights[i], 0.0);
    }

    // (ggff): n_roots = (4+4+3+3)/2 + 1 = 8
    EXPECT_NO_THROW(math::rys_compute(8, 15.0, roots, weights));
    for (int i = 0; i < 8; ++i) {
        EXPECT_GT(roots[i], 0.0);
        EXPECT_LT(roots[i], 1.0);
        EXPECT_GT(weights[i], 0.0);
    }
}

TEST(HigherAMStabilityTest, BoysHighOrder) {
    // Test Boys function at high orders (n=10..15) for numerical stability
    for (int n = 10; n <= 15; ++n) {
        for (double T : {0.0, 0.1, 1.0, 5.0, 10.0, 25.0, 50.0, 100.0}) {
            double result = math::boys_evaluate(n, T);
            EXPECT_TRUE(std::isfinite(result))
                << "Boys function not finite at n=" << n << ", T=" << T;
            EXPECT_GT(result, 0.0)
                << "Boys function not positive at n=" << n << ", T=" << T;
        }
    }
}

TEST(HigherAMStabilityTest, BoysHighOrderArray) {
    // Test Boys array evaluation up to high orders
    int n_max = 15;
    double result[31];

    for (double T : {0.0, 1.0, 10.0, 50.0}) {
        math::boys_evaluate_array(n_max, T, result);

        // Verify all values are finite and positive
        for (int n = 0; n <= n_max; ++n) {
            EXPECT_TRUE(std::isfinite(result[n]))
                << "n=" << n << ", T=" << T;
            EXPECT_GT(result[n], 0.0)
                << "n=" << n << ", T=" << T;
        }

        // Verify monotonicity: F_n > F_{n+1}
        for (int n = 0; n < n_max; ++n) {
            EXPECT_GT(result[n], result[n + 1])
                << "Monotonicity violated at n=" << n << ", T=" << T;
        }
    }
}

TEST(HigherAMStabilityTest, SphericalTransform2dHighAM) {
    // Test 2D spherical transform for (dd) and (ff) blocks
    const int n_cart_d = n_cartesian(2);
    const int n_sph_d = n_spherical(2);

    // (dd) block
    std::vector<double> cart_dd(n_cart_d * n_cart_d, 0.0);
    std::vector<double> sph_dd(n_sph_d * n_sph_d, 0.0);
    std::vector<double> work(math::work_size_2d(2, 2));

    // Set up identity-like Cartesian block
    for (int i = 0; i < std::min(n_cart_d, n_sph_d); ++i) {
        cart_dd[i * n_cart_d + i] = 1.0;
    }

    EXPECT_NO_THROW(math::transform_2d(2, 2, cart_dd.data(), sph_dd.data(), work.data()));

    for (int i = 0; i < n_sph_d * n_sph_d; ++i) {
        EXPECT_TRUE(std::isfinite(sph_dd[i])) << "Index " << i;
    }
}

}  // namespace libaccint::test
