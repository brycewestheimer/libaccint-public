// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2026 Bryce M. Westheimer. All rights reserved.

/// @file test_types.cpp
/// @brief Unit tests for fundamental types

#include <gtest/gtest.h>
#include <libaccint/core/types.hpp>

namespace libaccint {

// ============================================================================
// Test n_cartesian function
// ============================================================================

TEST(TypesTest, CartesianFunctionCount) {
    // Test for each angular momentum from S to I
    EXPECT_EQ(n_cartesian(0), 1);   // S: 1 function
    EXPECT_EQ(n_cartesian(1), 3);   // P: 3 functions
    EXPECT_EQ(n_cartesian(2), 6);   // D: 6 functions
    EXPECT_EQ(n_cartesian(3), 10);  // F: 10 functions
    EXPECT_EQ(n_cartesian(4), 15);  // G: 15 functions
    EXPECT_EQ(n_cartesian(5), 21);  // H: 21 functions
    EXPECT_EQ(n_cartesian(6), 28);  // I: 28 functions
}

// ============================================================================
// Test n_spherical function
// ============================================================================

TEST(TypesTest, SphericalFunctionCount) {
    // Test for each angular momentum from S to I
    EXPECT_EQ(n_spherical(0), 1);   // S: 1 function
    EXPECT_EQ(n_spherical(1), 3);   // P: 3 functions
    EXPECT_EQ(n_spherical(2), 5);   // D: 5 functions
    EXPECT_EQ(n_spherical(3), 7);   // F: 7 functions
    EXPECT_EQ(n_spherical(4), 9);   // G: 9 functions
    EXPECT_EQ(n_spherical(5), 11);  // H: 11 functions
    EXPECT_EQ(n_spherical(6), 13);  // I: 13 functions
}

// ============================================================================
// Test n_cartesian_pair function
// ============================================================================

TEST(TypesTest, CartesianPairCount) {
    // Test pair of P and D shells
    EXPECT_EQ(n_cartesian_pair(1, 2), 18);  // 3 * 6 = 18

    // Test additional pairs
    EXPECT_EQ(n_cartesian_pair(0, 0), 1);   // S-S: 1 * 1 = 1
    EXPECT_EQ(n_cartesian_pair(1, 1), 9);   // P-P: 3 * 3 = 9
    EXPECT_EQ(n_cartesian_pair(2, 3), 60);  // D-F: 6 * 10 = 60
}

// ============================================================================
// Test AngularMomentum enum
// ============================================================================

TEST(TypesTest, AngularMomentumEnum) {
    // Test enum values
    EXPECT_EQ(static_cast<int>(AngularMomentum::S), 0);
    EXPECT_EQ(static_cast<int>(AngularMomentum::P), 1);
    EXPECT_EQ(static_cast<int>(AngularMomentum::D), 2);
    EXPECT_EQ(static_cast<int>(AngularMomentum::F), 3);
    EXPECT_EQ(static_cast<int>(AngularMomentum::G), 4);
    EXPECT_EQ(static_cast<int>(AngularMomentum::H), 5);
    EXPECT_EQ(static_cast<int>(AngularMomentum::I), 6);
}

// ============================================================================
// Test to_int conversion
// ============================================================================

TEST(TypesTest, ToIntConversion) {
    EXPECT_EQ(to_int(AngularMomentum::S), 0);
    EXPECT_EQ(to_int(AngularMomentum::P), 1);
    EXPECT_EQ(to_int(AngularMomentum::D), 2);
    EXPECT_EQ(to_int(AngularMomentum::F), 3);
    EXPECT_EQ(to_int(AngularMomentum::G), 4);
    EXPECT_EQ(to_int(AngularMomentum::H), 5);
    EXPECT_EQ(to_int(AngularMomentum::I), 6);
}

// ============================================================================
// Test constants
// ============================================================================

TEST(TypesTest, Constants) {
    EXPECT_EQ(MAX_ANGULAR_MOMENTUM, 4);
    EXPECT_EQ(MAX_RYS_ROOTS, 15);
}

// ============================================================================
// Test Point3D
// ============================================================================

TEST(TypesTest, Point3DConstruction) {
    // Test default constructor
    Point3D p1;
    EXPECT_DOUBLE_EQ(p1.x, 0.0);
    EXPECT_DOUBLE_EQ(p1.y, 0.0);
    EXPECT_DOUBLE_EQ(p1.z, 0.0);

    // Test parameterized constructor
    Point3D p2(1.0, 2.0, 3.0);
    EXPECT_DOUBLE_EQ(p2.x, 1.0);
    EXPECT_DOUBLE_EQ(p2.y, 2.0);
    EXPECT_DOUBLE_EQ(p2.z, 3.0);
}

TEST(TypesTest, Point3DIndexOperator) {
    Point3D p(1.0, 2.0, 3.0);

    // Test const index operator
    EXPECT_DOUBLE_EQ(p[0], 1.0);
    EXPECT_DOUBLE_EQ(p[1], 2.0);
    EXPECT_DOUBLE_EQ(p[2], 3.0);

    // Test non-const index operator
    p[0] = 4.0;
    p[1] = 5.0;
    p[2] = 6.0;
    EXPECT_DOUBLE_EQ(p.x, 4.0);
    EXPECT_DOUBLE_EQ(p.y, 5.0);
    EXPECT_DOUBLE_EQ(p.z, 6.0);
}

TEST(TypesTest, Point3DDistanceSquared) {
    Point3D p1(0.0, 0.0, 0.0);
    Point3D p2(1.0, 0.0, 0.0);
    Point3D p3(1.0, 1.0, 0.0);
    Point3D p4(1.0, 1.0, 1.0);

    // Test distance calculations
    EXPECT_DOUBLE_EQ(p1.distance_squared(p2), 1.0);
    EXPECT_DOUBLE_EQ(p1.distance_squared(p3), 2.0);
    EXPECT_DOUBLE_EQ(p1.distance_squared(p4), 3.0);

    // Test symmetry
    EXPECT_DOUBLE_EQ(p2.distance_squared(p1), 1.0);

    // Test self-distance
    EXPECT_DOUBLE_EQ(p1.distance_squared(p1), 0.0);
}

// ============================================================================
// Test n_derivative_components
// ============================================================================

TEST(TypesTest, DerivativeComponents) {
    // Test energy (0th order) - always 1 component
    EXPECT_EQ((n_derivative_components<0, 1>()), 1);
    EXPECT_EQ((n_derivative_components<0, 2>()), 1);
    EXPECT_EQ((n_derivative_components<0, 3>()), 1);
    EXPECT_EQ((n_derivative_components<0, 4>()), 1);

    // Test gradient (1st order) - 3 * NCenters
    EXPECT_EQ((n_derivative_components<1, 1>()), 3);   // 3 * 1 = 3
    EXPECT_EQ((n_derivative_components<1, 2>()), 6);   // 3 * 2 = 6
    EXPECT_EQ((n_derivative_components<1, 3>()), 9);   // 3 * 3 = 9
    EXPECT_EQ((n_derivative_components<1, 4>()), 12);  // 3 * 4 = 12

    // Test hessian (2nd order) - n_first * (n_first + 1) / 2
    EXPECT_EQ((n_derivative_components<2, 1>()), 6);   // 3 * 4 / 2 = 6
    EXPECT_EQ((n_derivative_components<2, 2>()), 21);  // 6 * 7 / 2 = 21
    EXPECT_EQ((n_derivative_components<2, 3>()), 45);  // 9 * 10 / 2 = 45
    EXPECT_EQ((n_derivative_components<2, 4>()), 78);  // 12 * 13 / 2 = 78
}

// ============================================================================
// Test DerivativeOrder enum
// ============================================================================

TEST(TypesTest, DerivativeOrderEnum) {
    EXPECT_EQ(static_cast<int>(DerivativeOrder::Energy), 0);
    EXPECT_EQ(static_cast<int>(DerivativeOrder::Gradient), 1);
    EXPECT_EQ(static_cast<int>(DerivativeOrder::Hessian), 2);
}

// ============================================================================
// Test array types
// ============================================================================

TEST(TypesTest, AMArrayTypes) {
    // Test AMQuartet
    AMQuartet quartet = {0, 1, 2, 3};
    EXPECT_EQ(quartet.size(), 4);
    EXPECT_EQ(quartet[0], 0);
    EXPECT_EQ(quartet[3], 3);

    // Test AMPair
    AMPair pair = {1, 2};
    EXPECT_EQ(pair.size(), 2);
    EXPECT_EQ(pair[0], 1);
    EXPECT_EQ(pair[1], 2);

    // Test AMTriplet
    AMTriplet triplet = {0, 1, 2};
    EXPECT_EQ(triplet.size(), 3);
    EXPECT_EQ(triplet[0], 0);
    EXPECT_EQ(triplet[2], 2);
}

// ============================================================================
// Test Numeric concept
// ============================================================================

TEST(TypesTest, NumericConcept) {
    // These should compile if the concept is correctly defined
    static_assert(Numeric<float>, "float should satisfy Numeric");
    static_assert(Numeric<double>, "double should satisfy Numeric");
    static_assert(Numeric<long double>, "long double should satisfy Numeric");
    static_assert(!Numeric<int>, "int should not satisfy Numeric");
    static_assert(!Numeric<bool>, "bool should not satisfy Numeric");
}

// ============================================================================
// Test n_functions (Task 1.4.4)
// ============================================================================

TEST(TypesTest, NFunctions_Cartesian) {
    // n_functions(am, false) should equal n_cartesian(am)
    for (int am = 0; am <= 4; ++am) {
        EXPECT_EQ(n_functions(am, false), n_cartesian(am));
    }
}

TEST(TypesTest, NFunctions_Spherical) {
    // n_functions(am, true) should equal n_spherical(am)
    for (int am = 0; am <= 4; ++am) {
        EXPECT_EQ(n_functions(am, true), n_spherical(am));
    }
}

// ============================================================================
// Test n_spherical_pair (Task 1.4.4)
// ============================================================================

TEST(TypesTest, NSphericalPair_Combinations) {
    // n_spherical_pair(la, lb) = n_spherical(la) * n_spherical(lb) = (2*la+1) * (2*lb+1)
    EXPECT_EQ(n_spherical_pair(0, 0), 1);    // 1 * 1
    EXPECT_EQ(n_spherical_pair(0, 1), 3);    // 1 * 3
    EXPECT_EQ(n_spherical_pair(1, 0), 3);    // 3 * 1
    EXPECT_EQ(n_spherical_pair(1, 1), 9);    // 3 * 3
    EXPECT_EQ(n_spherical_pair(1, 2), 15);   // 3 * 5
    EXPECT_EQ(n_spherical_pair(2, 2), 25);   // 5 * 5
    EXPECT_EQ(n_spherical_pair(2, 3), 35);   // 5 * 7

    // Verify against definition
    for (int la = 0; la <= 4; ++la) {
        for (int lb = 0; lb <= 4; ++lb) {
            EXPECT_EQ(n_spherical_pair(la, lb), n_spherical(la) * n_spherical(lb));
        }
    }
}

// ============================================================================
// Additional Point3D tests (Task 1.4.4)
// ============================================================================

TEST(TypesTest, Point3D_DistanceSquared_KnownValues) {
    // (0,0,0) to (1,0,0) = 1.0
    Point3D origin(0.0, 0.0, 0.0);
    Point3D unit_x(1.0, 0.0, 0.0);
    EXPECT_DOUBLE_EQ(origin.distance_squared(unit_x), 1.0);

    // (1,2,3) to (4,5,6) = 9+9+9 = 27.0
    Point3D p1(1.0, 2.0, 3.0);
    Point3D p2(4.0, 5.0, 6.0);
    EXPECT_DOUBLE_EQ(p1.distance_squared(p2), 27.0);

    // Symmetry check
    EXPECT_DOUBLE_EQ(p2.distance_squared(p1), 27.0);
}

TEST(TypesTest, Point3D_OperatorBracket_MatchesFields) {
    Point3D p(1.5, 2.5, 3.5);
    EXPECT_EQ(p[0], p.x);
    EXPECT_EQ(p[1], p.y);
    EXPECT_EQ(p[2], p.z);
}

TEST(TypesTest, Point3D_OperatorBracketConst) {
    const Point3D p(10.0, 20.0, 30.0);
    EXPECT_DOUBLE_EQ(p[0], 10.0);
    EXPECT_DOUBLE_EQ(p[1], 20.0);
    EXPECT_DOUBLE_EQ(p[2], 30.0);
}

// ============================================================================
// Test n_derivative_components additional values (Task 1.4.4)
// ============================================================================

TEST(TypesTest, NDerivativeComponents_Values) {
    // DerivOrder=0: always 1
    EXPECT_EQ((n_derivative_components<0, 1>()), 1);
    EXPECT_EQ((n_derivative_components<0, 2>()), 1);

    // DerivOrder=1, 2 centers: 3 * 2 = 6
    EXPECT_EQ((n_derivative_components<1, 2>()), 6);

    // DerivOrder=1, 4 centers: 3 * 4 = 12
    EXPECT_EQ((n_derivative_components<1, 4>()), 12);

    // DerivOrder=2, 2 centers: n_first=6, 6*7/2 = 21
    EXPECT_EQ((n_derivative_components<2, 2>()), 21);

    // DerivOrder=2, 4 centers: n_first=12, 12*13/2 = 78
    EXPECT_EQ((n_derivative_components<2, 4>()), 78);
}

}  // namespace libaccint
