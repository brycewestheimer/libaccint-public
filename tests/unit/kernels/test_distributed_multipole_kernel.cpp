// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2026 Bryce M. Westheimer. All rights reserved.

/// @file test_distributed_multipole_kernel.cpp
/// @brief Validation tests for distributed multipole kernel

#include <libaccint/kernels/distributed_multipole_kernel.hpp>
#include <libaccint/basis/shell.hpp>
#include <libaccint/operators/operator_types.hpp>
#include <libaccint/utils/constants.hpp>
#include <gtest/gtest.h>
#include <cmath>

using namespace libaccint;

namespace {
constexpr Real TIGHT_TOL = 1e-12;
constexpr Real LOOSE_TOL = 1e-8;
}

// ============================================================================
// Test 1: Single point charge — compare to nuclear attraction
// ============================================================================
TEST(DistributedMultipoleKernelTest, SinglePointChargeStructure) {
    // A single unit point charge at some position should produce a matrix
    // with the same structure as a nuclear attraction integral
    Point3D A(0.0, 0.0, 0.0);
    Point3D B(0.0, 0.0, 0.0);
    Shell shell_a(AngularMomentum::S, A, {1.0}, {1.0});
    Shell shell_b(AngularMomentum::S, B, {1.0}, {1.0});

    DistributedMultipoleParams params;
    params.x = {0.0};
    params.y = {0.0};
    params.z = {1.0};  // charge at (0,0,1)
    params.charges = {1.0};

    OverlapBuffer buffer;

    kernels::compute_distributed_multipole(shell_a, shell_b, params, buffer);

    EXPECT_EQ(buffer.na(), 1);
    EXPECT_EQ(buffer.nb(), 1);

    // Should be a negative value (attractive potential, -Z/r integral)
    EXPECT_LT(buffer(0, 0), 0.0)
        << "Nuclear attraction from positive charge should give negative integral";
}

// ============================================================================
// Test 2: Multiple point charges — linearity
// ============================================================================
TEST(DistributedMultipoleKernelTest, MultipleChargesLinearity) {
    // Two unit charges should give twice the result of one charge at same position
    Point3D A(0.0, 0.0, 0.0);
    Shell shell_a(AngularMomentum::S, A, {1.0}, {1.0});
    Shell shell_b(AngularMomentum::S, A, {1.0}, {1.0});

    // Single charge
    DistributedMultipoleParams params1;
    params1.x = {0.0};
    params1.y = {0.0};
    params1.z = {1.0};
    params1.charges = {1.0};

    OverlapBuffer buf1;
    kernels::compute_distributed_multipole(shell_a, shell_b, params1, buf1);

    // Double charge
    DistributedMultipoleParams params2;
    params2.x = {0.0};
    params2.y = {0.0};
    params2.z = {1.0};
    params2.charges = {2.0};

    OverlapBuffer buf2;
    kernels::compute_distributed_multipole(shell_a, shell_b, params2, buf2);

    EXPECT_NEAR(buf2(0, 0), 2.0 * buf1(0, 0), LOOSE_TOL)
        << "Doubling charge should double the integral";
}

// ============================================================================
// Test 3: Symmetry — V(a,b) = V(b,a)
// ============================================================================
TEST(DistributedMultipoleKernelTest, Symmetry) {
    Point3D A(0.0, 0.0, 0.0);
    Point3D B(0.5, 0.3, -0.2);
    Shell shell_a(AngularMomentum::S, A, {1.5}, {1.0});
    Shell shell_b(AngularMomentum::S, B, {0.8}, {1.0});

    DistributedMultipoleParams params;
    params.x = {0.0};
    params.y = {0.0};
    params.z = {1.0};
    params.charges = {1.0};

    OverlapBuffer buf_ab, buf_ba;

    kernels::compute_distributed_multipole(shell_a, shell_b, params, buf_ab);
    kernels::compute_distributed_multipole(shell_b, shell_a, params, buf_ba);

    EXPECT_NEAR(buf_ab(0, 0), buf_ba(0, 0), LOOSE_TOL)
        << "Nuclear attraction integral should be symmetric";
}

// ============================================================================
// Test 4: Multiple sites — additivity
// ============================================================================
TEST(DistributedMultipoleKernelTest, MultiSiteAdditivity) {
    Point3D A(0.0, 0.0, 0.0);
    Shell shell_a(AngularMomentum::S, A, {1.0}, {1.0});
    Shell shell_b(AngularMomentum::S, A, {1.0}, {1.0});

    // Two separate charges at different positions
    DistributedMultipoleParams params_both;
    params_both.x = {0.0, 1.0};
    params_both.y = {0.0, 0.0};
    params_both.z = {1.0, 0.0};
    params_both.charges = {1.0, 0.5};

    OverlapBuffer buf_both;
    kernels::compute_distributed_multipole(shell_a, shell_b, params_both, buf_both);

    // Individual charges
    DistributedMultipoleParams params1;
    params1.x = {0.0};
    params1.y = {0.0};
    params1.z = {1.0};
    params1.charges = {1.0};

    DistributedMultipoleParams params2;
    params2.x = {1.0};
    params2.y = {0.0};
    params2.z = {0.0};
    params2.charges = {0.5};

    OverlapBuffer buf1, buf2;
    kernels::compute_distributed_multipole(shell_a, shell_b, params1, buf1);
    kernels::compute_distributed_multipole(shell_a, shell_b, params2, buf2);

    EXPECT_NEAR(buf_both(0, 0), buf1(0, 0) + buf2(0, 0), LOOSE_TOL)
        << "Multi-site result should equal sum of individual sites";
}

// ============================================================================
// Test 5: p-p shell pair structure
// ============================================================================
TEST(DistributedMultipoleKernelTest, PPShellPairSymmetry) {
    Point3D center(0.0, 0.0, 0.0);
    Shell shell(AngularMomentum::P, center, {1.0}, {1.0});

    DistributedMultipoleParams params;
    params.x = {0.0};
    params.y = {0.0};
    params.z = {1.0};
    params.charges = {1.0};

    OverlapBuffer buffer;
    kernels::compute_distributed_multipole(shell, shell, params, buffer);

    EXPECT_EQ(buffer.na(), 3);
    EXPECT_EQ(buffer.nb(), 3);

    // Matrix should be symmetric for same shells
    for (int a = 0; a < 3; ++a) {
        for (int b = 0; b <= a; ++b) {
            EXPECT_NEAR(buffer(a, b), buffer(b, a), LOOSE_TOL)
                << "Self-pair nuclear attraction should be symmetric, a=" << a << " b=" << b;
        }
    }
}

// ============================================================================
// Test 6: Params validation
// ============================================================================
TEST(DistributedMultipoleKernelTest, ParamsNSites) {
    DistributedMultipoleParams params;
    params.x = {0.0, 1.0};
    params.y = {0.0, 0.0};
    params.z = {1.0, 0.0};
    params.charges = {1.0, 0.5};

    EXPECT_EQ(params.n_sites(), 2);
    EXPECT_EQ(params.max_rank(), 0);
    EXPECT_TRUE(params.is_valid());
}

TEST(DistributedMultipoleKernelTest, ParamsWithDipoles) {
    DistributedMultipoleParams params;
    params.x = {0.0};
    params.y = {0.0};
    params.z = {1.0};
    params.charges = {1.0};
    params.dipole_x = {0.1};
    params.dipole_y = {0.2};
    params.dipole_z = {0.3};

    EXPECT_EQ(params.n_sites(), 1);
    EXPECT_EQ(params.max_rank(), 1);
    EXPECT_TRUE(params.is_valid());
}

TEST(DistributedMultipoleKernelTest, ParamsInvalid) {
    DistributedMultipoleParams params;
    params.x = {0.0};
    params.y = {0.0};
    // z intentionally omitted — size mismatch with charges
    params.charges = {1.0};

    EXPECT_FALSE(params.is_valid());
}
