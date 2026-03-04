// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2026 Bryce M. Westheimer. All rights reserved.

/// @file test_integral_buffer.cpp
/// @brief Tests for IntegralBuffer construction, sizing, and derivative support

#include <libaccint/engine/integral_buffer.hpp>
#include <libaccint/engine/engine.hpp>
#include <libaccint/basis/basis_set.hpp>
#include <libaccint/basis/shell.hpp>
#include <libaccint/core/derivative_utils.hpp>
#include <libaccint/core/types.hpp>
#include <libaccint/operators/operator.hpp>

#include <gtest/gtest.h>
#include <vector>

using namespace libaccint;
using namespace libaccint::engine;

namespace {

// =============================================================================
// STO-3G H2O Test Data
// =============================================================================

constexpr Point3D O_center{0.0, 0.0, 0.0};
constexpr Point3D H1_center{0.0, 1.43233673, -1.10866041};
constexpr Point3D H2_center{0.0, -1.43233673, -1.10866041};

std::vector<Shell> make_sto3g_h2o_shells() {
    std::vector<Shell> shells;
    shells.reserve(5);

    {
        Shell s(0, O_center,
                {130.7093200, 23.8088610, 6.4436083},
                {0.15432897, 0.53532814, 0.44463454});
        s.set_atom_index(0);
        shells.push_back(std::move(s));
    }
    {
        Shell s(0, O_center,
                {5.0331513, 1.1695961, 0.3803890},
                {-0.09996723, 0.39951283, 0.70011547});
        s.set_atom_index(0);
        shells.push_back(std::move(s));
    }
    {
        Shell s(1, O_center,
                {5.0331513, 1.1695961, 0.3803890},
                {0.15591627, 0.60768372, 0.39195739});
        s.set_atom_index(0);
        shells.push_back(std::move(s));
    }
    {
        Shell s(0, H1_center,
                {3.42525091, 0.62391373, 0.16885540},
                {0.15432897, 0.53532814, 0.44463454});
        s.set_atom_index(1);
        shells.push_back(std::move(s));
    }
    {
        Shell s(0, H2_center,
                {3.42525091, 0.62391373, 0.16885540},
                {0.15432897, 0.53532814, 0.44463454});
        s.set_atom_index(2);
        shells.push_back(std::move(s));
    }

    return shells;
}

constexpr Real TIGHT_TOL = 1e-10;

}  // anonymous namespace

// =============================================================================
// Default Construction
// =============================================================================

TEST(IntegralBufferTest, DefaultConstruction) {
    IntegralBuffer buf;

    EXPECT_TRUE(buf.empty());
    EXPECT_EQ(buf.n_integrals(), 0u);
    EXPECT_EQ(buf.n_shell_quartets(), 0u);
    EXPECT_EQ(buf.n_shell_pairs(), 0u);
    EXPECT_EQ(buf.deriv_order(), 0);
}

// =============================================================================
// Reserve and Capacity Tests
// =============================================================================

TEST(IntegralBufferTest, Reserve2E) {
    IntegralBuffer buf;

    const Size n_values = 81;   // e.g., 3*3*3*3 for p-shells
    const Size n_quartets = 1;

    buf.reserve_2e(n_values, n_quartets);

    // After reserve, buffer is not "empty" in capacity sense
    // but no quartets have been appended yet
    EXPECT_EQ(buf.n_shell_quartets(), 0u);
}

TEST(IntegralBufferTest, Reserve1E) {
    IntegralBuffer buf;

    const Size n_values = 9;  // 3*3 for p-shell pair
    const Size n_pairs = 1;

    buf.reserve_1e(n_values, n_pairs);

    EXPECT_EQ(buf.n_shell_pairs(), 0u);
}

// =============================================================================
// Append and Retrieve Tests
// =============================================================================

TEST(IntegralBufferTest, AppendQuartet) {
    IntegralBuffer buf;

    // Simulate a (ss|ss) quartet: 1*1*1*1 = 1 integral
    std::vector<Real> values = {3.14159};
    buf.reserve_2e(1, 1);
    buf.append_quartet(values, 0, 0, 0, 0, 1, 1, 1, 1);

    EXPECT_EQ(buf.n_shell_quartets(), 1u);

    const auto& meta = buf.quartet_meta(0);
    EXPECT_EQ(meta.fi, 0);
    EXPECT_EQ(meta.fj, 0);
    EXPECT_EQ(meta.fk, 0);
    EXPECT_EQ(meta.fl, 0);
    EXPECT_EQ(meta.na, 1);
    EXPECT_EQ(meta.nb, 1);
    EXPECT_EQ(meta.nc, 1);
    EXPECT_EQ(meta.nd, 1);

    auto data = buf.quartet_data(0);
    ASSERT_EQ(data.size(), 1u);
    EXPECT_NEAR(data[0], 3.14159, TIGHT_TOL);
}

TEST(IntegralBufferTest, AppendPair) {
    IntegralBuffer buf;

    // Simulate a (s|s) pair: 1*1 = 1 integral
    std::vector<Real> values = {2.71828};
    buf.reserve_1e(1, 1);
    buf.append_pair(values, 0, 0, 1, 1);

    EXPECT_EQ(buf.n_shell_pairs(), 1u);

    const auto& meta = buf.pair_meta(0);
    EXPECT_EQ(meta.fi, 0);
    EXPECT_EQ(meta.fj, 0);
    EXPECT_EQ(meta.na, 1);
    EXPECT_EQ(meta.nb, 1);

    auto data = buf.pair_data(0);
    ASSERT_EQ(data.size(), 1u);
    EXPECT_NEAR(data[0], 2.71828, TIGHT_TOL);
}

TEST(IntegralBufferTest, AppendMultipleQuartets) {
    IntegralBuffer buf;

    // Append two (ss|ss) quartets
    std::vector<Real> values1 = {1.0};
    std::vector<Real> values2 = {2.0};
    buf.reserve_2e(2, 2);
    buf.append_quartet(values1, 0, 0, 0, 0, 1, 1, 1, 1);
    buf.append_quartet(values2, 1, 1, 1, 1, 1, 1, 1, 1);

    EXPECT_EQ(buf.n_shell_quartets(), 2u);

    auto data0 = buf.quartet_data(0);
    auto data1 = buf.quartet_data(1);
    EXPECT_NEAR(data0[0], 1.0, TIGHT_TOL);
    EXPECT_NEAR(data1[0], 2.0, TIGHT_TOL);
}

// =============================================================================
// Clear Tests
// =============================================================================

TEST(IntegralBufferTest, Clear) {
    IntegralBuffer buf;

    std::vector<Real> values = {42.0};
    buf.reserve_2e(1, 1);
    buf.append_quartet(values, 0, 0, 0, 0, 1, 1, 1, 1);

    ASSERT_EQ(buf.n_shell_quartets(), 1u);
    ASSERT_FALSE(buf.empty());

    buf.clear();

    EXPECT_TRUE(buf.empty());
    EXPECT_EQ(buf.n_shell_quartets(), 0u);
    EXPECT_EQ(buf.n_shell_pairs(), 0u);
    EXPECT_EQ(buf.n_integrals(), 0u);
    EXPECT_EQ(buf.deriv_order(), 0);
}

// =============================================================================
// Data Access Tests
// =============================================================================

TEST(IntegralBufferTest, DataAccess) {
    IntegralBuffer buf;

    // sp shell pair: na=1, nb=3 → 3 integrals
    std::vector<Real> values = {1.1, 2.2, 3.3};
    buf.reserve_1e(3, 1);
    buf.append_pair(values, 0, 1, 1, 3);

    auto data = buf.pair_data(0);
    ASSERT_EQ(data.size(), 3u);
    EXPECT_NEAR(data[0], 1.1, TIGHT_TOL);
    EXPECT_NEAR(data[1], 2.2, TIGHT_TOL);
    EXPECT_NEAR(data[2], 3.3, TIGHT_TOL);
}

// =============================================================================
// Derivative Buffer Sizing Tests
// =============================================================================

TEST(IntegralBufferTest, GradientReserve1E) {
    IntegralBuffer buf;

    // N_DERIV_1E = 6 components for 1e gradients
    const Size n_base_values = 9;  // e.g., 3x3 p-shell pair
    const Size n_pairs = 1;

    buf.reserve_1e_gradient(n_base_values, n_pairs);

    EXPECT_EQ(buf.deriv_order(), 1);
    EXPECT_EQ(buf.n_deriv_components_1e(), N_DERIV_1E);
}

TEST(IntegralBufferTest, GradientReserve2E) {
    IntegralBuffer buf;

    // N_DERIV_2E = 12 components for 2e gradients
    const Size n_base_values = 1;  // (ss|ss) quartet
    const Size n_quartets = 1;

    buf.reserve_2e_gradient(n_base_values, n_quartets);

    EXPECT_EQ(buf.deriv_order(), 1);
    EXPECT_EQ(buf.n_deriv_components_2e(), N_DERIV_2E);
}

TEST(IntegralBufferTest, HessianReserve1E) {
    IntegralBuffer buf;

    const Size n_base_values = 1;  // (ss) pair
    const Size n_pairs = 1;

    buf.reserve_1e_hessian(n_base_values, n_pairs);

    EXPECT_EQ(buf.deriv_order(), 2);

    // Hessian 1e components = N_DERIV_1E * (N_DERIV_1E + 1) / 2 = 21
    int expected_hessian_1e = N_DERIV_1E * (N_DERIV_1E + 1) / 2;
    EXPECT_EQ(buf.n_deriv_components_1e(), expected_hessian_1e);
}

TEST(IntegralBufferTest, HessianReserve2E) {
    IntegralBuffer buf;

    const Size n_base_values = 1;
    const Size n_quartets = 1;

    buf.reserve_2e_hessian(n_base_values, n_quartets);

    EXPECT_EQ(buf.deriv_order(), 2);

    // Hessian 2e components = N_DERIV_2E * (N_DERIV_2E + 1) / 2 = 78
    int expected_hessian_2e = N_DERIV_2E * (N_DERIV_2E + 1) / 2;
    EXPECT_EQ(buf.n_deriv_components_2e(), expected_hessian_2e);
}

// =============================================================================
// Derivative Component Accessor Tests
// =============================================================================

TEST(IntegralBufferTest, DerivativeComponentAccess) {
    IntegralBuffer buf;

    // Set up a gradient buffer with 1 (ss) pair and 1 integral per pair
    // Total storage: n_deriv_1e * n_base = 6 * 1 = 6 values
    const Size n_base = 1;
    const Size n_pairs = 1;
    buf.reserve_1e_gradient(n_base, n_pairs);

    // Create data: 6 gradient components for dS/dAx, dS/dAy, dS/dAz, dS/dBx, dS/dBy, dS/dBz
    std::vector<Real> values(N_DERIV_1E * n_base);
    for (int c = 0; c < N_DERIV_1E; ++c) {
        values[c * n_base] = static_cast<Real>(c + 1);  // 1.0, 2.0, ..., 6.0
    }
    buf.append_pair(values, 0, 0, 1, 1);

    // Verify each derivative component
    for (int c = 0; c < N_DERIV_1E; ++c) {
        auto deriv_data = buf.pair_deriv_data(0, c);
        ASSERT_EQ(deriv_data.size(), n_base);
        EXPECT_NEAR(deriv_data[0], static_cast<Real>(c + 1), TIGHT_TOL)
            << "Derivative component " << c << " mismatch";
    }
}

// =============================================================================
// Angular Momentum Accessor Tests
// =============================================================================

TEST(IntegralBufferTest, AngularMomentum) {
    IntegralBuffer buf;

    buf.set_am(0, 1, 0, 1);

    EXPECT_EQ(buf.La(), 0);
    EXPECT_EQ(buf.Lb(), 1);
    EXPECT_EQ(buf.Lc(), 0);
    EXPECT_EQ(buf.Ld(), 1);
}

// =============================================================================
// Integration Test — compute_batch with Engine
// =============================================================================

TEST(IntegralBufferTest, ComputeBatchIntegration) {
    BasisSet basis(make_sto3g_h2o_shells());
    Engine engine(basis);

    Operator op = Operator::overlap();

    // compute_all_1e returns a vector of IntegralBuffers
    auto buffers = engine.compute_all_1e(op);

    ASSERT_GT(buffers.size(), 0u);

    // Count total shell pairs processed
    Size total_pairs = 0;
    for (const auto& buf : buffers) {
        total_pairs += buf.n_shell_pairs();
    }

    // STO-3G H2O has 5 shells → 15 unique pairs (with symmetry) or 25 total
    EXPECT_GT(total_pairs, 0u);

    // Check that at least one buffer has non-zero integral values
    bool found_nonzero = false;
    for (const auto& buf : buffers) {
        for (Size p = 0; p < buf.n_shell_pairs(); ++p) {
            auto data = buf.pair_data(p);
            for (Size i = 0; i < data.size(); ++i) {
                if (std::abs(data[i]) > 1e-15) {
                    found_nonzero = true;
                    break;
                }
            }
            if (found_nonzero) break;
        }
        if (found_nonzero) break;
    }
    EXPECT_TRUE(found_nonzero) << "Expected at least some non-zero overlap integrals";
}

