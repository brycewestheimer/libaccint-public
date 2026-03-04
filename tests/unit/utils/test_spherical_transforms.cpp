// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2026 Bryce M. Westheimer. All rights reserved.

/// @file test_spherical_transforms.cpp
/// @brief Unit tests for spherical transform wrapper classes (Task 1.3.6)

#include <libaccint/math/spherical_transform.hpp>
#include <libaccint/basis/basis_set.hpp>
#include <libaccint/basis/shell.hpp>

#include <gtest/gtest.h>
#include <cmath>
#include <numeric>
#include <vector>

using namespace libaccint;
using namespace libaccint::math;

// ============================================================================
// Helper: compute reference 2D transform using math-level API
// ============================================================================

static void reference_transform_2d(int La, int Lb,
                                   const std::vector<double>& cart,
                                   std::vector<double>& sph) {
    int n_cart_a = n_cartesian(La);
    int n_cart_b = n_cartesian(Lb);
    int n_sph_a = n_spherical(La);
    int n_sph_b = n_spherical(Lb);
    sph.resize(n_sph_a * n_sph_b);
    std::vector<double> work(work_size_2d(La, Lb));
    transform_2d(La, Lb, cart.data(), sph.data(), work.data());
}

// Helper: compute reference 4D transform using math-level API
static void reference_transform_4d(int La, int Lb, int Lc, int Ld,
                                   const std::vector<double>& cart,
                                   std::vector<double>& sph) {
    int n_sph_a = n_spherical(La);
    int n_sph_b = n_spherical(Lb);
    int n_sph_c = n_spherical(Lc);
    int n_sph_d = n_spherical(Ld);
    sph.resize(n_sph_a * n_sph_b * n_sph_c * n_sph_d);
    std::vector<double> work(work_size_4d(La, Lb, Lc, Ld));
    transform_4d(La, Lb, Lc, Ld, cart.data(), sph.data(), work.data());
}

// Helper: create a simple test input with deterministic but non-trivial values
static std::vector<double> make_test_input(int size) {
    std::vector<double> data(size);
    for (int i = 0; i < size; ++i) {
        data[i] = std::sin(0.7 * (i + 1)) + 0.5 * std::cos(1.3 * i);
    }
    return data;
}

// ============================================================================
// 1e Transform Tests (2-index, using SphericalTransformer)
// ============================================================================

class SphericalTransform1ETest : public ::testing::Test {
protected:
    SphericalTransformer transformer{4};  // max AM = 4 (G-functions)
};

TEST_F(SphericalTransform1ETest, Identity_SS) {
    // AM=0: (s|s) — identity, 1 Cartesian, 1 spherical
    std::vector<double> cart = {3.14};
    std::vector<double> sph(1);
    transformer.transform_1e(0, 0, cart.data(), sph.data());
    EXPECT_NEAR(sph[0], 3.14, 1e-14);
}

TEST_F(SphericalTransform1ETest, SP) {
    // (s|p): 1 x 3 Cartesian -> 1 x 3 spherical
    int La = 0, Lb = 1;
    int nc = n_cartesian(La) * n_cartesian(Lb);
    auto cart = make_test_input(nc);
    std::vector<double> sph, ref;

    sph.resize(n_spherical(La) * n_spherical(Lb));
    transformer.transform_1e(La, Lb, cart.data(), sph.data());
    reference_transform_2d(La, Lb, cart, ref);

    ASSERT_EQ(sph.size(), ref.size());
    for (std::size_t i = 0; i < sph.size(); ++i) {
        EXPECT_NEAR(sph[i], ref[i], 1e-12)
            << "Mismatch at index " << i << " for (s,p) transform";
    }
}

TEST_F(SphericalTransform1ETest, PP) {
    int La = 1, Lb = 1;
    int nc = n_cartesian(La) * n_cartesian(Lb);
    auto cart = make_test_input(nc);
    std::vector<double> sph, ref;

    sph.resize(n_spherical(La) * n_spherical(Lb));
    transformer.transform_1e(La, Lb, cart.data(), sph.data());
    reference_transform_2d(La, Lb, cart, ref);

    ASSERT_EQ(sph.size(), ref.size());
    for (std::size_t i = 0; i < sph.size(); ++i) {
        EXPECT_NEAR(sph[i], ref[i], 1e-12);
    }
}

TEST_F(SphericalTransform1ETest, PD) {
    int La = 1, Lb = 2;
    int nc = n_cartesian(La) * n_cartesian(Lb);
    auto cart = make_test_input(nc);
    std::vector<double> sph, ref;

    sph.resize(n_spherical(La) * n_spherical(Lb));
    transformer.transform_1e(La, Lb, cart.data(), sph.data());
    reference_transform_2d(La, Lb, cart, ref);

    ASSERT_EQ(sph.size(), ref.size());
    for (std::size_t i = 0; i < sph.size(); ++i) {
        EXPECT_NEAR(sph[i], ref[i], 1e-12);
    }
}

TEST_F(SphericalTransform1ETest, DD) {
    int La = 2, Lb = 2;
    int nc = n_cartesian(La) * n_cartesian(Lb);
    auto cart = make_test_input(nc);
    std::vector<double> sph, ref;

    sph.resize(n_spherical(La) * n_spherical(Lb));
    transformer.transform_1e(La, Lb, cart.data(), sph.data());
    reference_transform_2d(La, Lb, cart, ref);

    ASSERT_EQ(sph.size(), ref.size());
    for (std::size_t i = 0; i < sph.size(); ++i) {
        EXPECT_NEAR(sph[i], ref[i], 1e-12)
            << "Mismatch at index " << i << " for (d,d) transform";
    }
}

// ============================================================================
// 2e Transform Tests (4-index, using SphericalTransformer)
// ============================================================================

class SphericalTransform2ETest : public ::testing::Test {
protected:
    SphericalTransformer transformer{4};  // max AM = 4 (G-functions)
};

TEST_F(SphericalTransform2ETest, SSSS) {
    // (s,s|s,s): all AM=0, should be identity
    std::vector<double> cart = {2.718};
    std::vector<double> sph(1);
    transformer.transform_2e(0, 0, 0, 0, cart.data(), sph.data());
    EXPECT_NEAR(sph[0], 2.718, 1e-14);
}

TEST_F(SphericalTransform2ETest, SPSP) {
    int La = 0, Lb = 1, Lc = 0, Ld = 1;
    int nc = n_cartesian(La) * n_cartesian(Lb) *
             n_cartesian(Lc) * n_cartesian(Ld);
    auto cart = make_test_input(nc);
    std::vector<double> sph, ref;

    int ns = n_spherical(La) * n_spherical(Lb) *
             n_spherical(Lc) * n_spherical(Ld);
    sph.resize(ns);
    transformer.transform_2e(La, Lb, Lc, Ld, cart.data(), sph.data());
    reference_transform_4d(La, Lb, Lc, Ld, cart, ref);

    ASSERT_EQ(sph.size(), ref.size());
    for (std::size_t i = 0; i < sph.size(); ++i) {
        EXPECT_NEAR(sph[i], ref[i], 1e-12)
            << "Mismatch at index " << i << " for (s,p|s,p) transform";
    }
}

TEST_F(SphericalTransform2ETest, PPPP) {
    int La = 1, Lb = 1, Lc = 1, Ld = 1;
    int nc = n_cartesian(La) * n_cartesian(Lb) *
             n_cartesian(Lc) * n_cartesian(Ld);
    auto cart = make_test_input(nc);
    std::vector<double> sph, ref;

    int ns = n_spherical(La) * n_spherical(Lb) *
             n_spherical(Lc) * n_spherical(Ld);
    sph.resize(ns);
    transformer.transform_2e(La, Lb, Lc, Ld, cart.data(), sph.data());
    reference_transform_4d(La, Lb, Lc, Ld, cart, ref);

    ASSERT_EQ(sph.size(), ref.size());
    for (std::size_t i = 0; i < sph.size(); ++i) {
        EXPECT_NEAR(sph[i], ref[i], 1e-12)
            << "Mismatch at index " << i << " for (p,p|p,p) transform";
    }
}

// ============================================================================
// Repeated Calls (buffer reuse correctness)
// ============================================================================

TEST_F(SphericalTransform2ETest, RepeatedCalls) {
    int La = 1, Lb = 1, Lc = 0, Ld = 1;
    int nc = n_cartesian(La) * n_cartesian(Lb) *
             n_cartesian(Lc) * n_cartesian(Ld);
    int ns = n_spherical(La) * n_spherical(Lb) *
             n_spherical(Lc) * n_spherical(Ld);

    // Call transform multiple times with different input data
    for (int trial = 0; trial < 5; ++trial) {
        // Create unique input for each trial
        std::vector<double> cart(nc);
        for (int i = 0; i < nc; ++i) {
            cart[i] = std::sin(0.7 * (i + 1) + trial * 1.1);
        }

        std::vector<double> sph(ns), ref;
        transformer.transform_2e(La, Lb, Lc, Ld, cart.data(), sph.data());
        reference_transform_4d(La, Lb, Lc, Ld, cart, ref);

        ASSERT_EQ(sph.size(), ref.size());
        for (std::size_t i = 0; i < sph.size(); ++i) {
            EXPECT_NEAR(sph[i], ref[i], 1e-12)
                << "Mismatch at trial " << trial << " index " << i;
        }
    }
}

TEST_F(SphericalTransform1ETest, RepeatedCalls) {
    int La = 2, Lb = 1;
    int nc = n_cartesian(La) * n_cartesian(Lb);
    int ns = n_spherical(La) * n_spherical(Lb);

    for (int trial = 0; trial < 5; ++trial) {
        std::vector<double> cart(nc);
        for (int i = 0; i < nc; ++i) {
            cart[i] = std::cos(0.3 * (i + 1) + trial * 2.0);
        }

        std::vector<double> sph(ns), ref;
        transformer.transform_1e(La, Lb, cart.data(), sph.data());
        reference_transform_2d(La, Lb, cart, ref);

        ASSERT_EQ(sph.size(), ref.size());
        for (std::size_t i = 0; i < sph.size(); ++i) {
            EXPECT_NEAR(sph[i], ref[i], 1e-12)
                << "1e Mismatch at trial " << trial << " index " << i;
        }
    }
}
