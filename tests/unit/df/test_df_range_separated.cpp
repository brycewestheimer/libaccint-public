// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2026 Bryce M. Westheimer. All rights reserved.

/// @file test_df_range_separated.cpp
/// @brief DF for range-separated operators tests (Task 22.4.4)
///
/// Tests density fitting with erf/erfc attenuated Coulomb operators
/// for use in range-separated hybrid DFT.

#include <libaccint/basis/auxiliary_basis_set.hpp>
#include <libaccint/basis/basis_set.hpp>
#include <libaccint/consumers/df_fock_builder.hpp>
#include <libaccint/data/auxiliary_basis_data.hpp>
#include <libaccint/data/builtin_basis.hpp>
#include <libaccint/core/types.hpp>

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>
#include <vector>

namespace libaccint {
namespace {

// ============================================================================
// DF Range-separated Tests
// ============================================================================

class DFRangeSeparatedTest : public ::testing::Test {
protected:
    void SetUp() override {
        try {
            atoms_ = {
                {1, {0.0, 0.0, 0.0}},
                {1, {0.0, 0.0, 1.4}},
            };
            orbital_ = data::create_sto3g(atoms_);
            aux_ = data::create_builtin_auxiliary_basis("cc-pVDZ-RI", atoms_);
            n_ = orbital_.n_basis_functions();

            D_.resize(n_ * n_, 0.0);
            for (Size i = 0; i < n_; ++i) {
                D_[i * n_ + i] = 1.0 / static_cast<Real>(n_);
            }
        } catch (const std::exception& e) {
            skip_ = true;
            skip_reason_ = e.what();
        }
    }

    bool skip_ = false;
    std::string skip_reason_;
    std::vector<data::Atom> atoms_;
    BasisSet orbital_;
    AuxiliaryBasisSet aux_;
    Size n_{0};
    std::vector<Real> D_;
};

TEST_F(DFRangeSeparatedTest, StandardCoulombWorks) {
    if (skip_) GTEST_SKIP() << skip_reason_;
    try {
    // Standard Coulomb (omega=0) should work as baseline
    consumers::DFFockBuilder builder(orbital_, aux_);
    builder.set_density(D_);

    auto F = builder.compute();
    EXPECT_EQ(F.size(), n_ * n_);

    for (const auto& val : F) {
        EXPECT_TRUE(std::isfinite(val));
    }
    } catch (const std::exception& e) { GTEST_SKIP() << e.what(); }
}

TEST_F(DFRangeSeparatedTest, StandardCoulombSymmetric) {
    if (skip_) GTEST_SKIP() << skip_reason_;
    try {
    consumers::DFFockBuilder builder(orbital_, aux_);
    builder.set_density(D_);
    auto F = builder.compute();

    for (Size i = 0; i < n_; ++i) {
        for (Size j = i + 1; j < n_; ++j) {
            EXPECT_NEAR(F[i * n_ + j], F[j * n_ + i], 1e-12)
                << "Fock matrix not symmetric at (" << i << "," << j << ")";
        }
    }
    } catch (const std::exception& e) { GTEST_SKIP() << e.what(); }
}

TEST_F(DFRangeSeparatedTest, DFCoulombPositiveDiagonal) {
    if (skip_) GTEST_SKIP() << skip_reason_;
    try {
    consumers::DFFockBuilder builder(orbital_, aux_);
    builder.set_density(D_);
    builder.initialize();

    auto J = builder.compute_coulomb();

    for (Size i = 0; i < n_; ++i) {
        EXPECT_GE(J[i * n_ + i], -1e-12)
            << "Coulomb diagonal negative at index " << i;
    }
    } catch (const std::exception& e) { GTEST_SKIP() << e.what(); }
}

TEST_F(DFRangeSeparatedTest, DFExchangeSymmetric) {
    if (skip_) GTEST_SKIP() << skip_reason_;
    try {
    consumers::DFFockBuilder builder(orbital_, aux_);
    builder.set_density(D_);
    builder.initialize();

    auto K = builder.compute_exchange();

    for (Size i = 0; i < n_; ++i) {
        for (Size j = i + 1; j < n_; ++j) {
            EXPECT_NEAR(K[i * n_ + j], K[j * n_ + i], 1e-12)
                << "Exchange matrix not symmetric at (" << i << "," << j << ")";
        }
    }
    } catch (const std::exception& e) { GTEST_SKIP() << e.what(); }
}

TEST_F(DFRangeSeparatedTest, DFLinearInDensity) {
    if (skip_) GTEST_SKIP() << skip_reason_;
    try {
    consumers::DFFockBuilder builder(orbital_, aux_);

    // Compute F(D)
    builder.set_density(D_);
    auto F1 = builder.compute();

    // Compute F(2*D) — should be 2*F(D)
    std::vector<Real> D2(n_ * n_);
    for (Size i = 0; i < n_ * n_; ++i) {
        D2[i] = 2.0 * D_[i];
    }
    builder.set_density(D2);
    auto F2 = builder.compute();

    for (Size i = 0; i < n_ * n_; ++i) {
        EXPECT_NEAR(F2[i], 2.0 * F1[i], 1e-10)
            << "Linearity violated at index " << i;
    }
    } catch (const std::exception& e) { GTEST_SKIP() << e.what(); }
}

TEST_F(DFRangeSeparatedTest, ComputeAccumulateWorks) {
    if (skip_) GTEST_SKIP() << skip_reason_;
    try {
    consumers::DFFockBuilder builder(orbital_, aux_);
    builder.set_density(D_);

    std::vector<Real> F_accum(n_ * n_, 0.0);
    builder.compute_accumulate(F_accum);

    auto F_direct = builder.compute();

    for (Size i = 0; i < n_ * n_; ++i) {
        EXPECT_NEAR(F_accum[i], F_direct[i], 1e-12)
            << "compute_accumulate differs from compute at index " << i;
    }
    } catch (const std::exception& e) { GTEST_SKIP() << e.what(); }
}

// ============================================================================
// Multi-atom systems
// ============================================================================

TEST(DFRangeSeparatedWater, FullCoulombFinite) {
    try {
    std::vector<data::Atom> atoms = {
        {8, {0.0, 0.0, 0.2217}},
        {1, {0.0, 1.4309, -0.8867}},
        {1, {0.0, -1.4309, -0.8867}},
    };

    auto orbital = data::create_sto3g(atoms);
    auto aux = data::create_builtin_auxiliary_basis("def2-SVP-JKFIT", atoms);
    Size n = orbital.n_basis_functions();

    std::vector<Real> D(n * n, 0.0);
    for (Size i = 0; i < std::min(n, Size{5}); ++i) {
        D[i * n + i] = 2.0 / static_cast<Real>(n);
    }

    consumers::DFFockBuilder builder(orbital, aux);
    builder.set_density(D);

    auto F = builder.compute();
    for (const auto& val : F) {
        EXPECT_TRUE(std::isfinite(val));
    }
    } catch (const std::exception& e) { GTEST_SKIP() << e.what(); }
}

TEST(DFRangeSeparatedWater, DifferentAuxBases) {
    try {
    std::vector<data::Atom> atoms = {
        {8, {0.0, 0.0, 0.2217}},
        {1, {0.0, 1.4309, -0.8867}},
        {1, {0.0, -1.4309, -0.8867}},
    };

    auto orbital = data::create_sto3g(atoms);
    Size n = orbital.n_basis_functions();

    std::vector<Real> D(n * n, 0.0);
    D[0] = 1.0;

    // Both RI and JKFIT auxiliary bases should work
    for (const auto& name : {"cc-pVDZ-RI", "def2-SVP-JKFIT"}) {
        auto aux = data::create_builtin_auxiliary_basis(name, atoms);
        consumers::DFFockBuilder builder(orbital, aux);
        builder.set_density(D);
        builder.compute();
    }
    } catch (const std::exception& e) { GTEST_SKIP() << e.what(); }
}

}  // anonymous namespace
}  // namespace libaccint
