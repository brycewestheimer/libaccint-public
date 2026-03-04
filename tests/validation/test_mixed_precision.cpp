// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2026 Bryce M. Westheimer. All rights reserved.

/// @file test_mixed_precision.cpp
/// @brief Mixed precision accumulation tests (Task 24.3.3)
///
/// Validates the compute-float32-accumulate-float64 strategy where
/// individual integral values are computed in single precision but
/// accumulated into double-precision Fock matrix elements.

#include <libaccint/basis/basis_set.hpp>
#include <libaccint/buffers/two_electron_buffer.hpp>
#include <libaccint/consumers/fock_builder.hpp>
#include <libaccint/consumers/mixed_precision_fock_builder.hpp>
#include <libaccint/core/precision.hpp>
#include <libaccint/core/types.hpp>
#include <libaccint/data/builtin_basis.hpp>
#include <libaccint/engine/precision_dispatch.hpp>
#include <libaccint/kernels/eri_kernel.hpp>
#include <libaccint/operators/operator_types.hpp>

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <numeric>
#include <vector>

namespace libaccint::test {
namespace {

using namespace libaccint::consumers;
using namespace libaccint::data;

// ============================================================================
// MixedPrecisionFockBuilder Unit Tests
// ============================================================================

TEST(MixedPrecisionFockBuilder, Construction) {
    MixedPrecisionFockBuilder builder(10);
    EXPECT_EQ(builder.nbf(), 10u);
    EXPECT_EQ(builder.mode(), MixedPrecisionMode::Compute32Accumulate64);
    EXPECT_EQ(builder.n_float32_accumulations(), 0u);
    EXPECT_EQ(builder.n_float64_accumulations(), 0u);
}

TEST(MixedPrecisionFockBuilder, Reset) {
    MixedPrecisionFockBuilder builder(5);

    // Set density
    std::vector<Real> D(25, 0.1);
    builder.set_density(D.data(), 5);

    // Accumulate something
    TwoElectronBuffer<0, float> buf(1, 1, 1, 1);
    buf.clear();
    buf(0, 0, 0, 0) = 1.0f;
    builder.accumulate(buf, 0, 0, 0, 0, 1, 1, 1, 1);

    EXPECT_EQ(builder.n_float32_accumulations(), 1u);

    // Reset
    builder.reset();
    EXPECT_EQ(builder.n_float32_accumulations(), 0u);
    EXPECT_EQ(builder.n_float64_accumulations(), 0u);

    // All matrix elements should be zero
    auto J = builder.get_coulomb_matrix();
    for (Size i = 0; i < 25; ++i) {
        EXPECT_DOUBLE_EQ(J[i], 0.0);
    }
}

TEST(MixedPrecisionFockBuilder, Float32Accumulation) {
    constexpr Size nbf = 2;
    MixedPrecisionFockBuilder builder(nbf);

    std::vector<Real> D = {1.0, 0.0, 0.0, 1.0};
    builder.set_density(D.data(), nbf);

    // Create a simple (ss|ss) float buffer
    TwoElectronBuffer<0, float> buf(1, 1, 1, 1);
    buf.clear();
    buf(0, 0, 0, 0) = 0.5f;

    builder.accumulate(buf, 0, 0, 0, 0, 1, 1, 1, 1);

    auto J = builder.get_coulomb_matrix();
    auto K = builder.get_exchange_matrix();

    // J(0,0) += (00|00) * D(0,0) = 0.5 * 1.0 = 0.5
    EXPECT_NEAR(J[0], 0.5, 1e-6);

    // K(0,0) += (00|00) * D(0,0) = 0.5 * 1.0 = 0.5
    EXPECT_NEAR(K[0], 0.5, 1e-6);

    EXPECT_EQ(builder.n_float32_accumulations(), 1u);
}

TEST(MixedPrecisionFockBuilder, Float64Accumulation) {
    constexpr Size nbf = 2;
    MixedPrecisionFockBuilder builder(nbf);

    std::vector<Real> D = {1.0, 0.0, 0.0, 1.0};
    builder.set_density(D.data(), nbf);

    TwoElectronBuffer<0, double> buf(1, 1, 1, 1);
    buf.clear();
    buf(0, 0, 0, 0) = 0.5;

    builder.accumulate(buf, 0, 0, 0, 0, 1, 1, 1, 1);

    auto J = builder.get_coulomb_matrix();
    EXPECT_NEAR(J[0], 0.5, 1e-14);

    EXPECT_EQ(builder.n_float64_accumulations(), 1u);
}

TEST(MixedPrecisionFockBuilder, FockMatrixConstruction) {
    constexpr Size nbf = 2;
    MixedPrecisionFockBuilder builder(nbf);

    std::vector<Real> D = {1.0, 0.0, 0.0, 1.0};
    builder.set_density(D.data(), nbf);

    TwoElectronBuffer<0, float> buf(1, 1, 1, 1);
    buf.clear();
    buf(0, 0, 0, 0) = 0.5f;
    builder.accumulate(buf, 0, 0, 0, 0, 1, 1, 1, 1);

    // H_core = identity
    std::vector<Real> H_core = {1.0, 0.0, 0.0, 1.0};

    // F = H_core + J - K
    auto F = builder.get_fock_matrix(std::span<const Real>(H_core), 1.0);

    EXPECT_EQ(F.size(), 4u);
    // F(0,0) = H(0,0) + J(0,0) - K(0,0) = 1.0 + 0.5 - 0.5 = 1.0
    EXPECT_NEAR(F[0], 1.0, 1e-6);
}

// ============================================================================
// Mixed vs Pure Double Comparison
// ============================================================================

class MixedVsPureComparison : public ::testing::Test {
protected:
    void SetUp() override {
        atoms_ = {
            {1, {0.0, 0.0, 0.0}},
            {1, {0.0, 0.0, 1.4}}
        };
        try {
            basis_ = create_sto3g(atoms_);
            nbf_ = basis_.n_basis_functions();
        } catch (const std::exception& e) {
            skip_ = true;
            skip_reason_ = e.what();
        }
    }

    std::vector<Atom> atoms_;
    BasisSet basis_;
    Size nbf_{0};
    bool skip_{false};
    std::string skip_reason_;
};

TEST_F(MixedVsPureComparison, JMatrixAgreement) {
    if (skip_) GTEST_SKIP() << skip_reason_;

    auto shells = basis_.shells();
    Size nshells = shells.size();

    // Create density
    std::vector<Real> D(nbf_ * nbf_, 0.0);
    for (Size i = 0; i < nbf_; ++i) {
        D[i * nbf_ + i] = 1.0 / static_cast<Real>(nbf_);
    }

    FockBuilder pure_builder(nbf_);
    pure_builder.set_density(D.data(), nbf_);

    MixedPrecisionFockBuilder mixed_builder(nbf_);
    mixed_builder.set_density(D.data(), nbf_);

    // Compute all shell quartets
    for (Size i = 0; i < nshells; ++i) {
        for (Size j = 0; j < nshells; ++j) {
            for (Size k = 0; k < nshells; ++k) {
                for (Size l = 0; l < nshells; ++l) {
                    TwoElectronBuffer<0> d_buf;
                    kernels::compute_eri(shells[i], shells[j], shells[k], shells[l], d_buf);

                    auto fi = static_cast<Index>(shells[i].function_index());
                    auto fj = static_cast<Index>(shells[j].function_index());
                    auto fk = static_cast<Index>(shells[k].function_index());
                    auto fl = static_cast<Index>(shells[l].function_index());
                    int ni = n_cartesian(shells[i].angular_momentum());
                    int nj = n_cartesian(shells[j].angular_momentum());
                    int nk = n_cartesian(shells[k].angular_momentum());
                    int nl = n_cartesian(shells[l].angular_momentum());

                    // Pure double
                    pure_builder.accumulate(d_buf, fi, fj, fk, fl, ni, nj, nk, nl);

                    // Mixed: float32 compute, float64 accumulate
                    TwoElectronBuffer<0, float> f_buf;
                    f_buf.copy_from(d_buf);
                    mixed_builder.accumulate(f_buf, fi, fj, fk, fl, ni, nj, nk, nl);
                }
            }
        }
    }

    // Compare J matrices
    auto J_pure = pure_builder.get_coulomb_matrix();
    auto J_mixed = mixed_builder.get_coulomb_matrix();

    double max_abs_diff = 0.0;
    double max_rel_diff = 0.0;
    for (Size i = 0; i < nbf_ * nbf_; ++i) {
        double abs_diff = std::abs(J_pure[i] - J_mixed[i]);
        max_abs_diff = std::max(max_abs_diff, abs_diff);
        if (std::abs(J_pure[i]) > 1e-12) {
            double rel_diff = abs_diff / std::abs(J_pure[i]);
            max_rel_diff = std::max(max_rel_diff, rel_diff);
        }
    }

    // Mixed precision J should agree with pure double within float32 truncation
    EXPECT_LT(max_rel_diff, 1e-5)
        << "J matrix max relative difference: " << max_rel_diff
        << ", max absolute difference: " << max_abs_diff;
}

TEST_F(MixedVsPureComparison, KMatrixAgreement) {
    if (skip_) GTEST_SKIP() << skip_reason_;

    auto shells = basis_.shells();
    Size nshells = shells.size();

    std::vector<Real> D(nbf_ * nbf_, 0.0);
    for (Size i = 0; i < nbf_; ++i) {
        D[i * nbf_ + i] = 1.0 / static_cast<Real>(nbf_);
    }

    FockBuilder pure_builder(nbf_);
    pure_builder.set_density(D.data(), nbf_);

    MixedPrecisionFockBuilder mixed_builder(nbf_);
    mixed_builder.set_density(D.data(), nbf_);

    for (Size i = 0; i < nshells; ++i) {
        for (Size j = 0; j < nshells; ++j) {
            for (Size k = 0; k < nshells; ++k) {
                for (Size l = 0; l < nshells; ++l) {
                    TwoElectronBuffer<0> d_buf;
                    kernels::compute_eri(shells[i], shells[j], shells[k], shells[l], d_buf);

                    auto fi = static_cast<Index>(shells[i].function_index());
                    auto fj = static_cast<Index>(shells[j].function_index());
                    auto fk = static_cast<Index>(shells[k].function_index());
                    auto fl = static_cast<Index>(shells[l].function_index());
                    int ni = n_cartesian(shells[i].angular_momentum());
                    int nj = n_cartesian(shells[j].angular_momentum());
                    int nk = n_cartesian(shells[k].angular_momentum());
                    int nl = n_cartesian(shells[l].angular_momentum());

                    pure_builder.accumulate(d_buf, fi, fj, fk, fl, ni, nj, nk, nl);

                    TwoElectronBuffer<0, float> f_buf;
                    f_buf.copy_from(d_buf);
                    mixed_builder.accumulate(f_buf, fi, fj, fk, fl, ni, nj, nk, nl);
                }
            }
        }
    }

    auto K_pure = pure_builder.get_exchange_matrix();
    auto K_mixed = mixed_builder.get_exchange_matrix();

    double max_rel_diff = 0.0;
    for (Size i = 0; i < nbf_ * nbf_; ++i) {
        if (std::abs(K_pure[i]) > 1e-12) {
            double rel_diff = std::abs(K_pure[i] - K_mixed[i]) / std::abs(K_pure[i]);
            max_rel_diff = std::max(max_rel_diff, rel_diff);
        }
    }

    EXPECT_LT(max_rel_diff, 1e-5)
        << "K matrix max relative difference: " << max_rel_diff;
}

TEST_F(MixedVsPureComparison, AccumulationCountTracking) {
    if (skip_) GTEST_SKIP() << skip_reason_;

    MixedPrecisionFockBuilder builder(nbf_);
    std::vector<Real> D(nbf_ * nbf_, 0.1);
    builder.set_density(D.data(), nbf_);

    auto shells = basis_.shells();

    // Accumulate one float32 quartet
    TwoElectronBuffer<0, float> f_buf(1, 1, 1, 1);
    f_buf.clear();
    f_buf(0, 0, 0, 0) = 1.0f;
    auto fi = static_cast<Index>(shells[0].function_index());
    builder.accumulate(f_buf, fi, fi, fi, fi, 1, 1, 1, 1);

    // Accumulate one float64 quartet
    TwoElectronBuffer<0, double> d_buf(1, 1, 1, 1);
    d_buf.clear();
    d_buf(0, 0, 0, 0) = 1.0;
    builder.accumulate(d_buf, fi, fi, fi, fi, 1, 1, 1, 1);

    EXPECT_EQ(builder.n_float32_accumulations(), 1u);
    EXPECT_EQ(builder.n_float64_accumulations(), 1u);
}

// ============================================================================
// Precision Configuration Tests
// ============================================================================

TEST(PrecisionConfig, PureDoubleDefaults) {
    engine::PrecisionConfig cfg;
    EXPECT_EQ(cfg.compute_precision, Precision::Float64);
    EXPECT_EQ(cfg.accumulate_precision, Precision::Float64);
    EXPECT_EQ(cfg.mode, MixedPrecisionMode::Pure64);
    EXPECT_FALSE(cfg.adaptive_am);
}

TEST(PrecisionConfig, PureFloat) {
    auto cfg = engine::PrecisionConfig::pure_float();
    EXPECT_EQ(cfg.compute_precision, Precision::Float32);
    EXPECT_EQ(cfg.accumulate_precision, Precision::Float32);
    EXPECT_EQ(cfg.mode, MixedPrecisionMode::Pure32);
}

TEST(PrecisionConfig, Mixed) {
    auto cfg = engine::PrecisionConfig::mixed();
    EXPECT_EQ(cfg.compute_precision, Precision::Float32);
    EXPECT_EQ(cfg.accumulate_precision, Precision::Float64);
    EXPECT_EQ(cfg.mode, MixedPrecisionMode::Compute32Accumulate64);
}

TEST(PrecisionConfig, Adaptive) {
    auto cfg = engine::PrecisionConfig::adaptive(2);
    EXPECT_EQ(cfg.mode, MixedPrecisionMode::Adaptive);
    EXPECT_TRUE(cfg.adaptive_am);
    EXPECT_EQ(cfg.am_threshold_for_double, 2);

    // Low AM should use float32
    EXPECT_EQ(engine::select_precision_1e(cfg, 0, 0), Precision::Float32);
    EXPECT_EQ(engine::select_precision_1e(cfg, 1, 1), Precision::Float32);

    // High AM should use float64
    EXPECT_EQ(engine::select_precision_1e(cfg, 2, 0), Precision::Float64);
    EXPECT_EQ(engine::select_precision_1e(cfg, 0, 2), Precision::Float64);
}

TEST(PrecisionConfig, NonAdaptiveIgnoresAM) {
    auto cfg = engine::PrecisionConfig::pure_float();
    // Even for high AM, non-adaptive mode uses configured precision
    EXPECT_EQ(engine::select_precision_1e(cfg, 5, 5), Precision::Float32);
    EXPECT_EQ(engine::select_precision_2e(cfg, 3, 3, 3, 3), Precision::Float32);
}

// ============================================================================
// Accumulation Precision Verification
// ============================================================================

TEST(MixedPrecisionAccumulation, ManySmallValuesFromFloat) {
    // Verify that accumulating many small float32 values in float64
    // prevents catastrophic cancellation

    constexpr Size nbf = 2;
    MixedPrecisionFockBuilder builder(nbf);
    std::vector<Real> D = {1.0, 0.0, 0.0, 1.0};
    builder.set_density(D.data(), nbf);

    // Accumulate many small contributions
    constexpr int N = 1000;
    float small_integral = 1e-5f;

    for (int i = 0; i < N; ++i) {
        TwoElectronBuffer<0, float> buf(1, 1, 1, 1);
        buf.clear();
        buf(0, 0, 0, 0) = small_integral;
        builder.accumulate(buf, 0, 0, 0, 0, 1, 1, 1, 1);
    }

    auto J = builder.get_coulomb_matrix();
    // Expected: N * small_integral * D(0,0) = 1000 * 1e-5 * 1.0 = 0.01
    double expected = N * static_cast<double>(small_integral) * D[0];

    // Float64 accumulation should get this right
    EXPECT_NEAR(J[0], expected, 1e-10)
        << "Float64 accumulation of float32 values should be precise";
}

}  // namespace
}  // namespace libaccint::test
