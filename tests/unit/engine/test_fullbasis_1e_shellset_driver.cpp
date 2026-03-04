// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2026 Bryce M. Westheimer. All rights reserved.

/// @file test_fullbasis_1e_shellset_driver.cpp
/// @brief Regression tests verifying Engine full-basis 1e computation produces
///        correct results via ShellSetPair work-unit drivers.

#include <libaccint/engine/engine.hpp>
#include <libaccint/basis/basis_set.hpp>
#include <libaccint/basis/shell.hpp>
#include <libaccint/basis/shell_set_pair.hpp>
#include <libaccint/buffers/one_electron_buffer.hpp>
#include <libaccint/operators/one_electron_operator.hpp>
#include <libaccint/operators/operator.hpp>
#include <libaccint/core/types.hpp>

#include <gtest/gtest.h>
#include <cmath>
#include <vector>

using namespace libaccint;

namespace {

// =============================================================================
// Test Data: H2O / STO-3G (5 shells, 7 basis functions)
// =============================================================================

constexpr Point3D O_center{0.0, 0.0, 0.0};
constexpr Point3D H1_center{0.0, 1.43233673, -1.10866041};
constexpr Point3D H2_center{0.0, -1.43233673, -1.10866041};

std::vector<Shell> make_sto3g_h2o_shells() {
    std::vector<Shell> shells;
    shells.reserve(5);

    // O 1s
    {
        Shell s(0, O_center,
                {130.7093200, 23.8088610, 6.4436083},
                {0.15432897, 0.53532814, 0.44463454});
        s.set_atom_index(0);
        shells.push_back(std::move(s));
    }
    // O 2s
    {
        Shell s(0, O_center,
                {5.0331513, 1.1695961, 0.3803890},
                {-0.09996723, 0.39951283, 0.70011547});
        s.set_atom_index(0);
        shells.push_back(std::move(s));
    }
    // O 2p
    {
        Shell s(1, O_center,
                {5.0331513, 1.1695961, 0.3803890},
                {0.15591627, 0.60768372, 0.39195739});
        s.set_atom_index(0);
        shells.push_back(std::move(s));
    }
    // H1 1s
    {
        Shell s(0, H1_center,
                {3.42525091, 0.62391373, 0.16885540},
                {0.15432897, 0.53532814, 0.44463454});
        s.set_atom_index(1);
        shells.push_back(std::move(s));
    }
    // H2 1s
    {
        Shell s(0, H2_center,
                {3.42525091, 0.62391373, 0.16885540},
                {0.15432897, 0.53532814, 0.44463454});
        s.set_atom_index(2);
        shells.push_back(std::move(s));
    }

    return shells;
}

PointChargeParams make_h2o_charges() {
    PointChargeParams charges;
    charges.x = {0.0, 0.0, 0.0};
    charges.y = {0.0, 1.43233673, -1.43233673};
    charges.z = {0.0, -1.10866041, -1.10866041};
    charges.charge = {8.0, 1.0, 1.0};
    return charges;
}

// Tolerances
constexpr Real TIGHT_1E_TOL = 1e-14;
constexpr Real DIAG_TOL     = 1e-10;

}  // anonymous namespace

// =============================================================================
// Test Fixture
// =============================================================================

class FullBasis1eShellSetDriver : public ::testing::Test {
protected:
    void SetUp() override {
        basis_ = std::make_unique<BasisSet>(make_sto3g_h2o_shells());
        engine_ = std::make_unique<Engine>(*basis_);
        nbf_ = basis_->n_basis_functions();
        ASSERT_EQ(nbf_, 7u);
    }

    /// Compute a 1e matrix via manual shell-pair loop (reference).
    std::vector<Real> compute_1e_reference(const Operator& op) {
        std::vector<Real> M(nbf_ * nbf_, 0.0);
        OneElectronBuffer<0> buffer;

        for (Size i = 0; i < basis_->n_shells(); ++i) {
            const auto& si = basis_->shell(i);
            Index fi = si.function_index();
            int ni = si.n_functions();

            for (Size j = i; j < basis_->n_shells(); ++j) {
                const auto& sj = basis_->shell(j);
                Index fj = sj.function_index();
                int nj = sj.n_functions();

                engine_->compute(op, si, sj, buffer);

                for (int a = 0; a < ni; ++a) {
                    for (int b = 0; b < nj; ++b) {
                        M[(fi + a) * nbf_ + (fj + b)] = buffer(a, b);
                        M[(fj + b) * nbf_ + (fi + a)] = buffer(a, b);
                    }
                }
            }
        }
        return M;
    }

    std::unique_ptr<BasisSet> basis_;
    std::unique_ptr<Engine> engine_;
    Size nbf_{0};
};

// =============================================================================
// Tests
// =============================================================================

TEST_F(FullBasis1eShellSetDriver, OverlapViaShellSetPairs_MatchesDirectCompute) {
    // Full-basis compute (internally uses ShellSetPair driver)
    std::vector<Real> S;
    engine_->compute(OneElectronOperator(Operator::overlap()), S);

    // Reference: manual shell-pair loop
    auto S_ref = compute_1e_reference(Operator::overlap());

    ASSERT_EQ(S.size(), S_ref.size());
    for (Size idx = 0; idx < S.size(); ++idx) {
        EXPECT_NEAR(S[idx], S_ref[idx], TIGHT_1E_TOL)
            << "Mismatch at index " << idx
            << " (row=" << idx / nbf_ << ", col=" << idx % nbf_ << ")";
    }
}

TEST_F(FullBasis1eShellSetDriver, KineticViaShellSetPairs_MatchesDirectCompute) {
    std::vector<Real> T;
    engine_->compute(OneElectronOperator(Operator::kinetic()), T);

    auto T_ref = compute_1e_reference(Operator::kinetic());

    ASSERT_EQ(T.size(), T_ref.size());
    for (Size idx = 0; idx < T.size(); ++idx) {
        EXPECT_NEAR(T[idx], T_ref[idx], TIGHT_1E_TOL)
            << "Mismatch at index " << idx
            << " (row=" << idx / nbf_ << ", col=" << idx % nbf_ << ")";
    }
}

TEST_F(FullBasis1eShellSetDriver, NuclearViaShellSetPairs_MatchesDirectCompute) {
    auto charges = make_h2o_charges();

    std::vector<Real> V;
    engine_->compute(OneElectronOperator(Operator::nuclear(charges)), V);

    auto V_ref = compute_1e_reference(Operator::nuclear(charges));

    ASSERT_EQ(V.size(), V_ref.size());
    for (Size idx = 0; idx < V.size(); ++idx) {
        EXPECT_NEAR(V[idx], V_ref[idx], TIGHT_1E_TOL)
            << "Mismatch at index " << idx
            << " (row=" << idx / nbf_ << ", col=" << idx % nbf_ << ")";
    }
}

TEST_F(FullBasis1eShellSetDriver, OverlapMatrixSymmetry) {
    std::vector<Real> S;
    engine_->compute(OneElectronOperator(Operator::overlap()), S);

    ASSERT_EQ(S.size(), nbf_ * nbf_);
    for (Size i = 0; i < nbf_; ++i) {
        for (Size j = i + 1; j < nbf_; ++j) {
            EXPECT_NEAR(S[i * nbf_ + j], S[j * nbf_ + i], TIGHT_1E_TOL)
                << "Symmetry violated at S[" << i << "," << j << "]";
        }
    }
}

TEST_F(FullBasis1eShellSetDriver, OverlapDiagonalUnity) {
    std::vector<Real> S;
    engine_->compute(OneElectronOperator(Operator::overlap()), S);

    ASSERT_EQ(S.size(), nbf_ * nbf_);
    for (Size i = 0; i < nbf_; ++i) {
        EXPECT_NEAR(S[i * nbf_ + i], 1.0, DIAG_TOL)
            << "Diagonal element S[" << i << "," << i << "] should be 1.0";
    }
}

TEST_F(FullBasis1eShellSetDriver, ExplicitShellSetPairIteration_MatchesFullBasis) {
    // Compute S via explicit ShellSetPair iteration
    std::vector<Real> S(nbf_ * nbf_, 0.0);
    for (const auto& pair : basis_->shell_set_pairs()) {
        engine_->compute(Operator::overlap(), pair, S);
    }

    // Full-basis compute
    std::vector<Real> S2;
    engine_->compute(OneElectronOperator(Operator::overlap()), S2);

    ASSERT_EQ(S.size(), S2.size());
    for (Size idx = 0; idx < S.size(); ++idx) {
        EXPECT_NEAR(S[idx], S2[idx], TIGHT_1E_TOL)
            << "Mismatch at index " << idx
            << " (row=" << idx / nbf_ << ", col=" << idx % nbf_ << ")";
    }
}

TEST_F(FullBasis1eShellSetDriver, SpanOverloadMatchesExplicitShellSetPairIteration) {
    const auto& all_pairs = basis_->shell_set_pairs();
    ASSERT_FALSE(all_pairs.empty());

    std::vector<Real> batched;
    engine_->compute_1e(
        OneElectronOperator(Operator::overlap()),
        std::span<const ShellSetPair>(all_pairs.data(), all_pairs.size()),
        batched);

    std::vector<Real> explicit_pairs(nbf_ * nbf_, 0.0);
    for (const auto& pair : all_pairs) {
        engine_->compute(Operator::overlap(), pair, explicit_pairs);
    }

    ASSERT_EQ(batched.size(), explicit_pairs.size());
    for (Size idx = 0; idx < batched.size(); ++idx) {
        EXPECT_NEAR(batched[idx], explicit_pairs[idx], TIGHT_1E_TOL)
            << "Mismatch at index " << idx
            << " (row=" << idx / nbf_ << ", col=" << idx % nbf_ << ")";
    }
}
