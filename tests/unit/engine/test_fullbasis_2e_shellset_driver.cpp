// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2026 Bryce M. Westheimer. All rights reserved.

/// @file test_fullbasis_2e_shellset_driver.cpp
/// @brief Regression tests verifying Engine full-basis 2e computation produces
///        correct results via ShellSetQuartet work-unit drivers.

#include <libaccint/engine/engine.hpp>
#include <libaccint/consumers/fock_builder.hpp>
#include <libaccint/basis/basis_set.hpp>
#include <libaccint/basis/shell.hpp>
#include <libaccint/basis/shell_set.hpp>
#include <libaccint/basis/shell_set_pair.hpp>
#include <libaccint/basis/shell_set_quartet.hpp>
#include <libaccint/buffers/two_electron_buffer.hpp>
#include <libaccint/operators/operator.hpp>
#include <libaccint/core/types.hpp>

#include <gtest/gtest.h>
#include <cmath>
#include <span>
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

/// Build a simple diagonal density matrix for testing
std::vector<Real> make_diagonal_density(Size nbf) {
    std::vector<Real> D(nbf * nbf, 0.0);
    for (Size i = 0; i < nbf; ++i) {
        D[i * nbf + i] = 1.0 / static_cast<Real>(nbf);
    }
    return D;
}

// Tolerance for 2e Fock comparisons (more lenient due to FP ordering)
constexpr Real FOCK_TOL = 1e-10;

}  // anonymous namespace

// =============================================================================
// Test Fixture
// =============================================================================

class FullBasis2eShellSetDriver : public ::testing::Test {
protected:
    void SetUp() override {
        basis_ = std::make_unique<BasisSet>(make_sto3g_h2o_shells());
        engine_ = std::make_unique<Engine>(*basis_);
        nbf_ = basis_->n_basis_functions();
        ASSERT_EQ(nbf_, 7u);
        D_ = make_diagonal_density(nbf_);
    }

    /// Build Fock (J, K) via manual N^4 shell loop (reference).
    void build_fock_reference(consumers::FockBuilder& fock_ref) {
        const Size n_shells = basis_->n_shells();
        TwoElectronBuffer<0> buffer;

        for (Size i = 0; i < n_shells; ++i) {
            const auto& si = basis_->shell(i);
            Index fi = si.function_index();
            int ni = si.n_functions();

            for (Size j = 0; j < n_shells; ++j) {
                const auto& sj = basis_->shell(j);
                Index fj = sj.function_index();
                int nj = sj.n_functions();

                for (Size k = 0; k < n_shells; ++k) {
                    const auto& sk = basis_->shell(k);
                    Index fk = sk.function_index();
                    int nk = sk.n_functions();

                    for (Size l = 0; l < n_shells; ++l) {
                        const auto& sl = basis_->shell(l);
                        Index fl = sl.function_index();
                        int nl = sl.n_functions();

                        engine_->compute_2e_shell_quartet(
                            Operator::coulomb(), si, sj, sk, sl, buffer);
                        fock_ref.accumulate(buffer, fi, fj, fk, fl,
                                            ni, nj, nk, nl);
                    }
                }
            }
        }
    }

    /// Compare two Fock-sized spans element-wise.
    void compare_matrices(std::span<const Real> A,
                          std::span<const Real> B,
                          Real tol,
                          const char* label) {
        ASSERT_EQ(A.size(), B.size());
        for (Size idx = 0; idx < A.size(); ++idx) {
            EXPECT_NEAR(A[idx], B[idx], tol)
                << label << " mismatch at index " << idx
                << " (row=" << idx / nbf_ << ", col=" << idx % nbf_ << ")";
        }
    }

    std::unique_ptr<BasisSet> basis_;
    std::unique_ptr<Engine> engine_;
    Size nbf_{0};
    std::vector<Real> D_;
};

// =============================================================================
// Tests
// =============================================================================

TEST_F(FullBasis2eShellSetDriver, FockViaShellSetQuartets_MatchesDirectCompute) {
    // Full-basis Fock via compute_and_consume (uses ShellSetQuartet driver)
    consumers::FockBuilder fock1(nbf_);
    fock1.set_density(D_.data(), nbf_);
    engine_->compute(Operator::coulomb(), fock1);

    // Reference: manual N^4 shell loop
    consumers::FockBuilder fock_ref(nbf_);
    fock_ref.set_density(D_.data(), nbf_);
    build_fock_reference(fock_ref);

    compare_matrices(fock1.get_coulomb_matrix(),
                     fock_ref.get_coulomb_matrix(),
                     FOCK_TOL, "J");
    compare_matrices(fock1.get_exchange_matrix(),
                     fock_ref.get_exchange_matrix(),
                     FOCK_TOL, "K");
}

TEST_F(FullBasis2eShellSetDriver, FockCoulombMatrixSymmetry) {
    consumers::FockBuilder fock(nbf_);
    fock.set_density(D_.data(), nbf_);
    engine_->compute(Operator::coulomb(), fock);

    auto J = fock.get_coulomb_matrix();
    for (Size i = 0; i < nbf_; ++i) {
        for (Size j = i + 1; j < nbf_; ++j) {
            EXPECT_NEAR(J[i * nbf_ + j], J[j * nbf_ + i], FOCK_TOL)
                << "J symmetry violated at [" << i << "," << j << "]";
        }
    }
}

TEST_F(FullBasis2eShellSetDriver, FockExchangeMatrixSymmetry) {
    consumers::FockBuilder fock(nbf_);
    fock.set_density(D_.data(), nbf_);
    engine_->compute(Operator::coulomb(), fock);

    auto K = fock.get_exchange_matrix();
    for (Size i = 0; i < nbf_; ++i) {
        for (Size j = i + 1; j < nbf_; ++j) {
            EXPECT_NEAR(K[i * nbf_ + j], K[j * nbf_ + i], FOCK_TOL)
                << "K symmetry violated at [" << i << "," << j << "]";
        }
    }
}

TEST_F(FullBasis2eShellSetDriver, ExplicitQuartetIteration_MatchesFullBasis) {
    // Fock via explicit iteration over all ShellSet combinations
    // (matches the full N^4 iteration in compute_and_consume_impl)
    //
    // When OpenMP is enabled, compute_shell_set_quartet calls
    // prepare_parallel / finalize_parallel on each invocation.
    // To avoid repeated thread-local buffer resets we drive the
    // loop at the individual shell level, matching the reference.
    consumers::FockBuilder fock1(nbf_);
    fock1.set_density(D_.data(), nbf_);

    // Drive manually through all ShellSet tuples, computing each
    // shell quartet individually (sequential accumulation).
    auto sets = basis_->shell_sets();
    const Size n_sets = sets.size();
    TwoElectronBuffer<0> buffer;
    for (Size ia = 0; ia < n_sets; ++ia) {
        const auto& set_a = *sets[ia];
        for (Size ib = 0; ib < n_sets; ++ib) {
            const auto& set_b = *sets[ib];
            for (Size ic = 0; ic < n_sets; ++ic) {
                const auto& set_c = *sets[ic];
                for (Size id = 0; id < n_sets; ++id) {
                    const auto& set_d = *sets[id];
                    for (Size i = 0; i < set_a.n_shells(); ++i) {
                        const auto& shell_a = set_a.shell(i);
                        Index fi = shell_a.function_index();
                        int na = shell_a.n_functions();
                        for (Size j = 0; j < set_b.n_shells(); ++j) {
                            const auto& shell_b = set_b.shell(j);
                            Index fj = shell_b.function_index();
                            int nb = shell_b.n_functions();
                            for (Size k = 0; k < set_c.n_shells(); ++k) {
                                const auto& shell_c = set_c.shell(k);
                                Index fk = shell_c.function_index();
                                int nc = shell_c.n_functions();
                                for (Size l = 0; l < set_d.n_shells(); ++l) {
                                    const auto& shell_d = set_d.shell(l);
                                    Index fl = shell_d.function_index();
                                    int nl = shell_d.n_functions();
                                    engine_->compute_2e_shell_quartet(
                                        Operator::coulomb(),
                                        shell_a, shell_b, shell_c, shell_d,
                                        buffer);
                                    fock1.accumulate(buffer, fi, fj, fk, fl,
                                                     na, nb, nc, nl);
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    // Fock via full-basis compute_and_consume
    consumers::FockBuilder fock2(nbf_);
    fock2.set_density(D_.data(), nbf_);
    engine_->compute(Operator::coulomb(), fock2);

    compare_matrices(fock1.get_coulomb_matrix(),
                     fock2.get_coulomb_matrix(),
                     FOCK_TOL, "J (explicit vs full-basis)");
    compare_matrices(fock1.get_exchange_matrix(),
                     fock2.get_exchange_matrix(),
                     FOCK_TOL, "K (explicit vs full-basis)");
}

TEST_F(FullBasis2eShellSetDriver, WorklistOverload_MatchesDirectCompute) {
    // Build a complete worklist covering all ShellSet combinations
    auto sets = basis_->shell_sets();
    const Size n_sets = sets.size();
    std::vector<ShellSetPair> all_pairs;
    for (Size ia = 0; ia < n_sets; ++ia) {
        for (Size ib = 0; ib < n_sets; ++ib) {
            all_pairs.emplace_back(*sets[ia], *sets[ib]);
        }
    }
    std::vector<ShellSetQuartet> all_quartets;
    for (Size ip = 0; ip < all_pairs.size(); ++ip) {
        for (Size iq = 0; iq < all_pairs.size(); ++iq) {
            all_quartets.emplace_back(all_pairs[ip], all_pairs[iq]);
        }
    }

    // Fock via worklist (span) overload — use ThreadLocal strategy so
    // the per-quartet prepare_parallel/finalize_parallel cycles from
    // OpenMP-enabled compute_shell_set_quartet correctly accumulate
    // via thread-local reduction.
    consumers::FockBuilder fock1(nbf_);
    fock1.set_density(D_.data(), nbf_);
    fock1.set_threading_strategy(consumers::FockThreadingStrategy::ThreadLocal);
    engine_->compute(Operator::coulomb(),
                     std::span<const ShellSetQuartet>(all_quartets),
                     fock1);

    // Reference: manual N^4 shell loop
    consumers::FockBuilder fock_ref(nbf_);
    fock_ref.set_density(D_.data(), nbf_);
    build_fock_reference(fock_ref);

    compare_matrices(fock1.get_coulomb_matrix(),
                     fock_ref.get_coulomb_matrix(),
                     FOCK_TOL, "J (worklist vs direct)");
    compare_matrices(fock1.get_exchange_matrix(),
                     fock_ref.get_exchange_matrix(),
                     FOCK_TOL, "K (worklist vs direct)");
}
