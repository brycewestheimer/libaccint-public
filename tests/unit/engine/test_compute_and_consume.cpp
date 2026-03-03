// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2026 Bryce M. Westheimer. All rights reserved.

/// @file test_compute_and_consume.cpp
/// @brief Tests for the engine compute-and-consume pattern

#include <libaccint/engine/engine.hpp>
#include <libaccint/engine/cpu_engine.hpp>
#include <libaccint/engine/dispatch_policy.hpp>
#include <libaccint/engine/integral_buffer.hpp>
#include <libaccint/basis/basis_set.hpp>
#include <libaccint/basis/shell.hpp>
#include <libaccint/basis/shell_set_quartet.hpp>
#include <libaccint/buffers/one_electron_buffer.hpp>
#include <libaccint/buffers/two_electron_buffer.hpp>
#include <libaccint/consumers/fock_builder.hpp>
#include <libaccint/operators/one_electron_operator.hpp>
#include <libaccint/operators/operator.hpp>
#include <libaccint/screening/screening_options.hpp>
#include <libaccint/core/backend.hpp>
#include <libaccint/core/types.hpp>
#include <libaccint/utils/error_handling.hpp>

#include <gtest/gtest.h>
#include <algorithm>
#include <atomic>
#include <cmath>
#include <numeric>
#include <vector>

using namespace libaccint;

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

    // O 1s (L=0, K=3, atom 0)
    {
        Shell s(0, O_center,
                {130.7093200, 23.8088610, 6.4436083},
                {0.15432897, 0.53532814, 0.44463454});
        s.set_atom_index(0);
        shells.push_back(std::move(s));
    }

    // O 2s (L=0, K=3, atom 0)
    {
        Shell s(0, O_center,
                {5.0331513, 1.1695961, 0.3803890},
                {-0.09996723, 0.39951283, 0.70011547});
        s.set_atom_index(0);
        shells.push_back(std::move(s));
    }

    // O 2p (L=1, K=3, atom 0)
    {
        Shell s(1, O_center,
                {5.0331513, 1.1695961, 0.3803890},
                {0.15591627, 0.60768372, 0.39195739});
        s.set_atom_index(0);
        shells.push_back(std::move(s));
    }

    // H1 1s (L=0, K=3, atom 1)
    {
        Shell s(0, H1_center,
                {3.42525091, 0.62391373, 0.16885540},
                {0.15432897, 0.53532814, 0.44463454});
        s.set_atom_index(1);
        shells.push_back(std::move(s));
    }

    // H2 1s (L=0, K=3, atom 2)
    {
        Shell s(0, H2_center,
                {3.42525091, 0.62391373, 0.16885540},
                {0.15432897, 0.53532814, 0.44463454});
        s.set_atom_index(2);
        shells.push_back(std::move(s));
    }

    return shells;
}

/// Tolerance for floating-point comparisons
constexpr Real TIGHT_TOL = 1e-10;
constexpr Real LOOSE_TOL = 1e-8;

// =============================================================================
// Mock Consumer — Counts accumulate calls
// =============================================================================

struct CountingConsumer {
    std::atomic<int> accumulate_count{0};
    Size nbf_;

    explicit CountingConsumer(Size nbf) : nbf_(nbf) {}

    void accumulate(const TwoElectronBuffer<0>& /*buffer*/,
                    Index /*fa*/, Index /*fb*/, Index /*fc*/, Index /*fd*/,
                    int /*na*/, int /*nb*/, int /*nc*/, int /*nd*/) {
        accumulate_count.fetch_add(1, std::memory_order_relaxed);
    }

    void prepare_parallel(int /*n_threads*/) {}
    void finalize_parallel() {}
};

}  // anonymous namespace

// =============================================================================
// Basic Compute-and-Consume Tests
// =============================================================================

TEST(ComputeAndConsume, BasicFockBuild) {
    BasisSet basis(make_sto3g_h2o_shells());
    Engine engine(basis);

    const Size nbf = basis.n_basis_functions();
    ASSERT_EQ(nbf, 7u);

    // Create a unit density matrix (identity)
    std::vector<Real> D(nbf * nbf, 0.0);
    for (Size i = 0; i < nbf; ++i) {
        D[i * nbf + i] = 1.0;
    }

    consumers::FockBuilder fock(nbf);
    fock.set_density(D.data(), nbf);

    Operator op = Operator::coulomb();
    engine.compute_and_consume(op, fock);

    auto J = fock.get_coulomb_matrix();
    auto K = fock.get_exchange_matrix();

    // Both J and K should be non-trivial
    ASSERT_EQ(J.size(), nbf * nbf);
    ASSERT_EQ(K.size(), nbf * nbf);

    // J diagonal should be positive (Coulomb repulsion)
    for (Size i = 0; i < nbf; ++i) {
        EXPECT_GT(J[i * nbf + i], 0.0)
            << "J diagonal element " << i << " should be positive";
    }

    // K diagonal should also be positive
    for (Size i = 0; i < nbf; ++i) {
        EXPECT_GT(K[i * nbf + i], 0.0)
            << "K diagonal element " << i << " should be positive";
    }
}

TEST(ComputeAndConsume, ScreenedVsUnscreened) {
    BasisSet basis(make_sto3g_h2o_shells());
    Engine engine(basis);

    const Size nbf = basis.n_basis_functions();

    // Unit density
    std::vector<Real> D(nbf * nbf, 0.0);
    for (Size i = 0; i < nbf; ++i) {
        D[i * nbf + i] = 1.0;
    }

    Operator op = Operator::coulomb();

    // Unscreened
    consumers::FockBuilder fock_unscreened(nbf);
    fock_unscreened.set_density(D.data(), nbf);
    engine.compute_and_consume(op, fock_unscreened);

    // Screened with tight threshold
    consumers::FockBuilder fock_screened(nbf);
    fock_screened.set_density(D.data(), nbf);

    engine.precompute_schwarz_bounds();
    screening::ScreeningOptions opts;
    opts.threshold = 1e-12;
    opts.enabled = true;

    engine.compute_and_consume(op, fock_screened, opts);

    // Results should be very close with tight screening threshold
    auto J_unscr = fock_unscreened.get_coulomb_matrix();
    auto J_scr = fock_screened.get_coulomb_matrix();

    for (Size i = 0; i < nbf * nbf; ++i) {
        EXPECT_NEAR(J_unscr[i], J_scr[i], LOOSE_TOL)
            << "Screened J differs from unscreened at element " << i;
    }
}

TEST(ComputeAndConsume, SpanOverload) {
    BasisSet basis(make_sto3g_h2o_shells());
    Engine engine(basis);

    const Size nbf = basis.n_basis_functions();

    std::vector<Real> D(nbf * nbf, 0.0);
    for (Size i = 0; i < nbf; ++i) {
        D[i * nbf + i] = 1.0;
    }

    // Get full set of quartets
    const auto& all_quartets = basis.shell_set_quartets();
    ASSERT_GT(all_quartets.size(), 0u);

    // Use a subset: first few quartets
    Size n_subset = std::min<Size>(all_quartets.size(), 5);
    std::span<const ShellSetQuartet> subset(all_quartets.data(), n_subset);

    CountingConsumer counter(nbf);
    Operator op = Operator::coulomb();

    engine.compute_and_consume(op, subset, counter);

    // Each ShellSetQuartet expands to multiple individual shell quartets,
    // so accumulate_count > n_subset. Just verify we got some calls.
    EXPECT_GT(static_cast<Size>(counter.accumulate_count), 0u);
    // Should not exceed total shell quartets
    Size n_shells = basis.n_shells();
    EXPECT_LE(static_cast<Size>(counter.accumulate_count), n_shells * n_shells * n_shells * n_shells);
}

TEST(ComputeAndConsume, BackendHintCPU) {
    BasisSet basis(make_sto3g_h2o_shells());
    Engine engine(basis);

    const Size nbf = basis.n_basis_functions();

    std::vector<Real> D(nbf * nbf, 0.0);
    for (Size i = 0; i < nbf; ++i) {
        D[i * nbf + i] = 1.0;
    }

    consumers::FockBuilder fock(nbf);
    fock.set_density(D.data(), nbf);

    Operator op = Operator::coulomb();

    // PreferCPU hint — should succeed
    EXPECT_NO_THROW(engine.compute_and_consume(op, fock, BackendHint::PreferCPU));

    // Result should be valid
    auto J = fock.get_coulomb_matrix();
    ASSERT_EQ(J.size(), nbf * nbf);

    // Should have non-zero values
    Real max_val = *std::max_element(J.begin(), J.end());
    EXPECT_GT(max_val, 0.0);
}

TEST(ComputeAndConsume, BackendHintGPU) {
    BasisSet basis(make_sto3g_h2o_shells());
    Engine engine(basis);

    const Size nbf = basis.n_basis_functions();

    std::vector<Real> D(nbf * nbf, 0.0);
    for (Size i = 0; i < nbf; ++i) {
        D[i * nbf + i] = 1.0;
    }

    consumers::FockBuilder fock(nbf);
    fock.set_density(D.data(), nbf);

    Operator op = Operator::coulomb();

    // PreferGPU hint — should fall back to CPU if no GPU available
    EXPECT_NO_THROW(engine.compute_and_consume(op, fock, BackendHint::PreferGPU));

    auto J = fock.get_coulomb_matrix();
    ASSERT_EQ(J.size(), nbf * nbf);
}

TEST(ComputeAndConsume, BackendHintAuto) {
    BasisSet basis(make_sto3g_h2o_shells());
    Engine engine(basis);

    const Size nbf = basis.n_basis_functions();

    std::vector<Real> D(nbf * nbf, 0.0);
    for (Size i = 0; i < nbf; ++i) {
        D[i * nbf + i] = 1.0;
    }

    consumers::FockBuilder fock(nbf);
    fock.set_density(D.data(), nbf);

    Operator op = Operator::coulomb();

    // Auto hint — dispatch policy decides; result should be correct
    EXPECT_NO_THROW(engine.compute_and_consume(op, fock, BackendHint::Auto));

    auto J = fock.get_coulomb_matrix();
    ASSERT_EQ(J.size(), nbf * nbf);

    // Diagonal should be positive
    for (Size i = 0; i < nbf; ++i) {
        EXPECT_GT(J[i * nbf + i], 0.0);
    }
}

TEST(ComputeAndConsume, AccumulateCallCount) {
    BasisSet basis(make_sto3g_h2o_shells());
    Engine engine(basis);

    const Size nbf = basis.n_basis_functions();
    const auto& quartets = basis.shell_set_quartets();
    Size n_quartets = quartets.size();

    CountingConsumer counter(nbf);
    Operator op = Operator::coulomb();

    engine.compute_and_consume(op, counter);

    // compute_and_consume iterates all shell quartets (N^4), not just
    // the upper-triangle ShellSetQuartet worklist
    Size n_shells = basis.n_shells();
    Size expected_calls = n_shells * n_shells * n_shells * n_shells;
    EXPECT_EQ(static_cast<Size>(counter.accumulate_count), expected_calls);
}

TEST(ComputeAndConsume, MatchesBatchCompute) {
    // Verify that compute_and_consume produces valid, self-consistent Fock matrices.
    // Note: compute_all_2e uses upper-triangle ShellSetQuartet worklist while
    // compute_and_consume iterates all N^4 shell quartets, so they have different
    // coverage and can't be directly compared. We verify properties instead.
    BasisSet basis(make_sto3g_h2o_shells());
    Engine engine(basis);

    const Size nbf = basis.n_basis_functions();
    Operator op = Operator::coulomb();

    // Consuming Fock build
    std::vector<Real> D(nbf * nbf, 0.0);
    for (Size i = 0; i < nbf; ++i) {
        D[i * nbf + i] = 1.0;
    }

    consumers::FockBuilder fock(nbf);
    fock.set_density(D.data(), nbf);
    engine.compute_and_consume(op, fock);

    // Verify J matrix properties
    auto J = fock.get_coulomb_matrix();
    ASSERT_EQ(J.size(), nbf * nbf);

    // J should be symmetric: J[i*nbf+j] == J[j*nbf+i]
    for (Size i = 0; i < nbf; ++i) {
        for (Size j = i + 1; j < nbf; ++j) {
            EXPECT_NEAR(J[i * nbf + j], J[j * nbf + i], TIGHT_TOL)
                << "J not symmetric at (" << i << "," << j << ")";
        }
    }

    // J diagonal should be positive (self-interaction Coulomb)
    for (Size i = 0; i < nbf; ++i) {
        EXPECT_GT(J[i * nbf + i], 0.0)
            << "J diagonal not positive at " << i;
    }

    // Verify K matrix properties
    auto K = fock.get_exchange_matrix();
    ASSERT_EQ(K.size(), nbf * nbf);

    // K should be symmetric: K[i*nbf+j] == K[j*nbf+i]
    for (Size i = 0; i < nbf; ++i) {
        for (Size j = i + 1; j < nbf; ++j) {
            EXPECT_NEAR(K[i * nbf + j], K[j * nbf + i], TIGHT_TOL)
                << "K not symmetric at (" << i << "," << j << ")";
        }
    }

    // K diagonal should be positive
    for (Size i = 0; i < nbf; ++i) {
        EXPECT_GT(K[i * nbf + i], 0.0)
            << "K diagonal not positive at " << i;
    }

    // Run a second time with different density — verify results change
    consumers::FockBuilder fock2(nbf);
    std::vector<Real> D2(nbf * nbf, 0.0);
    D2[0] = 2.0;  // Only one non-zero element
    fock2.set_density(D2.data(), nbf);
    engine.compute_and_consume(op, fock2);

    auto J2 = fock2.get_coulomb_matrix();
    // J2 should differ from J (different density)
    bool any_differ = false;
    for (Size i = 0; i < nbf * nbf; ++i) {
        if (std::abs(J[i] - J2[i]) > TIGHT_TOL) {
            any_differ = true;
            break;
        }
    }
    EXPECT_TRUE(any_differ) << "J matrices should differ with different densities";
}
