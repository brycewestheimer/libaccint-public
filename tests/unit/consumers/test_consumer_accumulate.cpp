// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2026 Bryce M. Westheimer. All rights reserved.

/// @file test_consumer_accumulate.cpp
/// @brief Tests for consumer accumulate patterns: linearity, idempotency, reset

#include <libaccint/consumers/fock_builder.hpp>
#include <libaccint/engine/engine.hpp>
#include <libaccint/basis/basis_set.hpp>
#include <libaccint/basis/shell.hpp>
#include <libaccint/operators/operator.hpp>
#include <libaccint/core/types.hpp>

#include <gtest/gtest.h>
#include <cmath>
#include <vector>

using namespace libaccint;
using namespace libaccint::consumers;

namespace {

constexpr Real TOL = 1e-10;

std::vector<Shell> make_h2_shells() {
    std::vector<Shell> shells;

    Shell s0(0, Point3D{0.0, 0.0, 0.0},
             {3.42525091, 0.62391373, 0.16885540},
             {0.15432897, 0.53532814, 0.44463454});
    s0.set_atom_index(0);
    shells.push_back(std::move(s0));

    Shell s1(0, Point3D{1.4, 0.0, 0.0},
             {3.42525091, 0.62391373, 0.16885540},
             {0.15432897, 0.53532814, 0.44463454});
    s1.set_atom_index(1);
    shells.push_back(std::move(s1));

    return shells;
}

}  // anonymous namespace

// =============================================================================
// FockBuilder Accumulate Pattern Tests
// =============================================================================

TEST(ConsumerAccumulatePattern, FockBuilderDoubleAccumulateIsCumulative) {
    // Running compute_and_consume twice should give 2x the result
    BasisSet basis(make_h2_shells());
    Engine engine(basis);
    const Size nbf = basis.n_basis_functions();

    std::vector<Real> D(nbf * nbf, 0.0);
    D[0] = 1.0;

    Operator op = Operator::coulomb();

    // Single pass
    FockBuilder fock1(nbf);
    fock1.set_density(D.data(), nbf);
    engine.compute_and_consume(op, fock1);
    auto J1 = fock1.get_coulomb_matrix();

    // Double pass (no reset between)
    FockBuilder fock2(nbf);
    fock2.set_density(D.data(), nbf);
    engine.compute_and_consume(op, fock2);
    engine.compute_and_consume(op, fock2);
    auto J2 = fock2.get_coulomb_matrix();

    // J2 should be 2 * J1
    for (Size i = 0; i < nbf * nbf; ++i) {
        EXPECT_NEAR(J2[i], 2.0 * J1[i], TOL)
            << "Double accumulation not cumulative at " << i;
    }
}

TEST(ConsumerAccumulatePattern, FockBuilderResetBetweenPasses) {
    // Reset should give identical results to a fresh builder
    BasisSet basis(make_h2_shells());
    Engine engine(basis);
    const Size nbf = basis.n_basis_functions();

    std::vector<Real> D(nbf * nbf, 0.0);
    D[0] = 1.0;

    Operator op = Operator::coulomb();

    // First pass
    FockBuilder fock(nbf);
    fock.set_density(D.data(), nbf);
    engine.compute_and_consume(op, fock);
    auto J1 = fock.get_coulomb_matrix();

    // Reset and second pass
    fock.reset();
    fock.set_density(D.data(), nbf);
    engine.compute_and_consume(op, fock);
    auto J2 = fock.get_coulomb_matrix();

    for (Size i = 0; i < nbf * nbf; ++i) {
        EXPECT_NEAR(J1[i], J2[i], TOL)
            << "Reset + recompute differs at " << i;
    }
}

