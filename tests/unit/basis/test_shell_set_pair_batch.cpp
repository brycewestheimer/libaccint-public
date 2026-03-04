// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2026 Bryce M. Westheimer. All rights reserved.

/// @file test_shell_set_pair_batch.cpp
/// @brief Unit tests for ShellSetPair batch operations

#include <gtest/gtest.h>

#include <libaccint/basis/shell_set_pair.hpp>
#include <libaccint/basis/basis_set.hpp>
#include <libaccint/basis/shell.hpp>
#include <libaccint/basis/shell_set.hpp>

// Suppress deprecation warnings
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"

namespace libaccint::testing {

namespace {

constexpr Point3D O_center{0.0, 0.0, 0.0};
constexpr Point3D H1_center{0.0, 1.43233673, -1.10866041};
constexpr Point3D H2_center{0.0, -1.43233673, -1.10866041};

BasisSet make_h2o_basis() {
    std::vector<Shell> shells;

    { Shell s(0, O_center, {130.709320, 23.808861, 6.443608},
              {0.15432897, 0.53532814, 0.44463454});
      s.set_atom_index(0); shells.push_back(std::move(s)); }
    { Shell s(0, O_center, {5.033151, 1.169596, 0.380389},
              {-0.09996723, 0.39951283, 0.70011547});
      s.set_atom_index(0); shells.push_back(std::move(s)); }
    { Shell s(1, O_center, {5.033151, 1.169596, 0.380389},
              {0.15591627, 0.60768372, 0.39195739});
      s.set_atom_index(0); shells.push_back(std::move(s)); }
    { Shell s(0, H1_center, {3.42525091, 0.62391373, 0.16885540},
              {0.15432897, 0.53532814, 0.44463454});
      s.set_atom_index(1); shells.push_back(std::move(s)); }
    { Shell s(0, H2_center, {3.42525091, 0.62391373, 0.16885540},
              {0.15432897, 0.53532814, 0.44463454});
      s.set_atom_index(2); shells.push_back(std::move(s)); }

    return BasisSet(std::move(shells));
}

}  // anonymous namespace

// =============================================================================
// Batch Construction Tests
// =============================================================================

TEST(ShellSetPairBatchTest, PairsGenerated) {
    auto basis = make_h2o_basis();
    const auto& pairs = basis.shell_set_pairs();

    // STO-3G H2O: 2 ShellSets (s with K=3, p with K=3)
    // Upper triangle pairs: (ss), (sp), (pp) = 3
    EXPECT_GE(pairs.size(), 1u);
}

TEST(ShellSetPairBatchTest, PairProperties) {
    auto basis = make_h2o_basis();
    const auto& pairs = basis.shell_set_pairs();

    for (const auto& pair : pairs) {
        EXPECT_GE(pair.La(), 0);
        EXPECT_GE(pair.Lb(), 0);
        EXPECT_GT(pair.n_pairs(), 0u);
    }
}

TEST(ShellSetPairBatchTest, ShellSetReferences) {
    auto basis = make_h2o_basis();
    const auto& pairs = basis.shell_set_pairs();

    for (const auto& pair : pairs) {
        // ShellSets referenced by pairs should have valid properties
        const auto& set_a = pair.shell_set_a();
        const auto& set_b = pair.shell_set_b();

        EXPECT_GT(set_a.n_shells(), 0u);
        EXPECT_GT(set_b.n_shells(), 0u);
        EXPECT_EQ(set_a.angular_momentum(), pair.La());
        EXPECT_EQ(set_b.angular_momentum(), pair.Lb());
    }
}

// =============================================================================
// Multiple AM Combination Tests
// =============================================================================

TEST(ShellSetPairBatchTest, MultipleAMCombinations) {
    auto basis = make_h2o_basis();
    const auto& pairs = basis.shell_set_pairs();

    // With s and p shells, we should have multiple AM combinations
    bool has_ss = false;
    bool has_sp = false;
    bool has_pp = false;

    for (const auto& pair : pairs) {
        int la = pair.La();
        int lb = pair.Lb();
        if (la == 0 && lb == 0) has_ss = true;
        if ((la == 0 && lb == 1) || (la == 1 && lb == 0)) has_sp = true;
        if (la == 1 && lb == 1) has_pp = true;
    }

    EXPECT_TRUE(has_ss);
    EXPECT_TRUE(has_sp || has_pp);
}

// =============================================================================
// Pair Iteration Tests
// =============================================================================

TEST(ShellSetPairBatchTest, PairIteration) {
    auto basis = make_h2o_basis();
    const auto& pairs = basis.shell_set_pairs();

    for (const auto& pair : pairs) {
        // Each pair should have at least one shell pair
        EXPECT_GT(pair.n_pairs(), 0u);
    }
}

// =============================================================================
// Schwarz Bound Tests
// =============================================================================

TEST(ShellSetPairBatchTest, SchwarzBounds) {
    auto basis = make_h2o_basis();
    const auto& pairs = basis.shell_set_pairs();

    for (const auto& pair : pairs) {
        // Schwarz bound should be non-negative
        double bound = pair.schwarz_bound();
        EXPECT_GE(bound, 0.0);
    }
}

TEST(ShellSetPairBatchTest, PairsCountConsistency) {
    auto basis = make_h2o_basis();
    const auto& pairs = basis.shell_set_pairs();

    for (const auto& pair : pairs) {
        // n_pairs should equal n_shells_a * n_shells_b
        Size expected = pair.shell_set_a().n_shells() * pair.shell_set_b().n_shells();
        EXPECT_EQ(pair.n_pairs(), expected);
    }
}

}  // namespace libaccint::testing

#pragma GCC diagnostic pop
