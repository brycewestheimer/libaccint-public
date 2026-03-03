// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2026 Bryce M. Westheimer. All rights reserved.

/// @file test_shell_set_quartet_utils.cpp
/// @brief Unit tests for ShellSetQuartet utility functions

#include <gtest/gtest.h>

#include <libaccint/basis/shell_set_quartet_utils.hpp>
#include <libaccint/basis/basis_set.hpp>
#include <libaccint/basis/shell.hpp>

// Suppress deprecation warnings for deprecated API usage
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"

namespace libaccint::testing {

namespace {

/// Water geometry in bohr
constexpr Point3D O_center{0.0, 0.0, 0.0};
constexpr Point3D H1_center{0.0, 1.43233673, -1.10866041};
constexpr Point3D H2_center{0.0, -1.43233673, -1.10866041};

/// Build STO-3G H2O basis
BasisSet make_h2o_basis() {
    std::vector<Shell> shells;

    // O 1s
    { Shell s(0, O_center, {130.7093200, 23.8088610, 6.4436083},
              {0.15432897, 0.53532814, 0.44463454});
      s.set_atom_index(0); shells.push_back(std::move(s)); }
    // O 2s
    { Shell s(0, O_center, {5.0331513, 1.1695961, 0.3803890},
              {-0.09996723, 0.39951283, 0.70011547});
      s.set_atom_index(0); shells.push_back(std::move(s)); }
    // O 2p
    { Shell s(1, O_center, {5.0331513, 1.1695961, 0.3803890},
              {0.15591627, 0.60768372, 0.39195739});
      s.set_atom_index(0); shells.push_back(std::move(s)); }
    // H1 1s
    { Shell s(0, H1_center, {3.42525091, 0.62391373, 0.16885540},
              {0.15432897, 0.53532814, 0.44463454});
      s.set_atom_index(1); shells.push_back(std::move(s)); }
    // H2 1s
    { Shell s(0, H2_center, {3.42525091, 0.62391373, 0.16885540},
              {0.15432897, 0.53532814, 0.44463454});
      s.set_atom_index(2); shells.push_back(std::move(s)); }

    return BasisSet(std::move(shells));
}

}  // anonymous namespace

// =============================================================================
// AMClass Tests
// =============================================================================

TEST(AMClassTest, DefaultConstruction) {
    AMClass am;
    EXPECT_EQ(am.La, 0);
    EXPECT_EQ(am.Lb, 0);
    EXPECT_EQ(am.Lc, 0);
    EXPECT_EQ(am.Ld, 0);
    EXPECT_EQ(am.total(), 0);
}

TEST(AMClassTest, TotalAngularMomentum) {
    AMClass am{1, 2, 0, 1};
    EXPECT_EQ(am.total(), 4);
}

TEST(AMClassTest, Equality) {
    AMClass a{0, 0, 0, 0};
    AMClass b{0, 0, 0, 0};
    AMClass c{1, 0, 0, 0};
    EXPECT_EQ(a, b);
    EXPECT_NE(a, c);
}

TEST(AMClassTest, Ordering) {
    AMClass a{0, 0, 0, 0};
    AMClass b{0, 0, 0, 1};
    AMClass c{1, 0, 0, 0};
    EXPECT_TRUE(a < b);
    EXPECT_TRUE(a < c);
    EXPECT_TRUE(b < c);
    EXPECT_FALSE(a < a);
}

// =============================================================================
// get_am_class Tests
// =============================================================================

TEST(GetAMClassTest, FromQuartet) {
    auto basis = make_h2o_basis();
    const auto& quartets = basis.shell_set_quartets();

    if (!quartets.empty()) {
        AMClass am = get_am_class(quartets[0]);
        EXPECT_GE(am.La, 0);
        EXPECT_GE(am.Lb, 0);
        EXPECT_GE(am.Lc, 0);
        EXPECT_GE(am.Ld, 0);
    }
}

// =============================================================================
// sort_by_total_am Tests
// =============================================================================

TEST(SortByTotalAMTest, SortedOrder) {
    auto basis = make_h2o_basis();
    const auto& quartets = basis.shell_set_quartets();

    auto sorted = sort_by_total_am(quartets);

    EXPECT_EQ(sorted.size(), quartets.size());

    // Verify sorted by total AM
    for (size_t i = 1; i < sorted.size(); ++i) {
        EXPECT_LE(sorted[i - 1]->L_total(), sorted[i]->L_total());
    }
}

TEST(SortByTotalAMTest, EmptyInput) {
    std::vector<ShellSetQuartet> empty;
    auto sorted = sort_by_total_am(empty);
    EXPECT_TRUE(sorted.empty());
}

// =============================================================================
// group_by_am_class Tests
// =============================================================================

TEST(GroupByAMClassTest, GroupsAreNonEmpty) {
    auto basis = make_h2o_basis();
    const auto& quartets = basis.shell_set_quartets();

    auto groups = group_by_am_class(quartets);

    EXPECT_FALSE(groups.empty());

    // Each group non-empty and all quartets in group have same AM class
    Size total_in_groups = 0;
    for (const auto& grp : groups) {
        EXPECT_FALSE(grp.quartets.empty());
        for (const auto* q : grp.quartets) {
            EXPECT_EQ(get_am_class(*q), grp.am_class);
        }
        total_in_groups += grp.quartets.size();
    }
    // All quartets accounted for
    EXPECT_EQ(total_in_groups, quartets.size());
}

TEST(GroupByAMClassTest, GroupsSortedByAMClass) {
    auto basis = make_h2o_basis();
    const auto& quartets = basis.shell_set_quartets();

    auto groups = group_by_am_class(quartets);

    for (size_t i = 1; i < groups.size(); ++i) {
        EXPECT_TRUE(groups[i - 1].am_class < groups[i].am_class);
    }
}

TEST(GroupByAMClassTest, EmptyInput) {
    std::vector<ShellSetQuartet> empty;
    auto groups = group_by_am_class(empty);
    EXPECT_TRUE(groups.empty());
}

// =============================================================================
// estimate_quartet_cost Tests
// =============================================================================

TEST(EstimateQuartetCostTest, NonZeroCost) {
    auto basis = make_h2o_basis();
    const auto& quartets = basis.shell_set_quartets();

    for (const auto& q : quartets) {
        double cost = estimate_quartet_cost(q);
        EXPECT_GT(cost, 0.0);
    }
}

TEST(EstimateQuartetCostTest, HigherAMMoreExpensive) {
    auto basis = make_h2o_basis();
    const auto& quartets = basis.shell_set_quartets();

    // Find an (ssss) and a (pppp) or (spsp) quartet
    double min_cost = std::numeric_limits<double>::max();
    double max_cost = 0.0;
    int min_total_am = std::numeric_limits<int>::max();
    int max_total_am = 0;

    for (const auto& q : quartets) {
        double cost = estimate_quartet_cost(q);
        int total_am = q.L_total();
        if (total_am < min_total_am) {
            min_total_am = total_am;
            min_cost = cost;
        }
        if (total_am > max_total_am) {
            max_total_am = total_am;
            max_cost = cost;
        }
    }

    if (max_total_am > min_total_am) {
        EXPECT_GT(max_cost, min_cost);
    }
}

// =============================================================================
// generate_quartets Tests
// =============================================================================

TEST(GenerateQuartetsTest, SelfPairing) {
    auto basis = make_h2o_basis();
    const auto& pairs = basis.shell_set_pairs();

    auto quartets = generate_quartets(
        std::span<const ShellSetPair>(pairs),
        std::span<const ShellSetPair>(pairs));

    // Self-pairing → upper triangle: n*(n+1)/2
    Size n = pairs.size();
    EXPECT_EQ(quartets.size(), n * (n + 1) / 2);
}

TEST(GenerateQuartetsTest, CrossPairing) {
    auto basis = make_h2o_basis();
    const auto& pairs = basis.shell_set_pairs();

    if (pairs.size() >= 2) {
        // Use different sub-spans
        auto bra = std::span<const ShellSetPair>(pairs.data(), 1);
        auto ket = std::span<const ShellSetPair>(pairs.data() + 1, 1);

        auto quartets = generate_quartets(bra, ket);
        EXPECT_EQ(quartets.size(), 1u);
    }
}

// =============================================================================
// filter_symmetry_unique Tests
// =============================================================================

TEST(FilterSymmetryUniqueTest, SubsetOfInput) {
    auto basis = make_h2o_basis();
    const auto& quartets = basis.shell_set_quartets();

    auto unique = filter_symmetry_unique(quartets);

    EXPECT_LE(unique.size(), quartets.size());
    EXPECT_FALSE(unique.empty());
}

TEST(FilterSymmetryUniqueTest, AllCanonical) {
    auto basis = make_h2o_basis();
    const auto& quartets = basis.shell_set_quartets();

    auto unique = filter_symmetry_unique(quartets);

    // All returned quartets should have bra AM tuple <= ket AM tuple
    for (const auto* q : unique) {
        auto bra_am = std::make_pair(q->La(), q->Lb());
        auto ket_am = std::make_pair(q->Lc(), q->Ld());
        EXPECT_LE(bra_am, ket_am);
    }
}

TEST(FilterSymmetryUniqueTest, EmptyInput) {
    std::vector<ShellSetQuartet> empty;
    auto unique = filter_symmetry_unique(empty);
    EXPECT_TRUE(unique.empty());
}

}  // namespace libaccint::testing

#pragma GCC diagnostic pop
