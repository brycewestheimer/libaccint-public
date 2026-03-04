// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2026 Bryce M. Westheimer. All rights reserved.

/// @file test_shell_set_triple.cpp
/// @brief Unit tests for ShellSetTriple and generate_shell_set_triples

#include <gtest/gtest.h>

#include <libaccint/basis/shell_set_triple.hpp>
#include <libaccint/basis/basis_set.hpp>
#include <libaccint/basis/auxiliary_basis_set.hpp>
#include <libaccint/basis/shell.hpp>
#include <libaccint/basis/shell_set.hpp>

namespace libaccint::testing {

// =============================================================================
// Helpers
// =============================================================================

namespace {

/// Water geometry in bohr
constexpr Point3D O_center{0.0, 0.0, 0.0};
constexpr Point3D H1_center{0.0, 1.43233673, -1.10866041};
constexpr Point3D H2_center{0.0, -1.43233673, -1.10866041};

/// Create a minimal STO-3G-like H2O basis (5 shells)
BasisSet make_h2o_orbital() {
    std::vector<Shell> shells;

    // O 1s (L=0, K=3)
    {
        Shell s(0, O_center,
                {130.709320, 23.808861, 6.443608},
                {0.15432897, 0.53532814, 0.44463454});
        s.set_atom_index(0);
        shells.push_back(std::move(s));
    }
    // O 2s (L=0, K=3)
    {
        Shell s(0, O_center,
                {5.033151, 1.169596, 0.380389},
                {-0.09996723, 0.39951283, 0.70011547});
        s.set_atom_index(0);
        shells.push_back(std::move(s));
    }
    // O 2p (L=1, K=3)
    {
        Shell s(1, O_center,
                {5.033151, 1.169596, 0.380389},
                {0.15591627, 0.60768372, 0.39195739});
        s.set_atom_index(0);
        shells.push_back(std::move(s));
    }
    // H1 1s (L=0, K=3)
    {
        Shell s(0, H1_center,
                {3.42525091, 0.62391373, 0.16885540},
                {0.15432897, 0.53532814, 0.44463454});
        s.set_atom_index(1);
        shells.push_back(std::move(s));
    }
    // H2 1s (L=0, K=3)
    {
        Shell s(0, H2_center,
                {3.42525091, 0.62391373, 0.16885540},
                {0.15432897, 0.53532814, 0.44463454});
        s.set_atom_index(2);
        shells.push_back(std::move(s));
    }

    return BasisSet(std::move(shells));
}

/// Create a simple auxiliary basis with known structure
AuxiliaryBasisSet make_simple_auxiliary() {
    std::vector<Shell> shells;

    // Two s-shells (K=1) on H atoms
    shells.push_back(Shell(0, H1_center, {9.297}, {1.0}));
    shells.back().set_atom_index(1);
    shells.back().set_shell_index(0);

    shells.push_back(Shell(0, H2_center, {1.459}, {1.0}));
    shells.back().set_atom_index(2);
    shells.back().set_shell_index(1);

    // One p-shell (K=1) on O
    shells.push_back(Shell(1, O_center, {2.726}, {1.0}));
    shells.back().set_atom_index(0);
    shells.back().set_shell_index(2);

    return AuxiliaryBasisSet(std::move(shells), FittingType::RI, "test-RI");
}

}  // anonymous namespace

// =============================================================================
// ShellSetTripleKey Tests
// =============================================================================

TEST(ShellSetTripleKeyTest, DefaultConstruction) {
    ShellSetTripleKey key;
    EXPECT_EQ(key.am_a, 0);
    EXPECT_EQ(key.am_b, 0);
    EXPECT_EQ(key.am_P, 0);
}

TEST(ShellSetTripleKeyTest, ValueConstruction) {
    ShellSetTripleKey key(1, 0, 2, 3, 3, 1);
    EXPECT_EQ(key.am_a, 1);
    EXPECT_EQ(key.am_b, 0);
    EXPECT_EQ(key.am_P, 2);
    EXPECT_EQ(key.n_prim_a, 3);
    EXPECT_EQ(key.n_prim_b, 3);
    EXPECT_EQ(key.n_prim_P, 1);
}

TEST(ShellSetTripleKeyTest, Equality) {
    ShellSetTripleKey a(1, 0, 2, 3, 3, 1);
    ShellSetTripleKey b(1, 0, 2, 3, 3, 1);
    ShellSetTripleKey c(0, 0, 2, 3, 3, 1);
    EXPECT_EQ(a, b);
    EXPECT_NE(a, c);
}

TEST(ShellSetTripleKeyTest, Ordering) {
    ShellSetTripleKey a(0, 0, 0, 1, 1, 1);
    ShellSetTripleKey b(0, 0, 1, 1, 1, 1);
    ShellSetTripleKey c(1, 0, 0, 1, 1, 1);
    EXPECT_TRUE(a < b);
    EXPECT_TRUE(a < c);
    EXPECT_FALSE(a < a);
}

// =============================================================================
// ShellSetTriple Construction from ShellSets Tests
// =============================================================================

TEST(ShellSetTripleTest, ConstructFromShellSets) {
    // Create three ShellSets
    Shell s1(0, O_center, {3.0, 1.0, 0.3}, {0.15, 0.53, 0.44});
    Shell s2(0, H1_center, {3.0, 1.0, 0.3}, {0.15, 0.53, 0.44});
    Shell p1(1, O_center, {5.0, 1.0, 0.4}, {0.16, 0.61, 0.39});
    Shell aux_s(0, O_center, {9.3}, {1.0});

    ShellSet set_a(0, 3);
    set_a.add_shell(s1);
    set_a.add_shell(s2);

    ShellSet set_b(1, 3);
    set_b.add_shell(p1);

    ShellSet set_P(0, 1);
    set_P.add_shell(aux_s);

    ShellSetTriple triple(set_a, set_b, set_P, false);

    EXPECT_EQ(triple.am_a(), 0);
    EXPECT_EQ(triple.am_b(), 1);
    EXPECT_EQ(triple.am_P(), 0);
    EXPECT_EQ(triple.n_primitives_a(), 3);
    EXPECT_EQ(triple.n_primitives_b(), 3);
    EXPECT_EQ(triple.n_primitives_P(), 1);
    // 2 shells * 1 shell * 1 shell = 2 triples (no symmetry)
    EXPECT_EQ(triple.size(), 2u);
}

TEST(ShellSetTripleTest, SymmetricConstruction) {
    Shell s1(0, O_center, {3.0, 1.0, 0.3}, {0.15, 0.53, 0.44});
    s1.set_shell_index(0);
    Shell s2(0, H1_center, {3.0, 1.0, 0.3}, {0.15, 0.53, 0.44});
    s2.set_shell_index(1);
    Shell aux_s(0, O_center, {9.3}, {1.0});
    aux_s.set_shell_index(0);

    ShellSet set_ab(0, 3);
    set_ab.add_shell(s1);
    set_ab.add_shell(s2);

    ShellSet set_P(0, 1);
    set_P.add_shell(aux_s);

    // symmetric: same ShellSet for a and b, only i <= j
    ShellSetTriple triple(set_ab, set_ab, set_P, true);

    // With 2 shells and symmetric, pairs are (0,0), (0,1), (1,1) = 3
    // Each with 1 aux shell → 3 triples
    EXPECT_EQ(triple.size(), 3u);
}

TEST(ShellSetTripleTest, NonSymmetricConstruction) {
    Shell s1(0, O_center, {3.0, 1.0, 0.3}, {0.15, 0.53, 0.44});
    s1.set_shell_index(0);
    Shell s2(0, H1_center, {3.0, 1.0, 0.3}, {0.15, 0.53, 0.44});
    s2.set_shell_index(1);
    Shell aux_s(0, O_center, {9.3}, {1.0});
    aux_s.set_shell_index(0);

    ShellSet set_ab(0, 3);
    set_ab.add_shell(s1);
    set_ab.add_shell(s2);

    ShellSet set_P(0, 1);
    set_P.add_shell(aux_s);

    // non-symmetric: all pairs
    ShellSetTriple triple(set_ab, set_ab, set_P, false);

    // 2 * 2 * 1 = 4 triples
    EXPECT_EQ(triple.size(), 4u);
}

// =============================================================================
// Property Tests
// =============================================================================

TEST(ShellSetTripleTest, NFunctions) {
    Shell s1(0, O_center, {1.0}, {1.0});
    Shell p1(1, H1_center, {1.0}, {1.0});
    Shell d1(2, O_center, {1.0}, {1.0});

    ShellSet set_s(0, 1);
    set_s.add_shell(s1);
    ShellSet set_p(1, 1);
    set_p.add_shell(p1);
    ShellSet set_d(2, 1);
    set_d.add_shell(d1);

    ShellSetTriple triple(set_s, set_p, set_d, false);

    EXPECT_EQ(triple.n_functions_a(), 1);   // s: 1
    EXPECT_EQ(triple.n_functions_b(), 3);   // p: 3
    EXPECT_EQ(triple.n_functions_P(), 6);   // d: 6
    EXPECT_EQ(triple.total_am(), 3);        // 0+1+2
}

TEST(ShellSetTripleTest, AtBoundsChecking) {
    Shell s1(0, O_center, {1.0}, {1.0});
    s1.set_shell_index(0);

    ShellSet set_s(0, 1);
    set_s.add_shell(s1);

    ShellSetTriple triple(set_s, set_s, set_s, false);

    EXPECT_NO_THROW(triple.at(0));
    EXPECT_THROW(triple.at(999), std::out_of_range);
}

TEST(ShellSetTripleTest, EmptyTriple) {
    ShellSetTriple triple;
    EXPECT_TRUE(triple.empty());
    EXPECT_EQ(triple.size(), 0u);
}

TEST(ShellSetTripleTest, TripleIteration) {
    Shell s1(0, O_center, {1.0}, {1.0});
    s1.set_shell_index(0);
    Shell s2(0, H1_center, {1.0}, {1.0});
    s2.set_shell_index(1);

    ShellSet set_a(0, 1);
    set_a.add_shell(s1);
    ShellSet set_b(0, 1);
    set_b.add_shell(s2);
    ShellSet set_P(0, 1);
    set_P.add_shell(s1);

    ShellSetTriple triple(set_a, set_b, set_P, false);

    Size count = 0;
    for (const auto& t : triple) {
        EXPECT_TRUE(t.is_valid());
        ++count;
    }
    EXPECT_EQ(count, triple.size());
}

// =============================================================================
// estimate_triple_cost Tests
// =============================================================================

TEST(ShellSetTripleTest, EstimateCost) {
    Shell s1(0, O_center, {1.0}, {1.0});
    s1.set_shell_index(0);
    Shell p1(1, O_center, {1.0}, {1.0});
    p1.set_shell_index(1);

    ShellSet set_s(0, 1);
    set_s.add_shell(s1);
    ShellSet set_p(1, 1);
    set_p.add_shell(p1);

    ShellSetTriple triple_ss(set_s, set_s, set_s, false);
    ShellSetTriple triple_sp(set_s, set_p, set_s, false);

    Size cost_ss = estimate_triple_cost(triple_ss);
    Size cost_sp = estimate_triple_cost(triple_sp);

    EXPECT_GT(cost_ss, 0u);
    EXPECT_GT(cost_sp, 0u);
    // sp should be more expensive than ss
    EXPECT_GT(cost_sp, cost_ss);
}

// =============================================================================
// generate_shell_set_triples Tests
// =============================================================================

TEST(GenerateTripleTest, EmptyOrbitalBasis) {
    BasisSet empty;
    AuxiliaryBasisSet aux = make_simple_auxiliary();

    auto triples = generate_shell_set_triples(empty, aux, true);
    EXPECT_TRUE(triples.empty());
}

TEST(GenerateTripleTest, EmptyAuxiliaryBasis) {
    BasisSet orbital = make_h2o_orbital();
    AuxiliaryBasisSet empty;

    auto triples = generate_shell_set_triples(orbital, empty, true);
    EXPECT_TRUE(triples.empty());
}

TEST(GenerateTripleTest, NonEmptyResult) {
    BasisSet orbital = make_h2o_orbital();
    AuxiliaryBasisSet aux = make_simple_auxiliary();

    auto triples = generate_shell_set_triples(orbital, aux, true);
    EXPECT_FALSE(triples.empty());

    // Verify each triple has valid properties
    for (const auto& triple : triples) {
        EXPECT_FALSE(triple.empty());
        EXPECT_GE(triple.am_a(), 0);
        EXPECT_GE(triple.am_b(), 0);
        EXPECT_GE(triple.am_P(), 0);
        EXPECT_GT(triple.n_primitives_a(), 0);
        EXPECT_GT(triple.n_primitives_b(), 0);
        EXPECT_GT(triple.n_primitives_P(), 0);
    }
}

TEST(GenerateTripleTest, SymmetricVsNonSymmetric) {
    BasisSet orbital = make_h2o_orbital();
    AuxiliaryBasisSet aux = make_simple_auxiliary();

    auto sym_triples = generate_shell_set_triples(orbital, aux, true);
    auto nonsym_triples = generate_shell_set_triples(orbital, aux, false);

    // Symmetric should have fewer or equal triples
    // Non-symmetric generates all i,j pairs; symmetric generates i<=j
    EXPECT_GE(nonsym_triples.size(), sym_triples.size());
}

TEST(GenerateTripleTest, AuxiliaryPointersReferenceAuxiliaryBasis) {
    BasisSet orbital = make_h2o_orbital();
    AuxiliaryBasisSet aux = make_simple_auxiliary();

    const auto triples = generate_shell_set_triples(orbital, aux, true);
    ASSERT_FALSE(triples.empty());

    for (const auto& batch : triples) {
        for (const auto& triple : batch.triples()) {
            ASSERT_NE(triple.shell_P, nullptr);
            ASSERT_LT(triple.shell_P_idx, aux.n_shells());
            EXPECT_EQ(triple.shell_P, &aux.shell(triple.shell_P_idx));
        }
    }
}

}  // namespace libaccint::testing
