// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2026 Bryce M. Westheimer. All rights reserved.

#include <libaccint/basis/basis_set.hpp>
#include <libaccint/basis/shell.hpp>
#include <libaccint/basis/shell_set.hpp>
#include <libaccint/basis/shell_set_pair.hpp>
#include <libaccint/basis/shell_set_quartet.hpp>
#include <libaccint/utils/error_handling.hpp>
#include <gtest/gtest.h>

#include <algorithm>
#include <vector>

using namespace libaccint;

namespace {

// =============================================================================
// STO-3G H2O Test Data
// =============================================================================

/// Water geometry in bohr
constexpr Point3D O_center{0.0, 0.0, 0.0};
constexpr Point3D H1_center{0.0, 1.43233673, -1.10866041};
constexpr Point3D H2_center{0.0, -1.43233673, -1.10866041};

/// Build STO-3G H2O shells (5 shells, 7 basis functions)
/// Shell order: O 1s, O 2s, O 2p, H1 1s, H2 1s
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

}  // anonymous namespace

// =============================================================================
// Empty BasisSet
// =============================================================================

TEST(BasisSetTest, EmptyConstruction) {
    BasisSet basis;

    EXPECT_EQ(basis.n_shells(), 0u);
    EXPECT_EQ(basis.n_basis_functions(), 0u);
    EXPECT_EQ(basis.max_angular_momentum(), 0);
    EXPECT_EQ(basis.max_n_primitives(), 0);
    EXPECT_TRUE(basis.shells().empty());
    EXPECT_TRUE(basis.shell_sets().empty());
    EXPECT_EQ(basis.n_shell_sets(), 0u);
}

TEST(BasisSetTest, EmptyVectorConstruction) {
    std::vector<Shell> empty;
    BasisSet basis(std::move(empty));

    EXPECT_EQ(basis.n_shells(), 0u);
    EXPECT_EQ(basis.n_basis_functions(), 0u);
    EXPECT_TRUE(basis.shell_sets().empty());
}

// =============================================================================
// STO-3G H2O: Shell Counts and Basis Functions
// =============================================================================

TEST(BasisSetTest, STO3G_H2O_ShellCount) {
    BasisSet basis(make_sto3g_h2o_shells());

    // 5 shells: O 1s, O 2s, O 2p, H1 1s, H2 1s
    EXPECT_EQ(basis.n_shells(), 5u);
}

TEST(BasisSetTest, STO3G_H2O_BasisFunctionCount) {
    BasisSet basis(make_sto3g_h2o_shells());

    // 7 basis functions: 1 + 1 + 3 + 1 + 1
    EXPECT_EQ(basis.n_basis_functions(), 7u);
}

// =============================================================================
// ShellSet Grouping
// =============================================================================

TEST(BasisSetTest, STO3G_H2O_ShellSetGrouping) {
    BasisSet basis(make_sto3g_h2o_shells());

    // 2 ShellSets: {L=0, K=3} with 4 shells, {L=1, K=3} with 1 shell
    EXPECT_EQ(basis.n_shell_sets(), 2u);

    const auto& sets = basis.shell_sets();
    EXPECT_EQ(sets.size(), 2u);

    // Find the S and P shell sets
    const ShellSet* s_set = basis.shell_set(0, 3);
    const ShellSet* p_set = basis.shell_set(1, 3);

    ASSERT_NE(s_set, nullptr);
    ASSERT_NE(p_set, nullptr);

    EXPECT_EQ(s_set->n_shells(), 4u);
    EXPECT_EQ(s_set->angular_momentum(), 0);
    EXPECT_EQ(s_set->n_primitives_per_shell(), 3);

    EXPECT_EQ(p_set->n_shells(), 1u);
    EXPECT_EQ(p_set->angular_momentum(), 1);
    EXPECT_EQ(p_set->n_primitives_per_shell(), 3);
}

// =============================================================================
// Shell Index Assignment
// =============================================================================

TEST(BasisSetTest, STO3G_H2O_ShellIndices) {
    BasisSet basis(make_sto3g_h2o_shells());

    // Shell indices should be 0, 1, 2, 3, 4
    for (Size i = 0; i < basis.n_shells(); ++i) {
        EXPECT_EQ(basis.shell(i).shell_index(), static_cast<Index>(i))
            << "Shell " << i << " has wrong shell_index";
    }
}

TEST(BasisSetTest, STO3G_H2O_FunctionIndices) {
    BasisSet basis(make_sto3g_h2o_shells());

    // Function indices: O 1s -> 0, O 2s -> 1, O 2p -> 2, H1 1s -> 5, H2 1s -> 6
    EXPECT_EQ(basis.shell(0).function_index(), 0);  // O 1s: 1 function
    EXPECT_EQ(basis.shell(1).function_index(), 1);  // O 2s: 1 function
    EXPECT_EQ(basis.shell(2).function_index(), 2);  // O 2p: 3 functions
    EXPECT_EQ(basis.shell(3).function_index(), 5);  // H1 1s: 1 function
    EXPECT_EQ(basis.shell(4).function_index(), 6);  // H2 1s: 1 function
}

// =============================================================================
// Max Angular Momentum and Primitives
// =============================================================================

TEST(BasisSetTest, STO3G_H2O_MaxAngularMomentum) {
    BasisSet basis(make_sto3g_h2o_shells());

    EXPECT_EQ(basis.max_angular_momentum(), 1);
}

TEST(BasisSetTest, STO3G_H2O_MaxNPrimitives) {
    BasisSet basis(make_sto3g_h2o_shells());

    EXPECT_EQ(basis.max_n_primitives(), 3);
}

// =============================================================================
// ShellSet Lookup
// =============================================================================

TEST(BasisSetTest, STO3G_H2O_ShellSetLookup) {
    BasisSet basis(make_sto3g_h2o_shells());

    // Existing key
    const ShellSet* s_set = basis.shell_set(0, 3);
    ASSERT_NE(s_set, nullptr);
    EXPECT_EQ(s_set->angular_momentum(), 0);
    EXPECT_EQ(s_set->n_primitives_per_shell(), 3);

    // Non-existent key
    EXPECT_EQ(basis.shell_set(2, 3), nullptr);  // No D-shells
    EXPECT_EQ(basis.shell_set(0, 1), nullptr);  // No 1-primitive S-shells
}

TEST(BasisSetTest, STO3G_H2O_ShellSetsWithAM) {
    BasisSet basis(make_sto3g_h2o_shells());

    // S-shells: one ShellSet with {L=0, K=3}
    auto s_sets = basis.shell_sets_with_am(0);
    EXPECT_EQ(s_sets.size(), 1u);
    EXPECT_EQ(s_sets[0]->angular_momentum(), 0);
    EXPECT_EQ(s_sets[0]->n_shells(), 4u);

    // P-shells: one ShellSet with {L=1, K=3}
    auto p_sets = basis.shell_sets_with_am(1);
    EXPECT_EQ(p_sets.size(), 1u);
    EXPECT_EQ(p_sets[0]->angular_momentum(), 1);
    EXPECT_EQ(p_sets[0]->n_shells(), 1u);

    // D-shells: none
    auto d_sets = basis.shell_sets_with_am(2);
    EXPECT_TRUE(d_sets.empty());
}

// =============================================================================
// Pair Generation
// =============================================================================

TEST(BasisSetTest, STO3G_H2O_PairCount) {
    BasisSet basis(make_sto3g_h2o_shells());

    // 2 ShellSets -> pairs: (0,0), (0,1), (1,1) -> 3 pairs
    const auto& pairs = basis.shell_set_pairs();
    EXPECT_EQ(pairs.size(), 3u);
}

TEST(BasisSetTest, STO3G_H2O_PairCaching) {
    BasisSet basis(make_sto3g_h2o_shells());

    // Calling shell_set_pairs() twice should return the same vector
    const auto& pairs1 = basis.shell_set_pairs();
    const auto& pairs2 = basis.shell_set_pairs();

    EXPECT_EQ(&pairs1, &pairs2);
    EXPECT_EQ(pairs1.size(), pairs2.size());
}

TEST(BasisSetTest, STO3G_H2O_PairProperties) {
    BasisSet basis(make_sto3g_h2o_shells());

    const auto& pairs = basis.shell_set_pairs();
    ASSERT_EQ(pairs.size(), 3u);

    // Collect all (La, Lb) combinations
    // With 2 shell sets (S with 4 shells, P with 1 shell), pairs are:
    // (S,S), (S,P), (P,P)
    // The order depends on shell_sets_ order; verify the set of pairs is correct
    int ss_count = 0, sp_count = 0, pp_count = 0;
    for (const auto& pair : pairs) {
        int la = pair.La();
        int lb = pair.Lb();
        if (la == 0 && lb == 0) ++ss_count;
        else if ((la == 0 && lb == 1) || (la == 1 && lb == 0)) ++sp_count;
        else if (la == 1 && lb == 1) ++pp_count;
    }
    EXPECT_EQ(ss_count, 1);
    EXPECT_EQ(sp_count, 1);
    EXPECT_EQ(pp_count, 1);
}

// =============================================================================
// Quartet Generation
// =============================================================================

TEST(BasisSetTest, STO3G_H2O_QuartetCount) {
    BasisSet basis(make_sto3g_h2o_shells());

    // 3 pairs -> quartets: (0,0), (0,1), (0,2), (1,1), (1,2), (2,2) -> 6 quartets
    auto quartets = basis.shell_set_quartets();
    EXPECT_EQ(quartets.size(), 6u);
}

TEST(BasisSetTest, STO3G_H2O_QuartetReferencesAreValid) {
    BasisSet basis(make_sto3g_h2o_shells());

    auto quartets = basis.shell_set_quartets();
    ASSERT_EQ(quartets.size(), 6u);

    // Verify each quartet references valid pairs (not dangling)
    const auto& pairs = basis.shell_set_pairs();
    for (const auto& quartet : quartets) {
        // bra and ket should point into the cached pairs vector
        const auto* bra_ptr = &quartet.bra_pair();
        const auto* ket_ptr = &quartet.ket_pair();

        bool bra_found = false;
        bool ket_found = false;
        for (const auto& pair : pairs) {
            if (&pair == bra_ptr) bra_found = true;
            if (&pair == ket_ptr) ket_found = true;
        }
        EXPECT_TRUE(bra_found) << "Quartet bra pair is a dangling reference";
        EXPECT_TRUE(ket_found) << "Quartet ket pair is a dangling reference";
    }
}

TEST(BasisSetTest, STO3G_H2O_QuartetLTotals) {
    BasisSet basis(make_sto3g_h2o_shells());

    auto quartets = basis.shell_set_quartets();
    ASSERT_EQ(quartets.size(), 6u);

    // Verify all quartets have reasonable L_total values
    for (const auto& quartet : quartets) {
        EXPECT_GE(quartet.L_total(), 0);
        EXPECT_LE(quartet.L_total(), 4);  // max is 1+1+1+1 = 4 for P-shells
    }
}

// =============================================================================
// Atom Queries
// =============================================================================

TEST(BasisSetTest, STO3G_H2O_ShellsOnAtom) {
    BasisSet basis(make_sto3g_h2o_shells());

    // Atom 0 (O): 3 shells (O 1s, O 2s, O 2p)
    auto atom0_shells = basis.shells_on_atom(0);
    EXPECT_EQ(atom0_shells.size(), 3u);

    // Atom 1 (H1): 1 shell (H1 1s)
    auto atom1_shells = basis.shells_on_atom(1);
    EXPECT_EQ(atom1_shells.size(), 1u);

    // Atom 2 (H2): 1 shell (H2 1s)
    auto atom2_shells = basis.shells_on_atom(2);
    EXPECT_EQ(atom2_shells.size(), 1u);

    // Non-existent atom: 0 shells
    auto atom3_shells = basis.shells_on_atom(3);
    EXPECT_TRUE(atom3_shells.empty());
}

TEST(BasisSetTest, STO3G_H2O_ShellsOnAtomProperties) {
    BasisSet basis(make_sto3g_h2o_shells());

    auto atom0_shells = basis.shells_on_atom(0);
    ASSERT_EQ(atom0_shells.size(), 3u);

    // Check angular momenta: 0, 0, 1
    EXPECT_EQ(atom0_shells[0]->angular_momentum(), 0);
    EXPECT_EQ(atom0_shells[1]->angular_momentum(), 0);
    EXPECT_EQ(atom0_shells[2]->angular_momentum(), 1);

    // Check centers are all at O_center
    for (const auto* s : atom0_shells) {
        EXPECT_DOUBLE_EQ(s->center().x, O_center.x);
        EXPECT_DOUBLE_EQ(s->center().y, O_center.y);
        EXPECT_DOUBLE_EQ(s->center().z, O_center.z);
    }
}

// =============================================================================
// Out-of-Bounds Access
// =============================================================================

TEST(BasisSetTest, ShellOutOfBoundsThrows) {
    BasisSet basis(make_sto3g_h2o_shells());

    EXPECT_THROW((void)basis.shell(5), InvalidArgumentException);
    EXPECT_THROW((void)basis.shell(100), InvalidArgumentException);
}

TEST(BasisSetTest, EmptyBasisShellOutOfBoundsThrows) {
    BasisSet basis;

    EXPECT_THROW((void)basis.shell(0), InvalidArgumentException);
}

// =============================================================================
// Shells Span Access
// =============================================================================

TEST(BasisSetTest, ShellsSpanAccess) {
    BasisSet basis(make_sto3g_h2o_shells());

    auto span = basis.shells();
    EXPECT_EQ(span.size(), 5u);

    // Verify span points to the same shells
    for (Size i = 0; i < span.size(); ++i) {
        EXPECT_EQ(&span[i], &basis.shell(i));
    }
}

// =============================================================================
// Mixed Angular Momentum / Primitive Count
// =============================================================================

TEST(BasisSetTest, MixedBasisSet) {
    // Create a basis set with multiple (am, K) groups
    std::vector<Shell> shells;

    // S-shell with 3 primitives
    {
        Shell s(0, Point3D(0.0, 0.0, 0.0),
                {3.0, 1.0, 0.3}, {0.15, 0.53, 0.44});
        s.set_atom_index(0);
        shells.push_back(std::move(s));
    }

    // S-shell with 1 primitive (different K)
    {
        Shell s(0, Point3D(0.0, 0.0, 0.0),
                {0.5}, {1.0});
        s.set_atom_index(0);
        shells.push_back(std::move(s));
    }

    // P-shell with 3 primitives
    {
        Shell s(1, Point3D(0.0, 0.0, 0.0),
                {5.0, 1.2, 0.4}, {0.16, 0.61, 0.39});
        s.set_atom_index(0);
        shells.push_back(std::move(s));
    }

    // D-shell with 1 primitive
    {
        Shell s(2, Point3D(1.0, 0.0, 0.0),
                {0.8}, {1.0});
        s.set_atom_index(1);
        shells.push_back(std::move(s));
    }

    BasisSet basis(std::move(shells));

    // 4 shells
    EXPECT_EQ(basis.n_shells(), 4u);

    // Basis functions: 1 + 1 + 3 + 6 = 11
    EXPECT_EQ(basis.n_basis_functions(), 11u);

    // Max AM = 2 (D-shell)
    EXPECT_EQ(basis.max_angular_momentum(), 2);

    // Max primitives = 3
    EXPECT_EQ(basis.max_n_primitives(), 3);

    // 4 ShellSets: {L=0,K=3}, {L=0,K=1}, {L=1,K=3}, {L=2,K=1}
    EXPECT_EQ(basis.n_shell_sets(), 4u);

    // S-shells with am=0: 2 shell sets ({K=3} and {K=1})
    auto s_sets = basis.shell_sets_with_am(0);
    EXPECT_EQ(s_sets.size(), 2u);

    // Pair count: 4 shell sets -> 4*(4+1)/2 = 10 pairs
    const auto& pairs = basis.shell_set_pairs();
    EXPECT_EQ(pairs.size(), 10u);

    // Quartet count: 10 pairs -> 10*(10+1)/2 = 55 quartets
    auto quartets = basis.shell_set_quartets();
    EXPECT_EQ(quartets.size(), 55u);
}

// =============================================================================
// Single Shell BasisSet
// =============================================================================

TEST(BasisSetTest, SingleShell) {
    std::vector<Shell> shells;
    Shell s(0, Point3D(0.0, 0.0, 0.0), {1.0}, {1.0});
    s.set_atom_index(0);
    shells.push_back(std::move(s));

    BasisSet basis(std::move(shells));

    EXPECT_EQ(basis.n_shells(), 1u);
    EXPECT_EQ(basis.n_basis_functions(), 1u);
    EXPECT_EQ(basis.max_angular_momentum(), 0);
    EXPECT_EQ(basis.max_n_primitives(), 1);
    EXPECT_EQ(basis.n_shell_sets(), 1u);

    EXPECT_EQ(basis.shell(0).shell_index(), 0);
    EXPECT_EQ(basis.shell(0).function_index(), 0);

    // 1 shell set -> 1 pair, 1 quartet
    const auto& pairs = basis.shell_set_pairs();
    EXPECT_EQ(pairs.size(), 1u);

    auto quartets = basis.shell_set_quartets();
    EXPECT_EQ(quartets.size(), 1u);
}

// =============================================================================
// Empty Pairs/Quartets for Empty BasisSet
// =============================================================================

TEST(BasisSetTest, EmptyBasisPairsAndQuartets) {
    BasisSet basis;

    const auto& pairs = basis.shell_set_pairs();
    EXPECT_TRUE(pairs.empty());

    auto quartets = basis.shell_set_quartets();
    EXPECT_TRUE(quartets.empty());
}

// =============================================================================
// Atom Index Preservation
// =============================================================================

TEST(BasisSetTest, AtomIndexPreservation) {
    BasisSet basis(make_sto3g_h2o_shells());

    // Atom indices should be preserved from input
    EXPECT_EQ(basis.shell(0).atom_index(), 0);  // O 1s
    EXPECT_EQ(basis.shell(1).atom_index(), 0);  // O 2s
    EXPECT_EQ(basis.shell(2).atom_index(), 0);  // O 2p
    EXPECT_EQ(basis.shell(3).atom_index(), 1);  // H1 1s
    EXPECT_EQ(basis.shell(4).atom_index(), 2);  // H2 1s
}

// =============================================================================
// shell_set_pairs() Canonical Accessor
// =============================================================================

TEST(BasisSetTest, ShellSetPairsAccessor_ReturnsSameCachedVector) {
    BasisSet basis(make_sto3g_h2o_shells());

    // Repeated calls must return a reference to the same underlying vector
    const auto& first = basis.shell_set_pairs();
    const auto& second = basis.shell_set_pairs();

    EXPECT_EQ(&first, &second);
    EXPECT_EQ(first.size(), second.size());
}

TEST(BasisSetTest, ShellSetPairsAccessor_AllocationFreeOnRepeatedCall) {
    BasisSet basis(make_sto3g_h2o_shells());

    // First call populates the cache
    const auto& first = basis.shell_set_pairs();
    // Second call must return the exact same address (no reallocation)
    const auto& second = basis.shell_set_pairs();

    EXPECT_EQ(&first, &second);
    ASSERT_FALSE(first.empty());
    // Element-level address stability (no copy / move of elements)
    EXPECT_EQ(&first[0], &second[0]);
}

TEST(BasisSetTest, ShellSetPairsAccessor_UpperTriangleOrdering) {
    BasisSet basis(make_sto3g_h2o_shells());

    const auto& pairs = basis.shell_set_pairs();
    ASSERT_FALSE(pairs.empty());

    // Build a map from ShellSet pointer to its position in shell_sets()
    const auto sets = basis.shell_sets();
    std::unordered_map<const ShellSet*, Size> set_index;
    for (Size idx = 0; idx < sets.size(); ++idx) {
        set_index[sets[idx]] = idx;
    }

    // Every pair must satisfy i <= j (upper-triangle)
    for (const auto& pair : pairs) {
        const auto* bra_set = &pair.shell_set_a();
        const auto* ket_set = &pair.shell_set_b();
        auto it_bra = set_index.find(bra_set);
        auto it_ket = set_index.find(ket_set);
        ASSERT_NE(it_bra, set_index.end()) << "bra ShellSet not found in basis";
        ASSERT_NE(it_ket, set_index.end()) << "ket ShellSet not found in basis";
        EXPECT_LE(it_bra->second, it_ket->second)
            << "Pair violates upper-triangle ordering: bra index "
            << it_bra->second << " > ket index " << it_ket->second;
    }
}

// =============================================================================
// shell_set_quartets() Canonical Accessor
// =============================================================================

TEST(BasisSetTest, ShellSetQuartetsAccessor_ReturnsSameCachedVector) {
    BasisSet basis(make_sto3g_h2o_shells());

    // Repeated calls must return a reference to the same underlying vector
    const auto& first = basis.shell_set_quartets();
    const auto& second = basis.shell_set_quartets();

    EXPECT_EQ(&first, &second);
    EXPECT_EQ(first.size(), second.size());
}

TEST(BasisSetTest, ShellSetQuartetsAccessor_AllocationFreeOnRepeatedCall) {
    BasisSet basis(make_sto3g_h2o_shells());

    // First call populates the cache
    const auto& first = basis.shell_set_quartets();
    // Second call must return the exact same address (no reallocation)
    const auto& second = basis.shell_set_quartets();

    EXPECT_EQ(&first, &second);
    ASSERT_FALSE(first.empty());
    // Element-level address stability (no copy / move of elements)
    EXPECT_EQ(&first[0], &second[0]);
}

TEST(BasisSetTest, ShellSetQuartetsAccessor_CountMatchesUpperTriangle) {
    BasisSet basis(make_sto3g_h2o_shells());

    const auto& pairs = basis.shell_set_pairs();
    const auto& quartets = basis.shell_set_quartets();

    const Size n_pairs = pairs.size();
    const Size expected = n_pairs * (n_pairs + 1) / 2;
    EXPECT_EQ(quartets.size(), expected);
}

TEST(BasisSetTest, ShellSetQuartetsAccessor_ReferencesValidPairs) {
    BasisSet basis(make_sto3g_h2o_shells());

    const auto& pairs = basis.shell_set_pairs();
    const auto& quartets = basis.shell_set_quartets();

    ASSERT_FALSE(pairs.empty());
    ASSERT_FALSE(quartets.empty());

    // Compute the valid address range of the cached pairs vector
    const auto* pairs_begin = &pairs.front();
    const auto* pairs_end   = &pairs.back() + 1;

    for (const auto& quartet : quartets) {
        const auto* bra_ptr = &quartet.bra_pair();
        const auto* ket_ptr = &quartet.ket_pair();

        EXPECT_GE(bra_ptr, pairs_begin)
            << "Quartet bra pair pointer is before cached pairs range";
        EXPECT_LT(bra_ptr, pairs_end)
            << "Quartet bra pair pointer is past cached pairs range";
        EXPECT_GE(ket_ptr, pairs_begin)
            << "Quartet ket pair pointer is before cached pairs range";
        EXPECT_LT(ket_ptr, pairs_end)
            << "Quartet ket pair pointer is past cached pairs range";
    }
}
