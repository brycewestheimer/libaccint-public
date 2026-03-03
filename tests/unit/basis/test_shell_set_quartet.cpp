// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2026 Bryce M. Westheimer. All rights reserved.

#include <libaccint/basis/shell_set_quartet.hpp>
#include <libaccint/basis/shell_set_pair.hpp>
#include <libaccint/basis/shell_set.hpp>
#include <libaccint/basis/shell.hpp>
#include <gtest/gtest.h>

#include <functional>
#include <vector>

using namespace libaccint;

namespace {

/// Helper: create an S-shell (L=0) with given center and K primitives
Shell make_s_shell(Point3D center, Size k = 3) {
    std::vector<Real> exponents;
    std::vector<Real> coefficients;
    exponents.reserve(k);
    coefficients.reserve(k);
    for (Size i = 0; i < k; ++i) {
        exponents.push_back(3.0 / static_cast<Real>(i + 1));
        coefficients.push_back(1.0 / static_cast<Real>(k));
    }
    return Shell(AngularMomentum::S, center, std::move(exponents), std::move(coefficients));
}

/// Helper: create a P-shell (L=1) with given center and K primitives
Shell make_p_shell(Point3D center, Size k = 3) {
    std::vector<Real> exponents;
    std::vector<Real> coefficients;
    exponents.reserve(k);
    coefficients.reserve(k);
    for (Size i = 0; i < k; ++i) {
        exponents.push_back(2.0 / static_cast<Real>(i + 1));
        coefficients.push_back(1.0 / static_cast<Real>(k));
    }
    return Shell(AngularMomentum::P, center, std::move(exponents), std::move(coefficients));
}

/// Helper: create a D-shell (L=2) with given center and K primitives
Shell make_d_shell(Point3D center, Size k = 1) {
    std::vector<Real> exponents;
    std::vector<Real> coefficients;
    exponents.reserve(k);
    coefficients.reserve(k);
    for (Size i = 0; i < k; ++i) {
        exponents.push_back(1.5 / static_cast<Real>(i + 1));
        coefficients.push_back(1.0 / static_cast<Real>(k));
    }
    return Shell(AngularMomentum::D, center, std::move(exponents), std::move(coefficients));
}

/// Helper: create a ShellSet from a vector of Shells
ShellSet make_shell_set(const std::vector<Shell>& shells) {
    std::vector<std::reference_wrapper<const Shell>> refs;
    refs.reserve(shells.size());
    for (const auto& s : shells) {
        refs.push_back(std::cref(s));
    }
    return ShellSet(refs);
}

}  // anonymous namespace

// =============================================================================
// Construction Tests
// =============================================================================

TEST(ShellSetQuartetTest, ConstructFromTwoShellSetPairs) {
    Shell s1 = make_s_shell(Point3D(0.0, 0.0, 0.0));
    Shell s2 = make_s_shell(Point3D(1.0, 0.0, 0.0));
    Shell s3 = make_s_shell(Point3D(2.0, 0.0, 0.0));
    Shell s4 = make_s_shell(Point3D(3.0, 0.0, 0.0));

    std::vector<Shell> shells_a = {s1};
    std::vector<Shell> shells_b = {s2};
    std::vector<Shell> shells_c = {s3};
    std::vector<Shell> shells_d = {s4};

    ShellSet set_a = make_shell_set(shells_a);
    ShellSet set_b = make_shell_set(shells_b);
    ShellSet set_c = make_shell_set(shells_c);
    ShellSet set_d = make_shell_set(shells_d);

    ShellSetPair bra(set_a, set_b);
    ShellSetPair ket(set_c, set_d);

    ShellSetQuartet quartet(bra, ket);

    // Should construct successfully
    EXPECT_EQ(quartet.La(), 0);
    EXPECT_EQ(quartet.Lb(), 0);
    EXPECT_EQ(quartet.Lc(), 0);
    EXPECT_EQ(quartet.Ld(), 0);
}

TEST(ShellSetQuartetTest, ConstructWithSamePair) {
    Shell s1 = make_s_shell(Point3D(0.0, 0.0, 0.0));
    Shell s2 = make_s_shell(Point3D(1.0, 0.0, 0.0));

    std::vector<Shell> shells_a = {s1};
    std::vector<Shell> shells_b = {s2};

    ShellSet set_a = make_shell_set(shells_a);
    ShellSet set_b = make_shell_set(shells_b);

    ShellSetPair pair(set_a, set_b);

    // Should be able to use the same pair for both bra and ket
    ShellSetQuartet quartet(pair, pair);

    EXPECT_EQ(quartet.La(), 0);
    EXPECT_EQ(quartet.Lb(), 0);
    EXPECT_EQ(quartet.Lc(), 0);
    EXPECT_EQ(quartet.Ld(), 0);
    EXPECT_EQ(quartet.n_quartets(), 1u);  // 1 * 1 = 1
    EXPECT_EQ(&quartet.bra_pair(), &quartet.ket_pair());
}

// =============================================================================
// Accessor Tests
// =============================================================================

TEST(ShellSetQuartetTest, PairAccessors) {
    Shell s1 = make_s_shell(Point3D(0.0, 0.0, 0.0));
    Shell p1 = make_p_shell(Point3D(1.0, 0.0, 0.0));
    Shell s2 = make_s_shell(Point3D(2.0, 0.0, 0.0));
    Shell p2 = make_p_shell(Point3D(3.0, 0.0, 0.0));

    std::vector<Shell> shells_a = {s1};
    std::vector<Shell> shells_b = {p1};
    std::vector<Shell> shells_c = {s2};
    std::vector<Shell> shells_d = {p2};

    ShellSet set_a = make_shell_set(shells_a);
    ShellSet set_b = make_shell_set(shells_b);
    ShellSet set_c = make_shell_set(shells_c);
    ShellSet set_d = make_shell_set(shells_d);

    ShellSetPair bra(set_a, set_b);
    ShellSetPair ket(set_c, set_d);

    ShellSetQuartet quartet(bra, ket);

    // Accessors should return references to the original ShellSetPairs
    EXPECT_EQ(&quartet.bra_pair(), &bra);
    EXPECT_EQ(&quartet.ket_pair(), &ket);
}

TEST(ShellSetQuartetTest, AngularMomentumAccessors) {
    Shell s1 = make_s_shell(Point3D(0.0, 0.0, 0.0));
    Shell p1 = make_p_shell(Point3D(1.0, 0.0, 0.0));
    Shell d1 = make_d_shell(Point3D(2.0, 0.0, 0.0));
    Shell s2 = make_s_shell(Point3D(3.0, 0.0, 0.0));

    std::vector<Shell> shells_s = {s1};
    std::vector<Shell> shells_p = {p1};
    std::vector<Shell> shells_d = {d1};
    std::vector<Shell> shells_s2 = {s2};

    ShellSet set_s = make_shell_set(shells_s);
    ShellSet set_p = make_shell_set(shells_p);
    ShellSet set_d = make_shell_set(shells_d);
    ShellSet set_s2 = make_shell_set(shells_s2);

    // Test (SS|SS) quartet
    {
        ShellSetPair bra(set_s, set_s2);
        ShellSetPair ket(set_s, set_s2);
        ShellSetQuartet quartet(bra, ket);

        EXPECT_EQ(quartet.La(), 0);
        EXPECT_EQ(quartet.Lb(), 0);
        EXPECT_EQ(quartet.Lc(), 0);
        EXPECT_EQ(quartet.Ld(), 0);
        EXPECT_EQ(quartet.L_total(), 0);
    }

    // Test (SP|PD) quartet
    {
        ShellSetPair bra(set_s, set_p);
        ShellSetPair ket(set_p, set_d);
        ShellSetQuartet quartet(bra, ket);

        EXPECT_EQ(quartet.La(), 0);
        EXPECT_EQ(quartet.Lb(), 1);
        EXPECT_EQ(quartet.Lc(), 1);
        EXPECT_EQ(quartet.Ld(), 2);
        EXPECT_EQ(quartet.L_total(), 4);
    }

    // Test (PD|DS) quartet
    {
        ShellSetPair bra(set_p, set_d);
        ShellSetPair ket(set_d, set_s);
        ShellSetQuartet quartet(bra, ket);

        EXPECT_EQ(quartet.La(), 1);
        EXPECT_EQ(quartet.Lb(), 2);
        EXPECT_EQ(quartet.Lc(), 2);
        EXPECT_EQ(quartet.Ld(), 0);
        EXPECT_EQ(quartet.L_total(), 5);
    }

    // Test (DD|DD) quartet
    {
        ShellSetPair bra(set_d, set_d);
        ShellSetPair ket(set_d, set_d);
        ShellSetQuartet quartet(bra, ket);

        EXPECT_EQ(quartet.La(), 2);
        EXPECT_EQ(quartet.Lb(), 2);
        EXPECT_EQ(quartet.Lc(), 2);
        EXPECT_EQ(quartet.Ld(), 2);
        EXPECT_EQ(quartet.L_total(), 8);
    }
}

// =============================================================================
// n_quartets Tests
// =============================================================================

TEST(ShellSetQuartetTest, NQuartetsSingleShells) {
    Shell s1 = make_s_shell(Point3D(0.0, 0.0, 0.0));
    Shell s2 = make_s_shell(Point3D(1.0, 0.0, 0.0));
    Shell s3 = make_s_shell(Point3D(2.0, 0.0, 0.0));
    Shell s4 = make_s_shell(Point3D(3.0, 0.0, 0.0));

    std::vector<Shell> shells_a = {s1};
    std::vector<Shell> shells_b = {s2};
    std::vector<Shell> shells_c = {s3};
    std::vector<Shell> shells_d = {s4};

    ShellSet set_a = make_shell_set(shells_a);
    ShellSet set_b = make_shell_set(shells_b);
    ShellSet set_c = make_shell_set(shells_c);
    ShellSet set_d = make_shell_set(shells_d);

    ShellSetPair bra(set_a, set_b);
    ShellSetPair ket(set_c, set_d);

    ShellSetQuartet quartet(bra, ket);

    // 1 * 1 = 1 pair in bra, 1 * 1 = 1 pair in ket
    // n_quartets = 1 * 1 = 1
    EXPECT_EQ(quartet.n_quartets(), 1u);
}

TEST(ShellSetQuartetTest, NQuartetsMultipleShells) {
    // Create 2 S-shells in set A
    Shell s1 = make_s_shell(Point3D(0.0, 0.0, 0.0));
    Shell s2 = make_s_shell(Point3D(1.0, 0.0, 0.0));

    // Create 3 S-shells in set B
    Shell s3 = make_s_shell(Point3D(2.0, 0.0, 0.0));
    Shell s4 = make_s_shell(Point3D(3.0, 0.0, 0.0));
    Shell s5 = make_s_shell(Point3D(4.0, 0.0, 0.0));

    // Create 4 S-shells in set C
    Shell s6 = make_s_shell(Point3D(5.0, 0.0, 0.0));
    Shell s7 = make_s_shell(Point3D(6.0, 0.0, 0.0));
    Shell s8 = make_s_shell(Point3D(7.0, 0.0, 0.0));
    Shell s9 = make_s_shell(Point3D(8.0, 0.0, 0.0));

    // Create 5 S-shells in set D
    Shell s10 = make_s_shell(Point3D(9.0, 0.0, 0.0));
    Shell s11 = make_s_shell(Point3D(10.0, 0.0, 0.0));
    Shell s12 = make_s_shell(Point3D(11.0, 0.0, 0.0));
    Shell s13 = make_s_shell(Point3D(12.0, 0.0, 0.0));
    Shell s14 = make_s_shell(Point3D(13.0, 0.0, 0.0));

    std::vector<Shell> shells_a = {s1, s2};
    std::vector<Shell> shells_b = {s3, s4, s5};
    std::vector<Shell> shells_c = {s6, s7, s8, s9};
    std::vector<Shell> shells_d = {s10, s11, s12, s13, s14};

    ShellSet set_a = make_shell_set(shells_a);
    ShellSet set_b = make_shell_set(shells_b);
    ShellSet set_c = make_shell_set(shells_c);
    ShellSet set_d = make_shell_set(shells_d);

    ShellSetPair bra(set_a, set_b);
    ShellSetPair ket(set_c, set_d);

    ShellSetQuartet quartet(bra, ket);

    // bra has 2 * 3 = 6 pairs
    // ket has 4 * 5 = 20 pairs
    // n_quartets = 6 * 20 = 120
    EXPECT_EQ(quartet.n_quartets(), 120u);
}

TEST(ShellSetQuartetTest, NQuartetsWithReusedPair) {
    // Create shells
    Shell s1 = make_s_shell(Point3D(0.0, 0.0, 0.0));
    Shell s2 = make_s_shell(Point3D(1.0, 0.0, 0.0));
    Shell s3 = make_s_shell(Point3D(2.0, 0.0, 0.0));

    std::vector<Shell> shells_a = {s1, s2};
    std::vector<Shell> shells_b = {s3};

    ShellSet set_a = make_shell_set(shells_a);
    ShellSet set_b = make_shell_set(shells_b);

    ShellSetPair pair(set_a, set_b);

    // Use same pair for both bra and ket
    ShellSetQuartet quartet(pair, pair);

    // pair has 2 * 1 = 2 pairs
    // n_quartets = 2 * 2 = 4
    EXPECT_EQ(quartet.n_quartets(), 4u);
}

// =============================================================================
// Schwarz Bound Tests (Phase 2 - Full Implementation)
// =============================================================================

TEST(ShellSetQuartetTest, SchwarzBoundPositive) {
    // Test that Schwarz bounds are positive for all shell quartet combinations
    Shell s1 = make_s_shell(Point3D(0.0, 0.0, 0.0));
    Shell s2 = make_s_shell(Point3D(1.0, 0.0, 0.0));
    Shell s3 = make_s_shell(Point3D(2.0, 0.0, 0.0));
    Shell s4 = make_s_shell(Point3D(3.0, 0.0, 0.0));

    std::vector<Shell> shells_a = {s1};
    std::vector<Shell> shells_b = {s2};
    std::vector<Shell> shells_c = {s3};
    std::vector<Shell> shells_d = {s4};

    ShellSet set_a = make_shell_set(shells_a);
    ShellSet set_b = make_shell_set(shells_b);
    ShellSet set_c = make_shell_set(shells_c);
    ShellSet set_d = make_shell_set(shells_d);

    ShellSetPair bra(set_a, set_b);
    ShellSetPair ket(set_c, set_d);

    ShellSetQuartet quartet(bra, ket);

    // Schwarz bound = Q_ab * Q_cd should be positive
    Real bound = quartet.schwarz_bound();
    EXPECT_GT(bound, 0.0) << "Schwarz bound should be positive";

    // Verify it equals the product of pair bounds
    EXPECT_DOUBLE_EQ(bound, bra.schwarz_bound() * ket.schwarz_bound());
}

TEST(ShellSetQuartetTest, SchwarzBoundReasonableValues) {
    // Test that Schwarz bounds have reasonable magnitude for various AM combinations
    Shell s1 = make_s_shell(Point3D(0.0, 0.0, 0.0));
    Shell p1 = make_p_shell(Point3D(1.0, 0.0, 0.0));
    Shell d1 = make_d_shell(Point3D(2.0, 0.0, 0.0));
    Shell s2 = make_s_shell(Point3D(3.0, 0.0, 0.0));

    std::vector<Shell> shells_s = {s1};
    std::vector<Shell> shells_p = {p1};
    std::vector<Shell> shells_d = {d1};
    std::vector<Shell> shells_s2 = {s2};

    ShellSet set_s = make_shell_set(shells_s);
    ShellSet set_p = make_shell_set(shells_p);
    ShellSet set_d = make_shell_set(shells_d);
    ShellSet set_s2 = make_shell_set(shells_s2);

    // Test various combinations - all should have positive bounds
    {
        ShellSetPair bra(set_s, set_s2);
        ShellSetPair ket(set_s, set_s2);
        Real bound = ShellSetQuartet(bra, ket).schwarz_bound();
        EXPECT_GT(bound, 0.0) << "SS|SS bound should be positive";
    }

    {
        ShellSetPair bra(set_s, set_p);
        ShellSetPair ket(set_p, set_d);
        Real bound = ShellSetQuartet(bra, ket).schwarz_bound();
        EXPECT_GT(bound, 0.0) << "SP|PD bound should be positive";
    }

    {
        ShellSetPair bra(set_d, set_d);
        ShellSetPair ket(set_d, set_d);
        Real bound = ShellSetQuartet(bra, ket).schwarz_bound();
        EXPECT_GT(bound, 0.0) << "DD|DD bound should be positive";
    }
}

// =============================================================================
// Different Angular Momentum Combinations
// =============================================================================

TEST(ShellSetQuartetTest, SSSSQuartet) {
    Shell s1 = make_s_shell(Point3D(0.0, 0.0, 0.0));
    Shell s2 = make_s_shell(Point3D(1.0, 0.0, 0.0));
    Shell s3 = make_s_shell(Point3D(2.0, 0.0, 0.0));
    Shell s4 = make_s_shell(Point3D(3.0, 0.0, 0.0));

    std::vector<Shell> shells_a = {s1};
    std::vector<Shell> shells_b = {s2};
    std::vector<Shell> shells_c = {s3};
    std::vector<Shell> shells_d = {s4};

    ShellSet set_a = make_shell_set(shells_a);
    ShellSet set_b = make_shell_set(shells_b);
    ShellSet set_c = make_shell_set(shells_c);
    ShellSet set_d = make_shell_set(shells_d);

    ShellSetPair bra(set_a, set_b);
    ShellSetPair ket(set_c, set_d);

    ShellSetQuartet quartet(bra, ket);

    EXPECT_EQ(quartet.La(), 0);
    EXPECT_EQ(quartet.Lb(), 0);
    EXPECT_EQ(quartet.Lc(), 0);
    EXPECT_EQ(quartet.Ld(), 0);
    EXPECT_EQ(quartet.L_total(), 0);
    EXPECT_EQ(quartet.n_quartets(), 1u);
}

TEST(ShellSetQuartetTest, SPPSQuartet) {
    Shell s1 = make_s_shell(Point3D(0.0, 0.0, 0.0));
    Shell p1 = make_p_shell(Point3D(1.0, 0.0, 0.0));
    Shell p2 = make_p_shell(Point3D(2.0, 0.0, 0.0));
    Shell s2 = make_s_shell(Point3D(3.0, 0.0, 0.0));

    std::vector<Shell> shells_s1 = {s1};
    std::vector<Shell> shells_p1 = {p1};
    std::vector<Shell> shells_p2 = {p2};
    std::vector<Shell> shells_s2 = {s2};

    ShellSet set_s1 = make_shell_set(shells_s1);
    ShellSet set_p1 = make_shell_set(shells_p1);
    ShellSet set_p2 = make_shell_set(shells_p2);
    ShellSet set_s2 = make_shell_set(shells_s2);

    ShellSetPair bra(set_s1, set_p1);
    ShellSetPair ket(set_p2, set_s2);

    ShellSetQuartet quartet(bra, ket);

    EXPECT_EQ(quartet.La(), 0);
    EXPECT_EQ(quartet.Lb(), 1);
    EXPECT_EQ(quartet.Lc(), 1);
    EXPECT_EQ(quartet.Ld(), 0);
    EXPECT_EQ(quartet.L_total(), 2);
    EXPECT_EQ(quartet.n_quartets(), 1u);
}

TEST(ShellSetQuartetTest, PPPPQuartet) {
    Shell p1 = make_p_shell(Point3D(0.0, 0.0, 0.0));
    Shell p2 = make_p_shell(Point3D(1.0, 0.0, 0.0));
    Shell p3 = make_p_shell(Point3D(2.0, 0.0, 0.0));
    Shell p4 = make_p_shell(Point3D(3.0, 0.0, 0.0));
    Shell p5 = make_p_shell(Point3D(4.0, 0.0, 0.0));

    std::vector<Shell> shells_ab = {p1, p2};
    std::vector<Shell> shells_cd = {p3, p4, p5};

    ShellSet set_ab = make_shell_set(shells_ab);
    ShellSet set_cd = make_shell_set(shells_cd);

    ShellSetPair bra(set_ab, set_ab);
    ShellSetPair ket(set_cd, set_cd);

    ShellSetQuartet quartet(bra, ket);

    EXPECT_EQ(quartet.La(), 1);
    EXPECT_EQ(quartet.Lb(), 1);
    EXPECT_EQ(quartet.Lc(), 1);
    EXPECT_EQ(quartet.Ld(), 1);
    EXPECT_EQ(quartet.L_total(), 4);
    EXPECT_EQ(quartet.n_quartets(), 36u);  // (2*2) * (3*3) = 4 * 9 = 36
}

TEST(ShellSetQuartetTest, SDDPQuartet) {
    Shell s1 = make_s_shell(Point3D(0.0, 0.0, 0.0));
    Shell d1 = make_d_shell(Point3D(1.0, 0.0, 0.0));
    Shell d2 = make_d_shell(Point3D(2.0, 0.0, 0.0));
    Shell p1 = make_p_shell(Point3D(3.0, 0.0, 0.0));

    std::vector<Shell> shells_s = {s1};
    std::vector<Shell> shells_d1 = {d1};
    std::vector<Shell> shells_d2 = {d2};
    std::vector<Shell> shells_p = {p1};

    ShellSet set_s = make_shell_set(shells_s);
    ShellSet set_d1 = make_shell_set(shells_d1);
    ShellSet set_d2 = make_shell_set(shells_d2);
    ShellSet set_p = make_shell_set(shells_p);

    ShellSetPair bra(set_s, set_d1);
    ShellSetPair ket(set_d2, set_p);

    ShellSetQuartet quartet(bra, ket);

    EXPECT_EQ(quartet.La(), 0);
    EXPECT_EQ(quartet.Lb(), 2);
    EXPECT_EQ(quartet.Lc(), 2);
    EXPECT_EQ(quartet.Ld(), 1);
    EXPECT_EQ(quartet.L_total(), 5);
    EXPECT_EQ(quartet.n_quartets(), 1u);
}

TEST(ShellSetQuartetTest, DDDDQuartet) {
    Shell d1 = make_d_shell(Point3D(0.0, 0.0, 0.0));
    Shell d2 = make_d_shell(Point3D(1.0, 0.0, 0.0));
    Shell d3 = make_d_shell(Point3D(2.0, 0.0, 0.0));
    Shell d4 = make_d_shell(Point3D(3.0, 0.0, 0.0));

    std::vector<Shell> shells_a = {d1};
    std::vector<Shell> shells_b = {d2};
    std::vector<Shell> shells_c = {d3};
    std::vector<Shell> shells_d = {d4};

    ShellSet set_a = make_shell_set(shells_a);
    ShellSet set_b = make_shell_set(shells_b);
    ShellSet set_c = make_shell_set(shells_c);
    ShellSet set_d = make_shell_set(shells_d);

    ShellSetPair bra(set_a, set_b);
    ShellSetPair ket(set_c, set_d);

    ShellSetQuartet quartet(bra, ket);

    EXPECT_EQ(quartet.La(), 2);
    EXPECT_EQ(quartet.Lb(), 2);
    EXPECT_EQ(quartet.Lc(), 2);
    EXPECT_EQ(quartet.Ld(), 2);
    EXPECT_EQ(quartet.L_total(), 8);
    EXPECT_EQ(quartet.n_quartets(), 1u);
}

// =============================================================================
// Large Sets
// =============================================================================

TEST(ShellSetQuartetTest, LargeShellCounts) {
    // Test with larger shell counts to ensure n_quartets scales correctly
    constexpr Size n_shells_a = 10;
    constexpr Size n_shells_b = 8;
    constexpr Size n_shells_c = 12;
    constexpr Size n_shells_d = 6;

    std::vector<Shell> shells_a;
    for (Size i = 0; i < n_shells_a; ++i) {
        shells_a.push_back(make_s_shell(Point3D(static_cast<Real>(i), 0.0, 0.0)));
    }

    std::vector<Shell> shells_b;
    for (Size i = 0; i < n_shells_b; ++i) {
        shells_b.push_back(make_s_shell(Point3D(0.0, static_cast<Real>(i), 0.0)));
    }

    std::vector<Shell> shells_c;
    for (Size i = 0; i < n_shells_c; ++i) {
        shells_c.push_back(make_s_shell(Point3D(0.0, 0.0, static_cast<Real>(i))));
    }

    std::vector<Shell> shells_d;
    for (Size i = 0; i < n_shells_d; ++i) {
        shells_d.push_back(make_s_shell(Point3D(static_cast<Real>(i), static_cast<Real>(i), 0.0)));
    }

    ShellSet set_a = make_shell_set(shells_a);
    ShellSet set_b = make_shell_set(shells_b);
    ShellSet set_c = make_shell_set(shells_c);
    ShellSet set_d = make_shell_set(shells_d);

    ShellSetPair bra(set_a, set_b);
    ShellSetPair ket(set_c, set_d);

    ShellSetQuartet quartet(bra, ket);

    // bra has 10 * 8 = 80 pairs
    // ket has 12 * 6 = 72 pairs
    // n_quartets = 80 * 72 = 5760
    EXPECT_EQ(quartet.n_quartets(), 5760u);
    EXPECT_EQ(quartet.La(), 0);
    EXPECT_EQ(quartet.Lb(), 0);
    EXPECT_EQ(quartet.Lc(), 0);
    EXPECT_EQ(quartet.Ld(), 0);
    EXPECT_EQ(quartet.L_total(), 0);
}

// =============================================================================
// Lightweight Class Tests
// =============================================================================

TEST(ShellSetQuartetTest, IsLightweight) {
    // Verify that ShellSetQuartet is indeed lightweight (just two pointers)
    // This test documents the expected memory footprint
    constexpr Size expected_size = 2 * sizeof(const ShellSetPair*);

    EXPECT_EQ(sizeof(ShellSetQuartet), expected_size)
        << "ShellSetQuartet should be lightweight (two pointers only)";
}

TEST(ShellSetQuartetTest, CopyConstructible) {
    Shell s1 = make_s_shell(Point3D(0.0, 0.0, 0.0));
    Shell s2 = make_s_shell(Point3D(1.0, 0.0, 0.0));
    Shell s3 = make_s_shell(Point3D(2.0, 0.0, 0.0));
    Shell s4 = make_s_shell(Point3D(3.0, 0.0, 0.0));

    std::vector<Shell> shells_a = {s1};
    std::vector<Shell> shells_b = {s2};
    std::vector<Shell> shells_c = {s3};
    std::vector<Shell> shells_d = {s4};

    ShellSet set_a = make_shell_set(shells_a);
    ShellSet set_b = make_shell_set(shells_b);
    ShellSet set_c = make_shell_set(shells_c);
    ShellSet set_d = make_shell_set(shells_d);

    ShellSetPair bra(set_a, set_b);
    ShellSetPair ket(set_c, set_d);

    ShellSetQuartet quartet1(bra, ket);
    ShellSetQuartet quartet2 = quartet1;  // Copy construct

    // Both should refer to the same underlying ShellSetPairs
    EXPECT_EQ(&quartet1.bra_pair(), &quartet2.bra_pair());
    EXPECT_EQ(&quartet1.ket_pair(), &quartet2.ket_pair());
    EXPECT_EQ(quartet1.n_quartets(), quartet2.n_quartets());
    EXPECT_EQ(quartet1.La(), quartet2.La());
    EXPECT_EQ(quartet1.Lb(), quartet2.Lb());
    EXPECT_EQ(quartet1.Lc(), quartet2.Lc());
    EXPECT_EQ(quartet1.Ld(), quartet2.Ld());
}
