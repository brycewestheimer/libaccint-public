// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2026 Bryce M. Westheimer. All rights reserved.

#include <libaccint/basis/shell_set_pair.hpp>
#include <libaccint/basis/shell_set.hpp>
#include <libaccint/basis/shell.hpp>
#include <libaccint/math/gaussian_product.hpp>
#include <gtest/gtest.h>

#include <cmath>
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

TEST(ShellSetPairTest, ConstructFromTwoShellSets) {
    Shell s1 = make_s_shell(Point3D(0.0, 0.0, 0.0));
    Shell s2 = make_s_shell(Point3D(1.0, 0.0, 0.0));

    std::vector<Shell> shells_a = {s1};
    std::vector<Shell> shells_b = {s2};

    ShellSet set_a = make_shell_set(shells_a);
    ShellSet set_b = make_shell_set(shells_b);

    ShellSetPair pair(set_a, set_b);

    // Should construct successfully
    EXPECT_EQ(pair.La(), 0);
    EXPECT_EQ(pair.Lb(), 0);
}

TEST(ShellSetPairTest, ConstructWithSameShellSet) {
    Shell s1 = make_s_shell(Point3D(0.0, 0.0, 0.0));
    Shell s2 = make_s_shell(Point3D(1.0, 0.0, 0.0));

    std::vector<Shell> shells = {s1, s2};
    ShellSet set = make_shell_set(shells);

    // Should be able to pair a ShellSet with itself
    ShellSetPair pair(set, set);

    EXPECT_EQ(pair.La(), 0);
    EXPECT_EQ(pair.Lb(), 0);
    EXPECT_EQ(pair.n_pairs(), 4u);  // 2 * 2 = 4
    EXPECT_EQ(&pair.shell_set_a(), &pair.shell_set_b());
}

// =============================================================================
// Accessor Tests
// =============================================================================

TEST(ShellSetPairTest, ShellSetAccessors) {
    Shell s1 = make_s_shell(Point3D(0.0, 0.0, 0.0));
    Shell p1 = make_p_shell(Point3D(1.0, 0.0, 0.0));

    std::vector<Shell> shells_a = {s1};
    std::vector<Shell> shells_b = {p1};

    ShellSet set_a = make_shell_set(shells_a);
    ShellSet set_b = make_shell_set(shells_b);

    ShellSetPair pair(set_a, set_b);

    // Accessors should return references to the original ShellSets
    EXPECT_EQ(&pair.shell_set_a(), &set_a);
    EXPECT_EQ(&pair.shell_set_b(), &set_b);
}

TEST(ShellSetPairTest, AngularMomentumAccessors) {
    Shell s1 = make_s_shell(Point3D(0.0, 0.0, 0.0));
    Shell p1 = make_p_shell(Point3D(1.0, 0.0, 0.0));
    Shell d1 = make_d_shell(Point3D(2.0, 0.0, 0.0));

    std::vector<Shell> shells_s = {s1};
    std::vector<Shell> shells_p = {p1};
    std::vector<Shell> shells_d = {d1};

    ShellSet set_s = make_shell_set(shells_s);
    ShellSet set_p = make_shell_set(shells_p);
    ShellSet set_d = make_shell_set(shells_d);

    // Test SP pair
    {
        ShellSetPair pair(set_s, set_p);
        EXPECT_EQ(pair.La(), 0);
        EXPECT_EQ(pair.Lb(), 1);
        EXPECT_EQ(pair.L_total(), 1);
    }

    // Test PS pair
    {
        ShellSetPair pair(set_p, set_s);
        EXPECT_EQ(pair.La(), 1);
        EXPECT_EQ(pair.Lb(), 0);
        EXPECT_EQ(pair.L_total(), 1);
    }

    // Test PD pair
    {
        ShellSetPair pair(set_p, set_d);
        EXPECT_EQ(pair.La(), 1);
        EXPECT_EQ(pair.Lb(), 2);
        EXPECT_EQ(pair.L_total(), 3);
    }

    // Test DD pair
    {
        ShellSetPair pair(set_d, set_d);
        EXPECT_EQ(pair.La(), 2);
        EXPECT_EQ(pair.Lb(), 2);
        EXPECT_EQ(pair.L_total(), 4);
    }
}

// =============================================================================
// n_pairs Tests
// =============================================================================

TEST(ShellSetPairTest, NPairsSingleShells) {
    Shell s1 = make_s_shell(Point3D(0.0, 0.0, 0.0));
    Shell s2 = make_s_shell(Point3D(1.0, 0.0, 0.0));

    std::vector<Shell> shells_a = {s1};
    std::vector<Shell> shells_b = {s2};

    ShellSet set_a = make_shell_set(shells_a);
    ShellSet set_b = make_shell_set(shells_b);

    ShellSetPair pair(set_a, set_b);

    EXPECT_EQ(pair.n_pairs(), 1u);  // 1 * 1 = 1
}

TEST(ShellSetPairTest, NPairsMultipleShells) {
    // Create 3 S-shells in set A
    Shell s1 = make_s_shell(Point3D(0.0, 0.0, 0.0));
    Shell s2 = make_s_shell(Point3D(1.0, 0.0, 0.0));
    Shell s3 = make_s_shell(Point3D(2.0, 0.0, 0.0));

    // Create 4 S-shells in set B
    Shell s4 = make_s_shell(Point3D(3.0, 0.0, 0.0));
    Shell s5 = make_s_shell(Point3D(4.0, 0.0, 0.0));
    Shell s6 = make_s_shell(Point3D(5.0, 0.0, 0.0));
    Shell s7 = make_s_shell(Point3D(6.0, 0.0, 0.0));

    std::vector<Shell> shells_a = {s1, s2, s3};
    std::vector<Shell> shells_b = {s4, s5, s6, s7};

    ShellSet set_a = make_shell_set(shells_a);
    ShellSet set_b = make_shell_set(shells_b);

    ShellSetPair pair(set_a, set_b);

    EXPECT_EQ(pair.n_pairs(), 12u);  // 3 * 4 = 12
}

TEST(ShellSetPairTest, NPairsAsymmetric) {
    // Test that n_pairs is correctly computed even with asymmetric shell counts

    // 5 shells in A, 2 shells in B
    std::vector<Shell> shells_a;
    for (Size i = 0; i < 5; ++i) {
        shells_a.push_back(make_s_shell(Point3D(static_cast<Real>(i), 0.0, 0.0)));
    }

    std::vector<Shell> shells_b;
    for (Size i = 0; i < 2; ++i) {
        shells_b.push_back(make_s_shell(Point3D(0.0, static_cast<Real>(i), 0.0)));
    }

    ShellSet set_a = make_shell_set(shells_a);
    ShellSet set_b = make_shell_set(shells_b);

    ShellSetPair pair_ab(set_a, set_b);
    EXPECT_EQ(pair_ab.n_pairs(), 10u);  // 5 * 2 = 10

    ShellSetPair pair_ba(set_b, set_a);
    EXPECT_EQ(pair_ba.n_pairs(), 10u);  // 2 * 5 = 10 (commutative)
}

// =============================================================================
// Schwarz Bound Tests
// =============================================================================

TEST(ShellSetPairTest, SchwarzBoundPositive) {
    // Schwarz bound should be positive for overlapping shells
    Shell s1 = make_s_shell(Point3D(0.0, 0.0, 0.0));
    Shell s2 = make_s_shell(Point3D(1.0, 0.0, 0.0));

    std::vector<Shell> shells_a = {s1};
    std::vector<Shell> shells_b = {s2};

    ShellSet set_a = make_shell_set(shells_a);
    ShellSet set_b = make_shell_set(shells_b);

    ShellSetPair pair(set_a, set_b);

    // Schwarz bound Q_ab = sqrt(max (ab|ab)) should be positive
    Real Q = pair.schwarz_bound();
    EXPECT_GT(Q, 0.0) << "Schwarz bound should be positive for overlapping shells";
    EXPECT_TRUE(pair.schwarz_computed()) << "Schwarz bound should be marked as computed";
}

TEST(ShellSetPairTest, SchwarzBoundDifferentAM) {
    // Test different angular momentum combinations - all should have positive bounds
    Shell s1 = make_s_shell(Point3D(0.0, 0.0, 0.0));
    Shell p1 = make_p_shell(Point3D(1.0, 0.0, 0.0));
    Shell d1 = make_d_shell(Point3D(2.0, 0.0, 0.0));

    std::vector<Shell> shells_s = {s1};
    std::vector<Shell> shells_p = {p1};
    std::vector<Shell> shells_d = {d1};

    ShellSet set_s = make_shell_set(shells_s);
    ShellSet set_p = make_shell_set(shells_p);
    ShellSet set_d = make_shell_set(shells_d);

    // All combinations should have positive Schwarz bounds
    EXPECT_GT(ShellSetPair(set_s, set_s).schwarz_bound(), 0.0);
    EXPECT_GT(ShellSetPair(set_s, set_p).schwarz_bound(), 0.0);
    EXPECT_GT(ShellSetPair(set_s, set_d).schwarz_bound(), 0.0);
    EXPECT_GT(ShellSetPair(set_p, set_p).schwarz_bound(), 0.0);
    EXPECT_GT(ShellSetPair(set_p, set_d).schwarz_bound(), 0.0);
    EXPECT_GT(ShellSetPair(set_d, set_d).schwarz_bound(), 0.0);
}

TEST(ShellSetPairTest, SchwarzBoundLazyComputation) {
    Shell s1 = make_s_shell(Point3D(0.0, 0.0, 0.0));
    Shell s2 = make_s_shell(Point3D(1.0, 0.0, 0.0));

    std::vector<Shell> shells_a = {s1};
    std::vector<Shell> shells_b = {s2};

    ShellSet set_a = make_shell_set(shells_a);
    ShellSet set_b = make_shell_set(shells_b);

    ShellSetPair pair(set_a, set_b);

    // Not computed yet
    EXPECT_FALSE(pair.schwarz_computed());

    // Access the bound
    Real Q = pair.schwarz_bound();

    // Now it should be computed
    EXPECT_TRUE(pair.schwarz_computed());

    // Subsequent calls should return the same value
    EXPECT_DOUBLE_EQ(pair.schwarz_bound(), Q);
}

TEST(ShellSetPairTest, SchwarzBoundPrecompute) {
    Shell s1 = make_s_shell(Point3D(0.0, 0.0, 0.0));
    Shell s2 = make_s_shell(Point3D(1.0, 0.0, 0.0));

    std::vector<Shell> shells_a = {s1};
    std::vector<Shell> shells_b = {s2};

    ShellSet set_a = make_shell_set(shells_a);
    ShellSet set_b = make_shell_set(shells_b);

    ShellSetPair pair(set_a, set_b);

    // Not computed yet
    EXPECT_FALSE(pair.schwarz_computed());

    // Precompute
    pair.precompute_schwarz_bound();

    // Now it should be computed
    EXPECT_TRUE(pair.schwarz_computed());
}

// =============================================================================
// Different Angular Momentum Combinations
// =============================================================================

TEST(ShellSetPairTest, SSPair) {
    Shell s1 = make_s_shell(Point3D(0.0, 0.0, 0.0));
    Shell s2 = make_s_shell(Point3D(1.0, 0.0, 0.0));

    std::vector<Shell> shells_a = {s1};
    std::vector<Shell> shells_b = {s2};

    ShellSet set_a = make_shell_set(shells_a);
    ShellSet set_b = make_shell_set(shells_b);

    ShellSetPair pair(set_a, set_b);

    EXPECT_EQ(pair.La(), 0);
    EXPECT_EQ(pair.Lb(), 0);
    EXPECT_EQ(pair.L_total(), 0);
    EXPECT_EQ(pair.n_pairs(), 1u);
}

TEST(ShellSetPairTest, SPPair) {
    Shell s1 = make_s_shell(Point3D(0.0, 0.0, 0.0));
    Shell p1 = make_p_shell(Point3D(1.0, 0.0, 0.0));

    std::vector<Shell> shells_s = {s1};
    std::vector<Shell> shells_p = {p1};

    ShellSet set_s = make_shell_set(shells_s);
    ShellSet set_p = make_shell_set(shells_p);

    ShellSetPair pair(set_s, set_p);

    EXPECT_EQ(pair.La(), 0);
    EXPECT_EQ(pair.Lb(), 1);
    EXPECT_EQ(pair.L_total(), 1);
    EXPECT_EQ(pair.n_pairs(), 1u);
}

TEST(ShellSetPairTest, PPPair) {
    Shell p1 = make_p_shell(Point3D(0.0, 0.0, 0.0));
    Shell p2 = make_p_shell(Point3D(1.0, 0.0, 0.0));
    Shell p3 = make_p_shell(Point3D(2.0, 0.0, 0.0));

    std::vector<Shell> shells_a = {p1, p2};
    std::vector<Shell> shells_b = {p3};

    ShellSet set_a = make_shell_set(shells_a);
    ShellSet set_b = make_shell_set(shells_b);

    ShellSetPair pair(set_a, set_b);

    EXPECT_EQ(pair.La(), 1);
    EXPECT_EQ(pair.Lb(), 1);
    EXPECT_EQ(pair.L_total(), 2);
    EXPECT_EQ(pair.n_pairs(), 2u);  // 2 * 1 = 2
}

TEST(ShellSetPairTest, SDPair) {
    Shell s1 = make_s_shell(Point3D(0.0, 0.0, 0.0));
    Shell d1 = make_d_shell(Point3D(1.0, 0.0, 0.0));

    std::vector<Shell> shells_s = {s1};
    std::vector<Shell> shells_d = {d1};

    ShellSet set_s = make_shell_set(shells_s);
    ShellSet set_d = make_shell_set(shells_d);

    ShellSetPair pair(set_s, set_d);

    EXPECT_EQ(pair.La(), 0);
    EXPECT_EQ(pair.Lb(), 2);
    EXPECT_EQ(pair.L_total(), 2);
    EXPECT_EQ(pair.n_pairs(), 1u);
}

TEST(ShellSetPairTest, PDPair) {
    Shell p1 = make_p_shell(Point3D(0.0, 0.0, 0.0));
    Shell p2 = make_p_shell(Point3D(1.0, 0.0, 0.0));
    Shell d1 = make_d_shell(Point3D(2.0, 0.0, 0.0));
    Shell d2 = make_d_shell(Point3D(3.0, 0.0, 0.0));
    Shell d3 = make_d_shell(Point3D(4.0, 0.0, 0.0));

    std::vector<Shell> shells_p = {p1, p2};
    std::vector<Shell> shells_d = {d1, d2, d3};

    ShellSet set_p = make_shell_set(shells_p);
    ShellSet set_d = make_shell_set(shells_d);

    ShellSetPair pair(set_p, set_d);

    EXPECT_EQ(pair.La(), 1);
    EXPECT_EQ(pair.Lb(), 2);
    EXPECT_EQ(pair.L_total(), 3);
    EXPECT_EQ(pair.n_pairs(), 6u);  // 2 * 3 = 6
}

TEST(ShellSetPairTest, DDPair) {
    Shell d1 = make_d_shell(Point3D(0.0, 0.0, 0.0));
    Shell d2 = make_d_shell(Point3D(1.0, 0.0, 0.0));

    std::vector<Shell> shells_a = {d1};
    std::vector<Shell> shells_b = {d2};

    ShellSet set_a = make_shell_set(shells_a);
    ShellSet set_b = make_shell_set(shells_b);

    ShellSetPair pair(set_a, set_b);

    EXPECT_EQ(pair.La(), 2);
    EXPECT_EQ(pair.Lb(), 2);
    EXPECT_EQ(pair.L_total(), 4);
    EXPECT_EQ(pair.n_pairs(), 1u);
}

// =============================================================================
// Large Sets
// =============================================================================

TEST(ShellSetPairTest, LargeShellCounts) {
    // Test with larger shell counts to ensure n_pairs scales correctly
    constexpr Size n_shells_a = 20;
    constexpr Size n_shells_b = 15;

    std::vector<Shell> shells_a;
    for (Size i = 0; i < n_shells_a; ++i) {
        shells_a.push_back(make_s_shell(Point3D(static_cast<Real>(i), 0.0, 0.0)));
    }

    std::vector<Shell> shells_b;
    for (Size i = 0; i < n_shells_b; ++i) {
        shells_b.push_back(make_s_shell(Point3D(0.0, static_cast<Real>(i), 0.0)));
    }

    ShellSet set_a = make_shell_set(shells_a);
    ShellSet set_b = make_shell_set(shells_b);

    ShellSetPair pair(set_a, set_b);

    EXPECT_EQ(pair.n_pairs(), n_shells_a * n_shells_b);  // 20 * 15 = 300
    EXPECT_EQ(pair.La(), 0);
    EXPECT_EQ(pair.Lb(), 0);
    EXPECT_EQ(pair.L_total(), 0);
}

// =============================================================================
// Lightweight Class Tests
// =============================================================================

TEST(ShellSetPairTest, SizeDocumentation) {
    // Document the actual memory footprint of ShellSetPair
    // The class contains two pointers plus a shared_ptr for Schwarz cache
    constexpr Size min_size = 2 * sizeof(const ShellSet*) + sizeof(std::shared_ptr<void>);

    // Should be at least the minimum size
    EXPECT_GE(sizeof(ShellSetPair), min_size)
        << "ShellSetPair should contain at least two pointers and a shared_ptr";

    // Document actual size (for informational purposes)
    // This test will not fail but documents the current implementation
    EXPECT_LE(sizeof(ShellSetPair), 64u)  // Reasonable upper bound
        << "ShellSetPair should not be excessively large";
}

TEST(ShellSetPairTest, CopyConstructible) {
    Shell s1 = make_s_shell(Point3D(0.0, 0.0, 0.0));
    Shell s2 = make_s_shell(Point3D(1.0, 0.0, 0.0));

    std::vector<Shell> shells_a = {s1};
    std::vector<Shell> shells_b = {s2};

    ShellSet set_a = make_shell_set(shells_a);
    ShellSet set_b = make_shell_set(shells_b);

    ShellSetPair pair1(set_a, set_b);
    ShellSetPair pair2 = pair1;  // Copy construct

    // Both should refer to the same underlying ShellSets
    EXPECT_EQ(&pair1.shell_set_a(), &pair2.shell_set_a());
    EXPECT_EQ(&pair1.shell_set_b(), &pair2.shell_set_b());
    EXPECT_EQ(pair1.n_pairs(), pair2.n_pairs());
    EXPECT_EQ(pair1.La(), pair2.La());
    EXPECT_EQ(pair1.Lb(), pair2.Lb());
}

// =============================================================================
// PrimitivePairData Tests
// =============================================================================

TEST(ShellSetPairTest, PairDataNotReadyBeforeAccess) {
    Shell s1 = make_s_shell(Point3D(0.0, 0.0, 0.0));
    Shell s2 = make_s_shell(Point3D(1.0, 0.0, 0.0));

    std::vector<Shell> shells_a = {s1};
    std::vector<Shell> shells_b = {s2};

    ShellSet set_a = make_shell_set(shells_a);
    ShellSet set_b = make_shell_set(shells_b);

    ShellSetPair pair(set_a, set_b);

    EXPECT_FALSE(pair.pair_data_ready());
}

TEST(ShellSetPairTest, PairDataReadyAfterAccess) {
    Shell s1 = make_s_shell(Point3D(0.0, 0.0, 0.0));
    Shell s2 = make_s_shell(Point3D(1.0, 0.0, 0.0));

    std::vector<Shell> shells_a = {s1};
    std::vector<Shell> shells_b = {s2};

    ShellSet set_a = make_shell_set(shells_a);
    ShellSet set_b = make_shell_set(shells_b);

    ShellSetPair pair(set_a, set_b);

    (void)pair.pair_data();
    EXPECT_TRUE(pair.pair_data_ready());
}

TEST(ShellSetPairTest, PairDataPrecompute) {
    Shell s1 = make_s_shell(Point3D(0.0, 0.0, 0.0));
    Shell s2 = make_s_shell(Point3D(1.0, 0.0, 0.0));

    std::vector<Shell> shells_a = {s1};
    std::vector<Shell> shells_b = {s2};

    ShellSet set_a = make_shell_set(shells_a);
    ShellSet set_b = make_shell_set(shells_b);

    ShellSetPair pair(set_a, set_b);

    pair.precompute_pair_data();
    EXPECT_TRUE(pair.pair_data_ready());
}

TEST(ShellSetPairTest, PairDataDimensions) {
    // Two S-shells (K=3 each) in set_a, one S-shell (K=3) in set_b
    Shell s1 = make_s_shell(Point3D(0.0, 0.0, 0.0));
    Shell s2 = make_s_shell(Point3D(1.0, 0.0, 0.0));
    Shell s3 = make_s_shell(Point3D(2.0, 0.0, 0.0));

    std::vector<Shell> shells_a = {s1, s2};
    std::vector<Shell> shells_b = {s3};

    ShellSet set_a = make_shell_set(shells_a);
    ShellSet set_b = make_shell_set(shells_b);

    ShellSetPair pair(set_a, set_b);
    const auto& pd = pair.pair_data();

    EXPECT_EQ(pd.n_shells_a, 2u);
    EXPECT_EQ(pd.n_shells_b, 1u);
    EXPECT_EQ(pd.K_a, 3u);
    EXPECT_EQ(pd.K_b, 3u);
    EXPECT_EQ(pd.n_total_pairs, 2u * 1u * 3u * 3u);  // 18
    EXPECT_EQ(pd.primitives_per_shell_pair(), 9u);  // 3 * 3
    EXPECT_FALSE(pd.empty());
}

TEST(ShellSetPairTest, PairDataMatchesOnTheFly) {
    // Verify that cached pair data matches on-the-fly computation
    Shell s1 = make_s_shell(Point3D(0.0, 0.0, 0.0), 2);
    Shell s2 = make_s_shell(Point3D(1.4, 0.0, 0.0), 2);

    std::vector<Shell> shells_a = {s1};
    std::vector<Shell> shells_b = {s2};

    ShellSet set_a = make_shell_set(shells_a);
    ShellSet set_b = make_shell_set(shells_b);

    ShellSetPair pair(set_a, set_b);
    const auto& pd = pair.pair_data();

    // Compare each primitive pair with on-the-fly computation
    const auto& shell_a = set_a.shell(0);
    const auto& shell_b = set_b.shell(0);
    const auto& A = shell_a.center();
    const auto& B = shell_b.center();
    const auto exp_a = shell_a.exponents();
    const auto exp_b = shell_b.exponents();
    const auto coeff_a = shell_a.coefficients();
    const auto coeff_b = shell_b.coefficients();

    for (Size p = 0; p < static_cast<Size>(shell_a.n_primitives()); ++p) {
        for (Size q = 0; q < static_cast<Size>(shell_b.n_primitives()); ++q) {
            const auto gp = math::compute_gaussian_product(exp_a[p], A, exp_b[q], B);
            const Size idx = pd.pair_index(0, 0, p, q);

            EXPECT_NEAR(pd.Px[idx], gp.P.x, 1e-14);
            EXPECT_NEAR(pd.Py[idx], gp.P.y, 1e-14);
            EXPECT_NEAR(pd.Pz[idx], gp.P.z, 1e-14);
            EXPECT_NEAR(pd.zeta[idx], gp.zeta, 1e-14);
            EXPECT_NEAR(pd.one_over_2zeta[idx], 0.5 / gp.zeta, 1e-14);
            EXPECT_NEAR(pd.mu[idx], gp.mu, 1e-14);
            EXPECT_NEAR(pd.K_AB[idx], gp.K_AB, 1e-14);
            EXPECT_NEAR(pd.coeff_product[idx], coeff_a[p] * coeff_b[q], 1e-14);
            EXPECT_NEAR(pd.PA_x[idx], gp.P.x - A.x, 1e-14);
            EXPECT_NEAR(pd.PA_y[idx], gp.P.y - A.y, 1e-14);
            EXPECT_NEAR(pd.PA_z[idx], gp.P.z - A.z, 1e-14);
            EXPECT_NEAR(pd.PB_x[idx], gp.P.x - B.x, 1e-14);
            EXPECT_NEAR(pd.PB_y[idx], gp.P.y - B.y, 1e-14);
            EXPECT_NEAR(pd.PB_z[idx], gp.P.z - B.z, 1e-14);
        }
    }
}

TEST(ShellSetPairTest, PairDataMultipleShellPairs) {
    // Verify correct indexing across multiple shell pairs
    Shell s1 = make_s_shell(Point3D(0.0, 0.0, 0.0), 2);
    Shell s2 = make_s_shell(Point3D(1.0, 0.0, 0.0), 2);
    Shell s3 = make_s_shell(Point3D(0.0, 1.0, 0.0), 2);
    Shell s4 = make_s_shell(Point3D(1.0, 1.0, 0.0), 2);

    std::vector<Shell> shells_a = {s1, s2};
    std::vector<Shell> shells_b = {s3, s4};

    ShellSet set_a = make_shell_set(shells_a);
    ShellSet set_b = make_shell_set(shells_b);

    ShellSetPair pair(set_a, set_b);
    const auto& pd = pair.pair_data();

    EXPECT_EQ(pd.n_total_pairs, 2u * 2u * 2u * 2u);  // 16

    // Verify second shell pair (i=1, j=0) has correct center
    for (Size p = 0; p < 2; ++p) {
        for (Size q = 0; q < 2; ++q) {
            const Size idx = pd.pair_index(1, 0, p, q);
            const auto gp = math::compute_gaussian_product(
                set_a.shell(1).exponents()[p], set_a.shell(1).center(),
                set_b.shell(0).exponents()[q], set_b.shell(0).center());

            EXPECT_NEAR(pd.Px[idx], gp.P.x, 1e-14);
            EXPECT_NEAR(pd.zeta[idx], gp.zeta, 1e-14);
            EXPECT_NEAR(pd.K_AB[idx], gp.K_AB, 1e-14);
        }
    }
}

TEST(ShellSetPairTest, PairDataCopySharesCache) {
    Shell s1 = make_s_shell(Point3D(0.0, 0.0, 0.0));
    Shell s2 = make_s_shell(Point3D(1.0, 0.0, 0.0));

    std::vector<Shell> shells_a = {s1};
    std::vector<Shell> shells_b = {s2};

    ShellSet set_a = make_shell_set(shells_a);
    ShellSet set_b = make_shell_set(shells_b);

    ShellSetPair pair1(set_a, set_b);
    pair1.precompute_pair_data();

    ShellSetPair pair2 = pair1;

    // Copy shares the cache, so it should also report ready
    EXPECT_TRUE(pair2.pair_data_ready());

    // And the data should be identical
    const auto& pd1 = pair1.pair_data();
    const auto& pd2 = pair2.pair_data();
    EXPECT_EQ(&pd1, &pd2);  // Same object via shared_ptr
}
