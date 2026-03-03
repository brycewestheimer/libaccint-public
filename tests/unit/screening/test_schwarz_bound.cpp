// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2026 Bryce M. Westheimer. All rights reserved.

/// @file test_schwarz_bound.cpp
/// @brief Unit tests for Schwarz screening bounds

#include <libaccint/basis/shell_set_pair.hpp>
#include <libaccint/basis/shell_set.hpp>
#include <libaccint/basis/shell.hpp>
#include <libaccint/kernels/eri_kernel.hpp>
#include <libaccint/buffers/two_electron_buffer.hpp>
#include <gtest/gtest.h>

#include <cmath>
#include <functional>
#include <thread>
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

/// Helper: create STO-3G hydrogen 1s shell at given position
Shell make_sto3g_hydrogen(Point3D center) {
    std::vector<Real> exponents = {3.425250914, 0.6239137298, 0.168855404};
    std::vector<Real> coefficients = {0.1543289673, 0.5353281423, 0.4446345422};
    return Shell(AngularMomentum::S, center, exponents, coefficients);
}

}  // anonymous namespace

// =============================================================================
// Schwarz Bound Conservativeness Tests
// =============================================================================

TEST(SchwarzBoundTest, ConservativeForSSShellPair) {
    // The Schwarz bound Q_ab * Q_cd should be >= |any (ab|cd)|
    // Here we test with SS pairs
    Shell s1 = make_s_shell(Point3D(0.0, 0.0, 0.0));
    Shell s2 = make_s_shell(Point3D(1.5, 0.0, 0.0));

    std::vector<Shell> shells_a = {s1};
    std::vector<Shell> shells_b = {s2};

    ShellSet set_a = make_shell_set(shells_a);
    ShellSet set_b = make_shell_set(shells_b);

    ShellSetPair pair_ab(set_a, set_b);
    ShellSetPair pair_cd(set_a, set_b);

    Real Q_ab = pair_ab.schwarz_bound();
    Real Q_cd = pair_cd.schwarz_bound();

    // Compute actual integral (ab|cd)
    TwoElectronBuffer<0> buffer;
    kernels::compute_eri(s1, s2, s1, s2, buffer);

    // Check all function indices
    const int na = s1.n_functions();
    const int nb = s2.n_functions();

    for (int a = 0; a < na; ++a) {
        for (int b = 0; b < nb; ++b) {
            for (int c = 0; c < na; ++c) {
                for (int d = 0; d < nb; ++d) {
                    Real eri = std::abs(buffer(a, b, c, d));
                    EXPECT_LE(eri, Q_ab * Q_cd)
                        << "Schwarz bound should be conservative: "
                        << "|(" << a << "," << b << "|" << c << "," << d << ")| = " << eri
                        << " > Q_ab * Q_cd = " << (Q_ab * Q_cd);
                }
            }
        }
    }
}

TEST(SchwarzBoundTest, ConservativeForPPShellPair) {
    Shell p1 = make_p_shell(Point3D(0.0, 0.0, 0.0));
    Shell p2 = make_p_shell(Point3D(2.0, 1.0, 0.0));

    std::vector<Shell> shells_a = {p1};
    std::vector<Shell> shells_b = {p2};

    ShellSet set_a = make_shell_set(shells_a);
    ShellSet set_b = make_shell_set(shells_b);

    ShellSetPair pair_ab(set_a, set_b);

    Real Q_ab = pair_ab.schwarz_bound();

    // Compute actual integrals (ab|ab)
    TwoElectronBuffer<0> buffer;
    kernels::compute_eri(p1, p2, p1, p2, buffer);

    const int na = p1.n_functions();
    const int nb = p2.n_functions();

    for (int a = 0; a < na; ++a) {
        for (int b = 0; b < nb; ++b) {
            for (int c = 0; c < na; ++c) {
                for (int d = 0; d < nb; ++d) {
                    Real eri = std::abs(buffer(a, b, c, d));
                    EXPECT_LE(eri, Q_ab * Q_ab)
                        << "Schwarz bound should be conservative for PP pair";
                }
            }
        }
    }
}

TEST(SchwarzBoundTest, ConservativeForMixedAM) {
    // Test with SP and PD pairs
    Shell s1 = make_s_shell(Point3D(0.0, 0.0, 0.0));
    Shell p1 = make_p_shell(Point3D(1.0, 0.0, 0.0));
    Shell d1 = make_d_shell(Point3D(2.0, 0.0, 0.0));

    std::vector<Shell> shells_s = {s1};
    std::vector<Shell> shells_p = {p1};
    std::vector<Shell> shells_d = {d1};

    ShellSet set_s = make_shell_set(shells_s);
    ShellSet set_p = make_shell_set(shells_p);
    ShellSet set_d = make_shell_set(shells_d);

    ShellSetPair pair_sp(set_s, set_p);
    ShellSetPair pair_pd(set_p, set_d);

    Real Q_sp = pair_sp.schwarz_bound();
    Real Q_pd = pair_pd.schwarz_bound();

    // Compute (sp|pd) integrals
    TwoElectronBuffer<0> buffer;
    kernels::compute_eri(s1, p1, p1, d1, buffer);

    const int ns = s1.n_functions();
    const int np = p1.n_functions();
    const int nd = d1.n_functions();

    for (int a = 0; a < ns; ++a) {
        for (int b = 0; b < np; ++b) {
            for (int c = 0; c < np; ++c) {
                for (int d = 0; d < nd; ++d) {
                    Real eri = std::abs(buffer(a, b, c, d));
                    EXPECT_LE(eri, Q_sp * Q_pd)
                        << "Schwarz bound should be conservative for mixed AM";
                }
            }
        }
    }
}

// =============================================================================
// Schwarz Bound Positivity Tests
// =============================================================================

TEST(SchwarzBoundTest, PositiveForOverlappingShells) {
    // Schwarz bound should be strictly positive for overlapping shells
    Shell s1 = make_s_shell(Point3D(0.0, 0.0, 0.0));
    Shell s2 = make_s_shell(Point3D(0.5, 0.0, 0.0));  // Close shells

    std::vector<Shell> shells_a = {s1};
    std::vector<Shell> shells_b = {s2};

    ShellSet set_a = make_shell_set(shells_a);
    ShellSet set_b = make_shell_set(shells_b);

    ShellSetPair pair(set_a, set_b);

    EXPECT_GT(pair.schwarz_bound(), 0.0)
        << "Schwarz bound should be positive for overlapping shells";
}

TEST(SchwarzBoundTest, PositiveForSelfPair) {
    // Same shell paired with itself should have positive bound
    Shell s1 = make_s_shell(Point3D(0.0, 0.0, 0.0));

    std::vector<Shell> shells = {s1};
    ShellSet set = make_shell_set(shells);

    ShellSetPair pair(set, set);

    EXPECT_GT(pair.schwarz_bound(), 0.0)
        << "Schwarz bound should be positive for self-pair";
}

TEST(SchwarzBoundTest, DecreasesWithDistance) {
    // Schwarz bound should decrease as shells become more separated
    Shell s1 = make_s_shell(Point3D(0.0, 0.0, 0.0));

    std::vector<Shell> shells_a = {s1};
    ShellSet set_a = make_shell_set(shells_a);

    Real prev_bound = std::numeric_limits<Real>::max();

    for (Real R = 0.5; R <= 10.0; R += 2.0) {
        Shell s2 = make_s_shell(Point3D(R, 0.0, 0.0));
        std::vector<Shell> shells_b = {s2};
        ShellSet set_b = make_shell_set(shells_b);

        ShellSetPair pair(set_a, set_b);
        Real bound = pair.schwarz_bound();

        EXPECT_LT(bound, prev_bound)
            << "Schwarz bound should decrease with distance at R=" << R;
        prev_bound = bound;
    }
}

// =============================================================================
// Thread Safety Tests
// =============================================================================

TEST(SchwarzBoundTest, ThreadSafeConcurrentAccess) {
    Shell s1 = make_s_shell(Point3D(0.0, 0.0, 0.0));
    Shell s2 = make_s_shell(Point3D(1.0, 0.0, 0.0));

    std::vector<Shell> shells_a = {s1};
    std::vector<Shell> shells_b = {s2};

    ShellSet set_a = make_shell_set(shells_a);
    ShellSet set_b = make_shell_set(shells_b);

    ShellSetPair pair(set_a, set_b);

    // Launch multiple threads accessing schwarz_bound() concurrently
    constexpr int n_threads = 8;
    std::vector<std::thread> threads;
    std::vector<Real> results(n_threads);

    for (int i = 0; i < n_threads; ++i) {
        threads.emplace_back([&pair, &results, i]() {
            results[i] = pair.schwarz_bound();
        });
    }

    for (auto& t : threads) {
        t.join();
    }

    // All threads should get the same value
    for (int i = 1; i < n_threads; ++i) {
        EXPECT_DOUBLE_EQ(results[0], results[i])
            << "All threads should get the same Schwarz bound";
    }
}

// =============================================================================
// Lazy vs Eager Computation Tests
// =============================================================================

TEST(SchwarzBoundTest, LazyVsEagerSameResults) {
    Shell s1 = make_s_shell(Point3D(0.0, 0.0, 0.0));
    Shell s2 = make_s_shell(Point3D(1.0, 0.0, 0.0));

    std::vector<Shell> shells_a = {s1};
    std::vector<Shell> shells_b = {s2};

    ShellSet set_a = make_shell_set(shells_a);
    ShellSet set_b = make_shell_set(shells_b);

    // Create two pairs - one with lazy, one with eager computation
    ShellSetPair pair_lazy(set_a, set_b);
    ShellSetPair pair_eager(set_a, set_b);

    // Precompute for the eager pair
    pair_eager.precompute_schwarz_bound();

    // Access lazy pair (triggers computation)
    Real lazy_bound = pair_lazy.schwarz_bound();
    Real eager_bound = pair_eager.schwarz_bound();

    EXPECT_DOUBLE_EQ(lazy_bound, eager_bound)
        << "Lazy and eager computation should give the same result";
}

// =============================================================================
// STO-3G H2O Tests
// =============================================================================

TEST(SchwarzBoundTest, STO3G_H2O_AllBoundsPositive) {
    // H2O geometry (bohr)
    Point3D O_pos(0.0, 0.0, 0.0);
    Point3D H1_pos(0.0, 1.43233673, -1.10866041);
    Point3D H2_pos(0.0, -1.43233673, -1.10866041);

    Shell O_1s = make_sto3g_hydrogen(O_pos);  // Actually H for simplicity
    Shell H1_1s = make_sto3g_hydrogen(H1_pos);
    Shell H2_1s = make_sto3g_hydrogen(H2_pos);

    // Create shell sets
    std::vector<Shell> shells_O = {O_1s};
    std::vector<Shell> shells_H1 = {H1_1s};
    std::vector<Shell> shells_H2 = {H2_1s};

    ShellSet set_O = make_shell_set(shells_O);
    ShellSet set_H1 = make_shell_set(shells_H1);
    ShellSet set_H2 = make_shell_set(shells_H2);

    // All pair combinations should have positive Schwarz bounds
    EXPECT_GT(ShellSetPair(set_O, set_O).schwarz_bound(), 0.0);
    EXPECT_GT(ShellSetPair(set_O, set_H1).schwarz_bound(), 0.0);
    EXPECT_GT(ShellSetPair(set_O, set_H2).schwarz_bound(), 0.0);
    EXPECT_GT(ShellSetPair(set_H1, set_H1).schwarz_bound(), 0.0);
    EXPECT_GT(ShellSetPair(set_H1, set_H2).schwarz_bound(), 0.0);
    EXPECT_GT(ShellSetPair(set_H2, set_H2).schwarz_bound(), 0.0);
}
