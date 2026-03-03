// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2026 Bryce M. Westheimer. All rights reserved.

/// @file test_screening_symmetry.cpp
/// @brief Tests for 8-fold permutation symmetry and density screening integration
/// (Tasks 8.3.2, 8.3.3, 8.3.4)

#include <libaccint/screening/schwarz_bounds.hpp>
#include <libaccint/screening/density_screening.hpp>
#include <libaccint/screening/screening_options.hpp>
#include <libaccint/basis/shell.hpp>
#include <libaccint/basis/basis_set.hpp>
#include <libaccint/engine/engine.hpp>
#include <libaccint/operators/operator.hpp>
#include <libaccint/consumers/fock_builder.hpp>

#include <gtest/gtest.h>
#include <cmath>
#include <vector>
#include <numeric>
#include <algorithm>

using namespace libaccint;
using namespace libaccint::screening;

namespace {

Shell make_s_shell(Point3D center) {
    std::vector<Real> exponents = {3.0, 1.0, 0.3};
    std::vector<Real> coefficients = {0.3, 0.5, 0.2};
    return Shell(AngularMomentum::S, center, exponents, coefficients);
}

Shell make_p_shell(Point3D center) {
    std::vector<Real> exponents = {2.0, 0.5};
    std::vector<Real> coefficients = {0.6, 0.4};
    return Shell(AngularMomentum::P, center, exponents, coefficients);
}

BasisSet make_h2o_basis() {
    std::vector<Shell> shells;
    shells.push_back(make_s_shell(Point3D(0.0, 0.0, 0.0)));
    shells.push_back(make_p_shell(Point3D(0.0, 0.0, 0.0)));
    shells.push_back(make_s_shell(Point3D(0.0, 1.43, -1.11)));
    shells.push_back(make_s_shell(Point3D(0.0, -1.43, -1.11)));
    return BasisSet(std::move(shells));
}

std::vector<Real> make_identity_density(Size nbf) {
    std::vector<Real> D(nbf * nbf, 0.0);
    for (Size i = 0; i < nbf; ++i) D[i * nbf + i] = 1.0;
    return D;
}

std::vector<Real> make_uniform_density(Size nbf) {
    return std::vector<Real>(nbf * nbf, 1.0);
}

}  // namespace

// =========================================================================
// Task 8.3.3: Symmetry Exploitation Tests
// =========================================================================

TEST(SymmetryExploitationTest, UniqueQuartetCount) {
    // With 8-fold symmetry, the number of unique quartets should be ~N^4/8
    auto basis = make_h2o_basis();
    Size n = basis.n_shells();

    // Count unique quartets (i<=j, k<=l, ij<=kl)
    Size unique = 0;
    Size total = n * n * n * n;

    for (Size i = 0; i < n; ++i)
        for (Size j = i; j < n; ++j)
            for (Size k = i; k < n; ++k) {
                Size l_start = (k == i) ? j : k;
                for (Size l = l_start; l < n; ++l) {
                    ++unique;
                }
            }

    // For n=4 shells: total = 256, unique should be significantly less
    EXPECT_LT(unique, total);
    // Exact: n_pairs = 4*5/2 = 10, unique = 10*11/2 = 55
    EXPECT_EQ(unique, 55u);
}

TEST(SymmetryExploitationTest, FactorSumEqualsTotal) {
    // Sum of symmetry factors over all canonical quartets should equal N^4
    auto basis = make_h2o_basis();
    Size n = basis.n_shells();

    Size factor_sum = 0;
    for (Size i = 0; i < n; ++i)
        for (Size j = i; j < n; ++j)
            for (Size k = i; k < n; ++k) {
                Size l_start = (k == i) ? j : k;
                for (Size l = l_start; l < n; ++l) {
                    int factor = 8;
                    if (i == j) factor /= 2;
                    if (k == l) factor /= 2;
                    if (i == k && j == l) factor /= 2;
                    factor_sum += static_cast<Size>(factor);
                }
            }

    Size total = n * n * n * n;
    EXPECT_EQ(factor_sum, total);
}

TEST(SymmetryExploitationTest, SpecificFactorValues) {
    // Check known factor values for specific quartets
    auto basis = make_h2o_basis();
    Size n = basis.n_shells();

    for (Size i = 0; i < n; ++i)
        for (Size j = i; j < n; ++j)
            for (Size k = i; k < n; ++k) {
                Size l_start = (k == i) ? j : k;
                for (Size l = l_start; l < n; ++l) {
                    int factor = 8;
                    if (i == j) factor /= 2;
                    if (k == l) factor /= 2;
                    if (i == k && j == l) factor /= 2;

                    // All-same: i=j=k=l → factor should be 1
                    if (i == j && k == l && i == k) {
                        EXPECT_EQ(factor, 1);
                    }
                    // General case: all different → factor should be 8
                    if (i != j && k != l && !(i == k && j == l)) {
                        EXPECT_EQ(factor, 8);
                    }
                }
            }
}

TEST(SymmetryExploitationTest, FockEquivalence) {
    // The symmetry-exploited path should produce the same Fock matrix
    // as the full N^4 path (with screening)
    auto basis = make_h2o_basis();
    Engine engine(basis);
    Size nbf = basis.n_basis_functions();

    auto D = make_uniform_density(nbf);

    // Full N^4 path
    consumers::FockBuilder fock_full(nbf);
    fock_full.set_density(D.data(), nbf);

    auto opts_full = ScreeningOptions::normal();
    opts_full.use_permutation_symmetry = false;
    engine.compute_and_consume(Operator::coulomb(), fock_full, opts_full);

    // Symmetry-exploited path
    consumers::FockBuilder fock_sym(nbf);
    fock_sym.set_density(D.data(), nbf);

    auto opts_sym = ScreeningOptions::normal();
    opts_sym.use_permutation_symmetry = true;
    engine.compute_and_consume(Operator::coulomb(), fock_sym, opts_sym);

    // Compare J matrices
    auto J_full = fock_full.get_coulomb_matrix();
    auto J_sym = fock_sym.get_coulomb_matrix();

    Real max_J_diff = 0.0;
    for (Size i = 0; i < nbf * nbf; ++i) {
        max_J_diff = std::max(max_J_diff, std::abs(J_full[i] - J_sym[i]));
    }
    EXPECT_LT(max_J_diff, 1e-12)
        << "Symmetry-exploited J matrix differs from full path";

    // Compare K matrices
    auto K_full = fock_full.get_exchange_matrix();
    auto K_sym = fock_sym.get_exchange_matrix();

    Real max_K_diff = 0.0;
    for (Size i = 0; i < nbf * nbf; ++i) {
        max_K_diff = std::max(max_K_diff, std::abs(K_full[i] - K_sym[i]));
    }
    EXPECT_LT(max_K_diff, 1e-12)
        << "Symmetry-exploited K matrix differs from full path";
}

TEST(SymmetryExploitationTest, BypassOption) {
    // With use_permutation_symmetry = false, the full N^4 path is used
    auto basis = make_h2o_basis();
    Engine engine(basis);
    Size nbf = basis.n_basis_functions();

    auto D = make_identity_density(nbf);

    consumers::FockBuilder fock(nbf);
    fock.set_density(D.data(), nbf);

    auto opts = ScreeningOptions::normal();
    opts.use_permutation_symmetry = false;

    // Should work without errors (using full path)
    EXPECT_NO_THROW(
        engine.compute_and_consume(Operator::coulomb(), fock, opts));
}

// =========================================================================
// Task 8.3.2: Density-Weighted Screening Tests
// =========================================================================

TEST(DensityWeightedScreeningTest, DensityWeightedRejectsWithoutDensity) {
    // density_weighted=true without set_density_matrix should throw
    auto basis = make_h2o_basis();
    Engine engine(basis);
    Size nbf = basis.n_basis_functions();

    auto D = make_identity_density(nbf);
    consumers::FockBuilder fock(nbf);
    fock.set_density(D.data(), nbf);

    auto opts = ScreeningOptions::normal();
    opts.density_weighted = true;
    // Engine's density matrix not set → should throw
    EXPECT_THROW(
        engine.compute_and_consume(Operator::coulomb(), fock, opts),
        InvalidArgumentException);
}

TEST(DensityWeightedScreeningTest, DensityWeightedWithDensityWorks) {
    // With density matrix set, density_weighted=true should work
    auto basis = make_h2o_basis();
    Engine engine(basis);
    Size nbf = basis.n_basis_functions();

    auto D = make_uniform_density(nbf);
    engine.set_density_matrix(D.data(), nbf);

    consumers::FockBuilder fock(nbf);
    fock.set_density(D.data(), nbf);

    auto opts = ScreeningOptions::normal();
    opts.density_weighted = true;

    EXPECT_NO_THROW(
        engine.compute_and_consume(Operator::coulomb(), fock, opts));
}

TEST(DensityWeightedScreeningTest, SparseDensityScreensMore) {
    // A sparse density matrix should screen more quartets than a dense one
    auto basis = make_h2o_basis();
    SchwarzBounds bounds(basis);
    Size nbf = basis.n_basis_functions();
    Size n = basis.n_shells();

    // Dense density
    auto D_dense = make_uniform_density(nbf);
    DensityScreening screen_dense(basis);
    screen_dense.update_density(D_dense.data(), nbf);

    // Sparse density (only diagonal)
    auto D_sparse = make_identity_density(nbf);
    DensityScreening screen_sparse(basis);
    screen_sparse.update_density(D_sparse.data(), nbf);

    Real threshold = 1e-12;
    Size pass_dense = 0, pass_sparse = 0;

    for (Size i = 0; i < n; ++i)
        for (Size j = 0; j < n; ++j)
            for (Size k = 0; k < n; ++k)
                for (Size l = 0; l < n; ++l) {
                    if (screen_dense.passes_screening(i, j, k, l, bounds, threshold))
                        ++pass_dense;
                    if (screen_sparse.passes_screening(i, j, k, l, bounds, threshold))
                        ++pass_sparse;
                }

    // Sparse density should pass fewer or equal quartets
    EXPECT_LE(pass_sparse, pass_dense);
}

TEST(DensityWeightedScreeningTest, UniformDensityEquivalence) {
    // With uniform unit density, density-weighted screening should pass
    // exactly the same quartets as Schwarz-only screening (D_max=1 for all)
    auto basis = make_h2o_basis();
    SchwarzBounds bounds(basis);
    Size nbf = basis.n_basis_functions();
    Size n = basis.n_shells();

    auto D_unit = make_uniform_density(nbf);
    DensityScreening screen(basis);
    screen.update_density(D_unit.data(), nbf);

    Real threshold = 1e-12;

    for (Size i = 0; i < n; ++i)
        for (Size j = 0; j < n; ++j)
            for (Size k = 0; k < n; ++k)
                for (Size l = 0; l < n; ++l) {
                    bool schwarz_pass = bounds.passes_screening(i, j, k, l, threshold);
                    bool density_pass = screen.passes_screening(i, j, k, l, bounds, threshold);
                    // With D_max >= 1 (uniform density), density screening
                    // should pass everything that Schwarz passes
                    if (schwarz_pass) {
                        EXPECT_TRUE(density_pass)
                            << "Uniform density should not reject Schwarz-passing quartet ("
                            << i << "," << j << "," << k << "," << l << ")";
                    }
                }
}

TEST(DensityWeightedScreeningTest, FockMatchesFull) {
    // Density-weighted screening with uniform density should produce the same
    // Fock matrix as unscreened computation (D_max=1 for all pairs, so all
    // Schwarz-passing quartets also pass density screening)
    auto basis = make_h2o_basis();
    Engine engine(basis);
    Size nbf = basis.n_basis_functions();

    // Use uniform density so D_max=1 for all shell pairs
    auto D = make_uniform_density(nbf);

    // Unscreened Fock
    consumers::FockBuilder fock_ref(nbf);
    fock_ref.set_density(D.data(), nbf);
    engine.compute_and_consume(Operator::coulomb(), fock_ref);

    // Density-weighted screened Fock
    consumers::FockBuilder fock_dw(nbf);
    fock_dw.set_density(D.data(), nbf);
    engine.set_density_matrix(D.data(), nbf);

    auto opts = ScreeningOptions::normal();
    opts.density_weighted = true;
    engine.compute_and_consume(Operator::coulomb(), fock_dw, opts);

    auto J_ref = fock_ref.get_coulomb_matrix();
    auto J_dw = fock_dw.get_coulomb_matrix();

    Real max_diff = 0.0;
    for (Size i = 0; i < nbf * nbf; ++i) {
        max_diff = std::max(max_diff, std::abs(J_ref[i] - J_dw[i]));
    }
    EXPECT_LT(max_diff, 1e-10)
        << "Density-weighted screened Fock with uniform D differs from unscreened";
}

// =========================================================================
// Task 8.3.4: Threshold Sensitivity Tests
// =========================================================================

TEST(ThresholdSensitivityTest, Monotonicity) {
    // Tighter threshold should pass more quartets (fewer screened out)
    auto basis = make_h2o_basis();
    SchwarzBounds bounds(basis);

    std::vector<Real> thresholds = {1e-4, 1e-6, 1e-8, 1e-10, 1e-12, 1e-14};
    std::vector<Size> counts;

    for (Real t : thresholds) {
        counts.push_back(bounds.count_passing_quartets(t));
    }

    // Each threshold should pass more quartets than the previous (or equal)
    for (Size i = 1; i < counts.size(); ++i) {
        EXPECT_GE(counts[i], counts[i-1])
            << "Monotonicity violated at threshold " << thresholds[i];
    }
}

TEST(ThresholdSensitivityTest, ZeroThreshold) {
    // Zero threshold should pass all quartets
    auto basis = make_h2o_basis();
    SchwarzBounds bounds(basis);
    Size n = bounds.n_shells();

    Size n_pairs = n * (n + 1) / 2;
    Size total_unique = n_pairs * (n_pairs + 1) / 2;

    Size count = bounds.count_passing_quartets(0.0);
    EXPECT_EQ(count, total_unique);
}

TEST(ThresholdSensitivityTest, LargeThresholdScreensAll) {
    // Very large threshold should screen almost all quartets
    auto basis = make_h2o_basis();
    SchwarzBounds bounds(basis);

    Size count = bounds.count_passing_quartets(1e6);
    EXPECT_EQ(count, 0u) << "Very large threshold should screen all quartets";
}

TEST(ThresholdSensitivityTest, AccuracyVsThreshold) {
    // Fock matrix accuracy should improve with tighter thresholds
    auto basis = make_h2o_basis();
    Engine engine(basis);
    Size nbf = basis.n_basis_functions();

    auto D = make_identity_density(nbf);

    // Reference: unscreened
    consumers::FockBuilder fock_ref(nbf);
    fock_ref.set_density(D.data(), nbf);
    engine.compute_and_consume(Operator::coulomb(), fock_ref);
    auto J_ref = fock_ref.get_coulomb_matrix();

    std::vector<Real> thresholds = {1e-8, 1e-10, 1e-12, 1e-14};
    std::vector<Real> max_errors;

    for (Real t : thresholds) {
        consumers::FockBuilder fock(nbf);
        fock.set_density(D.data(), nbf);

        auto opts = ScreeningOptions{.threshold = t, .enabled = true};
        engine.compute_and_consume(Operator::coulomb(), fock, opts);

        auto J = fock.get_coulomb_matrix();
        Real max_err = 0.0;
        for (Size i = 0; i < nbf * nbf; ++i) {
            max_err = std::max(max_err, std::abs(J[i] - J_ref[i]));
        }
        max_errors.push_back(max_err);
    }

    // Errors should be monotonically non-increasing with tighter thresholds
    for (Size i = 1; i < max_errors.size(); ++i) {
        EXPECT_LE(max_errors[i], max_errors[i-1] + 1e-15)
            << "Accuracy should improve with tighter thresholds";
    }
}
