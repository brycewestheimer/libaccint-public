// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2026 Bryce M. Westheimer. All rights reserved.

/// @file test_density_screening.cpp
/// @brief Unit tests for density-weighted Schwarz screening

#include <libaccint/screening/density_screening.hpp>
#include <libaccint/screening/schwarz_bounds.hpp>
#include <libaccint/basis/shell.hpp>
#include <libaccint/basis/basis_set.hpp>
#include <gtest/gtest.h>

#include <cmath>
#include <vector>

using namespace libaccint;
using namespace libaccint::screening;

namespace {

/// Helper: create an S-shell (L=0) with given center
Shell make_s_shell(Point3D center) {
    std::vector<Real> exponents = {3.0, 1.0, 0.3};
    std::vector<Real> coefficients = {0.3, 0.5, 0.2};
    return Shell(AngularMomentum::S, center, exponents, coefficients);
}

/// Helper: create a P-shell (L=1) with given center
Shell make_p_shell(Point3D center) {
    std::vector<Real> exponents = {2.0, 0.5};
    std::vector<Real> coefficients = {0.6, 0.4};
    return Shell(AngularMomentum::P, center, exponents, coefficients);
}

/// Helper: create a simple H2O-like basis set
BasisSet make_h2o_basis() {
    std::vector<Shell> shells;

    // Oxygen (at origin)
    shells.push_back(make_s_shell(Point3D(0.0, 0.0, 0.0)));
    shells.push_back(make_p_shell(Point3D(0.0, 0.0, 0.0)));

    // Hydrogen 1
    shells.push_back(make_s_shell(Point3D(0.0, 1.43, -1.11)));

    // Hydrogen 2
    shells.push_back(make_s_shell(Point3D(0.0, -1.43, -1.11)));

    return BasisSet(std::move(shells));
}

}  // anonymous namespace

// =============================================================================
// DensityScreening Tests
// =============================================================================

TEST(DensityScreeningTest, ConstructFromBasis) {
    BasisSet basis = make_h2o_basis();
    DensityScreening screen(basis);

    // Not initialized until density is set
    EXPECT_FALSE(screen.is_initialized());
}

TEST(DensityScreeningTest, UpdateDensity) {
    BasisSet basis = make_h2o_basis();
    DensityScreening screen(basis);

    Size nbf = basis.n_basis_functions();
    std::vector<Real> D(nbf * nbf, 1.0);  // All ones

    screen.update_density(D.data(), nbf);

    EXPECT_TRUE(screen.is_initialized());
    EXPECT_DOUBLE_EQ(screen.max_d_max(), 1.0);
}

TEST(DensityScreeningTest, ShellPairDmax) {
    BasisSet basis = make_h2o_basis();
    DensityScreening screen(basis);

    Size nbf = basis.n_basis_functions();

    // Create a density matrix with specific values
    std::vector<Real> D(nbf * nbf, 0.0);

    // Set D[0, 0] = 2.0 (shell 0 is s-type with 1 function)
    D[0] = 2.0;

    screen.update_density(D.data(), nbf);

    // D_max for shell pair (0, 0) should be 2.0
    EXPECT_DOUBLE_EQ(screen.shell_pair_d_max(0, 0), 2.0);

    // Symmetry: (j, i) == (i, j)
    EXPECT_DOUBLE_EQ(screen.shell_pair_d_max(1, 0), screen.shell_pair_d_max(0, 1));
}

TEST(DensityScreeningTest, QuartetDmax) {
    BasisSet basis = make_h2o_basis();
    DensityScreening screen(basis);

    Size nbf = basis.n_basis_functions();
    std::vector<Real> D(nbf * nbf, 0.0);

    // H2O basis layout:
    // Shell 0 (S): AO 0
    // Shell 1 (P): AOs 1,2,3
    // Shell 2 (S): AO 4
    // Shell 3 (S): AO 5
    //
    // quartet_d_max(0, 1, 2, 3) looks at exchange-type pairs:
    // D_max(0,2): max |D[0, 4]|
    // D_max(0,3): max |D[0, 5]|
    // D_max(1,2): max |D[1:4, 4]|
    // D_max(1,3): max |D[1:4, 5]|
    //
    // Set density at exchange-type positions
    D[0 * nbf + 4] = 1.5;    // D[0, 4] - shell pair (0, 2)
    D[1 * nbf + 5] = 2.0;    // D[1, 5] - shell pair (1, 3)

    screen.update_density(D.data(), nbf);

    // quartet_d_max returns max over exchange-type pairs
    Real d_max = screen.quartet_d_max(0, 1, 2, 3);

    // Should be max of D_max(0,2), D_max(0,3), D_max(1,2), D_max(1,3)
    // D_max(0,2) = 1.5, D_max(1,3) = 2.0, others = 0
    EXPECT_DOUBLE_EQ(d_max, 2.0);
}

TEST(DensityScreeningTest, PassesScreening) {
    BasisSet basis = make_h2o_basis();
    SchwarzBounds schwarz(basis);
    DensityScreening screen(basis);

    Size nbf = basis.n_basis_functions();
    std::vector<Real> D(nbf * nbf, 1.0);  // Uniform density

    screen.update_density(D.data(), nbf);

    // With uniform density D_max = 1, screening should behave like basic Schwarz
    Real threshold = 1e-10;

    // All quartets that pass basic Schwarz should also pass density-weighted
    for (Size i = 0; i < basis.n_shells(); ++i) {
        for (Size j = i; j < basis.n_shells(); ++j) {
            for (Size k = i; k < basis.n_shells(); ++k) {
                Size l_start = (k == i) ? j : k;
                for (Size l = l_start; l < basis.n_shells(); ++l) {
                    bool passes_basic = schwarz.passes_screening(i, j, k, l, threshold);
                    bool passes_density = screen.passes_screening(i, j, k, l, schwarz, threshold);

                    // With D_max = 1, both should give same result
                    EXPECT_EQ(passes_density, passes_basic)
                        << "Mismatch at quartet (" << i << "," << j << "|" << k << "," << l << ")";
                }
            }
        }
    }
}

TEST(DensityScreeningTest, SparseDensityScreensMore) {
    BasisSet basis = make_h2o_basis();
    SchwarzBounds schwarz(basis);
    DensityScreening screen(basis);

    Size nbf = basis.n_basis_functions();

    // Create a sparse density (only diagonal elements)
    std::vector<Real> D_sparse(nbf * nbf, 0.0);
    for (Size i = 0; i < nbf; ++i) {
        D_sparse[i * nbf + i] = 1.0;
    }

    screen.update_density(D_sparse.data(), nbf);

    Real threshold = 1e-10;

    // Count quartets passing with density weighting
    Size count_density = 0;
    Size count_basic = 0;

    for (Size i = 0; i < basis.n_shells(); ++i) {
        for (Size j = i; j < basis.n_shells(); ++j) {
            for (Size k = i; k < basis.n_shells(); ++k) {
                Size l_start = (k == i) ? j : k;
                for (Size l = l_start; l < basis.n_shells(); ++l) {
                    if (schwarz.passes_screening(i, j, k, l, threshold)) {
                        ++count_basic;
                    }
                    if (screen.passes_screening(i, j, k, l, schwarz, threshold)) {
                        ++count_density;
                    }
                }
            }
        }
    }

    // Sparse density should screen more quartets (or same)
    EXPECT_LE(count_density, count_basic);
}

TEST(DensityScreeningTest, UpdateDensityFromSpan) {
    BasisSet basis = make_h2o_basis();
    DensityScreening screen(basis);

    Size nbf = basis.n_basis_functions();
    std::vector<Real> D(nbf * nbf, 0.5);

    std::span<const Real> D_span(D);
    screen.update_density(D_span, nbf);

    EXPECT_TRUE(screen.is_initialized());
    EXPECT_DOUBLE_EQ(screen.max_d_max(), 0.5);
}

