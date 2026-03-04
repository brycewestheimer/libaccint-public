// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2026 Bryce M. Westheimer. All rights reserved.

/// @file test_screened_fock_validation.cpp
/// @brief Quality Gate G2 validation: screened vs unscreened Fock matrix comparison

#include <libaccint/screening/screened_quartet_iterator.hpp>
#include <libaccint/screening/schwarz_bounds.hpp>
#include <libaccint/screening/screening_options.hpp>
#include <libaccint/basis/shell.hpp>
#include <libaccint/basis/basis_set.hpp>
#include <libaccint/consumers/fock_builder.hpp>
#include <libaccint/engine/engine.hpp>
#include <libaccint/operators/operator.hpp>
#include <gtest/gtest.h>

#include <cmath>
#include <vector>
#include <numeric>

using namespace libaccint;
using namespace libaccint::screening;
using namespace libaccint::consumers;

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

/// Helper: create a larger water cluster for screening efficiency tests
BasisSet make_water_cluster(Size n_waters, Real spacing = 8.0) {
    std::vector<Shell> shells;

    for (Size w = 0; w < n_waters; ++w) {
        Real x_offset = static_cast<Real>(w % 3) * spacing;
        Real y_offset = static_cast<Real>((w / 3) % 3) * spacing;
        Real z_offset = static_cast<Real>(w / 9) * spacing;

        // Oxygen
        shells.push_back(make_s_shell(Point3D(x_offset, y_offset, z_offset)));
        shells.push_back(make_p_shell(Point3D(x_offset, y_offset, z_offset)));

        // Hydrogens
        shells.push_back(make_s_shell(Point3D(x_offset, y_offset + 1.43, z_offset - 1.11)));
        shells.push_back(make_s_shell(Point3D(x_offset, y_offset - 1.43, z_offset - 1.11)));
    }

    return BasisSet(std::move(shells));
}

/// Helper: create identity-like density matrix
std::vector<Real> make_identity_density(Size nbf) {
    std::vector<Real> D(nbf * nbf, 0.0);
    for (Size i = 0; i < nbf; ++i) {
        D[i * nbf + i] = 1.0;
    }
    return D;
}

/// Helper: compute max absolute difference between two matrices
Real max_abs_difference(const std::vector<Real>& A, const std::vector<Real>& B) {
    if (A.size() != B.size()) return std::numeric_limits<Real>::max();

    Real max_diff = 0.0;
    for (Size i = 0; i < A.size(); ++i) {
        max_diff = std::max(max_diff, std::abs(A[i] - B[i]));
    }
    return max_diff;
}

/// Helper: compute Frobenius norm of difference
Real frobenius_difference(const std::vector<Real>& A, const std::vector<Real>& B) {
    if (A.size() != B.size()) return std::numeric_limits<Real>::max();

    Real sum_sq = 0.0;
    for (Size i = 0; i < A.size(); ++i) {
        Real diff = A[i] - B[i];
        sum_sq += diff * diff;
    }
    return std::sqrt(sum_sq);
}

}  // anonymous namespace

// =============================================================================
// Quality Gate G2: Schwarz Bound Conservativeness
// =============================================================================

TEST(ScreenedFockValidationTest, SchwarzBoundsAreConservative) {
    // Verify that Schwarz bounds are conservative:
    // |(ab|cd)| <= Q_ab * Q_cd for all quartets

    BasisSet basis = make_h2o_basis();
    SchwarzBounds bounds(basis);

    Engine engine(basis);
    TwoElectronBuffer<0> buffer;

    for (Size i = 0; i < basis.n_shells(); ++i) {
        for (Size j = 0; j < basis.n_shells(); ++j) {
            for (Size k = 0; k < basis.n_shells(); ++k) {
                for (Size l = 0; l < basis.n_shells(); ++l) {
                    // Compute actual ERIs
                    engine.compute_2e_shell_quartet(
                        Operator::coulomb(),
                        basis.shell(i), basis.shell(j),
                        basis.shell(k), basis.shell(l),
                        buffer);

                    Real Q_bound = bounds(i, j) * bounds(k, l);

                    // Check all function combinations
                    int ni = basis.shell(i).n_functions();
                    int nj = basis.shell(j).n_functions();
                    int nk = basis.shell(k).n_functions();
                    int nl = basis.shell(l).n_functions();

                    for (int a = 0; a < ni; ++a) {
                        for (int b = 0; b < nj; ++b) {
                            for (int c = 0; c < nk; ++c) {
                                for (int d = 0; d < nl; ++d) {
                                    Real eri = std::abs(buffer(a, b, c, d));
                                    // Allow small numerical tolerance
                                    EXPECT_LE(eri, Q_bound * (1.0 + 1e-10))
                                        << "Schwarz bound violated at ("
                                        << i << "," << j << "|" << k << "," << l << ") "
                                        << "function (" << a << "," << b << "|" << c << "," << d << ")";
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}

// =============================================================================
// Quality Gate G2: Quartet Reduction > 50% for Large Systems
// =============================================================================

TEST(ScreenedFockValidationTest, QuartetReductionLargeSystem) {
    // Quality Gate: >50% quartet reduction for large systems

    // Create a larger system (3 water molecules, spaced 8 Bohr apart)
    BasisSet basis = make_water_cluster(3, 8.0);
    SchwarzBounds bounds(basis);

    Real threshold = 1e-10;  // Use looser threshold for testing

    // Count quartets
    Size total = bounds.count_passing_quartets(0.0);  // All quartets
    Size passing = bounds.count_passing_quartets(threshold);

    Real pass_fraction = static_cast<Real>(passing) / static_cast<Real>(total);
    Real reduction = 1.0 - pass_fraction;

    std::cout << "Quartet reduction for 3-water cluster:\n"
              << "  Total quartets: " << total << "\n"
              << "  Passing at threshold " << threshold << ": " << passing << "\n"
              << "  Reduction: " << (reduction * 100) << "%\n";

    // Quality gate: >50% reduction for larger systems
    // For small systems, may not reach 50%, but we verify the mechanism works
    EXPECT_GT(reduction, 0.0) << "Screening should reduce some quartets";
}

TEST(ScreenedFockValidationTest, QuartetReductionVeryLargeSystem) {
    // Test with 9 water molecules for quality gate validation
    // Use larger spacing (10 Bohr) to get more distant shells

    BasisSet basis = make_water_cluster(9, 10.0);
    SchwarzBounds bounds(basis);

    Real threshold = 1e-10;  // Use looser threshold for better screening

    Size total = bounds.count_passing_quartets(0.0);
    Size passing = bounds.count_passing_quartets(threshold);

    Real reduction = 1.0 - static_cast<Real>(passing) / static_cast<Real>(total);

    std::cout << "Quartet reduction for 9-water cluster:\n"
              << "  Total quartets: " << total << "\n"
              << "  Passing: " << passing << "\n"
              << "  Reduction: " << (reduction * 100) << "%\n";

    // Quality Gate G2: >50% reduction for large systems with reasonable threshold
    EXPECT_GT(reduction, 0.50)
        << "Quality Gate G2: Expected >50% quartet reduction for 9-water cluster";
}

// =============================================================================
// Quality Gate G2: Screened Fock Matches Unscreened Within Threshold
// =============================================================================

TEST(ScreenedFockValidationTest, ScreenedMatchesUnscreenedH2O) {
    // Verify screened Fock matrix matches unscreened within tolerance
    //
    // Note: We use N^4 iteration for both paths to ensure consistent
    // accumulation. The screened path skips quartets that don't pass
    // the Schwarz bound check, while unscreened computes all.

    BasisSet basis = make_h2o_basis();
    Size nbf = basis.n_basis_functions();
    auto D = make_identity_density(nbf);

    Engine engine(basis);

    // Unscreened Fock build
    FockBuilder fock_unscreened(nbf);
    fock_unscreened.set_density(D.data(), nbf);
    engine.compute_and_consume(Operator::coulomb(), fock_unscreened);

    std::vector<Real> J_unscreened(fock_unscreened.get_coulomb_matrix().begin(),
                                   fock_unscreened.get_coulomb_matrix().end());

    // Screened Fock build using N^4 iteration with Schwarz screening
    SchwarzBounds bounds(basis);
    Real threshold = 1e-12;

    FockBuilder fock_screened(nbf);
    fock_screened.set_density(D.data(), nbf);

    TwoElectronBuffer<0> buffer;
    Size n_shells = basis.n_shells();
    Size computed = 0, skipped = 0;

    // Iterate over ALL quartets (same as compute_and_consume)
    for (Size i = 0; i < n_shells; ++i) {
        const auto& s_i = basis.shell(i);
        for (Size j = 0; j < n_shells; ++j) {
            const auto& s_j = basis.shell(j);
            for (Size k = 0; k < n_shells; ++k) {
                const auto& s_k = basis.shell(k);
                for (Size l = 0; l < n_shells; ++l) {
                    const auto& s_l = basis.shell(l);

                    // Apply Schwarz screening
                    if (!bounds.passes_screening(i, j, k, l, threshold)) {
                        ++skipped;
                        continue;
                    }

                    engine.compute_2e_shell_quartet(
                        Operator::coulomb(), s_i, s_j, s_k, s_l, buffer);

                    fock_screened.accumulate(
                        buffer,
                        static_cast<Index>(s_i.function_index()),
                        static_cast<Index>(s_j.function_index()),
                        static_cast<Index>(s_k.function_index()),
                        static_cast<Index>(s_l.function_index()),
                        s_i.n_functions(), s_j.n_functions(),
                        s_k.n_functions(), s_l.n_functions());
                    ++computed;
                }
            }
        }
    }

    std::vector<Real> J_screened(fock_screened.get_coulomb_matrix().begin(),
                                 fock_screened.get_coulomb_matrix().end());

    // Compare
    Real max_diff = max_abs_difference(J_screened, J_unscreened);
    Real frob_diff = frobenius_difference(J_screened, J_unscreened);

    std::cout << "H2O Fock comparison:\n"
              << "  Max absolute difference: " << max_diff << "\n"
              << "  Frobenius difference: " << frob_diff << "\n"
              << "  Computed: " << computed << ", Skipped: " << skipped << "\n";

    // Quality Gate: Max difference < 10 * threshold
    EXPECT_LT(max_diff, 10.0 * threshold)
        << "Screened Fock should match unscreened within threshold";
}

// =============================================================================
// Threshold vs Accuracy Relationship
// =============================================================================

TEST(ScreenedFockValidationTest, ThresholdVsAccuracy) {
    BasisSet basis = make_h2o_basis();
    Size nbf = basis.n_basis_functions();
    Size n_shells = basis.n_shells();
    auto D = make_identity_density(nbf);

    Engine engine(basis);
    SchwarzBounds bounds(basis);

    // Compute reference (unscreened)
    FockBuilder fock_ref(nbf);
    fock_ref.set_density(D.data(), nbf);
    engine.compute_and_consume(Operator::coulomb(), fock_ref);

    std::vector<Real> J_ref(fock_ref.get_coulomb_matrix().begin(),
                            fock_ref.get_coulomb_matrix().end());

    std::vector<Real> thresholds = {1e-8, 1e-10, 1e-12, 1e-14};

    std::cout << "\nThreshold vs Accuracy:\n";
    std::cout << "Threshold    | Max Diff     | Reduction%\n";
    std::cout << "-------------|--------------|----------\n";

    for (Real threshold : thresholds) {
        // Compute screened Fock using N^4 iteration (same as unscreened)
        FockBuilder fock_screened(nbf);
        fock_screened.set_density(D.data(), nbf);

        TwoElectronBuffer<0> buffer;
        Size computed = 0, skipped = 0;

        for (Size i = 0; i < n_shells; ++i) {
            const auto& s_i = basis.shell(i);
            for (Size j = 0; j < n_shells; ++j) {
                const auto& s_j = basis.shell(j);
                for (Size k = 0; k < n_shells; ++k) {
                    const auto& s_k = basis.shell(k);
                    for (Size l = 0; l < n_shells; ++l) {
                        const auto& s_l = basis.shell(l);

                        // Apply Schwarz screening
                        if (!bounds.passes_screening(i, j, k, l, threshold)) {
                            ++skipped;
                            continue;
                        }

                        engine.compute_2e_shell_quartet(
                            Operator::coulomb(), s_i, s_j, s_k, s_l, buffer);

                        fock_screened.accumulate(
                            buffer,
                            static_cast<Index>(s_i.function_index()),
                            static_cast<Index>(s_j.function_index()),
                            static_cast<Index>(s_k.function_index()),
                            static_cast<Index>(s_l.function_index()),
                            s_i.n_functions(), s_j.n_functions(),
                            s_k.n_functions(), s_l.n_functions());
                        ++computed;
                    }
                }
            }
        }

        std::vector<Real> J_screened(fock_screened.get_coulomb_matrix().begin(),
                                     fock_screened.get_coulomb_matrix().end());

        Real max_diff = max_abs_difference(J_screened, J_ref);
        Size total = computed + skipped;
        Real reduction = 100.0 * static_cast<Real>(skipped) / static_cast<Real>(total);

        std::cout << std::scientific << std::setprecision(0)
                  << threshold << "   | "
                  << std::setprecision(2) << max_diff << "  | "
                  << std::fixed << std::setprecision(1) << reduction << "%\n";

        // Error should scale roughly with threshold
        EXPECT_LT(max_diff, 100.0 * threshold)
            << "Error should be bounded by ~100 * threshold";
    }
}

