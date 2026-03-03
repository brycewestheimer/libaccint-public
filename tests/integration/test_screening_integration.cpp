// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2026 Bryce M. Westheimer. All rights reserved.

/// @file test_screening_integration.cpp
/// @brief End-to-end integration tests for screening module (Task 8.4.3)
///
/// Validates the complete screening pipeline from basis set through
/// Schwarz/density screening to final Fock matrix construction, using
/// real molecular geometries and production basis sets.

#include <libaccint/engine/engine.hpp>
#include <libaccint/consumers/fock_builder.hpp>
#include <libaccint/data/builtin_basis.hpp>
#include <libaccint/operators/operator.hpp>
#include <libaccint/screening/schwarz_bounds.hpp>
#include <libaccint/screening/density_screening.hpp>
#include <libaccint/screening/screening_options.hpp>
#include <libaccint/kernels/eri_kernel.hpp>
#include <libaccint/buffers/two_electron_buffer.hpp>
#include <libaccint/core/types.hpp>

#include <gtest/gtest.h>
#include <cmath>
#include <random>
#include <vector>

using namespace libaccint;
using namespace libaccint::data;
using namespace libaccint::consumers;
using namespace libaccint::screening;
using namespace libaccint::kernels;

namespace {

constexpr Real TOL_EXACT  = 1e-12;  // For results that must be bitwise identical
constexpr Real TOL_SCREEN = 1e-8;   // For screened vs unscreened tolerance

BasisSet make_h2o_sto3g() {
    std::vector<Atom> atoms = {
        {8, {0.0, 0.0, 0.0}},
        {1, {0.0, 1.43233673, -1.10866041}},
        {1, {0.0, -1.43233673, -1.10866041}},
    };
    return create_sto3g(atoms);
}

BasisSet make_ch4_sto3g() {
    // Tetrahedral CH4, bond length ~2.05 Bohr
    constexpr double d = 2.05 / std::numbers::sqrt3;
    std::vector<Atom> atoms = {
        {6, {0.0, 0.0, 0.0}},
        {1, { d,  d,  d}},
        {1, { d, -d, -d}},
        {1, {-d,  d, -d}},
        {1, {-d, -d,  d}},
    };
    return create_sto3g(atoms);
}

std::vector<Real> make_random_density(Size nbf, unsigned seed = 42) {
    std::mt19937 gen(seed);
    std::uniform_real_distribution<double> dist(-0.5, 0.5);
    std::vector<Real> D(nbf * nbf);
    for (Size i = 0; i < nbf; ++i) {
        for (Size j = i; j < nbf; ++j) {
            double val = dist(gen);
            D[i * nbf + j] = val;
            D[j * nbf + i] = val;
        }
    }
    return D;
}

std::vector<Real> make_uniform_density(Size nbf, Real val = 1.0) {
    std::vector<Real> D(nbf * nbf, 0.0);
    for (Size i = 0; i < nbf; ++i) {
        for (Size j = 0; j < nbf; ++j) {
            D[i * nbf + j] = val;
        }
    }
    return D;
}

Real max_abs_difference(const Real* A, const Real* B, Size n) {
    Real max_diff = 0.0;
    for (Size i = 0; i < n * n; ++i) {
        max_diff = std::max(max_diff, std::abs(A[i] - B[i]));
    }
    return max_diff;
}

void expect_matrices_equal(const Real* A, const Real* B, Size n,
                           Real tol, const std::string& name) {
    for (Size i = 0; i < n; ++i) {
        for (Size j = 0; j < n; ++j) {
            EXPECT_NEAR(A[i * n + j], B[i * n + j], tol)
                << name << "[" << i << "," << j << "] mismatch";
        }
    }
}

}  // anonymous namespace

// =============================================================================
// Screened vs Unscreened Fock Correctness
// =============================================================================

TEST(ScreeningIntegrationTest, SchwarzScreenedMatchesUnscreenedH2O) {
    auto basis = make_h2o_sto3g();
    const Size nbf = basis.n_basis_functions();
    auto D = make_random_density(nbf);

    Engine engine(basis);

    // Unscreened reference
    FockBuilder fock_ref(nbf);
    fock_ref.set_density(D.data(), nbf);
    engine.compute_and_consume(Operator::coulomb(), fock_ref);

    // Schwarz-screened
    ScreeningOptions opts;
    opts.enabled = true;
    opts.threshold = 1e-12;

    FockBuilder fock_screened(nbf);
    fock_screened.set_density(D.data(), nbf);
    engine.compute_and_consume(Operator::coulomb(), fock_screened, opts);

    expect_matrices_equal(fock_ref.get_coulomb_matrix().data(),
                          fock_screened.get_coulomb_matrix().data(),
                          nbf, TOL_SCREEN, "J_schwarz");
    expect_matrices_equal(fock_ref.get_exchange_matrix().data(),
                          fock_screened.get_exchange_matrix().data(),
                          nbf, TOL_SCREEN, "K_schwarz");
}

TEST(ScreeningIntegrationTest, SchwarzScreenedMatchesUnscreenedCH4) {
    auto basis = make_ch4_sto3g();
    const Size nbf = basis.n_basis_functions();
    auto D = make_random_density(nbf, 123);

    Engine engine(basis);

    // Unscreened reference
    FockBuilder fock_ref(nbf);
    fock_ref.set_density(D.data(), nbf);
    engine.compute_and_consume(Operator::coulomb(), fock_ref);

    // Schwarz-screened with tight threshold
    ScreeningOptions opts;
    opts.enabled = true;
    opts.threshold = 1e-14;

    FockBuilder fock_screened(nbf);
    fock_screened.set_density(D.data(), nbf);
    engine.compute_and_consume(Operator::coulomb(), fock_screened, opts);

    expect_matrices_equal(fock_ref.get_coulomb_matrix().data(),
                          fock_screened.get_coulomb_matrix().data(),
                          nbf, TOL_SCREEN, "J_ch4");
    expect_matrices_equal(fock_ref.get_exchange_matrix().data(),
                          fock_screened.get_exchange_matrix().data(),
                          nbf, TOL_SCREEN, "K_ch4");
}

// =============================================================================
// Density-Weighted Screening
// =============================================================================

TEST(ScreeningIntegrationTest, DensityWeightedScreeningCorrectness) {
    auto basis = make_h2o_sto3g();
    const Size nbf = basis.n_basis_functions();
    // Use uniform density so density screening passes same quartets as Schwarz-only
    auto D = make_uniform_density(nbf);

    Engine engine(basis);

    // Schwarz-only reference
    ScreeningOptions opts_schwarz;
    opts_schwarz.enabled = true;
    opts_schwarz.threshold = 1e-10;

    FockBuilder fock_schwarz(nbf);
    fock_schwarz.set_density(D.data(), nbf);
    engine.compute_and_consume(Operator::coulomb(), fock_schwarz, opts_schwarz);

    // Density-weighted: uniform density means D_max=1 everywhere,
    // so result should match pure Schwarz
    ScreeningOptions opts_density;
    opts_density.enabled = true;
    opts_density.threshold = 1e-10;
    opts_density.density_weighted = true;

    engine.set_density_matrix(D.data(), nbf);

    FockBuilder fock_density(nbf);
    fock_density.set_density(D.data(), nbf);
    engine.compute_and_consume(Operator::coulomb(), fock_density, opts_density);

    expect_matrices_equal(fock_schwarz.get_coulomb_matrix().data(),
                          fock_density.get_coulomb_matrix().data(),
                          nbf, TOL_EXACT, "J_density_uniform");
    expect_matrices_equal(fock_schwarz.get_exchange_matrix().data(),
                          fock_density.get_exchange_matrix().data(),
                          nbf, TOL_EXACT, "K_density_uniform");
}

TEST(ScreeningIntegrationTest, DensityWeightedRequiresDensityMatrix) {
    auto basis = make_h2o_sto3g();
    const Size nbf = basis.n_basis_functions();
    auto D = make_random_density(nbf);

    Engine engine(basis);

    ScreeningOptions opts;
    opts.enabled = true;
    opts.density_weighted = true;

    FockBuilder fock(nbf);
    fock.set_density(D.data(), nbf);

    // Should throw because no density matrix set on engine
    EXPECT_THROW(engine.compute_and_consume(Operator::coulomb(), fock, opts),
                 InvalidArgumentException);
}

// =============================================================================
// 8-Fold Symmetry Integration
// =============================================================================

TEST(ScreeningIntegrationTest, SymmetryScreenedMatchesFullH2O) {
    auto basis = make_h2o_sto3g();
    const Size nbf = basis.n_basis_functions();
    auto D = make_random_density(nbf);

    Engine engine(basis);

    // Full (no symmetry) reference
    ScreeningOptions opts_full;
    opts_full.enabled = true;
    opts_full.threshold = 1e-12;
    opts_full.use_permutation_symmetry = false;

    FockBuilder fock_full(nbf);
    fock_full.set_density(D.data(), nbf);
    engine.compute_and_consume(Operator::coulomb(), fock_full, opts_full);

    // With 8-fold symmetry
    ScreeningOptions opts_sym;
    opts_sym.enabled = true;
    opts_sym.threshold = 1e-12;
    opts_sym.use_permutation_symmetry = true;

    FockBuilder fock_sym(nbf);
    fock_sym.set_density(D.data(), nbf);
    engine.compute_and_consume(Operator::coulomb(), fock_sym, opts_sym);

    expect_matrices_equal(fock_full.get_coulomb_matrix().data(),
                          fock_sym.get_coulomb_matrix().data(),
                          nbf, TOL_EXACT, "J_sym");
    expect_matrices_equal(fock_full.get_exchange_matrix().data(),
                          fock_sym.get_exchange_matrix().data(),
                          nbf, TOL_EXACT, "K_sym");
}

TEST(ScreeningIntegrationTest, SymmetryScreenedMatchesFullCH4) {
    auto basis = make_ch4_sto3g();
    const Size nbf = basis.n_basis_functions();
    auto D = make_random_density(nbf, 99);

    Engine engine(basis);

    // Full reference
    FockBuilder fock_full(nbf);
    fock_full.set_density(D.data(), nbf);
    engine.compute_and_consume(Operator::coulomb(), fock_full);

    // With symmetry
    ScreeningOptions opts_sym;
    opts_sym.enabled = true;
    opts_sym.threshold = 1e-14;
    opts_sym.use_permutation_symmetry = true;

    FockBuilder fock_sym(nbf);
    fock_sym.set_density(D.data(), nbf);
    engine.compute_and_consume(Operator::coulomb(), fock_sym, opts_sym);

    expect_matrices_equal(fock_full.get_coulomb_matrix().data(),
                          fock_sym.get_coulomb_matrix().data(),
                          nbf, TOL_EXACT, "J_ch4_sym");
    expect_matrices_equal(fock_full.get_exchange_matrix().data(),
                          fock_sym.get_exchange_matrix().data(),
                          nbf, TOL_EXACT, "K_ch4_sym");
}

// =============================================================================
// Combined: Symmetry + Density-Weighted Screening
// =============================================================================

TEST(ScreeningIntegrationTest, CombinedSymmetryAndDensityScreening) {
    auto basis = make_h2o_sto3g();
    const Size nbf = basis.n_basis_functions();
    auto D = make_random_density(nbf);

    Engine engine(basis);
    engine.set_density_matrix(D.data(), nbf);

    // Unscreened reference
    FockBuilder fock_ref(nbf);
    fock_ref.set_density(D.data(), nbf);
    engine.compute_and_consume(Operator::coulomb(), fock_ref);

    // All screening features enabled
    ScreeningOptions opts;
    opts.enabled = true;
    opts.threshold = 1e-12;
    opts.density_weighted = true;
    opts.use_permutation_symmetry = true;

    FockBuilder fock_combined(nbf);
    fock_combined.set_density(D.data(), nbf);
    engine.compute_and_consume(Operator::coulomb(), fock_combined, opts);

    // Combined screening should still produce accurate results
    expect_matrices_equal(fock_ref.get_coulomb_matrix().data(),
                          fock_combined.get_coulomb_matrix().data(),
                          nbf, TOL_SCREEN, "J_combined");
    expect_matrices_equal(fock_ref.get_exchange_matrix().data(),
                          fock_combined.get_exchange_matrix().data(),
                          nbf, TOL_SCREEN, "K_combined");
}

// =============================================================================
// Parallel + Screening Pipeline
// =============================================================================

TEST(ScreeningIntegrationTest, ParallelScreenedMatchesSerial) {
    auto basis = make_h2o_sto3g();
    const Size nbf = basis.n_basis_functions();
    auto D = make_random_density(nbf);

    Engine engine(basis);
    engine.precompute_schwarz_bounds();

    ScreeningOptions opts;
    opts.enabled = true;
    opts.threshold = 1e-12;

    // Serial screened reference
    FockBuilder fock_serial(nbf);
    fock_serial.set_density(D.data(), nbf);
    engine.compute_and_consume(Operator::coulomb(), fock_serial, opts);

    // Parallel screened
    FockBuilder fock_parallel(nbf);
    fock_parallel.set_threading_strategy(FockThreadingStrategy::ThreadLocal);
    fock_parallel.set_density(D.data(), nbf);
    fock_parallel.prepare_parallel(4);
    engine.compute_and_consume_screened_parallel(
        Operator::coulomb(), fock_parallel, opts, 4);
    fock_parallel.finalize_parallel();

    expect_matrices_equal(fock_serial.get_coulomb_matrix().data(),
                          fock_parallel.get_coulomb_matrix().data(),
                          nbf, TOL_EXACT, "J_parallel");
    expect_matrices_equal(fock_serial.get_exchange_matrix().data(),
                          fock_parallel.get_exchange_matrix().data(),
                          nbf, TOL_EXACT, "K_parallel");
}

// =============================================================================
// Screening Preset Integration
// =============================================================================

TEST(ScreeningIntegrationTest, PresetsProduceAccurateResults) {
    auto basis = make_h2o_sto3g();
    const Size nbf = basis.n_basis_functions();
    auto D = make_random_density(nbf);

    Engine engine(basis);

    // Unscreened reference
    FockBuilder fock_ref(nbf);
    fock_ref.set_density(D.data(), nbf);
    engine.compute_and_consume(Operator::coulomb(), fock_ref);

    // Test all presets
    for (auto preset : {ScreeningPreset::Tight, ScreeningPreset::Normal,
                        ScreeningPreset::Loose}) {
        auto opts = ScreeningOptions::from_preset(preset);

        FockBuilder fock_test(nbf);
        fock_test.set_density(D.data(), nbf);
        engine.compute_and_consume(Operator::coulomb(), fock_test, opts);

        Real j_diff = max_abs_difference(
            fock_ref.get_coulomb_matrix().data(),
            fock_test.get_coulomb_matrix().data(), nbf);
        Real k_diff = max_abs_difference(
            fock_ref.get_exchange_matrix().data(),
            fock_test.get_exchange_matrix().data(), nbf);

        // Even loose screening should be within reasonable tolerance
        EXPECT_LT(j_diff, 1e-6) << "J error too large for preset " << static_cast<int>(preset);
        EXPECT_LT(k_diff, 1e-6) << "K error too large for preset " << static_cast<int>(preset);
    }
}

// =============================================================================
// Schwarz Bounds Pipeline Validation
// =============================================================================

TEST(ScreeningIntegrationTest, SchwarzBoundsAreUpperBounds) {
    auto basis = make_h2o_sto3g();
    const Size nshells = basis.n_shells();

    SchwarzBounds bounds(basis);

    Engine engine(basis);

    // Verify that Schwarz bound >= actual integral for all shell quartets
    for (Size i = 0; i < nshells; ++i) {
        for (Size j = i; j < nshells; ++j) {
            for (Size k = i; k < nshells; ++k) {
                Size l_start = (k == i) ? j : k;
                for (Size l = l_start; l < nshells; ++l) {
                    // bounds(i,j) already returns Q_ij = sqrt(max|(ij|ij)|)
                    // Schwarz inequality: |(ij|kl)| <= Q_ij * Q_kl
                    Real bound = bounds(i, j) * bounds(k, l);

                    // Compute actual integrals for this shell quartet
                    auto& si = basis.shell(i);
                    auto& sj = basis.shell(j);
                    auto& sk = basis.shell(k);
                    auto& sl = basis.shell(l);

                    Size ni = si.n_functions();
                    Size nj = sj.n_functions();
                    Size nk = sk.n_functions();
                    Size nl = sl.n_functions();

                    TwoElectronBuffer<0> buffer;
                    kernels::compute_eri(si, sj, sk, sl, buffer);

                    Real max_eri = 0.0;
                    for (Size a = 0; a < ni; ++a)
                        for (Size b = 0; b < nj; ++b)
                            for (Size c = 0; c < nk; ++c)
                                for (Size d = 0; d < nl; ++d)
                                    max_eri = std::max(max_eri,
                                        std::abs(buffer(a, b, c, d)));

                    EXPECT_LE(max_eri, bound * 1.01)  // 1% tolerance for numerics
                        << "Schwarz bound violated for (" << i << "," << j
                        << "|" << k << "," << l << ")";
                }
            }
        }
    }
}
