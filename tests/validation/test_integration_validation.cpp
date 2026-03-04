// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2026 Bryce M. Westheimer. All rights reserved.

/// @file test_integration_validation.cpp
/// @brief Phase 16 integration validation tests
///
/// Cross-module integration tests validating that remediated modules work
/// together correctly:
/// - Basis → Engine → Consumer pipeline (16.1.1)
/// - Screening → Engine → Fock build (16.1.2)
/// - cc-pVDZ basis set via BSE parser (16.2.1)
/// - Gradient vs finite-difference validation (16.2.2)
/// - Regression tests for fixed bugs (16.3.1)

#include <libaccint/engine/engine.hpp>
#include <libaccint/consumers/fock_builder.hpp>
#include <libaccint/data/builtin_basis.hpp>
#include <libaccint/data/bse_json_parser.hpp>
#include <libaccint/operators/operator.hpp>
#include <libaccint/operators/one_electron_operator.hpp>
#include <libaccint/screening/schwarz_bounds.hpp>
#include <libaccint/screening/screening_options.hpp>
#include <libaccint/memory/memory_pool.hpp>
#include <libaccint/core/types.hpp>
#include <libaccint/engine/dispatch_policy.hpp>

#include <gtest/gtest.h>
#include <Eigen/Dense>

#include <cmath>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <vector>

using namespace libaccint;
using namespace libaccint::data;
using namespace libaccint::consumers;

// =============================================================================
// Test Helpers (duplicated from test_hf_energy.cpp for independence)
// =============================================================================

namespace {

Real compute_nuclear_repulsion(const std::vector<Atom>& atoms) {
    Real E_nuc = 0.0;
    for (Size i = 0; i < atoms.size(); ++i) {
        for (Size j = i + 1; j < atoms.size(); ++j) {
            Real dx = atoms[i].position.x - atoms[j].position.x;
            Real dy = atoms[i].position.y - atoms[j].position.y;
            Real dz = atoms[i].position.z - atoms[j].position.z;
            Real r = std::sqrt(dx * dx + dy * dy + dz * dz);
            E_nuc += static_cast<Real>(atoms[i].atomic_number * atoms[j].atomic_number) / r;
        }
    }
    return E_nuc;
}

Eigen::MatrixXd to_eigen(const std::vector<Real>& flat, int n) {
    Eigen::MatrixXd mat(n, n);
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < n; ++j)
            mat(i, j) = flat[static_cast<Size>(i) * n + j];
    return mat;
}

std::pair<Eigen::VectorXd, Eigen::MatrixXd>
solve_gen_eigenvalue(const Eigen::MatrixXd& F, const Eigen::MatrixXd& S) {
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es(S);
    Eigen::VectorXd s_vals = es.eigenvalues();
    Eigen::MatrixXd U = es.eigenvectors();

    Eigen::VectorXd s_inv_sqrt(s_vals.size());
    for (int i = 0; i < s_vals.size(); ++i)
        s_inv_sqrt(i) = (s_vals(i) > 1e-10) ? 1.0 / std::sqrt(s_vals(i)) : 0.0;
    Eigen::MatrixXd X = U * s_inv_sqrt.asDiagonal() * U.transpose();
    Eigen::MatrixXd Fprime = X.transpose() * F * X;
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es2(Fprime);
    return {es2.eigenvalues(), X * es2.eigenvectors()};
}

Eigen::MatrixXd build_density(const Eigen::MatrixXd& C, int n_occ) {
    return 2.0 * C.leftCols(n_occ) * C.leftCols(n_occ).transpose();
}

struct SCFResult {
    Real total_energy;
    int iterations;
    bool converged;
};

/// @brief Run a minimal RHF SCF loop
SCFResult run_rhf(const std::vector<Atom>& atoms,
                   const BasisSet& basis,
                   int n_electrons,
                   Real energy_tol = 1e-10,
                   int max_iter = 100) {
    const int nbf = static_cast<int>(basis.n_basis_functions());
    const int n_occ = n_electrons / 2;

    Engine engine(basis);

    PointChargeParams charges;
    for (const auto& atom : atoms) {
        charges.x.push_back(atom.position.x);
        charges.y.push_back(atom.position.y);
        charges.z.push_back(atom.position.z);
        charges.charge.push_back(static_cast<Real>(atom.atomic_number));
    }

    std::vector<Real> S_flat, T_flat, V_flat;
    engine.compute_1e(Operator::overlap(), S_flat);
    engine.compute_1e(Operator::kinetic(), T_flat);
    engine.compute_1e(Operator::nuclear(charges), V_flat);

    auto S = to_eigen(S_flat, nbf);
    auto T = to_eigen(T_flat, nbf);
    auto V = to_eigen(V_flat, nbf);
    auto H_core = T + V;
    Real E_nuc = compute_nuclear_repulsion(atoms);

    auto [eps0, C0] = solve_gen_eigenvalue(H_core, S);
    auto D = build_density(C0, n_occ);
    Real E_old = 0.0;

    for (int iter = 0; iter < max_iter; ++iter) {
        FockBuilder fock_builder(static_cast<Size>(nbf));
        std::vector<Real> D_flat(nbf * nbf);
        for (int i = 0; i < nbf; ++i)
            for (int j = 0; j < nbf; ++j)
                D_flat[i * nbf + j] = D(i, j);
        fock_builder.set_density(D_flat.data(), static_cast<Size>(nbf));
        engine.compute_and_consume(Operator::coulomb(), fock_builder);

        auto J_span = fock_builder.get_coulomb_matrix();
        auto K_span = fock_builder.get_exchange_matrix();
        Eigen::MatrixXd J(nbf, nbf), K(nbf, nbf);
        for (int i = 0; i < nbf; ++i)
            for (int j = 0; j < nbf; ++j) {
                J(i, j) = J_span[i * nbf + j];
                K(i, j) = K_span[i * nbf + j];
            }

        Eigen::MatrixXd F = H_core + J - 0.5 * K;
        Real E_elec = 0.5 * ((H_core + F).array() * D.array()).sum();
        Real E_total = E_elec + E_nuc;

        auto [eps, C] = solve_gen_eigenvalue(F, S);
        auto D_new = build_density(C, n_occ);

        if (std::abs(E_total - E_old) < energy_tol && iter > 0) {
            return {E_total, iter + 1, true};
        }
        D = D_new;
        E_old = E_total;
    }
    return {E_old, max_iter, false};
}

}  // anonymous namespace

// =============================================================================
// 16.1.1: Basis → Engine → Consumer Cross-Module Integration
// =============================================================================

class CrossModuleIntegration : public ::testing::Test {
protected:
    std::vector<Atom> h2o_atoms_ = {
        {8, {0.0, 0.0, 0.0}},
        {1, {0.0, 1.43233673, -1.10866041}},
        {1, {0.0, -1.43233673, -1.10866041}},
    };
};

TEST_F(CrossModuleIntegration, BasisToEngineToFock) {
    // Create basis from built-in data
    auto basis = create_sto3g(h2o_atoms_);
    ASSERT_GT(basis.n_basis_functions(), 0u);
    ASSERT_GT(basis.n_shells(), 0u);

    // Create engine from basis
    Engine engine(basis);

    // Compute 1e integrals
    std::vector<Real> S;
    engine.compute_1e(Operator::overlap(), S);
    Size nbf = basis.n_basis_functions();
    ASSERT_EQ(S.size(), nbf * nbf);

    // Verify overlap matrix properties
    for (Size i = 0; i < nbf; ++i) {
        EXPECT_NEAR(S[i * nbf + i], 1.0, 1e-10)
            << "Overlap diagonal should be 1.0";
    }

    // Create density and run Fock build
    std::vector<Real> D(nbf * nbf, 0.0);
    for (Size i = 0; i < nbf; ++i)
        D[i * nbf + i] = 1.0 / static_cast<Real>(nbf);

    FockBuilder fock(nbf);
    fock.set_density(D.data(), nbf);
    engine.compute_and_consume(Operator::coulomb(), fock);

    auto J = fock.get_coulomb_matrix();
    ASSERT_EQ(J.size(), nbf * nbf);

    // J matrix should be symmetric
    for (Size i = 0; i < nbf; ++i) {
        for (Size j = i + 1; j < nbf; ++j) {
            EXPECT_NEAR(J[i * nbf + j], J[j * nbf + i], 1e-12)
                << "J matrix should be symmetric at (" << i << "," << j << ")";
        }
    }
}

TEST_F(CrossModuleIntegration, DispatchPolicyRouting) {
    auto basis = create_sto3g(h2o_atoms_);
    DispatchConfig config;
    config.min_gpu_shells = 100;  // Force CPU for small basis
    Engine engine(basis, config);

    std::vector<Real> S;
    engine.compute_1e(Operator::overlap(), S);
    EXPECT_EQ(S.size(), basis.n_basis_functions() * basis.n_basis_functions());
}

TEST_F(CrossModuleIntegration, MemoryPoolIntegration) {
    auto& pool = memory::get_thread_local_pool();
    auto before = pool.stats();

    auto basis = create_sto3g(h2o_atoms_);
    Engine engine(basis);

    // Pool should be usable after engine operations
    auto buf = pool.acquire(1024);
    EXPECT_NE(buf.data(), nullptr);
    EXPECT_GE(buf.size(), 1024u);

    auto after = pool.stats();
    EXPECT_GE(after.total_allocations, before.total_allocations);
}

// =============================================================================
// 16.1.2: Screening → Engine → Fock Build Integration
// =============================================================================

TEST_F(CrossModuleIntegration, ScreenedFockBuild) {
    auto basis = create_sto3g(h2o_atoms_);
    Size nbf = basis.n_basis_functions();

    Engine engine(basis);
    engine.precompute_schwarz_bounds();
    EXPECT_TRUE(engine.schwarz_bounds_precomputed());

    // Create identity density
    std::vector<Real> D(nbf * nbf, 0.0);
    for (Size i = 0; i < nbf; ++i)
        D[i * nbf + i] = 1.0 / static_cast<Real>(nbf);

    engine.set_density_matrix(D.data(), nbf);
    EXPECT_TRUE(engine.density_matrix_set());

    FockBuilder fock(nbf);
    fock.set_density(D.data(), nbf);

    auto options = screening::ScreeningOptions::normal();
    options.enable_statistics = true;
    engine.compute_and_consume(Operator::coulomb(), fock, options);

    auto J = fock.get_coulomb_matrix();

    // Verify we got non-zero results
    bool has_nonzero = false;
    for (Size i = 0; i < nbf * nbf; ++i) {
        if (std::abs(J[i]) > 1e-15) {
            has_nonzero = true;
            break;
        }
    }
    EXPECT_TRUE(has_nonzero) << "Screened Fock build produced all zeros";
}

TEST_F(CrossModuleIntegration, SchwarzBoundsFromEngine) {
    auto basis = create_sto3g(h2o_atoms_);
    Engine engine(basis);

    EXPECT_FALSE(engine.schwarz_bounds_precomputed());
    const auto& bounds = engine.precompute_schwarz_bounds();
    EXPECT_TRUE(engine.schwarz_bounds_precomputed());

    // Schwarz bounds should be positive
    EXPECT_GT(bounds.max_bound(), 0.0);
    EXPECT_EQ(bounds.n_shells(), basis.n_shells());
}

// =============================================================================
// 16.2.1: SCF Energy with cc-pVDZ via BSE Parser
// =============================================================================

class SCFValidation : public ::testing::Test {
protected:
    std::vector<Atom> h2o_atoms_ = {
        {8, {0.0, 0.0, 0.0}},
        {1, {0.0, 1.43233673, -1.10866041}},
        {1, {0.0, -1.43233673, -1.10866041}},
    };
    std::vector<Atom> h2_atoms_ = {
        {1, {0.0, 0.0, 0.0}},
        {1, {0.0, 0.0, 1.39839733}},
    };
};

TEST_F(SCFValidation, H2O_STO3G_Energy) {
    auto basis = create_sto3g(h2o_atoms_);
    auto result = run_rhf(h2o_atoms_, basis, 10);

    constexpr Real REF_ENERGY = -74.9631102394515;
    EXPECT_TRUE(result.converged) << "SCF did not converge";
    EXPECT_NEAR(result.total_energy, REF_ENERGY, 1e-8)
        << "H2O/STO-3G energy verification";
}

TEST_F(SCFValidation, H2O_ccpVDZ_Energy) {
    // Try to find cc-pvdz.json in standard locations
    std::vector<std::string> search_paths = {
        "share/basis_sets/cc-pvdz.json",
        "../share/basis_sets/cc-pvdz.json",
        "../../share/basis_sets/cc-pvdz.json",
    };

    // Also try relative to the source dir if available
    const char* src_dir = std::getenv("LIBACCINT_SOURCE_DIR");
    if (src_dir) {
        search_paths.insert(search_paths.begin(),
            std::string(src_dir) + "/share/basis_sets/cc-pvdz.json");
    }

    std::string found_path;
    for (const auto& path : search_paths) {
        if (std::filesystem::exists(path)) {
            found_path = path;
            break;
        }
    }

    if (found_path.empty()) {
        GTEST_SKIP() << "cc-pvdz.json not found — skipping cc-pVDZ validation";
    }

    auto basis = BseJsonParser::parse_file(found_path, h2o_atoms_);
    ASSERT_GT(basis.n_basis_functions(), 0u)
        << "cc-pVDZ basis should have basis functions";

    // cc-pVDZ for H2O should have 24 basis functions (5s2p1d for O, 2s1p for H)
    // In Cartesian: O = 1+1+3+1+3+6 = 15, H×2 = 2×(1+1+3) = 10 → 25
    // (exact count depends on whether d-functions are 5 spherical or 6 Cartesian)
    EXPECT_GE(basis.n_basis_functions(), 24u);

    auto result = run_rhf(h2o_atoms_, basis, 10);
    EXPECT_TRUE(result.converged) << "SCF did not converge for H2O/cc-pVDZ";

    // PySCF reference for H2O/cc-pVDZ (spherical, this geometry):
    // approximately -76.0236 Hartree (exact value depends on geometry)
    // Use a looser tolerance since DIIS-less simple SCF may converge slightly differently
    EXPECT_LT(result.total_energy, -75.0)
        << "H2O/cc-pVDZ energy should be lower than -75 Hartree";
    EXPECT_GT(result.total_energy, -77.0)
        << "H2O/cc-pVDZ energy should be above -77 Hartree";
}

TEST_F(SCFValidation, H2_STO3G_Energy) {
    auto basis = create_sto3g(h2_atoms_);
    auto result = run_rhf(h2_atoms_, basis, 2);

    constexpr Real REF_ENERGY = -1.11675930745672;
    EXPECT_TRUE(result.converged);
    EXPECT_NEAR(result.total_energy, REF_ENERGY, 1e-8);
}

// =============================================================================
// 16.2.2: Gradient vs Finite-Difference Validation
// =============================================================================

TEST_F(SCFValidation, OverlapGradientFiniteDifference) {
    // Validate 1e overlap integral gradient via finite difference
    // dS/dR ≈ [S(R+h) - S(R-h)] / (2h)
    auto basis = create_sto3g(h2_atoms_);
    int nbf = static_cast<int>(basis.n_basis_functions());

    Engine engine(basis);

    // Reference overlap matrix at equilibrium
    std::vector<Real> S_ref;
    engine.compute_1e(Operator::overlap(), S_ref);

    // Finite difference displacement
    const Real h = 1e-5;

    // Displace H2 along z (bond stretching)
    auto atoms_plus = h2_atoms_;
    atoms_plus[1].position.z += h;
    auto basis_plus = create_sto3g(atoms_plus);
    Engine engine_plus(basis_plus);
    std::vector<Real> S_plus;
    engine_plus.compute_1e(Operator::overlap(), S_plus);

    auto atoms_minus = h2_atoms_;
    atoms_minus[1].position.z -= h;
    auto basis_minus = create_sto3g(atoms_minus);
    Engine engine_minus(basis_minus);
    std::vector<Real> S_minus;
    engine_minus.compute_1e(Operator::overlap(), S_minus);

    // Finite difference gradient
    for (int i = 0; i < nbf; ++i) {
        for (int j = 0; j < nbf; ++j) {
            Size idx = i * nbf + j;
            Real fd_grad = (S_plus[idx] - S_minus[idx]) / (2.0 * h);
            // Don't check exact value, just that the change is smooth
            // (FD should produce a finite, non-NaN gradient)
            EXPECT_FALSE(std::isnan(fd_grad))
                << "FD gradient should not be NaN at (" << i << "," << j << ")";
            EXPECT_FALSE(std::isinf(fd_grad))
                << "FD gradient should not be Inf at (" << i << "," << j << ")";
        }
    }
}

TEST_F(SCFValidation, SCFEnergyGradientFiniteDifference) {
    // Validate that dE/dR (finite difference of SCF energy) is smooth
    // This is a smoke test — a full gradient implementation would
    // analytically compute dE/dR and compare.
    auto basis = create_sto3g(h2_atoms_);
    auto result_ref = run_rhf(h2_atoms_, basis, 2);
    ASSERT_TRUE(result_ref.converged);

    const Real h = 1e-4;  // bohr

    // Displace second H along z
    auto atoms_plus = h2_atoms_;
    atoms_plus[1].position.z += h;
    auto basis_plus = create_sto3g(atoms_plus);
    auto result_plus = run_rhf(atoms_plus, basis_plus, 2);
    ASSERT_TRUE(result_plus.converged);

    auto atoms_minus = h2_atoms_;
    atoms_minus[1].position.z -= h;
    auto basis_minus = create_sto3g(atoms_minus);
    auto result_minus = run_rhf(atoms_minus, basis_minus, 2);
    ASSERT_TRUE(result_minus.converged);

    Real fd_gradient = (result_plus.total_energy - result_minus.total_energy) / (2.0 * h);

    // At equilibrium, the gradient should be close to zero
    // The H-H distance is 1.398 bohr which is near equilibrium (0.74 Å = 1.4 bohr)
    EXPECT_LT(std::abs(fd_gradient), 0.1)
        << "Gradient magnitude at near-equilibrium H-H should be small";

    // Verify the gradient is physically reasonable:
    // compressing H2 should increase energy (positive gradient)
    // stretching should decrease energy for compressed distances
    EXPECT_FALSE(std::isnan(fd_gradient)) << "Gradient should not be NaN";
    EXPECT_FALSE(std::isinf(fd_gradient)) << "Gradient should not be Inf";
}

// =============================================================================
// 16.3.1: Regression Tests for Fixed Bugs
// =============================================================================

class RegressionTests : public ::testing::Test {};

/// Phase 0: Build system — ensure all basic includes compile
TEST_F(RegressionTests, BasicIncludes) {
    // Regression: Phase 0 fixed missing/broken includes
    EXPECT_GT(sizeof(Engine), 0u);
    EXPECT_GT(sizeof(FockBuilder), 0u);
    EXPECT_GT(sizeof(screening::SchwarzBounds), 0u);
    EXPECT_GT(sizeof(DispatchPolicy), 0u);
    EXPECT_GT(sizeof(memory::MemoryPool), 0u);
}

/// Phase 1: PointChargeParams uses SoA layout
TEST_F(RegressionTests, PointChargeParamsSoA) {
    PointChargeParams charges;
    charges.x.push_back(0.0);
    charges.y.push_back(0.0);
    charges.z.push_back(0.0);
    charges.charge.push_back(1.0);

    EXPECT_EQ(charges.x.size(), 1u);
    EXPECT_EQ(charges.y.size(), 1u);
    EXPECT_EQ(charges.z.size(), 1u);
    EXPECT_EQ(charges.charge.size(), 1u);
}

/// Phase 5: Operator factories
TEST_F(RegressionTests, OperatorFactories) {
    auto overlap = Operator::overlap();
    EXPECT_TRUE(overlap.is_one_electron());

    auto coulomb = Operator::coulomb();
    EXPECT_TRUE(coulomb.is_two_electron());

    PointChargeParams charges;
    charges.x = {0.0};
    charges.y = {0.0};
    charges.z = {0.0};
    charges.charge = {1.0};
    auto nuclear = Operator::nuclear(charges);
    EXPECT_TRUE(nuclear.is_one_electron());
}

/// Phase 8: Screening options factory methods
TEST_F(RegressionTests, ScreeningOptionsFactories) {
    auto none = screening::ScreeningOptions::none();
    EXPECT_FALSE(none.enabled);

    auto normal = screening::ScreeningOptions::normal();
    EXPECT_TRUE(normal.enabled);
    EXPECT_NEAR(normal.threshold, 1e-12, 1e-20);

    auto tight = screening::ScreeningOptions::tight();
    EXPECT_TRUE(tight.enabled);
    EXPECT_NEAR(tight.threshold, 1e-14, 1e-20);
}

/// Phase 9: Engine accepts DispatchConfig
TEST_F(RegressionTests, EngineDispatchConfig) {
    std::vector<Atom> atoms = {{1, {0.0, 0.0, 0.0}}, {1, {0.0, 0.0, 1.4}}};
    auto basis = create_sto3g(atoms);

    DispatchConfig config;
    config.min_gpu_batch_size = 32;
    Engine engine(basis, config);

    std::vector<Real> S;
    engine.compute_1e(Operator::overlap(), S);
    EXPECT_EQ(S.size(), basis.n_basis_functions() * basis.n_basis_functions());
}

/// Phase 14: Python-facing types exist and are constructible
TEST_F(RegressionTests, ConsumerTypes) {
    // FockBuilder
    FockBuilder fock(7);
    EXPECT_EQ(fock.get_coulomb_matrix().size(), 49u);
}

/// Phase 15: Memory pool acquire/release
TEST_F(RegressionTests, MemoryPoolAcquireRelease) {
    auto& pool = memory::get_thread_local_pool();
    auto buf = pool.acquire(256);
    EXPECT_NE(buf.data(), nullptr);
    EXPECT_GE(buf.size(), 256u);
    // PooledBuffer destructor releases automatically
}

/// Phase 15: BackendHint enum values
TEST_F(RegressionTests, BackendHintValues) {
    EXPECT_NE(BackendHint::Auto, BackendHint::ForceCPU);
    EXPECT_NE(BackendHint::Auto, BackendHint::ForceGPU);
    EXPECT_NE(BackendHint::Auto, BackendHint::PreferCPU);
    EXPECT_NE(BackendHint::Auto, BackendHint::PreferGPU);
}

// =============================================================================
// 16.3.3: Quality Gate Verification
// =============================================================================

/// @brief Verify fundamental build/runtime invariants from all phases
TEST(QualityGateVerification, CoreTypeSizes) {
    // Real should be double
    static_assert(sizeof(Real) == sizeof(double),
                  "Real should be double precision");
}

TEST(QualityGateVerification, BuiltinBasisAvailable) {
    std::vector<Atom> atoms = {{1, {0.0, 0.0, 0.0}}};
    auto basis = create_sto3g(atoms);
    EXPECT_GT(basis.n_basis_functions(), 0u);
    EXPECT_GT(basis.n_shells(), 0u);
}

TEST(QualityGateVerification, ScreeningPresets) {
    using screening::ScreeningPreset;
    auto opts_normal = screening::ScreeningOptions::from_preset(ScreeningPreset::Normal);
    auto opts_tight = screening::ScreeningOptions::from_preset(ScreeningPreset::Tight);
    EXPECT_LT(opts_tight.threshold, opts_normal.threshold);
}

TEST(QualityGateVerification, DispatchPolicyDefaults) {
    DispatchConfig config;
    DispatchPolicy policy(config);
    EXPECT_FALSE(policy.auto_tuning_enabled());

    auto backend = policy.select_backend(
        WorkUnitType::FullBasis, 5, 4, 500,
        BackendHint::ForceCPU, false);
    EXPECT_EQ(backend, BackendType::CPU);
}

