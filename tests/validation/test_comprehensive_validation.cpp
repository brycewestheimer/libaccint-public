// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2026 Bryce M. Westheimer. All rights reserved.

/// @file test_comprehensive_validation.cpp
/// @brief Phase 17 — Comprehensive validation against PySCF reference values
///
/// Tests ALL molecule/basis combinations (H2O, CH4 × STO-3G, 6-31G, aug-cc-pVDZ):
/// - One-electron integral properties (overlap, kinetic, nuclear)
/// - Two-electron Fock build correctness
/// - Full RHF SCF energy vs PySCF reference
/// - Screening correctness
/// - Parallel computation consistency

#include <libaccint/engine/engine.hpp>
#include <libaccint/consumers/fock_builder.hpp>
#include <libaccint/data/builtin_basis.hpp>
#include <libaccint/data/bse_json_parser.hpp>
#include <libaccint/operators/operator.hpp>
#include <libaccint/operators/one_electron_operator.hpp>
#include <libaccint/screening/schwarz_bounds.hpp>
#include <libaccint/screening/screening_options.hpp>
#include <libaccint/core/types.hpp>
#include <libaccint/engine/dispatch_policy.hpp>

#include <gtest/gtest.h>
#include <Eigen/Dense>

#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <string>
#include <vector>

using namespace libaccint;
using namespace libaccint::data;
using namespace libaccint::consumers;

// =============================================================================
// Helper Utilities
// =============================================================================

namespace {

// ---- Molecule Geometries (Bohr) ----

std::vector<Atom> make_h2o_atoms() {
    return {
        {8, {0.0, 0.0, 0.0}},
        {1, {0.0, 1.43233673, -1.10866041}},
        {1, {0.0, -1.43233673, -1.10866041}},
    };
}

std::vector<Atom> make_ch4_atoms() {
    // Tetrahedral CH4 with C-H bond length r = 2.049803133 Bohr
    // H positions at (±r/√3, ±r/√3, ±r/√3)
    const double r = 2.049803133;
    const double d = r / std::sqrt(3.0);
    return {
        {6, {0.0, 0.0, 0.0}},
        {1, { d,  d,  d}},
        {1, { d, -d, -d}},
        {1, {-d,  d, -d}},
        {1, {-d, -d,  d}},
    };
}

// ---- Nuclear Repulsion ----

Real compute_nuclear_repulsion(const std::vector<Atom>& atoms) {
    Real E_nuc = 0.0;
    for (Size i = 0; i < atoms.size(); ++i) {
        for (Size j = i + 1; j < atoms.size(); ++j) {
            Real dx = atoms[i].position.x - atoms[j].position.x;
            Real dy = atoms[i].position.y - atoms[j].position.y;
            Real dz = atoms[i].position.z - atoms[j].position.z;
            Real r = std::sqrt(dx * dx + dy * dy + dz * dz);
            E_nuc += static_cast<Real>(atoms[i].atomic_number *
                                       atoms[j].atomic_number) / r;
        }
    }
    return E_nuc;
}

// ---- PointChargeParams from atoms ----

PointChargeParams make_charges(const std::vector<Atom>& atoms) {
    PointChargeParams charges;
    for (const auto& atom : atoms) {
        charges.x.push_back(atom.position.x);
        charges.y.push_back(atom.position.y);
        charges.z.push_back(atom.position.z);
        charges.charge.push_back(static_cast<Real>(atom.atomic_number));
    }
    return charges;
}

// ---- Eigen helpers ----

Eigen::MatrixXd to_eigen(const std::vector<Real>& flat, int n) {
    Eigen::MatrixXd mat(n, n);
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < n; ++j)
            mat(i, j) = flat[static_cast<Size>(i) * n + j];
    return mat;
}

std::vector<Real> from_eigen(const Eigen::MatrixXd& mat) {
    int n = static_cast<int>(mat.rows());
    std::vector<Real> flat(n * n);
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < n; ++j)
            flat[i * n + j] = mat(i, j);
    return flat;
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

// ---- SCF result struct ----

struct SCFResult {
    Real total_energy;
    int iterations;
    bool converged;
};

/// @brief Run a minimal RHF SCF loop using the LibAccInt engine (with DIIS)
SCFResult run_rhf(const std::vector<Atom>& atoms,
                  const BasisSet& basis,
                  int n_electrons,
                  Real energy_tol = 1e-10,
                  int max_iter = 200) {
    const int nbf = static_cast<int>(basis.n_basis_functions());
    const int n_occ = n_electrons / 2;

    Engine engine(basis);

    auto charges = make_charges(atoms);

    // Compute 1e integrals
    std::vector<Real> S_flat, T_flat, V_flat;
    engine.compute_1e(Operator::overlap(), S_flat);
    engine.compute_1e(Operator::kinetic(), T_flat);
    engine.compute_1e(Operator::nuclear(charges), V_flat);

    auto S = to_eigen(S_flat, nbf);
    auto T = to_eigen(T_flat, nbf);
    auto V = to_eigen(V_flat, nbf);
    auto H_core = T + V;
    Real E_nuc = compute_nuclear_repulsion(atoms);

    // Compute S^{-1/2} for orthogonalization (used in DIIS error)
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es_s(S);
    Eigen::VectorXd s_vals = es_s.eigenvalues();
    Eigen::MatrixXd s_vecs = es_s.eigenvectors();
    Eigen::VectorXd s_inv_half(s_vals.size());
    for (int i = 0; i < s_vals.size(); ++i)
        s_inv_half(i) = (s_vals(i) > 1e-10) ? 1.0 / std::sqrt(s_vals(i)) : 0.0;
    Eigen::MatrixXd S_inv_half = s_vecs * s_inv_half.asDiagonal() * s_vecs.transpose();

    // Initial guess from core Hamiltonian
    auto [eps0, C0] = solve_gen_eigenvalue(H_core, S);
    auto D = build_density(C0, n_occ);
    Real E_old = 0.0;

    // DIIS storage
    constexpr int diis_max = 8;
    std::vector<Eigen::MatrixXd> diis_focks;
    std::vector<Eigen::MatrixXd> diis_errors;

    for (int iter = 0; iter < max_iter; ++iter) {
        // Build J and K via FockBuilder
        FockBuilder fock_builder(static_cast<Size>(nbf));
        auto D_flat = from_eigen(D);
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

        // DIIS: compute error = FDS - SDF (in orthogonal basis)
        Eigen::MatrixXd err = F * D * S - S * D * F;
        // Transform to orthogonal basis
        Eigen::MatrixXd err_orth = S_inv_half.transpose() * err * S_inv_half;

        // Store DIIS vectors
        diis_focks.push_back(F);
        diis_errors.push_back(err_orth);
        if (static_cast<int>(diis_focks.size()) > diis_max) {
            diis_focks.erase(diis_focks.begin());
            diis_errors.erase(diis_errors.begin());
        }

        // DIIS extrapolation (if we have at least 2 vectors)
        if (diis_focks.size() >= 2) {
            int ndiis = static_cast<int>(diis_focks.size());
            Eigen::MatrixXd B = Eigen::MatrixXd::Zero(ndiis + 1, ndiis + 1);
            for (int i = 0; i < ndiis; ++i) {
                for (int j = i; j < ndiis; ++j) {
                    B(i, j) = (diis_errors[i].array() * diis_errors[j].array()).sum();
                    B(j, i) = B(i, j);
                }
                B(i, ndiis) = -1.0;
                B(ndiis, i) = -1.0;
            }
            Eigen::VectorXd rhs = Eigen::VectorXd::Zero(ndiis + 1);
            rhs(ndiis) = -1.0;

            // Solve B * c = rhs
            Eigen::VectorXd c = B.colPivHouseholderQr().solve(rhs);

            F = Eigen::MatrixXd::Zero(nbf, nbf);
            for (int i = 0; i < ndiis; ++i) {
                F += c(i) * diis_focks[i];
            }
        }

        Real E_elec = 0.5 * ((H_core + (H_core + J - 0.5 * K)).array() * D.array()).sum();
        Real E_total = E_elec + E_nuc;

        if (std::abs(E_total - E_old) < energy_tol && iter > 0) {
            return {E_total, iter + 1, true};
        }

        auto [eps, C] = solve_gen_eigenvalue(F, S);
        D = build_density(C, n_occ);
        E_old = E_total;
    }
    return {E_old, max_iter, false};
}

// ---- Basis set path finding ----

std::string find_basis_file(const std::string& basis_filename) {
    std::vector<std::string> search_paths = {
        "share/basis_sets/" + basis_filename,
        "../share/basis_sets/" + basis_filename,
        "../../share/basis_sets/" + basis_filename,
        "../../../share/basis_sets/" + basis_filename,
    };

    const char* src_dir = std::getenv("LIBACCINT_SOURCE_DIR");
    if (src_dir) {
        search_paths.insert(search_paths.begin(),
            std::string(src_dir) + "/share/basis_sets/" + basis_filename);
    }

    for (const auto& path : search_paths) {
        if (std::filesystem::exists(path)) {
            return path;
        }
    }
    return {};
}

}  // anonymous namespace

// =============================================================================
// Parameterized Test Fixture
// =============================================================================

struct MolBasisParam {
    std::string mol_name;
    std::string basis_name;
    std::string basis_filename;  // e.g. "sto-3g.json"
    std::vector<Atom> atoms;
    int n_electrons;
    Real ref_energy;    // PySCF RHF total energy (Hartree)
    Size expected_nbf;  // expected number of basis functions
};

class ComprehensiveValidation
    : public ::testing::TestWithParam<MolBasisParam> {};

// Provide all 6 molecule/basis combinations
static std::vector<MolBasisParam> make_params() {
    auto h2o = make_h2o_atoms();
    auto ch4 = make_ch4_atoms();
    return {
        // PySCF RHF energies computed with exact Bohr geometries, cart=True, conv_tol=1e-12
        {"H2O", "STO-3G",       "sto-3g.json",       h2o, 10,
         -74.963110239453101,  7},
        {"H2O", "6-31G",        "6-31g.json",        h2o, 10,
         -75.983978013076069, 13},
        {"H2O", "aug-cc-pVDZ",  "aug-cc-pvdz.json",  h2o, 10,
         -76.041908887006002, 43},  // 43 with Cartesian d (6 per d-shell)
        {"CH4", "STO-3G",       "sto-3g.json",       ch4, 10,
         -39.726853932949453,  9},
        {"CH4", "6-31G",        "6-31g.json",        ch4, 10,
         -40.180535577368161, 17},
        {"CH4", "aug-cc-pVDZ",  "aug-cc-pvdz.json",  ch4, 10,
         -40.199621700722801, 61},  // 61 with Cartesian d (6 per d-shell)
    };
}

INSTANTIATE_TEST_SUITE_P(
    AllCombinations,
    ComprehensiveValidation,
    ::testing::ValuesIn(make_params()),
    [](const ::testing::TestParamInfo<MolBasisParam>& info) {
        // Sanitize name for gtest (no slashes, dashes, etc.)
        std::string name = info.param.mol_name + "_" + info.param.basis_name;
        for (auto& c : name) {
            if (c == '-' || c == ' ') c = '_';
        }
        return name;
    });

// =============================================================================
// 1. Basis Loading
// =============================================================================

TEST_P(ComprehensiveValidation, BasisLoads) {
    const auto& p = GetParam();
    std::string path = find_basis_file(p.basis_filename);
    if (path.empty()) {
        GTEST_SKIP() << "Basis file not found: " << p.basis_filename;
    }

    auto basis = BseJsonParser::parse_file(path, p.atoms);
    EXPECT_GT(basis.n_basis_functions(), 0u)
        << p.mol_name << "/" << p.basis_name << " should have basis functions";
    EXPECT_EQ(basis.n_basis_functions(), p.expected_nbf)
        << p.mol_name << "/" << p.basis_name
        << " expected " << p.expected_nbf << " basis functions";
}

// =============================================================================
// 2. One-Electron Integral Properties
// =============================================================================

TEST_P(ComprehensiveValidation, OverlapDiagonalIsOne) {
    const auto& p = GetParam();
    std::string path = find_basis_file(p.basis_filename);
    if (path.empty()) GTEST_SKIP() << "Basis file not found: " << p.basis_filename;

    auto basis = BseJsonParser::parse_file(path, p.atoms);
    Size nbf = basis.n_basis_functions();
    Engine engine(basis);

    std::vector<Real> S;
    engine.compute_1e(Operator::overlap(), S);
    ASSERT_EQ(S.size(), nbf * nbf);

    for (Size i = 0; i < nbf; ++i) {
        EXPECT_NEAR(S[i * nbf + i], 1.0, 1e-10)
            << p.mol_name << "/" << p.basis_name
            << " overlap diagonal S(" << i << "," << i << ")";
    }
}

TEST_P(ComprehensiveValidation, OverlapSymmetric) {
    const auto& p = GetParam();
    std::string path = find_basis_file(p.basis_filename);
    if (path.empty()) GTEST_SKIP() << "Basis file not found: " << p.basis_filename;

    auto basis = BseJsonParser::parse_file(path, p.atoms);
    Size nbf = basis.n_basis_functions();
    Engine engine(basis);

    std::vector<Real> S;
    engine.compute_1e(Operator::overlap(), S);

    for (Size i = 0; i < nbf; ++i) {
        for (Size j = i + 1; j < nbf; ++j) {
            EXPECT_NEAR(S[i * nbf + j], S[j * nbf + i], 1e-12)
                << "S(" << i << "," << j << ") != S(" << j << "," << i << ")";
        }
    }
}

TEST_P(ComprehensiveValidation, KineticSymmetricAndPositiveSemiDefinite) {
    const auto& p = GetParam();
    std::string path = find_basis_file(p.basis_filename);
    if (path.empty()) GTEST_SKIP() << "Basis file not found: " << p.basis_filename;

    auto basis = BseJsonParser::parse_file(path, p.atoms);
    int nbf = static_cast<int>(basis.n_basis_functions());
    Engine engine(basis);

    std::vector<Real> T_flat;
    engine.compute_1e(Operator::kinetic(), T_flat);

    // Symmetry check
    for (int i = 0; i < nbf; ++i) {
        for (int j = i + 1; j < nbf; ++j) {
            EXPECT_NEAR(T_flat[i * nbf + j], T_flat[j * nbf + i], 1e-12)
                << "T(" << i << "," << j << ") != T(" << j << "," << i << ")";
        }
    }

    // Positive semi-definiteness: all eigenvalues >= 0
    auto T = to_eigen(T_flat, nbf);
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es(T);
    for (int i = 0; i < nbf; ++i) {
        EXPECT_GE(es.eigenvalues()(i), -1e-10)
            << "Kinetic eigenvalue " << i << " should be >= 0";
    }
}

TEST_P(ComprehensiveValidation, NuclearSymmetricAndNegativeSemiDefinite) {
    const auto& p = GetParam();
    std::string path = find_basis_file(p.basis_filename);
    if (path.empty()) GTEST_SKIP() << "Basis file not found: " << p.basis_filename;

    auto basis = BseJsonParser::parse_file(path, p.atoms);
    int nbf = static_cast<int>(basis.n_basis_functions());
    Engine engine(basis);

    auto charges = make_charges(p.atoms);
    std::vector<Real> V_flat;
    engine.compute_1e(Operator::nuclear(charges), V_flat);

    // Symmetry check
    for (int i = 0; i < nbf; ++i) {
        for (int j = i + 1; j < nbf; ++j) {
            EXPECT_NEAR(V_flat[i * nbf + j], V_flat[j * nbf + i], 1e-12)
                << "V(" << i << "," << j << ") != V(" << j << "," << i << ")";
        }
    }

    // Negative semi-definiteness: all eigenvalues <= 0
    auto V = to_eigen(V_flat, nbf);
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es(V);
    for (int i = 0; i < nbf; ++i) {
        EXPECT_LE(es.eigenvalues()(i), 1e-10)
            << "Nuclear eigenvalue " << i << " should be <= 0";
    }
}

// =============================================================================
// 3. Two-Electron Fock Build Properties
// =============================================================================

TEST_P(ComprehensiveValidation, FockMatricesSymmetric) {
    const auto& p = GetParam();
    std::string path = find_basis_file(p.basis_filename);
    if (path.empty()) GTEST_SKIP() << "Basis file not found: " << p.basis_filename;

    auto basis = BseJsonParser::parse_file(path, p.atoms);
    Size nbf = basis.n_basis_functions();
    Engine engine(basis);

    // Use an identity-like density matrix for testing
    std::vector<Real> D(nbf * nbf, 0.0);
    for (Size i = 0; i < nbf; ++i)
        D[i * nbf + i] = 1.0 / static_cast<Real>(nbf);

    FockBuilder fock(nbf);
    fock.set_density(D.data(), nbf);
    engine.compute_and_consume(Operator::coulomb(), fock);

    auto J = fock.get_coulomb_matrix();
    auto K = fock.get_exchange_matrix();

    // J should be symmetric
    for (Size i = 0; i < nbf; ++i) {
        for (Size j = i + 1; j < nbf; ++j) {
            EXPECT_NEAR(J[i * nbf + j], J[j * nbf + i], 1e-12)
                << "J(" << i << "," << j << ") != J(" << j << "," << i << ")";
        }
    }

    // K should be symmetric
    for (Size i = 0; i < nbf; ++i) {
        for (Size j = i + 1; j < nbf; ++j) {
            EXPECT_NEAR(K[i * nbf + j], K[j * nbf + i], 1e-12)
                << "K(" << i << "," << j << ") != K(" << j << "," << i << ")";
        }
    }
}

// =============================================================================
// 4. RHF SCF Energy vs PySCF Reference
// =============================================================================

TEST_P(ComprehensiveValidation, SCFEnergyMatchesPySCF) {
    const auto& p = GetParam();
    std::string path = find_basis_file(p.basis_filename);
    if (path.empty()) GTEST_SKIP() << "Basis file not found: " << p.basis_filename;

    auto basis = BseJsonParser::parse_file(path, p.atoms);
    ASSERT_EQ(basis.n_basis_functions(), p.expected_nbf)
        << "Unexpected nbf for " << p.mol_name << "/" << p.basis_name;

    auto result = run_rhf(p.atoms, basis, p.n_electrons);
    EXPECT_TRUE(result.converged)
        << p.mol_name << "/" << p.basis_name
        << " SCF did not converge in " << result.iterations << " iterations";

    // Tolerance: 5e-8 Hartree (covers observed STO-3G parser/built-in drift)
    EXPECT_NEAR(result.total_energy, p.ref_energy, 5e-8)
        << p.mol_name << "/" << p.basis_name
        << " SCF energy does not match PySCF reference\n"
        << "  computed: " << result.total_energy << "\n"
        << "  expected: " << p.ref_energy;
}

// =============================================================================
// 5. Screening Correctness
// =============================================================================

TEST_P(ComprehensiveValidation, ScreenedFockNonZero) {
    const auto& p = GetParam();
    std::string path = find_basis_file(p.basis_filename);
    if (path.empty()) GTEST_SKIP() << "Basis file not found: " << p.basis_filename;

    auto basis = BseJsonParser::parse_file(path, p.atoms);
    Size nbf = basis.n_basis_functions();
    Engine engine(basis);

    // Precompute Schwarz bounds
    engine.precompute_schwarz_bounds();

    // Use identity-like density
    std::vector<Real> D(nbf * nbf, 0.0);
    for (Size i = 0; i < nbf; ++i)
        D[i * nbf + i] = 1.0 / static_cast<Real>(nbf);

    engine.set_density_matrix(D.data(), nbf);

    FockBuilder fock(nbf);
    fock.set_density(D.data(), nbf);
    engine.compute_and_consume(Operator::coulomb(), fock,
                               screening::ScreeningOptions::normal());

    auto J = fock.get_coulomb_matrix();

    // Verify we got non-zero results
    bool has_nonzero = false;
    for (Size i = 0; i < nbf * nbf; ++i) {
        if (std::abs(J[i]) > 1e-15) {
            has_nonzero = true;
            break;
        }
    }
    EXPECT_TRUE(has_nonzero)
        << p.mol_name << "/" << p.basis_name
        << " screened Fock build produced all zeros";
}

TEST_P(ComprehensiveValidation, ScreenedFockMatchesUnscreened) {
    const auto& p = GetParam();
    std::string path = find_basis_file(p.basis_filename);
    if (path.empty()) GTEST_SKIP() << "Basis file not found: " << p.basis_filename;

    auto basis = BseJsonParser::parse_file(path, p.atoms);
    Size nbf = basis.n_basis_functions();
    Engine engine(basis);

    // Build a simple density
    std::vector<Real> D(nbf * nbf, 0.0);
    for (Size i = 0; i < nbf; ++i)
        D[i * nbf + i] = 1.0 / static_cast<Real>(nbf);

    // Unscreened Fock
    FockBuilder fock_unscreened(nbf);
    fock_unscreened.set_density(D.data(), nbf);
    engine.compute_and_consume(Operator::coulomb(), fock_unscreened);
    auto J_unscreened = fock_unscreened.get_coulomb_matrix();
    auto K_unscreened = fock_unscreened.get_exchange_matrix();

    // Screened Fock (tight screening, should still be accurate)
    engine.precompute_schwarz_bounds();
    engine.set_density_matrix(D.data(), nbf);

    FockBuilder fock_screened(nbf);
    fock_screened.set_density(D.data(), nbf);
    engine.compute_and_consume(Operator::coulomb(), fock_screened,
                               screening::ScreeningOptions::tight());
    auto J_screened = fock_screened.get_coulomb_matrix();
    auto K_screened = fock_screened.get_exchange_matrix();

    // Tight screening should give results very close to unscreened
    for (Size i = 0; i < nbf * nbf; ++i) {
        EXPECT_NEAR(J_screened[i], J_unscreened[i], 1e-10)
            << "J mismatch at index " << i;
        EXPECT_NEAR(K_screened[i], K_unscreened[i], 1e-10)
            << "K mismatch at index " << i;
    }
}

// =============================================================================
// 6. Parallel Computation Consistency
// =============================================================================

TEST_P(ComprehensiveValidation, Parallel1eMatchesSerial) {
    const auto& p = GetParam();
    std::string path = find_basis_file(p.basis_filename);
    if (path.empty()) GTEST_SKIP() << "Basis file not found: " << p.basis_filename;

    auto basis = BseJsonParser::parse_file(path, p.atoms);
    Size nbf = basis.n_basis_functions();
    Engine engine(basis);

    // Serial overlap
    std::vector<Real> S_serial;
    engine.compute_1e(Operator::overlap(), S_serial);

    // Parallel overlap (4 threads)
    std::vector<Real> S_parallel;
    engine.compute_1e_parallel<0>(
        OneElectronOperator(Operator::overlap()), S_parallel, 4);

    ASSERT_EQ(S_serial.size(), S_parallel.size());
    for (Size i = 0; i < S_serial.size(); ++i) {
        EXPECT_NEAR(S_serial[i], S_parallel[i], 1e-14)
            << "Parallel 1e overlap mismatch at index " << i;
    }
}

TEST_P(ComprehensiveValidation, ParallelFockMatchesSerial) {
    const auto& p = GetParam();
    std::string path = find_basis_file(p.basis_filename);
    if (path.empty()) GTEST_SKIP() << "Basis file not found: " << p.basis_filename;

    auto basis = BseJsonParser::parse_file(path, p.atoms);
    Size nbf = basis.n_basis_functions();
    Engine engine(basis);

    // Build a simple density
    std::vector<Real> D(nbf * nbf, 0.0);
    for (Size i = 0; i < nbf; ++i)
        D[i * nbf + i] = 1.0 / static_cast<Real>(nbf);

    // Serial Fock build
    FockBuilder fock_serial(nbf);
    fock_serial.set_density(D.data(), nbf);
    engine.compute_and_consume(Operator::coulomb(), fock_serial);
    auto J_serial = fock_serial.get_coulomb_matrix();
    auto K_serial = fock_serial.get_exchange_matrix();

    // Parallel Fock build (4 threads)
    FockBuilder fock_parallel(nbf);
    fock_parallel.set_density(D.data(), nbf);
    engine.compute_and_consume_parallel(Operator::coulomb(), fock_parallel, 4);
    auto J_parallel = fock_parallel.get_coulomb_matrix();
    auto K_parallel = fock_parallel.get_exchange_matrix();

    for (Size i = 0; i < nbf * nbf; ++i) {
        EXPECT_NEAR(J_serial[i], J_parallel[i], 1e-12)
            << "Parallel J mismatch at index " << i;
        EXPECT_NEAR(K_serial[i], K_parallel[i], 1e-12)
            << "Parallel K mismatch at index " << i;
    }
}

// =============================================================================
// Non-Parameterized Supplementary Tests
// =============================================================================

/// Verify nuclear repulsion energies are physically reasonable
TEST(ComprehensiveValidationMisc, NuclearRepulsionReasonable) {
    auto h2o = make_h2o_atoms();
    auto ch4 = make_ch4_atoms();

    Real E_nuc_h2o = compute_nuclear_repulsion(h2o);
    Real E_nuc_ch4 = compute_nuclear_repulsion(ch4);

    // H2O nuclear repulsion should be positive and ~9.1 Hartree
    EXPECT_GT(E_nuc_h2o, 8.0);
    EXPECT_LT(E_nuc_h2o, 10.0);

    // CH4 nuclear repulsion should be positive and ~13.5 Hartree
    EXPECT_GT(E_nuc_ch4, 12.0);
    EXPECT_LT(E_nuc_ch4, 15.0);
}

/// Verify STO-3G via built-in basis matches BSE parser
TEST(ComprehensiveValidationMisc, BuiltinSTO3GMatchesBSE) {
    auto h2o = make_h2o_atoms();

    auto builtin = create_sto3g(h2o);
    Size nbf_builtin = builtin.n_basis_functions();

    std::string path = find_basis_file("sto-3g.json");
    if (path.empty()) {
        GTEST_SKIP() << "sto-3g.json not found";
    }

    auto bse = BseJsonParser::parse_file(path, h2o);
    Size nbf_bse = bse.n_basis_functions();

    EXPECT_EQ(nbf_builtin, nbf_bse)
        << "Built-in and BSE STO-3G should have the same number of basis functions";

    // Compare overlap matrices
    Engine engine_builtin(builtin);
    Engine engine_bse(bse);

    std::vector<Real> S_builtin, S_bse;
    engine_builtin.compute_1e(Operator::overlap(), S_builtin);
    engine_bse.compute_1e(Operator::overlap(), S_bse);

    ASSERT_EQ(S_builtin.size(), S_bse.size());
    for (Size i = 0; i < S_builtin.size(); ++i) {
        EXPECT_NEAR(S_builtin[i], S_bse[i], 5e-8)
            << "Built-in vs BSE overlap mismatch at index " << i;
    }
}

/// Stress test: SCF convergence for all molecules with tight tolerance
TEST(ComprehensiveValidationMisc, SCFConverges_H2O_STO3G_Builtin) {
    auto h2o = make_h2o_atoms();
    auto basis = create_sto3g(h2o);
    auto result = run_rhf(h2o, basis, 10);
    EXPECT_TRUE(result.converged);
    EXPECT_LT(result.iterations, 100)
        << "SCF should converge within 100 iterations";
    // Built-in STO-3G reference (may differ slightly from BSE normalization)
    EXPECT_NEAR(result.total_energy, -74.963110248964085, 1e-6);
}

/// Verify that screening with ScreeningOptions::none() produces same result as
/// unscreened path (API consistency check)
TEST(ComprehensiveValidationMisc, ScreeningNoneMatchesUnscreened) {
    auto h2o = make_h2o_atoms();
    auto basis = create_sto3g(h2o);
    Size nbf = basis.n_basis_functions();
    Engine engine(basis);

    std::vector<Real> D(nbf * nbf, 0.0);
    for (Size i = 0; i < nbf; ++i)
        D[i * nbf + i] = 1.0 / static_cast<Real>(nbf);

    // Unscreened
    FockBuilder fock1(nbf);
    fock1.set_density(D.data(), nbf);
    engine.compute_and_consume(Operator::coulomb(), fock1);
    auto J1 = fock1.get_coulomb_matrix();

    // Screening disabled
    FockBuilder fock2(nbf);
    fock2.set_density(D.data(), nbf);
    engine.compute_and_consume(Operator::coulomb(), fock2,
                               screening::ScreeningOptions::none());
    auto J2 = fock2.get_coulomb_matrix();

    for (Size i = 0; i < nbf * nbf; ++i) {
        EXPECT_NEAR(J1[i], J2[i], 1e-15)
            << "ScreeningOptions::none() should produce identical results";
    }
}
