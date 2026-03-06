// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2026 Bryce M. Westheimer. All rights reserved.

/// @file test_water_cluster_validation.cpp
/// @brief Water cluster validation against PySCF reference data
///
/// Tests (H2O)_N clusters (N=1,2,4,8) with aug-cc-pVTZ:
/// - One-electron integral matrices (S, T, V) vs PySCF
/// - Fock build (J, K from stored density) vs PySCF
/// - RHF SCF energy vs PySCF
/// - Sampled ERI quartets vs PySCF
/// - Overlap matrix properties (diagonal=1, symmetric, positive definite)

#include <libaccint/engine/engine.hpp>
#include <libaccint/consumers/fock_builder.hpp>
#include <libaccint/data/builtin_basis.hpp>
#include <libaccint/data/bse_json_parser.hpp>
#include <libaccint/data/basis_parser.hpp>
#include <libaccint/operators/operator.hpp>
#include <libaccint/operators/one_electron_operator.hpp>
#include <libaccint/core/types.hpp>

#include <gtest/gtest.h>
#include <Eigen/Dense>
#include <nlohmann/json.hpp>

#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

using namespace libaccint;
using namespace libaccint::data;
using namespace libaccint::consumers;
using json = nlohmann::json;

// =============================================================================
// Helper Utilities
// =============================================================================

namespace {

// ---- Water cluster geometry (must match Python generator exactly) ----

std::vector<Atom> make_water_cluster(int n) {
    const double r_oh = 1.8088;
    const double theta_hoh = 104.52 * M_PI / 180.0;
    const double half_theta = theta_hoh / 2.0;

    std::vector<Atom> atoms;

    if (n == 1) {
        double hy = r_oh * std::sin(half_theta);
        double hz = -r_oh * std::cos(half_theta);
        atoms.push_back({8, {0.0, 0.0, 0.0}});
        atoms.push_back({1, {0.0, hy, hz}});
        atoms.push_back({1, {0.0, -hy, hz}});
        return atoms;
    }

    // N >= 2: O atoms on regular N-gon, O-O distance = 5.4 Bohr
    const double oo_dist = 5.4;
    const double R = oo_dist / (2.0 * std::sin(M_PI / n));

    for (int i = 0; i < n; ++i) {
        double angle_o = 2.0 * M_PI * i / n;
        double ox = R * std::cos(angle_o);
        double oy = R * std::sin(angle_o);
        double oz = 0.0;
        atoms.push_back({8, {ox, oy, oz}});

        // Direction toward next O
        int next_i = (i + 1) % n;
        double angle_next = 2.0 * M_PI * next_i / n;
        double nx = R * std::cos(angle_next) - ox;
        double ny = R * std::sin(angle_next) - oy;
        double nlen = std::sqrt(nx * nx + ny * ny);
        double dx = nx / nlen;
        double dy = ny / nlen;

        // Perpendicular in XY plane
        double px = -dy;
        double py = dx;

        // H1: donor H
        double h1x = ox + r_oh * std::cos(half_theta) * dx;
        double h1y = oy + r_oh * std::cos(half_theta) * dy;
        double h1z = oz - r_oh * std::sin(half_theta);
        atoms.push_back({1, {h1x, h1y, h1z}});

        // H2: other H
        double h2x = ox + r_oh * std::cos(half_theta) * px;
        double h2y = oy + r_oh * std::cos(half_theta) * py;
        double h2z = oz + r_oh * std::sin(half_theta);
        atoms.push_back({1, {h2x, h2y, h2z}});
    }

    return atoms;
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

// ---- SCF ----

struct SCFResult {
    Real total_energy;
    int iterations;
    bool converged;
};

SCFResult run_rhf(const std::vector<Atom>& atoms,
                  const BasisSet& basis,
                  int n_electrons,
                  Real energy_tol = 1e-10,
                  int max_iter = 200) {
    const int nbf = static_cast<int>(basis.n_basis_functions());
    const int n_occ = n_electrons / 2;

    Engine engine(basis);
    auto charges = make_charges(atoms);

    std::vector<Real> S_flat, T_flat, V_flat;
    engine.compute_1e(Operator::overlap(), S_flat);
    engine.compute_1e(Operator::kinetic(), T_flat);
    engine.compute_1e(Operator::nuclear(charges), V_flat);

    auto S = to_eigen(S_flat, nbf);
    auto T = to_eigen(T_flat, nbf);
    auto V = to_eigen(V_flat, nbf);
    auto H_core = T + V;
    Real E_nuc = compute_nuclear_repulsion(atoms);

    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es_s(S);
    Eigen::VectorXd s_vals = es_s.eigenvalues();
    Eigen::MatrixXd s_vecs = es_s.eigenvectors();
    Eigen::VectorXd s_inv_half(s_vals.size());
    for (int i = 0; i < s_vals.size(); ++i)
        s_inv_half(i) = (s_vals(i) > 1e-10) ? 1.0 / std::sqrt(s_vals(i)) : 0.0;
    Eigen::MatrixXd S_inv_half = s_vecs * s_inv_half.asDiagonal() * s_vecs.transpose();

    auto [eps0, C0] = solve_gen_eigenvalue(H_core, S);
    auto D = build_density(C0, n_occ);
    Real E_old = 0.0;

    constexpr int diis_max = 8;
    std::vector<Eigen::MatrixXd> diis_focks;
    std::vector<Eigen::MatrixXd> diis_errors;

    for (int iter = 0; iter < max_iter; ++iter) {
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

        Eigen::MatrixXd err = F * D * S - S * D * F;
        Eigen::MatrixXd err_orth = S_inv_half.transpose() * err * S_inv_half;

        diis_focks.push_back(F);
        diis_errors.push_back(err_orth);
        if (static_cast<int>(diis_focks.size()) > diis_max) {
            diis_focks.erase(diis_focks.begin());
            diis_errors.erase(diis_errors.begin());
        }

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
            Eigen::VectorXd c = B.colPivHouseholderQr().solve(rhs);
            F = Eigen::MatrixXd::Zero(nbf, nbf);
            for (int i = 0; i < ndiis; ++i)
                F += c(i) * diis_focks[i];
        }

        Real E_elec = 0.5 * ((H_core + (H_core + J - 0.5 * K)).array() * D.array()).sum();
        Real E_total = E_elec + E_nuc;

        if (std::abs(E_total - E_old) < energy_tol && iter > 0)
            return {E_total, iter + 1, true};

        auto [eps, C] = solve_gen_eigenvalue(F, S);
        D = build_density(C, n_occ);
        E_old = E_total;
    }
    return {E_old, max_iter, false};
}

// ---- Reference data loading ----

std::string find_reference_file(int n_waters) {
    std::string filename = "water_" + std::to_string(n_waters) + "_aug_cc_pVTZ.json";
    std::vector<std::string> search_paths = {
        "tests/data/reference/" + filename,
        "../tests/data/reference/" + filename,
        "../../tests/data/reference/" + filename,
        "../../../tests/data/reference/" + filename,
    };
    const char* src_dir = std::getenv("LIBACCINT_SOURCE_DIR");
    if (src_dir) {
        search_paths.insert(search_paths.begin(),
            std::string(src_dir) + "/tests/data/reference/" + filename);
    }
    for (const auto& path : search_paths) {
        if (std::filesystem::exists(path))
            return path;
    }
    return {};
}

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
        if (std::filesystem::exists(path))
            return path;
    }
    return {};
}

json load_reference(int n_waters) {
    std::string path = find_reference_file(n_waters);
    if (path.empty()) return {};
    std::ifstream f(path);
    if (!f.is_open()) return {};
    json j;
    f >> j;
    return j;
}

}  // anonymous namespace

// =============================================================================
// Parameterized Test Fixture
// =============================================================================

struct WaterClusterParam {
    int n_waters;
    int n_electrons;
    Size expected_nbf;
};

class WaterClusterValidation
    : public ::testing::TestWithParam<WaterClusterParam> {};

static std::vector<WaterClusterParam> make_params() {
    return {
        {1, 10, 105},
        {2, 20, 210},
        {4, 40, 420},
        {8, 80, 840},
    };
}

INSTANTIATE_TEST_SUITE_P(
    WaterClusters,
    WaterClusterValidation,
    ::testing::ValuesIn(make_params()),
    [](const ::testing::TestParamInfo<WaterClusterParam>& info) {
        return "Water" + std::to_string(info.param.n_waters);
    });

// =============================================================================
// 1. Basis Loading
// =============================================================================

TEST_P(WaterClusterValidation, BasisLoads) {
    const auto& p = GetParam();
    std::string basis_path = find_basis_file("aug-cc-pvtz.json");
    if (basis_path.empty())
        GTEST_SKIP() << "aug-cc-pvtz.json not found";

    auto atoms = make_water_cluster(p.n_waters);
    auto basis = BseJsonParser::parse_file(basis_path, atoms);
    EXPECT_EQ(basis.n_basis_functions(), p.expected_nbf)
        << "(H2O)_" << p.n_waters << " expected " << p.expected_nbf << " BF";
}

// =============================================================================
// 2. Overlap vs PySCF
// =============================================================================

TEST_P(WaterClusterValidation, OverlapMatchesPySCF) {
    const auto& p = GetParam();
    auto ref = load_reference(p.n_waters);
    if (ref.empty()) GTEST_SKIP() << "Reference file not found for N=" << p.n_waters;
    std::string basis_path = find_basis_file("aug-cc-pvtz.json");
    if (basis_path.empty()) GTEST_SKIP() << "aug-cc-pvtz.json not found";

    auto atoms = make_water_cluster(p.n_waters);
    auto basis = BseJsonParser::parse_file(basis_path, atoms);
    int nbf = static_cast<int>(basis.n_basis_functions());

    Engine engine(basis);
    std::vector<Real> S;
    engine.compute_1e(Operator::overlap(), S);

    auto ref_S = ref["integrals"]["overlap"]["matrix"].get<std::vector<double>>();
    ASSERT_EQ(S.size(), ref_S.size());
    for (Size i = 0; i < S.size(); ++i) {
        EXPECT_NEAR(S[i], ref_S[i], 1e-10)
            << "(H2O)_" << p.n_waters << " overlap mismatch at index " << i;
    }
}

// =============================================================================
// 3. Kinetic vs PySCF
// =============================================================================

TEST_P(WaterClusterValidation, KineticMatchesPySCF) {
    const auto& p = GetParam();
    auto ref = load_reference(p.n_waters);
    if (ref.empty()) GTEST_SKIP() << "Reference file not found for N=" << p.n_waters;
    std::string basis_path = find_basis_file("aug-cc-pvtz.json");
    if (basis_path.empty()) GTEST_SKIP() << "aug-cc-pvtz.json not found";

    auto atoms = make_water_cluster(p.n_waters);
    auto basis = BseJsonParser::parse_file(basis_path, atoms);

    Engine engine(basis);
    std::vector<Real> T;
    engine.compute_1e(Operator::kinetic(), T);

    auto ref_T = ref["integrals"]["kinetic"]["matrix"].get<std::vector<double>>();
    ASSERT_EQ(T.size(), ref_T.size());
    for (Size i = 0; i < T.size(); ++i) {
        EXPECT_NEAR(T[i], ref_T[i], 1e-10)
            << "(H2O)_" << p.n_waters << " kinetic mismatch at index " << i;
    }
}

// =============================================================================
// 4. Nuclear Attraction vs PySCF
// =============================================================================

TEST_P(WaterClusterValidation, NuclearMatchesPySCF) {
    const auto& p = GetParam();
    auto ref = load_reference(p.n_waters);
    if (ref.empty()) GTEST_SKIP() << "Reference file not found for N=" << p.n_waters;
    std::string basis_path = find_basis_file("aug-cc-pvtz.json");
    if (basis_path.empty()) GTEST_SKIP() << "aug-cc-pvtz.json not found";

    auto atoms = make_water_cluster(p.n_waters);
    auto basis = BseJsonParser::parse_file(basis_path, atoms);

    Engine engine(basis);
    auto charges = make_charges(atoms);
    std::vector<Real> V;
    engine.compute_1e(Operator::nuclear(charges), V);

    auto ref_V = ref["integrals"]["nuclear_attraction"]["matrix"].get<std::vector<double>>();
    ASSERT_EQ(V.size(), ref_V.size());
    for (Size i = 0; i < V.size(); ++i) {
        EXPECT_NEAR(V[i], ref_V[i], 1e-8)
            << "(H2O)_" << p.n_waters << " nuclear mismatch at index " << i;
    }
}

// =============================================================================
// 5. Fock Build (J, K from stored density) vs PySCF
// =============================================================================

TEST_P(WaterClusterValidation, FockBuildMatchesPySCF) {
    const auto& p = GetParam();
    if (p.n_waters > 4) GTEST_SKIP() << "Fock test skipped for N>4 (too expensive)";

    const char* skip_large = std::getenv("LIBACCINT_SKIP_LARGE_TESTS");
    if (skip_large && p.n_waters >= 4)
        GTEST_SKIP() << "Skipped via LIBACCINT_SKIP_LARGE_TESTS";

    auto ref = load_reference(p.n_waters);
    if (ref.empty()) GTEST_SKIP() << "Reference file not found for N=" << p.n_waters;
    std::string basis_path = find_basis_file("aug-cc-pvtz.json");
    if (basis_path.empty()) GTEST_SKIP() << "aug-cc-pvtz.json not found";

    auto atoms = make_water_cluster(p.n_waters);
    auto basis = BseJsonParser::parse_file(basis_path, atoms);
    Size nbf = basis.n_basis_functions();

    // Load stored density from reference
    auto ref_D = ref["density_matrix"]["matrix"].get<std::vector<double>>();
    ASSERT_EQ(ref_D.size(), nbf * nbf);
    std::vector<Real> D(ref_D.begin(), ref_D.end());

    Engine engine(basis);
    FockBuilder fock(nbf);
    fock.set_density(D.data(), nbf);
    engine.compute_and_consume(Operator::coulomb(), fock);

    auto J = fock.get_coulomb_matrix();
    auto K = fock.get_exchange_matrix();

    auto ref_J = ref["fock_from_density"]["coulomb_matrix"].get<std::vector<double>>();
    auto ref_K = ref["fock_from_density"]["exchange_matrix"].get<std::vector<double>>();
    ASSERT_EQ(ref_J.size(), nbf * nbf);
    ASSERT_EQ(ref_K.size(), nbf * nbf);

    for (Size i = 0; i < nbf * nbf; ++i) {
        EXPECT_NEAR(J[i], ref_J[i], 1e-7)
            << "(H2O)_" << p.n_waters << " J mismatch at index " << i;
    }
    for (Size i = 0; i < nbf * nbf; ++i) {
        EXPECT_NEAR(K[i], ref_K[i], 1e-7)
            << "(H2O)_" << p.n_waters << " K mismatch at index " << i;
    }
}

// =============================================================================
// 6. SCF Energy vs PySCF
// =============================================================================

TEST_P(WaterClusterValidation, SCFEnergyMatchesPySCF) {
    const auto& p = GetParam();
    if (p.n_waters > 2) GTEST_SKIP() << "SCF test skipped for N>2 (too expensive for CI timeout)";

    const char* skip_large = std::getenv("LIBACCINT_SKIP_LARGE_TESTS");
    if (skip_large && p.n_waters >= 4)
        GTEST_SKIP() << "Skipped via LIBACCINT_SKIP_LARGE_TESTS";

    auto ref = load_reference(p.n_waters);
    if (ref.empty()) GTEST_SKIP() << "Reference file not found for N=" << p.n_waters;
    std::string basis_path = find_basis_file("aug-cc-pvtz.json");
    if (basis_path.empty()) GTEST_SKIP() << "aug-cc-pvtz.json not found";

    auto atoms = make_water_cluster(p.n_waters);
    auto basis = BseJsonParser::parse_file(basis_path, atoms);

    double ref_energy = ref["rhf"]["total_energy"].get<double>();

    auto result = run_rhf(atoms, basis, p.n_electrons);
    EXPECT_TRUE(result.converged)
        << "(H2O)_" << p.n_waters << " SCF did not converge";
    EXPECT_NEAR(result.total_energy, ref_energy, 5e-8)
        << "(H2O)_" << p.n_waters << " SCF energy mismatch\n"
        << "  computed: " << result.total_energy << "\n"
        << "  expected: " << ref_energy;
}

// =============================================================================
// 7. Sampled ERI vs PySCF
// =============================================================================

TEST_P(WaterClusterValidation, SampledERIMatchesPySCF) {
    const auto& p = GetParam();
    if (p.n_waters > 2) GTEST_SKIP() << "ERI sample test skipped for N>2 (too expensive)";
    auto ref = load_reference(p.n_waters);
    if (ref.empty()) GTEST_SKIP() << "Reference file not found for N=" << p.n_waters;
    if (!ref.contains("sampled_eri") || ref["sampled_eri"].empty())
        GTEST_SKIP() << "No sampled ERI data in reference for N=" << p.n_waters;
    std::string basis_path = find_basis_file("aug-cc-pvtz.json");
    if (basis_path.empty()) GTEST_SKIP() << "aug-cc-pvtz.json not found";

    auto atoms = make_water_cluster(p.n_waters);
    auto basis = BseJsonParser::parse_file(basis_path, atoms);
    Size nbf = basis.n_basis_functions();

    Engine engine(basis);

    // Reference ERI samples are stored as {indices: [i,j,k,l], value: (ij|kl)}
    // in LibAccInt convention and ordering.
    // We verify by computing (ij|kl) via delta-density trick:
    //   Set D(k,l) = D(l,k) = 1, compute J = sum_{kl} (ij|kl)*D(kl)
    //   J(i,j) = (ij|kl) + (ij|lk) = 2*(ij|kl) if k!=l, else (ij|kk)
    const auto& eri_samples = ref["sampled_eri"];
    int n_checked = 0;

    for (const auto& sample : eri_samples) {
        auto indices = sample["indices"].get<std::vector<int>>();
        double ref_val = sample["value"].get<double>();
        int i = indices[0], j = indices[1], k = indices[2], l = indices[3];

        std::vector<Real> D_delta(nbf * nbf, 0.0);
        D_delta[static_cast<Size>(k) * nbf + l] = 1.0;
        if (k != l)
            D_delta[static_cast<Size>(l) * nbf + k] = 1.0;

        FockBuilder fock(nbf);
        fock.set_density(D_delta.data(), nbf);
        engine.compute_and_consume(Operator::coulomb(), fock);
        auto J = fock.get_coulomb_matrix();

        double computed_J = J[static_cast<Size>(i) * nbf + j];
        double computed_eri = (k != l) ? computed_J / 2.0 : computed_J;

        EXPECT_NEAR(computed_eri, ref_val, 1e-10)
            << "(H2O)_" << p.n_waters
            << " ERI(" << i << "," << j << "|" << k << "," << l << ")";
        ++n_checked;
    }
    EXPECT_GT(n_checked, 0) << "No ERI samples were checked";
}

// =============================================================================
// 8. Overlap Properties
// =============================================================================

TEST_P(WaterClusterValidation, OverlapProperties) {
    const auto& p = GetParam();
    std::string basis_path = find_basis_file("aug-cc-pvtz.json");
    if (basis_path.empty()) GTEST_SKIP() << "aug-cc-pvtz.json not found";

    auto atoms = make_water_cluster(p.n_waters);
    auto basis = BseJsonParser::parse_file(basis_path, atoms);
    int nbf = static_cast<int>(basis.n_basis_functions());

    Engine engine(basis);
    std::vector<Real> S_flat;
    engine.compute_1e(Operator::overlap(), S_flat);

    // Diagonal = 1
    for (int i = 0; i < nbf; ++i) {
        EXPECT_NEAR(S_flat[static_cast<Size>(i) * nbf + i], 1.0, 1e-10)
            << "S(" << i << "," << i << ") should be 1.0";
    }

    // Symmetric
    for (int i = 0; i < nbf; ++i) {
        for (int j = i + 1; j < nbf; ++j) {
            EXPECT_NEAR(S_flat[static_cast<Size>(i) * nbf + j],
                        S_flat[static_cast<Size>(j) * nbf + i], 1e-12)
                << "S(" << i << "," << j << ") != S(" << j << "," << i << ")";
        }
    }

    // Positive definite
    auto S = to_eigen(S_flat, nbf);
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es(S);
    for (int i = 0; i < nbf; ++i) {
        EXPECT_GT(es.eigenvalues()(i), -1e-10)
            << "Overlap eigenvalue " << i << " should be positive";
    }
}
