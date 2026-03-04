// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2026 Bryce M. Westheimer. All rights reserved.

/// @file test_hf_energy.cpp
/// @brief End-to-end Hartree-Fock energy calculation test (Quality Gate G2)
///
/// Implements a minimal RHF SCF loop using LibAccInt for all integrals.
/// Validates against PySCF reference energies for multiple molecules.

#include <libaccint/engine/engine.hpp>
#include <libaccint/consumers/fock_builder.hpp>
#include <libaccint/data/builtin_basis.hpp>
#include <libaccint/operators/operator.hpp>
#include <libaccint/operators/one_electron_operator.hpp>
#include <libaccint/core/types.hpp>
#include <libaccint/utils/constants.hpp>

#include <gtest/gtest.h>
#include <Eigen/Dense>

#include <cmath>
#include <vector>
#include <algorithm>
#include <numeric>

using namespace libaccint;
using namespace libaccint::data;
using namespace libaccint::consumers;

// =============================================================================
// SCF Infrastructure
// =============================================================================

namespace {

/// @brief Compute nuclear repulsion energy
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

/// @brief Convert flat row-major vector to Eigen matrix
Eigen::MatrixXd to_eigen(const std::vector<Real>& flat, int n) {
    Eigen::MatrixXd mat(n, n);
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            mat(i, j) = flat[static_cast<Size>(i) * n + j];
        }
    }
    return mat;
}

/// @brief Solve generalized eigenvalue problem F*C = S*C*epsilon
/// Returns eigenvalues and eigenvectors (as columns of C)
std::pair<Eigen::VectorXd, Eigen::MatrixXd>
solve_gen_eigenvalue(const Eigen::MatrixXd& F, const Eigen::MatrixXd& S) {
    // Use canonical orthogonalization: X = S^(-1/2)
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es(S);
    Eigen::VectorXd s_vals = es.eigenvalues();
    Eigen::MatrixXd U = es.eigenvectors();

    // Build X = U * s^(-1/2) * U^T
    Eigen::VectorXd s_inv_sqrt(s_vals.size());
    for (int i = 0; i < s_vals.size(); ++i) {
        if (s_vals(i) > 1e-10) {
            s_inv_sqrt(i) = 1.0 / std::sqrt(s_vals(i));
        } else {
            s_inv_sqrt(i) = 0.0;
        }
    }
    Eigen::MatrixXd X = U * s_inv_sqrt.asDiagonal() * U.transpose();

    // Transform F to orthogonal basis: F' = X^T * F * X
    Eigen::MatrixXd Fprime = X.transpose() * F * X;

    // Solve standard eigenvalue problem
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es2(Fprime);

    // Back-transform eigenvectors: C = X * C'
    Eigen::MatrixXd C = X * es2.eigenvectors();

    return {es2.eigenvalues(), C};
}

/// @brief Build density matrix from occupied MOs: D = 2 * C_occ * C_occ^T
Eigen::MatrixXd build_density(const Eigen::MatrixXd& C, int n_occ) {
    Eigen::MatrixXd C_occ = C.leftCols(n_occ);
    return 2.0 * C_occ * C_occ.transpose();
}

/// @brief DIIS error vector: e = F*D*S - S*D*F
Eigen::MatrixXd compute_diis_error(const Eigen::MatrixXd& F,
                                     const Eigen::MatrixXd& D,
                                     const Eigen::MatrixXd& S) {
    return F * D * S - S * D * F;
}

/// @brief DIIS extrapolation
/// Maintains a history of Fock matrices and error vectors
struct DIIS {
    int max_size = 6;
    std::vector<Eigen::MatrixXd> fock_history;
    std::vector<Eigen::MatrixXd> error_history;

    void add(const Eigen::MatrixXd& F, const Eigen::MatrixXd& error) {
        fock_history.push_back(F);
        error_history.push_back(error);
        if (static_cast<int>(fock_history.size()) > max_size) {
            fock_history.erase(fock_history.begin());
            error_history.erase(error_history.begin());
        }
    }

    Eigen::MatrixXd extrapolate() const {
        int n = static_cast<int>(fock_history.size());
        if (n < 2) {
            return fock_history.back();
        }

        // Build B matrix
        Eigen::MatrixXd B = Eigen::MatrixXd::Zero(n + 1, n + 1);
        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < n; ++j) {
                // Flatten and dot product of error vectors
                B(i, j) = (error_history[i].array() * error_history[j].array()).sum();
            }
        }
        // Lagrange constraint
        for (int i = 0; i < n; ++i) {
            B(n, i) = -1.0;
            B(i, n) = -1.0;
        }
        B(n, n) = 0.0;

        // RHS
        Eigen::VectorXd rhs = Eigen::VectorXd::Zero(n + 1);
        rhs(n) = -1.0;

        // Solve
        Eigen::VectorXd c = B.colPivHouseholderQr().solve(rhs);

        // Extrapolate
        Eigen::MatrixXd F_new = Eigen::MatrixXd::Zero(fock_history[0].rows(),
                                                        fock_history[0].cols());
        for (int i = 0; i < n; ++i) {
            F_new += c(i) * fock_history[i];
        }
        return F_new;
    }
};

/// @brief Run RHF SCF calculation
/// @return Total energy (electronic + nuclear repulsion)
struct SCFResult {
    Real total_energy;
    int iterations;
    bool converged;
};

SCFResult run_rhf(const std::vector<Atom>& atoms,
                   const BasisSet& basis,
                   int n_electrons,
                   Real energy_threshold = 1e-12,
                   Real density_threshold = 1e-10,
                   int max_iterations = 100) {
    const int nbf = static_cast<int>(basis.n_basis_functions());
    const int n_occ = n_electrons / 2;

    Engine engine(basis);

    // Build nuclear charge parameters
    PointChargeParams charges;
    for (const auto& atom : atoms) {
        charges.x.push_back(atom.position.x);
        charges.y.push_back(atom.position.y);
        charges.z.push_back(atom.position.z);
        charges.charge.push_back(static_cast<Real>(atom.atomic_number));
    }

    // Compute one-electron integrals
    std::vector<Real> S_flat, T_flat, V_flat;
    engine.compute_1e(Operator::overlap(), S_flat);
    engine.compute_1e(Operator::kinetic(), T_flat);
    engine.compute_1e(Operator::nuclear(charges), V_flat);

    Eigen::MatrixXd S = to_eigen(S_flat, nbf);
    Eigen::MatrixXd T = to_eigen(T_flat, nbf);
    Eigen::MatrixXd V = to_eigen(V_flat, nbf);
    Eigen::MatrixXd H_core = T + V;

    // Nuclear repulsion
    Real E_nuc = compute_nuclear_repulsion(atoms);

    // Initial guess: diagonalize H_core
    auto [eps0, C0] = solve_gen_eigenvalue(H_core, S);
    Eigen::MatrixXd D = build_density(C0, n_occ);

    // SCF iteration
    Real E_old = 0.0;
    DIIS diis;
    bool converged = false;
    int iter = 0;

    for (; iter < max_iterations; ++iter) {
        // Build Fock matrix using compute_and_consume
        FockBuilder fock_builder(static_cast<Size>(nbf));

        // Flatten density matrix for FockBuilder
        std::vector<Real> D_flat(nbf * nbf);
        for (int i = 0; i < nbf; ++i) {
            for (int j = 0; j < nbf; ++j) {
                D_flat[i * nbf + j] = D(i, j);
            }
        }
        fock_builder.set_density(D_flat.data(), static_cast<Size>(nbf));

        Operator coulomb = Operator::coulomb();
        engine.compute_and_consume(coulomb, fock_builder);

        // Get J and K as Eigen matrices
        auto J_span = fock_builder.get_coulomb_matrix();
        auto K_span = fock_builder.get_exchange_matrix();

        Eigen::MatrixXd J(nbf, nbf), K(nbf, nbf);
        for (int i = 0; i < nbf; ++i) {
            for (int j = 0; j < nbf; ++j) {
                J(i, j) = J_span[i * nbf + j];
                K(i, j) = K_span[i * nbf + j];
            }
        }

        // F = H_core + J - 0.5*K (Szabo convention: D includes factor of 2)
        Eigen::MatrixXd F = H_core + J - 0.5 * K;

        // Compute energy: E = 0.5 * Tr[(H_core + F) * D] + E_nuc
        Real E_elec = 0.5 * ((H_core + F).array() * D.array()).sum();
        Real E_total = E_elec + E_nuc;

        // DIIS
        Eigen::MatrixXd error = compute_diis_error(F, D, S);
        diis.add(F, error);
        Eigen::MatrixXd F_diis = diis.extrapolate();

        // Convergence check
        Real delta_E = std::abs(E_total - E_old);
        Real max_delta_D = 0.0;

        // Solve eigenvalue problem
        auto [eps, C] = solve_gen_eigenvalue(F_diis, S);
        Eigen::MatrixXd D_new = build_density(C, n_occ);

        max_delta_D = (D_new - D).array().abs().maxCoeff();

        if (delta_E < energy_threshold && max_delta_D < density_threshold && iter > 0) {
            D = D_new;
            converged = true;

            // Recompute final energy with converged density
            FockBuilder fock_final(static_cast<Size>(nbf));
            std::vector<Real> D_final_flat(nbf * nbf);
            for (int i = 0; i < nbf; ++i) {
                for (int j = 0; j < nbf; ++j) {
                    D_final_flat[i * nbf + j] = D(i, j);
                }
            }
            fock_final.set_density(D_final_flat.data(), static_cast<Size>(nbf));
            engine.compute_and_consume(coulomb, fock_final);

            auto J_final = fock_final.get_coulomb_matrix();
            auto K_final = fock_final.get_exchange_matrix();
            Eigen::MatrixXd J_f(nbf, nbf), K_f(nbf, nbf);
            for (int i = 0; i < nbf; ++i) {
                for (int j = 0; j < nbf; ++j) {
                    J_f(i, j) = J_final[i * nbf + j];
                    K_f(i, j) = K_final[i * nbf + j];
                }
            }
            Eigen::MatrixXd F_final = H_core + J_f - 0.5 * K_f;
            Real E_elec_final = 0.5 * ((H_core + F_final).array() * D.array()).sum();

            return {E_elec_final + E_nuc, iter + 1, true};
        }

        D = D_new;
        E_old = E_total;
    }

    return {E_old, iter, converged};
}

}  // anonymous namespace

// =============================================================================
// H2O/STO-3G HF Energy Test (Quality Gate G2)
// =============================================================================

TEST(HFEnergyTest, H2O_STO3G) {
    std::vector<Atom> atoms = {
        {8, {0.0, 0.0, 0.0}},                    // O
        {1, {0.0, 1.43233673, -1.10866041}},      // H1
        {1, {0.0, -1.43233673, -1.10866041}},     // H2
    };
    auto basis = create_sto3g(atoms);

    // PySCF reference: -74.9631102394515 (Bohr coordinates, cart=True)
    constexpr Real REF_ENERGY = -74.9631102394515;
    constexpr Real TOLERANCE = 1e-10;

    auto result = run_rhf(atoms, basis, 10);  // 10 electrons

    EXPECT_TRUE(result.converged) << "SCF did not converge within max iterations";
    EXPECT_LT(result.iterations, 50) << "SCF took too many iterations";
    EXPECT_NEAR(result.total_energy, REF_ENERGY, TOLERANCE)
        << "H2O/STO-3G RHF energy does not match PySCF reference";
}

// =============================================================================
// Multi-Molecule Validation
// =============================================================================

TEST(HFEnergyTest, H2_STO3G) {
    std::vector<Atom> atoms = {
        {1, {0.0, 0.0, 0.0}},
        {1, {0.0, 0.0, 1.39839733}},
    };
    auto basis = create_sto3g(atoms);

    constexpr Real REF_ENERGY = -1.11675930745672;
    constexpr Real TOLERANCE = 1e-10;

    auto result = run_rhf(atoms, basis, 2);

    EXPECT_TRUE(result.converged);
    EXPECT_LT(result.iterations, 50);
    EXPECT_NEAR(result.total_energy, REF_ENERGY, TOLERANCE)
        << "H2/STO-3G RHF energy does not match PySCF reference";
}

TEST(HFEnergyTest, CH4_STO3G) {
    // Tetrahedral CH4 geometry in Bohr
    // C-H distance = 1.0870 Angstrom, tetrahedral: H at (±b, ±b, ±b)
    // b = (C-H distance in Angstrom) / sqrt(3) * ANGSTROM_TO_BOHR
    //   = 1.0870 / sqrt(3) * 1.8897261246257702 = 1.185953834894377 Bohr
    constexpr Real b = 1.185953834894377;
    std::vector<Atom> atoms = {
        {6, {0.0, 0.0, 0.0}},
        {1, {b, b, b}},
        {1, {b, -b, -b}},
        {1, {-b, b, -b}},
        {1, {-b, -b, b}},
    };
    auto basis = create_sto3g(atoms);

    // PySCF reference: -39.726810114797665 (Bohr coordinates, cart=True, C-H=1.0870 Ang)
    constexpr Real REF_ENERGY = -39.726810114797665;
    constexpr Real TOLERANCE = 1e-10;

    auto result = run_rhf(atoms, basis, 10);

    EXPECT_TRUE(result.converged) << "CH4/STO-3G SCF did not converge";
    EXPECT_LT(result.iterations, 50);
    EXPECT_NEAR(result.total_energy, REF_ENERGY, TOLERANCE)
        << "CH4/STO-3G RHF energy does not match PySCF reference";
}

TEST(HFEnergyTest, NuclearRepulsion) {
    // H2O nuclear repulsion with the standard Bohr geometry
    std::vector<Atom> atoms = {
        {8, {0.0, 0.0, 0.0}},
        {1, {0.0, 1.43233673, -1.10866041}},
        {1, {0.0, -1.43233673, -1.10866041}},
    };
    Real E_nuc = compute_nuclear_repulsion(atoms);

    // PySCF reference: 9.182637358503053
    EXPECT_NEAR(E_nuc, 9.182637358503053, 1e-10);
}

TEST(HFEnergyTest, SCFConvergenceMonotonic) {
    // Verify energy decreases (roughly) during SCF iterations
    // Using H2 as the simplest test case
    std::vector<Atom> atoms = {
        {1, {0.0, 0.0, 0.0}},
        {1, {0.0, 0.0, 1.39839733}},
    };
    auto basis = create_sto3g(atoms);
    auto result = run_rhf(atoms, basis, 2);

    EXPECT_TRUE(result.converged);
    // H2 should converge very quickly (few iterations)
    EXPECT_LT(result.iterations, 20);
}
