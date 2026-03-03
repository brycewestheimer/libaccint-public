// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2026 Bryce M. Westheimer. All rights reserved.
//
// hf_calculation.cpp
//
// Complete Restricted Hartree-Fock (RHF) SCF calculation for H2O using
// the cc-pVDZ basis set loaded from the Basis Set Exchange.
//
// This example demonstrates:
//   1. Molecule construction with atomic coordinates in Bohr
//   2. Basis set lookup by name string via the BSE parser
//   3. Core Hamiltonian assembly via ShellSetPair batching
//   4. Canonical orthogonalization (Lowdin S^{-1/2})
//   5. Full iterative SCF cycle with DIIS acceleration
//   6. Eigensolve and density construction at each iteration
//   7. Convergence monitoring with energy and density criteria
//   8. Final orbital energies and total energy printout
//
// Build:  cmake --preset cpu-release && cmake --build --preset cpu-release --target hf_calculation
// Run:    ./build/cpu-release/examples/hf_calculation
//
// Expected result: H2O/cc-pVDZ RHF energy ~ -76.02 Hartree

#include <libaccint/libaccint.hpp>
#include <libaccint/consumers/fock_builder.hpp>
#include <libaccint/data/basis_parser.hpp>
#include <libaccint/data/builtin_basis.hpp>

#include <Eigen/Dense>

#include <cmath>
#include <iomanip>
#include <iostream>
#include <vector>

using namespace libaccint;
using namespace libaccint::data;
using namespace libaccint::consumers;

// =============================================================================
// Helper functions
// =============================================================================

namespace {

/// Compute classical nuclear repulsion energy: sum_{A>B} Z_A * Z_B / R_AB
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

/// Convert a flat row-major std::vector to an Eigen matrix
Eigen::MatrixXd to_eigen(const std::vector<Real>& flat, int n) {
    Eigen::MatrixXd mat(n, n);
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            mat(i, j) = flat[static_cast<Size>(i) * n + j];
        }
    }
    return mat;
}

/// Convert an Eigen matrix to a flat row-major std::vector
std::vector<Real> from_eigen(const Eigen::MatrixXd& mat, int n) {
    std::vector<Real> flat(n * n);
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            flat[i * n + j] = mat(i, j);
        }
    }
    return flat;
}

/// Solve the generalized eigenvalue problem F*C = S*C*eps via canonical
/// orthogonalization. Returns {eigenvalues, eigenvectors (as columns of C)}.
std::pair<Eigen::VectorXd, Eigen::MatrixXd>
solve_gen_eigenvalue(const Eigen::MatrixXd& F, const Eigen::MatrixXd& S) {
    // Diagonalize overlap matrix: S = U * s * U^T
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es(S);
    Eigen::VectorXd s_vals = es.eigenvalues();
    Eigen::MatrixXd U = es.eigenvectors();

    // Build orthogonalization matrix X = U * s^{-1/2} * U^T
    Eigen::VectorXd s_inv_sqrt(s_vals.size());
    for (int i = 0; i < s_vals.size(); ++i) {
        s_inv_sqrt(i) = (s_vals(i) > 1e-10)
                             ? 1.0 / std::sqrt(s_vals(i))
                             : 0.0;
    }
    Eigen::MatrixXd X = U * s_inv_sqrt.asDiagonal() * U.transpose();

    // Transform to orthogonal basis: F' = X^T * F * X
    Eigen::MatrixXd Fprime = X.transpose() * F * X;

    // Solve standard eigenvalue problem in orthogonal basis
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es2(Fprime);

    // Back-transform eigenvectors to AO basis: C = X * C'
    Eigen::MatrixXd C = X * es2.eigenvectors();
    return {es2.eigenvalues(), C};
}

/// Build the RHF density matrix: D = 2 * C_occ * C_occ^T
Eigen::MatrixXd build_density(const Eigen::MatrixXd& C, int n_occ) {
    Eigen::MatrixXd C_occ = C.leftCols(n_occ);
    return 2.0 * C_occ * C_occ.transpose();
}

/// DIIS (Direct Inversion in the Iterative Subspace) accelerator.
/// Maintains a history of Fock matrices and commutator error vectors,
/// then extrapolates an improved Fock matrix via least-squares.
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
        if (n < 2) return fock_history.back();

        // Build the B matrix: B_ij = <e_i | e_j>
        Eigen::MatrixXd B = Eigen::MatrixXd::Zero(n + 1, n + 1);
        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < n; ++j) {
                B(i, j) = (error_history[i].array() *
                            error_history[j].array()).sum();
            }
        }
        // Lagrange multiplier constraint: sum(c_i) = 1
        for (int i = 0; i < n; ++i) {
            B(n, i) = -1.0;
            B(i, n) = -1.0;
        }
        B(n, n) = 0.0;

        Eigen::VectorXd rhs = Eigen::VectorXd::Zero(n + 1);
        rhs(n) = -1.0;

        Eigen::VectorXd c = B.colPivHouseholderQr().solve(rhs);

        // Extrapolated Fock matrix: F_diis = sum(c_i * F_i)
        Eigen::MatrixXd F_new = Eigen::MatrixXd::Zero(
            fock_history[0].rows(), fock_history[0].cols());
        for (int i = 0; i < n; ++i) {
            F_new += c(i) * fock_history[i];
        }
        return F_new;
    }
};

}  // anonymous namespace

// =============================================================================
// Main RHF SCF calculation
// =============================================================================

int main() {
    std::cout << "=== LibAccInt: Complete RHF Calculation ===\n";
    std::cout << "LibAccInt version: " << version() << "\n\n";

    // ── Step 1: Define the molecular geometry ────────────────────────────────
    // H2O in Bohr (atomic units). This geometry gives a bond angle of ~104.5°.
    std::vector<Atom> atoms = {
        {8, {0.000000,  0.000000,  0.117176}},   // O
        {1, {0.000000,  1.430665, -0.468706}},   // H
        {1, {0.000000, -1.430665, -0.468706}},   // H
    };
    const int n_electrons = 10;
    const int n_occ = n_electrons / 2;  // 5 doubly-occupied MOs

    // ── Step 2: Load the basis set from the Basis Set Exchange ───────────────
    // load_basis_set() looks up a QCSchema JSON file by name in share/basis_sets/
    BasisSet basis = load_basis_set("cc-pvdz", atoms);
    const int nbf = static_cast<int>(basis.n_basis_functions());

    std::cout << "Molecule: H2O\n";
    std::cout << "Basis set: cc-pVDZ (loaded from BSE)\n";
    std::cout << "Basis functions: " << nbf << "\n";
    std::cout << "Shells: " << basis.n_shells() << "\n";
    std::cout << "Max angular momentum: " << basis.max_angular_momentum() << "\n";
    std::cout << "Electrons: " << n_electrons
              << " (" << n_occ << " doubly-occupied MOs)\n\n";

    // ALTERNATIVE basis set loading methods:
    //   BasisSet basis = create_builtin_basis("STO-3G", atoms);
    //   BasisSet basis = load_basis_set_from_file("path/to/basis.json", atoms);

    // ── Step 3: Create the computation engine ────────────────────────────────
    Engine engine(basis);

    // ALTERNATIVE: Engine with custom dispatch configuration
    //   DispatchConfig config;
    //   config.min_gpu_batch_size = 32;
    //   Engine engine(basis, config);
    //
    // ALTERNATIVE: Check GPU availability
    //   if (engine.gpu_available()) { ... }

    // ── Step 4: Compute one-electron integrals via ShellSetPair batching ─────
    // Build nuclear charge parameters (SoA layout)
    PointChargeParams charges;
    for (const auto& atom : atoms) {
        charges.x.push_back(atom.position.x);
        charges.y.push_back(atom.position.y);
        charges.z.push_back(atom.position.z);
        charges.charge.push_back(static_cast<Real>(atom.atomic_number));
    }

    // Compute S, T, V using the convenience methods, which internally iterate
    // over ShellSetPairs with proper symmetry handling.
    std::vector<Real> S_flat, T_flat, V_flat;
    engine.compute_overlap_matrix(S_flat);
    engine.compute_kinetic_matrix(T_flat);
    engine.compute_nuclear_matrix(charges, V_flat);

    // ALTERNATIVE (one-shot core Hamiltonian):
    //   std::vector<Real> H_flat;
    //   engine.compute_core_hamiltonian(charges, H_flat);
    //
    // ALTERNATIVE (full-basis compute_1e with OneElectronOperator):
    //   engine.compute_1e(Operator::overlap(), S_flat);
    //
    // ALTERNATIVE (explicit ShellSetPair batching for fine-grained control):
    //   for (const auto& pair : basis.shell_set_pairs()) {
    //       engine.compute_shell_set_pair(Operator::overlap(), pair, S_flat);
    //   }
    //   // Note: compute_shell_set_pair accumulates into the upper triangle
    //   // only. The full-basis methods (compute_overlap_matrix, compute_1e)
    //   // handle symmetrization automatically.

    // Convert to Eigen matrices for linear algebra
    Eigen::MatrixXd S = to_eigen(S_flat, nbf);
    Eigen::MatrixXd T = to_eigen(T_flat, nbf);
    Eigen::MatrixXd V = to_eigen(V_flat, nbf);

    // ── Step 5: Build the core Hamiltonian H_core = T + V ────────────────────
    Eigen::MatrixXd H_core = T + V;

    std::cout << "One-electron integrals computed.\n";
    std::cout << "  Tr(S) = " << std::fixed << std::setprecision(6) << S.trace() << "\n";
    std::cout << "  Tr(T) = " << T.trace() << "\n";
    std::cout << "  Tr(V) = " << V.trace() << "\n\n";

    // ── Step 6: Canonical orthogonalization: X = S^{-1/2} ────────────────────
    // This is done inside solve_gen_eigenvalue(), but we document it here.
    // The orthogonalization matrix transforms the AO basis to an orthonormal
    // basis where the overlap matrix is the identity.

    // ── Step 7: Initial guess — diagonalize H_core in orthogonal basis ───────
    auto [eps0, C0] = solve_gen_eigenvalue(H_core, S);
    Eigen::MatrixXd D = build_density(C0, n_occ);

    // ── Step 8: SCF iteration ────────────────────────────────────────────────
    constexpr int max_iterations = 100;
    constexpr Real energy_threshold = 1e-10;
    constexpr Real density_threshold = 1e-8;

    Real E_nuc = compute_nuclear_repulsion(atoms);
    Real E_old = 0.0;
    DIIS diis;
    bool converged = false;
    int iter = 0;

    // Store final eigenvalues for orbital energy printout
    Eigen::VectorXd orbital_energies;

    std::cout << "Starting SCF iteration (max " << max_iterations << " cycles)...\n";
    std::cout << std::string(72, '-') << "\n";
    std::cout << std::setw(5) << "Iter"
              << std::setw(22) << "E_total (Hartree)"
              << std::setw(18) << "Delta_E"
              << std::setw(18) << "Max|Delta_D|"
              << std::setw(8) << "Status"
              << "\n";
    std::cout << std::string(72, '-') << "\n";

    for (; iter < max_iterations; ++iter) {
        // (a) Build the Fock matrix using the compute-and-consume pattern.
        //     FockBuilder accumulates J (Coulomb) and K (exchange) matrices
        //     as two-electron integrals are computed batch-by-batch.
        FockBuilder fock_builder(static_cast<Size>(nbf));

        std::vector<Real> D_flat = from_eigen(D, nbf);
        fock_builder.set_density(D_flat.data(), static_cast<Size>(nbf));

        Operator coulomb = Operator::coulomb();
        engine.compute_and_consume(coulomb, fock_builder);

        // ALTERNATIVE (parallel):
        //   engine.compute_and_consume_parallel(coulomb, fock_builder, /*n_threads=*/4);
        //
        // ALTERNATIVE (with Schwarz screening for larger systems):
        //   screening::ScreeningOptions opts = screening::ScreeningOptions::normal();
        //   engine.compute_and_consume_screened_parallel(coulomb, fock_builder, opts, 4);
        //
        // ALTERNATIVE (manual thread-safe accumulation):
        //   fock_builder.set_threading_strategy(FockThreadingStrategy::ThreadLocal);
        //   fock_builder.prepare_parallel(4);
        //   engine.compute_and_consume_parallel(coulomb, fock_builder, 4);
        //   fock_builder.finalize_parallel();

        // (b) Extract J and K, form Fock matrix: F = H_core + J - 0.5*K
        //     The 0.5 factor is because our density matrix includes a factor
        //     of 2 from double occupancy (Szabo & Ostlund convention).
        auto J_span = fock_builder.get_coulomb_matrix();
        auto K_span = fock_builder.get_exchange_matrix();

        Eigen::MatrixXd J(nbf, nbf), K(nbf, nbf);
        for (int i = 0; i < nbf; ++i) {
            for (int j = 0; j < nbf; ++j) {
                J(i, j) = J_span[i * nbf + j];
                K(i, j) = K_span[i * nbf + j];
            }
        }

        Eigen::MatrixXd F = H_core + J - 0.5 * K;

        // (c) Compute electronic energy: E_elec = 0.5 * Tr[D * (H_core + F)]
        Real E_elec = 0.5 * ((H_core + F).array() * D.array()).sum();
        Real E_total = E_elec + E_nuc;

        // (d) DIIS: compute commutator error e = FDS - SDF, then extrapolate
        Eigen::MatrixXd error = F * D * S - S * D * F;
        diis.add(F, error);
        Eigen::MatrixXd F_diis = diis.extrapolate();

        // (e) Convergence check
        Real delta_E = std::abs(E_total - E_old);

        // (f) Diagonalize the DIIS-extrapolated Fock matrix
        auto [eps, C] = solve_gen_eigenvalue(F_diis, S);
        Eigen::MatrixXd D_new = build_density(C, n_occ);

        Real max_delta_D = (D_new - D).array().abs().maxCoeff();

        // Print iteration info
        std::cout << std::setw(5) << iter + 1
                  << std::fixed << std::setprecision(12)
                  << std::setw(22) << E_total
                  << std::scientific << std::setprecision(4)
                  << std::setw(18) << delta_E
                  << std::setw(18) << max_delta_D;

        if (delta_E < energy_threshold && max_delta_D < density_threshold && iter > 0) {
            std::cout << std::setw(8) << "CONV" << "\n";
            D = D_new;
            orbital_energies = eps;
            converged = true;

            // Recompute final energy with the fully converged density
            FockBuilder fock_final(static_cast<Size>(nbf));
            std::vector<Real> D_final_flat = from_eigen(D, nbf);
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
            E_elec = 0.5 * ((H_core + F_final).array() * D.array()).sum();
            E_total = E_elec + E_nuc;

            // ── Step 9: Print final results ──────────────────────────────
            std::cout << std::string(72, '-') << "\n\n";
            std::cout << "SCF converged in " << iter + 1 << " iterations.\n\n";

            std::cout << std::fixed << std::setprecision(12);
            std::cout << "Electronic energy:     " << E_elec << " Hartree\n";
            std::cout << "Nuclear repulsion:     " << E_nuc << " Hartree\n";
            std::cout << "Total RHF energy:      " << E_total << " Hartree\n\n";

            std::cout << "Orbital energies (Hartree):\n";
            std::cout << std::fixed << std::setprecision(6);
            for (int i = 0; i < nbf; ++i) {
                std::cout << "  " << std::setw(3) << i + 1 << ": "
                          << std::setw(14) << orbital_energies(i)
                          << (i < n_occ ? "  (occupied)" : "  (virtual)")
                          << "\n";
            }
            std::cout << "\n";

            return 0;
        }

        std::cout << "\n";

        D = D_new;
        E_old = E_total;
        orbital_energies = eps;
    }

    // If we get here, SCF did not converge
    std::cout << std::string(72, '-') << "\n";
    std::cerr << "WARNING: SCF did not converge within "
              << max_iterations << " iterations.\n";
    std::cout << "Last energy: " << std::fixed << std::setprecision(12)
              << E_old << " Hartree\n";

    // ── GPU acceleration notes ───────────────────────────────────────────────
#if LIBACCINT_USE_CUDA
    // When compiled with CUDA support, the Engine can dispatch to GPU:
    //
    //   Engine engine(basis);
    //   if (engine.gpu_available()) {
    //       // Force all 2e integrals to GPU (serial)
    //       engine.compute_and_consume(Operator::coulomb(), fock,
    //                                  BackendHint::ForceGPU);
    //
    //       // Or use the dedicated CudaEngine for fused 1e integrals:
    //       CudaEngine* cuda = engine.cuda_engine();
    //       std::vector<Real> S_gpu, T_gpu, V_gpu;
    //       cuda->compute_all_1e_fused(charges, S_gpu, T_gpu, V_gpu);
    //
    //       // GpuFockBuilder keeps J/K on device for minimal transfers:
    //       GpuFockBuilder gpu_fock(nbf);
    //       gpu_fock.set_density(D_flat.data(), nbf);
    //       engine.compute_and_consume(Operator::coulomb(), gpu_fock,
    //                                  BackendHint::ForceGPU);
    //       gpu_fock.synchronize();
    //   }
    //
    // CONCURRENT GPU EXECUTION:
    //   Multiple host threads can safely call compute_batch() on the same
    //   Engine — CudaEngine internally manages a pool of GPU execution slots
    //   (each with its own CUDA stream and device buffers).
    //
    //   // Parallel batch computation across multiple GPU streams:
    //   auto results = engine.compute_batch_parallel(
    //       Operator::coulomb(), basis.shell_set_quartets(),
    //       /*n_threads=*/4, BackendHint::ForceGPU);
    //
    //   // Control concurrency for memory-limited GPUs:
    //   DispatchConfig config;
    //   config.n_gpu_slots = 2;  // 2 concurrent streams instead of default 4
    //   Engine engine(basis, config);
#endif

    return 1;
}
