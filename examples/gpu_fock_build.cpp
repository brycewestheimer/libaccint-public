// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2026 Bryce M. Westheimer. All rights reserved.
//
// gpu_fock_build.cpp
//
// Demonstrates GPU-accelerated Fock matrix construction using LibAccInt.
// When compiled with CUDA support (LIBACCINT_USE_CUDA), this example uses
// the GPU backend for two-electron integral computation. On CPU-only builds,
// it gracefully falls back to the CPU backend.
//
// Key concepts demonstrated:
//   - Backend detection and selection
//   - BackendHint for routing computation
//   - FockBuilder compute-and-consume pattern on GPU
//   - Performance timing comparison

#include <libaccint/libaccint.hpp>
#include <libaccint/consumers/fock_builder.hpp>
#include <libaccint/data/builtin_basis.hpp>
#include <libaccint/config.hpp>

#include <chrono>
#include <iomanip>
#include <iostream>
#include <vector>

using namespace libaccint;

int main() {
    std::cout << "=== LibAccInt GPU Fock Build Example ===\n";
    std::cout << "LibAccInt version: " << version() << "\n";
    std::cout << "CUDA backend: " << (has_cuda_backend() ? "enabled" : "disabled") << "\n\n";

    // ── Define water molecule (H₂O) ─────────────────────────────────────────
    // Geometry in Bohr (atomic units)
    std::vector<data::Atom> atoms = {
        {8, {0.0000,  0.0000, 0.2217}},   // O
        {1, {0.0000,  1.4309, -0.8867}},  // H
        {1, {0.0000, -1.4309, -0.8867}},  // H
    };

    BasisSet basis = data::create_builtin_basis("STO-3G", atoms);
    const Size nbf = basis.n_basis_functions();

    std::cout << "Molecule: H2O\n";
    std::cout << "Basis set: STO-3G\n";
    std::cout << "Basis functions: " << nbf << "\n";
    std::cout << "Shells: " << basis.n_shells() << "\n\n";

    Engine engine(basis);

    // Report GPU status
    if (engine.gpu_available()) {
        std::cout << "GPU backend is available — will use GPU acceleration\n\n";
    } else {
        std::cout << "GPU backend not available — using CPU backend\n";
        std::cout << "(Build with -DLIBACCINT_USE_CUDA=ON for GPU support)\n\n";
    }

    // ── Compute one-electron integrals (always CPU for 1e) ──────────────────
    std::vector<Real> S(nbf * nbf, 0.0);
    engine.compute_overlap_matrix(S);

    PointChargeParams nuclear_charges;
    for (const auto& atom : atoms) {
        nuclear_charges.x.push_back(atom.position.x);
        nuclear_charges.y.push_back(atom.position.y);
        nuclear_charges.z.push_back(atom.position.z);
        nuclear_charges.charge.push_back(static_cast<Real>(atom.atomic_number));
    }

    std::vector<Real> H_core(nbf * nbf, 0.0);
    engine.compute_core_hamiltonian(nuclear_charges, H_core);

    // ── Set up density matrix (identity for demo) ─────────────────────────
    std::vector<Real> D(nbf * nbf, 0.0);
    for (Size i = 0; i < nbf; ++i) {
        D[i * nbf + i] = 1.0 / static_cast<Real>(nbf);
    }

    // ── CPU Fock build with timing ─────────────────────────────────────────
    consumers::FockBuilder fock_cpu(nbf);
    fock_cpu.set_density(D.data(), nbf);

    auto t0 = std::chrono::high_resolution_clock::now();
    engine.compute(Operator::coulomb(), fock_cpu, BackendHint::ForceCPU);
    auto t1 = std::chrono::high_resolution_clock::now();

    double cpu_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    std::cout << "CPU Fock build: " << std::fixed << std::setprecision(3)
              << cpu_ms << " ms\n";

    // ── GPU Fock build with timing (if available) ──────────────────────────
    if (engine.gpu_available()) {
        consumers::FockBuilder fock_gpu(nbf);
        fock_gpu.set_density(D.data(), nbf);

        auto t2 = std::chrono::high_resolution_clock::now();
        engine.compute(Operator::coulomb(), fock_gpu, BackendHint::PreferGPU);
        auto t3 = std::chrono::high_resolution_clock::now();

        double gpu_ms = std::chrono::duration<double, std::milli>(t3 - t2).count();
        std::cout << "GPU Fock build: " << std::setprecision(3) << gpu_ms << " ms\n";
        std::cout << "Speedup: " << std::setprecision(2) << (cpu_ms / gpu_ms) << "x\n";

        // Verify CPU and GPU results match
        auto J_cpu = fock_cpu.get_coulomb_matrix();
        auto J_gpu = fock_gpu.get_coulomb_matrix();

        double max_diff = 0.0;
        for (Size i = 0; i < nbf * nbf; ++i) {
            max_diff = std::max(max_diff, std::abs(J_cpu[i] - J_gpu[i]));
        }
        std::cout << "Max |J_cpu - J_gpu| = " << std::scientific
                  << std::setprecision(2) << max_diff << "\n";
    }

    // ── Print Coulomb matrix ───────────────────────────────────────────────
    std::cout << "\nCoulomb matrix J (CPU):\n";
    auto J = fock_cpu.get_coulomb_matrix();
    std::cout << std::fixed << std::setprecision(8);
    for (Size i = 0; i < nbf; ++i) {
        for (Size j = 0; j < nbf; ++j) {
            std::cout << std::setw(14) << J[i * nbf + j];
        }
        std::cout << '\n';
    }

    return 0;
}
