// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2026 Bryce M. Westheimer. All rights reserved.
//
// gpu_hf_workflow.cpp
//
// Demonstrates GPU-accelerated Hartree-Fock energy calculation for H2O
// using LibAccInt. This example:
//   1. Constructs an STO-3G basis set for H2O
//   2. Uses CudaEngine directly for fused S+T+V computation
//   3. Uses GpuFockBuilder for device-side ERI accumulation
//   4. Builds the Fock matrix and computes the HF energy
//   5. Compares timing between CPU and GPU paths
//   6. Demonstrates multi-threaded concurrent GPU execution
//   7. Falls back gracefully to CPU when CUDA is not available
//
// Build with: cmake --preset cuda-release && cmake --build --preset cuda-release
// Run:        ./build/cuda-release/examples/gpu_hf_workflow

#include <libaccint/libaccint.hpp>
#include <libaccint/consumers/fock_builder.hpp>
#include <libaccint/data/builtin_basis.hpp>

#if LIBACCINT_USE_CUDA
#include <libaccint/engine/cuda_engine.hpp>
#include <libaccint/consumers/fock_builder_gpu.hpp>
#endif

#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <thread>
#include <vector>

using namespace libaccint;

/// Helper: compute electronic energy E_elec = 0.5 * Tr[D * (H + F)]
static Real compute_electronic_energy(const std::vector<Real>& D,
                                       const std::vector<Real>& H_core,
                                       const std::vector<Real>& F,
                                       Size nbf) {
    Real E = 0.0;
    for (Size i = 0; i < nbf; ++i) {
        for (Size j = 0; j < nbf; ++j) {
            E += D[i * nbf + j] * (H_core[i * nbf + j] + F[i * nbf + j]);
        }
    }
    return 0.5 * E;
}

/// Helper: compute nuclear repulsion energy
static Real compute_nuclear_repulsion(const std::vector<data::Atom>& atoms) {
    Real V_nn = 0.0;
    for (size_t i = 0; i < atoms.size(); ++i) {
        for (size_t j = i + 1; j < atoms.size(); ++j) {
            Real dx = atoms[i].position.x - atoms[j].position.x;
            Real dy = atoms[i].position.y - atoms[j].position.y;
            Real dz = atoms[i].position.z - atoms[j].position.z;
            Real r = std::sqrt(dx * dx + dy * dy + dz * dz);
            V_nn += static_cast<Real>(atoms[i].atomic_number *
                                       atoms[j].atomic_number) / r;
        }
    }
    return V_nn;
}

int main() {
    std::cout << "=== LibAccInt GPU HF Workflow Example ===\n";
    std::cout << "LibAccInt version: " << version() << "\n";
    std::cout << "CUDA backend: " << (has_cuda_backend() ? "compiled" : "not compiled") << "\n";
    std::cout << "OpenMP: " << (has_openmp() ? "available" : "not available") << "\n\n";

    // ── Step 1: Define H2O molecule ──────────────────────────────────────────
    // Geometry in Bohr (atomic units)
    std::vector<data::Atom> atoms = {
        {8, {0.000000,  0.000000,  0.117176}},  // O
        {1, {0.000000,  1.430665, -0.468706}},  // H
        {1, {0.000000, -1.430665, -0.468706}},  // H
    };

    // ── Step 2: Create basis set ─────────────────────────────────────────────
    BasisSet basis = data::create_builtin_basis("STO-3G", atoms);
    const Size nbf = basis.n_basis_functions();

    std::cout << "Molecule: H2O\n";
    std::cout << "Basis set: STO-3G\n";
    std::cout << "Basis functions: " << nbf << "\n";
    std::cout << "Shells: " << basis.n_shells() << "\n\n";

    // Nuclear charges for V integral
    PointChargeParams nuclear_charges;
    for (const auto& atom : atoms) {
        nuclear_charges.x.push_back(atom.position.x);
        nuclear_charges.y.push_back(atom.position.y);
        nuclear_charges.z.push_back(atom.position.z);
        nuclear_charges.charge.push_back(static_cast<Real>(atom.atomic_number));
    }

    // Simple density matrix (diagonal approximation)
    std::vector<Real> D(nbf * nbf, 0.0);
    for (Size i = 0; i < nbf; ++i) {
        D[i * nbf + i] = (i < 5) ? 1.0 : 0.0;  // 5 occupied MOs for H2O/STO-3G
    }

    // ── Step 3: CPU path for comparison ──────────────────────────────────────
    std::cout << "--- CPU Path ---\n";
    auto cpu_start = std::chrono::high_resolution_clock::now();

    Engine engine(basis);

    std::vector<Real> S_cpu, T_cpu, V_cpu;
    engine.compute_overlap_matrix(S_cpu, BackendHint::ForceCPU);
    engine.compute_kinetic_matrix(T_cpu, BackendHint::ForceCPU);
    engine.compute_nuclear_matrix(nuclear_charges, V_cpu, BackendHint::ForceCPU);

    std::vector<Real> H_core_cpu(nbf * nbf, 0.0);
    for (Size i = 0; i < nbf * nbf; ++i) {
        H_core_cpu[i] = T_cpu[i] + V_cpu[i];
    }

    consumers::FockBuilder fock_cpu(nbf);
    fock_cpu.set_density(D.data(), nbf);
    engine.compute_and_consume(Operator::coulomb(), fock_cpu, BackendHint::ForceCPU);

    auto F_cpu = fock_cpu.get_fock_matrix(
        std::span<const Real>(H_core_cpu), 0.5);
    Real E_elec_cpu = compute_electronic_energy(D, H_core_cpu, F_cpu, nbf);

    auto cpu_end = std::chrono::high_resolution_clock::now();
    double cpu_ms = std::chrono::duration<double, std::milli>(cpu_end - cpu_start).count();

    Real V_nn = compute_nuclear_repulsion(atoms);
    Real E_total_cpu = E_elec_cpu + V_nn;

    std::cout << std::fixed << std::setprecision(10);
    std::cout << "Electronic energy: " << E_elec_cpu << " Hartree\n";
    std::cout << "Nuclear repulsion: " << V_nn << " Hartree\n";
    std::cout << "Total energy:      " << E_total_cpu << " Hartree\n";
    std::cout << "CPU time:          " << std::setprecision(2) << cpu_ms << " ms\n\n";

    // ── Step 4: GPU path ─────────────────────────────────────────────────────
#if LIBACCINT_USE_CUDA
    if (engine.gpu_available()) {
        std::cout << "--- GPU Path ---\n";
        auto gpu_start = std::chrono::high_resolution_clock::now();

        CudaEngine* cuda_eng = engine.cuda_engine();

        // Fused S+T+V in a single kernel launch
        std::vector<Real> S_gpu, T_gpu, V_gpu;
        cuda_eng->compute_all_1e_fused(nuclear_charges, S_gpu, T_gpu, V_gpu);

        std::vector<Real> H_core_gpu(nbf * nbf, 0.0);
        for (Size i = 0; i < nbf * nbf; ++i) {
            H_core_gpu[i] = T_gpu[i] + V_gpu[i];
        }

        // GPU Fock builder for device-side ERI accumulation
        consumers::GpuFockBuilder fock_gpu(nbf);
        fock_gpu.set_density(D.data(), nbf);

        // Drive ERI computation through the Engine with GPU hint
        engine.compute_and_consume(Operator::coulomb(), fock_gpu, BackendHint::ForceGPU);
        fock_gpu.synchronize();

        auto F_gpu = fock_gpu.get_fock_matrix(
            std::span<const Real>(H_core_gpu), 0.5);
        Real E_elec_gpu = compute_electronic_energy(D, H_core_gpu, F_gpu, nbf);

        auto gpu_end = std::chrono::high_resolution_clock::now();
        double gpu_ms = std::chrono::duration<double, std::milli>(gpu_end - gpu_start).count();

        Real E_total_gpu = E_elec_gpu + V_nn;

        std::cout << std::fixed << std::setprecision(10);
        std::cout << "Electronic energy: " << E_elec_gpu << " Hartree\n";
        std::cout << "Nuclear repulsion: " << V_nn << " Hartree\n";
        std::cout << "Total energy:      " << E_total_gpu << " Hartree\n";
        std::cout << "GPU time:          " << std::setprecision(2) << gpu_ms << " ms\n\n";

        // Compare results
        Real energy_diff = std::abs(E_total_gpu - E_total_cpu);
        std::cout << "--- Comparison ---\n";
        std::cout << std::scientific << std::setprecision(2);
        std::cout << "Energy difference:  " << energy_diff << " Hartree\n";
        std::cout << std::fixed << std::setprecision(2);
        std::cout << "Speedup:            " << cpu_ms / gpu_ms << "x\n";

        if (energy_diff < 1e-10) {
            std::cout << "Results agree within machine precision.\n";
        }
        // ── Step 5: Multi-threaded concurrent GPU execution ─────────────────
        // CudaEngine manages a pool of GPU execution slots (each with its own
        // CUDA stream and device buffers). Multiple host threads can safely
        // call compute_batch() on the same Engine — each thread automatically
        // acquires an independent slot.
        std::cout << "\n--- Concurrent GPU Path ---\n";
        std::cout << "Running compute_batch_parallel with "
                  << std::thread::hardware_concurrency() << " available cores\n";

        auto par_start = std::chrono::high_resolution_clock::now();

        // compute_batch_parallel dispatches quartets across OpenMP threads.
        // Each thread acquires its own GPU execution slot internally.
        auto par_results = engine.compute_batch_parallel(
            Operator::coulomb(),
            basis.shell_set_quartets(),
            /*n_threads=*/0,  // 0 = auto-detect
            BackendHint::ForceGPU);

        auto par_end = std::chrono::high_resolution_clock::now();
        double par_ms = std::chrono::duration<double, std::milli>(par_end - par_start).count();

        std::cout << "Parallel GPU time: " << std::fixed << std::setprecision(2)
                  << par_ms << " ms (" << par_results.size() << " batches)\n";

        // To control the number of concurrent GPU streams (e.g., on a
        // memory-limited GPU), configure n_gpu_slots via DispatchConfig:
        //
        //   DispatchConfig config;
        //   config.n_gpu_slots = 2;  // only 2 concurrent streams
        //   Engine engine(basis, config);
    } else {
        std::cout << "GPU not available — CUDA was compiled but no device found.\n";
        std::cout << "Skipping GPU path.\n";
    }
#else
    std::cout << "--- GPU Path ---\n";
    std::cout << "CUDA not compiled. Build with cmake --preset cuda-release\n";
    std::cout << "to enable GPU acceleration.\n";
#endif

    std::cout << "\nNote: This uses a non-SCF density. A real HF code would\n";
    std::cout << "iterate to self-consistency.\n";

    return 0;
}
