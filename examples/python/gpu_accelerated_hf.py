#!/usr/bin/env python3
# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Bryce M. Westheimer. All rights reserved.
#
# gpu_accelerated_hf.py
#
# Demonstrates GPU-accelerated Hartree-Fock energy calculation for H2O
# using the LibAccInt Python bindings. Features:
#   - Fused S+T+V computation via CudaEngine
#   - Device-side ERI accumulation via GpuFockBuilder
#   - Graceful fallback to CPU when CUDA is not available
#   - Timing comparison between CPU and GPU paths
#
# Usage:
#   PYTHONPATH=build/cuda-release/python python examples/python/gpu_accelerated_hf.py

import time

import numpy as np

import libaccint as lai


def compute_electronic_energy(D, H_core, F):
    """E_elec = 0.5 * Tr[D * (H + F)]"""
    return 0.5 * np.sum(D * (H_core + F))


def compute_nuclear_repulsion(atoms):
    """Compute nuclear repulsion energy V_nn = sum Z_A*Z_B / R_AB."""
    V_nn = 0.0
    for i in range(len(atoms)):
        for j in range(i + 1, len(atoms)):
            ai, aj = atoms[i], atoms[j]
            pi, pj = ai.position, aj.position
            dx = pi.x - pj.x
            dy = pi.y - pj.y
            dz = pi.z - pj.z
            r = np.sqrt(dx**2 + dy**2 + dz**2)
            V_nn += ai.atomic_number * aj.atomic_number / r
    return V_nn


def main():
    print("=== LibAccInt GPU-Accelerated HF (Python) ===")
    print(f"LibAccInt version: {lai.version()}")
    print(f"CUDA backend: {'compiled' if lai.has_cuda_backend() else 'not compiled'}")
    print(f"OpenMP: {'available' if lai.has_openmp() else 'not available'}")
    print()

    # ── Step 1: Define H2O molecule ──────────────────────────────────────
    atoms = [
        lai.Atom(8, [0.000000,  0.000000,  0.117176]),  # O
        lai.Atom(1, [0.000000,  1.430665, -0.468706]),  # H
        lai.Atom(1, [0.000000, -1.430665, -0.468706]),  # H
    ]

    # ── Step 2: Create basis set ─────────────────────────────────────────
    basis = lai.create_builtin_basis("sto-3g", atoms)
    nbf = basis.n_basis_functions()

    print(f"Molecule: H2O")
    print(f"Basis set: STO-3G")
    print(f"Basis functions: {nbf}")
    print(f"Shells: {basis.n_shells()}")
    print()

    # Simple density matrix (diagonal approximation for demo)
    D = np.zeros((nbf, nbf))
    for i in range(min(5, nbf)):  # 5 occupied MOs for H2O/STO-3G
        D[i, i] = 1.0

    V_nn = compute_nuclear_repulsion(atoms)

    # ── Step 3: CPU path ─────────────────────────────────────────────────
    print("--- CPU Path ---")
    cpu_start = time.perf_counter()

    engine = lai.Engine(basis)

    S_cpu = engine.compute_overlap_matrix(lai.BackendHint.ForceCPU)
    T_cpu = engine.compute_kinetic_matrix(lai.BackendHint.ForceCPU)
    V_cpu = engine.compute_nuclear_matrix(atoms, lai.BackendHint.ForceCPU)
    H_core_cpu = T_cpu + V_cpu

    fock_cpu = lai.FockBuilder(nbf)
    fock_cpu.set_density(D)
    engine.compute_and_consume(
        lai.Operator.coulomb(), fock_cpu, lai.BackendHint.ForceCPU
    )

    F_cpu = fock_cpu.get_fock_matrix(H_core_cpu, 0.5)
    E_elec_cpu = compute_electronic_energy(D, H_core_cpu, F_cpu)
    E_total_cpu = E_elec_cpu + V_nn

    cpu_time = (time.perf_counter() - cpu_start) * 1000

    print(f"Electronic energy: {E_elec_cpu:.10f} Hartree")
    print(f"Nuclear repulsion: {V_nn:.10f} Hartree")
    print(f"Total energy:      {E_total_cpu:.10f} Hartree")
    print(f"CPU time:          {cpu_time:.2f} ms")
    print()

    # ── Step 4: GPU path ─────────────────────────────────────────────────
    if engine.gpu_available() and lai.CudaEngine is not None:
        print("--- GPU Path ---")
        gpu_start = time.perf_counter()

        cuda_eng = engine.cuda_engine()

        # Fused S+T+V in a single GPU kernel
        S_gpu, T_gpu, V_gpu = cuda_eng.compute_all_1e_fused(atoms)
        H_core_gpu = T_gpu + V_gpu

        # GPU Fock builder for device-side ERI accumulation
        fock_gpu = lai.GpuFockBuilder(nbf)
        fock_gpu.set_density(D)

        engine.compute_and_consume(
            lai.Operator.coulomb(), fock_gpu, lai.BackendHint.ForceGPU
        )
        fock_gpu.synchronize()

        F_gpu = fock_gpu.get_fock_matrix(H_core_gpu, 0.5)
        E_elec_gpu = compute_electronic_energy(D, H_core_gpu, F_gpu)
        E_total_gpu = E_elec_gpu + V_nn

        gpu_time = (time.perf_counter() - gpu_start) * 1000

        print(f"Electronic energy: {E_elec_gpu:.10f} Hartree")
        print(f"Nuclear repulsion: {V_nn:.10f} Hartree")
        print(f"Total energy:      {E_total_gpu:.10f} Hartree")
        print(f"GPU time:          {gpu_time:.2f} ms")
        print()

        # Compare
        energy_diff = abs(E_total_gpu - E_total_cpu)
        print("--- Comparison ---")
        print(f"Energy difference:  {energy_diff:.2e} Hartree")
        print(f"Speedup:            {cpu_time / gpu_time:.2f}x")

        if energy_diff < 1e-10:
            print("Results agree within machine precision.")
    else:
        print("--- GPU Path ---")
        if not lai.has_cuda_backend():
            print("CUDA not compiled. Build with cmake --preset cuda-release")
            print("and set PYTHONPATH to the CUDA build's python directory.")
        else:
            print("GPU not available - CUDA compiled but no device found.")

    print()
    print("Note: This uses a non-SCF density. A real HF code would")
    print("iterate to self-consistency.")


if __name__ == "__main__":
    main()
