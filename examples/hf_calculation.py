#!/usr/bin/env python3
# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Bryce M. Westheimer. All rights reserved.
#
# hf_calculation.py
#
# Complete Restricted Hartree-Fock (RHF) SCF calculation for H2O using
# the cc-pVDZ basis set, powered by LibAccInt Python bindings.
#
# This example demonstrates:
#   1. Molecule construction with atomic coordinates in Bohr
#   2. Basis set lookup by name string
#   3. Core Hamiltonian assembly (T + V)
#   4. Canonical orthogonalization (Lowdin S^{-1/2})
#   5. Full iterative SCF cycle with DIIS acceleration
#   6. Eigensolve and density construction at each iteration
#   7. Convergence monitoring with energy and density criteria
#   8. Final orbital energies and total energy printout
#
# Requirements: numpy, scipy, libaccint (built with Python bindings)
#
# Run: python examples/hf_calculation.py

import numpy as np
from scipy.linalg import eigh

import libaccint


# =============================================================================
# Helper functions
# =============================================================================

def compute_nuclear_repulsion(atoms):
    """Compute classical nuclear repulsion energy: sum_{A>B} Z_A * Z_B / R_AB."""
    e_nuc = 0.0
    for i in range(len(atoms)):
        for j in range(i + 1, len(atoms)):
            pi = atoms[i].position
            pj = atoms[j].position
            dx = pi[0] - pj[0]
            dy = pi[1] - pj[1]
            dz = pi[2] - pj[2]
            r = np.sqrt(dx * dx + dy * dy + dz * dz)
            e_nuc += atoms[i].atomic_number * atoms[j].atomic_number / r
    return e_nuc


def canonical_orthogonalization(S, threshold=1e-10):
    """
    Compute the canonical orthogonalization matrix X = S^{-1/2}.

    Uses eigendecomposition of S to form X = U * s^{-1/2} * U^T,
    discarding near-zero eigenvalues to handle linear dependence.
    """
    s_vals, U = eigh(S)
    # Discard eigenvalues below threshold (linear dependence)
    s_inv_sqrt = np.where(s_vals > threshold,
                          1.0 / np.sqrt(s_vals),
                          0.0)
    X = U @ np.diag(s_inv_sqrt) @ U.T
    return X


class DIIS:
    """
    DIIS (Direct Inversion in the Iterative Subspace) accelerator.

    Maintains a history of Fock matrices and commutator error vectors,
    then extrapolates an improved Fock matrix via least-squares.
    """

    def __init__(self, max_size=6):
        self.max_size = max_size
        self.fock_history = []
        self.error_history = []

    def add(self, F, error):
        """Add a Fock matrix and its error vector to the history."""
        self.fock_history.append(F.copy())
        self.error_history.append(error.copy())
        if len(self.fock_history) > self.max_size:
            self.fock_history.pop(0)
            self.error_history.pop(0)

    def extrapolate(self):
        """Extrapolate an improved Fock matrix from the history."""
        n = len(self.fock_history)
        if n < 2:
            return self.fock_history[-1]

        # Build B matrix: B_ij = <e_i | e_j>
        B = np.zeros((n + 1, n + 1))
        for i in range(n):
            for j in range(n):
                B[i, j] = np.sum(self.error_history[i] * self.error_history[j])
        # Lagrange multiplier constraint: sum(c_i) = 1
        B[n, :n] = -1.0
        B[:n, n] = -1.0

        rhs = np.zeros(n + 1)
        rhs[n] = -1.0

        c = np.linalg.solve(B, rhs)

        # Extrapolated Fock matrix: F_diis = sum(c_i * F_i)
        F_new = sum(c[i] * self.fock_history[i] for i in range(n))
        return F_new


# =============================================================================
# Main RHF SCF calculation
# =============================================================================

def main():
    print("=== LibAccInt: Complete RHF Calculation (Python) ===")
    print(f"LibAccInt version: {libaccint.version()}\n")

    # -- Step 1: Define the molecular geometry ---------------------------------
    # H2O in Bohr (atomic units). This geometry gives a bond angle of ~104.5 deg.
    atoms = [
        libaccint.Atom(8, [0.000000,  0.000000,  0.117176]),   # O
        libaccint.Atom(1, [0.000000,  1.430665, -0.468706]),   # H
        libaccint.Atom(1, [0.000000, -1.430665, -0.468706]),   # H
    ]
    n_electrons = 10
    n_occ = n_electrons // 2  # 5 doubly-occupied MOs

    # -- Step 2: Load basis set ------------------------------------------------
    # basis_set() convenience currently resolves built-in sets only.
    basis = libaccint.basis_set("sto-3g", atoms)
    nbf = basis.n_basis_functions()

    # ALTERNATIVE: Use a built-in basis directly
    #   basis = libaccint.create_builtin_basis("sto-3g", atoms)

    print(f"Molecule: H2O")
    print(f"Basis set: STO-3G")
    print(f"Basis functions: {nbf}")
    print(f"Shells: {basis.n_shells()}")
    print(f"Max angular momentum: {basis.max_angular_momentum()}")
    print(f"Electrons: {n_electrons} ({n_occ} doubly-occupied MOs)\n")

    # -- Step 3: Create the computation engine ---------------------------------
    engine = libaccint.Engine(basis)

    # ALTERNATIVE: Engine with custom dispatch config
    #   config = libaccint.DispatchConfig()
    #   config.min_gpu_batch_size = 32
    #   engine = libaccint.Engine(basis, config)

    # -- Step 4: Compute one-electron integrals --------------------------------
    # The Python bindings return NumPy arrays directly.
    S = engine.compute_overlap_matrix()
    T = engine.compute_kinetic_matrix()
    V = engine.compute_nuclear_matrix(atoms)

    # ALTERNATIVE (one-shot core Hamiltonian):
    #   H_core = engine.compute_core_hamiltonian(atoms)
    #
    # ALTERNATIVE (convenience functions that create an Engine internally):
    #   S = libaccint.compute_overlap(basis)
    #   T = libaccint.compute_kinetic(basis)
    #   V = libaccint.compute_nuclear(basis, atoms)
    #   H_core = libaccint.compute_core_hamiltonian(basis, atoms)

    # -- Step 5: Build core Hamiltonian ----------------------------------------
    H_core = T + V

    print("One-electron integrals computed.")
    print(f"  Tr(S) = {np.trace(S):.6f}")
    print(f"  Tr(T) = {np.trace(T):.6f}")
    print(f"  Tr(V) = {np.trace(V):.6f}\n")

    # -- Step 6: Canonical orthogonalization -----------------------------------
    X = canonical_orthogonalization(S)

    # -- Step 7: Initial guess — diagonalize H_core in orthogonal basis --------
    H_prime = X.T @ H_core @ X
    eps0, C_prime = eigh(H_prime)
    C = X @ C_prime
    D = 2.0 * C[:, :n_occ] @ C[:, :n_occ].T

    # -- Step 8: SCF iteration -------------------------------------------------
    max_iterations = 100
    energy_threshold = 1e-10
    density_threshold = 1e-8

    E_nuc = compute_nuclear_repulsion(atoms)
    E_old = 0.0
    diis = DIIS(max_size=6)
    orbital_energies = eps0

    print(f"Starting SCF iteration (max {max_iterations} cycles)...")
    print("-" * 72)
    print(f"{'Iter':>5s}{'E_total (Hartree)':>22s}{'Delta_E':>18s}"
          f"{'Max|Delta_D|':>18s}{'Status':>8s}")
    print("-" * 72)

    for iteration in range(1, max_iterations + 1):
        # (a) Build Fock matrix using compute-and-consume pattern
        fock_builder = libaccint.FockBuilder(nbf)
        fock_builder.set_density(np.ascontiguousarray(D, dtype=np.float64))

        engine.compute_and_consume(libaccint.Operator.coulomb(), fock_builder)

        # ALTERNATIVE (parallel):
        #   engine.compute_and_consume_parallel(
        #       libaccint.Operator.coulomb(), fock_builder, n_threads=4)
        #
        # ALTERNATIVE (high-level convenience):
        #   F = libaccint.build_fock(engine, D, H_core, exchange_fraction=0.5)

        # (b) Extract J and K, form Fock matrix
        J = fock_builder.get_coulomb_matrix()
        K = fock_builder.get_exchange_matrix()

        # F = H_core + J - 0.5*K
        # The 0.5 factor is because D includes a factor of 2 from double
        # occupancy (Szabo & Ostlund convention).
        F = H_core + J - 0.5 * K

        # (c) Compute electronic energy: E_elec = 0.5 * Tr[D * (H_core + F)]
        E_elec = 0.5 * np.sum(D * (H_core + F))
        E_total = E_elec + E_nuc

        # (d) DIIS: compute commutator error e = FDS - SDF, then extrapolate
        error = F @ D @ S - S @ D @ F
        diis.add(F, error)
        F_diis = diis.extrapolate()

        # (e) Convergence check
        delta_E = abs(E_total - E_old)

        # (f) Diagonalize the DIIS-extrapolated Fock matrix
        F_prime = X.T @ F_diis @ X
        eps, C_prime = eigh(F_prime)
        C = X @ C_prime
        D_new = 2.0 * C[:, :n_occ] @ C[:, :n_occ].T

        max_delta_D = np.max(np.abs(D_new - D))

        # Print iteration info
        status = ""
        if delta_E < energy_threshold and max_delta_D < density_threshold and iteration > 1:
            status = "CONV"
        print(f"{iteration:5d}{E_total:22.12f}{delta_E:18.4e}{max_delta_D:18.4e}"
              f"{status:>8s}")

        if status == "CONV":
            D = D_new
            orbital_energies = eps

            # Recompute final energy with converged density
            fock_final = libaccint.FockBuilder(nbf)
            fock_final.set_density(np.ascontiguousarray(D, dtype=np.float64))
            engine.compute_and_consume(libaccint.Operator.coulomb(), fock_final)

            J_f = fock_final.get_coulomb_matrix()
            K_f = fock_final.get_exchange_matrix()
            F_final = H_core + J_f - 0.5 * K_f
            E_elec = 0.5 * np.sum(D * (H_core + F_final))
            E_total = E_elec + E_nuc

            # -- Step 9: Print final results -----------------------------------
            print("-" * 72)
            print(f"\nSCF converged in {iteration} iterations.\n")

            print(f"Electronic energy:     {E_elec:20.12f} Hartree")
            print(f"Nuclear repulsion:     {E_nuc:20.12f} Hartree")
            print(f"Total RHF energy:      {E_total:20.12f} Hartree\n")

            print("Orbital energies (Hartree):")
            for i in range(nbf):
                label = "(occupied)" if i < n_occ else "(virtual)"
                print(f"  {i+1:3d}: {orbital_energies[i]:14.6f}  {label}")
            print()
            return

        D = D_new
        E_old = E_total
        orbital_energies = eps

    # If we get here, SCF did not converge
    print("-" * 72)
    print(f"WARNING: SCF did not converge within {max_iterations} iterations.")
    print(f"Last energy: {E_old:.12f} Hartree")


if __name__ == "__main__":
    main()
