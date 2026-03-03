#!/usr/bin/env python3
# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Bryce M. Westheimer. All rights reserved.
#
# basic_hf.py
#
# Demonstrates basic Hartree–Fock energy calculation using LibAccInt's
# Python bindings. This example computes the HF energy for H₂ using
# the STO-3G basis set.
#
# Requirements:
#   pip install libaccint numpy scipy
#
# If running from the build directory:
#   PYTHONPATH=build/cpu-release/python python examples/python/basic_hf.py

"""
Basic HF Energy Calculation with LibAccInt Python Bindings
==========================================================

This example demonstrates:
  1. Creating atoms and basis sets
  2. Computing one-electron integrals (S, T, V)
  3. Computing two-electron integrals via FockBuilder
  4. A simple SCF energy calculation

Note: This requires the LibAccInt Python bindings to be installed.
If they are not available, the script will print a message and exit.
"""

import sys
import math

try:
    import numpy as np
except ImportError:
    print("NumPy is required: pip install numpy")
    sys.exit(1)

try:
    import libaccint as lai
except ImportError:
    print("LibAccInt Python bindings not found.")
    print("Build with -DLIBACCINT_BUILD_PYTHON=ON or install via pip.")
    print("\nThis example shows what the code would look like:\n")
    print("""
    import libaccint as lai
    import numpy as np

    # Define H2 molecule
    atoms = [
        lai.Atom(1, [0.0, 0.0, 0.0]),       # H
        lai.Atom(1, [0.0, 0.0, 1.4]),        # H (1.4 bohr apart)
    ]

    # Create basis set
    basis = lai.create_builtin_basis("STO-3G", atoms)
    nbf = basis.n_basis_functions()

    # Create engine
    engine = lai.Engine(basis)

    # Compute one-electron matrices
    S = engine.compute_overlap_matrix()     # Overlap
    T = engine.compute_kinetic_matrix()     # Kinetic
    V = engine.compute_nuclear_matrix(atoms)  # Nuclear attraction
    H_core = T + V                          # Core Hamiltonian

    # Set up density matrix (initial guess: identity/nbf)
    D = np.eye(nbf) / nbf

    # Compute two-electron contributions
    fock = lai.FockBuilder(nbf)
    fock.set_density(D)
    engine.compute_and_consume(lai.Operator.coulomb(), fock)

    J = fock.get_coulomb_matrix()
    K = fock.get_exchange_matrix()
    G = J - 0.5 * K                         # Two-electron part

    F = H_core + G                           # Fock matrix

    # Electronic energy: E = 0.5 * Tr[D * (H + F)]
    E_elec = 0.5 * np.trace(D @ (H_core + F))

    # Nuclear repulsion
    V_nn = 1.0 / 1.4  # Z_A * Z_B / R_AB for H2

    print(f"Electronic energy: {E_elec:.10f} Hartree")
    print(f"Nuclear repulsion: {V_nn:.10f} Hartree")
    print(f"Total energy:      {E_elec + V_nn:.10f} Hartree")
    """)
    sys.exit(0)

# If libaccint is available, run the actual calculation
def main():
    print("=== LibAccInt Python HF Example ===")
    print(f"LibAccInt version: {lai.__version__}\n")

    # Define H2 molecule (positions in Bohr)
    bond_length = 1.4
    atoms = [
        lai.Atom(1, [0.0, 0.0, 0.0]),
        lai.Atom(1, [0.0, 0.0, bond_length]),
    ]

    # Create STO-3G basis set
    basis = lai.create_builtin_basis("STO-3G", atoms)
    nbf = basis.n_basis_functions()
    print(f"Molecule: H2 (R = {bond_length} bohr)")
    print(f"Basis: STO-3G ({nbf} functions)\n")

    # Create computation engine
    engine = lai.Engine(basis)

    # Compute one-electron integrals
    S = engine.compute_overlap_matrix()
    T = engine.compute_kinetic_matrix()
    V = engine.compute_nuclear_matrix(atoms)

    H_core = T + V

    print("Overlap matrix S:")
    print(S)
    print(f"\nCore Hamiltonian H_core:")
    print(H_core)

    # Initial density: unit density for demo
    D = np.eye(nbf) * 0.5

    # Two-electron integrals via FockBuilder
    fock = lai.FockBuilder(nbf)
    fock.set_density(D)
    engine.compute_and_consume(lai.Operator.coulomb(), fock)

    J = fock.get_coulomb_matrix()
    K = fock.get_exchange_matrix()
    F = H_core + J - 0.5 * K

    # Electronic energy
    E_elec = 0.5 * np.trace(D @ (H_core + F))

    # Nuclear repulsion
    V_nn = 1.0 / bond_length

    print(f"\nElectronic energy: {E_elec:.10f} Hartree")
    print(f"Nuclear repulsion: {V_nn:.10f} Hartree")
    print(f"Total energy:      {E_elec + V_nn:.10f} Hartree")


if __name__ == "__main__":
    main()
