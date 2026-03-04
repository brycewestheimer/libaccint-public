.. _tutorial-python:

Tutorial 6: Python Bindings
===========================

This tutorial covers using LibAccInt from Python with NumPy integration.

Installation
------------

Build and install Python bindings:

.. code-block:: bash

   # Configure with Python bindings enabled
   cmake --preset cpu-release -DLIBACCINT_BUILD_PYTHON=ON
   cmake --build --preset cpu-release

   # Install in editable mode
   cd python
   pip install -e .

   # Verify installation
   python -c "import libaccint; print(libaccint.__version__)"

Basic Usage
-----------

The Python API mirrors the C++ API but with Pythonic conventions:

.. code-block:: python

   import libaccint
   import numpy as np

   # Define atoms (atomic_number, [x, y, z] in Bohr)
   atoms = [
       libaccint.Atom(8, [0.0, 0.0, 0.0]),           # Oxygen
       libaccint.Atom(1, [0.0, 1.43233673, -1.10866041]),   # Hydrogen
       libaccint.Atom(1, [0.0, -1.43233673, -1.10866041]),  # Hydrogen
   ]

   # Create basis set (convenience function)
   basis = libaccint.basis_set("sto-3g", atoms)

   print(f"Shells: {basis.n_shells()}")
   print(f"Basis functions: {basis.n_basis_functions()}")

   # Create engine
   engine = libaccint.Engine(basis)
   print(f"GPU available: {engine.gpu_available()}")

One-Electron Integrals
----------------------

Compute overlap, kinetic, and nuclear attraction matrices:

.. code-block:: python

   import libaccint

   atoms = [
       libaccint.Atom(8, [0.0, 0.0, 0.0]),
       libaccint.Atom(1, [0.0, 1.43, -1.11]),
       libaccint.Atom(1, [0.0, -1.43, -1.11]),
   ]

   basis = libaccint.basis_set("sto-3g", atoms)
   engine = libaccint.Engine(basis)

   # Convenience functions return NumPy arrays
   S = engine.compute_overlap_matrix()
   T = engine.compute_kinetic_matrix()
   V = engine.compute_nuclear_matrix(atoms)

   print(f"Overlap matrix shape: {S.shape}")
   print(f"Overlap diagonal: {np.diag(S)}")

   # Core Hamiltonian
   H_core = T + V

Convenience API
~~~~~~~~~~~~~~~

Use module-level functions for quick computations:

.. code-block:: python

   # One-liner API
   S = libaccint.compute_overlap(basis)
   T = libaccint.compute_kinetic(basis)
   V = libaccint.compute_nuclear(basis, atoms)
   H = libaccint.compute_core_hamiltonian(basis, atoms)

Two-Electron Integrals
----------------------

For Fock matrix construction:

.. code-block:: python

   import numpy as np

   nbf = basis.n_basis_functions()

   # Create a density matrix (e.g., from initial guess)
   D = np.eye(nbf) * 0.1

   # Build Fock matrix
   J, K = libaccint.build_fock(engine, D)

   # Or use FockBuilder directly
   fock = libaccint.FockBuilder(basis)
   fock.set_density(D)
   engine.compute_fock(fock)
   J = fock.get_coulomb()
   K = fock.get_exchange()

   # G = 2J - K (two-electron contribution)
   G = 2 * J - K

Backend Control
---------------

Control CPU/GPU dispatch from Python:

.. code-block:: python

   # Check backend availability
   print(f"CUDA available: {libaccint.is_backend_available(libaccint.BackendType.CUDA)}")

   # Force CPU
   S = engine.compute_overlap_matrix(hint=libaccint.BackendHint.ForceCPU)

   # Prefer GPU
   S = engine.compute_overlap_matrix(hint=libaccint.BackendHint.PreferGPU)

NumPy Integration
-----------------

All matrices are returned as NumPy arrays with proper shape:

.. code-block:: python

   import numpy as np

   S = engine.compute_overlap_matrix()

   # Use with NumPy/SciPy
   eigenvalues, eigenvectors = np.linalg.eigh(S)

   # Use with linear algebra
   from scipy.linalg import sqrtm, inv
   S_sqrt_inv = inv(sqrtm(S))

   # Integration with other QC libraries
   # (convert to appropriate format)

Complete RHF Example
--------------------

A minimal Restricted Hartree-Fock implementation:

.. code-block:: python

   import numpy as np
   from scipy.linalg import eigh
   import libaccint

   def run_rhf(atoms, basis_name="sto-3g", max_iter=50, tol=1e-8):
       """Run restricted Hartree-Fock calculation."""

       # Setup
       basis = libaccint.basis_set(basis_name, atoms)
       engine = libaccint.Engine(basis)
       nbf = basis.n_basis_functions()

       # Count electrons (sum of atomic numbers for neutral molecule)
       n_electrons = sum(atom.atomic_number for atom in atoms)
       n_occ = n_electrons // 2

       # One-electron integrals
       S = engine.compute_overlap_matrix()
       T = engine.compute_kinetic_matrix()
       V = engine.compute_nuclear_matrix(atoms)
       H_core = T + V

       # Symmetric orthogonalization: S^{-1/2}
       eigvals, eigvecs = eigh(S)
       X = eigvecs @ np.diag(1.0 / np.sqrt(eigvals)) @ eigvecs.T

       # Initial guess: diagonalize H_core
       Fp = X.T @ H_core @ X
       eps, Cp = eigh(Fp)
       C = X @ Cp
       D = 2.0 * C[:, :n_occ] @ C[:, :n_occ].T

       # SCF loop
       E_old = 0.0
       converged = False

       print(f"{'Iter':>4} {'Energy':>18} {'Delta E':>14}")
       print("-" * 40)

       for iteration in range(max_iter):
           # Build Fock matrix
           J, K = libaccint.build_fock(engine, D)
           G = 2 * J - K
           F = H_core + G

           # Calculate energy
           E_elec = 0.5 * np.sum(D * (H_core + F))
           E_nuc = compute_nuclear_repulsion(atoms)
           E_total = E_elec + E_nuc

           delta_E = E_total - E_old
           print(f"{iteration:4d} {E_total:18.10f} {delta_E:14.2e}")

           if abs(delta_E) < tol:
               converged = True
               break

           E_old = E_total

           # Diagonalize Fock matrix
           Fp = X.T @ F @ X
           eps, Cp = eigh(Fp)
           C = X @ Cp
           D = 2.0 * C[:, :n_occ] @ C[:, :n_occ].T

       if converged:
           print(f"\nConverged in {iteration + 1} iterations")
           print(f"Final energy: {E_total:.10f} Hartree")
       else:
           print("\nDid not converge!")

       return E_total, C, eps

   def compute_nuclear_repulsion(atoms):
       """Calculate nuclear repulsion energy."""
       E = 0.0
       for i, a in enumerate(atoms):
           for j, b in enumerate(atoms):
               if i < j:
                   dx = a.position[0] - b.position[0]
                   dy = a.position[1] - b.position[1]
                   dz = a.position[2] - b.position[2]
                   r = np.sqrt(dx*dx + dy*dy + dz*dz)
                   E += a.atomic_number * b.atomic_number / r
       return E

   # Run calculation
   if __name__ == "__main__":
       atoms = [
           libaccint.Atom(8, [0.0, 0.0, 0.0]),
           libaccint.Atom(1, [0.0, 1.43233673, -1.10866041]),
           libaccint.Atom(1, [0.0, -1.43233673, -1.10866041]),
       ]

       E, C, eps = run_rhf(atoms)

Error Handling
--------------

Python exceptions are raised for errors:

.. code-block:: python

   import libaccint

   try:
       # Invalid basis set name
       basis = libaccint.basis_set("invalid-basis", atoms)
   except libaccint.LibAccIntError as e:
       print(f"LibAccInt error: {e}")

   try:
       # GPU not available but forced
       S = engine.compute_overlap_matrix(hint=libaccint.BackendHint.ForceGPU)
   except libaccint.BackendError as e:
       print(f"Backend error: {e}")

API Reference
-------------

See :doc:`/api/python/index` for complete Python API documentation.

Next Steps
----------

- :doc:`/api/python/index` - Complete Python API reference
- :doc:`../cookbook` - Python recipes and patterns
