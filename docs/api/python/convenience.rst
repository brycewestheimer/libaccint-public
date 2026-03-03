.. _api-python-convenience:

Convenience Functions
=====================

.. module:: libaccint
   :no-index:

These functions provide a simplified API for common operations.

basis_set
---------

.. autofunction:: basis_set

   Create a basis set by name.

   **Parameters:**

   - ``name`` (str): Basis set name ("sto-3g", "cc-pvdz", etc.)
   - ``atoms`` (list[Atom]): List of atoms

   **Returns:** BasisSet

   **Example:**

   .. code-block:: python

      atoms = [
          libaccint.Atom(8, [0.0, 0.0, 0.0]),
          libaccint.Atom(1, [0.0, 1.43, -1.11]),
          libaccint.Atom(1, [0.0, -1.43, -1.11]),
      ]
      basis = libaccint.basis_set("sto-3g", atoms)

compute_overlap
---------------

.. autofunction:: compute_overlap

   Compute the overlap matrix.

   **Parameters:**

   - ``basis`` (BasisSet): The basis set

   **Returns:** numpy.ndarray of shape (nbf, nbf)

   **Example:**

   .. code-block:: python

      S = libaccint.compute_overlap(basis)
      print(f"Overlap matrix: {S.shape}")

compute_kinetic
---------------

.. autofunction:: compute_kinetic

   Compute the kinetic energy matrix.

   **Parameters:**

   - ``basis`` (BasisSet): The basis set

   **Returns:** numpy.ndarray of shape (nbf, nbf)

   **Example:**

   .. code-block:: python

      T = libaccint.compute_kinetic(basis)

compute_nuclear
---------------

.. autofunction:: compute_nuclear

   Compute the nuclear attraction matrix.

   **Parameters:**

   - ``basis`` (BasisSet): The basis set
   - ``atoms`` (list[Atom]): List of atoms (for charges and positions)

   **Returns:** numpy.ndarray of shape (nbf, nbf)

   **Example:**

   .. code-block:: python

      V = libaccint.compute_nuclear(basis, atoms)

compute_core_hamiltonian
------------------------

.. autofunction:: compute_core_hamiltonian

   Compute the core Hamiltonian (T + V).

   **Parameters:**

   - ``basis`` (BasisSet): The basis set
   - ``atoms`` (list[Atom]): List of atoms

   **Returns:** numpy.ndarray of shape (nbf, nbf)

   **Example:**

   .. code-block:: python

      H = libaccint.compute_core_hamiltonian(basis, atoms)
      # Equivalent to:
      # H = libaccint.compute_kinetic(basis) + libaccint.compute_nuclear(basis, atoms)

build_fock
----------

.. autofunction:: build_fock

   Build Coulomb and exchange matrices from a density matrix.

   **Parameters:**

   - ``engine`` (Engine): The integral engine
   - ``density`` (numpy.ndarray): Density matrix of shape (nbf, nbf)

   **Returns:** tuple[numpy.ndarray, numpy.ndarray] - (J, K) matrices

   **Example:**

   .. code-block:: python

      J, K = libaccint.build_fock(engine, D)
      G = 2 * J - K  # Two-electron contribution to Fock matrix
      F = H + G       # Full Fock matrix

compute_eri_tensor
------------------

.. autofunction:: compute_eri_tensor

   Compute the full 4-index electron repulsion integral tensor.

   **Parameters:**

   - ``engine`` (Engine): The integral engine
   - ``op`` (Operator, optional): Two-electron operator (default: Coulomb)

   **Returns:** numpy.ndarray of shape (nbf, nbf, nbf, nbf)

   **Example:**

   .. code-block:: python

      eri = libaccint.compute_eri_tensor(engine)
      print(f"ERI tensor shape: {eri.shape}")  # (nbf, nbf, nbf, nbf)

      # Verify 8-fold symmetry
      assert np.allclose(eri, eri.transpose(1, 0, 2, 3))  # (ij|kl) = (ji|kl)
      assert np.allclose(eri, eri.transpose(0, 1, 3, 2))  # (ij|kl) = (ij|lk)

compute_eri_block
-----------------

.. autofunction:: compute_eri_block

   Compute ERIs for a single shell quartet with local basis-function indexing.

   **Parameters:**

   - ``engine`` (Engine): The integral engine
   - ``quartet`` (ShellSetQuartet): The shell set quartet to compute
   - ``op`` (Operator, optional): Two-electron operator (default: Coulomb)

   **Returns:** numpy.ndarray — flat array of integral values for the quartet

   **Example:**

   .. code-block:: python

      quartets = basis.shell_set_quartets()
      block = libaccint.compute_eri_block(engine, quartets[0])
      print(f"Block size: {block.size}")

Complete Example
----------------

.. code-block:: python

   import numpy as np
   import libaccint

   # Define molecule
   atoms = [
       libaccint.Atom(8, [0.0, 0.0, 0.0]),
       libaccint.Atom(1, [0.0, 1.43, -1.11]),
       libaccint.Atom(1, [0.0, -1.43, -1.11]),
   ]

   # Create basis set
   basis = libaccint.basis_set("sto-3g", atoms)
   engine = libaccint.Engine(basis)

   # Compute one-electron integrals
   S = libaccint.compute_overlap(basis)
   H = libaccint.compute_core_hamiltonian(basis, atoms)

   # Initial density (zeros or from core Hamiltonian)
   nbf = basis.n_basis_functions()
   D = np.zeros((nbf, nbf))

   # Build Fock matrix
   J, K = libaccint.build_fock(engine, D)
   F = H + 2*J - K

   print(f"Basis functions: {nbf}")
   print(f"Fock matrix trace: {np.trace(F):.6f}")
