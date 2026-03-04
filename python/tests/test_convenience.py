# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Bryce M. Westheimer. All rights reserved.

"""
Tests for convenience API functions.
"""

import pytest
import numpy as np


def make_water_atoms():
    """Create water molecule atoms."""
    from libaccint import Atom
    return [
        Atom(8, [0.0, 0.0, 0.0]),
        Atom(1, [0.0, 1.43, -1.11]),
        Atom(1, [0.0, -1.43, -1.11]),
    ]


class TestBasisSetFactory:
    """Tests for basis_set() convenience factory."""

    def test_create_basis(self):
        from libaccint import basis_set
        atoms = make_water_atoms()
        basis = basis_set("sto-3g", atoms)
        assert basis.n_basis_functions() == 7


class TestComputeOverlap:
    """Tests for compute_overlap() convenience function."""

    def test_with_basis(self):
        from libaccint import basis_set, compute_overlap
        atoms = make_water_atoms()
        basis = basis_set("sto-3g", atoms)
        S = compute_overlap(basis)
        assert S.shape == (7, 7)
        np.testing.assert_array_almost_equal(S, S.T)

    def test_with_engine(self):
        from libaccint import basis_set, compute_overlap, Engine
        atoms = make_water_atoms()
        basis = basis_set("sto-3g", atoms)
        engine = Engine(basis)
        S = compute_overlap(engine)
        assert S.shape == (7, 7)


class TestComputeKinetic:
    """Tests for compute_kinetic() convenience function."""

    def test_with_basis(self):
        from libaccint import basis_set, compute_kinetic
        atoms = make_water_atoms()
        basis = basis_set("sto-3g", atoms)
        T = compute_kinetic(basis)
        assert T.shape == (7, 7)
        np.testing.assert_array_almost_equal(T, T.T)


class TestComputeNuclear:
    """Tests for compute_nuclear() convenience function."""

    def test_with_basis(self):
        from libaccint import basis_set, compute_nuclear
        atoms = make_water_atoms()
        basis = basis_set("sto-3g", atoms)
        V = compute_nuclear(basis, atoms)
        assert V.shape == (7, 7)
        np.testing.assert_array_almost_equal(V, V.T)


class TestComputeCoreHamiltonian:
    """Tests for compute_core_hamiltonian() convenience function."""

    def test_with_basis(self):
        from libaccint import basis_set, compute_core_hamiltonian
        from libaccint import compute_kinetic, compute_nuclear
        atoms = make_water_atoms()
        basis = basis_set("sto-3g", atoms)

        H = compute_core_hamiltonian(basis, atoms)
        T = compute_kinetic(basis)
        V = compute_nuclear(basis, atoms)

        np.testing.assert_array_almost_equal(H, T + V, decimal=12)


class TestBuildFock:
    """Tests for build_fock() convenience function."""

    def test_basic_fock_build(self):
        from libaccint import basis_set, compute_core_hamiltonian, build_fock, Engine
        atoms = make_water_atoms()
        basis = basis_set("sto-3g", atoms)
        engine = Engine(basis)
        nbf = basis.n_basis_functions()

        H = compute_core_hamiltonian(engine, atoms)
        D = np.eye(nbf) * 0.1

        F = build_fock(engine, D, H)
        assert F.shape == (nbf, nbf)
        np.testing.assert_array_almost_equal(F, F.T, decimal=12)

    def test_fock_without_hcore(self):
        from libaccint import basis_set, build_fock, Engine
        atoms = make_water_atoms()
        basis = basis_set("sto-3g", atoms)
        engine = Engine(basis)
        nbf = basis.n_basis_functions()

        D = np.eye(nbf) * 0.1

        # F = J - K when H_core is None
        F = build_fock(engine, D, None)
        assert F.shape == (nbf, nbf)


class TestAtomsFromXyz:
    """Tests for atoms_from_xyz() convenience function."""

    def test_create_atoms(self):
        from libaccint._core import atoms_from_xyz
        atoms = atoms_from_xyz([
            (8, 0.0, 0.0, 0.0),
            (1, 0.0, 1.43, -1.11),
            (1, 0.0, -1.43, -1.11),
        ])
        assert len(atoms) == 3
        assert atoms[0].atomic_number == 8
        assert atoms[1].atomic_number == 1


class TestQuickFunctions:
    """Tests for quick_* convenience functions."""

    def test_quick_overlap(self):
        from libaccint import basis_set
        from libaccint._core import quick_overlap
        atoms = make_water_atoms()
        basis = basis_set("sto-3g", atoms)
        S = quick_overlap(basis)
        assert S.shape == (7, 7)

    def test_quick_kinetic(self):
        from libaccint import basis_set
        from libaccint._core import quick_kinetic
        atoms = make_water_atoms()
        basis = basis_set("sto-3g", atoms)
        T = quick_kinetic(basis)
        assert T.shape == (7, 7)

    def test_quick_nuclear(self):
        from libaccint import basis_set
        from libaccint._core import quick_nuclear
        atoms = make_water_atoms()
        basis = basis_set("sto-3g", atoms)
        V = quick_nuclear(basis, atoms)
        assert V.shape == (7, 7)

    def test_quick_core_hamiltonian(self):
        from libaccint import basis_set
        from libaccint._core import quick_core_hamiltonian
        atoms = make_water_atoms()
        basis = basis_set("sto-3g", atoms)
        H = quick_core_hamiltonian(basis, atoms)
        assert H.shape == (7, 7)
