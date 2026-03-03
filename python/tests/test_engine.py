# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Bryce M. Westheimer. All rights reserved.

"""
Tests for Engine bindings.
"""

import pytest
import numpy as np


def make_h2_basis():
    """Create STO-3G basis for H2."""
    from libaccint import Atom, create_builtin_basis
    atoms = [
        Atom(1, [0.0, 0.0, 0.0]),
        Atom(1, [1.4, 0.0, 0.0]),  # ~0.74 Angstrom = 1.4 Bohr
    ]
    return create_builtin_basis("sto-3g", atoms), atoms


def make_water_basis():
    """Create STO-3G basis for water."""
    from libaccint import Atom, create_builtin_basis
    atoms = [
        Atom(8, [0.0, 0.0, 0.0]),
        Atom(1, [0.0, 1.43, -1.11]),
        Atom(1, [0.0, -1.43, -1.11]),
    ]
    return create_builtin_basis("sto-3g", atoms), atoms


class TestEngineConstruction:
    """Tests for Engine construction."""

    def test_create_engine(self):
        from libaccint import Engine
        basis, _ = make_h2_basis()
        engine = Engine(basis)
        assert engine.max_angular_momentum() == 0  # H2 STO-3G is all s

    def test_engine_with_config(self):
        from libaccint import Engine, DispatchConfig
        basis, _ = make_h2_basis()
        config = DispatchConfig()
        config.min_gpu_shells = 100
        engine = Engine(basis, config)
        assert engine is not None

    def test_gpu_available(self):
        from libaccint import Engine
        basis, _ = make_h2_basis()
        engine = Engine(basis)
        # Should return bool (usually False on most test systems)
        assert isinstance(engine.gpu_available(), bool)

    def test_basis_access(self):
        from libaccint import Engine
        basis, _ = make_h2_basis()
        engine = Engine(basis)
        assert engine.basis().n_shells() == basis.n_shells()

    def test_repr(self):
        from libaccint import Engine
        basis, _ = make_h2_basis()
        engine = Engine(basis)
        r = repr(engine)
        assert 'Engine' in r


class TestOverlapMatrix:
    """Tests for overlap matrix computation."""

    def test_h2_overlap(self):
        from libaccint import Engine
        basis, _ = make_h2_basis()
        engine = Engine(basis)
        S = engine.compute_overlap_matrix()

        assert isinstance(S, np.ndarray)
        assert S.shape == (2, 2)

        # S should be symmetric
        np.testing.assert_array_almost_equal(S, S.T)

        # Diagonal should be ~1 for normalized basis
        np.testing.assert_array_almost_equal(np.diag(S), [1.0, 1.0], decimal=10)

    def test_water_overlap(self):
        from libaccint import Engine
        basis, _ = make_water_basis()
        engine = Engine(basis)
        S = engine.compute_overlap_matrix()

        assert S.shape == (7, 7)

        # Should be symmetric
        np.testing.assert_array_almost_equal(S, S.T)

        # Diagonal should be 1
        np.testing.assert_array_almost_equal(np.diag(S), np.ones(7), decimal=10)


class TestKineticMatrix:
    """Tests for kinetic energy matrix computation."""

    def test_h2_kinetic(self):
        from libaccint import Engine
        basis, _ = make_h2_basis()
        engine = Engine(basis)
        T = engine.compute_kinetic_matrix()

        assert isinstance(T, np.ndarray)
        assert T.shape == (2, 2)

        # T should be symmetric
        np.testing.assert_array_almost_equal(T, T.T)

        # Kinetic energy should be positive
        eigvals = np.linalg.eigvalsh(T)
        assert all(eigvals > 0)

    def test_water_kinetic(self):
        from libaccint import Engine
        basis, _ = make_water_basis()
        engine = Engine(basis)
        T = engine.compute_kinetic_matrix()

        assert T.shape == (7, 7)
        np.testing.assert_array_almost_equal(T, T.T)


class TestNuclearMatrix:
    """Tests for nuclear attraction matrix computation."""

    def test_h2_nuclear(self):
        from libaccint import Engine
        basis, atoms = make_h2_basis()
        engine = Engine(basis)
        V = engine.compute_nuclear_matrix(atoms)

        assert isinstance(V, np.ndarray)
        assert V.shape == (2, 2)

        # V should be symmetric
        np.testing.assert_array_almost_equal(V, V.T)

        # Nuclear attraction should be negative (attractive)
        eigvals = np.linalg.eigvalsh(V)
        assert all(eigvals < 0)

    def test_water_nuclear(self):
        from libaccint import Engine
        basis, atoms = make_water_basis()
        engine = Engine(basis)
        V = engine.compute_nuclear_matrix(atoms)

        assert V.shape == (7, 7)
        np.testing.assert_array_almost_equal(V, V.T)


class TestCoreHamiltonian:
    """Tests for core Hamiltonian computation."""

    def test_h2_core_hamiltonian(self):
        from libaccint import Engine
        basis, atoms = make_h2_basis()
        engine = Engine(basis)

        H = engine.compute_core_hamiltonian(atoms)

        # Compare with T + V
        T = engine.compute_kinetic_matrix()
        V = engine.compute_nuclear_matrix(atoms)

        np.testing.assert_array_almost_equal(H, T + V, decimal=12)

    def test_water_core_hamiltonian(self):
        from libaccint import Engine
        basis, atoms = make_water_basis()
        engine = Engine(basis)

        H = engine.compute_core_hamiltonian(atoms)
        T = engine.compute_kinetic_matrix()
        V = engine.compute_nuclear_matrix(atoms)

        np.testing.assert_array_almost_equal(H, T + V, decimal=12)


class TestBackendHint:
    """Tests for backend hint functionality."""

    def test_force_cpu(self):
        from libaccint import Engine, BackendHint
        basis, _ = make_h2_basis()
        engine = Engine(basis)
        S = engine.compute_overlap_matrix(hint=BackendHint.ForceCPU)
        assert S.shape == (2, 2)

    def test_auto_hint(self):
        from libaccint import Engine, BackendHint
        basis, _ = make_h2_basis()
        engine = Engine(basis)
        S = engine.compute_overlap_matrix(hint=BackendHint.Auto)
        assert S.shape == (2, 2)
