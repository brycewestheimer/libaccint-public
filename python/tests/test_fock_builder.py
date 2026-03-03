# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Bryce M. Westheimer. All rights reserved.

"""
Tests for FockBuilder consumer bindings.
"""

import pytest
import numpy as np


def make_water_setup():
    """Create water STO-3G setup with engine."""
    from libaccint import Atom, create_builtin_basis, Engine
    atoms = [
        Atom(8, [0.0, 0.0, 0.0]),
        Atom(1, [0.0, 1.43, -1.11]),
        Atom(1, [0.0, -1.43, -1.11]),
    ]
    basis = create_builtin_basis("sto-3g", atoms)
    engine = Engine(basis)
    return engine, atoms, basis.n_basis_functions()


class TestFockBuilder:
    """Tests for FockBuilder class."""

    def test_construction(self):
        from libaccint import FockBuilder
        fb = FockBuilder(7)
        assert fb.nbf() == 7

    def test_set_density(self):
        from libaccint import FockBuilder
        fb = FockBuilder(7)
        D = np.eye(7) * 0.5
        fb.set_density(D)
        # Should not raise

    def test_set_density_rejects_non_c_contiguous(self):
        from libaccint import FockBuilder
        fb = FockBuilder(7)
        D = np.asfortranarray(np.eye(7) * 0.5)
        assert not D.flags.c_contiguous

        with pytest.raises((ValueError, RuntimeError), match="C-contiguous"):
            fb.set_density(D)

    def test_reset(self):
        from libaccint import FockBuilder
        fb = FockBuilder(7)
        fb.reset()
        # J and K should be zero after reset

    def test_get_coulomb_matrix(self):
        from libaccint import FockBuilder
        fb = FockBuilder(7)
        J = fb.get_coulomb_matrix()
        assert isinstance(J, np.ndarray)
        assert J.shape == (7, 7)
        np.testing.assert_array_equal(J, np.zeros((7, 7)))

    def test_get_exchange_matrix(self):
        from libaccint import FockBuilder
        fb = FockBuilder(7)
        K = fb.get_exchange_matrix()
        assert isinstance(K, np.ndarray)
        assert K.shape == (7, 7)
        np.testing.assert_array_equal(K, np.zeros((7, 7)))

    def test_threading_strategy(self):
        from libaccint import FockBuilder, FockThreadingStrategy
        fb = FockBuilder(7)
        assert fb.threading_strategy() == FockThreadingStrategy.Sequential

        fb.set_threading_strategy(FockThreadingStrategy.ThreadLocal)
        assert fb.threading_strategy() == FockThreadingStrategy.ThreadLocal

    def test_repr(self):
        from libaccint import FockBuilder
        fb = FockBuilder(7)
        r = repr(fb)
        assert 'FockBuilder' in r
        assert '7' in r


class TestFockBuild:
    """Tests for Fock matrix construction."""

    def test_compute_and_consume(self):
        from libaccint import FockBuilder, Operator
        engine, atoms, nbf = make_water_setup()

        # Create density matrix (use identity for simple test)
        D = np.eye(nbf) * 0.1

        # Create FockBuilder
        fb = FockBuilder(nbf)
        fb.set_density(D)

        # Compute ERIs and accumulate
        engine.compute_and_consume(Operator.coulomb(), fb)

        # Get J and K
        J = fb.get_coulomb_matrix()
        K = fb.get_exchange_matrix()

        # Both should be non-zero now
        assert J.shape == (nbf, nbf)
        assert K.shape == (nbf, nbf)

        # J and K should be symmetric
        np.testing.assert_array_almost_equal(J, J.T, decimal=12)
        np.testing.assert_array_almost_equal(K, K.T, decimal=12)

    def test_get_fock_matrix(self):
        from libaccint import FockBuilder, Operator
        engine, atoms, nbf = make_water_setup()

        # Get core Hamiltonian
        H = engine.compute_core_hamiltonian(atoms)

        # Density matrix
        D = np.eye(nbf) * 0.1

        # Build Fock matrix
        fb = FockBuilder(nbf)
        fb.set_density(D)
        engine.compute_and_consume(Operator.coulomb(), fb)

        # Get Fock matrix with exchange fraction 1.0
        F = fb.get_fock_matrix(H, 1.0)

        assert F.shape == (nbf, nbf)
        np.testing.assert_array_almost_equal(F, F.T, decimal=12)

        # Verify F = H + J - K
        J = fb.get_coulomb_matrix()
        K = fb.get_exchange_matrix()
        F_expected = H + J - K
        np.testing.assert_array_almost_equal(F, F_expected, decimal=12)

    def test_get_fock_matrix_rejects_non_c_contiguous_hcore(self):
        from libaccint import FockBuilder
        fb = FockBuilder(7)
        H = np.asfortranarray(np.eye(7))
        assert not H.flags.c_contiguous

        with pytest.raises((ValueError, RuntimeError), match="C-contiguous"):
            fb.get_fock_matrix(H, 1.0)

    def test_different_exchange_fractions(self):
        from libaccint import FockBuilder, Operator
        engine, atoms, nbf = make_water_setup()

        H = engine.compute_core_hamiltonian(atoms)
        D = np.eye(nbf) * 0.1

        fb = FockBuilder(nbf)
        fb.set_density(D)
        engine.compute_and_consume(Operator.coulomb(), fb)

        J = fb.get_coulomb_matrix()
        K = fb.get_exchange_matrix()

        # Test different exchange fractions
        for x in [0.0, 0.5, 1.0]:
            F = fb.get_fock_matrix(H, x)
            F_expected = H + J - x * K
            np.testing.assert_array_almost_equal(F, F_expected, decimal=12)
