# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Bryce M. Westheimer. All rights reserved.

"""Phase 16 — Python round-trip integration tests.

Validates the full pipeline from Python:
  atoms → basis → engine → 1e integrals → Fock build → physical properties

This covers Task 16.1.4 (Python → C++ round-trip) and ensures all
remediated modules work together through the Python bindings.
"""

import numpy as np
import pytest

import libaccint
from libaccint._core import (
    Atom,
    BasisSet,
    Engine,
    FockBuilder,
    Operator,
    PointChargeParams,
    ScreeningOptions,
    ScreeningPreset,
    BackendHint,
    create_builtin_basis,
)


# =============================================================================
# 16.1.4: Full Python → C++ Round-Trip Tests
# =============================================================================


class TestWaterRoundTrip:
    """Full integration pipeline for H2O/STO-3G."""

    @pytest.fixture
    def water_setup(self):
        """Create water atoms, basis, and engine."""
        atoms = [
            Atom(8, [0.0, 0.0, 0.0]),
            Atom(1, [0.0, 1.43233673, -1.10866041]),
            Atom(1, [0.0, -1.43233673, -1.10866041]),
        ]
        basis = create_builtin_basis("sto-3g", atoms)
        engine = Engine(basis)
        return atoms, basis, engine

    def test_basis_creation(self, water_setup):
        """Atoms → BasisSet creation succeeds with expected properties."""
        atoms, basis, _ = water_setup
        assert basis.n_basis_functions() == 7  # O: 1s,2s,2px,2py,2pz + H×2: 1s
        assert basis.n_shells() > 0
        assert basis.n_shell_sets() > 0

    def test_one_electron_integrals(self, water_setup):
        """Full 1e integral computation and physical validation."""
        atoms, basis, engine = water_setup
        nbf = basis.n_basis_functions()

        S = engine.compute_overlap_matrix()
        T = engine.compute_kinetic_matrix()
        V = engine.compute_nuclear_matrix(atoms)
        H = engine.compute_core_hamiltonian(atoms)

        # Shape
        for M in (S, T, V, H):
            assert M.shape == (nbf, nbf)

        # Symmetry
        for M, name in [(S, "S"), (T, "T"), (V, "V"), (H, "H")]:
            np.testing.assert_array_almost_equal(
                M, M.T, decimal=12, err_msg=f"{name} not symmetric"
            )

        # Overlap diagonal = 1 (normalized basis)
        np.testing.assert_array_almost_equal(
            np.diag(S), np.ones(nbf), decimal=10
        )

        # Core Hamiltonian = T + V
        np.testing.assert_array_almost_equal(H, T + V, decimal=12)

        # Kinetic: positive semi-definite
        T_evals = np.linalg.eigvalsh(T)
        assert np.all(T_evals >= -1e-12), "T should be positive semi-definite"

        # Nuclear attraction: negative semi-definite (attractive potential)
        V_evals = np.linalg.eigvalsh(V)
        assert np.all(V_evals <= 1e-12), "V should be negative semi-definite"

    def test_fock_build(self, water_setup):
        """Full 2e integral Fock build through consumer."""
        atoms, basis, engine = water_setup
        nbf = basis.n_basis_functions()

        # Use a simple density matrix
        D = np.eye(nbf) * (1.0 / nbf)

        fb = FockBuilder(nbf)
        fb.set_density(D)
        engine.compute_and_consume(Operator.coulomb(), fb)

        J = fb.get_coulomb_matrix()
        K = fb.get_exchange_matrix()

        # Symmetry
        np.testing.assert_array_almost_equal(J, J.T, decimal=12)
        np.testing.assert_array_almost_equal(K, K.T, decimal=12)

        # Non-zero for non-zero density
        assert np.linalg.norm(J) > 1e-10, "J should be non-zero"
        assert np.linalg.norm(K) > 1e-10, "K should be non-zero"

    def test_fock_matrix_composition(self, water_setup):
        """Verify F = H + J - xK through get_fock_matrix."""
        atoms, basis, engine = water_setup
        nbf = basis.n_basis_functions()

        H = engine.compute_core_hamiltonian(atoms)
        D = np.eye(nbf) * (1.0 / nbf)

        fb = FockBuilder(nbf)
        fb.set_density(D)
        engine.compute_and_consume(Operator.coulomb(), fb)

        J = fb.get_coulomb_matrix()
        K = fb.get_exchange_matrix()
        F = fb.get_fock_matrix(H, 1.0)  # HF exchange scaling

        # F = H + J - K (with x=1.0)
        expected = H + J - K
        np.testing.assert_array_almost_equal(F, expected, decimal=12)

    def test_full_pipeline_consistency(self, water_setup):
        """Verify consistent results across engine calls."""
        atoms, basis, engine = water_setup

        S1 = engine.compute_overlap_matrix()
        S2 = engine.compute_overlap_matrix()
        np.testing.assert_array_equal(S1, S2)

    def test_screened_fock_build(self, water_setup):
        """Screened 2e integrals produce physically valid results."""
        atoms, basis, engine = water_setup
        nbf = basis.n_basis_functions()

        D = np.eye(nbf) * (1.0 / nbf)

        # Screened path
        engine.precompute_schwarz_bounds()
        engine.set_density_matrix(D, nbf)

        fb = FockBuilder(nbf)
        fb.set_density(D)
        options = ScreeningOptions.from_preset(ScreeningPreset.NORMAL)
        engine.compute_and_consume_screened(
            Operator.coulomb(), fb, options
        )
        J_screened = fb.get_coulomb_matrix()

        # Screened J should be non-zero for non-zero density
        assert np.linalg.norm(J_screened) > 1e-10, \
            "Screened J should be non-zero"

        # Diagonal elements should be positive (self-interactions)
        for i in range(nbf):
            assert J_screened[i, i] >= 0.0, \
                f"J diagonal [{i},{i}] should be non-negative"


class TestH2RoundTrip:
    """Minimal H2 round-trip (2 basis functions)."""

    @pytest.fixture
    def h2_setup(self):
        atoms = [
            Atom(1, [0.0, 0.0, 0.0]),
            Atom(1, [0.0, 0.0, 1.39839733]),
        ]
        basis = create_builtin_basis("sto-3g", atoms)
        engine = Engine(basis)
        return atoms, basis, engine

    def test_minimal_pipeline(self, h2_setup):
        """End-to-end for 2-function system."""
        atoms, basis, engine = h2_setup
        assert basis.n_basis_functions() == 2

        S = engine.compute_overlap_matrix()
        assert S.shape == (2, 2)
        assert abs(S[0, 0] - 1.0) < 1e-10
        assert 0.0 < S[0, 1] < 1.0  # Overlap should be positive

        H = engine.compute_core_hamiltonian(atoms)
        assert H.shape == (2, 2)
        np.testing.assert_array_almost_equal(H, H.T, decimal=12)

        D = np.eye(2) * 0.5
        fb = FockBuilder(2)
        fb.set_density(D)
        engine.compute_and_consume(Operator.coulomb(), fb)

        J = fb.get_coulomb_matrix()
        np.testing.assert_array_almost_equal(J, J.T, decimal=12)


class TestConvenienceRoundTrip:
    """Round-trip using convenience API wrappers."""

    def test_convenience_pipeline(self):
        atoms = [
            Atom(8, [0.0, 0.0, 0.0]),
            Atom(1, [0.0, 1.43233673, -1.10866041]),
            Atom(1, [0.0, -1.43233673, -1.10866041]),
        ]

        basis = libaccint.basis_set("sto-3g", atoms)
        engine = Engine(basis)
        nbf = basis.n_basis_functions()

        S = libaccint.compute_overlap(engine)
        T = libaccint.compute_kinetic(engine)
        V = libaccint.compute_nuclear(engine, atoms)
        H = libaccint.compute_core_hamiltonian(engine, atoms)

        np.testing.assert_array_almost_equal(H, T + V, decimal=12)

        D = np.eye(nbf) * (1.0 / nbf)
        F = libaccint.build_fock(engine, D, H)
        assert F.shape == (nbf, nbf)
        np.testing.assert_array_almost_equal(F, F.T, decimal=12)

    def test_atoms_from_xyz_pipeline(self):
        """Test atoms_from_xyz convenience into full pipeline."""
        atoms = libaccint.atoms_from_xyz([
            (1, 0.0, 0.0, 0.0),
            (1, 0.0, 0.0, 1.4),
        ])
        basis = libaccint.basis_set("sto-3g", atoms)
        S = libaccint.compute_overlap(basis)
        assert S.shape == (2, 2)
        assert abs(S[0, 0] - 1.0) < 1e-10


# =============================================================================
# 16.3.1: Python-side Regression Tests
# =============================================================================


class TestPythonRegressions:
    """Regression tests for Python-specific fixes made during remediation."""

    def test_fock_builder_reset(self):
        """FockBuilder.reset() clears accumulated data."""
        nbf = 7
        fb = FockBuilder(nbf)
        atoms = [
            Atom(8, [0.0, 0.0, 0.0]),
            Atom(1, [0.0, 1.43, -1.11]),
            Atom(1, [0.0, -1.43, -1.11]),
        ]
        basis = create_builtin_basis("sto-3g", atoms)
        engine = Engine(basis)

        D = np.eye(nbf) * 0.1
        fb.set_density(D)
        engine.compute_and_consume(Operator.coulomb(), fb)
        J1 = np.array(fb.get_coulomb_matrix())  # copy

        fb.reset()
        J_zero = fb.get_coulomb_matrix()
        assert np.linalg.norm(J_zero) < 1e-15, "After reset, J should be zero"

    def test_operator_factories_python(self):
        """All operator factories produce valid operators."""
        _ = Operator.overlap()
        _ = Operator.kinetic()
        _ = Operator.coulomb()

        params = PointChargeParams()
        params.x = [0.0]
        params.y = [0.0]
        params.z = [0.0]
        params.charges = [1.0]
        _ = Operator.nuclear(params)

    def test_basis_keeps_alive(self):
        """Engine keeps basis alive (Phase 14 py::keep_alive fix)."""
        atoms = [Atom(1, [0.0, 0.0, 0.0]), Atom(1, [0.0, 0.0, 1.4])]
        basis = create_builtin_basis("sto-3g", atoms)
        engine = Engine(basis)
        del basis  # Would crash without keep_alive

        # Engine should still work
        S = engine.compute_overlap_matrix()
        assert S.shape == (2, 2)
        assert abs(S[0, 0] - 1.0) < 1e-10

    def test_screening_options_presets(self):
        """Screening preset constructors work from Python."""
        opts_none = ScreeningOptions.none()
        assert not opts_none.enabled

        opts_normal = ScreeningOptions.from_preset(ScreeningPreset.NORMAL)
        assert opts_normal.enabled

        opts_tight = ScreeningOptions.from_preset(ScreeningPreset.TIGHT)
        assert opts_tight.enabled
        assert opts_tight.threshold < opts_normal.threshold
