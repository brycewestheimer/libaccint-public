# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Bryce M. Westheimer. All rights reserved.

"""Tests for DispatchConfig and Engine screening methods."""

import pytest
import numpy as np


class TestDispatchConfig:
    """Tests for DispatchConfig with expanded fields."""

    def test_default_values(self):
        from libaccint import DispatchConfig
        config = DispatchConfig()
        assert config.min_gpu_batch_size > 0
        assert config.min_gpu_shells > 0
        assert config.min_gpu_primitives > 0
        assert config.high_am_threshold >= 0

    def test_all_fields_readwrite(self):
        from libaccint import DispatchConfig
        config = DispatchConfig()

        config.min_gpu_batch_size = 32
        assert config.min_gpu_batch_size == 32

        config.min_gpu_primitives = 500
        assert config.min_gpu_primitives == 500

        config.high_am_threshold = 3
        assert config.high_am_threshold == 3

        config.min_gpu_shells = 20
        assert config.min_gpu_shells == 20

        config.enable_auto_tuning = True
        assert config.enable_auto_tuning is True

        config.auto_tune_min_batch = 100
        assert config.auto_tune_min_batch == 100

    def test_repr(self):
        from libaccint import DispatchConfig
        config = DispatchConfig()
        r = repr(config)
        assert 'DispatchConfig' in r
        assert 'min_gpu_batch_size' in r

    def test_engine_with_config(self):
        from libaccint import Atom, create_builtin_basis, Engine, DispatchConfig
        config = DispatchConfig()
        config.min_gpu_shells = 100

        atoms = [Atom(1, [0, 0, 0]), Atom(1, [0, 0, 1.4])]
        basis = create_builtin_basis("sto-3g", atoms)
        engine = Engine(basis, config)
        # Should not raise


class TestEngineScreening:
    """Tests for Engine screening methods."""

    @pytest.fixture
    def water_setup(self):
        from libaccint import Atom, create_builtin_basis, Engine
        atoms = [
            Atom(8, [0.0, 0.0, 0.0]),
            Atom(1, [0.0, 1.43, -1.11]),
            Atom(1, [0.0, -1.43, -1.11]),
        ]
        basis = create_builtin_basis("sto-3g", atoms)
        engine = Engine(basis)
        nbf = basis.n_basis_functions()
        return engine, atoms, nbf, basis

    def test_schwarz_not_precomputed_initially(self, water_setup):
        engine, atoms, nbf, basis = water_setup
        assert not engine.schwarz_bounds_precomputed()

    def test_precompute_schwarz_bounds(self, water_setup):
        engine, atoms, nbf, basis = water_setup
        engine.precompute_schwarz_bounds()
        assert engine.schwarz_bounds_precomputed()

    def test_set_density_matrix(self, water_setup):
        engine, atoms, nbf, basis = water_setup
        D = np.eye(nbf) * 0.1
        engine.set_density_matrix(D, nbf)
        assert engine.density_matrix_set()

    def test_set_density_matrix_rejects_non_c_contiguous(self, water_setup):
        engine, atoms, nbf, basis = water_setup
        D = np.asfortranarray(np.eye(nbf) * 0.1)
        assert not D.flags.c_contiguous

        with pytest.raises((ValueError, RuntimeError), match="C-contiguous"):
            engine.set_density_matrix(D, nbf)

    def test_set_dispatch_config(self, water_setup):
        from libaccint import DispatchConfig
        engine, atoms, nbf, basis = water_setup
        config = DispatchConfig()
        config.min_gpu_shells = 50
        engine.set_dispatch_config(config)

    def test_compute_and_consume_screened(self, water_setup):
        from libaccint import FockBuilder, Operator, ScreeningOptions
        engine, atoms, nbf, basis = water_setup

        D = np.eye(nbf) * 0.1
        fb = FockBuilder(nbf)
        fb.set_density(D)

        engine.precompute_schwarz_bounds()
        opts = ScreeningOptions.normal()

        engine.compute_and_consume_screened(Operator.coulomb(), fb, opts)

        J = fb.get_coulomb_matrix()
        K = fb.get_exchange_matrix()
        assert J.shape == (nbf, nbf)
        assert K.shape == (nbf, nbf)
        # J and K should have non-zero elements
        assert np.any(np.abs(J) > 1e-10)
        assert np.any(np.abs(K) > 1e-10)

    def test_compute_and_consume_screened_requires_density_when_weighted(self, water_setup):
        from libaccint import FockBuilder, Operator, ScreeningOptions
        engine, atoms, nbf, basis = water_setup

        D = np.eye(nbf) * 0.1
        fb = FockBuilder(nbf)
        fb.set_density(D)

        opts = ScreeningOptions.normal()
        opts.density_weighted = True

        with pytest.raises((ValueError, RuntimeError), match="density matrix"):
            engine.compute_and_consume_screened(Operator.coulomb(), fb, opts)

    def test_screened_vs_unscreened(self, water_setup):
        """Both screened and unscreened Fock builds should complete and produce results."""
        from libaccint import FockBuilder, Operator, ScreeningOptions
        engine, atoms, nbf, basis = water_setup

        D = np.eye(nbf) * 0.1

        # Unscreened
        fb1 = FockBuilder(nbf)
        fb1.set_density(D)
        engine.compute_and_consume(Operator.coulomb(), fb1)
        J1 = fb1.get_coulomb_matrix()

        # Screened
        fb2 = FockBuilder(nbf)
        fb2.set_density(D)
        engine.precompute_schwarz_bounds()
        opts = ScreeningOptions.normal()
        engine.compute_and_consume_screened(Operator.coulomb(), fb2, opts)
        J2 = fb2.get_coulomb_matrix()

        # Both should have non-zero results
        assert np.any(np.abs(J1) > 1e-10)
        assert np.any(np.abs(J2) > 1e-10)
        assert J1.shape == J2.shape
