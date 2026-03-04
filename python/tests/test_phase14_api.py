"""Tests for Phase 14 Python API bindings.

Tests cover non-consumer compute(), ERI tensor/block, parallel batch,
screened batch, and screening statistics.
"""

import pytest
import numpy as np


def _make_h2_engine():
    """Create a minimal H2 Engine for testing."""
    import libaccint
    atoms = [
        libaccint.Atom(1, [0.0, 0.0, 0.0]),
        libaccint.Atom(1, [0.0, 0.0, 1.4]),
    ]
    basis = libaccint.basis_set("sto-3g", atoms)
    engine = libaccint.Engine(basis)
    return engine, basis


class TestNonConsumerCompute:
    """Test non-consumer compute_quartet and compute_pair."""

    def test_compute_quartet_returns_buffer(self):
        """compute_quartet should return an IntegralBuffer."""
        import libaccint
        engine, basis = _make_h2_engine()
        op = libaccint.Operator.coulomb()
        quartets = basis.shell_set_quartets()
        if len(quartets) > 0:
            buf = engine.compute_quartet(op, quartets[0])
            assert isinstance(buf, libaccint.IntegralBuffer)

    def test_compute_pair_returns_buffer(self):
        """compute_pair should return an IntegralBuffer."""
        import libaccint
        engine, basis = _make_h2_engine()
        op = libaccint.Operator.overlap()
        pairs = basis.shell_set_pairs()
        if len(pairs) > 0:
            buf = engine.compute_pair(op, pairs[0])
            assert isinstance(buf, libaccint.IntegralBuffer)


class TestERITensor:
    """Test compute_eri_tensor."""

    def test_eri_tensor_shape(self):
        """ERI tensor should have shape (nbf, nbf, nbf, nbf)."""
        import libaccint
        engine, basis = _make_h2_engine()
        nbf = basis.n_basis_functions()
        tensor = engine.compute_eri_tensor()
        assert tensor.shape == (nbf, nbf, nbf, nbf)

    def test_eri_tensor_symmetry(self):
        """ERI tensor should satisfy (ij|kl) = (kl|ij)."""
        import libaccint
        engine, basis = _make_h2_engine()
        tensor = engine.compute_eri_tensor()
        # Check permutation symmetry: (ij|kl) = (kl|ij)
        np.testing.assert_allclose(
            tensor, tensor.transpose(2, 3, 0, 1), atol=1e-12
        )

    def test_eri_tensor_nonzero(self):
        """ERI tensor should have nonzero values."""
        import libaccint
        engine, _ = _make_h2_engine()
        tensor = engine.compute_eri_tensor()
        assert np.any(np.abs(tensor) > 1e-15)


class TestERIBlock:
    """Test compute_eri_block."""

    def test_eri_block_returns_array(self):
        """compute_eri_block should return a numpy array."""
        import libaccint
        engine, basis = _make_h2_engine()
        op = libaccint.Operator.coulomb()
        quartets = basis.shell_set_quartets()
        if len(quartets) > 0:
            block = engine.compute_eri_block(op, quartets[0])
            assert isinstance(block, np.ndarray)
            assert block.size > 0


class TestParallelBatch:
    """Test parallel batch computation."""

    def test_compute_all_2e_parallel(self):
        """Parallel 2e should match serial results."""
        import libaccint
        engine, _ = _make_h2_engine()
        op = libaccint.Operator.coulomb()
        serial = engine.compute_all_2e(op)
        parallel = engine.compute_all_2e_parallel(op, n_threads=2)
        assert len(serial) == len(parallel)

    def test_compute_batch_parallel(self):
        """compute_batch_parallel should return list of IntegralBuffer."""
        import libaccint
        engine, basis = _make_h2_engine()
        op = libaccint.Operator.coulomb()
        quartets = list(basis.shell_set_quartets())
        if len(quartets) > 0:
            results = engine.compute_batch_parallel(op, quartets)
            assert len(results) == len(quartets)
            assert all(isinstance(r, libaccint.IntegralBuffer) for r in results)


class TestScreenedBatch:
    """Test screened batch computation."""

    def test_compute_batch_screened(self):
        """Screened batch should return list of IntegralBuffer."""
        import libaccint
        engine, _ = _make_h2_engine()
        op = libaccint.Operator.coulomb()
        screening = libaccint.ScreeningOptions()
        results = engine.compute_batch_screened(op, screening)
        assert isinstance(results, list)

    def test_screening_statistics(self):
        """compute_screening_statistics should return valid counts."""
        import libaccint
        engine, _ = _make_h2_engine()
        op = libaccint.Operator.coulomb()
        screening = libaccint.ScreeningOptions()
        results = engine.compute_batch_screened(op, screening)
        stats = libaccint.Engine.compute_screening_statistics(results)
        assert stats.total_quartets == len(results)
        assert stats.computed_quartets + stats.skipped_quartets == stats.total_quartets
