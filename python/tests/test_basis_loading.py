# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Bryce M. Westheimer. All rights reserved.

"""Tests for basis set loading infrastructure improvements."""

import pytest


def _h2o_atoms():
    """Create H2O atoms for testing."""
    from libaccint import Atom
    return [
        Atom(8, [0.0, 0.0, 0.0]),
        Atom(1, [0.0, 1.43233673, -1.10866041]),
        Atom(1, [0.0, -1.43233673, -1.10866041]),
    ]


class TestPopleStarNotation:
    """Test that Pople star notation resolves correctly."""

    def test_631g_star(self):
        from libaccint import basis_set
        basis = basis_set("6-31G*", _h2o_atoms())
        assert basis.n_shells() > 5
        assert basis.n_basis_functions() > 7

    def test_631g_double_star(self):
        from libaccint import basis_set
        basis = basis_set("6-31G**", _h2o_atoms())
        assert basis.n_basis_functions() > 7

    def test_def2_svp(self):
        from libaccint import basis_set
        basis = basis_set("def2-SVP", _h2o_atoms())
        assert basis.n_shells() > 0


class TestListAvailableBasisSets:
    """Test list_available_basis_sets()."""

    def test_returns_nonempty(self):
        from libaccint import list_available_basis_sets
        names = list_available_basis_sets()
        assert len(names) >= 40

    def test_is_sorted(self):
        from libaccint import list_available_basis_sets
        names = list_available_basis_sets()
        assert names == sorted(names)

    def test_contains_known_bases(self):
        from libaccint import list_available_basis_sets
        names = list_available_basis_sets()
        assert "sto-3g" in names
        assert "cc-pvdz" in names
        assert "def2-svp" in names
