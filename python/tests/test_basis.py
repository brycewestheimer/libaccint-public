# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Bryce M. Westheimer. All rights reserved.

"""
Tests for basis set bindings (Atom, Shell, ShellSet, BasisSet).
"""

import pytest
import numpy as np


class TestAtom:
    """Tests for Atom class."""

    def test_constructor_with_point3d(self):
        from libaccint import Atom, Point3D
        pos = Point3D(1.0, 2.0, 3.0)
        atom = Atom(8, pos)
        assert atom.atomic_number == 8
        assert atom.position.x == 1.0
        assert atom.position.y == 2.0
        assert atom.position.z == 3.0

    def test_constructor_with_array(self):
        from libaccint import Atom
        atom = Atom(1, np.array([0.0, 1.43, -1.11]))
        assert atom.atomic_number == 1
        assert atom.position.x == 0.0
        assert abs(atom.position.y - 1.43) < 1e-10

    def test_constructor_with_list(self):
        from libaccint import Atom
        atom = Atom(6, [1.0, 2.0, 3.0])
        assert atom.atomic_number == 6
        assert atom.position.x == 1.0

    def test_repr(self):
        from libaccint import Atom
        atom = Atom(8, [0.0, 0.0, 0.0])
        r = repr(atom)
        assert 'Atom' in r
        assert '8' in r


class TestShell:
    """Tests for Shell class."""

    def test_s_shell(self):
        from libaccint import Shell
        # Simple s-shell at origin with 1 primitive
        shell = Shell(0, [0.0, 0.0, 0.0], [3.42], [1.0])
        assert shell.angular_momentum() == 0
        assert shell.n_primitives() == 1
        assert shell.n_functions() == 1  # S has 1 Cartesian function
        assert shell.valid() is True

    def test_p_shell(self):
        from libaccint import Shell
        shell = Shell(1, [0.0, 0.0, 0.0], [1.0, 0.5], [0.5, 0.5])
        assert shell.angular_momentum() == 1
        assert shell.n_primitives() == 2
        assert shell.n_functions() == 3  # P has 3 Cartesian functions

    def test_d_shell(self):
        from libaccint import Shell
        shell = Shell(2, [0.0, 0.0, 0.0], [1.0], [1.0])
        assert shell.angular_momentum() == 2
        assert shell.n_functions() == 6  # D has 6 Cartesian functions

    def test_shell_with_numpy_arrays(self):
        from libaccint import Shell
        center = np.array([1.0, 2.0, 3.0])
        exps = np.array([3.42, 0.62])
        coeffs = np.array([0.15, 0.85])
        shell = Shell(0, center, exps, coeffs)
        assert shell.angular_momentum() == 0
        assert shell.n_primitives() == 2

    def test_exponents_and_coefficients(self):
        from libaccint import Shell
        exps = [3.42, 0.62, 0.16]
        coeffs = [0.15, 0.5, 0.35]
        shell = Shell(0, [0.0, 0.0, 0.0], exps, coeffs)

        exp_arr = shell.exponents()
        assert isinstance(exp_arr, np.ndarray)
        assert len(exp_arr) == 3
        np.testing.assert_array_almost_equal(exp_arr, exps)

    def test_center_access(self):
        from libaccint import Shell
        shell = Shell(0, [1.0, 2.0, 3.0], [1.0], [1.0])
        center = shell.center()
        assert center.x == 1.0
        assert center.y == 2.0
        assert center.z == 3.0

    def test_tracking_indices(self):
        from libaccint import Shell
        shell = Shell(0, [0.0, 0.0, 0.0], [1.0], [1.0])
        # Before adding to BasisSet, indices should be -1
        assert shell.atom_index() == -1
        assert shell.shell_index() == -1
        assert shell.function_index() == -1

    def test_repr(self):
        from libaccint import Shell
        shell = Shell(2, [0.0, 0.0, 0.0], [1.0], [1.0])
        r = repr(shell)
        assert 'Shell' in r
        assert 'D' in r  # Angular momentum character


class TestBasisSet:
    """Tests for BasisSet class."""

    def test_empty_basis(self):
        from libaccint import BasisSet
        basis = BasisSet()
        assert basis.n_shells() == 0
        assert basis.n_basis_functions() == 0

    def test_basis_from_shells(self):
        from libaccint import BasisSet, Shell
        shells = [
            Shell(0, [0.0, 0.0, 0.0], [1.0], [1.0]),
            Shell(1, [0.0, 0.0, 0.0], [1.0], [1.0]),
        ]
        basis = BasisSet(shells)
        assert basis.n_shells() == 2
        assert basis.n_basis_functions() == 4  # 1 (S) + 3 (P)

    def test_shell_access(self):
        from libaccint import BasisSet, Shell
        shells = [
            Shell(0, [0.0, 0.0, 0.0], [1.0], [1.0]),
            Shell(1, [1.0, 0.0, 0.0], [1.0], [1.0]),
        ]
        basis = BasisSet(shells)

        s1 = basis.shell(0)
        assert s1.angular_momentum() == 0

        s2 = basis.shell(1)
        assert s2.angular_momentum() == 1

    def test_max_angular_momentum(self):
        from libaccint import BasisSet, Shell
        shells = [
            Shell(0, [0.0, 0.0, 0.0], [1.0], [1.0]),
            Shell(2, [0.0, 0.0, 0.0], [1.0], [1.0]),
            Shell(1, [0.0, 0.0, 0.0], [1.0], [1.0]),
        ]
        basis = BasisSet(shells)
        assert basis.max_angular_momentum() == 2

    def test_len(self):
        from libaccint import BasisSet, Shell
        shells = [
            Shell(0, [0.0, 0.0, 0.0], [1.0], [1.0]),
            Shell(0, [1.0, 0.0, 0.0], [1.0], [1.0]),
            Shell(0, [2.0, 0.0, 0.0], [1.0], [1.0]),
        ]
        basis = BasisSet(shells)
        assert len(basis) == 3

    def test_repr(self):
        from libaccint import BasisSet, Shell
        shells = [Shell(0, [0.0, 0.0, 0.0], [1.0], [1.0])]
        basis = BasisSet(shells)
        r = repr(basis)
        assert 'BasisSet' in r


class TestBuiltinBasis:
    """Tests for built-in basis set creation."""

    def test_create_sto3g_hydrogen(self):
        from libaccint import Atom, create_builtin_basis
        atoms = [Atom(1, [0.0, 0.0, 0.0])]
        basis = create_builtin_basis("sto-3g", atoms)
        assert basis.n_shells() == 1
        assert basis.n_basis_functions() == 1

    def test_create_sto3g_water(self):
        from libaccint import Atom, create_builtin_basis
        # Water molecule
        atoms = [
            Atom(8, [0.0, 0.0, 0.0]),          # O
            Atom(1, [0.0, 1.43, -1.11]),       # H
            Atom(1, [0.0, -1.43, -1.11]),      # H
        ]
        basis = create_builtin_basis("sto-3g", atoms)
        # STO-3G for water: O has 2 shells (1s, 2sp), each H has 1 shell
        # Total basis functions: 5 (O) + 1 (H) + 1 (H) = 7
        assert basis.n_basis_functions() == 7

    def test_create_sto3g_case_insensitive(self):
        from libaccint import Atom, create_builtin_basis
        atoms = [Atom(1, [0.0, 0.0, 0.0])]
        # Should work with different cases
        basis1 = create_builtin_basis("sto-3g", atoms)
        basis2 = create_builtin_basis("STO-3G", atoms)
        assert basis1.n_basis_functions() == basis2.n_basis_functions()
