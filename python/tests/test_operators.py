# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Bryce M. Westheimer. All rights reserved.

"""
Tests for operator bindings.
"""

import pytest
import numpy as np


class TestOperatorKind:
    """Tests for OperatorKind enum."""

    def test_one_electron_kinds(self):
        from libaccint import OperatorKind
        assert OperatorKind.Overlap is not None
        assert OperatorKind.Kinetic is not None
        assert OperatorKind.Nuclear is not None
        assert OperatorKind.PointCharge is not None

    def test_two_electron_kinds(self):
        from libaccint import OperatorKind
        assert OperatorKind.Coulomb is not None
        assert OperatorKind.ErfCoulomb is not None
        assert OperatorKind.ErfcCoulomb is not None


class TestPointChargeParams:
    """Tests for PointChargeParams class."""

    def test_empty_params(self):
        from libaccint import PointChargeParams
        params = PointChargeParams()
        assert params.n_centers() == 0
        assert len(params) == 0

    def test_params_from_arrays(self):
        from libaccint import PointChargeParams
        params = PointChargeParams(
            x=[0.0, 1.0, 2.0],
            y=[0.0, 0.0, 0.0],
            z=[0.0, 0.0, 0.0],
            charges=[8.0, 1.0, 1.0]
        )
        assert params.n_centers() == 3
        assert len(params) == 3

    def test_params_properties(self):
        from libaccint import PointChargeParams
        params = PointChargeParams(
            x=[1.0],
            y=[2.0],
            z=[3.0],
            charges=[6.0]
        )
        assert params.x == [1.0]
        assert params.y == [2.0]
        assert params.z == [3.0]
        assert params.charges == [6.0]

    def test_repr(self):
        from libaccint import PointChargeParams
        params = PointChargeParams(x=[0.0], y=[0.0], z=[0.0], charges=[1.0])
        r = repr(params)
        assert 'PointChargeParams' in r
        assert '1' in r  # n_centers


class TestRangeSeparatedParams:
    """Tests for RangeSeparatedParams class."""

    def test_default(self):
        from libaccint import RangeSeparatedParams
        params = RangeSeparatedParams()
        assert params.omega == 0.0

    def test_with_omega(self):
        from libaccint import RangeSeparatedParams
        params = RangeSeparatedParams(0.3)
        assert abs(params.omega - 0.3) < 1e-10

    def test_repr(self):
        from libaccint import RangeSeparatedParams
        params = RangeSeparatedParams(0.5)
        r = repr(params)
        assert 'RangeSeparatedParams' in r


class TestOperator:
    """Tests for Operator class."""

    def test_overlap(self):
        from libaccint import Operator, OperatorKind
        op = Operator.overlap()
        assert op.kind() == OperatorKind.Overlap
        assert op.is_one_electron() is True
        assert op.is_two_electron() is False

    def test_kinetic(self):
        from libaccint import Operator, OperatorKind
        op = Operator.kinetic()
        assert op.kind() == OperatorKind.Kinetic
        assert op.is_one_electron() is True

    def test_coulomb(self):
        from libaccint import Operator, OperatorKind
        op = Operator.coulomb()
        assert op.kind() == OperatorKind.Coulomb
        assert op.is_two_electron() is True
        assert op.is_one_electron() is False

    def test_nuclear(self):
        from libaccint import Operator, OperatorKind, PointChargeParams
        params = PointChargeParams(
            x=[0.0], y=[0.0], z=[0.0], charges=[8.0]
        )
        op = Operator.nuclear(params)
        assert op.kind() == OperatorKind.Nuclear
        assert op.is_one_electron() is True

    def test_erf_coulomb(self):
        from libaccint import Operator, OperatorKind
        op = Operator.erf_coulomb(0.3)
        assert op.kind() == OperatorKind.ErfCoulomb
        assert op.is_two_electron() is True

    def test_erfc_coulomb(self):
        from libaccint import Operator, OperatorKind
        op = Operator.erfc_coulomb(0.3)
        assert op.kind() == OperatorKind.ErfcCoulomb
        assert op.is_two_electron() is True

    def test_repr(self):
        from libaccint import Operator
        op = Operator.overlap()
        r = repr(op)
        assert 'Operator' in r


class TestOneElectronOperator:
    """Tests for OneElectronOperator class."""

    def test_single_operator(self):
        from libaccint import Operator, OneElectronOperator
        op = OneElectronOperator(Operator.kinetic())
        assert op.n_contributions() == 1

    def test_composed_operators(self):
        from libaccint import Operator, OneElectronOperator, PointChargeParams
        op = OneElectronOperator(Operator.kinetic())
        params = PointChargeParams(x=[0.0], y=[0.0], z=[0.0], charges=[8.0])
        op.add(Operator.nuclear(params))
        assert op.n_contributions() == 2

    def test_repr(self):
        from libaccint import Operator, OneElectronOperator
        op = OneElectronOperator(Operator.overlap())
        r = repr(op)
        assert 'OneElectronOperator' in r
