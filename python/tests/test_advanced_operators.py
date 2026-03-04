# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Bryce M. Westheimer. All rights reserved.

"""Tests for new operator types, factory methods, and classification functions."""

import pytest
import numpy as np


class TestOriginParams:
    """Tests for OriginParams struct."""

    def test_default_construction(self):
        from libaccint import OriginParams
        o = OriginParams()
        # Default origin should exist

    def test_from_array(self):
        from libaccint import OriginParams
        o = OriginParams(np.array([1.0, 2.0, 3.0]))
        # Should not raise

    def test_repr(self):
        from libaccint import OriginParams
        o = OriginParams()
        r = repr(o)
        assert 'OriginParams' in r


class TestDistributedMultipoleParams:
    """Tests for DistributedMultipoleParams struct."""

    def test_empty_construction(self):
        from libaccint import DistributedMultipoleParams
        dmp = DistributedMultipoleParams()
        assert dmp.n_sites() == 0
        assert dmp.is_valid()

    def test_charge_site(self):
        from libaccint import DistributedMultipoleParams
        dmp = DistributedMultipoleParams()
        dmp.x = [0.0]
        dmp.y = [0.0]
        dmp.z = [0.0]
        dmp.charges = [1.0]
        assert dmp.n_sites() == 1

    def test_max_rank_with_dipole(self):
        from libaccint import DistributedMultipoleParams
        dmp = DistributedMultipoleParams()
        dmp.x = [0.0]
        dmp.y = [0.0]
        dmp.z = [0.0]
        dmp.charges = [1.0]
        dmp.dipole_x = [0.1]
        dmp.dipole_y = [0.0]
        dmp.dipole_z = [0.0]
        assert dmp.max_rank() >= 1


class TestProjectionOperatorParams:
    """Tests for ProjectionOperatorParams struct."""

    def test_empty_construction(self):
        from libaccint import ProjectionOperatorParams
        pop = ProjectionOperatorParams()
        assert pop.is_valid()


class TestAdvancedOperatorKinds:
    """Tests for new OperatorKind values."""

    def test_advanced_kinds_exist(self):
        from libaccint import OperatorKind
        assert OperatorKind.DistributedMultipole is not None
        assert OperatorKind.ProjectionOperator is not None

    def test_property_kinds_exist(self):
        from libaccint import OperatorKind
        assert OperatorKind.ElectricDipole is not None
        assert OperatorKind.ElectricQuadrupole is not None
        assert OperatorKind.ElectricOctupole is not None
        assert OperatorKind.LinearMomentum is not None
        assert OperatorKind.AngularMomentum is not None


class TestOperatorFactoryMethods:
    """Tests for new Operator factory methods."""

    def test_electric_dipole(self):
        from libaccint import Operator, OperatorKind
        op = Operator.electric_dipole()
        assert op.kind() == OperatorKind.ElectricDipole
        assert op.is_one_electron()

    def test_electric_quadrupole(self):
        from libaccint import Operator, OperatorKind
        op = Operator.electric_quadrupole()
        assert op.kind() == OperatorKind.ElectricQuadrupole
        assert op.is_one_electron()

    def test_electric_octupole(self):
        from libaccint import Operator, OperatorKind
        op = Operator.electric_octupole()
        assert op.kind() == OperatorKind.ElectricOctupole
        assert op.is_one_electron()

    def test_linear_momentum(self):
        from libaccint import Operator, OperatorKind
        op = Operator.linear_momentum()
        assert op.kind() == OperatorKind.LinearMomentum
        assert op.is_one_electron()

    def test_angular_momentum(self):
        from libaccint import Operator, OperatorKind
        op = Operator.angular_momentum()
        assert op.kind() == OperatorKind.AngularMomentum
        assert op.is_one_electron()

    def test_dipole_with_origin(self):
        from libaccint import Operator, OriginParams
        origin = OriginParams(np.array([1.0, 0.0, 0.0]))
        op = Operator.electric_dipole(origin)
        assert op.is_one_electron()

    def test_point_charges(self):
        from libaccint import Operator, PointChargeParams
        params = PointChargeParams(
            [0.0, 1.0], [0.0, 0.0], [0.0, 0.0], [-1.0, 1.0]
        )
        op = Operator.point_charges(params)
        assert op.is_one_electron()


class TestClassificationFunctions:
    """Tests for operator classification utility functions."""

    def test_is_one_electron(self):
        from libaccint import is_one_electron, OperatorKind
        assert is_one_electron(OperatorKind.Overlap)
        assert is_one_electron(OperatorKind.Kinetic)
        assert is_one_electron(OperatorKind.Nuclear)
        assert not is_one_electron(OperatorKind.Coulomb)

    def test_is_two_electron(self):
        from libaccint import is_two_electron, OperatorKind
        assert is_two_electron(OperatorKind.Coulomb)
        assert is_two_electron(OperatorKind.ErfCoulomb)
        assert not is_two_electron(OperatorKind.Overlap)

    def test_is_multi_component(self):
        from libaccint import is_multi_component, OperatorKind
        assert is_multi_component(OperatorKind.ElectricDipole)
        assert not is_multi_component(OperatorKind.Overlap)

    def test_is_property_integral(self):
        from libaccint import is_property_integral, OperatorKind
        assert is_property_integral(OperatorKind.ElectricDipole)
        assert is_property_integral(OperatorKind.ElectricQuadrupole)

    def test_component_count(self):
        from libaccint import component_count, OperatorKind
        assert component_count(OperatorKind.ElectricDipole) == 3
        assert component_count(OperatorKind.ElectricQuadrupole) == 6
        assert component_count(OperatorKind.ElectricOctupole) == 10
        assert component_count(OperatorKind.Overlap) == 1
