# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Bryce M. Westheimer. All rights reserved.

"""
Tests for core type bindings (Point3D, enums, etc.).
"""

import pytest
import numpy as np


def test_import():
    """Test that the module can be imported."""
    import libaccint
    assert hasattr(libaccint, '__version__')


def test_version():
    """Test version string."""
    import libaccint
    assert libaccint.__version__ == "0.1.0a2"


class TestPoint3D:
    """Tests for Point3D class."""

    def test_default_constructor(self):
        from libaccint import Point3D
        p = Point3D()
        assert p.x == 0.0
        assert p.y == 0.0
        assert p.z == 0.0

    def test_constructor_with_values(self):
        from libaccint import Point3D
        p = Point3D(1.0, 2.0, 3.0)
        assert p.x == 1.0
        assert p.y == 2.0
        assert p.z == 3.0

    def test_constructor_from_array(self):
        from libaccint import Point3D
        arr = np.array([1.0, 2.0, 3.0])
        p = Point3D(arr)
        assert p.x == 1.0
        assert p.y == 2.0
        assert p.z == 3.0

    def test_getitem(self):
        from libaccint import Point3D
        p = Point3D(1.0, 2.0, 3.0)
        assert p[0] == 1.0
        assert p[1] == 2.0
        assert p[2] == 3.0

    def test_setitem(self):
        from libaccint import Point3D
        p = Point3D()
        p[0] = 1.0
        p[1] = 2.0
        p[2] = 3.0
        assert p.x == 1.0
        assert p.y == 2.0
        assert p.z == 3.0

    def test_to_numpy(self):
        from libaccint import Point3D
        p = Point3D(1.0, 2.0, 3.0)
        arr = p.to_numpy()
        assert isinstance(arr, np.ndarray)
        assert arr.shape == (3,)
        np.testing.assert_array_equal(arr, [1.0, 2.0, 3.0])

    def test_distance_squared(self):
        from libaccint import Point3D
        p1 = Point3D(0.0, 0.0, 0.0)
        p2 = Point3D(1.0, 0.0, 0.0)
        assert p1.distance_squared(p2) == 1.0

        p3 = Point3D(1.0, 1.0, 1.0)
        assert abs(p1.distance_squared(p3) - 3.0) < 1e-10

    def test_repr(self):
        from libaccint import Point3D
        p = Point3D(1.0, 2.0, 3.0)
        r = repr(p)
        assert 'Point3D' in r
        assert '1' in r


class TestAngularMomentum:
    """Tests for AngularMomentum enum."""

    def test_values(self):
        from libaccint import AngularMomentum
        assert int(AngularMomentum.S) == 0
        assert int(AngularMomentum.P) == 1
        assert int(AngularMomentum.D) == 2
        assert int(AngularMomentum.F) == 3
        assert int(AngularMomentum.G) == 4
        assert int(AngularMomentum.H) == 5
        assert int(AngularMomentum.I) == 6


class TestBackendType:
    """Tests for BackendType enum."""

    def test_values(self):
        from libaccint import BackendType
        assert BackendType.CPU is not None
        assert BackendType.CUDA is not None
        assert BackendType.HIP is not None

    def test_backend_available(self):
        from libaccint import is_backend_available, BackendType
        # CPU should always be available
        assert is_backend_available(BackendType.CPU) is True


class TestBackendHint:
    """Tests for BackendHint enum."""

    def test_values(self):
        from libaccint import BackendHint
        assert BackendHint.Auto is not None
        assert BackendHint.PreferCPU is not None
        assert BackendHint.PreferGPU is not None
        assert BackendHint.ForceCPU is not None
        assert BackendHint.ForceGPU is not None


class TestHelperFunctions:
    """Tests for helper functions."""

    def test_n_cartesian(self):
        from libaccint._core import n_cartesian
        assert n_cartesian(0) == 1   # S
        assert n_cartesian(1) == 3   # P
        assert n_cartesian(2) == 6   # D
        assert n_cartesian(3) == 10  # F
        assert n_cartesian(4) == 15  # G

    def test_n_spherical(self):
        from libaccint._core import n_spherical
        assert n_spherical(0) == 1   # S
        assert n_spherical(1) == 3   # P
        assert n_spherical(2) == 5   # D
        assert n_spherical(3) == 7   # F
        assert n_spherical(4) == 9   # G
