# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Bryce M. Westheimer. All rights reserved.

"""
Tests for integral buffer bindings.
"""

import pytest
import numpy as np


class TestOneElectronBuffer:
    """Tests for OneElectronBuffer class."""

    def test_empty_buffer(self):
        from libaccint import OneElectronBuffer
        buf = OneElectronBuffer()
        # Should be empty initially

    def test_sized_buffer(self):
        from libaccint import OneElectronBuffer
        buf = OneElectronBuffer(3, 3)  # For p-p shell pair
        # Should have 9 elements

    def test_resize(self):
        from libaccint import OneElectronBuffer
        buf = OneElectronBuffer()
        buf.resize(6, 6)  # For d-d shell pair
        # Should now have 36 elements

    def test_clear(self):
        from libaccint import OneElectronBuffer
        buf = OneElectronBuffer(2, 2)
        buf.clear()
        # All elements should be zero

    def test_getitem_setitem(self):
        from libaccint import OneElectronBuffer
        buf = OneElectronBuffer(3, 3)
        buf.clear()

        # Set a value
        buf[1, 2] = 1.5
        assert abs(buf[1, 2] - 1.5) < 1e-10

    def test_to_numpy(self):
        from libaccint import OneElectronBuffer
        buf = OneElectronBuffer(3, 3)
        buf.clear()
        arr = buf.to_numpy()
        assert isinstance(arr, np.ndarray)
        assert arr.shape == (3, 3)

    def test_repr(self):
        from libaccint import OneElectronBuffer
        buf = OneElectronBuffer(3, 3)
        r = repr(buf)
        assert 'OneElectronBuffer' in r


class TestTwoElectronBuffer:
    """Tests for TwoElectronBuffer class."""

    def test_empty_buffer(self):
        from libaccint import TwoElectronBuffer
        buf = TwoElectronBuffer()

    def test_sized_buffer(self):
        from libaccint import TwoElectronBuffer
        buf = TwoElectronBuffer(1, 1, 1, 1)  # For (s s | s s)

    def test_resize(self):
        from libaccint import TwoElectronBuffer
        buf = TwoElectronBuffer()
        buf.resize(3, 3, 3, 3)  # For (p p | p p)

    def test_clear(self):
        from libaccint import TwoElectronBuffer
        buf = TwoElectronBuffer(1, 1, 1, 1)
        buf.clear()

    def test_getitem_setitem(self):
        from libaccint import TwoElectronBuffer
        buf = TwoElectronBuffer(2, 2, 2, 2)
        buf.clear()

        buf[0, 1, 0, 1] = 2.5
        assert abs(buf[0, 1, 0, 1] - 2.5) < 1e-10

    def test_to_numpy(self):
        from libaccint import TwoElectronBuffer
        buf = TwoElectronBuffer(2, 2, 2, 2)
        buf.clear()
        arr = buf.to_numpy()
        assert isinstance(arr, np.ndarray)
        assert arr.shape == (2, 2, 2, 2)

    def test_repr(self):
        from libaccint import TwoElectronBuffer
        buf = TwoElectronBuffer(2, 2, 2, 2)
        r = repr(buf)
        assert 'TwoElectronBuffer' in r
