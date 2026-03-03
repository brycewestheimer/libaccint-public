# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Bryce M. Westheimer. All rights reserved.

"""Tests for screening module bindings."""

import pytest
import numpy as np


class TestScreeningPreset:
    """Tests for ScreeningPreset enum."""

    def test_enum_values(self):
        from libaccint import ScreeningPreset
        assert ScreeningPreset.NONE is not None
        assert ScreeningPreset.LOOSE is not None
        assert ScreeningPreset.NORMAL is not None
        assert ScreeningPreset.TIGHT is not None
        assert ScreeningPreset.CUSTOM is not None


class TestScreeningOptions:
    """Tests for ScreeningOptions struct."""

    def test_default_construction(self):
        from libaccint import ScreeningOptions
        opts = ScreeningOptions()
        assert opts.enabled

    def test_field_readwrite(self):
        from libaccint import ScreeningOptions
        opts = ScreeningOptions()
        opts.threshold = 1e-10
        assert abs(opts.threshold - 1e-10) < 1e-20
        opts.enabled = False
        assert not opts.enabled

    def test_factory_none(self):
        from libaccint import ScreeningOptions
        opts = ScreeningOptions.none()
        assert not opts.enabled

    def test_factory_loose(self):
        from libaccint import ScreeningOptions
        opts = ScreeningOptions.loose()
        assert opts.enabled
        assert opts.threshold > 1e-12  # Loose is less strict

    def test_factory_normal(self):
        from libaccint import ScreeningOptions
        opts = ScreeningOptions.normal()
        assert opts.enabled

    def test_factory_tight(self):
        from libaccint import ScreeningOptions
        opts = ScreeningOptions.tight()
        assert opts.enabled
        assert opts.threshold < 1e-12  # Tight is more strict

    def test_from_preset(self):
        from libaccint import ScreeningOptions, ScreeningPreset
        opts = ScreeningOptions.from_preset(ScreeningPreset.TIGHT)
        assert opts.enabled

    def test_preset_name(self):
        from libaccint import ScreeningOptions
        opts = ScreeningOptions.normal()
        name = opts.preset_name()
        assert isinstance(name, str)

    def test_repr(self):
        from libaccint import ScreeningOptions
        opts = ScreeningOptions()
        r = repr(opts)
        assert 'ScreeningOptions' in r


class TestScreeningStatistics:
    """Tests for ScreeningStatistics struct."""

    def test_default_construction(self):
        from libaccint import ScreeningStatistics
        stats = ScreeningStatistics()
        assert stats.total_quartets == 0
        assert stats.computed_quartets == 0
        assert stats.skipped_quartets == 0

    def test_field_readwrite(self):
        from libaccint import ScreeningStatistics
        stats = ScreeningStatistics()
        stats.total_quartets = 100
        stats.computed_quartets = 80
        stats.skipped_quartets = 20
        assert stats.total_quartets == 100
        assert stats.computed_quartets == 80
        assert stats.skipped_quartets == 20

    def test_efficiency(self):
        from libaccint import ScreeningStatistics
        stats = ScreeningStatistics()
        stats.total_quartets = 100
        stats.computed_quartets = 80
        stats.skipped_quartets = 20
        eff = stats.efficiency()
        # efficiency = skipped / total
        assert abs(eff - 0.2) < 1e-10

    def test_skip_percentage(self):
        from libaccint import ScreeningStatistics
        stats = ScreeningStatistics()
        stats.total_quartets = 100
        stats.computed_quartets = 80
        stats.skipped_quartets = 20
        pct = stats.skip_percentage()
        assert abs(pct - 20.0) < 1e-10

    def test_reset(self):
        from libaccint import ScreeningStatistics
        stats = ScreeningStatistics()
        stats.total_quartets = 50
        stats.reset()
        assert stats.total_quartets == 0

    def test_repr(self):
        from libaccint import ScreeningStatistics
        stats = ScreeningStatistics()
        r = repr(stats)
        assert 'ScreeningStatistics' in r


class TestSchwarzBounds:
    """Tests for SchwarzBounds class."""

    @pytest.fixture
    def h2_basis(self):
        from libaccint import Atom, create_builtin_basis
        atoms = [Atom(1, [0, 0, 0]), Atom(1, [0, 0, 1.4])]
        return create_builtin_basis("sto-3g", atoms)

    def test_construction(self, h2_basis):
        from libaccint import SchwarzBounds
        bounds = SchwarzBounds(h2_basis)
        assert bounds.n_shells() == h2_basis.n_shells()

    def test_call_returns_positive(self, h2_basis):
        from libaccint import SchwarzBounds
        bounds = SchwarzBounds(h2_basis)
        val = bounds(0, 0)
        assert val > 0.0

    def test_passes_screening(self, h2_basis):
        from libaccint import SchwarzBounds
        bounds = SchwarzBounds(h2_basis)
        # Very loose threshold should pass
        assert bounds.passes_screening(0, 0, 0, 0, 1e-20)
        # Very tight threshold may not pass
        result = bounds.passes_screening(0, 0, 0, 0, 1e100)
        assert isinstance(result, bool)

    def test_quartet_bound(self, h2_basis):
        from libaccint import SchwarzBounds
        bounds = SchwarzBounds(h2_basis)
        qb = bounds.quartet_bound(0, 0, 0, 0)
        assert qb > 0.0

    def test_max_bound(self, h2_basis):
        from libaccint import SchwarzBounds
        bounds = SchwarzBounds(h2_basis)
        mb = bounds.max_bound()
        assert mb > 0.0

    def test_count_passing_quartets(self, h2_basis):
        from libaccint import SchwarzBounds
        bounds = SchwarzBounds(h2_basis)
        n = bounds.count_passing_quartets(1e-12)
        assert n > 0

    def test_estimate_pass_fraction(self, h2_basis):
        from libaccint import SchwarzBounds
        bounds = SchwarzBounds(h2_basis)
        frac = bounds.estimate_pass_fraction(1e-12)
        assert 0.0 <= frac <= 1.0

    def test_repr(self, h2_basis):
        from libaccint import SchwarzBounds
        bounds = SchwarzBounds(h2_basis)
        r = repr(bounds)
        assert 'SchwarzBounds' in r
