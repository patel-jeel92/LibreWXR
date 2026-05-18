# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Joshua Kimsey
"""Tests for the source-discovery scaffolding (Phase 0+ of the sources
refactor — see ``.claude-context/sources-refactor-plan.md``).

The walker, protocols, and contribution dataclasses must be importable
and the wiring into ``data.regions`` / ``data.fetcher`` must not break
anything as source packages migrate one by one.
"""
from __future__ import annotations

import pytest

pytestmark = pytest.mark.sources


def test_iter_source_packages_returns_at_least_the_subtrees():
    """The two top-level subtrees (``world`` and ``regional``) must both
    be discovered.  As sources migrate, deeper packages join them."""
    from librewxr.sources import iter_source_packages

    pkg_names = {mod.__name__ for mod in iter_source_packages()}
    assert "librewxr.sources.world" in pkg_names
    assert "librewxr.sources.regional" in pkg_names


def test_radar_providers_are_callables():
    """Every entry in ``RADAR_PROVIDERS`` must be a callable so the
    fetcher can invoke it with ``settings`` and get a contribution back
    (or ``None``)."""
    from librewxr.sources import RADAR_PROVIDERS

    assert all(callable(p) for p in RADAR_PROVIDERS)


def test_mmd_provider_is_registered():
    """Phase 1 step 3: MET Malaysia is the first source on the registry
    path.  When ``mmd_enabled`` is True (default) the provider should
    return a contribution covering both Malaysia regions."""
    from librewxr.config import settings
    from librewxr.sources import RADAR_PROVIDERS
    from librewxr.sources._base import RadarSourceContribution

    contribs = [p(settings) for p in RADAR_PROVIDERS]
    mmd = [
        c for c in contribs
        if isinstance(c, RadarSourceContribution) and c.group == "SOUTHEAST_ASIA"
    ]
    assert len(mmd) == 1, "expected exactly one SOUTHEAST_ASIA contribution"
    region_names = {r.name for r in mmd[0].regions}
    assert region_names == {"MYPENINSULAR", "MYEAST"}


def test_protocols_and_contribution_dataclasses_importable():
    from librewxr.sources._base import (
        NWPContribution,
        NWPGrid,
        RadarSource,
        RadarSourceContribution,
    )

    # Smoke-construct the contribution dataclasses with minimal args to
    # confirm the field shapes match the plan.
    radar = RadarSourceContribution(regions=[], instance=None, group="X")  # type: ignore[arg-type]
    assert radar.regions == []
    assert radar.preempts == ()
    assert radar.station_map == {}
    assert radar.range_overrides == {}

    nwp = NWPContribution(instance=None, priority=10, name="X")  # type: ignore[arg-type]
    assert nwp.priority == 10

    # Protocols themselves should be importable + runtime_checkable.
    assert hasattr(RadarSource, "__protocol_attrs__") or hasattr(
        RadarSource, "_is_runtime_protocol"
    )
    assert hasattr(NWPGrid, "__protocol_attrs__") or hasattr(
        NWPGrid, "_is_runtime_protocol"
    )


def test_regions_module_imports_cleanly_with_discovery_wired():
    """``data.regions._merge_discovered_regions()`` runs at import time;
    in Phase 0 it has nothing to merge but must not raise."""
    from librewxr.data import regions

    # Existing hand-defined regions are still present (proving the
    # merge didn't clobber them).
    assert "USCOMP" in regions.REGIONS
    assert "OPERA" in regions.REGIONS
    assert "MYPENINSULAR" in regions.REGIONS


def test_fetcher_sees_registry_providers():
    """RadarFetcher.__init__ iterates ``RADAR_PROVIDERS`` to populate
    ``self._sources``.  As migrations land, the registry grows; at
    minimum the MMD provider should be visible from the fetcher's
    import path."""
    from librewxr.data.fetcher import RADAR_PROVIDERS

    assert len(RADAR_PROVIDERS) >= 1
    assert all(callable(p) for p in RADAR_PROVIDERS)
