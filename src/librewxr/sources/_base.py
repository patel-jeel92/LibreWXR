# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Joshua Kimsey
"""Protocols and contribution dataclasses for the sources registry.

Each source package under ``librewxr.sources`` exports a provider
function (``radar_provider`` or ``nwp_provider``) returning one of the
contribution dataclasses below.  The discovery walker in
``librewxr.sources.__init__`` collects those providers; ``fetcher.py``
and ``main.py`` call them to assemble the actual radar source map and
NWP chain at startup.

Source-specific config still lives in ``librewxr.config.settings`` and
is passed into each provider — Phase 0 of the refactor (2026-05-17)
deliberately kept config centralized to avoid Pydantic gymnastics.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Protocol, runtime_checkable

import numpy as np

from librewxr.data.regions import RegionDef


@runtime_checkable
class RadarSource(Protocol):
    """Shape every radar source class must satisfy.

    Concrete implementations are duck-typed against this — they don't
    need to inherit from it.  ``runtime_checkable`` is included so
    tests can ``isinstance``-check without imports.
    """

    async def fetch_frame(
        self, region: RegionDef
    ) -> tuple[datetime, np.ndarray] | None: ...

    async def fetch_archive_frame(
        self, region: RegionDef, when: datetime
    ) -> tuple[datetime, np.ndarray] | None: ...

    async def close(self) -> None: ...


@runtime_checkable
class NWPGrid(Protocol):
    """Shape every NWP grid class must satisfy.

    ``fetch()`` signature varies by grid (different return types, kwargs,
    timestep semantics) so it's intentionally not declared here — the
    NWPChain in ``data/fetcher.py`` knows what to call.  This protocol
    only pins down the universally-required parts.
    """

    name: str

    async def close(self) -> None: ...


@runtime_checkable
class SatelliteSource(Protocol):
    """Shape every satellite source class must satisfy.

    A satellite source ingests one channel of imagery (e.g. GMGSI LW or
    GMGSI VIS) on its own cadence, stores frames internally, and lets
    the renderer sample lat/lon grids out of any stored timestamp.

    Same ``fetch`` ergonomics as NWPGrid: signature varies per source
    (channel-specific kwargs, native cadence), so it's declared by the
    concrete class.  The pipeline introspects the signature via
    ``inspect.signature`` when dispatching, the same way it does for
    NWPGrid implementations.
    """

    name: str
    timestamps: list[int]

    async def close(self) -> None: ...


@dataclass
class RadarSourceContribution:
    """Return value from a source package's ``radar_provider(settings)``.

    A single contribution may cover multiple regions (e.g. MMD's one
    instance serves both MYPENINSULAR and MYEAST from one HTTP fetch).

    ``station_map`` is keyed by region name and feeds the coverage-mask
    builder in ``data/coverage.py``.  Regions without a station list get
    no mask (full-region coverage is assumed).  ``range_overrides``
    likewise feeds the mask builder — any region missing here uses the
    240 km default Doppler reach.
    """

    regions: list[RegionDef]
    instance: RadarSource
    group: str
    preempts: tuple[str, ...] = ()
    station_map: dict[str, list[tuple[float, float]]] = field(default_factory=dict)
    range_overrides: dict[str, float] = field(default_factory=dict)


@dataclass
class NWPContribution:
    """Return value from a source package's ``nwp_provider(settings, cache_dir)``.

    ``priority`` controls position in the NWPChain: lower runs earlier,
    so narrower / higher-resolution domains should use lower numbers.

    ``slug`` overrides the auto-generated snapshot / ``/health`` key.
    Leave it ``None`` to derive the key from ``name`` (lowercase,
    non-alphanumerics replaced with underscores, ``_grid`` suffixed).
    Three sources opt out: IFS (legacy key drops the "_ifs"), and the
    two AROME-OM variants whose display names include non-ASCII
    characters that wouldn't round-trip through the slugger.

    ``regional`` flags the contribution as part of the regional NWP
    chain — when ``regional_nwp_enabled`` is False, the central
    collector drops every regional contribution and lets IFS carry
    the chain alone.  Defaults to True so new sources don't have to
    opt in to the gate.  IFS itself sets ``regional=False``.
    """

    instance: NWPGrid
    priority: int
    name: str
    slug: str | None = None
    regional: bool = True


@dataclass
class SatelliteContribution:
    """Return value from a source package's ``satellite_provider(settings, cache_dir)``.

    One contribution per channel: a multi-channel source family (like
    GMGSI with LW + VIS) returns multiple contributions, each carrying
    its own instance and slug.  The renderer references them by slug
    when assembling composites.

    ``priority`` is reserved for the day a second satellite-mosaic
    source appears (none planned).  Lower number wins for any pixel
    both cover.  Single-source today, so the field is mostly
    ceremonial.

    ``slug`` overrides the auto-generated snapshot / ``/health`` key.
    Leave it ``None`` to derive the key from ``name`` (lowercase,
    non-alphanumerics replaced with underscores, ``_grid`` suffixed).
    """

    instance: SatelliteSource
    priority: int
    name: str
    slug: str | None = None
