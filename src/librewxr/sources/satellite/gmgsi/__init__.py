# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Joshua Kimsey
"""NOAA GMGSI — self-contained satellite source package.

Hourly pre-composited global mosaic (LW + VIS) from NESDIS, backing
the existing Rain Viewer-compatible ``/v2/satellite/...`` endpoint via
a VIS-over-LW composite renderer.  See ``docs/satellite-implementation-
plan.md`` for the full plan.

Phase 1 ships LW only; VIS is added in Phase 2.
"""
from __future__ import annotations

from librewxr.sources._base import SatelliteContribution

from .source import GMGSILWSource, GMGSIVISSource

__all__ = ["GMGSILWSource", "GMGSIVISSource", "satellite_provider"]


def satellite_provider(settings, cache_dir) -> list[SatelliteContribution]:
    """Return one ``SatelliteContribution`` per enabled GMGSI channel.

    The collector in ``librewxr.sources.__init__`` flattens the list
    into the global satellite-contribution registry.  Returning ``[]``
    when every channel is disabled is equivalent to returning ``None``.

    Both channels are ingested independently; the composite renderer
    in ``tiles/satellite_renderer.py`` blends them at render time.
    Disabling VIS while LW stays on degrades the composite to LW-only.
    """
    contributions: list[SatelliteContribution] = []
    retention = getattr(settings, "satellite_max_frames", 12)

    if getattr(settings, "gmgsi_lw_enabled", True):
        contributions.append(
            SatelliteContribution(
                instance=GMGSILWSource(cache_dir=cache_dir, max_frames=retention),
                priority=10,
                name="GMGSI LW",
                slug="gmgsi_lw_grid",
            ),
        )

    if getattr(settings, "gmgsi_vis_enabled", True):
        contributions.append(
            SatelliteContribution(
                instance=GMGSIVISSource(cache_dir=cache_dir, max_frames=retention),
                priority=11,
                name="GMGSI VIS",
                slug="gmgsi_vis_grid",
            ),
        )

    return contributions
