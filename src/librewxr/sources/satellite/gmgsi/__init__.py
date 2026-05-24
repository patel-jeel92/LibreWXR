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

from .source import GMGSILWSource

__all__ = ["GMGSILWSource", "satellite_provider"]


def satellite_provider(settings, cache_dir) -> list[SatelliteContribution]:
    """Return one ``SatelliteContribution`` per enabled GMGSI channel.

    The collector in ``librewxr.sources.__init__`` flattens the list
    into the global satellite-contribution registry.  Returning ``[]``
    when every channel is disabled is equivalent to returning ``None``.

    Phase 1 has LW only; VIS lands in Phase 2 and will join this list
    behind its own ``gmgsi_vis_enabled`` toggle.
    """
    contributions: list[SatelliteContribution] = []

    if getattr(settings, "gmgsi_lw_enabled", True):
        retention = getattr(settings, "gmgsi_retention_hours", 12)
        contributions.append(
            SatelliteContribution(
                instance=GMGSILWSource(cache_dir=cache_dir, max_frames=retention),
                priority=10,
                name="GMGSI LW",
                slug="gmgsi_lw_grid",
            ),
        )

    return contributions
