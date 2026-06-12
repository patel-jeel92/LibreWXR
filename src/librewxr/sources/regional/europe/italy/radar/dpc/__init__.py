# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Joshua Kimsey
"""DPC Italian national radar composite — self-contained source package.

Single region ``ITCOMP`` in the ``EUROPE`` group, fetched as GeoTIFF
from the open Radar-DPC v2 REST API (``radar-api.protezionecivile.it``).
5-min cadence, no auth, CC-BY-SA 4.0.

Italy is not in the EUMETNET OPERA station list, so what the
pan-European OPERA layer shows over Italian airspace today is
edge-of-range data from neighbouring countries (France Côte d'Azur,
Switzerland, Slovenia, …) — visible as residual clutter.  ITCOMP
sits below OPERA in ``pixel_size`` (0.009 vs 0.01) so it wins the
multi-region compositor's "finest resolution first" sort over Italy,
while OPERA continues to cover the rest of the European group.

Discovered automatically by ``librewxr.sources``.
"""
from __future__ import annotations

from librewxr.sources._base import RadarSourceContribution

from .regions import ITCOMP, REGIONS
from .source import DPCSource, _decode_vmi_dbz
from .stations import (
    COVERAGE_POLYGONS,
    RANGE_OVERRIDES,
    STATION_MAP,
    STATIONS,
)

REGION_GROUP = "EUROPE"

__all__ = [
    "COVERAGE_POLYGONS",
    "DPCSource",
    "ITCOMP",
    "RANGE_OVERRIDES",
    "REGIONS",
    "REGION_GROUP",
    "STATIONS",
    "STATION_MAP",
    "_decode_vmi_dbz",
    "radar_provider",
]


def radar_provider(settings) -> RadarSourceContribution | None:
    """Return a DPC Italy contribution unless explicitly disabled.

    Like every shipping radar source, defaults to enabled — operator
    opts out via ``LIBREWXR_DPC_ENABLED=false`` or by removing
    ``ITCOMP`` / the ``EUROPE`` group from
    ``LIBREWXR_ENABLED_REGIONS``.
    """
    if not getattr(settings, "dpc_enabled", True):
        return None
    instance = DPCSource(settings.dpc_base_url)
    return RadarSourceContribution(
        regions=list(REGIONS),
        instance=instance,
        group=REGION_GROUP,
        station_map={k: list(v) for k, v in STATION_MAP.items()},
        range_overrides=dict(RANGE_OVERRIDES),
        coverage_polygons={
            k: [list(ring) for ring in v]
            for k, v in COVERAGE_POLYGONS.items()
        },
    )
