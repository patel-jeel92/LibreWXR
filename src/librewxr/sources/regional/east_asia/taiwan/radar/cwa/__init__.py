# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Joshua Kimsey
"""Taiwan CWA QPESUMS composite — self-contained source package.

Single region (``TWCOMP``) in the ``TAIWAN`` group, fetched from
anonymous AWS S3 (``cwaopendata`` in ``ap-northeast-1``).  10-min XML
frames at ~9 MB each.

Discovered automatically by ``librewxr.sources``; ``radar_provider``
wires the source into the fetcher, and ``REGIONS`` /  ``REGION_GROUP``
feed ``librewxr.data.regions``.
"""
from __future__ import annotations

from librewxr.sources._base import RadarSourceContribution

from .regions import REGIONS, TWCOMP
from .source import CWASource, _parse_cwa_xml
from .stations import RANGE_OVERRIDES, STATION_MAP, STATIONS

REGION_GROUP = "TAIWAN"

__all__ = [
    "CWASource",
    "RANGE_OVERRIDES",
    "REGIONS",
    "REGION_GROUP",
    "STATIONS",
    "STATION_MAP",
    "TWCOMP",
    "_parse_cwa_xml",
    "radar_provider",
]


def radar_provider(settings) -> RadarSourceContribution | None:
    """Return a CWA Taiwan contribution.

    CWA has no enable toggle — the user controls inclusion via region
    selection (``LIBREWXR_ENABLED_REGIONS=TAIWAN`` or similar).  The
    fetcher only wires the source to ``TWCOMP`` when the user actually
    enabled that region, so the eager instantiation here is cheap
    (HTTP client opens lazily on first fetch).
    """
    instance = CWASource(settings.cwa_base_url)
    return RadarSourceContribution(
        regions=list(REGIONS),
        instance=instance,
        group=REGION_GROUP,
        station_map={k: list(v) for k, v in STATION_MAP.items()},
        range_overrides=dict(RANGE_OVERRIDES),
    )
