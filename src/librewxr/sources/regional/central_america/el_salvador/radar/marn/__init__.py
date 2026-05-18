# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Joshua Kimsey
"""El Salvador MARN/SNET radar — self-contained source package.

Single S-band radar at San Andrés volcano serving the ``SVCOMP``
region in the ``CENTRAL_AMERICA`` group.  Anonymous GCS bucket
``radar-images-sv``; 5-min cadence; HSV-gradient PNG decode.

Discovered automatically by ``librewxr.sources``; ``radar_provider``
wires the source into the fetcher, and ``REGIONS`` / ``REGION_GROUP``
feed ``librewxr.data.regions``.
"""
from __future__ import annotations

from librewxr.sources._base import RadarSourceContribution

from .regions import REGIONS, SVCOMP
from .source import (
    MARNSource,
    _decode_marn_png,
    _MARN_DBZ_MAX,
    _MARN_DBZ_MIN,
    _MARN_HUE_MAX,
    _MARN_HUE_MIN,
)
from .stations import RANGE_OVERRIDES, STATION_MAP, STATIONS

REGION_GROUP = "CENTRAL_AMERICA"

__all__ = [
    "MARNSource",
    "RANGE_OVERRIDES",
    "REGIONS",
    "REGION_GROUP",
    "STATIONS",
    "STATION_MAP",
    "SVCOMP",
    "_MARN_DBZ_MAX",
    "_MARN_DBZ_MIN",
    "_MARN_HUE_MAX",
    "_MARN_HUE_MIN",
    "_decode_marn_png",
    "radar_provider",
]


def radar_provider(settings) -> RadarSourceContribution | None:
    """Return a MARN El Salvador contribution.

    Like CWA, MARN has no enable toggle — user controls inclusion via
    region selection (``LIBREWXR_ENABLED_REGIONS=CENTRAL_AMERICA`` or
    similar).  The fetcher only assigns the source to ``SVCOMP`` when
    the user actually enabled it, so eager instantiation here is cheap
    (HTTP client opens lazily on first fetch).
    """
    instance = MARNSource(settings.marn_base_url)
    return RadarSourceContribution(
        regions=list(REGIONS),
        instance=instance,
        group=REGION_GROUP,
        station_map={k: list(v) for k, v in STATION_MAP.items()},
        range_overrides=dict(RANGE_OVERRIDES),
    )
