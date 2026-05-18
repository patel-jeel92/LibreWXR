# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Joshua Kimsey
"""MET Malaysia radar composite — self-contained source package.

Covers Peninsular Malaysia + East Malaysia (Borneo) + Brunei + Singapore
+ N. Sumatra via a single combined animated GIF served anonymously from
``api.met.gov.my`` under CC-BY-4.0.  Two regions ride on one HTTP fetch
per cycle: ``MYPENINSULAR`` and ``MYEAST``, both in the
``SOUTHEAST_ASIA`` group.

This package is auto-discovered by ``librewxr.sources`` at import time;
the ``radar_provider`` function below is what wires the source into the
fetcher.  ``REGIONS`` and ``REGION_GROUP`` are picked up by
``librewxr.data.regions`` to populate the global region map.
"""
from __future__ import annotations

from librewxr.sources._base import RadarSourceContribution

from .regions import MYEAST, MYPENINSULAR, REGIONS
from .source import MMDSource
from .stations import EAST_STATIONS, PENINSULAR_STATIONS, RANGE_OVERRIDES, STATION_MAP

# Discovery hooks — see librewxr.data.regions._merge_discovered_regions.
REGION_GROUP = "SOUTHEAST_ASIA"

__all__ = [
    "EAST_STATIONS",
    "MYEAST",
    "MYPENINSULAR",
    "MMDSource",
    "PENINSULAR_STATIONS",
    "RANGE_OVERRIDES",
    "REGIONS",
    "REGION_GROUP",
    "STATION_MAP",
    "radar_provider",
]


def radar_provider(settings) -> RadarSourceContribution | None:
    """Return an MMD contribution, or ``None`` when disabled.

    Honours ``settings.mmd_enabled`` (default ``True``).  When disabled,
    the fetcher sees no source for ``MYPENINSULAR`` / ``MYEAST`` and
    drops both regions from its working set even if a user's region
    spec is a group alias (e.g. ``SOUTHEAST_ASIA``, ``ALL``) that would
    otherwise pull them in.
    """
    if not getattr(settings, "mmd_enabled", True):
        return None
    instance = MMDSource(
        settings.mmd_base_url,
        publish_lag_sec=settings.mmd_publish_lag_sec,
    )
    return RadarSourceContribution(
        regions=list(REGIONS),
        instance=instance,
        group=REGION_GROUP,
        station_map={k: list(v) for k, v in STATION_MAP.items()},
        range_overrides=dict(RANGE_OVERRIDES),
    )
