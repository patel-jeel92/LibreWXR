# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Joshua Kimsey
"""USA radar — shared region + station inventory for IEM and MRMS.

The discovery walker picks up ``REGIONS`` and ``REGION_GROUP`` from
this package (not from the per-source sub-packages) so the 5 US
NEXRAD-fed regions are registered once, regardless of which source is
active.  Per-source ``radar_provider`` functions live in ``iem/`` and
``mrms/``.
"""
from __future__ import annotations

from .regions import AKCOMP, GUCOMP, HICOMP, PRCOMP, REGIONS, USCOMP
from .stations import (
    NEXRAD_ALASKA,
    NEXRAD_CONUS,
    NEXRAD_GUAM,
    NEXRAD_HAWAII,
    NEXRAD_PUERTO_RICO,
    STATION_MAP,
)

REGION_GROUP = "US"

__all__ = [
    "AKCOMP",
    "GUCOMP",
    "HICOMP",
    "NEXRAD_ALASKA",
    "NEXRAD_CONUS",
    "NEXRAD_GUAM",
    "NEXRAD_HAWAII",
    "NEXRAD_PUERTO_RICO",
    "PRCOMP",
    "REGIONS",
    "REGION_GROUP",
    "STATION_MAP",
    "USCOMP",
]
