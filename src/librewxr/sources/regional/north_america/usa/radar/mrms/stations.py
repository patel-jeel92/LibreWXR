# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Joshua Kimsey
"""MRMS contributing radar inventory.

MRMS ingests both the NEXRAD network (US) and the ECCC Canadian network
(into the CONUS product), so coverage masks for USCOMP/CACOMP combine
both networks.  AKCOMP / HICOMP / PRCOMP / GUCOMP are NEXRAD-only.
"""
from __future__ import annotations

from librewxr.sources.regional.north_america.canada.radar.msc_canada.stations import (
    STATIONS as _CANADA_STATIONS,
)
from librewxr.sources.regional.north_america.usa.radar.stations import (
    NEXRAD_ALASKA,
    NEXRAD_CONUS,
    NEXRAD_GUAM,
    NEXRAD_HAWAII,
    NEXRAD_PUERTO_RICO,
)


# Per-region station lists for coverage masks when MRMS is the active
# source.  USCOMP and CACOMP share the combined NEXRAD + Canadian list
# because the MRMS CONUS product spans both networks.
STATION_MAP: dict[str, list[tuple[float, float]]] = {
    "USCOMP": NEXRAD_CONUS + _CANADA_STATIONS,
    "CACOMP": NEXRAD_CONUS + _CANADA_STATIONS,
    "AKCOMP": NEXRAD_ALASKA,
    "HICOMP": NEXRAD_HAWAII,
    "PRCOMP": NEXRAD_PUERTO_RICO,
    "GUCOMP": NEXRAD_GUAM,
}
