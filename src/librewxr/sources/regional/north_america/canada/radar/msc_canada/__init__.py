# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Joshua Kimsey
"""MSC Canada (ECCC GeoMet) radar composite — self-contained source package.

Single region (``CACOMP``) in the ``CANADA`` group, fetched from MSC's
GeoMet WMS as a pre-coloured 14-bucket PNG (no raw-radar access from
ECCC).  Decoder reverses the discrete palette back to mm/h, then maps to
dBZ via Marshall-Palmer.

Discovered automatically by ``librewxr.sources``; ``radar_provider``
wires the source into the fetcher.

MSC is also consumed *outside* the discovery path — ``data/fetcher.py``
imports ``MSCCanadaSource`` directly for the CACOMP MRMS-blend and the
US-group MRMS fallback (``_blend_cacomp`` / ``_try_fallback``).  Those
cross-source helpers stay in fetcher.py per the plan; Phase 4 may
factor them into a ``_resolve_group_policy`` helper.
"""
from __future__ import annotations

from librewxr.sources._base import RadarSourceContribution

from .regions import CACOMP, REGIONS
from .source import (
    MSCCanadaSource,
    _decode_msc_canada_png,
    _mmhr_to_dbz,
    _MSC_CANADA_MAX_RGB_DIST,
    _MSC_CANADA_PALETTE,
)
from .stations import STATION_MAP, STATIONS

REGION_GROUP = "CANADA"

__all__ = [
    "CACOMP",
    "MSCCanadaSource",
    "REGIONS",
    "REGION_GROUP",
    "STATIONS",
    "STATION_MAP",
    "_MSC_CANADA_MAX_RGB_DIST",
    "_MSC_CANADA_PALETTE",
    "_decode_msc_canada_png",
    "_mmhr_to_dbz",
    "radar_provider",
]


def radar_provider(settings) -> RadarSourceContribution | None:
    """Return an MSC Canada contribution.

    Like CWA / MARN / OPERA, MSC Canada has no enable toggle — user
    controls inclusion via region selection
    (``LIBREWXR_ENABLED_REGIONS=CANADA`` or similar).  The fetcher only
    assigns the source to ``CACOMP`` when the user actually enabled it,
    so eager instantiation here is cheap (HTTP client opens lazily on
    first fetch).
    """
    instance = MSCCanadaSource(settings.msc_canada_base_url)
    return RadarSourceContribution(
        regions=list(REGIONS),
        instance=instance,
        group=REGION_GROUP,
        station_map={k: list(v) for k, v in STATION_MAP.items()},
    )
