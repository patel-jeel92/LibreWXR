# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Joshua Kimsey
"""NOAA MRMS — self-contained source package.

Serves USCOMP / AKCOMP / HICOMP / PRCOMP / GUCOMP (plus CACOMP via
the CONUS product) from the NCEP MRMS GRIB2 endpoint when
``settings.na_source`` is ``mrms`` or ``mrms_fallback``.  When
``na_source == "iem"``, IEM owns the slots and the MRMS provider
returns ``None``.

US regions and the IEM-flavored station inventory live one level up at
``sources/regional/north_america/usa/radar/`` because they're shared
with IEM.  MRMS contributes its own multi-product source, the
``MRMS_PRODUCTS`` / ``MRMS_EXTENTS`` tables, and the NEXRAD+Canadian
station combination it actually ingests.
"""
from __future__ import annotations

from librewxr.sources._base import RadarSourceContribution

from ..regions import REGIONS as USA_REGIONS
from .products import MRMS_EXTENTS, MRMS_PRODUCTS
from .source import (
    MRMSCompositeSource,
    MRMSSource,
    _parse_mrms_grib2,
    _resample_mrms_to_region,
    _suppress_eccodes_stderr,
)
from .stations import STATION_MAP

__all__ = [
    "MRMSCompositeSource",
    "MRMSSource",
    "MRMS_EXTENTS",
    "MRMS_PRODUCTS",
    "STATION_MAP",
    "_parse_mrms_grib2",
    "_resample_mrms_to_region",
    "_suppress_eccodes_stderr",
    "radar_provider",
]


def radar_provider(settings) -> RadarSourceContribution | None:
    """Return an MRMS contribution when MRMS is the primary NA source.

    Active when ``settings.na_source`` is ``mrms`` or ``mrms_fallback``.
    In ``iem`` mode this returns ``None`` and the IEM provider owns
    the US-group slots; the cross-source ``_iem_fallback`` /
    ``_blend_cacomp`` flows in ``data/fetcher.py`` are not gated by
    provider state.

    A single ``MRMSCompositeSource`` covers all six US-group regions;
    inside that wrapper, per-product ``MRMSSource`` instances are
    created lazily on first request.
    """
    if getattr(settings, "na_source", "mrms_fallback") not in (
        "mrms", "mrms_fallback"
    ):
        return None
    instance = MRMSCompositeSource(settings.mrms_base_url)
    return RadarSourceContribution(
        regions=list(USA_REGIONS),
        instance=instance,
        group="US",
        station_map={k: list(v) for k, v in STATION_MAP.items()},
    )
