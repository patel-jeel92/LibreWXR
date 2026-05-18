# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Joshua Kimsey
"""IEM NEXRAD N0Q composite — self-contained source package.

Serves USCOMP / AKCOMP / HICOMP / PRCOMP / GUCOMP from IEM's image
endpoints when ``settings.na_source == "iem"`` (the legacy single-source
North America profile).  When ``na_source`` is ``mrms`` or
``mrms_fallback``, MRMS owns the slot and IEM is reduced to a fallback
role wired directly in ``data/fetcher.py`` (``_iem_fallback``) — the
provider here returns ``None`` to keep ``self._sources`` clean.

The 5 US regions and the NEXRAD station list live one level up at
``sources/regional/north_america/usa/radar/`` so MRMS can share them.
"""
from __future__ import annotations

from librewxr.sources._base import RadarSourceContribution

from ..regions import REGIONS as USA_REGIONS
from ..stations import STATION_MAP as USA_STATION_MAP
from .source import IEMSource, _parse_n0q_png

__all__ = [
    "IEMSource",
    "_parse_n0q_png",
    "radar_provider",
]


def radar_provider(settings) -> RadarSourceContribution | None:
    """Return an IEM contribution when ``na_source == "iem"``.

    In MRMS modes (``mrms`` / ``mrms_fallback``), MRMS is the primary
    North America source and IEM is either unused or reached only via
    ``_iem_fallback`` in ``data/fetcher.py``.  Returning ``None`` here
    keeps the discovery loop from clobbering MRMS's slot.
    """
    if getattr(settings, "na_source", "mrms_fallback") != "iem":
        return None
    instance = IEMSource(settings.iem_base_url)
    return RadarSourceContribution(
        regions=list(USA_REGIONS),
        instance=instance,
        group="US",
        station_map={k: list(v) for k, v in USA_STATION_MAP.items()},
    )
