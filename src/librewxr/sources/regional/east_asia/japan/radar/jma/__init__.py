# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Joshua Kimsey
"""JMA HRPN (High-Resolution Precipitation Nowcast) composite.

Analysis-leg only: ``radar_provider()`` returns the N1 manifest frames
(basetime==validtime, radar + AMeDAS gauge composite QPE) as a standard
``RadarSourceContribution``.  JPCOMP nowcast frames are produced by
LibreWXR's internal optical-flow extrapolation, same as every other
region — the JMA N2 forecast leg was tried and removed because the
5-min validtime cadence didn't fit cleanly into the 10-min sampling
rhythm and the dispatch complexity wasn't worth the win for one
region.  Future re-attempt should probably treat it as a regional NWP
overlay instead, or pair with a Japanese mesoscale NWP model.

Endpoint: ``https://www.jma.go.jp/bosai/jmatile/data/nowc/`` — anonymous,
S3-backed CDN, no auth, no WAF.

Licence: JMA Public Data License v1.0 (CC-BY equivalent, commercial
reuse explicitly permitted with attribution).  Caveat: Article 17 of
Japan's Meteorological Service Act restricts "provision of meteorological
services in Japan" — does not apply to LibreWXR redistributing globally,
but worth noting for any operator running a Japan-domestic service.
"""
from __future__ import annotations

from librewxr.sources._base import RadarSourceContribution

from .regions import REGIONS, REGION_GROUP
from .source import JMAAnalysisSource, JMAFetcher
from .stations import STATION_MAP


# Module-level singleton fetcher.  Kept as a singleton in case a future
# nowcast / forecast leg is re-added — sharing the manifest and tile
# caches across legs avoids duplicate work.
_shared_fetcher: JMAFetcher | None = None


def _get_or_create_fetcher(settings) -> JMAFetcher:
    global _shared_fetcher
    if _shared_fetcher is None:
        _shared_fetcher = JMAFetcher(
            base_url=getattr(
                settings,
                "jma_base_url",
                "https://www.jma.go.jp/bosai/jmatile/data/nowc",
            ),
            zoom=getattr(settings, "jma_zoom", 8),
        )
    return _shared_fetcher


def radar_provider(settings) -> RadarSourceContribution | None:
    """Return the JMA analysis-leg radar contribution, or None if disabled."""
    if not getattr(settings, "jma_enabled", True):
        return None
    fetcher = _get_or_create_fetcher(settings)
    return RadarSourceContribution(
        regions=REGIONS,
        instance=JMAAnalysisSource(fetcher),
        group=REGION_GROUP,
        station_map=STATION_MAP,
        range_overrides={},
    )
