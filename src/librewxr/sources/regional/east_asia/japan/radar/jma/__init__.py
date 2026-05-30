# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Joshua Kimsey
"""JMA HRPN (High-Resolution Precipitation Nowcast) composite.

Provides two contributions from one source package:

- ``radar_provider()`` — analysis leg (N1 manifest, basetime==validtime),
  radar + AMeDAS gauge composite QPE.  Treated as standard radar source.
- ``nowcast_provider()`` — forecast leg (N2 manifest, validtime>basetime),
  JMA's own model-extrapolated nowcast out to 60 minutes.  Bypasses
  LibreWXR's internal optical-flow extrapolation for JPCOMP only.

Both legs share one ``JMAFetcher`` instance internally so the manifest
and tile caches are not duplicated.

Endpoint: ``https://www.jma.go.jp/bosai/jmatile/data/nowc/`` — anonymous,
S3-backed CDN, no auth, no WAF.

Licence: JMA Public Data License v1.0 (CC-BY equivalent, commercial
reuse explicitly permitted with attribution).  Caveat: Article 17 of
Japan's Meteorological Service Act restricts "provision of meteorological
services in Japan" — does not apply to LibreWXR redistributing globally,
but worth noting for any operator running a Japan-domestic service.
"""
from __future__ import annotations

from librewxr.sources._base import (
    NowcastContribution,
    RadarSourceContribution,
)

from .regions import JPCOMP, REGIONS, REGION_GROUP
from .source import JMAAnalysisSource, JMAFetcher, JMANowcastSource
from .stations import RANGE_OVERRIDES, STATION_MAP


# Module-level singleton fetcher so analysis and nowcast contributions
# share manifest + tile caches.  Lazily created on first provider call.
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
            zoom=getattr(settings, "jma_zoom", 7),
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


def nowcast_provider(settings) -> NowcastContribution | None:
    """Return the JMA forecast-leg nowcast contribution, or None if disabled."""
    if not getattr(settings, "jma_enabled", True):
        return None
    if not getattr(settings, "jma_nowcast_enabled", True):
        return None
    fetcher = _get_or_create_fetcher(settings)
    return NowcastContribution(
        region_name=JPCOMP.name,
        instance=JMANowcastSource(fetcher),
        horizon_minutes=60,
    )
