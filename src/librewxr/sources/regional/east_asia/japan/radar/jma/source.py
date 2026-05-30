# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Joshua Kimsey
"""JMA HRPN (High-Resolution Precipitation Nowcast) source classes.

Three classes:

- ``JMAFetcher`` — shared base with httpx client, manifest cache, tile
  cache, and concurrent tile-fetch logic.  Not a RadarSource itself.
- ``JMAAnalysisSource`` — implements the RadarSource interface
  (``fetch_frame`` / ``fetch_archive_frame``) over N1 manifest frames
  (basetime == validtime, observation-derived QPE).
- ``JMANowcastSource`` — implements the NowcastSource interface
  (``fetch_forecast``) over N2 manifest frames (validtime > basetime,
  JMA's own model-extrapolated nowcast to 60 minutes ahead).

The two source classes share one ``JMAFetcher`` instance internally so
the manifest and tile caches are not duplicated.
"""
from __future__ import annotations

import asyncio
import json
import logging
import time
from datetime import datetime, timezone

import httpx
import numpy as np

from librewxr.data.regions import RegionDef
from librewxr.data.retry import retry_get

from .decoder import (
    compute_tile_range,
    decode_jma_tile,
    resample_to_region,
)

logger = logging.getLogger(__name__)


_CADENCE_SEC = 300  # JMA HRPN native 5-min cadence
_MANIFEST_TTL_SEC = 60  # refresh manifest at most once per minute
_TILE_CACHE_MAX = 4096  # tile cache cap (110 tiles × ~37 frames worth)
_FRAME_CACHE_MAX = 12  # decoded region frames to retain
_HTTP_TIMEOUT = httpx.Timeout(20.0, connect=10.0)
_MAX_CONCURRENT_TILES = 24  # bounded parallelism for tile fetches


def _parse_jma_ts(ts: str) -> int:
    """JMA YYYYMMDDHHMMSS (UTC) → unix timestamp."""
    dt = datetime.strptime(ts, "%Y%m%d%H%M%S").replace(tzinfo=timezone.utc)
    return int(dt.timestamp())


class JMAFetcher:
    """Shared HTTP/cache state for JMA HRPN tiles.

    One instance is shared between the analysis source and the nowcast
    source so they don't duplicate the manifest or tile caches.  The
    analysis leg reads ``targetTimes_N1.json``; the nowcast leg reads
    ``targetTimes_N2.json``.  Tiles are namespaced by ``(basetime,
    validtime, element, z, x, y)`` so the two legs share the cache
    seamlessly when they happen to reference the same tile.
    """

    def __init__(self, base_url: str, zoom: int):
        self._base_url = base_url.rstrip("/")
        self._zoom = zoom
        self._client: httpx.AsyncClient | None = None
        # Manifest cache: {"N1" | "N2": (fetched_unix, parsed_frames)}
        self._manifest_cache: dict[
            str, tuple[float, list[dict]]
        ] = {}
        self._manifest_lock = asyncio.Lock()
        # Tile cache: {(basetime, validtime, element, z, x, y): uint8[256, 256]}
        self._tile_cache: dict[tuple[str, str, str, int, int, int], np.ndarray] = {}
        self._tile_cache_order: list[tuple[str, str, str, int, int, int]] = []
        self._tile_sem = asyncio.Semaphore(_MAX_CONCURRENT_TILES)

    @property
    def zoom(self) -> int:
        return self._zoom

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(
                timeout=_HTTP_TIMEOUT,
                follow_redirects=True,
            )
        return self._client

    async def close(self) -> None:
        if self._client is not None and not self._client.is_closed:
            await self._client.aclose()

    async def get_manifest(self, leg: str) -> list[dict] | None:
        """Fetch and parse a manifest (``"N1"`` or ``"N2"``), with caching."""
        async with self._manifest_lock:
            cached = self._manifest_cache.get(leg)
            if cached is not None and time.time() - cached[0] < _MANIFEST_TTL_SEC:
                return cached[1]

            url = f"{self._base_url}/targetTimes_{leg}.json"
            client = await self._get_client()
            resp = await retry_get(client, url, log_name=f"JMA-{leg}")
            if resp is None or resp.status_code != 200:
                logger.warning(
                    "JMA manifest fetch failed for %s (status=%s)",
                    leg, getattr(resp, "status_code", None),
                )
                if cached is not None:
                    return cached[1]
                return None
            try:
                parsed = json.loads(resp.content)
            except json.JSONDecodeError:
                logger.warning("JMA manifest %s: invalid JSON", leg)
                return None
            self._manifest_cache[leg] = (time.time(), parsed)
            return parsed

    async def fetch_region_frame(
        self,
        basetime: str,
        validtime: str,
        element: str,
        region: RegionDef,
    ) -> np.ndarray:
        """Fetch all tiles for a (basetime, validtime, element) and resample."""
        x_min, x_max, y_min, y_max = compute_tile_range(region, self._zoom)
        coords = [
            (x, y)
            for x in range(x_min, x_max + 1)
            for y in range(y_min, y_max + 1)
        ]
        tasks = [
            self._fetch_one_tile(basetime, validtime, element, x, y)
            for (x, y) in coords
        ]
        tiles = await asyncio.gather(*tasks)
        tile_grid: dict[tuple[int, int], np.ndarray] = {
            (x, y): tile
            for (x, y), tile in zip(coords, tiles)
            if tile is not None
        }
        return resample_to_region(tile_grid, self._zoom, region)

    async def _fetch_one_tile(
        self,
        basetime: str,
        validtime: str,
        element: str,
        x: int,
        y: int,
    ) -> np.ndarray | None:
        key = (basetime, validtime, element, self._zoom, x, y)
        cached = self._tile_cache.get(key)
        if cached is not None:
            return cached

        url = (
            f"{self._base_url}/{basetime}/none/{validtime}/surf/"
            f"{element}/{self._zoom}/{x}/{y}.png"
        )

        async with self._tile_sem:
            client = await self._get_client()
            resp = await retry_get(client, url, log_name="JMA-tile")

        if resp is None:
            return None
        if resp.status_code == 404:
            return None
        if resp.status_code != 200:
            logger.warning("JMA tile fetch failed: HTTP %d (%s)", resp.status_code, url)
            return None

        try:
            decoded = decode_jma_tile(resp.content)
        except Exception:
            logger.exception("JMA tile decode failed: %s", url)
            return None

        self._tile_cache[key] = decoded
        self._tile_cache_order.append(key)
        if len(self._tile_cache_order) > _TILE_CACHE_MAX:
            evict = self._tile_cache_order.pop(0)
            self._tile_cache.pop(evict, None)
        return decoded


class JMAAnalysisSource:
    """Analysis-leg (N1) JMA HRPN source — implements RadarSource."""

    def __init__(self, fetcher: JMAFetcher):
        self._fetcher = fetcher
        # Decoded per-region frames keyed by unix timestamp.
        self._frame_cache: dict[tuple[str, int], np.ndarray] = {}
        self._cache_order: list[tuple[str, int]] = []

    async def fetch_frame(
        self, region: RegionDef, minutes_ago: int,
    ) -> np.ndarray | None:
        now_aligned = (int(time.time()) // _CADENCE_SEC) * _CADENCE_SEC
        target_ts = now_aligned - minutes_ago * 60
        return await self._fetch_for_ts(region, target_ts)

    async def fetch_archive_frame(
        self, region: RegionDef, dt: datetime,
    ) -> np.ndarray | None:
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return await self._fetch_for_ts(region, int(dt.timestamp()))

    async def _fetch_for_ts(
        self, region: RegionDef, ts: int,
    ) -> np.ndarray | None:
        ts_aligned = (ts // _CADENCE_SEC) * _CADENCE_SEC
        key = (region.name, ts_aligned)
        cached = self._frame_cache.get(key)
        if cached is not None:
            return cached

        manifest = await self._fetcher.get_manifest("N1")
        if manifest is None:
            return None

        # N1 frames have basetime == validtime; find the entry matching ts_aligned.
        frame_meta = None
        for entry in manifest:
            entry_ts = _parse_jma_ts(entry["basetime"])
            if entry_ts == ts_aligned and "hrpns" in entry.get("elements", []):
                frame_meta = entry
                break
        if frame_meta is None:
            return None

        decoded = await self._fetcher.fetch_region_frame(
            frame_meta["basetime"],
            frame_meta["validtime"],
            "hrpns",
            region,
        )
        self._frame_cache[key] = decoded
        self._cache_order.append(key)
        if len(self._cache_order) > _FRAME_CACHE_MAX:
            evict = self._cache_order.pop(0)
            self._frame_cache.pop(evict, None)
        return decoded

    async def close(self) -> None:
        # Closing is owned by the shared fetcher; nothing to do here.
        pass


class JMANowcastSource:
    """Forecast-leg (N2) JMA HRPN source — implements NowcastSource.

    Returns ``[(validtime_unix, frame_data), ...]`` covering the latest
    forecast cycle (12 frames at 5-min steps from T+5 to T+60).
    """

    def __init__(self, fetcher: JMAFetcher):
        self._fetcher = fetcher
        # Cache the most recently fetched forecast cycle keyed by basetime.
        self._last_basetime: str | None = None
        self._last_frames: dict[str, list[tuple[int, np.ndarray]]] = {}

    async def fetch_forecast(
        self, region: RegionDef,
    ) -> list[tuple[int, np.ndarray]] | None:
        manifest = await self._fetcher.get_manifest("N2")
        if manifest is None or not manifest:
            return None

        latest_basetime = manifest[0]["basetime"]

        cached = self._last_frames.get(region.name)
        if cached is not None and self._last_basetime == latest_basetime:
            return cached

        # Forecast frames share one basetime; iterate validtimes oldest first.
        forecast_entries = sorted(
            (e for e in manifest if e["basetime"] == latest_basetime),
            key=lambda e: e["validtime"],
        )

        frames: list[tuple[int, np.ndarray]] = []
        for entry in forecast_entries:
            if "hrpns" not in entry.get("elements", []):
                continue
            decoded = await self._fetcher.fetch_region_frame(
                entry["basetime"],
                entry["validtime"],
                "hrpns",
                region,
            )
            frames.append((_parse_jma_ts(entry["validtime"]), decoded))

        if not frames:
            return None

        if self._last_basetime != latest_basetime:
            self._last_frames.clear()
            self._last_basetime = latest_basetime
        self._last_frames[region.name] = frames
        return frames

    async def close(self) -> None:
        pass
