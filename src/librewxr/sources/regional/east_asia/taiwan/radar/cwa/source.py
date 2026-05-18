# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Joshua Kimsey
"""Taiwan CWA QPESUMS composite reflectivity (O-A0059-001) source.

Fetches the 7-radar QPESUMS composite published by the Central Weather
Administration to anonymous AWS S3 (``cwaopendata`` in ``ap-northeast-1``).
Each frame is a UTF-8 XML at 10-min cadence (clock-aligned in Taipei
local time, UTC+8 with no DST), with ~9 MB of comma-separated
scientific-notation floats inside a single ``<content>`` element.

Sentinels:
    -99    invalid
    -999   outside radar range / QC-removed

Archive key format uses NO separator dot between the timestamp and the
product name (``{YYYYMMDDHHMM}compref_mosaic.xml``).  The sibling
QPESUMS gauge keys ``{YYYYMMDDHHMM}.QPESUMS_GAUGE.10M.xml`` *do* use a
dot — easy to mix up.

License: data.gov.tw Open Government Data License v1.0, attribution
required.  See README and ``docs/coverage.md`` for the citation.
"""
import logging
import time
from datetime import datetime, timedelta, timezone
from xml.etree import ElementTree as ET

import httpx
import numpy as np

from librewxr.data.regions import RegionDef
from librewxr.data.retry import retry_get
# Shared dBZ → uint8 encoder.  Currently lives in ``data/sources.py``;
# moves out of there once that file is empty of legacy classes.
from librewxr.sources._helpers import _dbz_float_to_uint8

logger = logging.getLogger(__name__)


_CWA_NAMESPACE = "urn:cwa:gov:tw:cwacommon:0.1"


class CWASource:
    """Taiwan CWA QPESUMS composite reflectivity (O-A0059-001) source.

    Anonymous S3 (``cwaopendata`` in ``ap-northeast-1``).  10-min
    cadence, clock-aligned in Taipei local time (UTC+8).  Decoder
    parses scientific-notation floats from the XML ``<content>``
    element and flips vertically (south-to-north → north-up).
    """

    _ARCHIVE_PREFIX = "/history/Observation"
    _LOCAL_TZ_OFFSET = 8           # Taipei is UTC+8 year-round (no DST)
    _CADENCE_SEC = 600             # 10 minutes
    # Files publish ~6 min after their frame time, so the most-recent
    # 1-2 slots are often 404.  Walk back up to 3 older slots.
    _MAX_FALLBACK_STEPS = 3

    def __init__(
        self,
        base_url: str = "https://cwaopendata.s3.ap-northeast-1.amazonaws.com",
    ):
        self._base_url = base_url.rstrip("/")
        self._client: httpx.AsyncClient | None = None

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(
                timeout=httpx.Timeout(90.0, connect=15.0),
                follow_redirects=True,
            )
        return self._client

    def _url_for_timestamp(self, ts: int) -> str:
        """Build the archive URL for a unix timestamp.

        Rounds to a 10-min slot, converts UTC → UTC+8, and formats
        as ``{YYYYMMDDHHMM}compref_mosaic.xml`` (note: no separator
        dot between timestamp and product name).
        """
        rounded = (ts // self._CADENCE_SEC) * self._CADENCE_SEC
        local = datetime.fromtimestamp(
            rounded, tz=timezone.utc
        ) + timedelta(hours=self._LOCAL_TZ_OFFSET)
        fname = local.strftime("%Y%m%d%H%M") + "compref_mosaic.xml"
        return f"{self._base_url}{self._ARCHIVE_PREFIX}/{fname}"

    async def fetch_frame(
        self, region: RegionDef, minutes_ago: int
    ) -> np.ndarray | None:
        now_rounded = (
            int(time.time() // self._CADENCE_SEC) * self._CADENCE_SEC
        )
        target_ts = now_rounded - minutes_ago * 60
        return await self._fetch_xml(target_ts, region)

    async def fetch_archive_frame(
        self, region: RegionDef, dt: datetime
    ) -> np.ndarray | None:
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return await self._fetch_xml(int(dt.timestamp()), region)

    async def _fetch_xml(
        self, ts: int, region: RegionDef
    ) -> np.ndarray | None:
        """Download and parse, falling back to older slots on 404."""
        client = await self._get_client()
        for step in range(self._MAX_FALLBACK_STEPS + 1):
            url = self._url_for_timestamp(ts - step * self._CADENCE_SEC)
            resp = await retry_get(client, url, log_name="CWA")
            if resp is None:
                return None
            if resp.status_code == 200:
                return _parse_cwa_xml(resp.content, region)
            if resp.status_code == 404 and step < self._MAX_FALLBACK_STEPS:
                continue
            logger.warning(
                "CWA fetch failed: HTTP %d (%s)",
                resp.status_code, url.rsplit("/", 1)[-1],
            )
            return None
        return None

    async def close(self) -> None:
        if self._client and not self._client.is_closed:
            await self._client.aclose()


def _parse_cwa_xml(data: bytes, region: RegionDef) -> np.ndarray | None:
    """Parse a CWA O-A0059-001 XML into a uint8 dBZ array.

    Steps:
    1. Locate the namespaced ``<content>`` element.
    2. Parse comma-separated scientific-notation floats into a 1D array.
    3. Reshape to ``(height, width)`` (the XML order is row-major,
       south-to-north — first value is the SW corner) and flip
       vertically so row 0 ends up north.
    4. Map sentinels (``-99``, ``-999``) to the shared no-data sentinel
       (-33), then encode to uint8 via the shared encoder.
    """
    try:
        root = ET.fromstring(data)
    except ET.ParseError:
        logger.exception("Failed to parse CWA XML")
        return None

    content_el = root.find(f".//{{{_CWA_NAMESPACE}}}content")
    if content_el is None or content_el.text is None:
        logger.warning("CWA XML missing <content> element")
        return None

    try:
        flat = np.fromstring(content_el.text, sep=",", dtype=np.float32)
    except Exception:
        logger.exception("Failed to parse CWA <content> floats")
        return None

    expected = region.width * region.height
    if flat.size != expected:
        logger.warning(
            "CWA grid size mismatch for %s: %d (expected %d)",
            region.name, flat.size, expected,
        )
        return None

    # Row-major south-to-north → flip vertically to north-up convention.
    grid = flat.reshape(region.height, region.width)[::-1]

    # Sentinels: -99 (invalid) and -999 (outside radar range / QC-removed)
    invalid = (grid <= -99.0)
    grid = np.where(invalid, -33.0, grid)
    return _dbz_float_to_uint8(grid)
