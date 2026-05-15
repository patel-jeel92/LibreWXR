# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Joshua Kimsey
from datetime import datetime, timezone

import numpy as np
import pytest

pytestmark = pytest.mark.sources

from librewxr.data.regions import REGIONS, resolve_regions
from librewxr.data.sources import CWASource, _parse_cwa_xml
from librewxr.tiles.coordinates import tile_overlaps_region


class TestTwcompRegion:
    def test_twcomp_in_regions(self):
        assert "TWCOMP" in REGIONS

    def test_twcomp_group_and_proj(self):
        r = REGIONS["TWCOMP"]
        assert r.proj == "latlon"
        assert r.group == "TAIWAN"

    def test_twcomp_bounds(self):
        # Bounds derived from CWA O-A0059-001 metadata: SW corner
        # (115.0°E, 18.0°N), 921×881 cells at 0.0125° spacing.
        r = REGIONS["TWCOMP"]
        assert r.west == 115.0
        assert r.east == pytest.approx(126.5125)
        assert r.south == 18.0
        assert r.north == pytest.approx(29.0125)

    def test_twcomp_dimensions(self):
        r = REGIONS["TWCOMP"]
        assert r.width == 921
        assert r.height == 881

    def test_taiwan_group_resolution(self):
        assert resolve_regions("TAIWAN") == ["TWCOMP"]

    def test_all_includes_twcomp(self):
        assert "TWCOMP" in resolve_regions("ALL")


class TestUrlBuilder:
    def test_utc_to_local_offset(self):
        # 06:00 UTC = 14:00 Taipei → filename embeds 202605151400
        src = CWASource()
        ts = int(datetime(2026, 5, 15, 6, 0, 0, tzinfo=timezone.utc).timestamp())
        url = src._url_for_timestamp(ts)
        assert url.endswith("202605151400compref_mosaic.xml")

    def test_rounds_to_10min(self):
        # Mid-slot 06:07:23 UTC must round DOWN to 06:00 UTC → 14:00 local
        src = CWASource()
        ts = int(datetime(2026, 5, 15, 6, 7, 23, tzinfo=timezone.utc).timestamp())
        url = src._url_for_timestamp(ts)
        assert url.endswith("202605151400compref_mosaic.xml")

    def test_uses_archive_path(self):
        src = CWASource()
        ts = int(datetime(2026, 5, 15, 6, 0, 0, tzinfo=timezone.utc).timestamp())
        url = src._url_for_timestamp(ts)
        # Confirm we hit the /history/Observation/ path, not the
        # 'latest' endpoint at /Observation/<file>.
        assert "/history/Observation/" in url

    def test_no_separator_dot_in_archive_key(self):
        # Critical: archive keys are {YYYYMMDDHHMM}compref_mosaic.xml
        # with NO separator dot.  The QPESUMS gauge keys do use a dot
        # ({ts}.QPESUMS_GAUGE.10M.xml) — different pattern.
        src = CWASource()
        ts = int(datetime(2026, 5, 15, 6, 0, 0, tzinfo=timezone.utc).timestamp())
        fname = src._url_for_timestamp(ts).rsplit("/", 1)[-1]
        assert fname == "202605151400compref_mosaic.xml"
        assert ".compref_mosaic" not in fname

    def test_custom_base_url(self):
        src = CWASource(base_url="https://example.test/")
        ts = int(datetime(2026, 1, 1, 0, 0, 0, tzinfo=timezone.utc).timestamp())
        url = src._url_for_timestamp(ts)
        # Trailing slash must be stripped; UTC 00:00 = Taipei 08:00 same day
        assert url.startswith("https://example.test/history/Observation/")
        assert url.endswith("202601010800compref_mosaic.xml")


def _build_xml(dimx: int, dimy: int, values: list[float]) -> bytes:
    """Build a minimal valid CWA-style XML body for testing.

    Reproduces just enough structure that the decoder finds the
    namespaced ``<content>`` element and parses the float string.
    """
    content = ",".join(f"{v:.3E}" for v in values)
    body = f"""<?xml version="1.0" encoding="UTF-8"?>
<cwaopendata xmlns="urn:cwa:gov:tw:cwacommon:0.1">
  <dataset>
    <datasetInfo>
      <parameterSet>
        <GridDimensionX>{dimx}</GridDimensionX>
        <GridDimensionY>{dimy}</GridDimensionY>
      </parameterSet>
    </datasetInfo>
    <contents>
      <content>{content}</content>
    </contents>
  </dataset>
</cwaopendata>
"""
    return body.encode("utf-8")


def _tiny_region(w: int, h: int):
    """A small stand-in region with TWCOMP's structure but tiny grid."""
    from librewxr.data.regions import RegionDef
    return RegionDef(
        name="TWCOMP", west=0.0, east=1.0, south=0.0, north=1.0,
        pixel_size=1.0, group="TAIWAN",
        grid_width=w, grid_height=h,
    )


class TestXmlDecoder:
    def test_parses_correct_shape(self):
        # 3×2 grid, 6 values, all 25 dBZ
        xml = _build_xml(3, 2, [25.0] * 6)
        out = _parse_cwa_xml(xml, _tiny_region(3, 2))
        assert out is not None
        assert out.shape == (2, 3)

    def test_vertical_flip(self):
        # XML order is south-to-north (first value = SW corner).  After
        # flip, row 0 should hold the LAST row of values (the north row).
        # 2×2 grid:
        #   XML order: [SW, SE, NW, NE]
        #   After flip: row 0 = [NW, NE]; row 1 = [SW, SE]
        # Use distinct dBZ values so we can identify each cell.
        xml = _build_xml(2, 2, [10.0, 20.0, 30.0, 40.0])
        out = _parse_cwa_xml(xml, _tiny_region(2, 2))
        # uint8 = (dBZ + 32) * 2 → 10→84, 20→104, 30→124, 40→144
        assert out[0, 0] == 124   # NW (was index 2 in XML)
        assert out[0, 1] == 144   # NE (was index 3)
        assert out[1, 0] == 84    # SW (was index 0)
        assert out[1, 1] == 104   # SE (was index 1)

    def test_minus_99_sentinel_is_no_data(self):
        # -99 = invalid → uint8 0
        xml = _build_xml(2, 1, [-99.0, 25.0])
        out = _parse_cwa_xml(xml, _tiny_region(2, 1))
        assert out[0, 0] == 0
        assert out[0, 1] == 114   # 25 dBZ → (25+32)*2 = 114

    def test_minus_999_sentinel_is_no_data(self):
        # -999 = outside radar range → uint8 0
        xml = _build_xml(2, 1, [-999.0, 25.0])
        out = _parse_cwa_xml(xml, _tiny_region(2, 1))
        assert out[0, 0] == 0
        assert out[0, 1] == 114

    def test_dbz_round_trip(self):
        # Encoder is (dBZ + 32) * 2 clipped to [0, 255].
        # 5 dBZ → 74, 70 dBZ → 204.
        xml = _build_xml(2, 1, [5.0, 70.0])
        out = _parse_cwa_xml(xml, _tiny_region(2, 1))
        assert out[0, 0] == 74
        assert out[0, 1] == 204

    def test_size_mismatch_returns_none(self):
        # 4 values but region expects 2×3 = 6 → decoder should refuse.
        xml = _build_xml(2, 3, [1.0, 2.0, 3.0, 4.0])
        out = _parse_cwa_xml(xml, _tiny_region(2, 3))
        assert out is None

    def test_malformed_xml_returns_none(self):
        assert _parse_cwa_xml(b"not xml at all", _tiny_region(1, 1)) is None

    def test_missing_content_element_returns_none(self):
        body = b"""<?xml version="1.0" encoding="UTF-8"?>
<cwaopendata xmlns="urn:cwa:gov:tw:cwacommon:0.1">
  <dataset></dataset>
</cwaopendata>
"""
        out = _parse_cwa_xml(body, _tiny_region(1, 1))
        assert out is None

    def test_dbz_below_minus_32_is_no_data(self):
        # The shared encoder treats anything ≤ -32 as no-data.  A
        # legitimate weak return of -14.5 dBZ should still encode as
        # positive uint8 (35).
        xml = _build_xml(2, 1, [-14.5, -40.0])
        out = _parse_cwa_xml(xml, _tiny_region(2, 1))
        assert out[0, 0] == 35    # (-14.5 + 32) * 2
        assert out[0, 1] == 0     # below noise floor → no-data


class TestTwcompTileOverlap:
    def test_tile_over_taiwan_overlaps(self):
        region = REGIONS["TWCOMP"]
        # z=5 tile (26, 13) covers roughly 112.5..123.75°E × 21.9..31.9°N
        assert tile_overlaps_region(region, z=5, x=26, y=13)

    def test_tile_over_europe_does_not_overlap(self):
        region = REGIONS["TWCOMP"]
        # z=4 tile (8, 5) covers central Europe — far from Taiwan.
        assert not tile_overlaps_region(region, z=4, x=8, y=5)
