# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Joshua Kimsey
import io
from datetime import datetime, timezone

import numpy as np
import pytest
from PIL import Image

pytestmark = pytest.mark.sources

from librewxr.data.regions import REGIONS, resolve_regions
from librewxr.data.sources import (
    MARNSource,
    _decode_marn_png,
    _MARN_DBZ_MAX,
    _MARN_DBZ_MIN,
)
from librewxr.tiles.coordinates import tile_overlaps_region


class TestSvcompRegion:
    def test_svcomp_in_regions(self):
        assert "SVCOMP" in REGIONS

    def test_svcomp_group_and_proj(self):
        r = REGIONS["SVCOMP"]
        assert r.proj == "latlon"
        assert r.group == "CENTRAL_AMERICA"

    def test_svcomp_bounds(self):
        # Bounds copied verbatim from the SNET viewer JS variables
        # pointSW120 / pointNE120 — must stay in sync if the viewer changes.
        r = REGIONS["SVCOMP"]
        assert r.west == -90.833
        assert r.east == -87.044
        assert r.south == 12.112
        assert r.north == 15.244

    def test_svcomp_dimensions(self):
        r = REGIONS["SVCOMP"]
        # Native SNET 120 km product is 409x342 — fixed grid dimensions
        # avoid pixel-size fencepost rounding ambiguity.
        assert r.width == 409
        assert r.height == 342

    def test_central_america_group_resolution(self):
        assert resolve_regions("CENTRAL_AMERICA") == ["SVCOMP"]

    def test_all_includes_svcomp(self):
        assert "SVCOMP" in resolve_regions("ALL")


class TestFilenameToUtc:
    def test_local_to_utc_offset(self):
        # El Salvador is UTC-6 year-round (no DST), so 23:25 local on
        # day N is 05:25 UTC on day N+1.
        dt = MARNSource._filename_to_utc(
            "esar82/Images/2026-05-14 23-25-00.png"
        )
        assert dt == datetime(2026, 5, 15, 5, 25, 0, tzinfo=timezone.utc)

    def test_handles_midnight_boundary(self):
        # 00:00 local = 06:00 UTC same day
        dt = MARNSource._filename_to_utc(
            "esar82/Images/2026-05-15 00-00-00.png"
        )
        assert dt == datetime(2026, 5, 15, 6, 0, 0, tzinfo=timezone.utc)

    def test_unparseable_returns_none(self):
        assert MARNSource._filename_to_utc("esar82/Images/garbage.png") is None
        assert MARNSource._filename_to_utc("not even a path") is None

    def test_strips_directory_prefix(self):
        # Should work whether the listing returns full keys or basenames.
        a = MARNSource._filename_to_utc("2026-05-14 23-25-00.png")
        b = MARNSource._filename_to_utc(
            "esar82/Images/2026-05-14 23-25-00.png"
        )
        assert a == b


class TestHueDecoder:
    """The 120 km SNET product encodes dBZ as a continuous HSV-style gradient:

    * green→cyan arc  (G=255, R=0, B varies)  → low dBZ
    * cyan→blue arc   (B=255, R=0, G varies)  → mid dBZ
    * blue→magenta arc (B=255, G=0, R varies) → high dBZ
    """

    def _make_png(self, pixels: list[tuple[int, int, int, int]]) -> bytes:
        arr = np.array([pixels], dtype=np.uint8)
        img = Image.fromarray(arr, mode="RGBA")
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        return buf.getvalue()

    def _fake_region(self, w: int):
        # SVCOMP's exact shape isn't needed for decode-correctness tests,
        # but the decoder logs a warning on shape mismatch.  Build a tiny
        # stand-in region matching the test PNG shape.
        from librewxr.data.regions import RegionDef
        return RegionDef(
            name="SVCOMP", west=0.0, east=1.0, south=0.0, north=1.0,
            pixel_size=1.0, group="CENTRAL_AMERICA",
            grid_width=w, grid_height=1,
        )

    def test_transparent_is_no_data(self):
        # tRNS isn't easy to embed via Pillow's save; rely on the
        # decoder's alpha=0 → no-data behaviour (the `.convert("RGBA")`
        # in _decode_marn_png turns tRNS into alpha=0 at load time).
        png = self._make_png([(0, 0, 0, 0)])
        out = _decode_marn_png(png, self._fake_region(1))
        assert out is not None
        assert out.shape == (1, 1)
        assert out[0, 0] == 0

    def test_pure_green_is_lowest_dbz(self):
        # Green (0,255,0) sits at hue 120 — the low end of the gradient.
        # → dBZ = _MARN_DBZ_MIN → uint8 = (dBZ + 32) * 2
        png = self._make_png([(0, 255, 0, 255)])
        out = _decode_marn_png(png, self._fake_region(1))
        expected = int(round((_MARN_DBZ_MIN + 32.0) * 2.0))
        assert out[0, 0] == expected

    def test_pure_magenta_is_highest_dbz(self):
        # Magenta (255,0,255) sits at hue 300 — the high end.
        png = self._make_png([(255, 0, 255, 255)])
        out = _decode_marn_png(png, self._fake_region(1))
        expected = int(round((_MARN_DBZ_MAX + 32.0) * 2.0))
        assert out[0, 0] == expected

    def test_cyan_midpoint(self):
        # Pure cyan (0,255,255) is hue 180 — quarter way along the
        # 180° span, so dBZ = MIN + (MAX - MIN) * (60/180).
        png = self._make_png([(0, 255, 255, 255)])
        out = _decode_marn_png(png, self._fake_region(1))
        expected_dbz = _MARN_DBZ_MIN + (_MARN_DBZ_MAX - _MARN_DBZ_MIN) / 3
        expected = int(round((expected_dbz + 32.0) * 2.0))
        assert abs(int(out[0, 0]) - expected) <= 1

    def test_blue_midpoint(self):
        # Pure blue (0,0,255) is hue 240 — two thirds along the span.
        png = self._make_png([(0, 0, 255, 255)])
        out = _decode_marn_png(png, self._fake_region(1))
        expected_dbz = _MARN_DBZ_MIN + (_MARN_DBZ_MAX - _MARN_DBZ_MIN) * 2 / 3
        expected = int(round((expected_dbz + 32.0) * 2.0))
        assert abs(int(out[0, 0]) - expected) <= 1

    def test_monotonic_along_each_arc(self):
        # Within each arc, increasing the varying channel should give
        # monotonically increasing dBZ.
        arc1 = [(0, 255, b, 255) for b in (0, 64, 128, 192, 255)]
        arc2 = [(0, g, 255, 255) for g in (200, 128, 64, 0)]  # G:255→0
        arc3 = [(r, 0, 255, 255) for r in (0, 64, 128, 192, 255)]
        for arc in (arc1, arc2, arc3):
            png = self._make_png(arc)
            out = _decode_marn_png(png, self._fake_region(len(arc)))
            diffs = np.diff(out[0].astype(int))
            assert np.all(diffs >= 0), f"non-monotonic decode for {arc}"
            assert np.any(diffs > 0), f"all-equal decode for {arc}"

    def test_off_palette_color_is_no_data(self):
        # White (255,255,255) doesn't sit on any of the three arcs and
        # should decode as no-data.  This protects against
        # decoder-injected colours from upstream tools (e.g. a future
        # overlay layer).
        png = self._make_png([(255, 255, 255, 255)])
        out = _decode_marn_png(png, self._fake_region(1))
        assert out[0, 0] == 0

    def test_bad_png_returns_none(self):
        assert _decode_marn_png(b"not a png", self._fake_region(1)) is None


class TestSvcompTileOverlap:
    def test_tile_over_central_america_overlaps(self):
        region = REGIONS["SVCOMP"]
        # z=6, x=15, y=29 covers roughly El Salvador / Guatemala
        assert tile_overlaps_region(region, z=6, x=15, y=29)

    def test_tile_over_europe_does_not_overlap(self):
        region = REGIONS["SVCOMP"]
        # z=4, x=8, y=5 is over central Europe — no overlap.
        assert not tile_overlaps_region(region, z=4, x=8, y=5)
