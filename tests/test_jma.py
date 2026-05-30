# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Joshua Kimsey
"""Tests for the JMA HRPN source package."""
from __future__ import annotations

import io
import math

import numpy as np
import pytest
from PIL import Image

from librewxr.sources._base import NowcastContribution, NowcastSource, RadarSourceContribution
from librewxr.sources.regional.east_asia.japan.radar.jma import (
    nowcast_provider,
    radar_provider,
)
from librewxr.sources.regional.east_asia.japan.radar.jma.decoder import (
    _DBZ_BY_INDEX,
    compute_tile_range,
    decode_jma_tile,
    lat_to_tile_y,
    lon_to_tile_x,
    resample_to_region,
    tile_x_to_lon,
    tile_y_to_lat,
)
from librewxr.sources.regional.east_asia.japan.radar.jma.regions import JPCOMP
from librewxr.sources.regional.east_asia.japan.radar.jma.source import (
    JMAAnalysisSource,
    JMAFetcher,
    JMANowcastSource,
    _parse_jma_ts,
)


pytestmark = pytest.mark.sources


# Reconstruct JMA's 10-stop palette for use in synthetic fixture tiles.
# These exact RGB values were extracted via live PLTE probe of a real
# populated tile on 2026-05-30.
_JMA_PLTE_RGB: list[tuple[int, int, int]] = [
    (255, 255, 255),  # 0: nodata
    (255, 255, 255),  # 1: nodata
    (242, 242, 255),  # 2: 0.1-1 mm/h
    (160, 210, 255),  # 3: 1-5
    (33, 140, 255),   # 4: 5-10
    (0, 65, 255),     # 5: 10-20
    (250, 245, 0),    # 6: 20-30
    (255, 153, 0),    # 7: 30-50
    (255, 40, 0),     # 8: 50-80
    (180, 0, 104),    # 9: 80+
]
_JMA_TRNS = bytes([0, 0] + [255] * 8)


def _make_palette_tile(indices: np.ndarray) -> bytes:
    """Build a JMA-style 4-bit palette PNG from a (256, 256) uint8 index array."""
    assert indices.shape == (256, 256)
    assert indices.dtype == np.uint8
    img = Image.fromarray(indices, mode="P")
    palette_flat = []
    for r, g, b in _JMA_PLTE_RGB:
        palette_flat.extend((r, g, b))
    # Pad palette to 256 entries (PIL pads automatically when shorter)
    img.putpalette(palette_flat)
    buf = io.BytesIO()
    img.save(buf, format="PNG", transparency=_JMA_TRNS, bits=4)
    return buf.getvalue()


def _make_empty_rgba_tile() -> bytes:
    """Build a JMA-style 8-bit RGBA all-transparent sentinel tile."""
    img = Image.new("RGBA", (256, 256), (0, 0, 0, 0))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


class TestPaletteEncoding:
    def test_nodata_indices_map_to_zero(self):
        assert _DBZ_BY_INDEX[0] == 0
        assert _DBZ_BY_INDEX[1] == 0

    def test_dbz_values_monotonically_increase(self):
        # Indices 2-9 represent increasing mm/h bins, so encoded uint8
        # dBZ must increase monotonically across them.
        for i in range(2, 9):
            assert _DBZ_BY_INDEX[i] < _DBZ_BY_INDEX[i + 1], (
                f"dBZ at index {i} ({_DBZ_BY_INDEX[i]}) "
                f">= index {i+1} ({_DBZ_BY_INDEX[i+1]})"
            )

    def test_lightest_precip_above_zero_uint8(self):
        # 0.3 mm/h M-P → ~14.5 dBZ → uint8 (14.5 + 32) * 2 = 93.
        assert _DBZ_BY_INDEX[2] == pytest.approx(93, abs=2)

    def test_heaviest_precip_within_uint8_range(self):
        # 100 mm/h M-P → ~55 dBZ → uint8 (55 + 32) * 2 = 174.
        assert _DBZ_BY_INDEX[9] == pytest.approx(174, abs=2)
        assert _DBZ_BY_INDEX[9] < 255


class TestTileDecoding:
    def test_decode_all_zeros_palette(self):
        indices = np.zeros((256, 256), dtype=np.uint8)
        data = _make_palette_tile(indices)
        decoded = decode_jma_tile(data)
        assert decoded.shape == (256, 256)
        assert decoded.dtype == np.uint8
        assert decoded.max() == 0

    def test_decode_uniform_precipitation(self):
        # Fill the entire tile with index 5 (10-20 mm/h)
        indices = np.full((256, 256), 5, dtype=np.uint8)
        data = _make_palette_tile(indices)
        decoded = decode_jma_tile(data)
        expected = _DBZ_BY_INDEX[5]
        assert decoded.min() == expected
        assert decoded.max() == expected

    def test_decode_mixed_indices(self):
        indices = np.zeros((256, 256), dtype=np.uint8)
        indices[0:64, :] = 3      # light rain band
        indices[64:128, :] = 5    # heavier rain band
        indices[128:192, :] = 7   # intense rain band
        indices[192:256, :] = 9   # extreme band
        data = _make_palette_tile(indices)
        decoded = decode_jma_tile(data)
        assert decoded[10, 10] == _DBZ_BY_INDEX[3]
        assert decoded[80, 80] == _DBZ_BY_INDEX[5]
        assert decoded[150, 150] == _DBZ_BY_INDEX[7]
        assert decoded[220, 220] == _DBZ_BY_INDEX[9]

    def test_decode_rgba_empty_sentinel(self):
        data = _make_empty_rgba_tile()
        decoded = decode_jma_tile(data)
        assert decoded.shape == (256, 256)
        assert decoded.dtype == np.uint8
        assert decoded.max() == 0

    def test_decode_rejects_wrong_size(self):
        img = Image.new("RGBA", (128, 128), (0, 0, 0, 0))
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        with pytest.raises(ValueError, match="unexpected size"):
            decode_jma_tile(buf.getvalue())


class TestMercatorMath:
    def test_lon_to_tile_x_roundtrip(self):
        for z in (4, 6, 7, 8):
            for lon in (-179.0, -90.0, 0.0, 90.0, 139.69, 179.0):
                x = lon_to_tile_x(lon, z)
                back = tile_x_to_lon(x, z)
                assert back == pytest.approx(lon, abs=1e-6), (
                    f"z={z} lon={lon} x={x} back={back}"
                )

    def test_lat_to_tile_y_roundtrip(self):
        # asinh/tan composition has small numerical error; allow 1e-3°
        # tolerance across the Web Mercator latitude range.
        for z in (4, 6, 7, 8):
            for lat in (-80.0, -45.0, 0.0, 35.68, 45.0, 80.0):
                y = lat_to_tile_y(lat, z)
                back = tile_y_to_lat(y, z)
                assert back == pytest.approx(lat, abs=1e-3), (
                    f"z={z} lat={lat} y={y} back={back}"
                )

    def test_tile_y_clamps_at_mercator_pole(self):
        # latitudes outside ±85.05° clamp to the pole (no NaN)
        y_high = lat_to_tile_y(89.0, 7)
        assert math.isfinite(y_high)
        y_low = lat_to_tile_y(-89.0, 7)
        assert math.isfinite(y_low)

    def test_tokyo_tile_coordinates(self):
        # Tokyo at (35.68, 139.69) at z=7 should land in tile (~113, ~50)
        x = lon_to_tile_x(139.69, 7)
        y = lat_to_tile_y(35.68, 7)
        assert int(x) in (113, 114), f"Tokyo lon -> tile_x = {x}"
        assert int(y) in (50, 51), f"Tokyo lat -> tile_y = {y}"


class TestTileRange:
    def test_jpcomp_tile_range_z7(self):
        x_min, x_max, y_min, y_max = compute_tile_range(JPCOMP, 7)
        nx = x_max - x_min + 1
        ny = y_max - y_min + 1
        # Live probe found 10 x 11 = 110 tiles cover JPCOMP at z=7.
        assert (nx, ny) == (10, 11), f"Expected 10x11 tiles, got {nx}x{ny}"

    def test_jpcomp_tile_range_z6_smaller(self):
        z6 = compute_tile_range(JPCOMP, 6)
        z7 = compute_tile_range(JPCOMP, 7)
        n6 = (z6[1] - z6[0] + 1) * (z6[3] - z6[2] + 1)
        n7 = (z7[1] - z7[0] + 1) * (z7[3] - z7[2] + 1)
        assert n6 < n7, "z=6 should require fewer tiles than z=7"


class TestResample:
    def test_resample_output_shape_matches_region(self):
        # Construct one populated and one empty tile, place into the
        # corner of the tile range, confirm output shape matches region.
        x_min, x_max, y_min, y_max = compute_tile_range(JPCOMP, 7)
        indices = np.full((256, 256), 4, dtype=np.uint8)
        tile_data = decode_jma_tile(_make_palette_tile(indices))
        tile_grid = {(x_min, y_min): tile_data}
        out = resample_to_region(tile_grid, 7, JPCOMP)
        assert out.shape == (JPCOMP.height, JPCOMP.width)
        assert out.dtype == np.uint8

    def test_resample_zero_when_no_tiles(self):
        out = resample_to_region({}, 7, JPCOMP)
        assert out.shape == (JPCOMP.height, JPCOMP.width)
        assert out.max() == 0


class TestTimestampParsing:
    def test_parse_jma_ts_round_trip_via_datetime(self):
        from datetime import datetime, timezone
        dt = datetime(2026, 5, 30, 3, 50, 0, tzinfo=timezone.utc)
        assert _parse_jma_ts("20260530035000") == int(dt.timestamp())

    def test_parse_jma_ts_treats_as_utc(self):
        # Two timestamps 5 minutes apart should differ by exactly 300s.
        a = _parse_jma_ts("20260530035000")
        b = _parse_jma_ts("20260530035500")
        assert b - a == 300


class TestProviderShape:
    def test_radar_provider_returns_contribution_when_enabled(self):
        class _S:
            jma_enabled = True
            jma_base_url = "https://example.invalid/nowc"
            jma_zoom = 7
        c = radar_provider(_S())
        assert isinstance(c, RadarSourceContribution)
        assert c.group == "JAPAN"
        assert len(c.regions) == 1
        assert c.regions[0].name == "JPCOMP"
        assert "JPCOMP" in c.station_map

    def test_radar_provider_returns_none_when_disabled(self):
        class _S:
            jma_enabled = False
        assert radar_provider(_S()) is None

    def test_nowcast_provider_returns_contribution_when_enabled(self):
        class _S:
            jma_enabled = True
            jma_nowcast_enabled = True
            jma_base_url = "https://example.invalid/nowc"
            jma_zoom = 7
        c = nowcast_provider(_S())
        assert isinstance(c, NowcastContribution)
        assert c.region_name == "JPCOMP"
        assert c.horizon_minutes == 60
        assert isinstance(c.instance, NowcastSource)

    def test_nowcast_provider_returns_none_when_nowcast_disabled(self):
        class _S:
            jma_enabled = True
            jma_nowcast_enabled = False
        assert nowcast_provider(_S()) is None

    def test_nowcast_provider_returns_none_when_jma_disabled(self):
        class _S:
            jma_enabled = False
            jma_nowcast_enabled = True
        assert nowcast_provider(_S()) is None


class TestSourceClasses:
    def test_analysis_source_implements_radarsource_shape(self):
        f = JMAFetcher("https://example.invalid/nowc", 7)
        s = JMAAnalysisSource(f)
        # Duck-type check: fetch_frame, fetch_archive_frame, close all exist.
        assert callable(s.fetch_frame)
        assert callable(s.fetch_archive_frame)
        assert callable(s.close)

    def test_nowcast_source_implements_nowcastsource_shape(self):
        f = JMAFetcher("https://example.invalid/nowc", 7)
        s = JMANowcastSource(f)
        assert callable(s.fetch_forecast)
        assert callable(s.close)
