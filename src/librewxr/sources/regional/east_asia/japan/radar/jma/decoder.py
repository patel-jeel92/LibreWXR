# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Joshua Kimsey
"""Decoder and tile-grid math for the JMA HRPN composite.

Pure functions only — no HTTP, no caching, no I/O state.  ``source.py``
owns those.  This module:

  1. Parses JMA's 4-bit palette + tRNS PNG tiles (10-stop precipitation
     palette) into mm/h arrays, and recognises the 8-bit RGBA
     all-transparent sentinel JMA uses for empty tiles.
  2. Provides standard Web Mercator XYZ tile math (lon/lat ↔ tile coords).
  3. Computes the tile grid covering a ``RegionDef`` at a given zoom.
  4. Resamples a stitched Mercator mosaic onto the region's lat/lon grid
     via ``cv2.remap`` bilinear interpolation.
  5. Applies Marshall-Palmer (Z = 200·R^1.6) for mm/h → dBZ conversion.

The 10-stop palette mapping was confirmed by extracting the PLTE chunk
from a populated JMA tile (live probe 2026-05-30) and cross-referencing
the RGB triplets against JMA's published "降水ナウキャスト" (Precipitation
Nowcast) legend at ``www.jma.go.jp/bosai/nowc/``.  Bin boundaries match
JMA's standard public-facing precipitation legend (0.1, 1, 5, 10, 20,
30, 50, 80 mm/h).
"""
from __future__ import annotations

import io
import math

import cv2
import numpy as np
from PIL import Image

from librewxr.data.regions import RegionDef
from librewxr.sources._helpers import _dbz_float_to_uint8


# JMA HRPN 10-stop palette → representative mm/h.
#
# Indices 0 and 1 are both (255, 255, 255) with alpha=0 in the tRNS
# chunk — "no data" / "no precipitation observed".  Indices 2-9 are the
# eight precipitation bins, with the representative mm/h taken as the
# geometric midpoint of each open-on-the-right interval (except the
# closed upper bin 9, where we use a representative value above the
# 80 mm/h threshold).  Marshall-Palmer maps these to dBZ values in the
# range 18-55, well within standard radar colour-scheme bounds.
_MMH_BY_INDEX: np.ndarray = np.array(
    [
        0.0,    # 0: nodata
        0.0,    # 1: nodata (alternate "no detection" marker)
        0.3,    # 2: 0.1 ≤ R < 1 mm/h
        2.0,    # 3: 1   ≤ R < 5 mm/h
        7.0,    # 4: 5   ≤ R < 10 mm/h
        14.0,   # 5: 10  ≤ R < 20 mm/h
        25.0,   # 6: 20  ≤ R < 30 mm/h
        40.0,   # 7: 30  ≤ R < 50 mm/h
        65.0,   # 8: 50  ≤ R < 80 mm/h
        100.0,  # 9: R ≥ 80 mm/h
    ],
    dtype=np.float32,
)


def _build_dbz_table() -> np.ndarray:
    """Marshall-Palmer dBZ table indexed by palette index.

    Uses the project's canonical uint8 encoding via
    ``_dbz_float_to_uint8`` (pixel = clamp((dBZ + 32) * 2, 0, 255)).
    Indices 0-1 (no data) are post-mapped to 0 explicitly so the
    canonical encoder's NODATA-mask logic kicks in.
    """
    mmh = _MMH_BY_INDEX
    dbz_float = np.where(
        mmh > 0,
        23.0103 + 16.0 * np.log10(np.maximum(mmh, 0.001)),
        -99.0,  # below the -32 NODATA threshold in _dbz_float_to_uint8
    )
    return _dbz_float_to_uint8(dbz_float.astype(np.float32))


_DBZ_BY_INDEX: np.ndarray = _build_dbz_table()


# JMA serves all-transparent tiles as an 8-bit RGBA PNG of fixed size
# (~334 bytes for the typical case).  Populated tiles use 4-bit palette
# PNG with PLTE+tRNS chunks.  Detecting the format from the PIL mode
# avoids parsing the full image when the tile is known-empty.
_TILE_PX = 256
_EMPTY_TILE: np.ndarray = np.zeros((_TILE_PX, _TILE_PX), dtype=np.uint8)


def decode_jma_tile(png_bytes: bytes) -> np.ndarray:
    """Decode one JMA HRPN tile to a 256×256 uint8 dBZ array.

    Returns the project's standard dBZ encoding (0 = no data).  Handles
    both the 4-bit palette PNG format (populated tiles) and the 8-bit
    RGBA all-transparent sentinel (empty tiles).
    """
    img = Image.open(io.BytesIO(png_bytes))
    if img.size != (_TILE_PX, _TILE_PX):
        raise ValueError(
            f"JMA tile unexpected size {img.size}, expected ({_TILE_PX}, {_TILE_PX})"
        )
    if img.mode == "P":
        indices = np.array(img, dtype=np.uint8)
        if indices.max() >= len(_DBZ_BY_INDEX):
            indices = np.clip(indices, 0, len(_DBZ_BY_INDEX) - 1)
        return _DBZ_BY_INDEX[indices]
    if img.mode in ("RGBA", "LA"):
        return _EMPTY_TILE.copy()
    raise ValueError(f"JMA tile unexpected mode {img.mode!r}")


# ── Web Mercator tile math ─────────────────────────────────────────


def lon_to_tile_x(lon: float, z: int) -> float:
    return (lon + 180.0) / 360.0 * (1 << z)


def lat_to_tile_y(lat: float, z: int) -> float:
    lat_rad = math.radians(max(min(lat, 85.05112878), -85.05112878))
    return (1.0 - math.asinh(math.tan(lat_rad)) / math.pi) / 2.0 * (1 << z)


def tile_x_to_lon(x: float, z: int) -> float:
    return x / (1 << z) * 360.0 - 180.0


def tile_y_to_lat(y: float, z: int) -> float:
    n = math.pi * (1.0 - 2.0 * y / (1 << z))
    return math.degrees(math.atan(math.sinh(n)))


def compute_tile_range(
    region: RegionDef, zoom: int,
) -> tuple[int, int, int, int]:
    """Return (x_min, x_max, y_min, y_max) inclusive tile range for a region."""
    x_w = lon_to_tile_x(region.west, zoom)
    x_e = lon_to_tile_x(region.east, zoom)
    y_n = lat_to_tile_y(region.north, zoom)
    y_s = lat_to_tile_y(region.south, zoom)
    return (
        int(math.floor(x_w)),
        int(math.floor(x_e)),
        int(math.floor(y_n)),
        int(math.floor(y_s)),
    )


def resample_to_region(
    tile_grid: dict[tuple[int, int], np.ndarray],
    zoom: int,
    region: RegionDef,
) -> np.ndarray:
    """Stitch a tile dict {(x, y): array} and resample onto region's grid.

    ``tile_grid`` keys are XYZ tile coords; values are 256×256 uint8 dBZ
    arrays from ``decode_jma_tile``.  Missing tiles are treated as zero.
    """
    x_min, x_max, y_min, y_max = compute_tile_range(region, zoom)
    nx = x_max - x_min + 1
    ny = y_max - y_min + 1

    mosaic = np.zeros((ny * _TILE_PX, nx * _TILE_PX), dtype=np.uint8)
    for (tx, ty), tile in tile_grid.items():
        col = tx - x_min
        row = ty - y_min
        if 0 <= col < nx and 0 <= row < ny:
            mosaic[
                row * _TILE_PX : (row + 1) * _TILE_PX,
                col * _TILE_PX : (col + 1) * _TILE_PX,
            ] = tile

    out_w = region.width
    out_h = region.height
    ps_x = region.pixel_size
    ps_y = region.pixel_size_y if region.pixel_size_y > 0 else region.pixel_size

    n_tiles = 1 << zoom
    lons = region.west + (np.arange(out_w) + 0.5) * ps_x
    lats = region.north - (np.arange(out_h) + 0.5) * ps_y

    map_x_tile = (lons + 180.0) / 360.0 * n_tiles
    lats_clamped = np.clip(lats, -85.05112878, 85.05112878)
    lat_rad = np.radians(lats_clamped)
    map_y_tile = (1.0 - np.arcsinh(np.tan(lat_rad)) / np.pi) / 2.0 * n_tiles

    map_x_px = (map_x_tile - x_min) * _TILE_PX
    map_y_px = (map_y_tile - y_min) * _TILE_PX

    grid_x, grid_y = np.meshgrid(map_x_px, map_y_px)

    return cv2.remap(
        mosaic,
        grid_x.astype(np.float32),
        grid_y.astype(np.float32),
        cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
