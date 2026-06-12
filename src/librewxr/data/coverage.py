# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Joshua Kimsey
"""Precomputed radar coverage masks.

At startup, build a boolean mask per region marking which pixels (in a
coarse lat/lon grid) lie within range of any radar station. The tile
renderer consults this to decide whether an empty pixel should receive
ECMWF fallback — previously we relied on ``values == 0``, but IEM N0Q
encodes "clear sky within radar range" and "outside radar range"
identically, causing either bleed-through or cutouts at the coverage
boundary.
"""
from __future__ import annotations

import logging
import math

import cv2
import numpy as np

from librewxr.data.regions import REGIONS, RegionDef

logger = logging.getLogger(__name__)

# Default effective precipitation detection range (km).  Per-source
# ``range_overrides`` (passed into ``build_coverage_masks``) replace this
# for individual regions; the OPERA C-band fleet, El Salvador's 120 km
# overlay, and CWA Taiwan's typhoon-tracking extent all do so.
DEFAULT_RADAR_RANGE_KM = 240.0

# Coarse grid resolution for coverage masks. 0.05° ≈ 5.5 km at the equator,
# much finer than the ~240 km radar range so blob edges are smooth.
MASK_RESOLUTION_DEG = 0.05

# Station coverage mask cache: region name -> (mask, west, south, dx, dy)
_COVERAGE_MASKS: dict[str, tuple[np.ndarray, float, float, float, float]] = {}


def _build_region_mask(
    region: RegionDef,
    stations: list[tuple[float, float]],
    range_km: float,
) -> None:
    """Build a boolean coverage mask for one region and store it.

    Uses an equirectangular approximation (valid for regional bboxes):
    distance ≈ sqrt((Δlat·111)² + (Δlon·111·cos(lat))²) in km.
    """
    west, east = region.west, region.east
    south, north = region.south, region.north

    # Mask grid covering the region's bbox at MASK_RESOLUTION_DEG.
    dx = MASK_RESOLUTION_DEG
    dy = MASK_RESOLUTION_DEG
    nx = max(1, int(math.ceil((east - west) / dx)))
    ny = max(1, int(math.ceil((north - south) / dy)))

    # Pixel centers
    lon_axis = west + (np.arange(nx) + 0.5) * dx
    lat_axis = south + (np.arange(ny) + 0.5) * dy

    lat_grid, lon_grid = np.meshgrid(lat_axis, lon_axis, indexing="ij")

    mask = np.zeros((ny, nx), dtype=bool)
    range_km_sq = range_km * range_km

    for st_lat, st_lon in stations:
        dlat_km = (lat_grid - st_lat) * 111.0
        # Use station's own latitude for cos factor (good enough within 240 km).
        dlon_km = (lon_grid - st_lon) * 111.0 * math.cos(math.radians(st_lat))
        d2 = dlat_km * dlat_km + dlon_km * dlon_km
        mask |= d2 <= range_km_sq

    _COVERAGE_MASKS[region.name] = (mask, west, south, dx, dy)
    logger.info(
        "coverage mask %s: %dx%d @ %.2f° (%d stations, %.1f%% covered)",
        region.name, ny, nx, MASK_RESOLUTION_DEG, len(stations),
        100.0 * mask.mean(),
    )


def _build_region_polygon_mask(
    region: RegionDef,
    polygon: list[tuple[float, float]] | list[list[tuple[float, float]]],
) -> None:
    """Build a coverage mask by rasterising a polygon over the region grid.

    ``polygon`` is either:
      - a single ring: ``list[(lat, lon)]`` of perimeter vertices, or
      - a multi-polygon: ``list[list[(lat, lon)]]`` for disjoint regions
        (e.g. Italy's mainland + Sicily + Sardinia + smaller islands, or
        DPC's open-ocean tendrils south of Sicily where neither OPERA
        nor any other neighbour covers).

    Winding direction doesn't matter — ``cv2.fillPoly`` rasterises the
    interior regardless.  Vertices are converted to mask grid pixel
    coordinates before the fill.
    """
    west, east = region.west, region.east
    south, north = region.south, region.north

    dx = MASK_RESOLUTION_DEG
    dy = MASK_RESOLUTION_DEG
    nx = max(1, int(math.ceil((east - west) / dx)))
    ny = max(1, int(math.ceil((north - south) / dy)))

    # Normalise to multi-polygon shape: list of rings.  A single-ring
    # input is detected by inspecting the first element — a ring is a
    # sequence of (lat, lon) pairs, so polygon[0] is a tuple/list of two
    # floats; a multi-polygon's polygon[0] is itself a ring.
    if polygon and isinstance(polygon[0][0], (int, float)):
        rings = [polygon]
    else:
        rings = polygon

    # Convert (lat, lon) → (col, row) in mask pixel space.  cv2.fillPoly
    # accepts a list of int32 arrays, each shaped ``(N, 2)`` in (x, y) order.
    pts = [
        np.array(
            [
                [int(round((lon - west) / dx)),
                 int(round((lat - south) / dy))]
                for lat, lon in ring
            ],
            dtype=np.int32,
        )
        for ring in rings
    ]

    canvas = np.zeros((ny, nx), dtype=np.uint8)
    cv2.fillPoly(canvas, pts, 255)
    mask = canvas > 0

    _COVERAGE_MASKS[region.name] = (mask, west, south, dx, dy)
    total_verts = sum(len(r) for r in rings)
    logger.info(
        "coverage mask %s: %dx%d @ %.2f° (polygon, %d ring(s), "
        "%d vertices, %.1f%% covered)",
        region.name, ny, nx, MASK_RESOLUTION_DEG, len(rings),
        total_verts, 100.0 * mask.mean(),
    )


def build_coverage_masks(
    station_map: dict[str, list[tuple[float, float]]],
    range_overrides: dict[str, float] | None = None,
    coverage_polygons: (
        dict[str, list[tuple[float, float]] | list[list[tuple[float, float]]]]
        | None
    ) = None,
) -> None:
    """Build coverage masks for every region with station data or a polygon.

    Polygons take precedence over station circles when both are
    provided for the same region — the polygon is the authoritative
    statement of the published product extent.

    Args:
        station_map: Mapping of region name to its contributing radar
            stations.  Typically assembled by
            ``librewxr.sources.collect_radar_coverage_metadata`` from the
            active radar providers — but any dict works, which keeps the
            mask builder testable in isolation.
        range_overrides: Optional mapping of region name to a custom
            effective range (km).  Regions absent here use
            ``DEFAULT_RADAR_RANGE_KM``.  Used by OPERA (300 km C-band
            reach), SVCOMP (120 km product), CWA TWCOMP (450 km typhoon
            buffer), and the MET Malaysia regions.
        coverage_polygons: Optional mapping of region name to a polygon
            describing the published coverage extent.  Used by
            gauge-corrected QPE composites whose product extent
            doesn't match individual Doppler ranges — JMA HRPN's tile
            pyramid traces a tilted polygon along the archipelago,
            extending well past 240 km Doppler reach into the offshore
            Pacific.  Vertices are ``(latitude, longitude)`` tuples.
            Each region's value is either a single ring (``list[(lat,
            lon)]``) for a connected coverage shape or a list of rings
            (``list[list[(lat, lon)]]``) for disjoint coverage —
            e.g. DPC Italy's mainland + Sicily + Sardinia + open-ocean
            tendrils south of Sicily where neither OPERA nor any other
            neighbour covers.
    """
    range_overrides = range_overrides or {}
    coverage_polygons = coverage_polygons or {}

    for region_name, polygon in coverage_polygons.items():
        region = REGIONS.get(region_name)
        if region is None:
            continue
        _build_region_polygon_mask(region, polygon)

    for region_name, stations in station_map.items():
        if region_name in coverage_polygons:
            continue
        region = REGIONS.get(region_name)
        if region is None:
            continue
        range_km = range_overrides.get(region_name, DEFAULT_RADAR_RANGE_KM)
        _build_region_mask(region, stations, range_km)


def sample_coverage(
    region_name: str, lat_grid: np.ndarray, lon_grid: np.ndarray,
) -> np.ndarray:
    """Return a boolean array: True where the point is within radar range.

    ``lat_grid`` and ``lon_grid`` have matching shape.  If no station
    mask exists for the region — the convention used by gauge-corrected
    QPE composites whose product extent is defined upstream (JMA HRPN,
    MRMS-style fusions) rather than by individual Doppler ranges — the
    region's full bbox is treated as covered.  Still bbox-bounded so a
    tile straddling the region edge correctly hands off to the NWP
    chain outside.
    """
    entry = _COVERAGE_MASKS.get(region_name)
    if entry is None:
        region = REGIONS.get(region_name)
        if region is None:
            return np.ones(lat_grid.shape, dtype=bool)
        return (
            (lon_grid >= region.west)
            & (lon_grid <= region.east)
            & (lat_grid >= region.south)
            & (lat_grid <= region.north)
        )

    mask, west, south, dx, dy = entry
    ny, nx = mask.shape

    col = np.floor((lon_grid - west) / dx).astype(np.int32)
    row = np.floor((lat_grid - south) / dy).astype(np.int32)

    in_bounds = (col >= 0) & (col < nx) & (row >= 0) & (row < ny)
    # Clamp for safe indexing, then mask out-of-bounds to False.
    col_c = np.clip(col, 0, nx - 1)
    row_c = np.clip(row, 0, ny - 1)
    result = mask[row_c, col_c]
    return result & in_bounds


# ---------------------------------------------------------------------------
# Feather masks: distance-transform gradient at coverage boundaries
# ---------------------------------------------------------------------------
# Used by nowcast blending to smoothly transition between extrapolated radar
# and IFS forecast at the edge of radar coverage, preventing hard seams.

# Distance (in coverage mask pixels) over which the feather ramps from 0 to 1.
# At MASK_RESOLUTION_DEG=0.05°, 15 pixels ≈ 0.75° ≈ 80 km at mid-latitudes.
FEATHER_DISTANCE_PX = 15

# Feather mask cache: region name -> (feather, west, south, dx, dy)
_FEATHER_MASKS: dict[str, tuple[np.ndarray, float, float, float, float]] = {}


def build_feather_masks() -> None:
    """Build feather masks from existing coverage masks.

    Must be called after ``build_coverage_masks()``.  For each region
    with a coverage mask, computes a distance transform from the mask
    boundary inward and normalizes to 0–1 over ``FEATHER_DISTANCE_PX``.
    """
    for region_name, (mask, west, south, dx, dy) in _COVERAGE_MASKS.items():
        # cv2.distanceTransform needs uint8: 255 inside coverage, 0 outside
        mask_uint8 = mask.astype(np.uint8) * 255
        dist = cv2.distanceTransform(mask_uint8, cv2.DIST_L2, 5)
        feather = np.clip(dist / FEATHER_DISTANCE_PX, 0.0, 1.0).astype(np.float32)
        _FEATHER_MASKS[region_name] = (feather, west, south, dx, dy)
        logger.info(
            "feather mask %s: %dx%d, feather_px=%d",
            region_name, feather.shape[0], feather.shape[1], FEATHER_DISTANCE_PX,
        )


def sample_feather(
    region_name: str, lat_grid: np.ndarray, lon_grid: np.ndarray,
) -> np.ndarray:
    """Return a float array 0–1: how far inside radar coverage each point is.

    0.0 = at the coverage boundary or outside,
    1.0 = well inside coverage (≥ ``FEATHER_DISTANCE_PX`` mask pixels from edge).

    For regions with no feather mask — gauge-corrected QPE composites
    that skip the station-circle mask in the first place (see
    ``sample_coverage``) — returns 1.0 inside the region's bbox and 0.0
    outside.  No soft transition; the radar simply dominates inside the
    region and hands off to NWP at the edge.
    """
    entry = _FEATHER_MASKS.get(region_name)
    if entry is None:
        region = REGIONS.get(region_name)
        if region is None:
            return np.ones(lat_grid.shape, dtype=np.float32)
        inside = (
            (lon_grid >= region.west)
            & (lon_grid <= region.east)
            & (lat_grid >= region.south)
            & (lat_grid <= region.north)
        )
        return inside.astype(np.float32)

    feather, west, south, dx, dy = entry
    ny, nx = feather.shape

    col = np.floor((lon_grid - west) / dx).astype(np.int32)
    row = np.floor((lat_grid - south) / dy).astype(np.int32)

    in_bounds = (col >= 0) & (col < nx) & (row >= 0) & (row < ny)
    col_c = np.clip(col, 0, nx - 1)
    row_c = np.clip(row, 0, ny - 1)
    result = feather[row_c, col_c]
    result[~in_bounds] = 0.0
    return result
