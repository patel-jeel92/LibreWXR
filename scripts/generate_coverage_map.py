#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Joshua Kimsey
"""Generate the LibreWXR coverage maps.

Writes two PNGs to ``docs/``:

  * ``coverage-map-radar.png``  — radar composites
  * ``coverage-map-models.png`` — regional NWP grids

ECMWF IFS provides global coverage and is not drawn.

One-off authoring tool — not a runtime dependency.  Regenerate after
adding or removing a radar source or regional NWP grid.

Each polygon is rendered by walking its grid's perimeter in the
source's native projection (LCC, polar stereographic, LAEA, rotated
lat/lon, or regular lat/lon) and inverse-projecting each edge sample
to WGS84 lat/lon, so the resulting outline follows the actual curved
domain shape rather than a misleading lat/lon bounding box.

Basemap: Natural Earth Vector 1:110m country polygons (CC0).

# Regenerate with:
#   python3 -m venv /tmp/coverage-map-venv
#   /tmp/coverage-map-venv/bin/pip install matplotlib pyproj
#   curl -L https://raw.githubusercontent.com/nvkelso/natural-earth-vector/master/geojson/ne_110m_admin_0_countries.geojson -o /tmp/ne_countries.geojson
#   /tmp/coverage-map-venv/bin/python scripts/generate_coverage_map.py
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from math import asin, atan2, cos, degrees, radians, sin
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch, PathPatch
from matplotlib.path import Path as MplPath
from pyproj import CRS, Transformer

REPO_ROOT = Path(__file__).resolve().parent.parent
RADAR_OUTPUT = REPO_ROOT / "docs" / "coverage-map-radar.png"
MODEL_OUTPUT = REPO_ROOT / "docs" / "coverage-map-models.png"
BASEMAP_PATH = Path("/tmp/ne_countries.geojson")


# ── Coverage source definitions ────────────────────────────────────────


@dataclass(frozen=True)
class Source:
    """A coverage polygon — radar composite or regional NWP grid."""
    label: str
    color: str
    polygon: np.ndarray  # shape (N, 2), [lon, lat]


def project_grid_perimeter(
    crs_grid: CRS,
    x_min: float, x_max: float, y_min: float, y_max: float,
    samples_per_edge: int = 80,
) -> np.ndarray:
    """Walk a rectangular grid's perimeter in projected (x, y) space and
    inverse-project to (lon, lat).  Smooth enough to capture curvature
    of LCC / polar stereographic / LAEA projections in geographic
    coordinates."""
    transformer = Transformer.from_crs(crs_grid, "EPSG:4326", always_xy=True)
    edge_x = np.linspace(x_min, x_max, samples_per_edge)
    edge_y = np.linspace(y_min, y_max, samples_per_edge)
    xs = np.concatenate([
        edge_x,                              # top
        np.full(samples_per_edge, x_max),    # right
        edge_x[::-1],                        # bottom
        np.full(samples_per_edge, x_min),    # left
    ])
    ys = np.concatenate([
        np.full(samples_per_edge, y_max),
        edge_y[::-1],
        np.full(samples_per_edge, y_min),
        edge_y,
    ])
    lon, lat = transformer.transform(xs, ys)
    return np.column_stack([lon, lat])


def latlon_box(west: float, east: float, south: float, north: float) -> np.ndarray:
    """Simple geographic bounding-box polygon."""
    return np.array([
        [west, north], [east, north], [east, south], [west, south], [west, north],
    ])


# Explicit rotated-pole inverse via Cartesian rotation on the unit
# sphere.  pyproj's ob_tran proved fiddly with the ECCC encoding —
# rolling our own keeps the convention obvious.
def _rotated_inv_xy(
    rlat: np.ndarray, rlon: np.ndarray,
    pole_lat: float, pole_lon: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Rotated lat/lon (deg) → geographic lat/lon (deg).

    Inverse of the COSMO / WMO north-pole rotated_ll forward used by
    the HRDPS grid module: the rotated north pole sits at geographic
    (pole_lat, pole_lon), and rotated coordinates measure latitude
    from that pole.
    """
    phi_r = np.radians(rlat)
    lam_r = np.radians(rlon)
    # Cartesian point on unit sphere in the rotated frame.
    x = np.cos(phi_r) * np.cos(lam_r)
    y = np.cos(phi_r) * np.sin(lam_r)
    z = np.sin(phi_r)
    # Reverse step: rotate by +(90 - pole_lat) about the y-axis to
    # restore the rotated pole back to its geographic latitude.
    cb = cos(radians(90 - pole_lat))
    sb = sin(radians(90 - pole_lat))
    x1 = cb * x + sb * z
    y1 = y
    z1 = -sb * x + cb * z
    # Reverse step: rotate by +pole_lon about the z-axis to restore
    # the rotated pole back to its geographic longitude.
    ca = cos(radians(pole_lon))
    sa = sin(radians(pole_lon))
    x2 = ca * x1 - sa * y1
    y2 = sa * x1 + ca * y1
    z2 = z1
    lat = np.degrees(np.arcsin(np.clip(z2, -1.0, 1.0)))
    lon = np.degrees(np.arctan2(y2, x2))
    return lat, lon


def _rotated_fwd_xy(
    lat: float, lon: float, pole_lat: float, pole_lon: float,
) -> tuple[float, float]:
    """Scalar forward rotation, used to compute rotated bounds from a
    known geographic corner."""
    phi = radians(lat); lam = radians(lon)
    x = cos(phi) * cos(lam); y = cos(phi) * sin(lam); z = sin(phi)
    # Step 1: rotate by pole_lon about z.
    ca = cos(radians(pole_lon)); sa = sin(radians(pole_lon))
    x1 = ca * x + sa * y; y1 = -sa * x + ca * y; z1 = z
    # Step 2: rotate by (90 - pole_lat) about y.
    cb = cos(radians(90 - pole_lat)); sb = sin(radians(90 - pole_lat))
    x2 = cb * x1 - sb * z1; y2 = y1; z2 = sb * x1 + cb * z1
    return degrees(asin(z2)), degrees(atan2(y2, x2))


def rotated_pole_polygon(
    pole_lat: float, pole_lon: float,
    rlat_min: float, rlat_max: float,
    rlon_min: float, rlon_max: float,
    samples_per_edge: int = 80,
) -> np.ndarray:
    """Perimeter polygon for a rotated-pole grid, in geographic coords."""
    edge_rlon = np.linspace(rlon_min, rlon_max, samples_per_edge)
    edge_rlat = np.linspace(rlat_min, rlat_max, samples_per_edge)
    rlons = np.concatenate([
        edge_rlon,
        np.full(samples_per_edge, rlon_max),
        edge_rlon[::-1],
        np.full(samples_per_edge, rlon_min),
    ])
    rlats = np.concatenate([
        np.full(samples_per_edge, rlat_max),
        edge_rlat[::-1],
        np.full(samples_per_edge, rlat_min),
        edge_rlat,
    ])
    lat, lon = _rotated_inv_xy(rlats, rlons, pole_lat, pole_lon)
    return np.column_stack([lon, lat])


# ── Source definitions ─────────────────────────────────────────────────


def build_radar_sources() -> list[Source]:
    radar: list[Source] = []
    # MRMS — five lat/lon composites all coloured the same since they
    # share an upstream operator.
    mrms_color = "#1f77b4"
    for label, w, e, s, n in [
        ("MRMS — CONUS",       -126.0, -65.0, 23.0, 50.0),
        ("MRMS — Alaska",      -170.5, -130.5, 53.2, 68.7),
        ("MRMS — Hawaii",      -162.4, -152.4, 15.4, 24.4),
        ("MRMS — Puerto Rico", -71.1, -61.1, 13.1, 23.1),
        ("MRMS — Guam",        140.5, 149.0, 9.2, 17.7),
    ]:
        radar.append(Source(label, mrms_color, latlon_box(w, e, s, n)))

    radar.append(Source(
        "MSC Canada radar", "#17becf",
        latlon_box(-141.0, -52.0, 41.0, 84.0),
    ))

    # OPERA — LAEA projection, full grid extent
    opera_crs = CRS.from_proj4(
        "+proj=laea +lat_0=55 +lon_0=10 +x_0=1950000 +y_0=-2100000 +ellps=WGS84"
    )
    radar.append(Source(
        "OPERA (Europe)", "#9467bd",
        project_grid_perimeter(opera_crs, 0, 3_800_000, -4_400_000, 0),
    ))
    return radar


def build_model_sources() -> list[Source]:
    models: list[Source] = []

    # HRRR-CONUS (LCC)
    hrrr_crs = CRS.from_proj4(
        "+proj=lcc +lat_0=38.5 +lon_0=-97.5 +lat_1=38.5 +lat_2=38.5 "
        "+R=6371229 +no_defs"
    )
    hrrr_x_max = -2699020.143 + 1798 * 3000
    hrrr_y_min = 1588193.847 - 1058 * 3000
    models.append(Source(
        "HRRR-CONUS", "#d62728",
        project_grid_perimeter(
            hrrr_crs, -2699020.143, hrrr_x_max, hrrr_y_min, 1588193.847,
        ),
    ))

    # HRRR-Alaska (polar stereographic)
    hrrr_ak_crs = CRS.from_proj4(
        "+proj=stere +lat_0=90 +lat_ts=60 +lon_0=-135 +R=6371229 +no_defs"
    )
    ak_x_max = -3425051.0 + 1298 * 3000
    ak_y_min = -1344804.1 - 918 * 3000
    models.append(Source(
        "HRRR-Alaska", "#ff9896",
        project_grid_perimeter(
            hrrr_ak_crs, -3425051.0, ak_x_max, ak_y_min, -1344804.1,
        ),
    ))

    # HRDPS (rotated lat/lon).  Pole at (36.08852°N, 65.30514°E); grid
    # 2540 x 1290 at 0.0225° rotated spacing.  Compute rotated bounds
    # by forward-projecting two known geographic corners (SW and SE)
    # and walking the perimeter between them — this script's Cartesian
    # rotation convention differs by 180° from the closed-form COSMO
    # formulas the HRDPS module uses, so spans look "backwards" relative
    # to the module's documented orientation but the resulting polygon
    # is identical in geographic coordinates.
    pole_lat, pole_lon = 36.08852, 65.30514
    rlat_sw, rlon_sw = _rotated_fwd_xy(39.6260, -133.6295, pole_lat, pole_lon)
    rlat_se, rlon_se = _rotated_fwd_xy(27.285, -66.966, pole_lat, pole_lon)
    # SW → SE crosses the ±180° rlon line; unwrap the SE rlon so
    # linspace traces the correct (short) arc rather than the long way
    # around the rotated sphere.
    if rlon_se < rlon_sw:
        rlon_se += 360.0
    rlat_min = rlat_sw
    rlat_max = rlat_sw + 1289 * 0.0225
    models.append(Source(
        "HRDPS (Canada)", "#e377c2",
        rotated_pole_polygon(
            pole_lat=pole_lat, pole_lon=pole_lon,
            rlat_min=rlat_min, rlat_max=rlat_max,
            rlon_min=rlon_sw, rlon_max=rlon_se,
        ),
    ))

    # DMI HARMONIE DINI (LCC over Europe)
    dini_crs = CRS.from_proj4(
        "+proj=lcc +lat_0=55.5 +lon_0=-8.0 +lat_1=55.5 +lat_2=55.5 "
        "+R=6371229 +no_defs"
    )
    dini_fwd = Transformer.from_crs("EPSG:4326", dini_crs, always_xy=True)
    dini_x0, dini_y0 = dini_fwd.transform(-25.422, 39.671)
    dini_x_max = dini_x0 + 1905 * 2000
    dini_y_max = dini_y0 + 1605 * 2000
    models.append(Source(
        "DMI HARMONIE DINI", "#2ca02c",
        project_grid_perimeter(dini_crs, dini_x0, dini_x_max, dini_y0, dini_y_max),
    ))

    # DWD ICON-EU (regular lat/lon)
    models.append(Source(
        "DWD ICON-EU", "#bcbd22",
        latlon_box(-23.5, 62.5, 29.5, 70.5),
    ))

    # AROME Antilles (regular lat/lon, longitudes shifted from 0..360E)
    models.append(Source(
        "AROME Antilles", "#ff7f0e",
        latlon_box(-75.3, -51.7, 9.7, 22.9),
    ))

    # WRF-SMN Argentina (LCC, southern hemisphere, 6,370 km sphere)
    smn_crs = CRS.from_proj4(
        "+proj=lcc +lat_0=-35 +lon_0=-65 +lat_1=-35 +lat_2=-35 "
        "+R=6370000 +no_defs"
    )
    smn_fwd = Transformer.from_crs("EPSG:4326", smn_crs, always_xy=True)
    smn_x0, smn_y0 = smn_fwd.transform(-94.3308, -54.3868)
    smn_x_max = smn_x0 + 998 * 4000
    smn_y_max = smn_y0 + 1248 * 4000
    models.append(Source(
        "WRF-SMN Argentina", "#8c564b",
        project_grid_perimeter(smn_crs, smn_x0, smn_x_max, smn_y0, smn_y_max),
    ))

    return models


# ── Rendering ─────────────────────────────────────────────────────────


def load_countries_polygons(path: Path) -> list[list[np.ndarray]]:
    """Return list of polygons; each is a list of rings (outer + holes)."""
    with path.open() as f:
        gj = json.load(f)
    polygons: list[list[np.ndarray]] = []
    for feature in gj["features"]:
        geom = feature["geometry"]
        if geom is None:
            continue
        gtype = geom["type"]
        if gtype == "Polygon":
            polygons.append([np.asarray(r) for r in geom["coordinates"]])
        elif gtype == "MultiPolygon":
            for poly in geom["coordinates"]:
                polygons.append([np.asarray(r) for r in poly])
    return polygons


def _draw_basemap(ax) -> None:
    for rings in load_countries_polygons(BASEMAP_PATH):
        outer = rings[0]
        if outer.shape[0] < 3:
            continue
        ax.add_patch(PathPatch(
            MplPath(outer), facecolor="#f7f3eb",
            edgecolor="#8b8b8b", linewidth=0.4, zorder=1,
        ))


def _draw_polygon(ax, src: Source, alpha_fill: float, hatch: str | None) -> None:
    """Split on antimeridian wraps so a polygon spanning ±180 doesn't
    smear horizontally across the plot."""
    poly = src.polygon
    lon = poly[:, 0]
    wrap_idx = np.where(np.abs(np.diff(lon)) > 180)[0]
    if len(wrap_idx) == 0:
        segments = [poly]
    else:
        segments = []
        start = 0
        for idx in wrap_idx:
            segments.append(poly[start:idx + 1])
            start = idx + 1
        segments.append(poly[start:])
    for seg in segments:
        if seg.shape[0] < 3:
            continue
        ax.fill(
            seg[:, 0], seg[:, 1],
            facecolor=src.color, edgecolor=src.color,
            alpha=alpha_fill, linewidth=1.2,
            hatch=hatch, zorder=2,
        )


def render(
    sources: list[Source],
    output_path: Path,
    title: str,
    subtitle: str,
    legend_title: str,
    alpha_fill: float = 0.45,
    hatch: str | None = None,
    dedupe_label_prefix: str | None = None,
) -> None:
    """Render one map with the given polygon set.

    ``dedupe_label_prefix`` collapses multiple entries that share a
    common prefix (e.g. "MRMS — CONUS", "MRMS — Alaska") into one
    legend entry titled with the prefix.
    """
    fig, ax = plt.subplots(figsize=(16, 9), dpi=120)
    ax.set_xlim(-180, 180)
    ax.set_ylim(-65, 85)
    ax.set_aspect(1.3)
    ax.set_facecolor("#e8f1f8")

    _draw_basemap(ax)
    for src in sources:
        _draw_polygon(ax, src, alpha_fill=alpha_fill, hatch=hatch)

    ax.set_xticks(np.arange(-180, 181, 30))
    ax.set_yticks(np.arange(-60, 81, 30))
    ax.grid(True, which="major", color="#c0c0c0", linewidth=0.4, zorder=0)
    ax.tick_params(axis="both", labelsize=8)
    ax.set_xlabel("Longitude (°)", fontsize=9)
    ax.set_ylabel("Latitude (°)", fontsize=9)
    ax.set_title(f"{title}\n{subtitle}", fontsize=13, pad=12)

    # Legend
    seen: set[str] = set()
    handles: list[Patch] = []
    for s in sources:
        key = s.label
        if dedupe_label_prefix and s.label.startswith(dedupe_label_prefix):
            key = dedupe_label_prefix.rstrip(" —")
        if key in seen:
            continue
        seen.add(key)
        label = key
        if dedupe_label_prefix and key == dedupe_label_prefix.rstrip(" —"):
            # Count how many entries share the prefix
            cnt = sum(1 for x in sources if x.label.startswith(dedupe_label_prefix))
            label = f"{key} ({cnt} composites)"
        handles.append(Patch(
            facecolor=s.color, edgecolor=s.color,
            alpha=min(0.95, alpha_fill + 0.10), hatch=hatch,
            label=label,
        ))
    ax.legend(
        handles=handles, loc="lower left",
        title=legend_title, fontsize=9, title_fontsize=10,
        framealpha=0.95,
    )

    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight", dpi=120, facecolor="white")
    print(f"Wrote {output_path}")


if __name__ == "__main__":
    if not BASEMAP_PATH.exists():
        raise SystemExit(
            f"Basemap not found at {BASEMAP_PATH}. Download it first:\n"
            "  curl -L https://raw.githubusercontent.com/nvkelso/"
            "natural-earth-vector/master/geojson/"
            "ne_110m_admin_0_countries.geojson -o /tmp/ne_countries.geojson"
        )

    RADAR_OUTPUT.parent.mkdir(parents=True, exist_ok=True)

    render(
        sources=build_radar_sources(),
        output_path=RADAR_OUTPUT,
        title="LibreWXR — Radar Composite Coverage",
        subtitle="NOAA MRMS · MSC Canada · OPERA Europe",
        legend_title="Radar composites",
        alpha_fill=0.40,
        hatch="//",
        dedupe_label_prefix="MRMS — ",
    )
    render(
        sources=build_model_sources(),
        output_path=MODEL_OUTPUT,
        title="LibreWXR — Regional NWP Model Coverage",
        subtitle="Chain-dispatched per pixel; ECMWF IFS provides global coverage everywhere else",
        legend_title="Regional NWP models",
        alpha_fill=0.50,
        hatch=None,
    )
