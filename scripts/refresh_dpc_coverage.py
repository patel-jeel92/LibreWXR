#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Joshua Kimsey
"""Refresh the static DPC Italy coverage polygon.

Unlike JMA, DPC does not publish their own coverage shape — the
``findLastProductByType?type=SITES`` endpoint exists but
``/downloadProduct`` rejects it.  We derive the polygon offline from:

  1. Natural Earth ``ne_10m_admin_0_countries`` (public domain) — the
     authoritative shape of Italy proper, including Sicily, Sardinia,
     and the smaller inhabited islands.
  2. The DPC station list in ``stations.py`` — 24 sites, 150 km
     effective reach each.
  3. The OPERA station list in
     ``sources/regional/europe/radar/opera/stations.py`` — 233 sites,
     300 km reach each.

The build rule (Shapely set operations, all in EPSG:3035 LAEA so the
buffer is meter-accurate across Italy's 35–48° latitude span):

    A = italy_admin0.buffer(30 km)              # always-DPC core
    D = union(150 km circle per DPC station)    # actual DPC reach
    O = union(300 km circle per OPERA station)  # neighbour reach

    coverage = D ∩ (A ∪ ¬O)
             = (D ∩ A) ∪ (D − O)

That gives DPC priority over Italy + 30 km buffer (where the published
composite is most trustworthy) and full 150 km reach into the open
ocean south / east of Italy where no OPERA neighbour can fill behind.
Where OPERA *can* fill (Adriatic, Ligurian, around Slovenia / France /
Switzerland / Croatia), DPC's claim is tight against the Italian
coast — no Alpine donut, no W. Austria hole.

# Regenerate with:
#   python3 -m venv /tmp/dpc-refresh-venv
#   /tmp/dpc-refresh-venv/bin/pip install httpx shapely pyproj pyshp
#   /tmp/dpc-refresh-venv/bin/python scripts/refresh_dpc_coverage.py

The polygon is quasi-static — only changes when DPC adds/removes a
radar, OPERA gains/loses a member, or Natural Earth pushes a coastline
update.  Refresh manually at maintenance time, not at every startup.
"""
from __future__ import annotations

import io
import json
import math
import sys
import zipfile
from pathlib import Path

import httpx
from shapely.geometry import (
    MultiPolygon,
    Point,
    Polygon,
    shape,
)
from shapely.ops import transform, unary_union
from pyproj import CRS, Transformer


# Natural Earth 1:10m Cultural — public domain.  10m is overkill for our
# 0.05° mask grid but the file is small (~5 MB zipped) and the extra
# precision survives any future tightening of the buffer.
NE_URL = (
    "https://naciscdn.org/naturalearth/10m/cultural/"
    "ne_10m_admin_0_countries.zip"
)

DPC_RANGE_KM = 150.0
OPERA_RANGE_KM = 300.0
ITALY_BUFFER_KM = 30.0
SIMPLIFY_TOLERANCE_DEG = 0.005  # ~500 m; well below the 0.05° mask grid

# WGS84 → Europe LAEA (EPSG:3035) for meter-accurate buffer / circle
# generation across the latitude span Italy occupies.
WGS84 = CRS.from_epsg(4326)
LAEA = CRS.from_epsg(3035)
TO_LAEA = Transformer.from_crs(WGS84, LAEA, always_xy=True).transform
TO_WGS84 = Transformer.from_crs(LAEA, WGS84, always_xy=True).transform

REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_ROOT = REPO_ROOT / "src" / "librewxr"
OUTPUT_PATH = (
    SRC_ROOT / "sources" / "regional" / "europe" / "italy" / "radar" / "dpc"
    / "dpc_coverage.geojson"
)


def _load_stations(rel_path: str) -> list[tuple[float, float]]:
    """Import ``STATIONS`` from one of the source packages.

    Done with an exec-on-source dance rather than a regular import so
    this script stays runnable from the throwaway venv (which doesn't
    have the project installed) — only the standard library is touched
    during evaluation because both ``stations.py`` files just declare
    list literals.
    """
    text = (SRC_ROOT / rel_path).read_text()
    ns: dict[str, object] = {"__name__": "__stations__"}
    # Strip the ``from __future__ import annotations`` line so the
    # module evaluates standalone; it's harmless either way but the
    # other top-level imports (json, Path) would resolve fine too.
    exec(compile(text, str(SRC_ROOT / rel_path), "exec"), ns)
    stations = ns.get("STATIONS")
    if not isinstance(stations, list) or not stations:
        raise RuntimeError(f"{rel_path}: STATIONS missing or empty")
    return [(float(lat), float(lon)) for lat, lon in stations]


def fetch_italy_admin0() -> MultiPolygon | Polygon:
    """Download Natural Earth admin-0 and return Italy's geometry."""
    print(f"  fetching {NE_URL}")
    resp = httpx.get(NE_URL, timeout=120.0, follow_redirects=True)
    resp.raise_for_status()
    zf = zipfile.ZipFile(io.BytesIO(resp.content))
    # Natural Earth ships a shapefile bundle.  Read the .shp + .dbf
    # directly via a minimal in-memory shapefile reader — avoiding a
    # full geopandas dep keeps the throwaway venv lean (httpx + shapely
    # + pyproj only).
    shp_name = next(n for n in zf.namelist() if n.endswith(".shp"))
    dbf_name = shp_name[:-4] + ".dbf"
    import shapefile  # pyshp — pull in only if we got here

    with zf.open(shp_name) as shp_f, zf.open(dbf_name) as dbf_f:
        # pyshp needs seekable file objects; copy to BytesIO.
        reader = shapefile.Reader(
            shp=io.BytesIO(shp_f.read()),
            dbf=io.BytesIO(dbf_f.read()),
        )
        fields = [f[0] for f in reader.fields[1:]]  # skip deletion flag
        name_idx = fields.index("ADMIN")
        for record, shape_rec in zip(reader.records(), reader.shapes()):
            if record[name_idx] == "Italy":
                geom = shape(shape_rec.__geo_interface__)
                if not isinstance(geom, (Polygon, MultiPolygon)):
                    raise RuntimeError(
                        f"unexpected Italy geometry type: {type(geom).__name__}"
                    )
                return geom
    raise RuntimeError("Italy not found in Natural Earth admin-0")


def km_to_laea_buffer(km: float) -> float:
    """LAEA is in metres."""
    return km * 1000.0


def build_station_union(
    stations: list[tuple[float, float]], range_km: float,
) -> Polygon | MultiPolygon:
    """Union of range-km circles around each station, in LAEA coords."""
    circles = []
    radius_m = km_to_laea_buffer(range_km)
    for lat, lon in stations:
        x, y = TO_LAEA(lon, lat)
        # 64 segments per quarter circle is plenty for a 150 km radius
        # at our 0.005° simplification tolerance.
        circles.append(Point(x, y).buffer(radius_m, quad_segs=64))
    return unary_union(circles)


def build_coverage(
    italy: MultiPolygon | Polygon,
    dpc_stations: list[tuple[float, float]],
    opera_stations: list[tuple[float, float]],
) -> MultiPolygon | Polygon:
    """Apply the (D ∩ A) ∪ (D − O) rule and return the WGS84 geometry."""
    italy_laea = transform(TO_LAEA, italy)
    italy_buffered_laea = italy_laea.buffer(
        km_to_laea_buffer(ITALY_BUFFER_KM),
        quad_segs=32,
    )
    dpc_union_laea = build_station_union(dpc_stations, DPC_RANGE_KM)
    opera_union_laea = build_station_union(opera_stations, OPERA_RANGE_KM)

    core = dpc_union_laea.intersection(italy_buffered_laea)
    ocean_tendrils = dpc_union_laea.difference(opera_union_laea)
    coverage_laea = unary_union([core, ocean_tendrils])

    # Back to WGS84 for storage / rasterisation.
    coverage_wgs84 = transform(TO_WGS84, coverage_laea)

    # Cheap post-clean: drop slivers (< ~10 km²) introduced by floating-
    # point boundary jitter.  10 km² is far below any radar pixel; the
    # mask grid is ~30 km² per cell.
    if isinstance(coverage_wgs84, MultiPolygon):
        keepers = [
            poly for poly in coverage_wgs84.geoms
            if _approx_area_km2(poly) >= 10.0
        ]
        coverage_wgs84 = (
            MultiPolygon(keepers) if len(keepers) > 1 else keepers[0]
        )
    return coverage_wgs84


def _approx_area_km2(poly: Polygon) -> float:
    """Quick equirectangular area estimate, good enough for sliver test."""
    cy = poly.centroid.y
    deg_area = poly.area
    km_per_deg_lat = 111.0
    km_per_deg_lon = 111.0 * math.cos(math.radians(cy))
    return deg_area * km_per_deg_lat * km_per_deg_lon


def simplify(
    geom: MultiPolygon | Polygon, tolerance_deg: float,
) -> MultiPolygon | Polygon:
    return geom.simplify(tolerance_deg, preserve_topology=True)


def to_rings(
    geom: MultiPolygon | Polygon,
) -> list[list[tuple[float, float]]]:
    """Flatten a (Multi)Polygon to a list of exterior rings ([lon, lat])."""
    polys = list(geom.geoms) if isinstance(geom, MultiPolygon) else [geom]
    return [list(p.exterior.coords) for p in polys]


def write_geojson(
    geom: MultiPolygon | Polygon,
    output_path: Path,
) -> None:
    """Write the coverage geometry as a single MultiPolygon feature.

    Vertex order is ``[lon, lat]`` per the GeoJSON spec; the runtime
    loader in ``stations.py`` swaps to ``(lat, lon)`` for project
    convention.
    """
    rings = to_rings(geom)
    coords = [[list(r)] for r in rings]
    if isinstance(geom, Polygon):
        geometry = {
            "type": "Polygon",
            "coordinates": coords[0],
        }
    else:
        geometry = {
            "type": "MultiPolygon",
            "coordinates": coords,
        }
    feature = {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "properties": {
                    "name": "DPC Italy coverage",
                    "source": (
                        "Natural Earth 10m admin-0 (Italy) ∪ DPC station "
                        "150 km union, clipped by OPERA 300 km union "
                        "outside Italy + 30 km buffer"
                    ),
                    "natural_earth_url": NE_URL,
                    "italy_buffer_km": ITALY_BUFFER_KM,
                    "dpc_range_km": DPC_RANGE_KM,
                    "opera_range_km": OPERA_RANGE_KM,
                    "simplify_tolerance_deg": SIMPLIFY_TOLERANCE_DEG,
                    "ring_count": len(rings),
                    "vertex_count": sum(len(r) for r in rings),
                },
                "geometry": geometry,
            }
        ],
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w") as f:
        json.dump(feature, f, indent=2)
        f.write("\n")


def main() -> int:
    print("[1/5] Loading DPC + OPERA station lists")
    dpc = _load_stations(
        "sources/regional/europe/italy/radar/dpc/stations.py"
    )
    opera = _load_stations(
        "sources/regional/europe/radar/opera/stations.py"
    )
    print(f"      DPC: {len(dpc)} stations  OPERA: {len(opera)} stations")

    print("[2/5] Fetching Italy admin-0 from Natural Earth")
    italy = fetch_italy_admin0()
    print(f"      Italy: {type(italy).__name__} "
          f"({len(list(italy.geoms)) if isinstance(italy, MultiPolygon) else 1} parts)")

    print(f"[3/5] Building coverage: (D ∩ Italy+{ITALY_BUFFER_KM:g}km) ∪ (D − O)")
    coverage = build_coverage(italy, dpc, opera)
    raw_rings = to_rings(coverage)
    print(f"      raw: {len(raw_rings)} ring(s), "
          f"{sum(len(r) for r in raw_rings)} vertices")

    print(f"[4/5] Simplifying with Douglas-Peucker at {SIMPLIFY_TOLERANCE_DEG}°")
    coverage = simplify(coverage, SIMPLIFY_TOLERANCE_DEG)
    rings = to_rings(coverage)
    print(f"      simplified: {len(rings)} ring(s), "
          f"{sum(len(r) for r in rings)} vertices")

    print(f"[5/5] Writing {OUTPUT_PATH.relative_to(REPO_ROOT)}")
    write_geojson(coverage, OUTPUT_PATH)
    size_kb = OUTPUT_PATH.stat().st_size / 1024
    print(f"      wrote {size_kb:.1f} KB")
    return 0


if __name__ == "__main__":
    sys.exit(main())
