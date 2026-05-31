#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Joshua Kimsey
"""Refresh the static JMA HRPN coverage polygon.

Downloads JMA's authoritative ``hrpns_nd`` no-data polygon from the
nowcast tile-server, extracts its inner ring (the JPCOMP coverage
hole), simplifies it with Douglas-Peucker, and writes a compact
GeoJSON to
``src/librewxr/sources/regional/east_asia/japan/radar/jma/jpcomp_coverage.geojson``.

The runtime loads this file at import time as ``JPCOMP_COVERAGE_POLYGON``.
The polygon is quasi-static (only changes when JMA's HRPN network
itself changes, i.e. a new radar site or coverage extension), so we
fetch once at maintenance time rather than at every startup.

# Regenerate with:
#   python3 -m venv /tmp/jma-refresh-venv
#   /tmp/jma-refresh-venv/bin/pip install httpx shapely
#   /tmp/jma-refresh-venv/bin/python scripts/refresh_jma_coverage.py

JMA URL convention (discovered by inspecting bosai/en_nowc/ XML config):
  https://www.jma.go.jp/bosai/jmatile/data/nowc/targetTimes_N1.json
    -> array of {basetime, validtime, elements: [..., "hrpns_nd"]}
  https://www.jma.go.jp/bosai/jmatile/data/nowc/{basetime}/none/{validtime}/surf/hrpns_nd/data.geojson
    -> gzipped GeoJSON: single Polygon feature with
       outer ring = world bbox (no-data everywhere)
       inner ring = HRPN coverage area (the "hole" in the no-data polygon)
"""
from __future__ import annotations

import gzip
import json
import sys
from pathlib import Path

import httpx
from shapely.geometry import LinearRing


TIMES_URL = "https://www.jma.go.jp/bosai/jmatile/data/nowc/targetTimes_N1.json"
GEOJSON_URL_TEMPLATE = (
    "https://www.jma.go.jp/bosai/jmatile/data/nowc/"
    "{basetime}/none/{validtime}/surf/hrpns_nd/data.geojson"
)
SIMPLIFY_TOLERANCE_DEG = 0.005  # ~500 m; well below our 0.05° mask resolution

REPO_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_PATH = (
    REPO_ROOT
    / "src" / "librewxr" / "sources" / "regional" / "east_asia"
    / "japan" / "radar" / "jma" / "jpcomp_coverage.geojson"
)


def fetch_latest_basetime() -> tuple[str, str]:
    """Return the most recent (basetime, validtime) pair from targetTimes_N1."""
    resp = httpx.get(TIMES_URL, timeout=30.0)
    resp.raise_for_status()
    entries = resp.json()
    if not entries:
        raise RuntimeError("targetTimes_N1.json is empty")
    latest = entries[0]
    return latest["basetime"], latest["validtime"]


def fetch_coverage_polygon(basetime: str, validtime: str) -> list[tuple[float, float]]:
    """Fetch and decode the hrpns_nd GeoJSON, returning the inner ring.

    Returns a list of ``(lon, lat)`` tuples — GeoJSON ordering.
    """
    url = GEOJSON_URL_TEMPLATE.format(basetime=basetime, validtime=validtime)
    resp = httpx.get(url, timeout=60.0)
    resp.raise_for_status()
    raw = gzip.decompress(resp.content) if resp.content[:2] == b"\x1f\x8b" else resp.content
    gj = json.loads(raw)

    features = gj.get("features", [])
    if not features:
        raise RuntimeError("hrpns_nd geojson has no features")
    geom = features[0]["geometry"]
    if geom["type"] != "Polygon":
        raise RuntimeError(f"expected Polygon, got {geom['type']!r}")
    rings = geom["coordinates"]
    if len(rings) < 2:
        raise RuntimeError(
            "hrpns_nd polygon has no inner ring — coverage hole missing"
        )
    inner = rings[1]
    return [(float(x), float(y)) for x, y in inner]


def simplify_ring(
    inner: list[tuple[float, float]], tolerance_deg: float,
) -> list[tuple[float, float]]:
    """Douglas-Peucker simplification preserving topology."""
    ring = LinearRing(inner)
    simplified = ring.simplify(tolerance_deg, preserve_topology=True)
    return [(float(x), float(y)) for x, y in simplified.coords]


def write_geojson(
    ring: list[tuple[float, float]],
    basetime: str,
    validtime: str,
    output_path: Path,
) -> None:
    """Write the inner ring as a single Polygon feature.

    The output is human-readable JSON (indented) so git diffs are
    meaningful if JMA ever republishes a different shape.
    """
    feature = {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "properties": {
                    "name": "JMA HRPN coverage",
                    "source": "JMA hrpns_nd GeoJSON (inner ring)",
                    "fetched_basetime": basetime,
                    "fetched_validtime": validtime,
                    "simplify_tolerance_deg": SIMPLIFY_TOLERANCE_DEG,
                    "vertex_count": len(ring),
                },
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [[[lon, lat] for lon, lat in ring]],
                },
            }
        ],
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w") as f:
        json.dump(feature, f, indent=2)
        f.write("\n")


def main() -> int:
    print(f"[1/4] Fetching latest basetime from {TIMES_URL}")
    basetime, validtime = fetch_latest_basetime()
    print(f"      basetime={basetime}  validtime={validtime}")

    print(f"[2/4] Fetching hrpns_nd GeoJSON")
    inner = fetch_coverage_polygon(basetime, validtime)
    print(f"      raw inner ring: {len(inner)} vertices")

    print(f"[3/4] Simplifying with Douglas-Peucker at {SIMPLIFY_TOLERANCE_DEG}°")
    simplified = simplify_ring(inner, SIMPLIFY_TOLERANCE_DEG)
    print(f"      simplified ring: {len(simplified)} vertices")

    print(f"[4/4] Writing {OUTPUT_PATH.relative_to(REPO_ROOT)}")
    write_geojson(simplified, basetime, validtime, OUTPUT_PATH)
    size_kb = OUTPUT_PATH.stat().st_size / 1024
    print(f"      wrote {size_kb:.1f} KB")
    return 0


if __name__ == "__main__":
    sys.exit(main())
