# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Joshua Kimsey
"""JMA C-band Doppler radar station locations.

20 operational sites covering Japan.  Kept for documentation /
coverage-map purposes only — JPCOMP is intentionally NOT registered
in ``STATION_MAP`` (see below).

Station coordinates from the JMA Observations page
(``www.jma.go.jp/jma/en/Activities/observations.html``).  XRAIN X-band
sites are not listed here — they're fused into HRPN upstream and don't
need separate mask handling.

Why JPCOMP has no station mask:
HRPN is JMA's gauge-corrected QPE composite — it fuses the 20 C-band
Doppler radars with XRAIN X-band radars and the AMeDAS rain-gauge
network into one product whose published extent extends well past
individual Doppler reach.  A 240 km station-circle union dramatically
under-represents the real product footprint, so anywhere offshore
where HRPN genuinely has data falls outside the union mask and the
renderer's NWP-fill path paints model precipitation on top of the
radar pixels.  The clean fix is to use the authoritative coverage
polygon JMA themselves publish — see ``JPCOMP_COVERAGE_POLYGON``.
"""
from __future__ import annotations

import json
from pathlib import Path


# (latitude, longitude) — 20 JMA C-band Doppler radars.  Documentation
# only; consumed by the coverage-map script (``scripts/generate_coverage_map.py``)
# but NOT by the runtime coverage mask builder.
STATIONS: list[tuple[float, float]] = [
    (43.063, 141.349),   # Sapporo (Ishikari)
    (43.910, 144.069),   # Kitami (Mombetsu area)
    (42.998, 144.494),   # Kushiro
    (41.775, 140.739),   # Hakodate
    (40.190, 140.797),   # Akita
    (38.262, 140.902),   # Sendai
    (37.392, 138.616),   # Niigata
    (36.069, 139.769),   # Tokyo (Kashiwa)
    (35.243, 138.973),   # Mt. Fuji (Shizuoka)
    (35.180, 136.906),   # Nagoya (Komaki)
    (34.694, 135.502),   # Osaka (Tanigawa)
    (35.452, 133.066),   # Matsue
    (34.013, 131.067),   # Hiroshima (Sera)
    (33.595, 130.451),   # Fukuoka
    (32.745, 129.866),   # Nagasaki (Seburi)
    (32.749, 132.949),   # Muroto Cape (Kochi)
    (31.790, 130.393),   # Kagoshima (Tanegashima)
    (28.380, 129.547),   # Naze (Amami Oshima)
    (26.205, 127.687),   # Naha (Okinawa main)
    (24.453, 122.951),   # Yonaguni (westernmost Ryukyu)
]


# Empty intentionally — coverage comes from the polygon below, not from
# station circles.  See module docstring.
STATION_MAP: dict[str, list[tuple[float, float]]] = {}

# Empty intentionally — no per-station range overrides apply when no
# station-circle mask is built in the first place.
RANGE_OVERRIDES: dict[str, dict[tuple[float, float], float]] = {}


# JPCOMP coverage polygon — vertices in (latitude, longitude) order.
# Loaded from ``jpcomp_coverage.geojson`` at import time.  That file is
# JMA's own ``hrpns_nd`` no-data GeoJSON (inner ring, Douglas-Peucker-
# simplified at 0.005° ≈ 500 m), discovered via the bosai/nowc viewer
# config.  Refresh by running ``scripts/refresh_jma_coverage.py``;
# JMA only republishes a new shape when the HRPN network itself changes.
_COVERAGE_FILE = Path(__file__).with_name("jpcomp_coverage.geojson")


def _load_coverage_polygon() -> list[tuple[float, float]]:
    """Load the JPCOMP coverage polygon as (lat, lon) tuples.

    GeoJSON stores coordinates as ``[lon, lat]``; the project convention
    for polygon mask builders is ``(lat, lon)``.  The order swap happens
    here so consumers don't have to think about it.
    """
    gj = json.loads(_COVERAGE_FILE.read_text())
    ring = gj["features"][0]["geometry"]["coordinates"][0]
    return [(float(lat), float(lon)) for lon, lat in ring]


JPCOMP_COVERAGE_POLYGON: list[tuple[float, float]] = _load_coverage_polygon()


COVERAGE_POLYGONS: dict[str, list[tuple[float, float]]] = {
    "JPCOMP": JPCOMP_COVERAGE_POLYGON,
}
