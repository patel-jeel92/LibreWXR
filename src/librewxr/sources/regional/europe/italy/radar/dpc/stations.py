# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Joshua Kimsey
"""DPC Italian national radar network — station list + coverage polygon.

Sourced from the official DPC document
``LA RETE RADAR METEOROLOGICA NAZIONALE``
(Allegato 1, ANAC publication 2024) — the 11 DPC-direct radars in
Tabella 1 (Gematronik Meteor 600 C and 50 DX) carry exact coordinates
from that document.  The 13 partner radars in Tabella 2 are not given
coordinates there; the values below come from publicly-known site
locations (airport ICAO codes, mountain-top toponyms) and should be
accurate to within a few hundred metres — fine for documentation +
coverage-polygon construction but flagged here for any future
application that needs survey precision.

The DPC platform does not (currently) expose a SITES download endpoint
despite `findLastProductByType?type=SITES` reporting one — the
documentation lists SITES as a product type but `/downloadProduct`
rejects it with `productType non supportato`.  Until that changes, this
list is hand-maintained.

The composite covers all of Italy from a mix of DPC + partner radars;
unlisted partner radars only mean the derived coverage polygon
slightly under-extends in their direction (pixels that *are* in radar
range get marked as model-fill), not gaps in the actual composite data.

Why ITCOMP uses a polygon mask rather than 150 km station circles:
A uniform 150 km circle union over-extends into W. Austria, S. Slovenia,
and S. France where DPC has no signal but the circle says it does —
the renderer then "claims" those pixels for ITCOMP and locks OPERA's
real data out (see GitHub issue #5).  The polygon is built offline by
``scripts/refresh_dpc_coverage.py`` using the rule

    coverage = (D ∩ italy+30km) ∪ (D − OPERA300km)

where ``D`` is the 150 km DPC station union, ``italy+30km`` is the
Natural Earth admin-0 shape buffered by 30 km, and ``OPERA300km`` is
the union of 300 km circles around all OPERA stations.  The polygon
keeps DPC priority over Italy proper + a coastal buffer, lets OPERA
take over wherever it can reach (Adriatic / Ligurian / Alpine border),
and preserves DPC's full 150 km reach into open ocean south of Sicily
and east of the Ionian where no OPERA neighbour fills behind.
"""
from __future__ import annotations

import json
from pathlib import Path


# Tabella 1 of Allegato 1 — DPC-direct radars (lat/lon authoritative).
_DPC_DIRECT: list[tuple[float, float]] = [
    # 7 C-band Gematronik Meteor 600 C (dual-polarization Doppler)
    (41.939, 14.624),   # Monte II Monte (Tufillo, CH)
    (43.956, 10.607),   # Monte Crocione (Villa Basilica, LU)
    (42.856, 12.791),   # Monte Serano (Campello sul Clitunno, PG)
    (39.373, 16.624),   # Monte Pettinascura (Longobucco, CS)
    (46.556, 12.974),   # Monte Zoufplan (Paluzza, UD)
    (37.123, 14.824),   # Monte Lauro (Buccheri, SI)
    (39.873,  9.491),   # Monte Armidda (Gairo, NU)
    # 4 X-band Gematronik Meteor 50 DX (airport sites)
    (40.880, 14.290),   # Aeroporto Napoli Capodichino
    (38.050, 15.650),   # Aeroporto Reggio Calabria
    (41.139, 16.760),   # Aeroporto Bari Palese
    (37.460, 15.050),   # Aeroporto Catania Fontanarossa
]

# Tabella 2 — Partner sub-network (regional admins + ENAV + Aeronautica
# Militare).  Coordinates derived from public site information; nominal
# precision a few hundred metres at worst.
_DPC_PARTNERS: list[tuple[float, float]] = [
    (45.0367,  7.7325),   # Bric della Croce (Torino — ARPA Piemonte)
    (44.2467,  8.2008),   # Monte Settepani (Piemonte / Liguria border)
    (44.7872, 10.4972),   # Gattatico (Reggio Emilia — ARPASIM)
    (44.6539, 11.6231),   # San Pietro Capofiume (Molinella, BO — ARPASIM)
    (45.3469, 11.6708),   # Monte Grande (Teolo, PD — ARPA Veneto)
    (45.7600, 12.8500),   # Concordia Sagittaria (VE — ARPA Veneto)
    (42.0408, 13.1764),   # Monte Midia (AQ — Regione Abruzzo)
    (46.4683, 11.1839),   # Monte Macaion (TN — PAA di Trento)
    (45.4451,  9.2767),   # Linate (Milano — ENAV)
    (41.8003, 12.2389),   # Fiumicino (Roma — ENAV)
    (45.7256, 13.4581),   # Fossalon di Grado (UD — ARPA FVG)
    (40.5750,  8.1717),   # Capocaccia (SS — Aeronautica Militare)
    (40.4080,  9.0233),   # Monte Rasu (SS — ARPAS Sardegna)
]

STATIONS: list[tuple[float, float]] = _DPC_DIRECT + _DPC_PARTNERS


# Empty intentionally — ITCOMP coverage comes from the polygon below,
# not from station circles.  See module docstring.
STATION_MAP: dict[str, list[tuple[float, float]]] = {}

# Empty intentionally — no per-station range applies when no station-
# circle mask is built in the first place.
RANGE_OVERRIDES: dict[str, float] = {}


# ITCOMP coverage polygon — vertices in (latitude, longitude) order.
# Loaded from ``dpc_coverage.geojson`` at import time.  Refresh by
# running ``scripts/refresh_dpc_coverage.py``; that script encodes the
# build rule (D ∩ italy+30km) ∪ (D − OPERA300km), so any change to the
# DPC or OPERA station lists, or a coastline update from Natural Earth,
# is picked up by re-running it.
_COVERAGE_FILE = Path(__file__).with_name("dpc_coverage.geojson")


def _load_coverage_polygon() -> list[list[tuple[float, float]]]:
    """Load the ITCOMP coverage shape as a list of (lat, lon) rings.

    GeoJSON stores coordinates as ``[lon, lat]``; the project convention
    for polygon mask builders is ``(lat, lon)``.  The order swap happens
    here so consumers don't have to think about it.

    Returns a multi-polygon (list of rings) even when the underlying
    GeoJSON is a single Polygon, so downstream consumers can handle one
    shape.
    """
    gj = json.loads(_COVERAGE_FILE.read_text())
    geom = gj["features"][0]["geometry"]
    if geom["type"] == "Polygon":
        polys = [geom["coordinates"]]
    elif geom["type"] == "MultiPolygon":
        polys = geom["coordinates"]
    else:
        raise RuntimeError(
            f"unexpected geometry type in {_COVERAGE_FILE.name}: {geom['type']}"
        )
    return [
        [(float(lat), float(lon)) for lon, lat in poly[0]]
        for poly in polys
    ]


ITCOMP_COVERAGE_POLYGON: list[list[tuple[float, float]]] = (
    _load_coverage_polygon()
)


COVERAGE_POLYGONS: dict[str, list[list[tuple[float, float]]]] = {
    "ITCOMP": ITCOMP_COVERAGE_POLYGON,
}
