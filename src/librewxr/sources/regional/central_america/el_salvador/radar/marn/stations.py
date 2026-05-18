# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Joshua Kimsey
"""SNET (El Salvador) radar station inventory.

Single S-band radar at San Andrés volcano.  Coordinates from the
viewer's ``center = [13.687, -88.883]`` JS variable
(snet.gob.sv/googlemaps/radares/radaresSV8.php).

Range override: the SNET ``esar82`` product is explicitly the 120 km
range overlay (single S-band radar at San Andrés).  The default 240 km
would overstate coverage by 2× and bleed past the product's footprint
into Honduras / Nicaragua where no actual returns exist.
"""
from __future__ import annotations


STATIONS: list[tuple[float, float]] = [
    (13.687, -88.883),   # San Andrés
]

STATION_MAP: dict[str, list[tuple[float, float]]] = {
    "SVCOMP": STATIONS,
}

RANGE_OVERRIDES: dict[str, float] = {
    "SVCOMP": 120.0,
}
