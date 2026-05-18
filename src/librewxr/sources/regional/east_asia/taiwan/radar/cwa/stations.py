# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Joshua Kimsey
"""Taiwan QPESUMS contributing radars.

7 contributing radars operated by the Central Weather Administration.
Approximate coordinates from publicly-documented siting and CWA's radar
inventory.  Two additional Taiwan radars exist (RCMK military, RCWF
civil aviation) but are not part of this composite.

The 240 km default range would cover all of Taiwan + a substantial
offshore buffer, but CWA QPESUMS publishes coverage out to its full
bbox edges (~565 km from the nearest station) for typhoon tracking, as
verified from the ``-999`` sentinel pattern in a real frame.  We
previously used 550 km to match that "claimed" reach, but Taiwan's
western and southern stations are close enough to land that the
resulting circles bleed visibly onto Fujian / Hong Kong (Qigu's 550 km
reach hit 114.7°E) and over the northern tip of Luzon (Kenting's reach
hit 17°N, past Cape Bojeador).  At 450 km the mask still covers ~94%
of CWA's claimed in-range area and ~450 km of Pacific buffer east of
Hualien (well into the typhoon corridor); the trimmed halo over the
SCS / W. Pacific gets filled by IFS instead, which resolves typhoons
at 9 km perfectly well.  Per-station ranges are the proper long-term
fix (small western stations could go to ~150 km, eastern stations
could keep ~450 km) but ``range_overrides`` is per-region today.
Tighten further to 300 km if bleed onto Fujian / Luzon is still
visible.

Exact lat/lons to the metre don't matter here because the radar
circles overlap heavily over the island; the resulting union polygon
is insensitive to small per-station shifts.
"""
from __future__ import annotations


STATIONS: list[tuple[float, float]] = [
    (25.071, 121.773),   # 五分山 / Wufenshan  (NE Taiwan, original 4)
    (23.991, 121.622),   # 花蓮   / Hualien    (E Taiwan,  original 4)
    (23.146, 120.094),   # 七股   / Qigu/Cigu  (SW Taiwan, original 4)
    (21.902, 120.853),   # 墾丁   / Kenting    (S Taiwan,  original 4)
    (24.998, 121.420),   # 樹林   / Shulin     (N gap-fill)
    (24.140, 120.620),   # 南屯   / Nantun     (central gap-fill)
    (22.510, 120.397),   # 林園   / Linyuan    (S gap-fill, Kaohsiung)
]

STATION_MAP: dict[str, list[tuple[float, float]]] = {
    "TWCOMP": STATIONS,
}

RANGE_OVERRIDES: dict[str, float] = {
    "TWCOMP": 450.0,
}
