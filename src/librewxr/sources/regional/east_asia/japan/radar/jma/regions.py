# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Joshua Kimsey
"""Region definition for the JMA HRPN composite.

Single national region (``JPCOMP``) covering the Japanese archipelago
plus offshore buffer for typhoon tracking.  The native HRPN product is
250 m resolution over the home islands; we resample to 0.0125°/px
(~1.4 km) to match TWCOMP's pixel_size — keeps the rendered output
visually consistent at the geographic boundary between Taiwan and
Japan coverage areas.

Bounds rationale:
- West=122°E: covers Ryukyu Islands (Yonaguni at 122.95°E)
- East=149°E: covers Ogasawara/Chichijima (~142.2°E) plus Pacific buffer
  for landfalling typhoons
- South=22°N: covers Yonaguni (24.45°N) plus offshore buffer
- North=46°N: covers Hokkaido (Soya Cape ~45.5°N) plus a small margin

These bounds intentionally exclude the South Kuril Islands / Northern
Territories (which JMA does not publish radar coverage for) and
exclude Sakhalin (Russian Federation, no public radar — see
project_russia_radar_excluded.md for why this isn't a gap to fill).

Equirectangular treatment for serving — the upstream HRPN product is
published as Web Mercator XYZ tiles, but we resample down to a regular
lat/lon grid in the decoder.  Same simplification as MMD's Albers ->
equirect treatment.
"""
from __future__ import annotations

from librewxr.data.regions import RegionDef


JPCOMP = RegionDef(
    name="JPCOMP",
    west=122.0, east=149.0, south=22.0, north=46.0,
    pixel_size=0.0125,          # matches TWCOMP for visual continuity
    group="JAPAN",
    grid_width=2160,             # (149 - 122) / 0.0125
    grid_height=1920,            # (46 - 22) / 0.0125
)


REGIONS: list[RegionDef] = [JPCOMP]
REGION_GROUP = "JAPAN"
