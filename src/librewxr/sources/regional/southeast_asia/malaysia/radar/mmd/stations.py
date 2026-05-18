# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Joshua Kimsey
"""MET Malaysia radar station inventory.

12-radar national network feeding the combined Peninsular + East
composite GIF.  Coordinates are approximate (taken from each radar's
host airport / city — the operational siting is usually co-located or
within a few km).

Cross-checked against the station inventory on
rainviewer.com/radars/malaysia.html (12 stations: MY2809–MY2819,
MY2865).

Range overrides: MET Malaysia's CAPPI composite extends well past the
240 km Doppler default — empirically measured 372 km max extent over
Peninsular Malaysia and 332 km over East Malaysia from the nearest
station, with p99 distances of 313 km and 302 km respectively.  The
default would clip ~10% of legitimate radar data (visible as IFS bleed
at frame T+0 and a "chunk of radar rain blinking out" at the first
nowcast frame as the blend switches to model outside the mask).  Ranges
chosen to snugly cover the measured max with small margin: a looser fit
would help if maxes fluctuate day-to-day, but it would also hide IFS
precipitation over the South China Sea / Celebes Sea (which, unlike
CWA's open Pacific halo, gets real rain).  If clipping reappears on a
heavy convective day, bump by 25 km.
"""
from __future__ import annotations


PENINSULAR_STATIONS: list[tuple[float, float]] = [
    (6.20, 100.40),    # MY2810 Alor Setar
    (5.47, 100.39),    # MY2819 Butterworth
    (2.04, 103.32),    # MY2818 Kluang
    (6.17, 102.29),    # MY2815 Kota Bharu
    (3.78, 103.21),    # MY2817 Kuantan
    (3.13, 101.55),    # MY2816 Subang
    (2.74, 101.71),    # MY2865 TDR KLIA
]

EAST_STATIONS: list[tuple[float, float]] = [
    (3.16, 113.05),    # MY2812 Bintulu
    (5.94, 116.05),    # MY2809 Kota Kinabalu
    (1.48, 110.34),    # MY2814 Kuching
    (4.32, 113.99),    # MY2813 Miri
    (5.90, 118.06),    # MY2811 Sandakan
]

STATION_MAP: dict[str, list[tuple[float, float]]] = {
    "MYPENINSULAR": PENINSULAR_STATIONS,
    "MYEAST": EAST_STATIONS,
}

RANGE_OVERRIDES: dict[str, float] = {
    "MYPENINSULAR": 375.0,
    "MYEAST": 350.0,
}
