# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Joshua Kimsey
"""JMA C-band Doppler radar station locations.

20 operational sites covering Japan, used by ``data/coverage.py`` to
build the coverage mask.  Station coordinates from the JMA Observations
page (``www.jma.go.jp/jma/en/Activities/observations.html``).  XRAIN
X-band sites are not listed here — they're fused into HRPN upstream
and don't need separate mask handling.

Default Doppler range (240 km) applies to all stations; sites with
non-standard range can override via ``RANGE_OVERRIDES`` when verified.
"""
from __future__ import annotations


# (latitude, longitude) — 20 JMA C-band Doppler radars
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


STATION_MAP: dict[str, list[tuple[float, float]]] = {
    "JPCOMP": [
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
    ],
}

# Empty by default — populate per-station if any radar has a range
# materially different from the 240 km default Doppler reach.
RANGE_OVERRIDES: dict[str, dict[tuple[float, float], float]] = {}
