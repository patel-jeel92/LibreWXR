# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Joshua Kimsey
"""NWPSource Protocol and NWPChain dispatcher for multi-model NWP fallback.

Phase 1 of the multi-model NWP integration: defines the contract that any
numerical-weather-prediction source (ECMWF IFS, NOAA HRRR, DWD ICON-D2, ...)
must satisfy, plus a chain dispatcher that walks sources in priority order
and fills pixels from the first source with both coverage and data.

Each source handles its own quirks internally — Z-R conversion, projection
sampling, fetch cadence — so the renderer talks to a single uniform interface.
"""

from typing import Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class NWPSource(Protocol):
    """A numerical weather prediction data source."""

    name: str

    def sample(
        self,
        lat: np.ndarray,
        lon: np.ndarray,
        timestamp: int | None = None,
        bilinear: bool = False,
    ) -> np.ndarray:
        """Return uint8 dBZ-encoded precipitation at each (lat, lon) point.

        Encoding matches the radar pipeline: pixel = (dBZ + 32) * 2.
        Output shape == lat.shape.
        """
        ...

    def get_snow_mask(
        self,
        lat: np.ndarray,
        lon: np.ndarray,
        timestamp: int | None = None,
    ) -> np.ndarray:
        """Return bool mask: True where precipitation is snow. Shape == lat.shape."""
        ...

    def domain_mask(self, lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
        """Return bool mask: True where this source has coverage. Shape == lat.shape."""
        ...

    def has_data_at(self, timestamp: int) -> bool:
        """Whether this source can answer for the given valid time right now."""
        ...

    def has_data(self) -> bool:
        """Whether this source has any data loaded at all."""
        ...


class NWPChain:
    """Dispatches sample/snow_mask queries across NWP sources in priority order.

    For each pixel, the first source with both coverage AND data fills it.
    Pixels not covered by any source remain at the default fill value
    (0 = no precipitation in dBZ encoding, False in the snow mask).

    A chain with a single source behaves identically to calling that source
    directly — Phase 1 wraps ECMWFGrid in a one-element chain to introduce
    the abstraction without changing behavior.
    """

    def __init__(self, sources: list[NWPSource]):
        self._sources = list(sources)

    @property
    def sources(self) -> list[NWPSource]:
        return list(self._sources)

    def has_data(self) -> bool:
        """True if any registered source has data loaded."""
        return any(src.has_data() for src in self._sources)

    def sample(
        self,
        lat: np.ndarray,
        lon: np.ndarray,
        timestamp: int | None = None,
        bilinear: bool = False,
    ) -> np.ndarray:
        out = np.zeros(lat.shape, dtype=np.uint8)
        unfilled = np.ones(lat.shape, dtype=bool)
        for src in self._sources:
            if timestamp is not None and not src.has_data_at(timestamp):
                continue
            if timestamp is None and not src.has_data():
                continue
            domain = src.domain_mask(lat, lon)
            mask = unfilled & domain
            if not mask.any():
                continue
            sub_lat = lat[mask]
            sub_lon = lon[mask]
            out[mask] = src.sample(sub_lat, sub_lon, timestamp, bilinear)
            unfilled &= ~domain
            if not unfilled.any():
                break
        return out

    def get_snow_mask(
        self,
        lat: np.ndarray,
        lon: np.ndarray,
        timestamp: int | None = None,
    ) -> np.ndarray:
        out = np.zeros(lat.shape, dtype=bool)
        unfilled = np.ones(lat.shape, dtype=bool)
        for src in self._sources:
            if timestamp is not None and not src.has_data_at(timestamp):
                continue
            if timestamp is None and not src.has_data():
                continue
            domain = src.domain_mask(lat, lon)
            mask = unfilled & domain
            if not mask.any():
                continue
            sub_lat = lat[mask]
            sub_lon = lon[mask]
            out[mask] = src.get_snow_mask(sub_lat, sub_lon, timestamp)
            unfilled &= ~domain
            if not unfilled.any():
                break
        return out
