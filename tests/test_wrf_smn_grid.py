# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Joshua Kimsey
"""Unit tests for WRF-SMN Argentina grid math, decode, and chain integration."""
from __future__ import annotations

import io
from datetime import datetime, timezone

import numpy as np
import pytest

pytestmark = pytest.mark.wrf_smn

from librewxr.data.wrf_smn_grid import (
    BRACKET_INTERVAL_SECONDS,
    CYCLE_INTERVAL_SECONDS,
    WRF_SMN_GRID_HEIGHT,
    WRF_SMN_GRID_WIDTH,
    WRF_SMN_LA1_SOUTH,
    WRF_SMN_LO1_SOUTH,
    WRFSMNGrid,
    bracket_lead_seconds,
    decode_pp_message,
    domain_mask,
    feather_mask,
    file_url,
    floor_cycle,
    grid_indices,
    latest_published_run,
    lcc_forward,
    precip_rate_to_dbz_encoded,
)
from librewxr.data.nwp_source import NWPChain, NWPSource


# ── LCC projection + grid ─────────────────────────────────────────────


class TestLCCProjection:
    @pytest.mark.parametrize(
        "name,lat,lon,inside",
        [
            # Cities expected INSIDE the verified WRF-SMN extent
            ("Buenos Aires", -34.61, -58.38, True),
            ("Cordoba",      -31.42, -64.18, True),
            ("Mendoza",      -32.89, -68.85, True),
            ("Santiago",     -33.46, -70.65, True),   # Chile
            ("Asuncion",     -25.27, -57.58, True),   # Paraguay
            ("Montevideo",   -34.90, -56.16, True),   # Uruguay
            ("La Paz",       -16.50, -68.15, True),   # Bolivia
            ("Sao Paulo",    -23.55, -46.63, True),   # S. Brazil
            ("Ushuaia",      -54.80, -68.30, True),   # Tierra del Fuego
            # Cities expected OUTSIDE the WRF-SMN domain
            ("Lima",         -12.05, -77.04, False),  # past N. edge (-11.65)
            ("Bogota",         4.71, -74.07, False),  # too far north
            ("Rio",          -22.91, -43.17, False),  # past E. edge
            ("Caracas",       10.50, -66.92, False),  # too far north
            ("Madrid",        40.42,  -3.70, False),  # off-continent
            ("Tokyo",         35.68, 139.69, False),
            ("New York",      40.71, -74.01, False),
        ],
    )
    def test_domain_mask_known_points(self, name, lat, lon, inside):
        m = domain_mask(np.array([lat]), np.array([lon]))
        assert bool(m[0]) is inside, name

    def test_grid_origin_at_south_west(self):
        # The native (un-flipped) GRIB scan puts the SW corner at
        # row=0, col=0.  After flip, it's at row HEIGHT-1, col 0.  We
        # verify via the documented (LA1, LO1) anchor.
        row, col = grid_indices(
            np.array([WRF_SMN_LA1_SOUTH]),
            np.array([WRF_SMN_LO1_SOUTH]),
        )
        assert abs(row[0] - (WRF_SMN_GRID_HEIGHT - 1)) < 1e-3
        assert abs(col[0] - 0) < 1e-3

    def test_lcc_centred_on_central_meridian(self):
        # At the standard parallel + central meridian, LCC projects to
        # x=0 (everything is on the y-axis at LoV).
        x, _ = lcc_forward(np.array([-35.0]), np.array([-65.0]))
        assert abs(float(x[0])) < 1e-3

    def test_lcc_round_trip_at_central_lat(self):
        # On the standard parallel, LCC scaling is unity; ±1° lon at
        # -35°N projects to ~91 km of x-distance (= 111 km × cos 35°).
        # Sanity range 70-100 km.
        x_at_lon0, _ = lcc_forward(np.array([-35.0]), np.array([-65.0]))
        x_at_lon1, _ = lcc_forward(np.array([-35.0]), np.array([-64.0]))
        dx = float(x_at_lon1[0] - x_at_lon0[0])
        assert 70_000 < dx < 100_000


# ── Feather ───────────────────────────────────────────────────────────


class TestFeatherMask:
    def test_inside_full_weight(self):
        # Buenos Aires — well inside the domain
        f = feather_mask(np.array([-34.61]), np.array([-58.38]))
        assert f.dtype == np.float32
        assert f[0] == pytest.approx(1.0)

    def test_outside_zero(self):
        # Madrid is off the grid entirely.
        f = feather_mask(np.array([40.42]), np.array([-3.70]))
        assert f[0] == 0.0

    def test_taper_monotonic_walking_off_north_edge(self):
        # Walk lat from inside (-15°S) to outside (5°N) along -65°W.
        # Feather should be non-increasing.
        lats = np.linspace(-15.0, 5.0, 25)
        lons = np.full_like(lats, -65.0)
        f = feather_mask(lats, lons)
        diffs = np.diff(f)
        assert (diffs <= 1e-6).all()


# ── Timing helpers ────────────────────────────────────────────────────


class TestTiming:
    def test_floor_cycle_6h(self):
        ts = int(datetime(2026, 5, 1, 14, 23, tzinfo=timezone.utc).timestamp())
        floored = floor_cycle(ts)
        expected = int(datetime(2026, 5, 1, 12, 0, tzinfo=timezone.utc).timestamp())
        assert floored == expected

    def test_floor_cycle_at_boundary(self):
        ts = int(datetime(2026, 5, 1, 18, 0, tzinfo=timezone.utc).timestamp())
        assert floor_cycle(ts) == ts

    def test_latest_published_run(self):
        now = int(datetime(2026, 5, 1, 14, 0, tzinfo=timezone.utc).timestamp())
        # 4h delay → floor_cycle(10:00) = 06:00
        run = latest_published_run(now, 4 * 3600)
        expected = int(datetime(2026, 5, 1, 6, 0, tzinfo=timezone.utc).timestamp())
        assert run == expected

    @pytest.mark.parametrize(
        "lead_min,l0_min,l1_min,alpha",
        [
            (0,    0,   60, 0.0),
            (30,   0,   60, 0.5),
            (60,   60, 120, 0.0),
            (90,   60, 120, 0.5),
            (120, 120, 180, 0.0),
        ],
    )
    def test_bracket_lead_seconds(self, lead_min, l0_min, l1_min, alpha):
        l0, l1, a = bracket_lead_seconds(lead_min * 60)
        assert l0 == l0_min * 60
        assert l1 == l1_min * 60
        assert a == pytest.approx(alpha)

    def test_cycle_interval_constants(self):
        assert CYCLE_INTERVAL_SECONDS == 6 * 3600
        assert BRACKET_INTERVAL_SECONDS == 3600


# ── URL construction ──────────────────────────────────────────────────


class TestFileUrl:
    def test_format_matches_smn_pattern(self):
        run = datetime(2026, 5, 8, 0, tzinfo=timezone.utc)
        url = file_url(run, 6)
        assert url.endswith(
            "DATA/WRF/DET/2026/05/08/00/WRFDETAR_01H_20260508_00_006.nc"
        )

    def test_step_zero_padded_to_three_digits(self):
        run = datetime(2026, 5, 8, 6, tzinfo=timezone.utc)
        url = file_url(run, 0)
        assert url.endswith("_06_000.nc")

    def test_step_72_padded(self):
        run = datetime(2026, 5, 8, 6, tzinfo=timezone.utc)
        url = file_url(run, 72)
        assert url.endswith("_06_072.nc")

    def test_uses_settings_bucket_and_region(self):
        run = datetime(2026, 5, 8, 0, tzinfo=timezone.utc)
        url = file_url(run, 1)
        assert "smn-ar-wrf" in url
        assert "us-west-2" in url


# ── Z-R conversion ────────────────────────────────────────────────────


class TestZR:
    def test_zero_rate_zero_encoded(self):
        encoded = precip_rate_to_dbz_encoded(np.array([0.0, 0.0]))
        assert (encoded == 0).all()

    def test_higher_rate_higher_dbz(self):
        encoded = precip_rate_to_dbz_encoded(np.array([0.5, 5.0, 50.0]))
        assert encoded[0] < encoded[1] < encoded[2]
        assert abs(int(encoded[2]) - 164) <= 2

    def test_handles_nan_and_negative(self):
        encoded = precip_rate_to_dbz_encoded(np.array([np.nan, -1.0, 1.0]))
        assert encoded[0] == 0
        assert encoded[1] == 0
        assert encoded[2] > 0

    def test_dbz_offset_shifts_uniformly(self):
        rates = np.array([1.0, 5.0, 25.0])
        base = precip_rate_to_dbz_encoded(rates, dbz_offset=0.0)
        shifted = precip_rate_to_dbz_encoded(rates, dbz_offset=6.0)
        for b, s in zip(base, shifted):
            if b > 0:
                assert int(s) - int(b) == 12

    def test_zero_rate_offset_still_zero(self):
        encoded = precip_rate_to_dbz_encoded(
            np.array([0.0, 0.0]), dbz_offset=10.0,
        )
        assert (encoded == 0).all()


# ── NetCDF4 / HDF5 decode ─────────────────────────────────────────────


def _build_synthetic_nc(
    pp: np.ndarray, lat: np.ndarray, lon: np.ndarray,
) -> bytes:
    """Build a minimal in-memory HDF5/NetCDF4 buffer with PP, lat, lon."""
    import h5py

    buf = io.BytesIO()
    with h5py.File(buf, "w") as f:
        f.create_dataset("PP", data=pp[None, ...].astype(np.float32))
        f.create_dataset("lat", data=lat.astype(np.float32))
        f.create_dataset("lon", data=lon.astype(np.float32))
    return buf.getvalue()


class TestDecodeOrientation:
    def test_decode_flips_south_up(self, tmp_path):
        # Synthetic file: PP has marker at row 0 (south), the file's
        # native scan order.  After decode flip, the marker should
        # land at row HEIGHT-1.
        pp = np.zeros((WRF_SMN_GRID_HEIGHT, WRF_SMN_GRID_WIDTH), dtype=np.float32)
        pp[0, 100] = 5.0     # south marker
        pp[-1, 200] = 8.0    # north marker

        # 2-D lat/lon arrays where lat INCREASES with row index (south-up).
        lat = np.broadcast_to(
            np.linspace(-54.0, -12.0, WRF_SMN_GRID_HEIGHT)[:, None],
            (WRF_SMN_GRID_HEIGHT, WRF_SMN_GRID_WIDTH),
        ).astype(np.float32)
        lon = np.broadcast_to(
            np.linspace(-94.0, -36.0, WRF_SMN_GRID_WIDTH)[None, :],
            (WRF_SMN_GRID_HEIGHT, WRF_SMN_GRID_WIDTH),
        ).astype(np.float32)

        nc_bytes = _build_synthetic_nc(pp, lat, lon)
        arr = decode_pp_message(nc_bytes)
        assert arr is not None
        assert arr.shape == (WRF_SMN_GRID_HEIGHT, WRF_SMN_GRID_WIDTH)
        # After flip, south marker (row 0 in file) → our row -1.
        assert arr[-1, 100] == 5.0
        assert arr[0, 200] == 8.0

    def test_decode_no_flip_when_north_up(self):
        # Defensive: if SMN ever ships north-up files, we should NOT flip.
        pp = np.zeros((WRF_SMN_GRID_HEIGHT, WRF_SMN_GRID_WIDTH), dtype=np.float32)
        pp[0, 100] = 5.0
        pp[-1, 200] = 8.0

        # lat DECREASES with row index → north-up file.
        lat = np.broadcast_to(
            np.linspace(-12.0, -54.0, WRF_SMN_GRID_HEIGHT)[:, None],
            (WRF_SMN_GRID_HEIGHT, WRF_SMN_GRID_WIDTH),
        ).astype(np.float32)
        lon = np.broadcast_to(
            np.linspace(-94.0, -36.0, WRF_SMN_GRID_WIDTH)[None, :],
            (WRF_SMN_GRID_HEIGHT, WRF_SMN_GRID_WIDTH),
        ).astype(np.float32)

        nc_bytes = _build_synthetic_nc(pp, lat, lon)
        arr = decode_pp_message(nc_bytes)
        assert arr is not None
        # No flip: row 0 in file stays at our row 0.
        assert arr[0, 100] == 5.0
        assert arr[-1, 200] == 8.0

    def test_decode_handles_fillvalue(self):
        # SMN files use 1e20 as _FillValue; ensure it doesn't leak through.
        pp = np.full(
            (WRF_SMN_GRID_HEIGHT, WRF_SMN_GRID_WIDTH), 1e20, dtype=np.float32,
        )
        pp[100, 100] = 3.5  # one real value
        # north-up so no flip
        lat = np.broadcast_to(
            np.linspace(-12.0, -54.0, WRF_SMN_GRID_HEIGHT)[:, None],
            (WRF_SMN_GRID_HEIGHT, WRF_SMN_GRID_WIDTH),
        ).astype(np.float32)
        lon = np.broadcast_to(
            np.linspace(-94.0, -36.0, WRF_SMN_GRID_WIDTH)[None, :],
            (WRF_SMN_GRID_HEIGHT, WRF_SMN_GRID_WIDTH),
        ).astype(np.float32)

        arr = decode_pp_message(_build_synthetic_nc(pp, lat, lon))
        assert arr is not None
        assert arr[100, 100] == pytest.approx(3.5)
        # Everywhere else is fill → 0
        assert arr[0, 0] == 0.0
        assert arr[500, 500] == 0.0

    def test_decode_returns_none_when_no_pp(self):
        # File with no PP variable → graceful None.
        import h5py
        buf = io.BytesIO()
        with h5py.File(buf, "w") as f:
            f.create_dataset("T2", data=np.zeros(
                (1, WRF_SMN_GRID_HEIGHT, WRF_SMN_GRID_WIDTH), dtype=np.float32,
            ))
        arr = decode_pp_message(buf.getvalue())
        assert arr is None


# ── Cumulative-to-rate diff ───────────────────────────────────────────


class TestAccumulationDiff:
    def test_step_zero_baseline_is_zero(self):
        grid = WRFSMNGrid()
        run_dt = datetime(2026, 5, 8, 0, tzinfo=timezone.utc)
        run_ts = int(run_dt.timestamp())
        grid._accum[(run_ts, 0)] = np.zeros(
            (WRF_SMN_GRID_HEIGHT, WRF_SMN_GRID_WIDTH), dtype=np.float32,
        )
        baseline = grid._accum[(run_ts, 0)]
        assert baseline.shape == (WRF_SMN_GRID_HEIGHT, WRF_SMN_GRID_WIDTH)
        assert (baseline == 0).all()

    def test_diff_yields_windowed_rate(self):
        accum_5 = np.full(
            (WRF_SMN_GRID_HEIGHT, WRF_SMN_GRID_WIDTH), 5.0, dtype=np.float32,
        )
        accum_6 = np.full(
            (WRF_SMN_GRID_HEIGHT, WRF_SMN_GRID_WIDTH), 11.0, dtype=np.float32,
        )
        rate = accum_6 - accum_5
        encoded = precip_rate_to_dbz_encoded(rate)
        assert (encoded > 0).all()


# ── Run picking ───────────────────────────────────────────────────────


class TestPickRun:
    def test_no_frames_returns_none(self):
        grid = WRFSMNGrid()
        ts = int(datetime(2026, 5, 8, 12, tzinfo=timezone.utc).timestamp())
        assert grid._pick_run(ts) is None

    def test_returns_run_only_when_bracket_loaded(self):
        grid = WRFSMNGrid()
        run_ts = int(
            datetime(2026, 5, 8, 6, tzinfo=timezone.utc).timestamp()
        )
        fake = np.zeros(
            (WRF_SMN_GRID_HEIGHT, WRF_SMN_GRID_WIDTH), dtype=np.uint8,
        )
        grid._frames[(run_ts, 3 * 3600)] = fake
        query_ts = run_ts + 3 * 3600 + 1800
        assert grid._pick_run(query_ts) is None
        grid._frames[(run_ts, 4 * 3600)] = fake
        assert grid._pick_run(query_ts) == run_ts

    def test_falls_back_to_older_run_when_freshest_incomplete(self):
        grid = WRFSMNGrid()
        old_ts = int(datetime(2026, 5, 8, 0, tzinfo=timezone.utc).timestamp())
        new_ts = int(datetime(2026, 5, 8, 6, tzinfo=timezone.utc).timestamp())
        fake = np.zeros(
            (WRF_SMN_GRID_HEIGHT, WRF_SMN_GRID_WIDTH), dtype=np.uint8,
        )
        grid._frames[(new_ts, 7 * 3600)] = fake
        grid._frames[(old_ts, 13 * 3600)] = fake
        grid._frames[(old_ts, 14 * 3600)] = fake
        query_ts = int(
            datetime(2026, 5, 8, 13, 30, tzinfo=timezone.utc).timestamp()
        )
        assert grid._pick_run(query_ts) == old_ts


# ── Protocol conformance ──────────────────────────────────────────────


class TestNWPSourceProtocol:
    def test_satisfies_protocol(self):
        grid = WRFSMNGrid()
        assert isinstance(grid, NWPSource)

    def test_chain_with_only_smn(self):
        grid = WRFSMNGrid()
        chain = NWPChain([grid])
        out = chain.sample(np.array([-34.61]), np.array([-58.38]))
        assert out.shape == (1,)
        assert out[0] == 0  # no data loaded

    def test_supports_snow_false(self):
        grid = WRFSMNGrid()
        assert grid.supports_snow is False
        out = grid.get_snow_mask(np.array([-34.61]), np.array([-58.38]))
        assert (out == False).all()
