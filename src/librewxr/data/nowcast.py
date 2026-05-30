# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Joshua Kimsey
"""Precipitation nowcasting via radar extrapolation and IFS blending.

Generates short-range forecast frames (default 60 minutes) by:

1. Computing optical flow between the two most recent radar frames
   (per region, with adaptive downscaling for speed).
2. Extrapolating the latest radar forward along the motion vectors.
3. Storing extrapolated frames in a lightweight ``NowcastStore`` with
   per-frame blend weights that tell the renderer how much to trust
   the extrapolation vs the ECMWF IFS forecast.

The renderer handles the actual blending — this module only produces
the extrapolated radar data and the temporal blend weight for each step.
"""
from __future__ import annotations

import asyncio
import logging
import os
import shutil
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from librewxr.config import settings
from librewxr.data.zr import mmh_to_uint8, uint8_to_mmh

logger = logging.getLogger(__name__)

# S-PROG cascade levels.  Default 6 covers ~2 km – 512 km scales on
# our typical regional grids, matching the pysteps default.
_SPROG_CASCADE_LEVELS = 6

# pyfftw is recommended but optional — pysteps falls back to numpy FFT
# (~2-3× slower) when the import fails.  We pick once at module load
# and pass the chosen backend to every sprog.forecast call.
try:
    import pyfftw  # noqa: F401

    _SPROG_FFT_METHOD = "pyfftw"
except ImportError:
    _SPROG_FFT_METHOD = "numpy"

# Target longest dimension for optical flow computation.
# Larger grids are downscaled to this for speed, then flow vectors are
# upscaled back to full resolution.
_TARGET_FLOW_DIM = 1000

# Farneback optical flow parameters (same tuning as ecmwf_interpolation).
_FARNEBACK = dict(
    pyr_scale=0.5,
    levels=3,
    winsize=15,
    iterations=3,
    poly_n=5,
    poly_sigma=1.2,
    flags=0,
)


# ---------------------------------------------------------------------------
# NowcastStore
# ---------------------------------------------------------------------------

@dataclass
class NowcastFrame:
    """A single nowcast frame with per-region extrapolated radar data."""
    timestamp: int
    regions: dict[str, np.ndarray] = field(default_factory=dict)
    blend_weight: float = 1.0  # 1.0 = trust radar, 0.0 = trust IFS


class NowcastStore:
    """Lightweight store for nowcast frames.

    Nowcast frames are regenerated every fetch cycle, so no persistence
    or max-frames eviction is needed — just an atomic swap of the frame
    dict each cycle.  Region arrays and flow fields are backed by
    memory-mapped temp files so the OS page cache manages physical RAM.
    """

    def __init__(self, cache_dir: Path | None = None):
        self._frames: dict[int, NowcastFrame] = {}
        self._flows: dict[str, np.ndarray] = {}
        self._lock = asyncio.Lock()
        if cache_dir is not None:
            self._memmap_dir = Path(cache_dir) / "nowcast"
            self._persistent = True
        else:
            self._memmap_dir = Path(tempfile.mkdtemp(prefix="librewxr_nowcast_"))
            self._persistent = False
        self._memmap_dir.mkdir(parents=True, exist_ok=True)
        for path in self._memmap_dir.glob("*.tmp"):
            path.unlink(missing_ok=True)
        logger.info(
            "Nowcast memmap directory: %s (persistent=%s)",
            self._memmap_dir, self._persistent,
        )

    def _to_memmap(self, name: str, data: np.ndarray) -> np.ndarray:
        """Write array to disk atomically and return a read-only memory-mapped view."""
        final = self._memmap_dir / f"{name}.dat"
        tmp = final.with_suffix(".dat.tmp")
        mm = np.memmap(tmp, dtype=data.dtype, mode="w+", shape=data.shape)
        mm[:] = data
        mm.flush()
        del mm
        os.replace(tmp, final)
        return np.memmap(final, dtype=data.dtype, mode="r", shape=data.shape)

    async def replace_all(
        self, frames: list[NowcastFrame],
    ) -> list[int]:
        """Atomically replace all nowcast frames.

        Returns the timestamps of the old frames that were removed
        (for tile cache invalidation).
        """
        async with self._lock:
            old_timestamps = list(self._frames.keys())

            # Clean up old frame memmap files
            for path in self._memmap_dir.glob("frame_*.dat"):
                try:
                    path.unlink()
                except OSError:
                    pass

            # Convert region arrays to memmaps
            for frame in frames:
                for name, data in list(frame.regions.items()):
                    frame.regions[name] = self._to_memmap(
                        f"frame_{frame.timestamp}_{name}", data
                    )

            self._frames = {f.timestamp: f for f in frames}
            return old_timestamps

    async def get_frame(
        self, timestamp: int,
    ) -> tuple[NowcastFrame | None, float]:
        """Return ``(frame, blend_weight)`` or ``(None, 0.0)``."""
        async with self._lock:
            frame = self._frames.get(timestamp)
            if frame is None:
                return None, 0.0
            return frame, frame.blend_weight

    async def get_timestamps(self) -> list[int]:
        async with self._lock:
            return sorted(self._frames.keys())

    async def replace_flows(self, flows: dict[str, np.ndarray]) -> None:
        """Update the latest optical flow vectors."""
        async with self._lock:
            # Clean up old flow memmap files
            for path in self._memmap_dir.glob("flow_*.dat"):
                try:
                    path.unlink()
                except OSError:
                    pass

            for name, data in list(flows.items()):
                flows[name] = self._to_memmap(f"flow_{name}", data)
            self._flows = flows

    async def get_flows(self) -> dict[str, np.ndarray]:
        """Return the latest per-region optical flow vectors."""
        async with self._lock:
            return dict(self._flows)

    @property
    def data_bytes(self) -> int:
        """Total bytes across all nowcast frame arrays and flow fields."""
        total = 0
        for frame in self._frames.values():
            for arr in frame.regions.values():
                total += arr.nbytes
        for arr in self._flows.values():
            total += arr.nbytes
        return total

    def clear(self) -> None:
        self._frames.clear()
        self._flows.clear()

    def __getstate__(self) -> dict:
        """Serialize state for cross-process reload (multi-worker mode)."""
        frames_state: list[dict] = []
        for ts, frame in self._frames.items():
            regions: dict[str, list] = {}
            for name, arr in frame.regions.items():
                regions[name] = [
                    os.path.basename(str(arr.filename)),
                    arr.dtype.str,
                    list(arr.shape),
                ]
            frames_state.append({
                "timestamp": ts,
                "blend_weight": frame.blend_weight,
                "regions": regions,
            })
        flows_state: dict[str, list] = {}
        for name, arr in self._flows.items():
            flows_state[name] = [
                os.path.basename(str(arr.filename)),
                arr.dtype.str,
                list(arr.shape),
            ]
        return {
            "memmap_dir": str(self._memmap_dir),
            "frames": frames_state,
            "flows": flows_state,
        }

    def __setstate__(self, state: dict) -> None:
        """Restore state from the dict produced by ``__getstate__``."""
        memmap_dir = Path(state["memmap_dir"])
        new_frames: dict[int, NowcastFrame] = {}
        for f_info in state["frames"]:
            ts = int(f_info["timestamp"])
            frame = NowcastFrame(
                timestamp=ts,
                blend_weight=float(f_info["blend_weight"]),
            )
            for name, (basename, dtype_str, shape) in f_info["regions"].items():
                frame.regions[name] = np.memmap(
                    memmap_dir / basename,
                    dtype=np.dtype(dtype_str), mode="r",
                    shape=tuple(shape),
                )
            new_frames[ts] = frame

        new_flows: dict[str, np.ndarray] = {}
        for name, (basename, dtype_str, shape) in state["flows"].items():
            new_flows[name] = np.memmap(
                memmap_dir / basename,
                dtype=np.dtype(dtype_str), mode="r",
                shape=tuple(shape),
            )

        self._memmap_dir = memmap_dir
        self._frames = new_frames
        self._flows = new_flows
        self._persistent = True
        if not hasattr(self, "_lock"):
            self._lock = asyncio.Lock()

    def cleanup(self) -> None:
        """Clear data; remove the memmap dir only when non-persistent."""
        self.clear()
        if self._persistent:
            logger.info("Nowcast memmaps retained on disk at %s", self._memmap_dir)
        else:
            shutil.rmtree(self._memmap_dir, ignore_errors=True)
            logger.info("Nowcast memmap directory cleaned up")


# ---------------------------------------------------------------------------
# NowcastGenerator
# ---------------------------------------------------------------------------

class NowcastGenerator:
    """Generates nowcast frames from the latest radar data."""

    def __init__(self, store, nowcast_store: NowcastStore, cache=None):
        self._store = store          # FrameStore (past radar)
        self._nowcast_store = nowcast_store
        self._cache = cache          # TileCache (for invalidation)

    async def generate(self) -> None:
        """Generate nowcast frames from the most recent radar frames.

        Called after each fetch cycle.  Runs the CPU-heavy work in a
        thread to avoid blocking the event loop.  ``settings.nowcast_method``
        selects between the legacy ``"extrapolation"`` path (2-frame
        Farneback + cv2.remap warp) and ``"sprog"`` (3-frame pysteps
        S-PROG spectral cascade).
        """
        if not settings.nowcast_enabled:
            return

        method = settings.nowcast_method
        n_history = 3 if method == "sprog" else 2

        history = await self._store.get_last_n_frames(n_history)
        if len(history) < n_history:
            logger.debug(
                "Nowcast (%s): need %d frames, have %d",
                method, n_history, len(history),
            )
            return

        latest_ts = history[-1].timestamp
        n_steps = settings.nowcast_frames
        interval = settings.fetch_interval

        if method == "sprog":
            nowcast_frames, flows = await asyncio.to_thread(
                self._generate_sync_sprog,
                history, latest_ts, n_steps, interval,
                settings.nowcast_blend_mode,
            )
        else:
            nowcast_frames, flows = await asyncio.to_thread(
                self._generate_sync,
                history[-2].regions, history[-1].regions,
                latest_ts, n_steps, interval,
                settings.nowcast_blend_mode,
            )

        # Swap into the store and invalidate old tile cache entries
        old_timestamps = await self._nowcast_store.replace_all(nowcast_frames)
        await self._nowcast_store.replace_flows(flows)
        if self._cache is not None:
            for ts in old_timestamps:
                self._cache.invalidate_timestamp(ts)

        if nowcast_frames:
            logger.info(
                "Nowcast updated (%s): %d frames (T+%d to T+%d min)",
                method, len(nowcast_frames),
                interval // 60,
                n_steps * interval // 60,
            )

    @staticmethod
    def _generate_sync(
        prev_regions: dict[str, np.ndarray],
        latest_regions: dict[str, np.ndarray],
        latest_ts: int,
        n_steps: int,
        interval: int,
        blend_mode: str = "blended",
    ) -> list[NowcastFrame]:
        """Synchronous extrapolation-path nowcast (runs in a thread)."""
        t0 = time.monotonic()

        # Pre-compute flow per region
        flows: dict[str, np.ndarray] = {}
        for region_name in latest_regions:
            data0 = prev_regions.get(region_name)
            data1 = latest_regions.get(region_name)
            if data0 is None or data1 is None:
                continue
            flows[region_name] = _compute_flow(data0, data1)

        if not flows:
            return [], {}

        # Generate extrapolated frames for each step
        frames: list[NowcastFrame] = []
        for step in range(1, n_steps + 1):
            nowcast_ts = latest_ts + step * interval
            blend_weight = _compute_blend_weight(step, interval, blend_mode)

            regions: dict[str, np.ndarray] = {}
            for region_name, flow in flows.items():
                data = latest_regions[region_name]
                regions[region_name] = _extrapolate_forward(data, flow, step)

            frames.append(NowcastFrame(
                timestamp=nowcast_ts,
                regions=regions,
                blend_weight=blend_weight,
            ))

        elapsed = time.monotonic() - t0
        logger.info(
            "Nowcast generation (extrapolation): %d frames × %d regions (%.1fs)",
            len(frames), len(flows), elapsed,
        )
        return frames, flows

    @staticmethod
    def _generate_sync_sprog(
        history: list,
        latest_ts: int,
        n_steps: int,
        interval: int,
        blend_mode: str = "blended",
    ) -> tuple[list[NowcastFrame], dict[str, np.ndarray]]:
        """Synchronous S-PROG nowcast (runs in a thread).

        ``history`` is a list of three ``RadarFrame`` objects in
        chronological order.  Each region present in all three frames
        gets a forecast; regions missing from any frame are skipped.
        """
        from pysteps.nowcasts import sprog

        t0 = time.monotonic()

        prev2_regions = history[0].regions
        prev_regions = history[1].regions
        latest_regions = history[2].regions

        # Pre-compute flow per region (same Farneback as the
        # extrapolation path — pysteps accepts any velocity field, no
        # benefit to swapping motion estimation here).
        flows: dict[str, np.ndarray] = {}
        precip_stacks: dict[str, np.ndarray] = {}
        for region_name, latest_data in latest_regions.items():
            data0 = prev2_regions.get(region_name)
            data1 = prev_regions.get(region_name)
            if data0 is None or data1 is None:
                continue
            flows[region_name] = _compute_flow(data1, latest_data)
            # Stack as (ar_order+1, H, W) in mm/h for S-PROG.  uint8 0
            # (NODATA) maps to 0 mm/h, which S-PROG treats as no-rain
            # via the precip_thr parameter below.
            precip_stacks[region_name] = np.stack([
                uint8_to_mmh(data0),
                uint8_to_mmh(data1),
                uint8_to_mmh(latest_data),
            ])

        if not flows:
            return [], {}

        # Run S-PROG per region.  Each call produces (n_steps, H, W)
        # forecast frames in mm/h.  We transpose flow from (H, W, 2)
        # to (2, H, W) to match the pysteps semilagrangian extrapolator
        # convention.
        per_region_forecasts: dict[str, np.ndarray] = {}
        for region_name, flow in flows.items():
            precip = precip_stacks[region_name]
            velocity = flow.transpose(2, 0, 1).astype(np.float32, copy=False)
            try:
                forecast_mmh = sprog.forecast(
                    precip=precip,
                    velocity=velocity,
                    timesteps=n_steps,
                    n_cascade_levels=_SPROG_CASCADE_LEVELS,
                    ar_order=2,
                    precip_thr=0.1,
                    probmatching_method="cdf",
                    fft_method=_SPROG_FFT_METHOD,
                )
            except Exception:
                logger.exception(
                    "S-PROG forecast failed for region %s; skipping", region_name,
                )
                continue
            per_region_forecasts[region_name] = forecast_mmh

        if not per_region_forecasts:
            return [], flows

        # Assemble NowcastFrames step-by-step in chronological order.
        frames: list[NowcastFrame] = []
        for step in range(1, n_steps + 1):
            nowcast_ts = latest_ts + step * interval
            blend_weight = _compute_blend_weight(step, interval, blend_mode)

            regions: dict[str, np.ndarray] = {}
            for region_name, forecast in per_region_forecasts.items():
                # S-PROG returns the step-1 forecast at index 0.
                regions[region_name] = mmh_to_uint8(forecast[step - 1])

            frames.append(NowcastFrame(
                timestamp=nowcast_ts,
                regions=regions,
                blend_weight=blend_weight,
            ))

        elapsed = time.monotonic() - t0
        logger.info(
            "Nowcast generation (sprog): %d frames × %d regions (%.1fs)",
            len(frames), len(per_region_forecasts), elapsed,
        )
        return frames, flows


def _compute_blend_weight(step: int, interval: int, blend_mode: str) -> float:
    """Temporal blend weight between radar nowcast and NWP forecast.

    1.0 = pure radar, 0.0 = pure NWP.  Beyond the 60-min radar-skill
    window, always falls back to pure NWP regardless of mode.
    """
    max_blend_steps = max(1, 3600 // interval)  # 6 at 10-min cadence
    if step > max_blend_steps:
        return 0.0
    if blend_mode == "radar":
        return 1.0
    if blend_mode == "model":
        return 0.0
    # "blended" (default).  Tuned for HRRR's native-resolution,
    # dBZ-matched output.  Starts at ~82% radar at T+10 for a smooth
    # transition off the live frame, crosses to model-dominant by T+40,
    # lands at 20% radar by T+60.  When the chain falls back to IFS
    # (outside HRRR's CONUS domain) the same curve still applies.
    #
    # Note: this curve is tuned for the legacy extrapolation path, where
    # high-frequency convective detail persists at full intensity all
    # the way to T+60.  S-PROG decays fine-scale features naturally via
    # the per-scale AR(2) process, so a future Phase 4 will revisit
    # this curve once S-PROG is live and verified.
    t = step / max_blend_steps
    return 0.20 + 0.80 * (1.0 - t) ** 1.4


# ---------------------------------------------------------------------------
# Optical flow helpers
# ---------------------------------------------------------------------------

def _compute_flow(frame0: np.ndarray, frame1: np.ndarray) -> np.ndarray:
    """Compute dense optical flow with adaptive downscaling.

    Downscales so the longest dimension is ~``_TARGET_FLOW_DIM`` pixels,
    computes Farneback flow, then upscales the flow vectors to the
    original resolution.
    """
    h, w = frame0.shape
    scale = min(_TARGET_FLOW_DIM / max(h, w), 1.0)

    if scale < 1.0:
        small_h = max(1, int(h * scale))
        small_w = max(1, int(w * scale))
        small0 = cv2.resize(frame0, (small_w, small_h), interpolation=cv2.INTER_AREA)
        small1 = cv2.resize(frame1, (small_w, small_h), interpolation=cv2.INTER_AREA)

        flow_small = cv2.calcOpticalFlowFarneback(
            small0, small1, flow=None, **_FARNEBACK,
        )

        flow = cv2.resize(flow_small, (w, h), interpolation=cv2.INTER_LINEAR)
        flow *= 1.0 / scale  # scale vectors to full resolution
    else:
        flow = cv2.calcOpticalFlowFarneback(
            frame0, frame1, flow=None, **_FARNEBACK,
        )

    return flow


def _extrapolate_forward(
    frame: np.ndarray, flow: np.ndarray, steps: int,
) -> np.ndarray:
    """Warp *frame* forward by *steps* × flow using inverse remap.

    For each output pixel p, samples *frame* at ``p − steps·flow(p)``.
    After warping, rescales the result to preserve the total precipitation
    energy of the source frame — bilinear interpolation in cv2.remap
    tends to smooth peak values, causing artificial intensity loss.
    """
    h, w = frame.shape
    ys, xs = np.mgrid[0:h, 0:w].astype(np.float32)

    map_x = xs - steps * flow[..., 0]
    map_y = ys - steps * flow[..., 1]

    warped = cv2.remap(
        frame, map_x, map_y,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT, borderValue=0,
    )

    # Note: intensity preservation (rescaling warped pixels to match
    # source mean) was removed because bilinear interpolation only
    # loses ~1-2% per step, while new low-value boundary pixels from
    # spreading inflated the correction ratio and caused a visible
    # intensity jump on the first forecast frame.

    return warped
