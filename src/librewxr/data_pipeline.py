# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Joshua Kimsey
"""Standalone data-pipeline process for the multi-worker tile-server split.

This is the "master" half of the split: a single asyncio process that
fetches radar frames + auxiliary NWP grids, runs the nowcast generator,
and dumps a cross-process state snapshot under ``LIBREWXR_CACHE_DIR``
after every cycle.

The companion render-only tile-server workers (``librewxr.main`` with
``LIBREWXR_RENDER_ONLY=1``) memory-map the same files and refresh their
in-memory store views by polling ``state.json``'s mtime.

Run with::

    python -m librewxr.data_pipeline

``LIBREWXR_CACHE_DIR`` is required.  Without it there is nothing to
share with the render workers and the script exits immediately.
"""
from __future__ import annotations

import asyncio
import logging
import signal
import sys
from pathlib import Path

from rich.logging import RichHandler

from librewxr.config import settings
from librewxr.data.alerts_fetcher import WMOAlertsFetcher
from librewxr.data.alerts_store import AlertsStore
from librewxr.data.arome_antilles_grid import AROMEAntillesGrid
from librewxr.data.cloud_grid import CloudGrid
from librewxr.data.coverage import build_coverage_masks, build_feather_masks
from librewxr.data.dmi_dini_grid import DMIDiniGrid
from librewxr.data.ecmwf_grid import ECMWFGrid
from librewxr.data.fetcher import RadarFetcher
from librewxr.data.hrdps_grid import HRDPSGrid
from librewxr.data.hrrr_alaska_grid import HRRRAlaskaGrid
from librewxr.data.hrrr_grid import HRRRGrid
from librewxr.data.icon_eu_grid import ICONEUGrid
from librewxr.data.master_state import dump_state
from librewxr.data.nowcast import NowcastGenerator, NowcastStore
from librewxr.data.nwp_source import NWPChain
from librewxr.data.radar_cache import RadarFrameCache
from librewxr.data.radar_stations import MRMS_STATIONS
from librewxr.data.regions import REGIONS
from librewxr.data.store import FrameStore
from librewxr.data.wrf_smn_grid import WRFSMNGrid
from librewxr.tiles.cache import TileCache

# The pipeline writes no tiles itself, but RadarFetcher invalidates a
# TileCache on frame eviction.  A shared one here would be useless to
# the render workers (different process, no cross-process invalidation),
# so we hand it a tiny no-op-effect cache and rely on the render workers
# to invalidate their own caches when they pick up a new state.json.

_LOG_TAGS = {
    "librewxr.data_pipeline": "pipeline",
    "librewxr.config": "config",
    "librewxr.data.sources": "radar",
    "librewxr.data.fetcher": "fetcher",
    "librewxr.data.store": "store",
    "librewxr.data.regions": "regions",
    "librewxr.data.coverage": "coverage",
    "librewxr.data.ecmwf_grid": "ifs",
    "librewxr.data.ecmwf_interpolation": "ifs",
    "librewxr.data.hrrr_grid": "hrrr",
    "librewxr.data.hrrr_alaska_grid": "hrrr-ak",
    "librewxr.data.icon_eu_grid": "icon-eu",
    "librewxr.data.dmi_dini_grid": "dmi-dini",
    "librewxr.data.hrdps_grid": "hrdps",
    "librewxr.data.arome_antilles_grid": "arome-ant",
    "librewxr.data.wrf_smn_grid": "wrf-smn",
    "librewxr.data.cloud_grid": "cloud",
    "librewxr.data.cloud_cache": "cloud",
    "librewxr.data.nowcast": "nowcast",
    "librewxr.data.master_state": "state",
    "librewxr.data.alerts_fetcher": "alerts",
    "librewxr.data.alerts_store": "alerts",
}


class _TagFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        record.tag = _LOG_TAGS.get(record.name, record.name.rsplit(".", 1)[-1])
        return super().format(record)


def _setup_logging() -> None:
    handler = RichHandler(rich_tracebacks=True, show_path=False)
    handler.setFormatter(_TagFormatter("[%(tag)s] %(message)s"))
    logging.basicConfig(level=logging.INFO, handlers=[handler], force=True)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)


logger = logging.getLogger(__name__)


async def run_pipeline() -> None:
    """Construct the pipeline, run until signalled, then shut down cleanly."""
    if not settings.cache_dir:
        raise SystemExit(
            "LIBREWXR_CACHE_DIR must be set when running the data pipeline — "
            "render-only workers need a shared directory to read snapshots from."
        )
    cache_dir = Path(settings.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    enabled = settings.get_enabled_regions()
    logger.info(
        "Pipeline starting (cache_dir=%s, regions=%s, fetch_interval=%ds)",
        cache_dir, ", ".join(enabled), settings.fetch_interval,
    )

    # All persistent stores share the same cache_dir so render workers can
    # memmap the same files and __setstate__ from a single state.json.
    store = FrameStore(max_frames=settings.max_frames, cache_dir=cache_dir)
    tile_cache = TileCache(max_mb=1)  # noop-effect (see module docstring)

    ecmwf_grid = ECMWFGrid(cache_dir=cache_dir) if settings.ecmwf_enabled else None
    if settings.na_nwp_source == "hrrr":
        hrrr_grid = HRRRGrid(cache_dir=cache_dir)
        hrrr_alaska_grid = HRRRAlaskaGrid(cache_dir=cache_dir)
    else:
        hrrr_grid = None
        hrrr_alaska_grid = None
    if settings.eu_nwp_profile in ("icon_eu_only", "dini_with_icon_eu"):
        icon_eu_grid = ICONEUGrid(cache_dir=cache_dir)
    else:
        icon_eu_grid = None
    if settings.eu_nwp_profile == "dini_with_icon_eu":
        dmi_dini_grid = DMIDiniGrid(cache_dir=cache_dir)
    else:
        dmi_dini_grid = None
    hrdps_grid = HRDPSGrid(cache_dir=cache_dir) if settings.hrdps_enabled else None
    arome_antilles_grid = (
        AROMEAntillesGrid(cache_dir=cache_dir)
        if settings.arome_antilles_enabled else None
    )
    wrf_smn_grid = WRFSMNGrid(cache_dir=cache_dir) if settings.wrf_smn_enabled else None

    chain_sources = []
    if hrrr_grid:
        chain_sources.append(hrrr_grid)
    if hrrr_alaska_grid:
        chain_sources.append(hrrr_alaska_grid)
    if hrdps_grid:
        chain_sources.append(hrdps_grid)
    if arome_antilles_grid:
        chain_sources.append(arome_antilles_grid)
    if dmi_dini_grid:
        chain_sources.append(dmi_dini_grid)
    if icon_eu_grid:
        chain_sources.append(icon_eu_grid)
    if wrf_smn_grid:
        chain_sources.append(wrf_smn_grid)
    if ecmwf_grid is not None:
        chain_sources.append(ecmwf_grid)
    nwp_chain = NWPChain(chain_sources)
    logger.info("NWP chain: [%s]", ", ".join(s.name for s in nwp_chain.sources))

    cloud_grid = CloudGrid(cache_dir=cache_dir) if settings.satellite_enabled else None

    coverage_overrides = (
        MRMS_STATIONS if settings.na_source in ("mrms", "mrms_fallback") else None
    )
    build_coverage_masks(station_overrides=coverage_overrides)
    build_feather_masks()

    nowcast_store = None
    nowcast_generator = None
    if settings.nowcast_enabled:
        nowcast_store = NowcastStore(cache_dir=cache_dir)
        nowcast_generator = NowcastGenerator(store, nowcast_store, cache=tile_cache)
        logger.info("Nowcast enabled: %d frames", settings.nowcast_frames)

    # RadarFrameCache lets the pipeline restart and re-populate its
    # FrameStore from the prior session's frames before the first fetch
    # completes — without it, render workers would see an empty
    # state.json on cold start.
    radar_cache = RadarFrameCache(cache_dir)
    regions_by_name = {name: REGIONS[name] for name in enabled}
    restored = radar_cache.load_frames(regions_by_name)
    if restored:
        for frame in restored:
            await store.add_frame(frame)
        logger.info(
            "Restored %d radar frame(s) from disk cache (%d → %d)",
            len(restored), restored[0].timestamp, restored[-1].timestamp,
        )

    # Stores keyed by the same names master_state expects.  None entries
    # are skipped by dump_state.
    stores = {
        "frame_store": store,
        "ecmwf_grid": ecmwf_grid,
        "hrrr_grid": hrrr_grid,
        "hrrr_alaska_grid": hrrr_alaska_grid,
        "hrdps_grid": hrdps_grid,
        "arome_antilles_grid": arome_antilles_grid,
        "wrf_smn_grid": wrf_smn_grid,
        "icon_eu_grid": icon_eu_grid,
        "dmi_dini_grid": dmi_dini_grid,
        "cloud_grid": cloud_grid,
        "nowcast_store": nowcast_store,
    }

    async def on_cycle_complete() -> None:
        try:
            dump_state(stores, cache_dir)
        except Exception:
            logger.exception("Failed to dump master state snapshot")

    fetcher = RadarFetcher(
        store, tile_cache,
        ecmwf_grid=ecmwf_grid,
        hrrr_grid=hrrr_grid,
        hrrr_alaska_grid=hrrr_alaska_grid,
        hrdps_grid=hrdps_grid,
        arome_antilles_grid=arome_antilles_grid,
        wrf_smn_grid=wrf_smn_grid,
        icon_eu_grid=icon_eu_grid,
        dmi_dini_grid=dmi_dini_grid,
        cloud_grid=cloud_grid,
        nowcast_generator=nowcast_generator,
        warmer=None,  # tile warming is the render workers' job
        radar_cache=radar_cache,
        on_cycle_complete=on_cycle_complete,
    )

    alerts_store = None
    alerts_fetcher = None
    if settings.alerts_enabled:
        alerts_cache = (
            cache_dir if settings.cache_dir
            else (Path(settings.alerts_cache_dir) if settings.alerts_cache_dir else None)
        )
        alerts_store = AlertsStore()
        alerts_fetcher = WMOAlertsFetcher(
            store=alerts_store,
            cache_dir=str(alerts_cache) if alerts_cache else None,
            interval=settings.alerts_fetch_interval,
            concurrency=settings.alerts_concurrency,
        )
        await alerts_fetcher.start()
        logger.info(
            "Alerts: WMO ingest started (interval=%ds)",
            settings.alerts_fetch_interval,
        )

    await fetcher.start()
    logger.info("Pipeline running — Ctrl-C / SIGTERM to stop")

    stop_event = asyncio.Event()
    loop = asyncio.get_running_loop()

    def _signal_stop(signame: str) -> None:
        logger.info("Received %s, shutting down…", signame)
        stop_event.set()

    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, _signal_stop, sig.name)
        except NotImplementedError:
            # Windows: signal handlers via the loop aren't supported, but
            # Ctrl-C still raises KeyboardInterrupt at the top level.
            pass

    try:
        await stop_event.wait()
    finally:
        await fetcher.stop()
        if alerts_fetcher is not None:
            await alerts_fetcher.close()
        if cloud_grid is not None:
            await cloud_grid.close()
        if nowcast_store is not None:
            nowcast_store.cleanup()
        store.cleanup()
        logger.info("Pipeline shutdown complete")


def main() -> None:
    _setup_logging()
    try:
        asyncio.run(run_pipeline())
    except KeyboardInterrupt:
        # Belt-and-braces: signal_handler path is the normal route, but on
        # platforms where add_signal_handler is unavailable we still exit
        # cleanly on Ctrl-C.
        sys.exit(0)


if __name__ == "__main__":
    main()
