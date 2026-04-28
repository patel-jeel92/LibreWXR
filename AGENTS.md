# AGENTS.md - LibreWRX

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

Always use the project venv `.venv/`; never install to system Python.

## Running & Testing

```bash
python -m librewxr.main   # dev server (uvicorn)
pytest                     # all tests
pytest -m api              # by marker (api|ecmwf|nowcast|sources|tiles|store)
pytest tests/test_renderer.py  # single file
pytest -k "test_tile_render"    # by name pattern
```

All tests are auto-async (`asyncio_mode = "auto"` in pyproject.toml). No explicit `@pytest.mark.asyncio` needed on individual async tests (though some older tests still have it).

No linter/formatter/typechecker is configured — there is no `ruff`, `mypy`, `black`, etc. in the project.

## Key Architecture Facts

- **Source layout:** `src/librewxr/` (hatchling build backend, editable install via `pip install -e ".[dev]"`)
- **Entry point:** `python -m librewxr.main` starts uvicorn; app object is `librewxr.main:app`
- **Shared state wiring:** Lifespan in `main.py` creates all singletons (store, cache, fetcher, ecmwf_grid, nowcast, cloud) and assigns them to `routes` module-level vars — dependencies are NOT injected via FastAPI's DI.
- **Region dispatch:** `RadarFetcher.__init__` maps region groups to source classes; adding a region requires modifying `regions.py`, `sources.py`, and `fetcher.py`
- **Data encoding:** Radar frames are `dict[str, np.ndarray]` keyed by region name (`USCOMP`, `CACOMP`, etc.), stored as uint8 dBZ values
- **Memory:** Heavily uses numpy memmap (temp files) for radar frames, ECMWF grids, and nowcast data. Memory grows with number of enabled regions and frames. See docker-compose.yml for RAM guidance.
- **Tile rendering:** Two separate thread pools — one for on-demand requests, one for background tile warming — so requests never queue behind warming tasks.

## Configuration

All config via `LIBREWXR_*` env vars or `.env` file. Settings defined in `src/librewxr/config.py`. Key ones an agent might need to know:

- `LIBREWXR_ENABLED_REGIONS`: `ALL`, `CONUS`, `US`, `CANADA`, `EUROPE`, or comma-separated region names
- `LIBREWXR_MAX_FRAMES`: default 12 (number of past radar frames to keep)
- `LIBREWXR_MAX_ZOOM`: default 12
- `LIBREWXR_NOWCAST_ENABLED`: default true
- `LIBREWXR_SATELLITE_ENABLED`: default true
- `LIBREWXR_CACHE_DIR`: persistent disk cache for satellite grids; empty = in-memory only

## Conventions

- **File headers:** `# SPDX-License-Identifier: AGPL-3.0-or-later` + `# Copyright (C) 2026 Joshua Kimsey` on every source file
- **Commit style:** imperative mood, concise (e.g., "Add precipitation motion arrows")
- **Docker:** `docker compose up --build` — exposes port 8080 (configurable via `LIBREWXR_PORT`)