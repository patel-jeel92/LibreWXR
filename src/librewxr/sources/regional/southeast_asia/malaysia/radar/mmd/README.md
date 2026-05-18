# MET Malaysia (MMD) radar composite

National radar composite from **Jabatan Meteorologi Malaysia**, served as
a single animated GIF from `api.met.gov.my/static/images/radar-latest.gif`.

## Coverage

| Region         | Footprint                                    |
| -------------- | -------------------------------------------- |
| `MYPENINSULAR` | Peninsular Malaysia + N. Sumatra + Singapore |
| `MYEAST`       | East Malaysia (Borneo) + Brunei              |

Both regions are sub-rectangles of the same 1352×570 combined GIF — one
HTTP fetch per cycle is shared between them.

## Cadence & latency

- Native cadence: **10 min** (matches LibreWXR's store cadence exactly).
- Each GIF carries **6 frames** (~60 min of backfill).
- MET publishes each 10-min slot ~11 min after its real data time, so
  frames are anchored to the current wall-clock 10-min slot rather than
  to `Last-Modified - lag` (see `_frame_timestamps` in `source.py`).

## Format

GIF89a with a discrete 18-stop colour palette (paired with dBZ via
Marshall-Palmer from labelled mm/h ticks).  Decoded by nearest-RGB match
in `source.py::_decode_mmd_palette`.  Burned-in state-border lines leave
hairline gaps which are bridged by a 3×3 morphological close
(`_fill_boundary_gaps`).

## License & attribution

**CC-BY-4.0 (METMalaysia / api.met.gov.my)**.  Attribution is recorded in
the top-level `README.md` and `docs/coverage.md`.

## Stations

12-radar national network (MY2809–MY2819, MY2865).  Coordinates and the
375 km / 350 km range overrides live in `stations.py` and feed
`data/coverage.py` via the `RadarSourceContribution`.
