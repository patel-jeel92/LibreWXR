# MARN/SNET El Salvador radar

Single S-band radar at San Andrés volcano, operated by the **Servicio
Nacional de Estudios Territoriales (SNET)** under MARN.  120 km range
product, published to the anonymous GCS bucket `radar-images-sv` under
the `esar82/Images/` prefix.

## Coverage

| Region   | Footprint                                                |
| -------- | -------------------------------------------------------- |
| `SVCOMP` | El Salvador + most of Honduras, Guatemala, S. Belize     |

Grid: 409 × 342, anisotropic ~0.926 km lon × ~1.02 km lat.

## Cadence & latency

- Native cadence: **5 min**.
- Filenames embed **El Salvador local time** (UTC-6, no DST) — must be
  converted to UTC on ingest (`MARNSource._filename_to_utc`).
- Files in the bucket archive ~24 h; older frames are pruned upstream.

## Format quirks

- PNG with a tRNS chunk marking RGB=(0,0,0) as transparent — no-data
  pixels are alpha=0 after the `convert("RGBA")` load.
- Reflectivity is encoded as an **HSV-style continuous hue gradient**
  (green → cyan → blue → magenta), not the discrete legend on the
  SNET site (which is for a different 60 km composite product).
- Decoder runs the inverse arc-detect map in `source.py::_decode_marn_png`.
- dBZ calibration is provisional (`_MARN_DBZ_MIN`=10, `_MARN_DBZ_MAX`=70);
  refine against overlapping NEXRAD coverage during the next tropical
  system that tracks the area.

## License & attribution

MARN explicitly permits full or partial reproduction **with citation**.
Citation block lives in the top-level `README.md` and `docs/coverage.md`.

## Stations

Single station — San Andrés volcano at (13.687, -88.883).  Coordinates
and the 120 km range override live in `stations.py` and feed
`data/coverage.py` via the `RadarSourceContribution`.  The override
matches the 120 km product (not the 240 km network default).
