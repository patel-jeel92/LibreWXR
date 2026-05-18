# MSC Canada (ECCC GeoMet) radar composite

Canadian radar composite operated by **Environment and Climate Change
Canada**, served via the MSC GeoMet WMS at `geo.weather.gc.ca/geomet`.

## Coverage

| Region   | Footprint                                          |
| -------- | -------------------------------------------------- |
| `CACOMP` | Canada (-141°W to -52°W, 41°N to 84°N)             |

Latlon grid at 0.025° (~3560 × 1720 cells).  Resolution chosen to keep
the single-request WMS tile under typical server caps.

## Cadence & latency

- Native cadence: **~6 min**.
- WMS `TIME` dimension exposes a rolling **~3-hour history**.
- The most recent 1–2 slots are often unpublished; the source walks
  back up to `_MAX_TIME_RETRIES` 6-min steps on XML
  `ServiceExceptionReport`.

## Format quirks

- MSC publishes **pre-coloured PNG only** — there is no WCS/TIFF path
  for raw radar values.
- Decoder uses the 14-bucket `Radar-Rain_Dis-14colors` palette and
  nearest-anchor lookup with an 8.0 Euclidean RGB distance threshold
  (composite pixels are within ±1 per channel of the legend colours
  due to server-side rendering rounding).
- Bucket values use the **geometric mean** of each [lower, upper) edge
  rather than the lower-edge label, so the encoded mm/h is the typical
  value within the bucket.  Using the lower edge would systematically
  under-report by ~30% and push the lowest bucket below the 10 dBZ
  noise floor.
- Top bucket (≥ 200 mm/h, unbounded) is represented as 250 mm/h — a
  reasonable "typical extreme" that preserves headroom under the
  clamp ceiling.
- mm/h is then converted to dBZ via **Marshall-Palmer**
  (`Z = 200 · R^1.6`, `dBZ = 10·log10(Z)`).

## Cross-source role

Beyond serving CACOMP as a standalone source, MSC Canada is also used:

1. As a **fallback** for the US-group MRMS regions in `mrms_fallback`
   mode (when MRMS misses, IEM is tried first, then MSC for CACOMP).
2. As a **blend partner** for CACOMP in `mrms_fallback` mode: MRMS data
   takes priority within its CONUS+border extent (~20–55°N), MSC fills
   gaps north of 55°N and outside the MRMS lon range.

Both of these live in `librewxr/data/fetcher.py` (`_try_fallback`,
`_blend_cacomp`).  They import `MSCCanadaSource` directly from this
package — the discovery path is only used to wire the standalone CACOMP
source.  Phase 4 of the sources refactor may factor the cross-source
policy into a `_resolve_group_policy` helper.

## License & attribution

ECCC publishes under the **Canadian Open Government Licence**.
Citation block in the top-level `README.md` and `docs/coverage.md`.

## Stations

32 S-band dual-pol radars (METEOR 1700S) after the 2023 ECCC network
modernization.  Full list in `stations.py`; feeds `data/coverage.py` via
the `RadarSourceContribution`.  Without the station-circle mask the
ECMWF fallback would be suppressed across the entire CACOMP bbox (open
Pacific, Arctic, Atlantic) where MSC has no real coverage.
