# CWA Taiwan QPESUMS composite

7-radar QPESUMS composite reflectivity (`O-A0059-001` / 雷達合成回波)
published by the **Central Weather Administration (中央氣象署)** to
anonymous AWS S3 (`cwaopendata` in `ap-northeast-1`).

## Coverage

| Region   | Footprint                                           |
| -------- | --------------------------------------------------- |
| `TWCOMP` | Taiwan + surrounding waters (115–126.5°E, 18–29°N)  |

Grid: 921 × 881 at 0.0125° (~1.4 km).

## Cadence & latency

- Native cadence: **10 min**, clock-aligned in **Taipei local time
  (UTC+8, no DST)**.
- Files publish ~6 min after their frame time, so the most-recent 1–2
  slots may 404 — the source walks back up to 3 older slots on
  fall-through (`_MAX_FALLBACK_STEPS = 3`).

## Format quirks

- Archive key uses NO separator dot between timestamp and product name
  (`{YYYYMMDDHHMM}compref_mosaic.xml`).  The sibling gauge keys
  `{YYYYMMDDHHMM}.QPESUMS_GAUGE.10M.xml` *do* use a dot — easy to mix
  up.
- XML data is row-major **south-to-north** — decoder vertically flips
  to north-up.
- Sentinels: `-99` invalid, `-999` outside radar range / QC-removed.
- Datum is TWD67; sub-pixel offset vs WGS84 is below rendering
  resolution.

## License & attribution

**Open Government Data License v1.0 (data.gov.tw)** — attribution
required.  See top-level `README.md` and `docs/coverage.md` for the
citation block.

## Stations

7-radar national network: Wufenshan, Hualien, Qigu, Kenting, Shulin,
Nantun, Linyuan.  Coordinates and the 450 km range override live in
`stations.py` and feed `data/coverage.py` via the
`RadarSourceContribution`.
