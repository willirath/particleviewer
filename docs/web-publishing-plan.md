# Plan: publishing the Fucus trajectories with an interactive web viewer

Status: draft, 2026-11-18. Estimates are marked **measured** (on local data) or
**estimated** (extrapolated or from hardware specs, ±50 % or worse).

## Goal

Publish the full Fucus dispersal trajectory archive on a website with a
trajshade-like viewer (pan / zoom, per-year and per-window aggregations), on a
single VM to start with.

## Data

| | |
|---|---|
| full archive | Nesh: `/gxfs_work/geomar/smomw122/2025_fucus_dispersal_outputs` — 3 regimes × ~10 years |
| measured on | local 2022 copies (one regime-year), `hextraj/external/Fucus_BSH_20220101-20220625_dt60min_N453440_seed123.zarr` and `…_20220701-20221230_dt60min_N470880_seed123.zarr` |
| store 1 | 453,440 trajectories × 5,304 hourly obs; 26 releases 2022-01-01 → 06-25; ends 2022-08-09 … 08-24 |
| store 2 | 366,240 trajectories × 3,432 hourly obs; 21 releases 2022-07-01 → 11-18; **all end 2022-11-20 19:00** (looks truncated vs. its name) |
| layout | Parcels 3.0.6 Zarr v2, (trajectory, obs) with obs = hours since release; float64 lon/lat/z/time, lz4 |
| size | 10 GB on disk per regime-year; ~2.2 G valid hourly positions (**measured**) |
| domain | North Sea / Baltic, 5–30 °E, 54–64 °N |

Caveats on the local copies:

- **They predate the release fix.** In store 2, 69 % of trajectories never leave
  their release position (median age at last move 0 d, distance 0 km; release sites
  in the Danish straits, Gulf of Riga, Swedish east coast). Only 32 % of 6-hourly
  positions move (**measured**). Fixed runs will have ~2–3× more moving positions.
- Store 1 was not checked for stationary particles.
- All tile and compression measurements below use store 2 only, sampled 6-hourly.

## Findings

### Trajectory compression (lon/lat, **measured** on 50 k trajectories, 6-hourly)

| encoding | B / vertex | max error |
|---|---|---|
| as stored (float64, lz4) | 3.14 | 0 |
| float32 + zstd | 0.74 | float32 |
| quantise 10 m + delta + zstd | 0.58 | 6 m |
| **quantise 100 m + delta + zstd** | **0.37** | 63 m |
| quantise 500 m + delta + zstd | 0.24 | 313 m |
| ZFP (1-D per trajectory), 100 m tolerance | 2.9 | 68 m |
| Douglas–Peucker 1 km + quantise | 0.16 | ~1.1 km (shape) |

- Quantise + delta + zstd is the effective error-bounded lossy coder (SZ-like);
  ZFP loses by 5–10× because stationary steps cost it a per-block floor.
- Per moving vertex (stationary runs collapsed, exact timing kept): 1.23 B at 100 m.
- Douglas–Peucker keeps 6–13 % of positions: little storage gain, but an
  8–15× rendering gain. It preserves shape, not timing (time-interpolation error
  up to ~50–65 km between kept vertices).
- `time`, `age_sec` are derivable; `z` ∈ {0, 1}; `cell_ID` is per release —
  none needs storing per position.

### Pre-rendered raster tiles (**measured**, store 2, one aggregation over the half-year)

Cumulative MB per layer down to zoom level:

| encoding | ≤ z9 (157 m/px) | ≤ z10 (79 m/px) | ≤ z11 (39 m/px) |
|---|---|---|---|
| PNG 8-bit palette | 16 | 59 | 201 |
| uint16 values + zstd | 16 | 57 | 188 |
| WebP q75 | 6 | 33 | 170 |
| AVIF q50 | 4 | 18 | 106 |

z10 is the useful limit for 6-hourly data (segments are km long). Lossy
formats save 2–4× at coarse zoom, little at z11 (sparse tiles). Visual quality of
lossy tiles not checked.

### Rendering throughput (**measured**, one CPU core)

datashader line `count`: ~27 M vertices/s; trajshade's particle-weighted numba
kernel (`trajectories`, `pmean`, `pstd`): ~65 M vertices/s. The current browser
viewer is limited by the image payload, not aggregation (issue #1: 10 MB float
image per update at 2× DPR).

## Architecture

1. **Raster tile pyramid** for standard layers (regime × year × month ×
   reduction, plus climatologies), z ≤ 10. Static files, cacheable, cheapest to
   serve and transfer. Built offline on Nesh.
2. **Tiled, quantised trajectory store** for live rendering: trajectories clipped
   per tile and zoom level, tile-local int16 coordinates (4096 grid), simplified
   to ~½ px per level, record index + particle id per piece (keeps time windows
   and particle-weighted reductions exact). Per-view cost is bounded (~10–50 M
   positions) regardless of archive size. ~100–200 MB per regime-year, ~3–6 GB
   total (**estimated**).
   - Own binary format (Zarr / PMTiles) read by trajshade kernels server-side —
     exact reductions. Preferred.
   - Standard MVT via tippecanoe + MapLibre / deck.gl client-side — quick
     prototype, approximate density only, per-position time needs feature splitting.
3. **Server-side rendering** (trajshade) for custom windows / reductions, with the
   issue #1 fixes (server-side shading, throttling).

Recommendation: 1 for coarse zoom and default layers, 2 + 3 for deep zoom and
custom queries. Stranding / release-site locations as a separate point layer
(one point per particle), not inside every aggregation.

## Resource estimates (full archive, 30 regime-years, **estimated**)

### Storage

| form | per regime-year | × 30 |
|---|---|---|
| as stored | ~10 GB | ~300 GB |
| hourly, 100 m, delta + zstd | 0.8–1.5 GB | 25–45 GB |
| 6-hourly, 100 m | 0.2–0.4 GB | 6–12 GB |
| tiled trajectory store | 0.1–0.2 GB | 3–6 GB |
| raster tiles, ~1,100 monthly layers to z10 | | ≤ 65 GB PNG / ≤ 20 GB AVIF |
| raster tiles, climatologies only (~110 layers) | | ~6 GB PNG / ~2 GB AVIF |

### VM

| role | vCPU | RAM | disk |
|---|---|---|---|
| raster tiles only | 2 | 4 GB | 50–100 GB |
| tiles + live viewer, one regime-year at a time | 8 | 32 GB | 100–150 GB |

Live CPU rendering: one regime-year at 6-hourly ≈ 0.3–0.4 G positions ≈ 3 GB RAM;
full-extent view ~1–2 s on 8 cores, bounded per view with tiling / LOD. The
whole archive (~10 G positions, ~80 GB float32) does not fit in RAM; load per run.

### GPU (**estimated** from specs, not benchmarked)

| per view | 8 CPU cores | A40 (48 GB, 0.7 TB/s) | A100 (40/80 GB, 1.6–2 TB/s) |
|---|---|---|---|
| LOD view, counts / means | 0.3–1 s | 10–50 ms | 5–25 ms |
| LOD view, particle-weighted | 0.5–1.5 s | 30–100 ms | 15–50 ms |
| full regime-year, particle-weighted | 2–4 s | 0.3–1 s | 0.15–0.5 s |
| 10 years of one regime, live | 15–30 s | 0.5–3 s | 0.2–1.5 s |

With LOD a GPU makes rendering nearly free; encoding (50–200 ms) and network then
dominate. The GPU case is live multi-year / multi-regime aggregation. A40 is the
better value for serving; A100's advantages (bandwidth, 40 MB L2) give ~1.5–3×.
Porting cost: datashader GPU covers simple reductions only; the particle-weighted
kernels need a numba.cuda / CuPy port (sort-based dedup), a few days.

### Example session transfer (**estimated**)

Full Baltic for one year → zoom to Bornholm Basin, flip 10 years → zoom out,
Kiel Bight, flip 10 years:

| | raster tiles | server-rendered views | vector tiles in browser |
|---|---|---|---|
| total | 20–25 MB PNG / 10–12 MB AVIF | 15–30 MB | 40–150 MB |

Raster cost scales with screen area (~20–30 kB per non-empty tile); vector cost
scales with data (~2 B per kept position), worst at coarse zoom. Repeat flips
hit the browser cache.

## Next steps

1. **Measure on Nesh**: one fixed regime-year — positions per run, moving share,
   tile sizes for monthly layers. Turns the ±50 % above into numbers.
2. trajshade issue #1: server-side shading, throttled updates, float32 output.
3. Prototype the tiled quantised trajectory store for one regime-year (own
   format) and render from it in trajshade; compare with a tippecanoe/PMTiles +
   MapLibre prototype.
4. Raster pyramid builder as a Nesh array job; decide layer set (reductions ×
   windows) and encoding (PNG vs AVIF, after a visual check).
5. Decide CPU-only vs GPU after 1–3; benchmark on an A40 if live multi-year
   aggregation is wanted.

## Open questions

- Which layers are the public default (reductions, windows, climatologies,
  regime differences)?
- Treatment of stranded / never-moving particles in the fixed runs.
- Hourly vs 6-hourly as the base for deep zoom.
- Hosting: institutional VM (GPU available?) vs cloud.
