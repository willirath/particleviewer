# trajshade

Interactive [datashader](https://datashader.org) aggregations of particle trajectories
from a Zarr store (kelp output format 2, or Parcels layout with trajectory × obs).
Every pan / zoom re-aggregates the lines at screen resolution.

```bash
uv sync
uv run trajshade tracks.zarr          # browser app (Panel / HoloViews / Bokeh)
uv run trajshade tracks.zarr --mpl    # native matplotlib window
uv run trajshade tracks.zarr --png out.png --how pstd --var z --records 0:120
```

## Reductions

Segment-weighted (datashader; every segment crossing a pixel counts, so particles
looping inside a pixel dominate):

| | |
|---|---|
| `segments` | segments crossing each pixel |
| `dwell` | records spent in each pixel (vertices only) |
| `mean` `std` `min` `max` `range` | of a per-vertex variable along the lines (`min(record)` = first arrival) |

Particle-weighted (own numba line raster; each particle counts once per pixel):

| | |
|---|---|
| `trajectories` | distinct particles touching each pixel |
| `pmean` | each particle's in-pixel mean of the variable, averaged over particles |
| `pstd` | std across particles of those in-pixel means (blank where < 2 particles) |

Variables are the store's 2-D fields plus `record` and `age_days`.

## Controls

Browser: reduction, variable, end-record player and span (trailing window),
particle stride (default ≈ 5 M vertices), norm, colormap, antialiasing, alive only,
graticule. The canvas follows the device pixel ratio (`--scale` to force).

Matplotlib window: `r`/`R` reduction, `v`/`V` variable, `n` norm, `a` antialiasing,
`,` `.` step the record window, `0` reset view.

The image must map 1:1 onto device pixels: both front ends draw it with
nearest-neighbour resampling, which would drop 1-px lines when downsampling.

Projection is polar stereographic for runs confined to high latitudes, plate
carrée otherwise (`--proj`).
