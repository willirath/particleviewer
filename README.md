# kelp-livetrack

A live terminal view of a running [kelp](../kelp) experiment. It is a sidecar: it
only reads the growing Zarr trajectory store (`data.output_path: ….zarr`) and
never touches the run.

```bash
uv sync
uv run kelp-live run.yaml                    # a kelp config: finds output_path and mesh.meshpath
uv run kelp-live tracks.zarr --mesh mesh.nc  # or point at the store directly
uv run kelp-live tracks.zarr --once --png g.png   # one gallery to stdout (+ PNG), no TUI
```

It waits for the store to appear and for the first complete record, then polls
every `--poll` seconds (default 2).

## Panels

| key | panel | |
|---|---|---|
| 1 | density | log-scaled alive particles per pixel over land / bathymetry |
| 2 | particles by status | braille dots; alive / beached / outside |
| 3 | trails | 250 particles, last 120 records, bright = recent (zoom in to see pathways) |
| 4 | field | cell mean of a sampled variable (`v` cycles; temperature first) |
| 5 | depth | cell mean particle depth (3-D) |
| 6 | distance from release | cell mean great-circle distance from the seed position |
| 7 | section | latitude × depth density, √z axis (3-D) |
| 8 | stats | progress, ETA, status counts, time-series sparklines, depth histogram |

The grid starts with 1, 4, 7, 8 for 3-D runs and 1, 2, 4, 8 for 2-D (`--show` to change).
A number key shows that panel full screen.

## Keys

`1`–`8` show that panel full screen (same key again, `esc` or `g`: back to the grid), `a` grid of all eight · `,` `.` step one record back/forward, `<` `>` ten,
`home` first, `f`/`end` back to live · `space`/`p` play/pause a replay of the
recorded history (from live it starts at record 1, at the newest record it hands
back to live), `[` `]` slower/faster (1–64 records/s; above 8/s it skips records),
also as `|◀ ▶ ▶|` buttons in the top bar · `P` pause polling · `+` `-` zoom,
arrows / `hjkl` pan, `0` reset · `v` next sampled variable · `s` save PNG · `q` quit.

## How it reads the store

kelp's Zarr writer buffers records in RAM and flushes a whole time-chunk at a time:
`time` is resized first, then each per-record variable in turn. A record counts
as complete only when it lies inside `time`, `lon`, `lat`, `z` and `status` and its
`status` column is not all fill (0). Read errors from a chunk being written are
retried on the next poll.

How often new records appear depends on the writer's chunk width: about
4 MB / (8 B × particles), between 1 and 256 records. Small runs therefore show up
in bursts; runs with ≥ ~16k particles appear record by record.

On start-up a long backlog is read with a stride (`--backlog`, default 120 records)
so the time series start fast; new records are then read one by one. Scrubbing
reads the requested record on demand; replay reads ahead in blocks into a cache
of about 1 GB.

## Background

Land and bathymetry come from the mesh by nearest-neighbour lookup from each
display pixel, as in [ushow](https://github.com/koldunovn/ushow):

- NEMO `mesh_mask` / `domain_cfg`: `glamt`/`gphit`, `tmaskutil` (or surface `tmask`), `mbathy` + `gdepw_1d`
- FESOM2 mesh diag and any file with lon/lat points: all points wet, land where there are none

Without a mesh the maps draw on a plain background. Projection is north/south
polar stereographic for runs confined to high latitudes, plate carrée otherwise
(`--proj` to force).

## Replay (demo / testing)

```bash
uv run kelp-live-replay finished.zarr /tmp/growing.zarr --rate 2 --block 4 &
uv run kelp-live /tmp/growing.zarr --mesh mesh_mask.nc
```

`kelp-live-replay` copies a finished store record block by record block, in the
same order as kelp's writer, so the viewer sees the same torn intermediate states.

## Limits

- Zarr only (NetCDF output holds an HDF5 lock while the run writes).
- Every new record is read in full (lon, lat, z, status, sampled variables):
  about 30 MB per million particles per variable.
- HEALPix and FESOM 1.4 ASCII meshes are not read for the background yet.

## Tests

```bash
uv run pytest
```
