"""Line and point aggregations of (particle, record) trajectory arrays.

Segment-weighted (datashader, every segment crossing a pixel counts):
  segments      count of segments; a particle looping inside a pixel counts many times
  dwell         count of vertices (records spent in each pixel; lines not drawn)
  mean/std/min/max/range   of a per-vertex value along the lines

Particle-weighted (own numba raster; each particle counts once per pixel):
  trajectories  distinct particles touching each pixel
  pmean         mean over particles of each particle's in-pixel mean value
  pstd          spread (std) across particles of those in-pixel means
"""
from __future__ import annotations

import datashader as ds
import numba
import numpy as np
import pandas as pd
import xarray as xr

REDUCTIONS = ("trajectories", "segments", "dwell", "mean", "std", "min", "max", "range",
              "pmean", "pstd")
COUNTS = ("segments", "trajectories", "dwell")
NEEDS_VALUE = tuple(r for r in REDUCTIONS if r not in COUNTS)
PARTICLE = ("trajectories", "pmean", "pstd")


def line_frame(x: np.ndarray, y: np.ndarray, v: np.ndarray | None = None) -> pd.DataFrame:
    """Flatten (P, K) arrays into one NaN-separated polyline column set (datashader axis=0)."""
    P = x.shape[0]
    gap = np.full((P, 1), np.nan)
    cols = {"x": np.concatenate([x, gap], 1).ravel(), "y": np.concatenate([y, gap], 1).ravel()}
    if v is not None:
        cols["v"] = np.concatenate([np.asarray(v, np.float32), gap.astype(np.float32)], 1).ravel()
    return pd.DataFrame(cols)


@numba.njit(cache=True, nogil=True, inline="always")
def _visit(j, val, tag, stamp, touched, nt, psum, pn):
    if stamp[j] != tag:
        stamp[j] = tag
        touched[nt] = j
        nt += 1
        psum[j] = 0.0
        pn[j] = 0
    if np.isfinite(val):
        psum[j] += val
        pn[j] += 1
    return nt


@numba.njit(cache=True, nogil=True)
def _particle_raster(x, y, v, x0, y0, sx, sy, W, H, cnt, vcnt, vsum, vsq):
    """Per pixel: distinct particles (cnt) and moments of their in-pixel means."""
    P, K = x.shape
    stamp = np.zeros(W * H, np.int64)
    touched = np.empty(W * H, np.int64)
    psum = np.zeros(W * H, np.float64)
    pn = np.zeros(W * H, np.int64)
    for p in range(P):
        tag = p + 1
        nt = 0
        last = -1
        for k in range(K):
            ax = (x[p, k] - x0) * sx
            ay = (y[p, k] - y0) * sy
            if not (np.isfinite(ax) and np.isfinite(ay)):
                last = -1
                continue
            val = v[p, k]
            has_next = k + 1 < K and np.isfinite(x[p, k + 1]) and np.isfinite(y[p, k + 1])
            if not has_next:
                # isolated / final vertex: mark its own pixel
                px, py = int(np.floor(ax)), int(np.floor(ay))
                if 0 <= px < W and 0 <= py < H and py * W + px != last:
                    nt = _visit(py * W + px, val, tag, stamp, touched, nt, psum, pn)
                last = -1
                continue
            bx = (x[p, k + 1] - x0) * sx
            by = (y[p, k + 1] - y0) * sy
            # Liang-Barsky clip of the segment to [0, W] x [0, H]
            t0, t1 = 0.0, 1.0
            dx, dy = bx - ax, by - ay
            ok = True
            for pp, qq in ((-dx, ax), (dx, W - ax), (-dy, ay), (dy, H - ay)):
                if pp == 0.0:
                    if qq < 0.0:
                        ok = False
                        break
                else:
                    r = qq / pp
                    if pp < 0.0:
                        if r > t1:
                            ok = False
                            break
                        if r > t0:
                            t0 = r
                    else:
                        if r < t0:
                            ok = False
                            break
                        if r < t1:
                            t1 = r
            if not ok:
                last = -1
                continue
            cx, cy = ax + t0 * dx, ay + t0 * dy
            ex, ey = ax + t1 * dx, ay + t1 * dy
            n = int(max(abs(ex - cx), abs(ey - cy))) + 1
            for s in range(n):  # end vertex belongs to the next segment
                t = s / n
                px = int(np.floor(cx + t * (ex - cx)))
                py = int(np.floor(cy + t * (ey - cy)))
                if 0 <= px < W and 0 <= py < H:
                    j = py * W + px
                    if j != last:  # one sample per pixel entry
                        nt = _visit(j, val, tag, stamp, touched, nt, psum, pn)
                        last = j
        for i in range(nt):
            j = touched[i]
            cnt[j] += 1
            if pn[j] > 0:
                m = psum[j] / pn[j]
                vcnt[j] += 1
                vsum[j] += m
                vsq[j] += m * m


def particle_aggregate(x, y, v, x_range, y_range, width, height):
    """(count, mean, std) on a (height, width) grid; each particle counts once per pixel."""
    n = width * height
    cnt, vcnt = np.zeros(n, np.int64), np.zeros(n, np.int64)
    vsum, vsq = np.zeros(n), np.zeros(n)
    if v is None:
        v = np.full(x.shape, np.nan, np.float32)
    sx = width / (x_range[1] - x_range[0])
    sy = height / (y_range[1] - y_range[0])
    _particle_raster(np.ascontiguousarray(x, np.float64), np.ascontiguousarray(y, np.float64),
                     np.ascontiguousarray(v, np.float64), float(x_range[0]), float(y_range[0]),
                     sx, sy, width, height, cnt, vcnt, vsum, vsq)
    with np.errstate(invalid="ignore", divide="ignore"):
        mean = vsum / vcnt
        std = np.sqrt(np.maximum(vsq / vcnt - mean * mean, 0))
    shape = (height, width)
    return cnt.reshape(shape), mean.reshape(shape), std.reshape(shape)


def distinct_trajectories(x, y, x_range, y_range, width, height) -> np.ndarray:
    return particle_aggregate(x, y, None, x_range, y_range, width, height)[0]


def aggregate(x, y, v=None, how="trajectories", x_range=None, y_range=None,
              width=800, height=800, antialias=False) -> xr.DataArray:
    """Aggregate (P, K) projected trajectories onto a width × height canvas.

    Returns a DataArray with ascending pixel-centre ``x``/``y`` coordinates
    (datashader's convention); empty pixels are NaN.
    """
    if x_range is None:
        x_range = (np.nanmin(x), np.nanmax(x))
    if y_range is None:
        y_range = (np.nanmin(y), np.nanmax(y))
    if how in NEEDS_VALUE and v is None:
        raise ValueError(f"{how} needs a value array")
    cvs = ds.Canvas(plot_width=width, plot_height=height, x_range=x_range, y_range=y_range)
    lw = 1 if antialias else 0
    if how in PARTICLE:
        cnt, mean, std = particle_aggregate(x, y, v if how != "trajectories" else None,
                                            x_range, y_range, width, height)
        a = {"trajectories": cnt, "pmean": mean, "pstd": std}[how].astype(np.float32)
        if how == "pstd":
            a[cnt < 2] = np.nan  # a spread needs at least two particles
        ref = cvs.points(pd.DataFrame({"x": [np.nan], "y": [np.nan]}), "x", "y")
        agg = xr.DataArray(a, coords=ref.coords, dims=ref.dims)
    elif how == "dwell":
        agg = cvs.points(line_frame(x, y), "x", "y", agg=ds.count()).astype(np.float32)
    elif how == "segments":
        agg = cvs.line(line_frame(x, y), "x", "y", agg=ds.count(), line_width=lw).astype(np.float32)
    elif how == "range":
        s = cvs.line(line_frame(x, y, v), "x", "y",
                     agg=ds.summary(mx=ds.max("v"), mn=ds.min("v")))
        agg = s["mx"] - s["mn"]
    elif how in ("mean", "std", "min", "max"):
        # datashader antialiases only count / sum / mean-like reductions
        agg = cvs.line(line_frame(x, y, v), "x", "y", agg=getattr(ds, how)("v"),
                       line_width=lw if how == "mean" else 0)
    else:
        raise ValueError(f"unknown reduction {how!r}; one of {REDUCTIONS}")
    if how in COUNTS:
        agg = agg.where(agg > 0)
    return agg.rename(how)
