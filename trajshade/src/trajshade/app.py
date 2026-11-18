"""Interactive Panel / HoloViews front end: re-aggregates on every pan, zoom or widget change."""
from __future__ import annotations

import time

import colorcet as cc
import holoviews as hv
import numpy as np
import panel as pn
from holoviews.streams import RangeXY

from .aggregate import COUNTS, REDUCTIONS, aggregate
from .geo import graticule, make_projection, project
from .store import Store

CMAPS = {"fire": cc.fire, "bmy": cc.bmy, "kbc": cc.kbc, "bgy": cc.bgy, "gray": cc.gray,
         "rainbow": cc.rainbow4, "coolwarm": cc.coolwarm, "viridis": cc.CET_L20}
HELP = {
    "trajectories": "distinct particles whose path touches each pixel",
    "segments": "line segments crossing each pixel (loops count repeatedly)",
    "dwell": "records spent in each pixel (vertices only, time-weighted)",
    "mean": "segment-weighted mean of the variable (loopers dominate)",
    "std": "segment-weighted spread (std) of the variable",
    "min": "minimum of the variable along the paths (min record = first arrival)",
    "max": "maximum of the variable along the paths",
    "range": "max − min of the variable along the paths",
    "pmean": "particle-weighted mean: each particle's in-pixel mean, averaged over particles",
    "pstd": "particle-weighted spread: std across particles of their in-pixel means (≥ 2 particles)",
}


class Selection:
    """Projected window arrays for the current (records, stride, variable) choice, cached."""

    def __init__(self, store: Store, proj: str = "auto"):
        self.store = store
        lon, lat, _ = store.window(0, 1, max(1, store.P // 20000), alive_only=False)
        self.proj = make_projection(proj, lon, lat)
        self._key, self._val = None, None

    def get(self, k0, k1, stride, var, alive_only):
        key = (k0, k1, stride, var, alive_only)
        if key != self._key:
            lon, lat, v = self.store.window(k0, k1, stride, var, alive_only)
            x, y = project(self.proj, lon, lat)
            self._key, self._val = key, (x, y, v)
        return self._val


def default_stride(P: int, T: int, target: float = 5e6) -> int:
    return max(1, int(np.ceil(P * T / target)))


def extent(x, y, pad=0.03):
    ok = np.isfinite(x) & np.isfinite(y)
    if not ok.any():
        return (-1, 1), (-1, 1)
    x0, x1 = np.percentile(x[ok], [0.1, 99.9])
    y0, y1 = np.percentile(y[ok], [0.1, 99.9])
    dx, dy = (x1 - x0) or 1, (y1 - y0) or 1
    return (x0 - pad * dx, x1 + pad * dx), (y0 - pad * dy, y1 + pad * dy)


def build(path: str, proj: str = "auto", size: int = 800, stride: int | None = None,
          scale: int | None = None):
    """``scale``: canvas pixels per CSS pixel; None follows the browser's device pixel ratio.

    The canvas must map 1:1 onto device pixels: the image is drawn with
    nearest-neighbour resampling, which drops 1-px lines when it downsamples.
    """
    hv.extension("bokeh")
    pn.extension()
    store = Store(path)
    sel = Selection(store, proj)
    P, T = store.P, store.T
    stride0 = stride or default_stride(P, T)

    # full extent from a strided sample of the whole run
    lon, lat, _ = store.window(0, T, max(stride0, P // 5000), alive_only=False)
    gx, gy = project(sel.proj, lon, lat)
    x_full, y_full = extent(gx, gy)

    w_how = pn.widgets.Select(name="reduction", options=list(REDUCTIONS), value="trajectories")
    w_var = pn.widgets.Select(name="variable (value reductions)", options=store.variables,
                              value="record")
    w_end = pn.widgets.Player(name="end record", start=1, end=T, value=T, step=1, interval=250,
                              loop_policy="loop", width=300)
    w_span = pn.widgets.IntSlider(name="span (records before end)", start=1, end=T, value=T)
    w_stride = pn.widgets.IntSlider(name="particle stride", start=1, end=max(2, min(P, 200)),
                                    value=stride0)
    w_norm = pn.widgets.RadioButtonGroup(name="norm", options=["eq_hist", "log", "linear"],
                                         value="eq_hist")
    w_cmap = pn.widgets.Select(name="colormap", options=list(CMAPS), value="fire")
    w_aa = pn.widgets.Checkbox(name="antialiased lines (slower)", value=False)
    w_alive = pn.widgets.Checkbox(name="alive only", value=True, disabled=not store.has_status)
    w_grat = pn.widgets.Checkbox(name="graticule", value=True)
    w_scale = pn.widgets.IntInput(value=scale or 1, visible=False)
    if scale is None and pn.state.browser_info is not None:
        def _dpr(*_):
            r = pn.state.browser_info.device_pixel_ratio
            if r:
                w_scale.value = max(1, round(r))
        pn.state.browser_info.param.watch(_dpr, "device_pixel_ratio")
        _dpr()
    status = pn.pane.Markdown("", sizing_mode="stretch_width")

    def _var_cmap(event):
        w_cmap.value = "fire" if event.new in COUNTS else "rainbow"
        w_norm.value = "eq_hist" if event.new in COUNTS else "linear"
    w_how.param.watch(_var_cmap, "value")

    def image(x_range, y_range, end, span, stride, how, var, antialias, alive_only, scale):
        x_range = x_range or x_full
        y_range = y_range or y_full
        k1 = int(end)
        k0 = max(0, k1 - int(span))
        t0 = time.perf_counter()
        x, y, v = sel.get(k0, k1, int(stride), var if how not in COUNTS else None, alive_only)
        t1 = time.perf_counter()
        agg = aggregate(x, y, v, how, x_range, y_range, size * scale, size * scale, antialias)
        t2 = time.perf_counter()
        n_part = int(np.isfinite(x).any(axis=1).sum())
        label = how if how in COUNTS else f"{how}({var})"
        status.object = (
            f"**{store.label(k0)} → {store.label(k1 - 1)}**  \n"
            f"{label}: {HELP[how]}  \n"
            f"{n_part:,} particles × {k1 - k0} records = {x.size:,} vertices · "
            f"{size * scale}² px · read {1e3 * (t1 - t0):.0f} ms · aggregate {1e3 * (t2 - t1):.0f} ms")
        return hv.Image(agg, kdims=["x", "y"], vdims=[hv.Dimension(how, label=label)])

    bound = pn.bind(image, end=w_end, span=w_span, stride=w_stride, how=w_how, var=w_var,
                    antialias=w_aa, alive_only=w_alive, scale=w_scale)
    dmap = hv.DynamicMap(bound, streams=[RangeXY()])
    dmap = dmap.apply.opts(cmap=pn.bind(CMAPS.get, w_cmap), cnorm=w_norm)
    dmap = dmap.opts(
        hv.opts.Image(frame_width=size, frame_height=size, data_aspect=1, colorbar=True,
                      tools=["hover"], bgcolor="#111", clipping_colors={"NaN": "#111"},
                      xaxis=None if not sel.proj.wraps else "bottom",
                      yaxis=None if not sel.proj.wraps else "left",
                      xlim=x_full, ylim=y_full, title=f"{store.path} · {sel.proj.name}"))

    lat_ok = lat[np.isfinite(lat)]
    grat = hv.Path([{"x": a, "y": b} for a, b in graticule(
        sel.proj, lat_min=float(lat_ok.min()) if lat_ok.size else -90,
        lat_max=float(lat_ok.max()) if lat_ok.size else 90)]).opts(
        color="#888", line_width=0.6, alpha=0.5)
    plot = pn.bind(lambda g: dmap * grat if g else dmap, w_grat)

    side = pn.Column(
        f"### trajshade\n`{store.kind}` store · {P:,} particles × {T} records",
        w_how, w_var, pn.pane.Markdown("**records**"), w_end, w_span, w_stride,
        pn.pane.Markdown("**display**"), w_norm, w_cmap, w_aa, w_alive, w_grat, status,
        width=340)
    return pn.Row(side, pn.panel(plot))


def static_png(path: str, out: str, how="trajectories", var="record", proj="auto",
               size=800, stride=None, k0=0, k1=None, norm="eq_hist", cmap=None,
               alive_only=True):
    """Render one aggregation to a PNG without a server."""
    import datashader.transfer_functions as tf

    store = Store(path)
    sel = Selection(store, proj)
    k1 = k1 or store.T
    x, y, v = sel.get(k0, k1, stride or default_stride(store.P, k1 - k0),
                      var if how not in COUNTS else None, alive_only)
    xr_, yr_ = extent(x, y)
    agg = aggregate(x, y, v, how, xr_, yr_, size, size)
    cm = CMAPS[cmap or ("fire" if how in COUNTS else "rainbow")]
    img = tf.set_background(tf.shade(agg, cmap=cm, how=norm), "black")
    img.to_pil().save(out)
    return agg
