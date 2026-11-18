"""Native matplotlib window that re-aggregates at device resolution on every zoom / pan.

Keys: r / R next / previous reduction · v / V next / previous variable ·
n cycle norm (eq_hist, log, linear) · a toggle antialiasing · , . step the
record window back / forward by its span · 0 reset view.
"""
from __future__ import annotations

import time

import colorcet as cc
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np

from .aggregate import COUNTS, REDUCTIONS, aggregate
from .app import Selection, default_stride, extent
from .geo import graticule
from .store import Store

NORMS = ("eq_hist", "log", "linear")


def eq_hist(a: np.ndarray, nbins: int = 4096) -> np.ndarray:
    """Histogram-equalised copy of ``a`` in [0, 1] (NaN stays NaN)."""
    ok = np.isfinite(a)
    out = np.full(a.shape, np.nan)
    if not ok.any():
        return out
    v = a[ok]
    hist, edges = np.histogram(v, bins=nbins)
    cdf = np.cumsum(hist).astype(float)
    cdf /= cdf[-1]
    out[ok] = np.interp(v, 0.5 * (edges[1:] + edges[:-1]), cdf)
    return out


class Viewer:
    def __init__(self, path, how="trajectories", var="record", proj="auto", stride=None,
                 k0=0, k1=None, norm="eq_hist", alive_only=True):
        self.store = Store(path)
        self.sel = Selection(self.store, proj)
        s = self.store
        self.k1 = k1 or s.T
        self.k0 = k0
        self.stride = stride or default_stride(s.P, self.k1 - self.k0)
        self.how, self.var, self.norm = how, var, norm
        self.antialias, self.alive_only = False, alive_only

        x, y, _ = self.data()
        self.x_full, self.y_full = extent(x, y)
        self.fig, self.ax = plt.subplots(figsize=(9, 8.5), layout="constrained")
        self.fig.set_facecolor("#111")
        self.ax.set_facecolor("#111")
        self.ax.set_aspect("equal")
        if not self.sel.proj.wraps:
            self.ax.set_axis_off()
        for gx, gy in graticule(self.sel.proj, lat_min=self.y_full[0] if self.sel.proj.wraps else 0):
            self.ax.plot(gx, gy, color="#888", lw=0.4, alpha=0.4, zorder=3)
        self.im = self.ax.imshow(np.full((2, 2), np.nan), origin="lower", interpolation="nearest",
                                 extent=(*self.x_full, *self.y_full), zorder=2)
        self.cb = self.fig.colorbar(self.im, ax=self.ax, shrink=0.8)
        self.cb.ax.tick_params(colors="#ccc")
        self.ax.set_xlim(self.x_full)
        self.ax.set_ylim(self.y_full)
        # lay out once, then freeze: a layout engine moving the axes at draw time
        # would leave the image a few pixels off its box (and drop 1-px lines)
        self.fig.canvas.draw()
        self.fig.set_layout_engine("none")
        self._busy = False
        self.ax.callbacks.connect("xlim_changed", lambda ax: self.redraw())
        self.ax.callbacks.connect("ylim_changed", lambda ax: self.redraw())
        self.fig.canvas.mpl_connect("key_press_event", self.on_key)
        self.fig.canvas.mpl_connect("resize_event", lambda e: self.redraw())
        self.redraw()

    def data(self):
        v = None if self.how in COUNTS else self.var
        return self.sel.get(self.k0, self.k1, self.stride, v, self.alive_only)

    def redraw(self):
        if self._busy:
            return
        self._busy = True
        try:
            # resolve the equal-aspect box first, or the image is computed for a
            # stale box and nearest-neighbour resampling drops 1-px lines
            self.ax.apply_aspect()
            xr_, yr_ = self.ax.get_xlim(), self.ax.get_ylim()
            bb = self.ax.get_window_extent()  # device pixels (HiDPI aware)
            W, H = max(round(bb.width), 2), max(round(bb.height), 2)
            t0 = time.perf_counter()
            x, y, v = self.data()
            t1 = time.perf_counter()
            agg = aggregate(x, y, v, self.how, xr_, yr_, W, H, self.antialias)
            t2 = time.perf_counter()
            a = agg.values.astype(float)
            cmap = (cc.cm.fire if self.how in COUNTS else cc.cm.rainbow4).copy()
            cmap.set_bad((0, 0, 0, 0))
            if self.norm == "eq_hist":
                img, norm = eq_hist(a), mcolors.Normalize(0, 1)
            elif self.norm == "log":
                pos = a[np.isfinite(a) & (a > 0)]
                img = np.where(a > 0, a, np.nan)
                norm = mcolors.LogNorm(*(np.percentile(pos, [0, 100]) if pos.size else (1, 10)))
            else:
                img = a
                fin = a[np.isfinite(a)]
                norm = mcolors.Normalize(*(np.percentile(fin, [1, 99]) if fin.size else (0, 1)))
            self.im.set_data(img)
            self.im.set_extent((*xr_, *yr_))
            self.im.set_cmap(cmap)
            self.im.set_norm(norm)
            label = self.how if self.how in COUNTS else f"{self.how}({self.var})"
            self.cb.set_label(label + (" (eq_hist rank)" if self.norm == "eq_hist" else ""),
                              color="#ccc")
            s = self.store
            self.ax.set_title(
                f"{label} · {s.label(self.k0)} → {s.label(self.k1 - 1)}\n"
                f"stride {self.stride} · {x.size:,} vertices · {W}×{H} px · "
                f"read {1e3 * (t1 - t0):.0f} ms · agg {1e3 * (t2 - t1):.0f} ms",
                color="#ccc", fontsize=9)
            self.fig.canvas.draw_idle()
        finally:
            self._busy = False

    def on_key(self, e):
        k = e.key
        cyc = lambda seq, cur, d: seq[(seq.index(cur) + d) % len(seq)]
        if k in ("r", "R"):
            self.how = cyc(REDUCTIONS, self.how, 1 if k == "r" else -1)
        elif k in ("v", "V"):
            self.var = cyc(self.store.variables, self.var, 1 if k == "v" else -1)
        elif k == "n":
            self.norm = cyc(NORMS, self.norm, 1)
        elif k == "a":
            self.antialias = not self.antialias
        elif k in (",", "."):
            span = self.k1 - self.k0
            d = span if k == "." else -span
            self.k0 = min(max(self.k0 + d, 0), self.store.T - span)
            self.k1 = self.k0 + span
        elif k == "0":
            self._busy = True
            self.ax.set_xlim(self.x_full)
            self.ax.set_ylim(self.y_full)
            self._busy = False
        else:
            return
        self.redraw()


def run(path, **kw):
    Viewer(path, **kw)
    plt.show()
