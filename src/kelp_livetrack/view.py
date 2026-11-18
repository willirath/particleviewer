"""Map view: projection, data extent, zoom/pan, pixel grids and cached backgrounds."""
from __future__ import annotations

import math
from collections import OrderedDict

import numpy as np

from .mesh import Mesh


class View:
    def __init__(self, proj, mesh: Mesh | None = None):
        self.proj = proj
        self.mesh = mesh or Mesh(None)
        self.data_extent: tuple[float, float, float, float] | None = None
        self.zoom = 1.0
        self.cx = self.cy = 0.0  # pan offset, in units of the data extent
        self._bg: OrderedDict = OrderedDict()
        self.section_range: tuple[float, float, float] | None = None  # lat0, lat1, zmax

    def grow_section(self, lat, z) -> tuple[float, float, float]:
        """Section axes that only ever widen, so the bulk does not appear to move
        when a few particles extend the range."""
        lat, z = np.asarray(lat, float), np.asarray(z, float)
        lat, z = lat[np.isfinite(lat)], z[np.isfinite(z)]
        if len(lat) and len(z):
            r = (float(lat.min()), float(lat.max()), max(float(z.max()), 1.0))
            if self.section_range is not None:
                o = self.section_range
                r = (min(r[0], o[0]), max(r[1], o[1]), max(r[2], o[2]))
            self.section_range = r
        return self.section_range or (0.0, 1.0, 1.0)

    # -- extent -------------------------------------------------------------
    def fit(self, lon, lat, pad: float = 0.08) -> bool:
        """Grow the data extent to include all finite positions. True if it changed."""
        x, y = self.proj.fwd(lon, lat)
        ok = np.isfinite(x) & np.isfinite(y)
        if not ok.any():
            return False
        x0, x1, y0, y1 = x[ok].min(), x[ok].max(), y[ok].min(), y[ok].max()
        if self.data_extent is not None:
            e = self.data_extent
            if x0 >= e[0] and x1 <= e[1] and y0 >= e[2] and y1 <= e[3]:
                return False
            x0, x1, y0, y1 = min(x0, e[0]), max(x1, e[1]), min(y0, e[2]), max(y1, e[3])
        px, py = max(x1 - x0, 1e-6) * pad, max(y1 - y0, 1e-6) * pad
        self.data_extent = (x0 - px, x1 + px, y0 - py, y1 + py)
        return True

    @property
    def extent(self):
        x0, x1, y0, y1 = self.data_extent
        w, h = (x1 - x0), (y1 - y0)
        cx, cy = (x0 + x1) / 2 + self.cx * w, (y0 + y1) / 2 + self.cy * h
        hw, hh = w / 2 / self.zoom, h / 2 / self.zoom
        return (cx - hw, cx + hw, cy - hh, cy + hh)

    def zoom_by(self, f: float):
        self.zoom = float(np.clip(self.zoom * f, 1.0, 64.0))
        if self.zoom == 1.0:
            self.cx = self.cy = 0.0

    def pan(self, dx: float, dy: float):
        """Pan by fractions of the visible width/height (dy > 0 = north/up)."""
        self.cx += dx / self.zoom
        self.cy += dy / self.zoom
        lim = 0.5
        self.cx, self.cy = float(np.clip(self.cx, -lim, lim)), float(np.clip(self.cy, -lim, lim))

    def reset(self):
        self.zoom, self.cx, self.cy = 1.0, 0.0, 0.0

    # -- pixel grids ----------------------------------------------------------
    def grid(self, cols: int, rows: int) -> tuple[int, int]:
        """Half-block pixel grid (H, W) with square pixels that fits cols x rows cells."""
        x0, x1, y0, y1 = self.extent
        aspect = (y1 - y0) / (x1 - x0)
        cols, rows = max(cols, 4), max(rows, 2)
        W, H = cols, 2 * rows
        if W * aspect > H:
            W = max(4, int(H / aspect))
        r = max(1, min(rows, math.ceil(W * aspect / 2)))
        return 2 * r, W

    def pixel_lonlat(self, H: int, W: int):
        x0, x1, y0, y1 = self.extent
        xs = x0 + (np.arange(W) + 0.5) * (x1 - x0) / W
        ys = y1 - (np.arange(H) + 0.5) * (y1 - y0) / H
        X, Y = np.meshgrid(xs, ys)
        return self.proj.inv(X, Y)

    def background(self, H: int, W: int) -> np.ndarray:
        """Half-block background raster (H, W, 3), cached per size and extent."""
        key = (H, W, self.extent)
        if key not in self._bg:
            self._bg[key] = self.mesh.raster(*self.pixel_lonlat(H, W))
            while len(self._bg) > 24:
                self._bg.popitem(last=False)
        self._bg.move_to_end(key)
        return self._bg[key]

    def cell_background(self, H: int, W: int) -> np.ndarray:
        """Per-cell background (H/2, W, 3) for braille panels: mean of the half-block pair."""
        hb = self.background(H, W)
        return ((hb[0::2].astype(int) + hb[1::2]) // 2).astype(np.uint8)

    def to_pixel(self, lon, lat, H: int, W: int):
        """Fractional pixel coordinates (px, py) on an H x W grid."""
        x, y = self.proj.fwd(lon, lat)
        x0, x1, y0, y1 = self.extent
        return (x - x0) / (x1 - x0) * W, (y1 - y) / (y1 - y0) * H

    def bin(self, lon, lat, H: int, W: int) -> np.ndarray:
        """Flat pixel index for each position, -1 outside the grid / non-finite."""
        px, py = self.to_pixel(lon, lat, H, W)
        ok = np.isfinite(px) & np.isfinite(py) & (px >= 0) & (px < W) & (py >= 0) & (py < H)
        idx = np.full(np.shape(px), -1, np.int64)
        idx[ok] = py[ok].astype(np.int64) * W + px[ok].astype(np.int64)
        return idx
