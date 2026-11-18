"""Projections (shared shape with kelp-live) and a graticule."""
from __future__ import annotations

import numpy as np


class PolarStereo:
    """Polar stereographic on the unit sphere; ``lon0`` points down the screen."""

    def __init__(self, north: bool = True, lon0: float = 0.0):
        self.s, self.lon0 = (1 if north else -1), lon0
        self.name = "north polar stereo" if north else "south polar stereo"
        self.wraps = False

    def fwd(self, lon, lat):
        colat = np.radians(90 - self.s * np.asarray(lat, float))
        r = 2 * np.tan(colat / 2)
        a = np.radians(np.asarray(lon, float) - self.lon0)
        return r * np.sin(a), -self.s * r * np.cos(a)


class Plate:
    """Plate carrée centred on ``lon0``; segments crossing the seam are cut."""

    def __init__(self, lon0: float = 0.0):
        self.lon0, self.name, self.wraps = lon0, "lon/lat", True

    def fwd(self, lon, lat):
        return (np.asarray(lon, float) - self.lon0 + 180) % 360 - 180, np.asarray(lat, float)


def auto_projection(lon, lat):
    """Polar stereo for runs confined to high latitudes, plate carrée otherwise."""
    lon, lat = np.asarray(lon, float), np.asarray(lat, float)
    ok = np.isfinite(lon) & np.isfinite(lat)
    if not ok.any():
        return Plate()
    lon, lat = lon[ok], lat[ok]
    if lat.min() > 25:
        return PolarStereo(True)
    if lat.max() < -25:
        return PolarStereo(False)
    rl = np.radians(lon)
    return Plate(float(np.degrees(np.arctan2(np.sin(rl).mean(), np.cos(rl).mean()))))


def make_projection(name: str, lon, lat):
    return {"auto": lambda: auto_projection(lon, lat), "plate": Plate,
            "north": lambda: PolarStereo(True), "south": lambda: PolarStereo(False)}[name]()


def project(proj, lon, lat):
    """Projected (x, y); for wrapping projections the vertex after a seam jump is NaN."""
    x, y = proj.fwd(lon, lat)
    if proj.wraps and x.shape[-1] > 1:
        jump = np.zeros(x.shape, bool)
        jump[:, 1:] = np.abs(np.diff(x, axis=1)) > 180
        x[jump] = np.nan
        y[jump] = np.nan
    return x, y


def graticule(proj, dlat=10, dlon=30, lat_min=-90, lat_max=90):
    """List of (x, y) polylines of parallels and meridians."""
    lines = []
    lons = np.linspace(-180, 180, 361)
    for la in np.arange(-80, 81, dlat):
        if lat_min - dlat <= la <= lat_max + dlat:
            lines.append(proj.fwd(lons, np.full_like(lons, la)))
    lats = np.linspace(max(lat_min - dlat, -89), min(lat_max + dlat, 89), 181)
    for lo in np.arange(-180, 180, dlon):
        lines.append(proj.fwd(np.full_like(lats, lo), lats))
    out = []
    for x, y in lines:
        if proj.wraps:  # split at the seam
            cut = np.flatnonzero(np.abs(np.diff(x)) > 180) + 1
            out += [(a, b) for a, b in zip(np.split(x, cut), np.split(y, cut)) if len(a) > 1]
        else:
            out.append((x, y))
    return out
