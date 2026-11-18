"""Projections and spherical helpers."""
from __future__ import annotations

import numpy as np

R_EARTH_KM = 6371.0


class PolarStereo:
    """Polar stereographic on the unit sphere; ``lon0`` points down the screen."""

    def __init__(self, north: bool = True, lon0: float = 0.0):
        self.s, self.lon0 = (1 if north else -1), lon0
        self.name = "north polar stereo" if north else "south polar stereo"

    def fwd(self, lon, lat):
        colat = np.radians(90 - self.s * np.asarray(lat, float))
        r = 2 * np.tan(colat / 2)
        a = np.radians(np.asarray(lon, float) - self.lon0)
        return r * np.sin(a), -self.s * r * np.cos(a)

    def inv(self, x, y):
        r = np.hypot(x, y)
        lat = self.s * (90 - np.degrees(2 * np.arctan(r / 2)))
        lon = self.lon0 + np.degrees(np.arctan2(x, -self.s * y))
        return (lon + 180) % 360 - 180, lat


class Plate:
    """Plate carrée centred on ``lon0`` (dateline-safe for regional runs)."""

    def __init__(self, lon0: float = 0.0):
        self.lon0, self.name = lon0, "lon/lat"

    def fwd(self, lon, lat):
        return (np.asarray(lon, float) - self.lon0 + 180) % 360 - 180, np.asarray(lat, float)

    def inv(self, x, y):
        return (x + self.lon0 + 180) % 360 - 180, y


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
    lon0 = float(np.degrees(np.arctan2(np.sin(rl).mean(), np.cos(rl).mean())))
    return Plate(lon0)


def xyz(lon, lat):
    lo, la = np.radians(lon), np.radians(lat)
    return np.stack([np.cos(la) * np.cos(lo), np.cos(la) * np.sin(lo), np.sin(la)], -1)


def haversine_km(lon1, lat1, lon2, lat2):
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dl, dp = np.radians(np.asarray(lon2) - lon1), p2 - p1
    a = np.sin(dp / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2) ** 2
    return 2 * R_EARTH_KM * np.arcsin(np.sqrt(np.clip(a, 0, 1)))
