"""Mesh background: land / bathymetry by nearest-neighbour lookup (as in ushow).

Everything is a point cloud with lon/lat. A display pixel takes the wet flag and
bottom depth of the nearest mesh point; pixels farther than a few mesh spacings
from any point count as land (point-cloud meshes) or no data.
"""
from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree

from .geo import xyz
from .render import blend

NO_DATA = (24, 30, 40)
LAND = (74, 70, 64)
OCEAN_SHALLOW = (44, 66, 86)
OCEAN_DEEP = (14, 22, 36)


def _open(path: str):
    import xarray as xr
    if path.rstrip("/").endswith(".zarr"):
        return xr.open_zarr(path)
    return xr.open_dataset(path)


def _nemo(ds):
    lon = np.squeeze(ds["glamt"].values)
    lat = np.squeeze(ds["gphit"].values)
    if "tmaskutil" in ds:
        wet = np.squeeze(ds["tmaskutil"].values) > 0
    elif "tmask" in ds:
        tm = ds["tmask"].values
        wet = tm.reshape(-1, *tm.shape[-2:])[0] > 0  # first (surface) level
    elif "top_level" in ds:  # domain_cfg
        wet = np.squeeze(ds["top_level"].values) > 0
    else:
        wet = np.ones(lon.shape, bool)
    depth = None
    if "mbathy" in ds and "gdepw_1d" in ds:
        kb = np.squeeze(ds["mbathy"].values).astype(int)
        gw = np.squeeze(ds["gdepw_1d"].values)
        depth = gw[np.clip(kb, 0, len(gw) - 1)]
    elif "bathy_metry" in ds:
        depth = np.squeeze(ds["bathy_metry"].values)
    elif "bottom_level" in ds and "e3t_1d" in ds:
        kb = np.squeeze(ds["bottom_level"].values).astype(int)
        gw = np.concatenate([[0.0], np.cumsum(np.squeeze(ds["e3t_1d"].values))])
        depth = gw[np.clip(kb, 0, len(gw) - 1)]
    return lon, lat, wet, depth


def _points(ds):
    if "lon" in ds and "lat" in ds:
        lon, lat = ds["lon"].values, ds["lat"].values
    elif "nodes" in ds:  # FESOM2 mesh diag variant: (2, nod2)
        nodes = np.asarray(ds["nodes"].values, float)
        if nodes.shape[0] != 2:
            nodes = nodes.T
        lon, lat = nodes
        if np.nanmax(np.abs(lon)) <= 2 * np.pi + 1e-6 and np.nanmax(np.abs(lat)) <= np.pi / 2 + 1e-6:
            lon, lat = np.degrees(lon), np.degrees(lat)
    else:
        for lo_n, la_n in (("longitude", "latitude"), ("nav_lon", "nav_lat"),
                           ("nod2_lon", "nod2_lat")):
            if lo_n in ds and la_n in ds:
                lon, lat = ds[lo_n].values, ds[la_n].values
                break
        else:
            raise ValueError("no lon/lat point coordinates found")
    lon, lat = np.asarray(lon, float), np.asarray(lat, float)
    if lon.ndim == 1 and lat.ndim == 1 and lon.shape != lat.shape:  # regular grid axes
        lon, lat = np.meshgrid(lon, lat)
    depth = next((ds[n].values for n in ("depth", "nod2_depth", "bathymetry", "topo")
                  if n in ds and ds[n].shape == lon.shape), None)
    return lon, lat, np.ones(lon.shape, bool), depth


class Mesh:
    """Mesh points with a wet flag and optional bottom depth; ``Mesh(None)`` = no background."""

    def __init__(self, path: str | None = None, *, lon=None, lat=None, wet=None, depth=None):
        self.tree = None
        self.path = path
        if path is not None:
            ds = _open(path)
            lon, lat, wet, depth = _nemo(ds) if ("glamt" in ds and "gphit" in ds) else _points(ds)
        if lon is None:
            return
        lon, lat = np.asarray(lon, float).ravel(), np.asarray(lat, float).ravel()
        wet = np.ones(lon.shape, bool) if wet is None else np.asarray(wet, bool).ravel()
        ok = np.isfinite(lon) & np.isfinite(lat)
        self.wet = wet[ok]
        self.depth = None if depth is None else np.abs(np.asarray(depth, float).ravel()[ok])
        pts = xyz(lon[ok], lat[ok])
        self.tree = cKDTree(pts)
        n = min(4000, len(pts))
        sample = pts[np.random.default_rng(0).choice(len(pts), n, replace=False)]
        d, _ = self.tree.query(sample, k=min(2, len(pts)))
        spacing = np.median(d[:, -1]) if d.ndim == 2 else 0.01
        self.max_dist = 2.5 * max(spacing, 1e-6)

    @property
    def describe(self) -> str:
        if self.tree is None:
            return "no mesh"
        return f"{self.tree.n} mesh points"

    def raster(self, lon: np.ndarray, lat: np.ndarray) -> np.ndarray:
        """Background RGB (H, W, 3) for pixel-centre lon/lat arrays (H, W)."""
        H, W = lon.shape
        rgb = np.empty((H, W, 3), np.uint8)
        rgb[:] = NO_DATA
        if self.tree is None:
            return rgb
        ok = np.isfinite(lon) & np.isfinite(lat)
        d = np.full(H * W, np.inf)
        i = np.zeros(H * W, int)
        q = ok.ravel()
        if q.any():
            d[q], i[q] = self.tree.query(xyz(lon.ravel()[q], lat.ravel()[q]))
        covered = (d < self.max_dist).reshape(H, W)
        wet = self.wet[i].reshape(H, W) & covered
        # a point-cloud mesh (all wet) has land wherever there are no points
        land = ~wet & (covered | (bool(self.wet.all()) & ok))
        rgb[land] = LAND
        if self.depth is not None:
            v = np.sqrt(np.clip(self.depth[i].reshape(H, W) / 5000.0, 0, 1))
            rgb[wet] = blend(OCEAN_SHALLOW, OCEAN_DEEP, v[wet][:, None])
        else:
            rgb[wet] = (24, 36, 52)
        return rgb
