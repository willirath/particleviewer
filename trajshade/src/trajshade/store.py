"""Read (particle, record) windows from a trajectory Zarr store.

Works for kelp output format 2 (``lon``/``lat``/``time`` with dims trajectory × time,
1-D ``time``) and Parcels-style stores (dims trajectory × obs, 2-D ``time``).
Windows are assembled from per-chunk blocks that are cached after the particle
stride is applied, so moving a time window or playing it back re-decodes little.
"""
from __future__ import annotations

from collections import OrderedDict

import numpy as np
import zarr

SKIP = {"lon", "lat", "status", "failure_reason", "trajectory", "time", "obs"}
DERIVED = ("record", "age_days")


class Store:
    def __init__(self, path: str, cache_bytes: float = 2e9):
        self.path = path
        self.g = zarr.open_group(path, mode="r")
        self.attrs = dict(self.g.attrs)
        if "lon" not in self.g or "lat" not in self.g:
            raise ValueError(f"{path}: no lon/lat arrays")
        self.P, self.T = self.g["lon"].shape
        self.kind = "kelp" if "kelp_output_format" in self.attrs else "parcels"
        arrays = dict(self.g.arrays())
        self.fields = sorted(n for n, a in arrays.items()
                             if a.shape == (self.P, self.T) and n not in SKIP
                             and a.dtype.kind in "fi" and not n.startswith("restart_"))
        self.has_status = "status" in arrays and arrays["status"].shape == (self.P, self.T)
        self.time = None
        self.epoch = None
        if "time" in arrays and arrays["time"].ndim == 1:
            self.time = np.asarray(arrays["time"][:], float)
            units = str(dict(arrays["time"].attrs).get("units", ""))
            if " since " in units:
                try:
                    self.epoch = np.datetime64(units.split(" since ")[1].strip().replace(" ", "T"))
                except ValueError:
                    pass
        self.release_time = (np.asarray(arrays["release_time"][:], float)
                             if "release_time" in arrays and self.time is not None else None)
        self._cache: OrderedDict[tuple, np.ndarray] = OrderedDict()
        self._cache_bytes = 0.0
        self.cache_limit = cache_bytes

    @property
    def variables(self) -> list[str]:
        """Fields usable as the value of a mean / min / max aggregation."""
        out = ["record"]
        if self.release_time is not None or "age_sec" in self.fields:
            out.append("age_days")
        return out + [f for f in self.fields if f != "age_sec"]

    def label(self, k: int) -> str:
        """Human label for record index ``k`` (0-based)."""
        if self.time is None:
            return f"obs {k}"
        t = self.time[min(k, self.T - 1)]
        if not np.isfinite(t):
            return f"record {k}"
        if self.epoch is None:
            return f"record {k} (t = {t / 86400:.2f} d)"
        return f"record {k} · " + str(self.epoch + np.timedelta64(int(t), "s"))[:16].replace("T", " ")

    # -- raw windows ---------------------------------------------------------------
    def _block(self, name: str, c: int, stride: int) -> np.ndarray:
        key = (name, c, stride)
        if key in self._cache:
            self._cache.move_to_end(key)
            return self._cache[key]
        a = self.g[name]
        ct = a.chunks[1]
        blk = np.asarray(a[:, c * ct:(c + 1) * ct])[::stride]
        if blk.dtype.kind == "f" and name not in ("lon", "lat"):
            blk = blk.astype(np.float32)
        self._cache[key] = blk
        self._cache_bytes += blk.nbytes
        while self._cache_bytes > self.cache_limit and len(self._cache) > 1:
            _, old = self._cache.popitem(last=False)
            self._cache_bytes -= old.nbytes
        return blk

    def raw(self, name: str, k0: int, k1: int, stride: int = 1) -> np.ndarray:
        """``name`` for every ``stride``-th particle and records [k0, k1)."""
        ct = self.g[name].chunks[1]
        parts = []
        for c in range(k0 // ct, (k1 - 1) // ct + 1):
            b = self._block(name, c, stride)
            lo, hi = max(k0 - c * ct, 0), min(k1 - c * ct, ct)
            parts.append(b[:, lo:hi])
        return np.concatenate(parts, axis=1) if len(parts) > 1 else parts[0]

    def window(self, k0: int, k1: int, stride: int = 1, var: str | None = None,
               alive_only: bool = True):
        """(lon, lat, value) for records [k0, k1); invalid vertices are NaN."""
        lon = self.raw("lon", k0, k1, stride).astype(np.float64)
        lat = self.raw("lat", k0, k1, stride).astype(np.float64)
        bad = ~(np.isfinite(lon) & np.isfinite(lat))
        if self.has_status:
            st = self.raw("status", k0, k1, stride)
            # 0 = not yet written, 4 = not released; 2 / 3 = beached / outside (frozen)
            bad |= (st != 1) if alive_only else ((st == 0) | (st == 4))
        lon[bad] = np.nan
        lat[bad] = np.nan
        val = None if var is None else self.value(var, k0, k1, stride)
        return lon, lat, val

    def value(self, var: str, k0: int, k1: int, stride: int = 1) -> np.ndarray:
        n = len(range(0, self.P, stride))
        if var == "record":
            return np.broadcast_to(np.arange(k0, k1, dtype=np.float32), (n, k1 - k0))
        if var == "age_days":
            if "age_sec" in self.fields:
                return self.raw("age_sec", k0, k1, stride) / np.float32(86400)
            rel = self.release_time[::stride, None]
            return ((self.time[None, k0:k1] - rel) / 86400).astype(np.float32)
        return self.raw(var, k0, k1, stride)
