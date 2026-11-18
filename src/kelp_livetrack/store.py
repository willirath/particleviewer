"""Incremental reading of a growing kelp Zarr trajectory store (output format 2).

kelp's Zarr writer buffers records and flushes a whole time-chunk at a time,
resizing ``time`` first and then each per-record variable in turn. A reader can
therefore see a ``time`` axis longer than ``lon``, or a resized variable whose
newest chunk is not written yet (it reads back as the fill value). A record
counts as complete only when it is inside every variable we read and its
``status`` column is not all fill (0).
"""
from __future__ import annotations

import threading
import time as _time
from collections import OrderedDict, deque
from dataclasses import dataclass, field

import numpy as np
import zarr

from .geo import haversine_km

STATUS = {1: "alive", 2: "beached", 3: "outside", 4: "not_released"}
RESERVED = {"lon", "lat", "z", "status", "failure_reason", "vel_u", "vel_v", "vel_w",
            "time", "trajectory"}
CORE = ("lon", "lat", "z", "status")


class Feed:
    """Thin, stateless-ish access to the store; every call sees the latest metadata."""

    def __init__(self, path: str):
        self.path = path
        g = self.group()
        if "lon" not in g or "time" not in g:
            raise ValueError(f"{path}: not a kelp trajectory store (no lon/time)")
        self.attrs = dict(g.attrs)
        self.P = int(g["lon"].shape[0])
        self.release = {k: np.asarray(g[k][:], float)
                        for k in ("release_lon", "release_lat", "release_z") if k in g}
        self.epoch = None
        units = str(dict(g["time"].attrs).get("units", ""))
        if " since " in units:
            try:
                self.epoch = np.datetime64(units.split(" since ")[1].strip().replace(" ", "T"))
            except ValueError:
                pass
        self.refresh(g)

    def group(self):
        # a fresh handle on every call: picks up resizes and lazily created arrays
        return zarr.open_group(self.path, mode="r")

    def refresh(self, g=None):
        """Re-scan the variables (kelp creates sampled / failure / velocity vars lazily)."""
        g = g or self.group()
        arrays = dict(g.arrays())
        self.vars = set(arrays)
        self.sampled = sorted(n for n, a in arrays.items()
                              if a.ndim == 2 and n not in RESERVED and not n.startswith("restart_"))
        self.units = {n: str(dict(arrays[n].attrs).get("units", "")) for n in self.sampled}
        self.is_3d = bool(self.attrs.get("mode_3d", "z" in arrays))

    def needed(self) -> list[str]:
        extra = ["failure_reason"] if "failure_reason" in self.vars else []
        return [v for v in CORE if v in self.vars] + extra + self.sampled

    def n_ready(self) -> int:
        g = self.group()
        names = [v for v in ("time", *CORE) if v in g]
        n = min(g[v].shape[-1] for v in names)
        while n > 0 and not np.any(g["status"][:, n - 1]):
            n -= 1
        return n

    def times(self, n: int) -> np.ndarray:
        return np.asarray(self.group()["time"][:n], float)

    def expected_records(self, times: np.ndarray) -> int | None:
        a = self.attrs
        if len(times) >= 2 and "n_steps" in a and "dt" in a and times[1] != times[0]:
            return int(round(a["n_steps"] * a["dt"] / abs(times[1] - times[0])))
        return None

    def label(self, t: float) -> str:
        if not np.isfinite(t):
            return "-"
        if self.epoch is None:
            return f"t = {t / 86400:.2f} d"
        return str(self.epoch + np.timedelta64(int(t), "s"))[:16].replace("T", " ")

    def read(self, k0: int, k1: int, names: list[str]) -> dict[str, np.ndarray]:
        """Records [k0, k1) of the given vars as (P, k1-k0) arrays (float32 for floats)."""
        g = self.group()
        out = {}
        for v in names:
            a = g[v][:, k0:k1]
            out[v] = a.astype(np.float32) if a.dtype.kind == "f" else a
        return out


@dataclass
class Frame:
    """Everything the panels need for one displayed record."""
    feed: Feed
    k: int                       # displayed record (0-based)
    n_ready: int
    time: float
    cols: dict[str, np.ndarray]  # per-particle values at record k (+ 'disp')
    series_k: np.ndarray         # record indices of the time series (<= k)
    series: dict[str, np.ndarray]
    trail: list[tuple[np.ndarray, np.ndarray]]
    n_trail: int
    following: bool = True
    rate_per_min: float | None = None
    last_new: float | None = None
    read_s: float = 0.0
    n_expected: int | None = None
    var: str | None = None       # selected field variable
    extra: dict = field(default_factory=dict)


class State:
    """Incrementally built from the store; safe to update from a worker thread."""

    def __init__(self, feed: Feed, backlog: int = 120, trail_len: int = 120,
                 n_trail: int = 250, cache_bytes: float = 1e9):
        self.feed = feed
        self.backlog, self.trail_len = backlog, trail_len
        self.lock = threading.RLock()
        rng = np.random.default_rng(1)
        self.trail_idx = np.sort(rng.choice(feed.P, min(n_trail, feed.P), replace=False))
        self.rec: list[int] = []                    # ingested record indices
        self.series: dict[str, list[float]] = {}
        self.trail_hist: list[tuple[np.ndarray, np.ndarray]] = []
        self.latest: dict[str, np.ndarray] | None = None
        self.n_ready = 0
        self.times = np.zeros(0)
        self._cache: OrderedDict[int, dict] = OrderedDict()
        self.cache_bytes = cache_bytes
        self.arrivals: deque = deque(maxlen=64)     # (wall time, n_ready)
        self.read_s = 0.0
        self.error: str | None = None

    # -- ingest (worker thread) ----------------------------------------------
    def update(self) -> bool:
        """Ingest records that appeared since the last call. True if anything is new."""
        f = self.feed
        try:
            n = f.n_ready()
            if n <= self.n_ready:
                return False
            f.refresh()
            last = self.rec[-1] if self.rec else -1
            todo = np.arange(last + 1, n)
            if len(todo) > self.backlog:  # catch up on a long backlog with a stride
                todo = np.unique(np.linspace(last + 1, n - 1, self.backlog).astype(int))
            names = f.needed()
            times = f.times(n)
            t0 = _time.perf_counter()
            batches = []
            for run in np.split(todo, np.where(np.diff(todo) != 1)[0] + 1):
                if not len(run):
                    continue
                for a in range(run[0], run[-1] + 1, 64):
                    b = min(run[-1] + 1, a + 64)
                    batches.append((a, b, f.read(a, b, names)))
            dt = _time.perf_counter() - t0
        except Exception as e:  # torn chunk mid-write, var appearing, ...: retry next poll
            self.error = f"{type(e).__name__}: {e}"
            return False
        with self.lock:
            for a, b, block in batches:
                for j in range(b - a):
                    col = {v: block[v][:, j] for v in names}
                    self._ingest(a + j, col)
            self.times, self.n_ready = times, n
            self.read_s += dt
            self.arrivals.append((_time.time(), n))
            self.error = None
        return True

    def _derive(self, col: dict) -> dict:
        rl = self.feed.release
        if "release_lon" in rl and "disp" not in col:
            col["disp"] = haversine_km(rl["release_lon"], rl["release_lat"],
                                       col["lon"], col["lat"]).astype(np.float32)
        return col

    def _ingest(self, k: int, col: dict):
        col = self._derive(col)
        st = col["status"]
        alive = st == 1
        s = self.series

        def put(name, value):
            s.setdefault(name, [np.nan] * len(self.rec)).append(value)

        def mean(v):
            if not alive.any():
                return np.nan
            with np.errstate(all="ignore"):
                return float(np.nanmean(v[alive]))

        for code, name in STATUS.items():
            put(name, float((st == code).sum()))
        if "disp" in col:
            put("mean disp [km]", mean(col["disp"]))
        if self.feed.is_3d and "z" in col:
            put("mean depth [m]", mean(col["z"]))
        for v in self.feed.sampled[:3]:
            if v in col:
                put(f"mean {v} [{self.feed.units.get(v, '')}]", mean(col[v]))
        self.rec.append(k)
        for name, vals in s.items():  # keep all series aligned
            if len(vals) < len(self.rec):
                vals.append(np.nan)
        self.trail_hist.append((col["lon"][self.trail_idx].copy(), col["lat"][self.trail_idx].copy()))
        self.latest = col

    # -- random access (scrubbing) -------------------------------------------
    def has(self, k: int) -> bool:
        with self.lock:
            return (self.rec and k == self.rec[-1]) or k in self._cache

    def record_bytes(self) -> int:
        col = self.latest or {}
        return max(1, sum(v.nbytes for v in col.values()))

    def cache_capacity(self) -> int:
        """Records the cache may hold (at least 3)."""
        return max(3, int(self.cache_bytes // self.record_bytes()))

    def load(self, k: int) -> None:
        """Read one record into the cache (worker thread)."""
        self.load_records([k])

    def load_records(self, ks) -> None:
        """Read the given records into the cache with one slice read per variable."""
        ks = [int(k) for k in ks if not self.has(int(k))]
        if not ks:
            return
        f = self.feed
        ks = np.unique(ks)
        t0 = _time.perf_counter()
        cols = {}
        # one slice read per contiguous run (never a wide slice for strided playback)
        for run in np.split(ks, np.where(np.diff(ks) != 1)[0] + 1):
            k0 = int(run[0])
            block = f.read(k0, int(run[-1]) + 1, f.needed())
            for k in run:
                cols[int(k)] = self._derive({v: a[:, k - k0].copy() for v, a in block.items()})
        with self.lock:
            self.read_s += _time.perf_counter() - t0
            for k, col in cols.items():
                self._cache[k] = col
                self._cache.move_to_end(k)
            cap = self.cache_capacity()
            while len(self._cache) > cap:
                self._cache.popitem(last=False)

    # -- snapshot for rendering (UI thread) ----------------------------------
    def rate(self) -> float | None:
        a = [x for x in self.arrivals if x[0] > _time.time() - 1800]
        if len(a) < 2 or a[-1][0] == a[0][0]:
            return None
        return (a[-1][1] - a[0][1]) / (a[-1][0] - a[0][0]) * 60

    def frame(self, k: int | None = None, var: str | None = None) -> Frame | None:
        with self.lock:
            if not self.rec:
                return None
            following = k is None or k == self.rec[-1]
            k = self.rec[-1] if k is None else k
            if k == self.rec[-1]:
                cols = self.latest
            elif k in self._cache:
                cols = self._cache[k]
            else:
                return None
            rec = np.asarray(self.rec)
            m = rec <= k
            series = {n: np.asarray(v)[m] for n, v in self.series.items()}
            sel = np.where(m & (rec > k - self.trail_len))[0]
            trail = [self.trail_hist[i] for i in sel]
            return Frame(
                feed=self.feed, k=k, n_ready=self.n_ready,
                time=float(self.times[k]) if k < len(self.times) else np.nan,
                cols=cols, series_k=rec[m], series=series, trail=trail,
                n_trail=len(self.trail_idx), following=following,
                rate_per_min=self.rate(),
                last_new=self.arrivals[-1][0] if self.arrivals else None,
                read_s=self.read_s, n_expected=self.feed.expected_records(self.times),
                var=var,
            )
