"""Synthetic stores in kelp's output format 2 (Zarr v3)."""
import numpy as np
import pytest
import zarr


def make_store(path, P=300, T=40, chunk=8, threed=True, sampled=("votemper",),
               lat0=70.0, written=None):
    """A finished store with drifting particles; ``written`` < T leaves later records unwritten."""
    rng = np.random.default_rng(0)
    g = zarr.open_group(str(path), mode="w")
    g.attrs.update({"kelp_output_format": 2, "mesh_type": "nemo", "integrator": "rk4",
                    "dt": 3600.0, "n_steps": T * 24, "mode_3d": int(threed)})
    lon0 = rng.uniform(-30, 30, P)
    la0 = rng.uniform(lat0, lat0 + 10, P)
    k = np.arange(T)
    lon = lon0[:, None] + 0.05 * k[None, :]
    lat = la0[:, None] + 0.02 * k[None, :]
    z = np.abs(rng.normal(300, 200, P))[:, None] + 2.0 * k[None, :]
    status = np.ones((P, T), np.int32)
    status[: P // 10, T // 2:] = 2  # some beach half way
    W = T if written is None else written

    def arr(name, data, fill, dims, chunks, attrs=None):
        a = g.create_array(name, shape=data.shape if data.ndim == 1 and name != "time" else
                           (data.shape[0], W) if data.ndim == 2 else (W,),
                           chunks=chunks, dtype=data.dtype, fill_value=fill, dimension_names=dims)
        if data.ndim == 2:
            a[:, :W] = data[:, :W]
        elif name == "time":
            a[:W] = data[:W]
        else:
            a[:] = data
        a.attrs.update(attrs or {})

    arr("time", (k + 1) * 86400.0, np.nan, ("time",), (chunk,),
        {"units": "seconds since 2011-01-01 00:00:00", "calendar": "gregorian"})
    arr("trajectory", np.arange(P, dtype=np.int32), 0, ("trajectory",), (P,))
    arr("release_lon", lon0, np.nan, ("trajectory",), (P,))
    arr("release_lat", la0, np.nan, ("trajectory",), (P,))
    arr("release_z", z[:, 0].copy(), np.nan, ("trajectory",), (P,))
    for name, data, fill in (("lon", lon, np.nan), ("lat", lat, np.nan), ("z", z, np.nan),
                             ("status", status, 0)):
        arr(name, data, fill, ("trajectory", "time"), (P, chunk))
    for v in sampled:
        arr(v, (5 - 0.01 * z).astype(float), np.nan, ("trajectory", "time"), (P, chunk),
            {"units": "degC"})
    return g


@pytest.fixture
def store(tmp_path):
    p = tmp_path / "tracks.zarr"
    make_store(p)
    return str(p)
