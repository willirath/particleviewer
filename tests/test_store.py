import numpy as np
import zarr

from kelp_livetrack.replay import append, per_record, start
from kelp_livetrack.store import Feed, State

from conftest import make_store


def test_feed_basics(store):
    f = Feed(store)
    assert f.P == 300 and f.is_3d
    assert f.sampled == ["votemper"] and f.units["votemper"] == "degC"
    assert f.n_ready() == 40
    assert f.label(86400.0) == "2011-01-02 00:00"
    assert f.expected_records(f.times(40)) == 40


def test_torn_flush_not_counted(tmp_path, store):
    """time resized and lon written, but status not yet: the record is not ready."""
    src, dst = start(store, str(tmp_path / "grow.zarr"))
    names = per_record(src)
    append(src, dst, 0, 5, names)
    f = Feed(str(tmp_path / "grow.zarr"))
    assert f.n_ready() == 5
    # emulate kelp mid-flush: time + lon/lat grown, status resized but chunk unwritten
    dst["time"].resize((8,))
    dst["time"][5:8] = src["time"][5:8]
    for n in ("lon", "lat"):
        dst[n].resize((300, 8))
        dst[n][:, 5:8] = src[n][:, 5:8]
    assert f.n_ready() == 5  # z/status still 5 long
    dst["z"].resize((300, 8))
    dst["status"].resize((300, 8))  # resized, fill 0, not written
    assert f.n_ready() == 5
    dst["status"][:, 5:8] = src["status"][:, 5:8]
    dst["z"][:, 5:8] = src["z"][:, 5:8]
    assert f.n_ready() == 8


def test_incremental_equals_full(tmp_path, store):
    full = State(Feed(store))
    assert full.update()
    src, dst = start(store, str(tmp_path / "grow.zarr"))
    names = per_record(src)
    inc = State(Feed(str(tmp_path / "grow.zarr")))
    assert not inc.update()
    for k0 in range(0, 40, 7):
        append(src, dst, k0, min(40, k0 + 7), names)
        assert inc.update()
    assert not inc.update()
    assert inc.rec == full.rec == list(range(40))
    for name in full.series:
        np.testing.assert_allclose(inc.series[name], full.series[name], equal_nan=True)
    assert inc.series["beached"][-1] == 30
    np.testing.assert_allclose(inc.latest["lon"], full.latest["lon"])


def test_backlog_stride(store):
    s = State(Feed(store), backlog=10)
    assert s.update()
    assert len(s.rec) == 10 and s.rec[0] == 0 and s.rec[-1] == 39
    assert all(len(v) == 10 for v in s.series.values())


def test_frame_scrub_and_trails(store):
    s = State(Feed(store), trail_len=5)
    s.update()
    fr = s.frame()
    assert fr.following and fr.k == 39 and len(fr.trail) == 5
    assert "disp" in fr.cols and np.nanmax(fr.cols["disp"]) > 0
    assert s.frame(10) is None  # not cached yet
    s.load(10)
    fr = s.frame(10)
    assert not fr.following and fr.k == 10
    assert fr.series_k[-1] == 10 and len(fr.trail) == 5
    np.testing.assert_allclose(fr.cols["lon"], zarr.open_group(store)["lon"][:, 10], rtol=1e-6)


def test_lazy_variable_appears(tmp_path):
    p = tmp_path / "t.zarr"
    make_store(p, sampled=())
    s = State(Feed(str(p)))
    s.update()
    assert s.feed.sampled == []
    g = zarr.open_group(str(p), mode="a")
    a = g.create_array("vosaline", shape=(300, 40), chunks=(300, 8), dtype="f8",
                       fill_value=np.nan, dimension_names=("trajectory", "time"))
    a[:] = 35.0
    a.attrs["units"] = "1e-3"
    # grow by one record so update() rescans
    for n in ("time", "lon", "lat", "z", "status", "vosaline"):
        arr = g[n]
        arr.resize((41,) if arr.ndim == 1 else (300, 41))
        arr[..., 40] = 1 if n == "status" else 35.0 if n == "vosaline" else 1.0
    assert s.update()
    assert s.feed.sampled == ["vosaline"]
    assert "mean vosaline [1e-3]" in s.series
