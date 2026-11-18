import asyncio
import re

import numpy as np
import pytest
from rich.console import Console

from kelp_livetrack.geo import Plate, PolarStereo, auto_projection, haversine_km
from kelp_livetrack.mesh import Mesh
from kelp_livetrack.panels import PANELS, render_panel
from kelp_livetrack.render import render_braille, render_halfblock, sparkline
from kelp_livetrack.static import gallery, save_png
from kelp_livetrack.store import Feed, State
from kelp_livetrack.view import View

from conftest import make_store

ANSI = re.compile(r"\x1b\[[0-9;]*m")


@pytest.mark.parametrize("proj", [PolarStereo(True), PolarStereo(False), Plate(170.0)])
def test_projection_roundtrip(proj):
    lon = np.array([-179.0, -40.0, 0.0, 60.0, 179.5])
    lat = np.array([75.0, 80.0, 89.0, 66.0, 70.0]) * (-1 if proj.name.startswith("south") else 1)
    x, y = proj.fwd(lon, lat)
    lo, la = proj.inv(x, y)
    np.testing.assert_allclose(la, lat, atol=1e-9)
    np.testing.assert_allclose((lo - lon + 180) % 360 - 180, 0, atol=1e-9)


def test_auto_projection():
    assert auto_projection(np.array([0, 10.0]), np.array([60, 80.0])).name.startswith("north")
    assert auto_projection(np.array([0, 10.0]), np.array([-60, -80.0])).name.startswith("south")
    p = auto_projection(np.array([170, -170.0]), np.array([0, 10.0]))
    assert isinstance(p, Plate) and abs(abs(p.lon0) - 180) < 1e-6


def test_haversine():
    assert abs(haversine_km(0, 0, 0, 1) - 111.19) < 0.1


def test_halfblock_and_braille_shapes():
    rgb = np.zeros((6, 5, 3), np.uint8)
    lines = render_halfblock(rgb)
    assert len(lines) == 3 and all(len(ANSI.sub("", ln)) == 5 for ln in lines)
    dots = np.zeros((8, 6), bool)
    dots[0, 0] = dots[7, 5] = True
    fg = np.full((2, 3, 3), 255, np.uint8)
    lines = render_braille(dots, fg, np.zeros_like(fg))
    plain = [ANSI.sub("", ln) for ln in lines]
    assert plain[0][0] == chr(0x2801) and plain[1][2] == chr(0x2880)
    assert sparkline([1, 2, 3, 4], 4) == "▁▃▆█"
    assert len(sparkline(np.arange(100.0), 10)) == 10


def test_mesh_point_cloud_land():
    lon, lat = np.meshgrid(np.arange(-20, 21, 1.0), np.arange(60, 81, 1.0))
    keep = ~((np.abs(lon) < 5) & (np.abs(lat - 70) < 5))  # a hole = land
    m = Mesh(None, lon=lon[keep], lat=lat[keep])
    rgb = m.raster(np.array([[0.0, -15.0]]), np.array([[70.0, 70.0]]))
    assert tuple(rgb[0, 0]) == (74, 70, 64)  # land in the hole
    assert tuple(rgb[0, 1]) != (74, 70, 64)


def _state(path, **kw):
    s = State(Feed(str(path)), **kw)
    assert s.update()
    v = View(auto_projection(s.latest["lon"], s.latest["lat"]))
    v.fit(s.latest["lon"], s.latest["lat"])
    return s, v


@pytest.mark.parametrize("threed", [True, False])
def test_all_panels_render(tmp_path, threed):
    p = tmp_path / "t.zarr"
    make_store(p, threed=threed, sampled=() if not threed else ("votemper",))
    s, v = _state(p)
    fr = s.frame()
    for k in PANELS:
        title, body = render_panel(k, fr, v, 40, 14)
        assert title.startswith(f"{k} ·")
        Console(width=60, file=open("/dev/null", "w")).print(body)


def test_zoom_pan_changes_extent(store):
    s, v = _state(store)
    e0 = v.extent
    v.zoom_by(2)
    v.pan(0.2, 0)
    assert v.extent != e0
    v.reset()
    assert v.extent == e0


def test_gallery_png(tmp_path, store):
    s, v = _state(store)
    con = Console(width=120, file=open("/dev/null", "w"))
    out = gallery(s.frame(), v, [1, 2, 3, 4, 5, 6, 7, 8], 119, 12)
    save_png(con, out, str(tmp_path / "g.png"))
    assert (tmp_path / "g.png").stat().st_size > 1000


def test_app_pilot(tmp_path, store):
    """Drive the Textual app headless: panels toggle, scrub, zoom, follow."""
    from kelp_livetrack.app import LiveApp, PanelWidget

    s, v = _state(store)

    async def run():
        app = LiveApp(s, v, [1, 4, 7, 8], poll=0.2, png_dir=str(tmp_path))
        async with app.run_test(size=(140, 45)) as pilot:
            await pilot.pause()
            assert {w.kind for w in app.query(PanelWidget) if w.display} == {1, 4, 7, 8}
            shown = lambda: {w.kind for w in app.query(PanelWidget) if w.display}
            await pilot.press("2")
            assert shown() == {2}
            await pilot.pause()
            p2 = app.query_one("#panel-2", PanelWidget)
            assert p2.size.width > 120 and p2.size.height > 35  # full screen
            await pilot.press("3")
            assert shown() == {3}
            await pilot.press("3")
            assert shown() == {1, 4, 7, 8}
            await pilot.press("5", "escape")
            assert shown() == {1, 4, 7, 8}
            await pilot.press("a")
            assert shown() == set(range(1, 9))
            await pilot.press("1")
            assert shown() == {1}
            await pilot.press("comma", "comma")
            await pilot.pause(0.5)
            assert app.k_view == 37 and app.frame.k == 37
            await pilot.press("plus", "left", "v", "0")
            await pilot.press("f")
            assert app.k_view is None and app.frame.k == 39
            await pilot.press("s")
            await pilot.pause()
        return app

    asyncio.run(run())
    assert list(tmp_path.glob("kelp-live_*.png"))


def test_app_playback(store):
    """Play replays from the first record, pauses, and hands over to live at the end."""
    from textual.widgets import Button

    from kelp_livetrack.app import LiveApp

    s, v = _state(store)

    async def run():
        app = LiveApp(s, v, [1, 8], poll=5)
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            await pilot.click("#btn-play")
            await pilot.pause(0.1)
            assert app.playing and app.k_view is not None
            assert str(app.query_one("#btn-play", Button).label) == "||"
            await pilot.pause(1.0)
            k1 = app.k_view
            assert 1 <= k1 < 39 and app.frame.k == k1  # 4 records/s
            await pilot.press("space")
            assert not app.playing
            await pilot.pause(0.5)
            assert app.k_view == k1  # paused stays put
            await pilot.press("right_square_bracket", "right_square_bracket", "right_square_bracket")
            assert app.speed_i == 5
            await pilot.press("space")
            for _ in range(40):
                await pilot.pause(0.1)
                if not app.playing:
                    break
            assert not app.playing and app.k_view is None and app.frame.k == 39  # live again
            await pilot.press("space", "home")  # play from live restarts at record 1
            await pilot.pause(0.2)
            assert app.playing
            await pilot.press("comma")  # stepping stops playback
            assert not app.playing

    asyncio.run(run())


def test_section_axes_only_widen(store):
    s, v = _state(store)
    r1 = v.grow_section(np.array([70.0, 80.0]), np.array([0.0, 500.0]))
    r2 = v.grow_section(np.array([75.0, 76.0]), np.array([10.0, 20.0]))
    assert r2 == r1
    r3 = v.grow_section(np.array([50.0]), np.array([4000.0]))
    assert r3[0] == 50.0 and r3[1] == r1[1] and r3[2] == 4000.0
