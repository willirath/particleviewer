"""kelp-live: watch a growing kelp Zarr trajectory store in the terminal.

    kelp-live run.yaml                    # a kelp config: output_path + mesh.meshpath
    kelp-live tracks.zarr --mesh mesh.nc  # or the store directly
    kelp-live tracks.zarr --once          # print one gallery and exit (no TUI)

Panels (a number key shows that panel full screen, esc returns to the grid;
--show sets the grid):
  1 density   2 particles by status   3 trails   4 field (sampled variable)
  5 depth     6 distance from release 7 section (latitude x depth)   8 stats
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np
from rich.console import Console

from .geo import Plate, PolarStereo, auto_projection
from .mesh import Mesh
from .store import Feed, State
from .view import View


def resolve_target(target: str, mesh: str | None) -> tuple[str, str | None]:
    """A store path, or a kelp YAML config -> (store path, mesh path)."""
    if target.endswith((".yaml", ".yml")):
        import yaml
        with open(target) as fh:
            cfg = yaml.safe_load(fh) or {}
        base = os.path.dirname(os.path.abspath(target))

        def res(p):
            if not p:
                return None
            p = os.path.expanduser(str(p))
            return p if os.path.isabs(p) else os.path.normpath(os.path.join(base, p))

        store = res((cfg.get("data") or {}).get("output_path"))
        if not store:
            raise SystemExit(f"{target}: data.output_path is not set")
        mesh = mesh or res((cfg.get("mesh") or {}).get("meshpath"))
        target = store
    if not target.rstrip("/").endswith(".zarr"):
        raise SystemExit(f"{target}: kelp-live reads Zarr stores only "
                         "(set data.output_path to a .zarr path for live viewing)")
    return target, mesh


def load_mesh(path: str | None, console: Console) -> Mesh:
    if not path:
        return Mesh(None)
    try:
        with console.status(f"reading mesh {path} …"):
            return Mesh(path)
    except Exception as e:  # noqa: BLE001 - the background is optional
        console.print(f"[yellow]no land/bathymetry background: {path}: {e}[/]")
        return Mesh(None)


def wait_for(store: str, poll: float, console: Console) -> tuple[Feed, State]:
    feed = None
    with console.status(f"waiting for {store} …") as status:
        while True:
            try:
                feed = feed or Feed(store)
                state = State(feed)
                if state.update():
                    return feed, state
                status.update(f"waiting for the first complete record in {store} …")
            except (FileNotFoundError, KeyError, ValueError) as e:
                status.update(f"waiting for {store} ({type(e).__name__}) …")
                feed = None
            time.sleep(poll)


def make_projection(name: str, lon, lat):
    if name == "north":
        return PolarStereo(True)
    if name == "south":
        return PolarStereo(False)
    if name == "plate":
        rl = np.radians(lon[np.isfinite(lon)])
        return Plate(float(np.degrees(np.arctan2(np.sin(rl).mean(), np.cos(rl).mean()))) if len(rl) else 0.0)
    return auto_projection(lon, lat)


def main(argv=None):
    ap = argparse.ArgumentParser(prog="kelp-live", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("target", help="kelp config (.yaml) or trajectory store (.zarr)")
    ap.add_argument("--mesh", help="mesh for land/bathymetry (NEMO mesh_mask/domain_cfg, FESOM2 mesh diag, any lon/lat file)")
    ap.add_argument("--show", help="panels in the grid, e.g. 1,4,8 (default: 1,4,7,8 for 3-D, 1,2,4,8 for 2-D)")
    ap.add_argument("--proj", choices=["auto", "north", "south", "plate"], default="auto")
    ap.add_argument("--poll", type=float, default=2.0, help="seconds between polls of the store")
    ap.add_argument("--backlog", type=int, default=120,
                    help="max records read when catching up (strided beyond that)")
    ap.add_argument("--once", action="store_true", help="print a panel gallery and exit")
    ap.add_argument("--record", type=int, help="with --once: 1-based record to show (default newest)")
    ap.add_argument("--rows", type=int, default=22, help="with --once: text rows per panel")
    ap.add_argument("--png", help="with --once: also save the gallery as PNG")
    args = ap.parse_args(argv)

    console = Console()
    store, mesh_path = resolve_target(args.target, args.mesh)
    feed, state = wait_for(store, args.poll, console) if not args.once else (None, None)
    if args.once:
        feed = Feed(store)
        state = State(feed, backlog=args.backlog)
        if not state.update():
            console.print(f"{store}: no complete record on disk yet")
            return 1
    state.backlog = args.backlog
    mesh = load_mesh(mesh_path, console)

    latest = state.latest
    rl = feed.release
    lon = np.concatenate([rl.get("release_lon", np.zeros(0)), latest["lon"]]).astype(float)
    lat = np.concatenate([rl.get("release_lat", np.zeros(0)), latest["lat"]]).astype(float)
    view = View(make_projection(args.proj, lon, lat), mesh)
    view.fit(lon, lat)
    zz = np.concatenate([rl.get("release_z", np.zeros(0)), latest["z"]]).astype(float) if "z" in latest else None
    if zz is not None:
        view.grow_section(lat, zz)

    if args.show:
        show = [int(x) for x in args.show.split(",") if x.strip()]
    else:
        show = [1, 4, 7, 8] if feed.is_3d else [1, 2, 4, 8]

    if args.once:
        k = None if args.record is None else max(0, min(state.n_ready, args.record) - 1)
        if k is not None and not state.has(k):
            state.load(k)
        fr = state.frame(k)
        from .static import gallery, save_png
        out = gallery(fr, view, show, console.width - 1, args.rows)
        console.print(out)
        if args.png:
            save_png(console, out, args.png)
            console.print(f"saved {args.png}")
        return 0

    from .app import LiveApp
    LiveApp(state, view, show, poll=args.poll).run()
    return 0


if __name__ == "__main__":
    sys.exit(main())
