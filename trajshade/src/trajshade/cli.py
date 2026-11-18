"""trajshade STORE.zarr — serve the interactive view, or --png for a single image."""
from __future__ import annotations

import argparse

from .aggregate import REDUCTIONS


def main(argv=None):
    p = argparse.ArgumentParser(prog="trajshade", description=__doc__)
    p.add_argument("store", help="trajectory Zarr store (kelp or Parcels layout)")
    p.add_argument("--proj", default="auto", choices=["auto", "plate", "north", "south"])
    p.add_argument("--size", type=int, default=800, help="canvas pixels per side")
    p.add_argument("--stride", type=int, help="use every n-th particle (default: ~5M vertices)")
    p.add_argument("--scale", type=int, default=None,
                   help="browser: canvas pixels per CSS pixel (default: device pixel ratio)")
    p.add_argument("--mpl", action="store_true",
                   help="native matplotlib window instead of the browser app")
    p.add_argument("--port", type=int, default=5006)
    p.add_argument("--no-show", action="store_true", help="don't open a browser")
    g = p.add_argument_group("static image / matplotlib window")
    g.add_argument("--png", help="write one aggregation to this PNG and exit")
    g.add_argument("--how", default="trajectories", choices=REDUCTIONS)
    g.add_argument("--var", default="record")
    g.add_argument("--records", default=None, help="k0:k1 record window")
    g.add_argument("--norm", default="eq_hist", choices=["eq_hist", "log", "linear"])
    a = p.parse_args(argv)

    if a.png:
        from .app import static_png
        k0, k1 = 0, None
        if a.records:
            s0, s1 = a.records.split(":")
            k0, k1 = int(s0 or 0), int(s1) if s1 else None
        agg = static_png(a.store, a.png, a.how, a.var, a.proj, a.size, a.stride, k0, k1, a.norm)
        print(f"{a.png}: {a.how} min {float(agg.min()):.3g} max {float(agg.max()):.3g}")
        return

    if a.mpl:
        from .mpl import run
        k0, k1 = 0, None
        if a.records:
            s0, s1 = a.records.split(":")
            k0, k1 = int(s0 or 0), int(s1) if s1 else None
        run(a.store, how=a.how, var=a.var, proj=a.proj, stride=a.stride, k0=k0, k1=k1,
            norm=a.norm)
        return

    import panel as pn
    from .app import build
    pn.serve(lambda: build(a.store, a.proj, a.size, a.stride, a.scale), port=a.port, show=not a.no_show,
             title="trajshade")


if __name__ == "__main__":
    main()
