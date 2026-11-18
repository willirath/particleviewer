"""Panels: each renders one view of a Frame into a cols x rows cell box.

Map panels return a list of ANSI lines (last line = legend / colour bar);
the stats panel returns a Rich renderable.
"""
from __future__ import annotations

import re
import time as _time

import numpy as np
from rich.table import Table
from rich.text import Text

from .render import RESET, blend, cmap, colorbar, render_braille, render_halfblock, sparkline
from .store import STATUS, Frame
from .view import View

STATUS_RGB = {1: (235, 235, 225), 2: (255, 150, 40), 3: (240, 60, 60), 4: (110, 110, 120)}

TITLES = {
    1: "density",
    2: "particles by status",
    3: "trails",
    4: "field",
    5: "depth",
    6: "distance from release",
    7: "section",
    8: "stats",
}


def _alive(fr: Frame):
    c = fr.cols
    m = (c["status"] == 1) & np.isfinite(c["lon"]) & np.isfinite(c["lat"])
    return c["lon"][m], c["lat"][m], m


def density(fr: Frame, view: View, cols: int, rows: int):
    H, W = view.grid(cols, rows - 1)
    lon, lat, _ = _alive(fr)
    rgb = view.background(H, W).copy()
    idx = view.bin(lon, lat, H, W)
    cnt = np.bincount(idx[idx >= 0], minlength=H * W).reshape(H, W)
    has = cnt > 0
    vmax = float(np.percentile(cnt[has], 99)) if has.any() else 1.0
    vmax = max(vmax, 1.0)
    rgb[has] = cmap("inferno", 0.12 + 0.88 * np.log1p(np.minimum(cnt[has], vmax)) / np.log1p(vmax))
    lines = render_halfblock(rgb)
    lines.append(colorbar("inferno", 1, vmax, W, "alive / px (log, p99)", "{:.0f}"))
    return f"density · {int(has.sum())} px", lines


def status(fr: Frame, view: View, cols: int, rows: int):
    H, W = view.grid(cols, rows - 1)
    Hd, Wd = 2 * H, 2 * W  # braille dots: 2 x 4 per cell
    R, C = H // 2, W
    c = fr.cols
    dots = np.zeros(Hd * Wd, bool)
    code = np.zeros(R * C, int)
    for k in (1, 2, 3):  # rarer states drawn last so they stay visible
        m = (c["status"] == k) & np.isfinite(c["lon"])
        idx = view.bin(c["lon"][m], c["lat"][m], Hd, Wd)
        idx = idx[idx >= 0]
        dots[idx] = True
        code[(idx // Wd // 4) * C + (idx % Wd) // 2] = k
    fg = np.zeros((R * C, 3), np.uint8)
    for k, rgb in STATUS_RGB.items():
        fg[code == k] = rgb
    lines = render_braille(dots.reshape(Hd, Wd), fg.reshape(R, C, 3), view.cell_background(H, W))
    lines.append("  ".join("\x1b[38;2;%d;%d;%dm⣿" % STATUS_RGB[k] + RESET
                           + f" {STATUS[k][:7]} {int((c['status'] == k).sum())}" for k in (1, 2, 3, 4)))
    return "particles by status", lines


def trails(fr: Frame, view: View, cols: int, rows: int):
    H, W = view.grid(cols, rows - 1)
    Hd, Wd = 2 * H, 2 * W
    R, C = H // 2, W
    age = np.zeros(Hd * Wd)
    n = len(fr.trail)
    prev = None
    for a, (lon, lat) in enumerate(fr.trail):
        w = (a + 1) / n
        px, py = view.to_pixel(lon, lat, Hd, Wd)
        pts = [(px, py)]
        if prev is not None:  # connect consecutive records, skip wraps / jumps
            qx, qy = prev
            ok = np.isfinite(px) & np.isfinite(qx) & (np.hypot(px - qx, py - qy) < 0.25 * Wd)
            for s in (0.2, 0.4, 0.6, 0.8):
                pts.append((np.where(ok, qx + s * (px - qx), np.nan),
                            np.where(ok, qy + s * (py - qy), np.nan)))
        for x, y in pts:
            ok = np.isfinite(x) & np.isfinite(y) & (x >= 0) & (x < Wd) & (y >= 0) & (y < Hd)
            np.maximum.at(age, y[ok].astype(int) * Wd + x[ok].astype(int), w)
        prev = (px, py)
    cell = age.reshape(R, 4, C, 2).max(axis=(1, 3))
    fg = blend((90, 110, 150), (255, 230, 150), cell[..., None])
    lines = render_braille((age > 0).reshape(Hd, Wd), fg, view.cell_background(H, W))
    lines.append(f"{fr.n_trail} particles · last {n} records · bright = recent")
    return "trails", lines


def _cell_mean(fr, view, cols, rows, values, cmap_name, title, unit, fmt="{:.3g}"):
    H, W = view.grid(cols, rows - 1)
    lon, lat, m = _alive(fr)
    v = np.asarray(values)[m]
    ok = np.isfinite(v)
    idx = view.bin(lon[ok], lat[ok], H, W)
    v = v[ok][idx >= 0]
    idx = idx[idx >= 0]
    rgb = view.background(H, W).copy()
    vmin, vmax = 0.0, 1.0
    if len(v):
        vmin, vmax = (float(x) for x in np.percentile(v, [2, 98]))
        vmax = vmax if vmax > vmin else vmin + 1e-6
        s = np.bincount(idx, weights=v, minlength=H * W)
        n = np.bincount(idx, minlength=H * W)
        has = (n > 0).reshape(H, W)
        mean = (s / np.maximum(n, 1)).reshape(H, W)
        rgb[has] = cmap(cmap_name, (mean[has] - vmin) / (vmax - vmin))
    lines = render_halfblock(rgb)
    lines.append(colorbar(cmap_name, vmin, vmax, W, f"{unit} cell mean (p2–p98)", fmt))
    return title, lines


def default_var(names: list[str]) -> str | None:
    """Temperature first if there is one, else the first sampled variable."""
    temps = [n for n in names if "temp" in n.lower() or n.lower() in ("t", "thetao", "to", "sst")]
    return (temps or names or [None])[0]


def field(fr: Frame, view: View, cols: int, rows: int):
    var = fr.var or default_var(fr.feed.sampled)
    if var is None or var not in fr.cols:
        return "field", ["(no sampled variables in this store)"]
    cm = "thermal" if "temp" in var.lower() or var.lower() in ("t", "thetao", "votemper") else "viridis"
    return _cell_mean(fr, view, cols, rows, fr.cols[var], cm, f"field · {var} [v]",
                      fr.feed.units.get(var, ""))


def depth(fr: Frame, view: View, cols: int, rows: int):
    if not fr.feed.is_3d or "z" not in fr.cols:
        return "depth", ["(2-D run)"]
    return _cell_mean(fr, view, cols, rows, fr.cols["z"], "deep", "depth", "m", "{:.0f}")


def displacement(fr: Frame, view: View, cols: int, rows: int):
    if "disp" not in fr.cols:
        return "distance from release", ["(no release positions in store)"]
    return _cell_mean(fr, view, cols, rows, fr.cols["disp"], "amp", "distance from release",
                      "km", "{:.0f}")


def section(fr: Frame, view: View, cols: int, rows: int):
    c = fr.cols
    if not fr.feed.is_3d or "z" not in c:
        return "section", ["(2-D run)"]
    m = (c["status"] == 1) & np.isfinite(c["lat"]) & np.isfinite(c["z"])
    lat, z = c["lat"][m], c["z"][m]
    if not len(z):
        return "section", ["(no alive particles)"]
    lab = 7
    W, H = max(8, cols - lab), 2 * max(2, rows - 2)
    la0, la1, zmax = view.grow_section(lat, z)
    ci = np.clip(((lat - la0) / max(la1 - la0, 1e-6) * W).astype(int), 0, W - 1)
    ri = np.clip((np.sqrt(np.clip(z, 0, None) / zmax) * H).astype(int), 0, H - 1)
    cnt = np.bincount(ri * W + ci, minlength=H * W).reshape(H, W)
    rgb = np.empty((H, W, 3), np.uint8)
    rgb[:] = (18, 24, 34)
    has = cnt > 0
    rgb[has] = cmap("viridis", 0.1 + 0.9 * np.log1p(cnt[has]) / np.log1p(cnt.max()))
    body = render_halfblock(rgb)
    step = max(1, len(body) // 6)
    lines = []
    for r, ln in enumerate(body):
        dz = zmax * ((2 * r + 1) / H) ** 2
        lines.append((f"{dz:5.0f}m " if r % step == 0 else " " * lab) + ln)
    ns = "N" if la1 >= 0 else "S"
    lines.append(" " * lab + f"{abs(la0):.1f}°".ljust(max(1, W - 7)) + f"{abs(la1):.1f}°{ns}")
    return "section · latitude × depth (√z)", lines


def _ago(t):
    if t is None:
        return "-"
    s = _time.time() - t
    return f"{s:.0f}s" if s < 120 else f"{s / 60:.0f}min" if s < 7200 else f"{s / 3600:.1f}h"


def stats(fr: Frame, view: View, cols: int, rows: int):
    f, a = fr.feed, fr.feed.attrs
    barw = max(8, cols - 27)
    tbl = Table.grid(padding=(0, 1))
    tbl.add_column(style="bold", no_wrap=True, width=14, overflow="ellipsis")
    tbl.add_column(no_wrap=True, overflow="ellipsis")
    tbl.add_column(justify="right", no_wrap=True, width=10, overflow="ellipsis")

    n_exp = fr.n_expected
    if n_exp:
        filled = int(round(barw * min(fr.n_ready / n_exp, 1)))
        tbl.add_row("records", "█" * filled + "░" * (barw - filled), f"{fr.n_ready}/{n_exp}")
    else:
        tbl.add_row("records", "", str(fr.n_ready))
    rate = fr.rate_per_min
    live = f"last new {_ago(fr.last_new)} ago"
    if rate:
        live += f" · {rate:.3g}/min"
        if n_exp and n_exp > fr.n_ready:
            eta = (n_exp - fr.n_ready) / rate
            live += f" · ETA {eta:.0f}min" if eta < 120 else f" · ETA {eta / 60:.1f}h"
    tbl.add_row("live", live, "")
    tbl.add_row("shown", f"#{fr.k + 1} · {f.label(fr.time)}", "LIVE" if fr.following else "history")
    tbl.add_row("run", f"{f.P} pcl · {a.get('mesh_type', '?')} · {a.get('integrator', '?')}"
                f" · {a.get('dt', 0):.0f}s", "3-D" if f.is_3d else "2-D")

    st = fr.cols["status"]
    counts = {k: int((st == k).sum()) for k in STATUS}
    sbar = "".join("\x1b[38;2;%d;%d;%dm" % STATUS_RGB[k] + "█" * int(round(barw * n / max(f.P, 1)))
                   for k, n in counts.items()) + RESET
    tbl.add_row("status", Text.from_ansi(sbar), f"{counts[1] / max(f.P, 1):.0%}")
    tbl.add_row("", " · ".join(f"{STATUS[k]} {n}" for k, n in counts.items() if n), "")
    if "failure_reason" in fr.cols:
        fr_ = fr.cols["failure_reason"]
        nz = {int(k): int((fr_ == k).sum()) for k in np.unique(fr_) if k}
        if nz:
            tbl.add_row("failures", " ".join(f"reason {k}: {n}" for k, n in nz.items()), "")

    tbl.add_row("", "", "")
    for name, vals in fr.series.items():
        if name in STATUS.values() and not np.nansum(vals):
            continue
        cur = vals[-1] if len(vals) else np.nan
        tbl.add_row(name, Text(sparkline(vals, barw), style="cyan"),
                    "-" if not np.isfinite(cur) else f"{cur:.4g}")

    z = fr.cols["z"][st == 1] if "z" in fr.cols and f.is_3d else np.zeros(0)
    z = z[np.isfinite(z)]
    if len(z) and z.max() > 0:
        tbl.add_row("", "", "")
        edges = [0, 10, 50, 100, 200, 500, 1000, 2000, 4000, np.inf]
        h, _ = np.histogram(z, edges)
        for lo, hi, n in zip(edges[:-1], edges[1:], h):
            w = int(round(barw * n / max(h.max(), 1)))
            lbl = f"{lo:.0f}–{hi:.0f} m" if np.isfinite(hi) else f"> {lo:.0f} m"
            tbl.add_row(Text(lbl, style="not bold"), Text("▇" * w, style="blue"), str(n))
    tbl.add_row("", "", "")
    tbl.add_row(Text("io", style="dim"), Text(f"{fr.read_s:.2f}s reading", style="dim"), "")
    return "stats", tbl


PANELS = {1: density, 2: status, 3: trails, 4: field, 5: depth, 6: displacement,
          7: section, 8: stats}


_ANSI = re.compile(r"\x1b\[[0-9;]*m")


def render_panel(kind: int, fr: Frame, view: View, cols: int, rows: int):
    """(title, Rich renderable) for one panel in a cols x rows content box; maps centred."""
    title, body = PANELS[kind](fr, view, cols, rows)
    if isinstance(body, list):
        w = len(_ANSI.sub("", body[0])) if body else 0
        pad = " " * max(0, (cols - w) // 2) if len(body) > 1 else ""
        body = Text.from_ansi("\n".join(pad + ln for ln in body), no_wrap=True, end="")
    return f"{kind} · {title}", body
