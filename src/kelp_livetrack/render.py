"""Colour maps and terminal raster rendering (half-blocks, braille, bars)."""
from __future__ import annotations

import numpy as np

RESET = "\x1b[0m"

# Anchor tables, linearly interpolated.
CMAPS = {
    "inferno": [(20, 11, 52), (66, 10, 104), (120, 28, 109), (171, 47, 94),
                (219, 80, 59), (243, 132, 21), (250, 193, 39), (252, 255, 164)],
    "thermal": [(4, 35, 51), (23, 51, 122), (85, 59, 157), (129, 79, 143),
                (175, 95, 130), (222, 112, 101), (249, 146, 66), (249, 196, 65),
                (232, 250, 91)],
    "viridis": [(68, 1, 84), (72, 40, 120), (62, 73, 137), (49, 104, 142),
                (38, 130, 142), (31, 158, 137), (53, 183, 121), (110, 206, 88),
                (181, 222, 43), (253, 231, 37)],
    "deep": [(253, 254, 204), (163, 222, 166), (89, 180, 170), (64, 135, 163),
             (62, 90, 154), (60, 49, 106), (40, 26, 44)],
    "amp": [(241, 237, 236), (230, 196, 182), (219, 155, 129), (206, 113, 81),
            (187, 70, 46), (155, 30, 36), (104, 12, 34), (60, 9, 17)],
}


def cmap(name: str, v) -> np.ndarray:
    """Map values in [0, 1] to uint8 RGB (..., 3); NaN maps to 0."""
    a = np.asarray(CMAPS[name], dtype=float)
    v = np.clip(np.nan_to_num(np.asarray(v, dtype=float)), 0, 1) * (len(a) - 1)
    i = np.minimum(v.astype(int), len(a) - 2)
    f = (v - i)[..., None]
    return (a[i] * (1 - f) + a[i + 1] * f).astype(np.uint8)


def blend(a, b, w) -> np.ndarray:
    return (np.asarray(a, float) * (1 - np.asarray(w, float))
            + np.asarray(b, float) * np.asarray(w, float)).astype(np.uint8)


def render_halfblock(rgb: np.ndarray) -> list[str]:
    """(H, W, 3) uint8 with H even -> lines of '▀' (fg = top pixel, bg = bottom pixel)."""
    H, W, _ = rgb.shape
    if H % 2:
        raise ValueError("half-block raster needs an even pixel height")
    px = rgb.tolist()
    lines = []
    for r in range(0, H, 2):
        top, bot = px[r], px[r + 1]
        out, last = [], None
        for c in range(W):
            key = (*top[c], *bot[c])
            if key != last:
                out.append("\x1b[38;2;%d;%d;%d;48;2;%d;%d;%dm" % key)
                last = key
            out.append("▀")
        out.append(RESET)
        lines.append("".join(out))
    return lines


# dot bit for (row 0..3, col 0..1) inside a braille cell
BRAILLE_BITS = np.array([[0x01, 0x08], [0x02, 0x10], [0x04, 0x20], [0x40, 0x80]])


def render_braille(dots: np.ndarray, fg: np.ndarray, bg: np.ndarray) -> list[str]:
    """dots (4R, 2C) bool; fg, bg (R, C, 3) uint8 per cell."""
    R, C = fg.shape[:2]
    bits = (dots.reshape(R, 4, C, 2) * BRAILLE_BITS[None, :, None, :]).sum(axis=(1, 3))
    fgl, bgl, bl = fg.tolist(), bg.tolist(), bits.tolist()
    lines = []
    for r in range(R):
        out, last = [], None
        for c in range(C):
            b = bl[r][c]
            if b:
                key = (*fgl[r][c], *bgl[r][c])
                if key != last:
                    out.append("\x1b[38;2;%d;%d;%d;48;2;%d;%d;%dm" % key)
                out.append(chr(0x2800 + b))
            else:
                key = (None, None, None, *bgl[r][c])
                if key != last:
                    out.append("\x1b[48;2;%d;%d;%dm" % key[3:])
                out.append(" ")
            last = key
        out.append(RESET)
        lines.append("".join(out))
    return lines


def colorbar(name: str, vmin: float, vmax: float, width: int, label: str = "",
             fmt: str = "{:.3g}") -> str:
    lo, hi = fmt.format(vmin), fmt.format(vmax)
    n = max(4, width - len(lo) - len(hi) - len(label) - 3)
    cols = cmap(name, np.linspace(0, 1, n)).tolist()
    bar = "".join("\x1b[48;2;%d;%d;%dm " % tuple(c) for c in cols) + RESET
    return f"{lo} {bar} {hi} {label}"[: 10 * width]


SPARK = "▁▂▃▄▅▆▇█"


def sparkline(vals, width: int) -> str:
    """Values -> one line of block characters, binned down to ``width``."""
    v = np.asarray(vals, float)
    if len(v) == 0 or width <= 0:
        return ""
    if len(v) > width:
        edges = np.linspace(0, len(v), width + 1).astype(int)
        with np.errstate(all="ignore"):
            v = np.array([np.nanmean(v[a:b]) if np.isfinite(v[a:b]).any() else np.nan
                          for a, b in zip(edges[:-1], edges[1:])])
    ok = np.isfinite(v)
    if not ok.any():
        return ""
    lo, hi = np.nanmin(v), np.nanmax(v)
    s = (v - lo) / (hi - lo) if hi > lo else np.full_like(v, 0.5)
    return "".join(SPARK[int(round(x * 7))] if np.isfinite(x) else " " for x in s)
