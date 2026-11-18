"""Non-interactive output: a panel gallery for plain terminals/logs, and PNG export."""
from __future__ import annotations

import numpy as np
from rich.console import Group
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from .panels import render_panel
from .render import BRAILLE_BITS
from .store import Frame
from .view import View


def gallery(fr: Frame, view: View, show: list[int], width: int, rows: int):
    """Panels of equal size laid out in a grid that fits ``width`` columns."""
    H, W = view.grid(width - 5, rows - 1)
    cols = W
    rows = H // 2 + 1
    per_row = max(1, min(len(show), width // (cols + 5)))
    panels = []
    for k in show:
        title, body = render_panel(k, fr, view, cols, rows)
        panels.append(Panel(body, title=title, title_align="left", width=cols + 4,
                            border_style="grey50", padding=(0, 1)))
    grid = Table.grid(padding=(0, 1))
    for _ in range(per_row):
        grid.add_column(vertical="top")
    for i in range(0, len(panels), per_row):
        row = panels[i:i + per_row]
        grid.add_row(*row, *[""] * (per_row - len(row)))
    f = fr.feed
    head = Text(f"{f.path} · record {fr.k + 1}/{fr.n_ready} · {f.label(fr.time)} · "
                f"{view.proj.name}", style="bold")
    return Group(head, grid)


def save_png(console, renderable, path: str, cw: int = 8, ch: int = 16) -> None:
    """Rasterise terminal cells exactly: half-blocks, braille dots, block bars, text."""
    from PIL import Image, ImageDraw, ImageFont

    lines = console.render_lines(renderable, console.options, pad=False)
    ncol = max((sum(len(s.text) for s in ln) for ln in lines), default=1)
    dbg, dfg = (24, 24, 27), (220, 220, 220)
    img = Image.new("RGB", (max(ncol, 1) * cw, max(len(lines), 1) * ch), dbg)
    dr = ImageDraw.Draw(img)
    font = None
    for cand in ("/System/Library/Fonts/Menlo.ttc", "DejaVuSansMono.ttf"):
        try:
            font = ImageFont.truetype(cand, 13)
            break
        except OSError:
            pass
    font = font or ImageFont.load_default()
    bars = "▁▂▃▄▅▆▇█"

    def rgb(c, d):
        return tuple(c.get_truecolor()) if c is not None else d

    for r, ln in enumerate(lines):
        x = 0
        for seg in ln:
            st = seg.style
            fg = rgb(st.color, dfg) if st else dfg
            bg = rgb(st.bgcolor, dbg) if st else dbg
            for c in seg.text:
                X, Y = x * cw, r * ch
                dr.rectangle([X, Y, X + cw - 1, Y + ch - 1], fill=bg)
                o = ord(c)
                if c == "▀":
                    dr.rectangle([X, Y, X + cw - 1, Y + ch // 2 - 1], fill=fg)
                elif 0x2800 < o <= 0x28FF:
                    b = o - 0x2800
                    for (dy, dx), bit in np.ndenumerate(BRAILLE_BITS):
                        if b & bit:
                            cx, cy = X + (dx + 0.5) * cw / 2, Y + (dy + 0.5) * ch / 4
                            dr.rectangle([cx - 1, cy - 1, cx + 1, cy + 1], fill=fg)
                elif c in bars:
                    h = (bars.index(c) + 1) * ch // 8
                    dr.rectangle([X, Y + ch - h, X + cw - 1, Y + ch - 1], fill=fg)
                elif c == "░":
                    dr.rectangle([X, Y + ch // 4, X + cw - 1, Y + 3 * ch // 4],
                                 fill=tuple(v // 3 for v in fg))
                elif c != " ":
                    dr.text((X, Y + 1), c, font=font, fill=fg)
                x += 1
    img.save(path)
