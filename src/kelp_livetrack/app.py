"""Textual app: live, zoomable panels over a growing kelp Zarr store."""
from __future__ import annotations

import io
import math
import time

from rich.console import Console
from rich.markup import escape
from rich.text import Text
from textual import work
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Grid, Horizontal
from textual.widgets import Button, Footer, Static

from .panels import TITLES, default_var, render_panel
from .static import gallery, save_png
from .store import State
from .view import View


class PanelWidget(Static):
    DEFAULT_CSS = """
    PanelWidget {
        border: round $panel-lighten-2;
        border-title-color: $text-muted;
        padding: 0 1;
        width: 1fr;
        height: 1fr;
        overflow: hidden;
    }
    """

    def __init__(self, kind: int):
        super().__init__(id=f"panel-{kind}")
        self.kind = kind
        self.border_title = f"{kind} · {TITLES[kind]}"
        self._last_size = None

    def on_resize(self) -> None:
        if self.size != self._last_size:
            self._last_size = self.size
            self.app.redraw_panel(self)


SPEEDS = (1, 2, 4, 8, 16, 32, 64)  # playback, records per second
MAX_FPS = 8                         # above this, playback skips records


class LiveApp(App):
    TITLE = "kelp-live"
    CSS = """
    #bar { height: 1; background: $boost; }
    #bar Button { min-width: 5; margin: 0 1 0 0; }
    #status { width: 1fr; height: 1; padding: 0 1; color: $text; }
    #panels { grid-gutter: 0 1; padding: 0 1; }
    """
    BINDINGS = [
        Binding("q", "quit", "quit"),
        Binding("space,p", "play", "play/pause"),
        Binding("left_square_bracket", "speed(-1)", "slower", show=False),
        Binding("right_square_bracket", "speed(1)", "faster", show=False),
        Binding("P", "toggle_pause", "pause polling", show=False),
        Binding("f,end", "follow", "follow live"),
        Binding("comma", "step(-1)", "◀ record"),
        Binding("full_stop", "step(1)", "record ▶"),
        Binding("less_than_sign", "step(-10)", "◀◀", show=False),
        Binding("greater_than_sign", "step(10)", "▶▶", show=False),
        Binding("home", "goto_first", "first", show=False),
        Binding("plus,equals_sign", "zoom(1.5)", "zoom in"),
        Binding("minus", "zoom(0.6666667)", "zoom out"),
        Binding("0", "reset_view", "reset view"),
        Binding("left,h", "pan(-0.2, 0)", "pan", show=False),
        Binding("right,l", "pan(0.2, 0)", "pan", show=False),
        Binding("up,k", "pan(0, 0.2)", "pan", show=False),
        Binding("down,j", "pan(0, -0.2)", "pan", show=False),
        Binding("v", "cycle_var", "variable"),
        Binding("s", "save_png", "png"),
        Binding("escape,g", "grid", "grid"),
        Binding("a", "show_all", "all panels", show=False),
        *[Binding(str(i), f"solo({i})", f"panel {i}", show=False) for i in range(1, 9)],
    ]

    def __init__(self, state: State, view: View, show: list[int], poll: float = 2.0,
                 png_dir: str = "."):
        super().__init__()
        self.state, self.view = state, view
        self.grid_set = set(show)       # panels in the grid view
        self.solo: int | None = None    # one panel full screen
        self.poll_s = poll
        self.png_dir = png_dir
        self.k_view: int | None = None  # None = follow the newest record
        self.paused = False
        self.var: str | None = None
        self.frame = None
        self._loading = False
        self.playing = False
        self.speed_i = 2

    # -- layout ----------------------------------------------------------------
    @property
    def shown(self) -> set[int]:
        return {self.solo} if self.solo is not None else self.grid_set

    def compose(self) -> ComposeResult:
        with Horizontal(id="bar"):
            for label, bid, action, tip in (
                    ("|◀", "btn-first", "app.goto_first", "first record (home)"),
                    ("▶", "btn-play", "app.play", "play / pause replay (space)"),
                    ("▶|", "btn-live", "app.follow", "back to live (f)")):
                b = Button(label, id=bid, action=action, compact=True, tooltip=tip)
                b.can_focus = False
                yield b
            yield Static(id="status")
        with Grid(id="panels"):
            for k in range(1, 9):
                yield PanelWidget(k)
        yield Footer()

    def on_mount(self) -> None:
        self.relayout()
        self.set_interval(self.poll_s, self.poll)
        self.refresh_frame()

    def relayout(self) -> None:
        n = max(1, len(self.shown))
        ncol = 1 if n == 1 else 2 if n <= 4 else 3 if n <= 9 else 4
        grid = self.query_one("#panels", Grid)
        grid.styles.grid_size_columns = ncol
        grid.styles.grid_size_rows = math.ceil(n / ncol)
        for w in self.query(PanelWidget):
            w.display = w.kind in self.shown
        self.call_after_refresh(self.redraw)

    # -- data --------------------------------------------------------------------
    def poll(self) -> None:
        if not self.paused:
            self.update_worker()

    @work(thread=True, exclusive=True, group="io")
    def update_worker(self) -> None:
        if self.state.update():
            self.call_from_thread(self.on_new_data)
        elif self.state.error:
            self.call_from_thread(self.update_status)

    @work(thread=True, exclusive=True, group="load")
    def load_worker(self, k: int) -> None:
        try:
            self.state.load(k)
        except Exception as e:  # noqa: BLE001
            self.call_from_thread(self.notify, escape(f"reading record {k + 1}: {e}"), severity="error")
        self.call_from_thread(self._loaded)

    def _loaded(self) -> None:
        self._loading = False
        self.refresh_frame()

    def on_new_data(self) -> None:
        latest = self.state.latest
        if latest is not None and self.view.fit(latest["lon"], latest["lat"]):
            pass  # extent grew: backgrounds recomputed lazily on redraw
        if self.k_view is None:
            self.refresh_frame()
        else:
            self.update_status()

    def refresh_frame(self) -> None:
        fr = self.state.frame(self.k_view, self.var)
        if fr is None and self.k_view is not None:
            if not self._loading:
                self._loading = True
                self.load_worker(self.k_view)
            self.update_status(loading=True)
            return
        self.frame = fr
        self.redraw()

    # -- drawing -------------------------------------------------------------------
    def redraw(self) -> None:
        self.update_status()
        for w in self.query(PanelWidget):
            if w.display:
                self.redraw_panel(w)

    def redraw_panel(self, w: PanelWidget) -> None:
        if self.frame is None or self.view.data_extent is None:
            w.update("waiting for the first complete record …")
            return
        cols, rows = w.size.width, w.size.height
        if cols < 8 or rows < 3:
            return
        try:
            title, body = render_panel(w.kind, self.frame, self.view, cols, rows)
        except Exception as e:  # noqa: BLE001 - a panel must never take the app down
            title, body = f"{w.kind} · {TITLES[w.kind]}", Text(f"{type(e).__name__}: {e}", style="red")
        w.border_title = title
        w.update(body)

    def update_status(self, loading: bool = False) -> None:
        st, fr = self.state, self.frame
        f = st.feed
        if fr is None:
            txt = f"{escape(f.path)} · waiting for records"
        else:
            if self.playing:
                mode = f"[b cyan]PLAY {SPEEDS[self.speed_i]}/s[/]"
            elif self.k_view is None:
                mode = "[b green]LIVE[/]"
            else:
                mode = "[b yellow]SCRUB[/]"
            if self.paused:
                mode += " [b red]PAUSED[/]"
            txt = (f"{mode}  record {fr.k + 1}/{st.n_ready}"
                   f"{'/' + str(fr.n_expected) if fr.n_expected else ''}  ·  {f.label(fr.time)}  ·  "
                   f"{self.view.proj.name} ×{self.view.zoom:.1f}  ·  {escape(f.path)}")
        if loading:
            txt += "  · [i]loading …[/]"
        if st.error:
            txt += f"  · [red]{escape(st.error[:80])}[/]"
        self.query_one("#status", Static).update(txt)

    # -- actions ---------------------------------------------------------------------
    def action_toggle_pause(self) -> None:
        self.paused = not self.paused
        self.update_status()

    def action_follow(self) -> None:
        self._stop_play()
        self.k_view = None
        self.paused = False
        self.refresh_frame()

    # -- playback ---------------------------------------------------------------------
    def action_play(self) -> None:
        """Replay recorded history; from live it starts at the first record."""
        if self.playing:
            self._stop_play()
            return
        if self.state.n_ready < 2:
            return
        if self.k_view is None or self.k_view >= self.state.n_ready - 1:
            self.k_view = 0
        self.playing = True
        self._sync_play()
        self.refresh_frame()
        self.play_worker()

    def action_speed(self, d: int) -> None:
        self.speed_i = max(0, min(len(SPEEDS) - 1, self.speed_i + d))
        self.update_status()

    def _stop_play(self) -> None:
        if self.playing:
            self.playing = False
            self._sync_play()

    def _sync_play(self) -> None:
        self.query_one("#btn-play", Button).label = "||" if self.playing else "▶"
        self.update_status()

    @work(thread=True, exclusive=True, group="play")
    def play_worker(self) -> None:
        st = self.state
        while self.playing:
            t0 = time.monotonic()
            speed = SPEEDS[self.speed_i]
            fps = min(speed, MAX_FPS)
            step = max(1, round(speed / fps))
            cur = self.k_view
            if cur is None:
                break
            nxt = cur + step
            n = st.n_ready
            if nxt >= n - 1:
                self.call_from_thread(self._play_done)
                break
            if not st.has(nxt):
                ahead = max(1, min(16, st.cache_capacity() // 2))
                try:
                    st.load_records(range(nxt, min(n - 1, nxt + step * ahead), step))
                except Exception as e:  # noqa: BLE001
                    self.call_from_thread(self.notify, escape(f"replay: {e}"), severity="error")
                    self.call_from_thread(self._stop_play)
                    break
            if not self.playing:
                break
            self.call_from_thread(self._show, nxt)
            time.sleep(max(0.0, 1 / fps - (time.monotonic() - t0)))

    def _show(self, k: int) -> None:
        if self.playing:
            self.k_view = k
            self.refresh_frame()

    def _play_done(self) -> None:
        self.playing = False
        self.k_view = None
        self._sync_play()
        self.refresh_frame()
        self.notify("replay reached the newest record: live")

    def _goto(self, k: int) -> None:
        n = self.state.n_ready
        if n == 0:
            return
        k = max(0, min(n - 1, k))
        self.k_view = None if k == n - 1 else k
        self.refresh_frame()

    def action_step(self, d: int) -> None:
        self._stop_play()
        cur = self.k_view if self.k_view is not None else self.state.n_ready - 1
        self._goto(cur + d)

    def action_goto_first(self) -> None:
        self._goto(0)

    def action_zoom(self, f: float) -> None:
        self.view.zoom_by(f)
        self.redraw()

    def action_pan(self, dx: float, dy: float) -> None:
        self.view.pan(dx, dy)
        self.redraw()

    def action_reset_view(self) -> None:
        self.view.reset()
        self.redraw()

    def action_cycle_var(self) -> None:
        names = self.state.feed.sampled
        if not names:
            self.notify("no sampled variables in this store")
            return
        cur = self.var if self.var in names else default_var(names)
        self.var = names[(names.index(cur) + 1) % len(names)]
        self.notify(f"field: {self.var}" + ("" if 4 in self.shown else " (panel 4)"))
        self.refresh_frame()

    def action_solo(self, k: int) -> None:
        """Show panel k full screen; the same key again returns to the grid."""
        self.solo = None if self.solo == k else k
        self.relayout()

    def action_grid(self) -> None:
        self.solo = None
        self.relayout()

    def action_show_all(self) -> None:
        self.grid_set = set(range(1, 9))
        self.solo = None
        self.relayout()

    def action_save_png(self) -> None:
        if self.frame is None:
            return
        import os
        width = self.size.width
        rows = max(8, (self.size.height - 4) // math.ceil(len(self.shown) / 3))
        con = Console(width=width, file=io.StringIO(), color_system="truecolor")
        view = gallery(self.frame, self.view, sorted(self.shown), width, rows)
        path = os.path.join(self.png_dir, f"kelp-live_{self.frame.k + 1:05d}.png")
        save_png(con, view, path)
        self.notify(f"saved {path}")
