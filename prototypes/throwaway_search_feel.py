"""THROWAWAY PROTOTYPE — not part of session-browser, do not import from this.

Question: WHERE does the `reading X of Y` indicator physically go, given the
real geometry? The search box has only 33-43 usable columns at common
terminal sizes, shared with the typed query — the full phrase does not fit.

Geometry here is measured from the real app, not invented:

    terminal   left pane   text area inside the search box
      100 x30      41              33
      144 x40      51              43
      180 x45      65              57

Press `p` to cycle placement, `1`/`2`/`3` to change terminal size, `t` to
toggle a long query (stresses the in-box option).

Settled on earlier rungs: no streaming arrival, no split sections, single
merged list, indicator wording is two-phase (`scanning N sessions…` then
`reading X of Y transcripts` — only ~64 of 1,345 survive the rg prefilter).

    .venv/bin/python prototypes/throwaway_search_feel.py
"""

from __future__ import annotations

from textual.app import App, ComposeResult
from textual.widgets import Static

SPINNER = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"

# (terminal cols, left pane width, usable text width inside the search box)
GEOMETRY = {"1": (100, 41, 33), "2": (144, 51, 43), "3": (180, 65, 57)}

TOTAL_SESSIONS = 1345
CANDIDATES = 64

SHORT_Q = "ratelimit"
LONG_Q = "ratelimit backoff redis"

ROWS = [
    ("claude", "session-browser", "2h", "fix the ratelimit backoff in herdr"),
    ("codex", "herdr", "4h", "ratelimit retry budget tuning"),
    ("opencode", "edge-proxy", "9h", "ratelimit middleware for the edge"),
    ("claude", "api-gateway", "1d", "ratelimit notes"),
    ("opencode", "notes", "1d", "why is the ratelimit 429ing on warm"),
    ("claude", "infra", "2d", "ratelimit: move counters into redis"),
    ("codex", "billing", "3d", "draft ratelimit policy for partners"),
    ("claude", "herdr", "4d", "ratelimit dashboards are wrong again"),
    ("opencode", "web", "6d", "ratelimit + burst credit design"),
    ("claude", "warehouse", "8d", "remove the old ratelimit shim"),
    ("codex", "dotfiles", "11d", "ratelimit tests are flaky in CI"),
    ("claude", "scratch", "2w", "ratelimit headers spec review"),
]

PLACEMENTS = ["pane-title", "in-box", "under-box", "app-header"]


class Proto(App):
    CSS = """
    Screen { background: $surface; }
    #screen { padding: 1 2; }
    """
    BINDINGS = [
        ("p", "placement", "cycle placement"),
        ("1", "size1", "100 cols"),
        ("2", "size2", "144 cols"),
        ("3", "size3", "180 cols"),
        ("t", "toggle_query", "long/short query"),
        ("s", "search", "restart scan"),
        ("q", "quit", "quit"),
    ]

    def compose(self) -> ComposeResult:
        yield Static(id="screen")

    def on_mount(self) -> None:
        self.geo = "2"
        self.placement = 0
        self.long_query = False
        self.t = 0.0
        self.frame = 0
        self.set_interval(0.08, self.tick)

    def action_placement(self) -> None:
        self.placement = (self.placement + 1) % len(PLACEMENTS)

    def action_size1(self) -> None:
        self.geo = "1"

    def action_size2(self) -> None:
        self.geo = "2"

    def action_size3(self) -> None:
        self.geo = "3"

    def action_toggle_query(self) -> None:
        self.long_query = not self.long_query

    def action_search(self) -> None:
        self.t = 0.0

    def tick(self) -> None:
        self.frame += 1
        self.t = (self.t + 0.08) % 9.0    # loop the scan forever
        self.render_screen()

    # -- the indicator -----------------------------------------------------

    def phase(self) -> list[str]:
        """Forms for this point in the scan, longest first.

        Whichever placement is chosen takes the longest one that fits its
        column budget — that is the whole question at 100 columns.
        """
        spin = SPINNER[self.frame % len(SPINNER)]
        if self.t < 1.0:                       # rg prefilter phase
            return [f"{spin} scanning {TOTAL_SESSIONS:,} sessions…",
                    f"{spin} scanning {TOTAL_SESSIONS:,}…",
                    f"{spin} scanning…",
                    f"{spin}"]
        if self.t < 7.0:                       # reading candidates
            done = int(CANDIDATES * (self.t - 1.0) / 6.0)
            return [f"{spin} reading {done} of {CANDIDATES} transcripts",
                    f"{spin} reading {done} of {CANDIDATES}",
                    f"{spin} {done}/{CANDIDATES}",
                    f"{spin}"]
        return ["✓ +41 from transcripts", "✓ +41 found", "✓ +41", "✓"]

    # -- rendering ---------------------------------------------------------

    def render_screen(self) -> None:
        cols, pane_w, box_w = GEOMETRY[self.geo]
        query = LONG_Q if self.long_query else SHORT_Q
        forms = self.phase()
        long_form = forms[0]
        mode = PLACEMENTS[self.placement]
        right_w = cols - pane_w - 3

        meta = f"{TOTAL_SESSIONS:,} sessions  ·  all projects"
        if mode == "app-header":
            meta = f"[yellow]{_esc(long_form)}[/]  [dim]·  {TOTAL_SESSIONS:,} sessions[/]"

        out = [
            "[bold]search feel prototype[/]  [dim]throwaway — real measured "
            "geometry[/]",
            "",
            "[dim]" + "═" * cols + "[/]",
        ]

        # --- app bar ---
        brand = "[bold cyan]◉[/]  session browser"
        gap = max(1, cols - 20 - len(_plain(meta)))
        out.append(f"{brand}{' ' * gap}{meta}")
        out.append("[dim]" + "─" * cols + "[/]")

        # --- panes ---
        left: list[str] = []
        if mode == "pane-title":
            # budget: pane width, less "SESSIONS" and two spaces of gap
            fit = _best_fit(forms, pane_w - 8 - 2)
            title = f"[dim]SESSIONS[/]  [yellow]{_esc(fit)}[/]" if fit \
                else "[dim]SESSIONS[/]"
            left.append(title)
        else:
            left.append("[dim]SESSIONS[/]")
        left.append(self._search_box(box_w, query, forms, mode))
        if mode == "under-box":
            left.append(f" [yellow]{_esc(_fit(long_form, box_w))}[/]")
        for prov, proj, age, summary in ROWS:
            left.append(" " + _row(prov, proj, age, summary, pane_w - 1))

        right = ["[dim]TRANSCRIPT[/]", "[dim]" + "·" * (right_w - 2) + "[/]"]
        right += ["[dim]" + "·" * (right_w - 2) + "[/]"] * 3

        for i in range(max(len(left), len(right))):
            l = left[i] if i < len(left) else ""
            r = right[i] if i < len(right) else ""
            pad = max(0, pane_w - len(_plain(l)))
            out.append(f"{l}{' ' * pad} [dim]│[/] {r}")

        out.append("[dim]" + "═" * cols + "[/]")
        out += [
            "",
            f"  placement: [bold yellow]{mode}[/]   "
            f"terminal: [bold]{cols}[/] cols   "
            f"box text width: [bold]{box_w}[/]   "
            f"query: [bold]{len(query)}[/] chars",
            f"  [dim]full phrase is {len(_plain(long_form))} chars; "
            f"pane-title budget is {pane_w - 10}, in-box budget is "
            f"{box_w - len(query) - 2}[/]",
            "",
            "  [cyan]p[/] placement   [cyan]1[/]/[cyan]2[/]/[cyan]3[/] terminal "
            "100/144/180   [cyan]t[/] long query   [cyan]s[/] restart   "
            "[cyan]q[/] quit",
        ]
        self.query_one("#screen", Static).update("\n".join(out))

    def _search_box(self, box_w: int, query: str, forms: list[str],
                    mode: str) -> str:
        """The input, with the in-box indicator overlaid at its right edge."""
        inner = query
        if mode == "in-box":
            room = box_w - len(query) - 2
            short = _best_fit(forms, room) or forms[-1]
            if room >= len(_plain(short)):
                pad = box_w - len(query) - len(_plain(short)) - 1
                inner = (f"{_esc(query)}[reverse] [/]{' ' * max(0, pad - 1)}"
                         f"[yellow]{_esc(short)}[/]")
            else:
                # no room — this is the failure mode worth seeing
                inner = (f"{_esc(query)}[reverse] [/]"
                         f"{' ' * max(0, box_w - len(query) - 4)}"
                         f"[red]▮▮▮[/]")
        else:
            inner = f"{_esc(query)}[reverse] [/]{' ' * (box_w - len(query) - 1)}"
        return f"[dim]╭[/]{inner}[dim]╮[/]"


PROV = {"claude": "[magenta]claude  [/]", "codex": "[cyan]codex   [/]",
        "opencode": "[green]opencode[/]"}


def _esc(s: str) -> str:
    return s.replace("[", "\\[")


def _plain(s: str) -> str:
    import re
    return re.sub(r"\[[^\]]*\]", "", s)


def _fit(s: str, w: int) -> str:
    return s if len(s) <= w else s[: w - 1] + "…"


def _best_fit(forms: list[str], width: int) -> str:
    """Longest form that fits `width`, or "" if even the shortest doesn't."""
    for f in forms:
        if len(_plain(f)) <= width:
            return f
    return ""


def _row(prov: str, proj: str, age: str, summary: str, w: int) -> str:
    head = f"{PROV[prov]} [dim]{age:>3}[/]  "
    room = max(4, w - 8 - 1 - 3 - 2)
    return head + _esc(_fit(summary, room))


if __name__ == "__main__":
    Proto().run()
