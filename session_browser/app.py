#!/usr/bin/env python3
"""Lightweight agent session browser TUI."""

from __future__ import annotations

import os
import re
import subprocess
import sys
from bisect import bisect_left
from collections.abc import Sequence
from functools import wraps
from typing import ClassVar

try:
    from rich.segment import Segment
    from textual import events
    from textual.app import App, ComposeResult
    from textual.binding import Binding
    from textual.containers import Horizontal, Vertical, VerticalScroll
    from textual.content import Content
    from textual.css.query import NoMatches
    from textual.message import Message
    from textual.screen import ModalScreen
    from textual.strip import Strip
    from textual.widgets import DataTable, Footer, Input, Label, Static
    from textual.worker import Worker, WorkerState, get_current_worker
except ImportError:
    print("textual is required: pip install 'textual>=0.80.0'")
    sys.exit(1)

from datetime import UTC

from . import multiplexer
from .discovery import Session, discover_all
from .resume import (
    build_chat_export,
    copy_to_clipboard,
    export_chat_to_file,
    resume_command,
)
from .transcript import (
    ContentHit,
    ContentSearchCache,
    Transcript,
    TranscriptEntry,
    TranscriptUnreadable,
    _offset_map,
    _OffsetMap,
    canonical_id,
    entry_label,
    find_text_spans,
    load_transcript,
    normalize_match_text,
    render_text,
    search_session_hits,
)

# ---------------------------------------------------------------------------
# Provider colour tags for the list
# ---------------------------------------------------------------------------

PROVIDER_COLOURS = {
    "claude": "[bold magenta]●[/] claude",
    "codex": "[bold green]●[/] codex",
    "opencode": "[bold cyan]●[/] opencode",
    "pi": "[bold yellow]●[/] pi",
}

# Coalesce bursts of keystrokes in the global search before filtering, so a
# content-search worker isn't launched on every character. Short enough to feel
# instant.
_FILTER_DEBOUNCE = 0.15

# The detail pane renders at most this many characters at once. This is
# strictly a presentation bound: search, match counts, and exports always
# see the full transcript, and the window slides to the selected match.
_DISPLAY_WINDOW = 200_000

# Content search takes ~1s on a large history — too quick to read partial
# results, long enough that the screen shouldn't sit frozen. So we hold the
# reveal until every match is in and spin this indicator meanwhile.
_SPINNER_FRAMES = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"
_SPINNER_INTERVAL = 0.08
# How long "✓ +N from transcripts" stays up after the second wave lands.
_DONE_FLASH_DURATION = 2.5

# Global shortcuts bound to printable keys: suppressed while a search box has
# focus, or the keystroke types a letter *and* fires the action. See
# SessionBrowser.check_action.
_EDIT_BLOCKED_ACTIONS = frozenset(
    {
        "quit",
        "focus_detail_search",
        "copy_resume",
        "copy_session_id",
        "open_terminal",
        "export_chat_clipboard",
        "export_chat_file",
        "next_match",
        "prev_match",
        "refresh_sessions",
        "toggle_here",
        "toggle_focus_mode",
        "show_help",
    }
)


def _progress_forms(
    glyph: str, progress: dict[str, tuple[int, int]] | None
) -> list[str]:
    """Wordings for the search indicator, widest first.

    Phases rather than one bar, because the work is genuinely several things
    and they do not run one after another: the file prefilter rules out most
    of the corpus in a fraction of a second, the survivors are read, and the
    conversation database is scanned *concurrently* with all of that, often
    outlasting it by seconds.

    So the phases are held together and the live one is chosen here, rather
    than letting whichever thread wrote last own the display — two counters
    taking turns in one line reads as a fault, not as progress.
    """
    if not progress:
        return []

    def ratio(phase: str) -> tuple[int, int] | None:
        got = progress.get(phase)
        return got if got and got[1] > 0 else None

    if found := ratio("indexing"):
        done, total = found
        return [
            f"{glyph} indexing {done:,} of {total:,} results",
            f"{glyph} indexing {done:,} of {total:,}",
            f"{glyph} indexing {done:,}",
            glyph,
        ]
    # Reading owns the line until it finishes; the database scan then takes
    # over, because it is what the search is still waiting on.
    reading = ratio("reading")
    if reading and reading[0] < reading[1]:
        done, total = reading
        return [
            f"{glyph} reading {done:,} of {total:,} transcripts",
            f"{glyph} reading {done:,} of {total:,}",
            f"{glyph} {done:,}/{total:,}",
            glyph,
        ]
    if "conversations" in progress:
        done, total = progress["conversations"]
        if total > 0:
            return [
                f"{glyph} reading {done:,} of {total:,} conversations",
                f"{glyph} reading {done:,} of {total:,} convos",
                f"{glyph} {done:,}/{total:,} convos",
                glyph,
            ]
        # The database prefilter is one long scan with no session boundaries
        # to count, so show the entries consumed: an unbounded number that
        # keeps moving still proves the search is alive, which is the job.
        return [
            f"{glyph} scanning conversations… {done:,}",
            f"{glyph} conversations… {done:,}",
            f"{glyph} conversations…",
            glyph,
        ]
    if reading:
        done, total = reading
        return [
            f"{glyph} reading {done:,} of {total:,} transcripts",
            f"{glyph} reading {done:,} of {total:,}",
            f"{glyph} {done:,}/{total:,}",
            glyph,
        ]
    if found := ratio("scanning"):
        return [
            f"{glyph} scanning {found[1]:,} sessions…",
            f"{glyph} scanning {found[1]:,}…",
            f"{glyph} scanning…",
            glyph,
        ]
    return [f"{glyph} searching…", f"{glyph} search…", glyph]


# ---------------------------------------------------------------------------
# CSS
# ---------------------------------------------------------------------------

APP_CSS = """\
Screen {
    background: $background;
}
#app-bar {
    height: 2;
    background: $panel;
    border-bottom: solid $accent;
}
#brand {
    width: auto;
    padding: 0 1;
    content-align: left middle;
    text-style: bold;
}
#app-meta {
    width: 1fr;
    padding: 0 1;
    content-align: right middle;
    text-align: right;
    color: $text-muted;
}
#main {
    height: 1fr;
    padding: 0 1;
}
#left-pane {
    width: 45%;
    min-width: 36;
    margin-right: 1;
    background: $surface;
    border: round $border;
}
#right-pane {
    width: 55%;
    min-width: 0;
    background: $surface;
    border: round $border;
}
#left-pane:focus-within, #right-pane:focus-within {
    border: round $accent;
    background-tint: $accent 3%;
}
.pane-title {
    height: 1;
    padding: 0 1;
    color: $text-muted;
    text-style: bold;
}
#sessions-header {
    height: 1;
}
#sessions-title {
    width: auto;
}
#sessions-status {
    width: 1fr;
    padding: 0 1;
    content-align: right middle;
}
#global-search {
    margin: 0 1;
}
#session-table {
    height: 1fr;
    border: none;
}
#detail-header {
    height: auto;
    max-height: 5;
    padding: 0 1;
    border-bottom: solid $border;
    overflow: hidden;
}
#detail-search {
    margin: 0 1;
}
/* Reserve border space on every focusable widget so that gaining/losing
   focus only changes the border *colour*, not the layout.  Without this
   the tall border appears/disappears and the whole pane shifts by 2 rows. */
SearchInput {
    border: round $border;
}
/* Highlight the focused search field so it's obvious where keystrokes go. */
SearchInput:focus {
    border: round $accent;
    background: $boost;
}
#detail-scroll {
    height: 1fr;
    border: none;
}
#detail-content {
    height: auto;
    padding: 0 1;
}
.transcript-entry {
    height: auto;
    margin: 0 1 1 1;
    padding: 0 1;
    border-left: solid gray;
    background: $panel;
}
.transcript-entry:focus {
    background: $boost;
}
.transcript-entry.-user {
    border-left: solid $primary;
}
.transcript-entry.-assistant {
    border-left: solid $success;
}
.transcript-entry.-tool-call {
    border-left: solid $warning;
}
.transcript-entry.-tool-output {
    border-left: solid gray;
    color: $text-muted;
}
.transcript-entry.-system {
    border-left: solid $secondary;
    color: $text-muted;
}
.transcript-gap {
    height: auto;
    margin: 0 1 1 1;
    padding: 0 1;
    color: $text-muted;
}
#match-counter {
    height: 0;
    text-align: right;
    color: $text-muted;
    padding: 0 1;
}
#match-counter.-visible {
    height: 1;
}
/* No dock here: the built-in Footer already docks to the bottom edge, and
   same-edge docks overlap — a docked status bar renders underneath the Footer
   and is never seen. As a flow child it gets the row above the Footer. */
#status-bar {
    height: 1;
    padding: 0 1;
    background: $panel;
    color: $text-muted;
}
/* Briefly pulse the bar on copy/confirm actions — it's otherwise muted and
   easy to miss. `auto` picks a contrasting foreground for the fill. */
#status-bar.-flash-ok {
    background: $success;
    color: auto;
    text-style: bold;
}
#status-bar.-flash-err {
    background: $error;
    color: auto;
    text-style: bold;
}
.detail-field {
    height: auto;
}
.detail-label {
    color: $text-muted;
}

/* Wide terminals become a reading canvas: keep the session browser useful,
   but cap it before it steals space from the transcript. */
Screen.-wide #left-pane {
    width: 38%;
    min-width: 48;
    max-width: 76;
}
Screen.-wide #right-pane {
    width: 1fr;
}

/* Narrow terminals are a true master/detail flow. Only one pane is mounted
   visibly at a time, so neither side is crushed into an unusable sliver. */
Screen.-micro #left-pane,
Screen.-compact #left-pane,
Screen.-micro #right-pane,
Screen.-compact #right-pane {
    width: 100%;
    min-width: 0;
    max-width: 100%;
    margin-right: 0;
}
Screen.-micro.-show-sessions #right-pane,
Screen.-compact.-show-sessions #right-pane,
Screen.-micro.-show-detail #left-pane,
Screen.-compact.-show-detail #left-pane {
    display: none;
}

/* z toggles this manual focus view at any non-compact width. Pane switching
   continues to work and moves the full-width view with focus. */
Screen.-focus-left #left-pane,
Screen.-focus-right #right-pane {
    display: block;
    width: 100%;
    min-width: 0;
    max-width: 100%;
    margin-right: 0;
}
Screen.-focus-left #right-pane,
Screen.-focus-right #left-pane {
    display: none;
}

/* Height is information architecture too: a short Magnet-sized window keeps
   rows and transcript content, while decorative chrome compresses away. */
Screen.-short #app-bar {
    height: 1;
    border-bottom: none;
}
Screen.-short #app-meta {
    display: none;
}
Screen.-short #main {
    padding: 0;
}
Screen.-short .pane-title {
    display: none;
}
Screen.-short #sessions-header {
    display: none;
}
Screen.-short SearchInput {
    height: 1;
    margin: 0 1;
    border: none;
}
Screen.-short SearchInput:focus {
    border: none;
    background: $boost;
}
Screen.-short #detail-header {
    max-height: 2;
    padding: 0 1;
}
Screen.-short Footer,
Screen.-micro Footer {
    display: none;
}
"""


def _relative_time(iso_str: str) -> str:
    """Human-friendly relative time like '2h ago' or '3d ago'."""
    if not iso_str:
        return "—"
    from datetime import datetime

    try:
        # Handle multiple ISO formats
        ts = iso_str.replace("Z", "+00:00")
        if "T" not in ts:
            return iso_str[:10]
        dt = datetime.fromisoformat(ts)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=UTC)
        now = datetime.now(UTC)
        delta = now - dt
        seconds = int(delta.total_seconds())
        if seconds < 60:
            return "just now"
        if seconds < 3600:
            return f"{seconds // 60}m ago"
        if seconds < 86400:
            return f"{seconds // 3600}h ago"
        days = seconds // 86400
        if days < 30:
            return f"{days}d ago"
        return dt.strftime("%Y-%m-%d")
    except Exception:
        return iso_str[:16] if iso_str else "—"


def _escape_markup(text: str) -> str:
    """Escape session content so it can sit inside app markup safely.

    Every "[" is escaped: Textual's content-markup grammar is far looser
    than Rich's — `[Session(id=...)]` parses as a tag — and neither
    rich.markup.escape nor textual.markup.escape (both only escape
    lowercase/#/@ tag starts) protects Static widgets from such content.

    A segment ending in a backslash would escape the bracket of a tag
    appended right after it (`…\\[/]` renders the tag as literal text and a
    later close tag then crashes the app with "nothing to close"). Neither
    markup dialect has an escape for backslash itself, so such segments get
    one trailing space, invisible in the terminal, to keep the tag intact.
    """
    escaped = text.replace("[", r"\[")
    if escaped.endswith("\\"):
        escaped += " "
    return escaped


def _project_name(cwd: str, max_len: int = 18) -> str:
    """Directory basename identifying the session's project, or '—'."""
    if not cwd:
        return "—"
    name = os.path.basename(cwd.rstrip(os.sep)) or cwd
    if len(name) > max_len:
        name = name[: max_len - 1] + "…"
    return name


def _ellipsize(text: str, max_len: int) -> str:
    """Fit user-controlled text on one terminal line."""
    clean = " ".join(text.split())
    if len(clean) <= max_len:
        return clean
    return clean[: max(1, max_len - 1)] + "…"


def _collapse_ws(text: str) -> str:
    """One-line whitespace collapse that keeps boundary spaces intact."""
    return re.sub(r"\s+", " ", text)


def _format_content_hit(hit: ContentHit) -> str:
    """Summary-cell markup for a content match: count + highlighted snippet.

    The session's opening summary would actively mislead here — the row is
    in the list because of *this* text, so show it, with the match itself
    highlighted the way the transcript pane highlights it.
    """
    before = _collapse_ws(hit.before)[-40:]
    match = _ellipsize(hit.match, 60)
    after = _collapse_ws(hit.after)[:60]
    count = f"[dim]{hit.count}× [/]" if hit.count > 1 else ""
    prefix = "[dim]…[/]" if hit.before else ""
    suffix = "[dim]…[/]" if hit.after else ""
    return (
        f"{count}{prefix}{_escape_markup(before)}"
        f"[black on yellow]{_escape_markup(match)}[/]"
        f"{_escape_markup(after)}{suffix}"
    )


def _teardown_safe(method):
    """Absorb widget queries that land after unmount.

    Timer callbacks and worker completions are delivered through the message
    loop, so they can fire while the app is tearing down — after the widgets
    they update have been unmounted, when any query raises NoMatches. Every
    async re-entry point that touches the DOM wears this guard; synchronous
    action/handler paths don't need it (they only run while mounted, and
    there a NoMatches is a real bug worth crashing on).
    """

    @wraps(method)
    def wrapper(self, *args, **kwargs):
        try:
            return method(self, *args, **kwargs)
        except NoMatches:
            return None

    return wrapper


# ---------------------------------------------------------------------------
# Widgets
# ---------------------------------------------------------------------------


class SearchInput(Input):
    """Input with vim/readline/fzf-style keybindings.

    - Escape: clear a populated search; from an empty global search, move back
      to the session list.
    - Ctrl+U: clear immediately (readline).
    - Down arrow: jump out of the field into the list / detail pane below it.
    - Ctrl+N / Ctrl+P: move the session-list cursor without leaving the field
      (fzf convention — keep typing to refine, keep ^N/^P to browse).
    """

    BINDINGS: ClassVar[list[Binding]] = [
        Binding("escape", "escape_input", "Clear / back", show=False),
        Binding("ctrl+u", "clear_input", "Clear", show=False),
        Binding("down", "focus_below", "Into list", show=False),
        Binding("ctrl+n", "list_cursor_down", "Next result", show=False),
        # priority=True so we win over Textual's default ctrl+p (command palette).
        Binding("ctrl+p", "list_cursor_up", "Prev result", show=False, priority=True),
    ]

    COMPONENT_CLASSES = Input.COMPONENT_CLASSES | {"search-input--spinner"}
    DEFAULT_CSS = """
    SearchInput > .search-input--spinner {
        color: $warning;
    }
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.spinner_glyph: str | None = None

    def set_spinner(self, glyph: str | None) -> None:
        self.spinner_glyph = glyph
        self.refresh()

    def render_line(self, y: int) -> Strip:
        strip = super().render_line(y)
        visible_width = self.scrollable_content_region.width
        if y != 0 or self.spinner_glyph is None or visible_width == 0:
            return strip
        spinner = Strip(
            [
                Segment(
                    self.spinner_glyph,
                    self.get_component_rich_style("search-input--spinner"),
                )
            ],
            1,
        )
        return Strip.join(
            (
                strip.crop(0, visible_width - 1),
                spinner,
                strip.crop(visible_width),
            )
        )

    def action_escape_input(self) -> None:
        if self.id == "global-search":
            if self.value:
                self.app.action_clear_global_search()
                return
            # The empty global search and session list form a two-way Escape
            # focus toggle.
            self.app.action_focus_session_list()
            return
        if self.value:
            self.value = ""
            return
        # Detail search: empty-Escape blurs to the list.
        self.app.action_focus_session_list()

    def action_clear_input(self) -> None:
        self.value = ""

    def action_focus_below(self) -> None:
        """Move focus out of the search field into the pane below it."""
        if self.id == "detail-search":
            scroll = self.app.query_one("#detail-scroll", VerticalScroll)
            scroll.focus()
        else:
            self.app.query_one("#session-table", DataTable).focus()

    def _step_list(self, delta: int) -> None:
        table = self.app.query_one("#session-table", DataTable)
        if not table.row_count:
            return
        new_row = max(0, min(table.cursor_row + delta, table.row_count - 1))
        table.move_cursor(row=new_row)

    def action_list_cursor_down(self) -> None:
        self._step_list(1)

    def action_list_cursor_up(self) -> None:
        self._step_list(-1)


class ResponsiveFooter(Footer):
    """Use labels when they fit and keycaps when they don't."""

    def on_resize(self, event: events.Resize) -> None:
        self.compact = event.size.width < 144
        # Screen consumes the terminal Resize event after applying breakpoint
        # classes, so this full-width child is our reliable viewport observer.
        self.app.call_after_refresh(self.app._sync_responsive_chrome)


class SessionTable(DataTable):
    """DataTable that hops back to the search field when cursor-up at row 0."""

    class ColumnsChanged(Message):
        """The list crossed a width threshold and needs different columns."""

        def __init__(self, mode: str) -> None:
            self.mode = mode
            super().__init__()

    class RebuildSettled(Message):
        """Marks the end of a rebuild's RowHighlighted echoes.

        DataTable posts every rebuild-induced RowHighlighted synchronously
        (from clear/add_row/move_cursor), so a message posted onto the table's
        queue right after a rebuild bubbles up to the app strictly after those
        echoes. Lifting highlight suppression on this signal is deterministic,
        unlike call_after_refresh, which races the message queue.
        """

        def __init__(self, generation: int) -> None:
            self.generation = generation
            super().__init__()

    def action_cursor_up(self) -> None:
        if self.row_count and self.cursor_row == 0:
            self.app.query_one("#global-search", Input).focus()
            return
        super().action_cursor_up()

    def action_cursor_right(self) -> None:
        """Right crosses into the detail pane instead of moving columns."""
        self.app.action_focus_right_pane()

    def on_resize(self, event: events.Resize) -> None:
        width = event.size.width
        mode = "minimal" if width < 42 else "compact" if width < 62 else "full"
        if mode != getattr(self, "_column_mode", None):
            self._column_mode = mode
            self.post_message(self.ColumnsChanged(mode))


class TranscriptEntryWidget(Static):
    """One structured transcript entry with display-only tool collapsing."""

    can_focus = True
    COLLAPSE_AT = 500
    PREVIEW_CHARS = 160
    BINDINGS: ClassVar[list[Binding]] = [
        Binding("enter", "toggle_collapsed", "Expand / collapse", show=False),
    ]

    def __init__(self, entry: TranscriptEntry, index: int, text_start: int) -> None:
        self.entry = entry
        self.entry_index = index
        self.text_start = text_start
        self._query = ""
        self._match_spans: list[tuple[int, int]] = []
        self._active_match_span: tuple[int, int] | None = None
        self._active_index: int | None = None
        # Calls as well as outputs. Both are unbounded -- measured over the
        # file-backed corpus on 2026-08-25, 10,513 of 30,689 tool calls run
        # past this threshold, 202 past 10,000 characters and the largest to
        # 92,248 -- and a call that arrives already expanded floods the pane
        # exactly as an output would.
        self.collapsible = entry.role == "tool" and len(entry.text) > self.COLLAPSE_AT
        # Nothing but ``action_toggle_collapsed`` ever writes this again, which
        # is what keeps an explicit choice explicit: navigation moves the
        # preview inside a collapsed block instead of unfolding it, so a block
        # the reader expanded stays expanded and one they folded stays folded.
        self.collapsed = self.collapsible
        classes = ["transcript-entry", f"-{entry.role}"]
        if entry.role == "tool":
            kind = (entry.metadata or {}).get("kind")
            classes.append("-tool-call" if kind == "call" else "-tool-output")
        super().__init__("", classes=" ".join(classes))

    def on_mount(self) -> None:
        self._refresh_content()

    def set_matches(
        self,
        query: str,
        absolute_spans: list[tuple[int, int]],
        active_absolute_span: tuple[int, int] | None = None,
    ) -> None:
        self._query = query
        self._match_spans = [
            (start - self.text_start, end - self.text_start)
            for start, end in absolute_spans
        ]
        self._active_match_span = (
            (
                active_absolute_span[0] - self.text_start,
                active_absolute_span[1] - self.text_start,
            )
            if active_absolute_span is not None
            else None
        )
        # The active match belongs to at most one widget, and the span alone
        # cannot say which: every widget is handed the same one. Resolving it
        # to a position in this entry's own spans answers both questions at
        # once -- whether it is here, and which of the matches here it is.
        self._active_index = None
        if self._active_match_span is not None:
            at = bisect_left(self._match_spans, self._active_match_span)
            if (
                at < len(self._match_spans)
                and self._match_spans[at] == self._active_match_span
            ):
                self._active_index = at
        self._refresh_content()

    def action_toggle_collapsed(self) -> None:
        if self.collapsible:
            self.collapsed = not self.collapsed
            self._refresh_content()

    def on_click(self) -> None:
        # Textual synthesizes Click whenever mouse-down and mouse-up land on
        # the same widget, even when the pointer moved to select text.  The
        # completed selection is available by the time Click is dispatched,
        # so copy it and leave the entry's layout untouched.
        if selected_text := self.screen.get_selected_text():
            self.app.copy_to_clipboard(selected_text)
            return
        self.action_toggle_collapsed()

    def _visible_text(self) -> tuple[str, int, str]:
        """Displayed slice of the entry, its offset into it, and a footer.

        A collapsed entry that *contains matches* previews one of them rather
        than its opening line: a match hidden behind the fold renders nothing
        at all, which reads as "search missed it" even though the counter
        includes it and n/N reaches it. Slicing around the match keeps the
        preview the same size, so collapsing still costs one short string per
        entry however large the tool block is.

        The footer says which match is on screen and how much is not, because
        a preview that silently showed the first of eight would be a smaller
        lie than the opening line but a lie of the same kind.
        """
        text = self.entry.text
        if not self.collapsed:
            return text, 0, ""
        start, end = self._preview_bounds(text)
        hidden = max(0, len(text) - (end - start))
        showing = (
            f" · showing match {(self._active_index or 0) + 1}"
            f" of {len(self._match_spans)}"
            if self._match_spans
            else ""
        )
        return (
            text[start:end],
            start,
            f"\n[dim]… {hidden:,} chars hidden{showing} · Enter/click to expand[/]",
        )

    def _preview_bounds(self, text: str) -> tuple[int, int]:
        """Bounds of the collapsed preview: the active match, else the first.

        Following the active match is what lets n/N traverse a block without
        unfolding it. The spans are already known, so choosing between them
        is an index; both branches below are then pure C-level string work --
        no scan over the entry in Python -- so a collapsed megabyte of tool
        output costs the same per navigation step whether or not the query hit
        it, and the same however many times it did.
        """
        if not self._match_spans:
            line_end = text.find("\n")
            if line_end == -1:
                line_end = len(text)
            return 0, min(line_end, self.PREVIEW_CHARS)
        match_start, match_end = self._match_spans[self._active_index or 0]
        # Stay inside the match's own line: a preview that ran on through
        # newlines would grow the collapsed widget by an unbounded number of
        # rendered rows.
        line_start = text.rfind("\n", 0, match_start) + 1
        line_end = text.find("\n", match_start)
        if line_end == -1:
            line_end = len(text)
        start = max(line_start, match_start - self.PREVIEW_CHARS // 3)
        end = min(line_end, start + self.PREVIEW_CHARS)
        if end < match_end:
            # A match longer than the preview: show it from its own start.
            start = match_start
            end = min(line_end, start + self.PREVIEW_CHARS)
        return start, end

    def _highlighted_text(self, text: str, offset: int = 0) -> str:
        """Escape *text*, which begins at *offset* within the entry, and wrap
        the match spans that fall inside it."""
        if not self._query or not self._match_spans:
            return _escape_markup(text)
        active = self._active_match_span
        if active is not None and offset:
            active = (active[0] - offset, active[1] - offset)
        parts: list[str] = []
        previous = 0
        # Walk only the spans that can land inside the slice. Collapsed, that
        # is a handful out of however many the entry holds; a Python loop over
        # every match of a common term in a multi-megabyte block would be the
        # per-navigation cost the preview exists to avoid. Indices rather than
        # a slice of the list, so the spans are not copied either.
        first = bisect_left(self._match_spans, (offset, 0))
        last = bisect_left(self._match_spans, (offset + len(text), 0))
        for index in range(first, last):
            start, end = self._match_spans[index]
            start -= offset
            end -= offset
            if start < previous or end > len(text) or end <= start:
                continue
            parts.append(_escape_markup(text[previous:start]))
            style = (
                "bold reverse yellow" if (start, end) == active else "black on yellow"
            )
            parts.append(f"[{style}]{_escape_markup(text[start:end])}[/]")
            previous = end
        parts.append(_escape_markup(text[previous:]))
        return "".join(parts)

    def _header_markup(self) -> str:
        timestamp = (
            f"  [dim]{_relative_time(self.entry.timestamp)}[/]"
            if self.entry.timestamp
            else ""
        )
        label = entry_label(self.entry)
        tool = (self.entry.metadata or {}).get("tool")
        if tool and self.entry.role == "tool":
            label = f"{label}: {tool}"
        count = len(self._match_spans)
        matches = (
            f"  [dim]· {count} match{'' if count == 1 else 'es'}[/]" if count else ""
        )
        return f"[bold]{_escape_markup(label)}[/]{matches}{timestamp}"

    def active_match_row(self) -> int | None:
        """Rendered row containing the active match within this widget.

        Measured against what is actually on screen. Collapsed, that is the
        preview -- so the height is computed over a bounded string rather than
        over the megabyte behind it, and the answer describes the rows the
        reader can see rather than the ones they cannot.
        """
        if self._active_match_span is None or self.size.width <= 0:
            return None
        text, offset, _ = self._visible_text()
        start = self._active_match_span[0] - offset
        if not 0 <= start <= len(text):
            return None
        elision = "[dim]…[/] " if offset else ""
        prefix = f"{self._header_markup()}\n{elision}{_escape_markup(text[:start])}"
        return max(
            0,
            Content.from_markup(prefix).get_height(self.styles, self.size.width) - 1,
        )

    def _refresh_content(self) -> None:
        text, offset, suffix = self._visible_text()
        prefix = "[dim]…[/] " if offset else ""
        body = self._highlighted_text(text, offset)
        self.update(f"{self._header_markup()}\n{prefix}{body}{suffix}")


class TranscriptGapWidget(Static):
    """A run of blocks the matching-only projection left out.

    It exists to stop the projection lying by omission. Two entries an hour
    apart, rendered adjacent with nothing between them, read as consecutive
    conversation; a marker saying how many blocks are missing is the
    difference between a filtered transcript and a misleading one.
    """

    def __init__(self, hidden: int) -> None:
        blocks = "block" if hidden == 1 else "blocks"
        super().__init__(
            f"[dim]⋯ {hidden:,} non-matching {blocks} hidden[/]",
            classes="transcript-gap",
        )


class ShortcutHelp(ModalScreen[None]):
    """Small, responsive key map kept out of the everyday footer."""

    CSS = """
    ShortcutHelp {
        align: center middle;
        background: $background;
    }
    #shortcut-help {
        width: 72;
        max-width: 94%;
        height: 23;
        max-height: 94%;
        padding: 1 2;
        background: $surface;
        border: round $accent;
    }
    """

    BINDINGS: ClassVar[list[Binding]] = [
        Binding("escape", "close_help", "Close", show=False),
        Binding("?", "close_help", "Close", show=False),
        Binding("q", "close_help", "Close", show=False),
    ]

    HELP = """\
[bold cyan]session browser[/]  [dim]keyboard map[/]

[bold]Move[/]       [cyan]j/k ↑/↓[/] rows or scroll   [cyan]gg / G[/] top / bottom
           [cyan]tab ← →[/] switch pane     [cyan]J/K[/] transcript entries
           [cyan]\\[ / ][/] previous / next tool output

[bold]Find[/]       [cyan]/[/] search active pane    [cyan]n / N[/] next / previous match
           [cyan]s[/] find in session       [cyan]m[/] matching blocks only
           [cyan]p[/] this-project scope

[bold]Layout[/]     [cyan]z[/] focus current pane    [cyan]Esc[/] step back

[bold]Use[/]        [cyan]c[/] copy resume command  [cyan]i[/] copy canonical id
           {terminal}[cyan]e / E[/] copy / export chat
           [cyan]r[/] refresh              [cyan]q[/] quit

[dim]On narrow screens, Enter or → opens the transcript full-width;
Esc or ← returns to sessions. Press ? or Esc to close.[/]"""

    # Visible width of the left column in the Use block. The ``t`` entry is
    # padded to it so the second column stays in line with the static rows
    # above and below whichever multiplexers it has to name.
    _USE_COLUMN = 23

    def __init__(self, terminal_names: Sequence[str] = ()) -> None:
        """``terminal_names`` are the multiplexers currently running; the ``t``
        entry names those, or reports that there is nowhere to hand off to."""
        super().__init__()
        self._terminal_names = list(terminal_names)

    def compose(self) -> ComposeResult:
        help_text = self.HELP.format(terminal=self._terminal_entry())
        yield VerticalScroll(Static(help_text), id="shortcut-help")

    def _terminal_entry(self) -> str:
        """The ``t`` entry, naming only the multiplexers actually running.

        With none running the key does nothing, so it is shown dimmed with the
        reason rather than silently omitted — a missing row reads as a bug,
        while "no live multiplexer" is the actual state. Pressing the key names
        the ones it looked for, so the row does not have to.
        """
        if self._terminal_names:
            text = f"t open in {' / '.join(self._terminal_names)}"
            markup = f"[cyan]t[/] open in {' / '.join(self._terminal_names)}"
        else:
            text = "t no live multiplexer"
            markup = f"[dim]{text}[/]"
        return markup + " " * max(1, self._USE_COLUMN - len(text))

    def action_close_help(self) -> None:
        self.dismiss()


class MultiplexerChoice(ModalScreen["multiplexer.Target | None"]):
    """Ask which multiplexer to hand a session to, when both are running."""

    CSS = """
    MultiplexerChoice {
        align: center middle;
        background: $background;
    }
    #multiplexer-choice {
        width: 44;
        max-width: 94%;
        height: auto;
        padding: 1 2;
        background: $surface;
        border: round $accent;
    }
    """

    BINDINGS: ClassVar[list[Binding]] = [
        Binding("escape", "cancel", "Cancel", show=False),
        Binding("q", "cancel", "Cancel", show=False),
    ]

    def __init__(self, targets: Sequence[multiplexer.Target]) -> None:
        super().__init__()
        self._targets = list(targets)

    def compose(self) -> ComposeResult:
        lines = [f"[cyan]{t.key}[/] {t.name}" for t in self._targets]
        yield Static(
            "[bold]Open in which multiplexer?[/]\n\n"
            + "   ".join(lines)
            + "\n\n[dim]Esc to cancel[/]",
            id="multiplexer-choice",
        )

    def on_key(self, event: events.Key) -> None:
        """Each target is picked by its own key rather than a cursor, so the
        choice costs one keystroke — the same as the handoff it replaces."""
        for target in self._targets:
            if event.key == target.key:
                event.stop()
                self.dismiss(target)
                return

    def action_cancel(self) -> None:
        self.dismiss(None)


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------


class SessionBrowser(App):
    """Lightweight agent session browser."""

    TITLE = "session browser"
    CSS = APP_CSS
    HORIZONTAL_BREAKPOINTS: ClassVar[list[tuple[int, str]]] = [
        (0, "-micro"),
        (72, "-compact"),
        (100, "-standard"),
        (144, "-wide"),
    ]
    VERTICAL_BREAKPOINTS: ClassVar[list[tuple[int, str]]] = [
        (0, "-short"),
        (28, "-tall"),
    ]
    # Free up ctrl+p for fzf-style "previous result" in search boxes.
    ENABLE_COMMAND_PALETTE = False

    BINDINGS: ClassVar[list[Binding]] = [
        Binding("q", "quit", "Quit"),
        Binding("escape", "escape", "Back", show=False),
        Binding("/", "focus_search", "Search"),
        Binding("s", "focus_detail_search", "Find in session", show=False),
        Binding("tab", "switch_pane", "Switch pane", priority=True),
        Binding("shift+tab", "switch_pane", "Switch pane", show=False, priority=True),
        Binding("z", "toggle_focus_mode", "Focus pane"),
        Binding("?", "show_help", "Keys"),
        Binding("left", "focus_left_pane", "Sessions", show=False),
        Binding("right", "focus_right_pane", "Transcript", show=False),
        Binding("c", "copy_resume", "Copy resume"),
        # Textual itself binds ctrl+c/super+c to copy the mouse text
        # selection, but tmux collapses the super (cmd) modifier into Meta
        # when forwarding keys, so cmd+c reaches the app as alt+c. Bind that
        # too so cmd+c copies when running under tmux (Ghostty must pass the
        # key through, e.g. `keybind = performable:super+c=copy_to_clipboard`).
        Binding("alt+c", "screen.copy_text", "Copy selection", show=False),
        Binding("i", "copy_session_id", "Copy id", show=False),
        Binding("t", "open_terminal", "Open in terminal", show=False),
        Binding("e", "export_chat_clipboard", "Copy chat", show=False),
        Binding("E", "export_chat_file", "Export chat", show=False),
        Binding("m", "toggle_matches_only", "Matching blocks", show=False),
        Binding("n", "next_match", "Next match", show=False),
        Binding("N", "prev_match", "Prev match", show=False),
        Binding("J", "next_entry", "Next entry", show=False),
        Binding("K", "prev_entry", "Previous entry", show=False),
        Binding("]", "next_tool", "Next tool", show=False),
        Binding("[", "prev_tool", "Previous tool", show=False),
        Binding("r", "refresh_sessions", "Refresh", show=False),
        Binding("p", "toggle_here", "This project", show=False),
        # Vim-style navigation (hidden from footer to keep it uncluttered)
        Binding("j", "nav_down", "Down", show=False),
        Binding("k", "nav_up", "Up", show=False),
        Binding("g", "nav_top", "Top", show=False),
        Binding("G", "nav_bottom", "Bottom", show=False),
        Binding("ctrl+d", "halfpage_down", "½ page down", show=False),
        Binding("ctrl+u", "halfpage_up", "½ page up", show=False),
    ]

    def __init__(self) -> None:
        super().__init__()
        self._all_sessions: list[Session] = []
        self._filtered: list[Session] = []
        self._table_sessions: list[Session] = []
        self._selected: Session | None = None
        self._transcript: Transcript | None = None
        self._entry_spans: list[tuple[int, int]] = []
        self._entry_widgets: list[TranscriptEntryWidget] = []
        self._highlighted_entry_indices: set[int] = set()
        self._detail_text: str = ""
        self._window_start: int = 0
        # (start, end) char spans of matches in _detail_text. Spans, not
        # offsets: markdown-insensitive matching means a match can be longer
        # in the original text than the query (stripped `/* inside it).
        self._matches: list[tuple[int, int]] = []
        self._match_idx: int = -1
        self._search_query: str = ""
        # Matching-blocks-only projection: a display mode over the match state
        # that already exists, holding no transcript of its own.
        self._matches_only: bool = False
        self._transcript_title: str = "TRANSCRIPT"
        # What the pane last mounted, entries and gaps in order, so a re-render
        # that would rebuild the same rows reuses them instead.
        self._rendered_plan: list[tuple[str, int]] = []
        self._projected_blocks: int = 0
        self._window_blocks: int = 0
        self._filter_query: str = ""
        self._content_hits: dict[str, ContentHit] = {}
        # (normalized query, hit ids) of the last *completed* content
        # search. Extending a query can only shrink its hit set (any text
        # containing the longer phrase contains the shorter one), so the
        # next search may be restricted to these sessions — the difference
        # between re-scanning 1,000+ transcripts per keystroke and a few.
        self._hits_basis: tuple[str, frozenset[str]] | None = None
        # Lazily-built (folded_text, offset_map) for _detail_text, shared by
        # every keystroke of the in-session search.
        self._detail_norm: tuple[str, _OffsetMap] | None = None
        # Per-entry lazy fold state for the structured detail search:
        # [folded, offset_map_or_None, map_ready]. Folding a multi-megabyte
        # transcript once is cheap; offset recovery stays lazy until an entry
        # actually matches and is then reused across keystrokes.
        self._entry_norms: list[list | None] | None = None
        # The query this app last auto-seeded into the detail search box, so
        # a user-typed detail query is never clobbered by global-search sync.
        self._detail_seed: str = ""
        self._content_search_cache = ContentSearchCache()
        self._filter_timer = None
        self._pending_query: str = ""
        self._spinner_timer = None
        self._spinner_frame: int = 0
        # phase -> (done, total), as reported by the content-search worker.
        # Several phases run at once, so each keeps its own slot and the
        # renderer picks the live one. Written from worker threads and read
        # by the spinner timer on the UI thread: whole-value dict item
        # assignment, so a torn read is impossible and neither a lock nor a
        # message per session is needed.
        self._search_progress: dict[str, tuple[int, int]] = {}
        # Bumped per search so a superseded worker's progress is discarded.
        self._search_generation: int = 0
        self._done_flash_timer = None
        self._suppress_table_highlights = False
        self._table_rebuild_generation = 0
        self._table_column_mode: str | None = None
        self._here_only: bool = False
        self._cwd_base: str = os.getcwd()
        self._focus_mode: bool = False
        # Terminal drivers may queue a burst such as `/speaker` before the
        # focus action for `/` has completed. While this is set, printable keys
        # which still arrive at the App are forwarded to the intended input.
        self._search_focus_handoff: str | None = None
        # Armed by the first `g` of a vim-style `gg` chord.
        self._pending_g: bool = False
        self._pending_g_timer = None

    def compose(self) -> ComposeResult:
        with Horizontal(id="app-bar"):
            yield Static("[bold cyan]◉[/]  session browser", id="brand")
            yield Static("Discovering sessions…", id="app-meta")
        with Horizontal(id="main"):
            with Vertical(id="left-pane"):
                with Horizontal(id="sessions-header"):
                    yield Static("SESSIONS", classes="pane-title", id="sessions-title")
                    yield Static("", classes="pane-title", id="sessions-status")
                yield SearchInput(placeholder="Search sessions…", id="global-search")
                yield SessionTable(id="session-table", cursor_type="row")
            with Vertical(id="right-pane"):
                yield Static("TRANSCRIPT", classes="pane-title", id="transcript-title")
                yield Static("", id="detail-header")
                yield SearchInput(
                    placeholder="Search transcript… (n/N to navigate)",
                    id="detail-search",
                )
                yield Label("", id="match-counter")
                yield VerticalScroll(
                    Static("Select a session to view details", id="detail-content"),
                    id="detail-scroll",
                )
        yield Static("Loading sessions…", id="status-bar")
        yield ResponsiveFooter()

    def on_mount(self) -> None:
        table = self.query_one("#session-table", DataTable)
        table.add_columns("Provider", "Project", "Age", "Summary")
        self._table_column_mode = "full"
        self.screen.set_focus(self.query_one("#session-table", DataTable))
        self._apply_pane_visibility("sessions")
        self._load_sessions()

    @property
    def _status(self) -> Static:
        return self.query_one("#status-bar", Static)

    @property
    def _app_meta(self) -> Static:
        return self.query_one("#app-meta", Static)

    # Held so a rapid second flash cancels the first's revert timer rather
    # than letting it clear the bar mid-pulse.
    _flash_timer = None
    _FLASH_DURATION = 1.2

    def _flash_status(self, message: str, *, ok: bool = True) -> None:
        """Update the status bar and pulse it (green on success, red on
        failure) so the confirmation registers visually — the bar is otherwise
        muted and easy to overlook."""
        bar = self._status
        bar.update(message)
        bar.remove_class("-flash-ok", "-flash-err")
        bar.add_class("-flash-ok" if ok else "-flash-err")
        if self._flash_timer is not None:
            self._flash_timer.stop()
        self._flash_timer = self.set_timer(
            self._FLASH_DURATION,
            self._restore_context_bar,
        )

    @_teardown_safe
    def _restore_context_bar(self) -> None:
        self._flash_timer = None
        self._status.remove_class("-flash-ok", "-flash-err")
        self._update_context_bar()

    @staticmethod
    def _pane_for_widget(widget) -> str:
        if isinstance(widget, TranscriptEntryWidget):
            return "detail"
        if widget is not None and widget.id in {"detail-search", "detail-scroll"}:
            return "detail"
        return "sessions"

    def _apply_pane_visibility(self, pane: str) -> None:
        """Synchronize auto-compact and manual focus layouts with navigation."""
        screen = self.screen
        if isinstance(screen, ShortcutHelp):
            return
        screen.remove_class(
            "-show-sessions", "-show-detail", "-focus-left", "-focus-right"
        )
        screen.add_class("-show-detail" if pane == "detail" else "-show-sessions")
        if self._focus_mode:
            screen.add_class("-focus-right" if pane == "detail" else "-focus-left")

    def _update_context_bar(self, widget=None) -> None:
        if self._status.has_class("-flash-ok") or self._status.has_class("-flash-err"):
            return
        focused = widget if widget is not None else self.focused
        pane = self._pane_for_widget(focused)
        if focused is not None and focused.id in {"global-search", "detail-search"}:
            location = "search"
        elif pane == "detail":
            location = "reading"
        else:
            location = "list"
        scope = "this project" if self._here_only else "all projects"
        if self.size.width < 100:
            hint = "[cyan]tab[/] switch  ·  [cyan]?[/] keys"
        elif self._focus_mode:
            hint = "[cyan]z[/] split  ·  [cyan]tab[/] switch  ·  [cyan]?[/] keys"
        else:
            hint = "[cyan]z[/] focus  ·  [cyan]tab[/] switch  ·  [cyan]?[/] keys"
        name = "Transcript" if pane == "detail" else "Sessions"
        self._status.update(
            f"[bold]{name}[/] [dim]› {location}  ·  {scope}  ·[/]  {hint}"
        )

    def on_descendant_focus(self, event: events.DescendantFocus) -> None:
        if isinstance(self.screen, ShortcutHelp):
            return
        pane = self._pane_for_widget(event.widget)
        self._apply_pane_visibility(pane)
        self._update_context_bar(event.widget)

    @_teardown_safe
    def _sync_responsive_chrome(self) -> None:
        """Refresh width/height-sensitive text after an in-place resize."""
        if isinstance(self.screen, ShortcutHelp):
            return
        if self._selected is not None:
            self._update_detail_header(self._selected)
        self._update_context_bar()

    def copy_to_clipboard(self, text: str) -> None:
        """Called by Textual when the user copies a mouse text selection
        (ctrl+c / cmd+c). The default implementation only emits an OSC 52
        escape sequence, which terminals may ignore silently — also push the
        text through the system clipboard tool and confirm in the status bar.
        """
        if not text:
            self._flash_status("Nothing selected", ok=False)
            return
        super().copy_to_clipboard(text)
        if copy_to_clipboard(text):
            self._flash_status(f"Copied selection ({len(text)} chars)")
        else:
            self._flash_status("No clipboard tool; sent OSC 52 escape only", ok=False)

    # -- Discovery (async worker) ------------------------------------------

    def _load_sessions(self) -> None:
        self._status.update("Discovering sessions…")
        self._app_meta.update("Discovering sessions…")
        self.run_worker(self._discover_worker, thread=True)

    def _discover_worker(self) -> list[Session]:
        return discover_all()

    @_teardown_safe
    def on_worker_state_changed(self, event: Worker.StateChanged) -> None:
        if not event.worker.is_finished:
            return
        if event.state is WorkerState.ERROR:
            # A crashed worker must not fail silently: without this, a dead
            # content search leaves the spinner spinning forever and the
            # app simply looks like "search doesn't work". (CANCELLED is
            # routine — exclusive workers supersede each other constantly.)
            if event.worker.name == "_content_search":
                self._stop_deep_search()
            self._flash_status(
                f"{event.worker.name.lstrip('_')} failed: {event.worker.error}",
                ok=False,
            )
            return
        if event.worker.name == "_discover_worker":
            if event.worker.result is not None:
                self._all_sessions = event.worker.result
                self._apply_filter("")
        elif event.worker.name == "_load_content":
            result = event.worker.result
            if result is None:
                return
            session, transcript = result
            # Discard stale loads — the selection moved on while this ran.
            if self._selected is None or session.id != self._selected.id:
                return
            self._on_transcript_loaded(transcript)
        elif event.worker.name == "_content_search":
            result = event.worker.result
            if result is None:
                return
            query, hits = result
            # Discard stale results — the filter query moved on while we ran.
            # A newer search is still in flight, so leave the spinner running.
            if query != self._filter_query:
                return
            # Count only the rows the transcript search actually added — the
            # sessions already matched by name or summary were on screen from
            # the first keystroke, so claiming them here would overstate it.
            added = sum(
                1 for s in self._all_sessions if s.id in hits and not s.matches(query)
            )
            self._stop_deep_search(found=added)
            self._content_hits = hits
            self._hits_basis = (normalize_match_text(query.strip()), frozenset(hits))
            self._rebuild_filtered(query)

    # -- Global search / filter --------------------------------------------

    def _schedule_filter(self, query: str) -> None:
        """Debounce: coalesce a burst of keystrokes into one filter pass."""
        self._pending_query = query
        if self._filter_timer is not None:
            self._filter_timer.stop()
        self._filter_timer = self.set_timer(_FILTER_DEBOUNCE, self._run_pending_filter)

    @_teardown_safe
    def _run_pending_filter(self) -> None:
        self._filter_timer = None
        # A debounce can outlive its query: the box may have been changed
        # (its Changed message — which would reschedule us — still in
        # flight) or a direct _apply_filter may have superseded it. Never
        # resurrect a query the search box no longer shows.
        if self._pending_query != self.query_one("#global-search", Input).value:
            return
        self._apply_filter(self._pending_query)

    def _cwd_scope(self, sessions: list[Session]) -> list[Session]:
        """Restrict to sessions rooted at (or under) the launch directory.

        Mirrors the CLI's --here: exact match or path-prefix, so a session
        opened one directory up from a subproject doesn't leak in. Sessions
        with no recorded cwd are excluded, same as --here.
        """
        if not self._here_only:
            return list(sessions)
        base = self._cwd_base
        return [
            s
            for s in sessions
            if s.cwd and (s.cwd == base or s.cwd.startswith(base + os.sep))
        ]

    def _update_meta_count(self) -> None:
        """Refresh only the header count, leaving the status bar alone."""
        scope = "this project" if self._here_only else "all projects"
        count = f"{len(self._filtered):,}"
        if len(self._filtered) != len(self._all_sessions):
            count = f"{count} of {len(self._all_sessions):,}"
        self._app_meta.update(f"[bold]{count}[/] sessions  ·  {scope}")

    def _update_status_count(self) -> None:
        self._update_meta_count()
        self._update_context_bar()

    def action_toggle_here(self) -> None:
        self._here_only = not self._here_only
        search = self.query_one("#global-search", SearchInput)
        search.placeholder = (
            "Search sessions… (this project)" if self._here_only else "Search sessions…"
        )
        if self._filter_query:
            self._rebuild_filtered(self._filter_query)
        else:
            self._filtered = self._cwd_scope(self._all_sessions)
            self._rebuild_table()
        self._update_status_count()
        self._flash_status(
            f"Scope: this project ({self._cwd_base})"
            if self._here_only
            else "Scope: all projects"
        )

    def _apply_filter(self, query: str) -> None:
        self._filter_query = query
        self._content_hits = {}
        if not query:
            self._stop_deep_search()
            self._filtered = self._cwd_scope(self._all_sessions)
            self._rebuild_table()
            self._update_status_count()
            self._sync_detail_search()
            return
        # Show metadata matches immediately, then merge transcript matches when
        # the slower content search completes.
        self._rebuild_filtered(query)
        self._start_deep_search()
        sessions_snapshot = self._all_sessions
        basis = self._hits_basis
        needle = normalize_match_text(query.strip())
        if basis is not None and basis[0] and basis[0] in needle:
            # Narrowing an already-searched query: only its hits can match.
            # (A session written to *since* that search could in principle
            # gain the phrase mid-typing; the full corpus is re-scanned as
            # soon as the query stops being an extension of the basis.)
            ids = basis[1]
            sessions_snapshot = [s for s in self._all_sessions if s.id in ids]
        self.run_worker(
            lambda q=query, ss=sessions_snapshot: self._content_search_worker(q, ss),
            thread=True,
            exclusive=True,
            group="content_search",
            name="_content_search",
        )

    def _content_search_worker(
        self, query: str, sessions: list[Session]
    ) -> tuple[str, dict[str, ContentHit]]:
        """Search with a cancellation check safe to use in nested threads."""
        worker = get_current_worker()
        generation = self._search_generation
        hits = search_session_hits(
            sessions,
            query,
            cache=self._content_search_cache,
            cancelled=lambda: worker.is_cancelled,
            progress=lambda phase, done, total: self._note_search_progress(
                generation, phase, done, total
            ),
        )
        return query, hits

    def action_clear_global_search(self) -> None:
        """Clear global search state and cooperatively stop its worker."""
        if self._filter_timer is not None:
            self._filter_timer.stop()
            self._filter_timer = None
        self._pending_query = ""
        self.workers.cancel_group(self, "content_search")
        search = self.query_one("#global-search", SearchInput)
        search.value = ""
        self._apply_filter("")

    def _note_search_progress(
        self, generation: int, phase: str, done: int, total: int
    ) -> None:
        """Progress sink for the content-search worker threads.

        Deliberately does nothing but store: posting a message per session
        would put thousands of updates through the UI message loop to feed a
        display that only repaints 12 times a second anyway.

        Superseded searches are ignored. Cancellation is cooperative and the
        database scan has no check inside it at all, so an abandoned search
        keeps running and reporting for a while; without the generation test
        its counters interleave with the live ones and the readout jumps
        backwards, which looks far more broken than no readout at all.
        """
        if generation != self._search_generation:
            return
        self._search_progress[phase] = (done, total)

    def _start_deep_search(self) -> None:
        self._spinner_frame = 0
        self._search_generation += 1
        self._search_progress = {}
        if self._done_flash_timer is not None:
            self._done_flash_timer.stop()
            self._done_flash_timer = None
        self._render_spinner()
        if self._spinner_timer is None:
            self._spinner_timer = self.set_interval(
                _SPINNER_INTERVAL, self._render_spinner
            )

    @_teardown_safe
    def _render_spinner(self) -> None:
        glyph = _SPINNER_FRAMES[self._spinner_frame % len(_SPINNER_FRAMES)]
        self._spinner_frame += 1
        shown = self._set_sessions_title(_progress_forms(glyph, self._search_progress))
        # A short terminal hides the pane titles entirely, which would leave
        # no indicator at all; there, keep the original in-box glyph.
        self.query_one("#global-search", SearchInput).set_spinner(
            None if shown else glyph
        )

    def _set_sessions_title(self, forms: list[str] | None) -> bool:
        """Write the right-aligned status readout with the widest form that fits.

        The readout lives in its own slot at the pane's right edge, so the
        "SESSIONS" title never moves. At a 100-column terminal the slot is
        ~31 columns wide, so the full phrase does not always fit; rather than
        truncate it to nonsense, fall back through progressively shorter
        wordings. Returns whether a status was actually displayed — when the
        title row is hidden or too narrow for even the shortest form, the
        caller needs to know the indicator did not land anywhere.
        """
        status = self.query_one("#sessions-status", Static)
        budget = status.size.width
        best = next((f for f in forms if len(f) <= budget), "") if forms else ""
        status.update(f"[$warning]{_escape_markup(best)}[/]" if best else "")
        return bool(best)

    def _stop_deep_search(self, found: int = 0) -> None:
        if self._spinner_timer is not None:
            self._spinner_timer.stop()
            self._spinner_timer = None
        self._search_progress = {}
        self.query_one("#global-search", SearchInput).set_spinner(None)
        if found <= 0:
            self._set_sessions_title(None)
            return
        # Name the second wave as it lands. Without this the extra rows are
        # the unexplained event the progress readout just spent five seconds
        # promising, and the user is back to "where did those come from".
        self._set_sessions_title(
            [
                f"✓ +{found:,} from transcripts",
                f"✓ +{found:,} found",
                f"✓ +{found:,}",
            ]
        )
        self._done_flash_timer = self.set_timer(
            _DONE_FLASH_DURATION, self._clear_done_flash
        )

    @_teardown_safe
    def _clear_done_flash(self) -> None:
        self._done_flash_timer = None
        if self._spinner_timer is None:
            self._set_sessions_title(None)

    def _match_tier(self, s: Session, query: str) -> int:
        """Demote echo pollution, otherwise leave recency in charge.

        0 — conversation: the query appears in metadata (summary is the
            opening prompt) or in a user/assistant message of the transcript.
        1 — tool/system traffic only: a session that merely grepped or
            quoted the phrase must not outrank the session that said it.

        User and assistant evidence deliberately share a tier: splitting
        them buried a minutes-old assistant-prose match under month-old
        user matches, which read as "search is broken".
        """
        if s.matches(query):
            return 0
        hit = self._content_hits.get(s.id)
        if hit is None or hit.role not in ("user", "assistant"):
            return 1
        return 0

    def _rebuild_filtered(self, query: str) -> None:
        hits = self._content_hits
        matched = [s for s in self._all_sessions if s.matches(query) or s.id in hits]
        # Stable sort: _all_sessions is newest-first, so recency still
        # orders sessions within the same evidence tier.
        matched.sort(key=lambda s: self._match_tier(s, query))
        self._filtered = self._cwd_scope(matched)
        self._rebuild_table()
        self._update_meta_count()
        self._sync_detail_search()

    def _rebuild_table(
        self, preserve_id: str | None = None, *, snapshot_cursor: bool = True
    ) -> None:
        table = self.query_one("#session-table", DataTable)
        # Snapshot the session under the table's actual cursor. RowHighlighted
        # is delivered asynchronously, so _selected can briefly lag behind the
        # visible highlight when the content-search second wave lands. Using
        # that stale detail selection would visibly yank the cursor elsewhere.
        # Track the id, not the row index, because new rows can be inserted
        # above it.
        if snapshot_cursor:
            cursor_row = table.cursor_row
            if 0 <= cursor_row < len(self._table_sessions):
                preserve_id = self._table_sessions[cursor_row].id
            else:
                preserve_id = self._selected.id if self._selected else None
        selected_is_visible = self._selected is not None and any(
            s.id == self._selected.id for s in self._filtered
        )
        self._table_rebuild_generation += 1
        rebuild_generation = self._table_rebuild_generation
        self._suppress_table_highlights = True
        table.clear()
        target_row = 0
        for idx, s in enumerate(self._filtered):
            provider_label = PROVIDER_COLOURS.get(s.provider, s.provider)
            project = f"[dim]{_escape_markup(_project_name(s.cwd))}[/]"
            hit = self._content_hits.get(s.id) if self._filter_query else None
            if hit is not None:
                summary = _format_content_hit(hit)
            else:
                summary = (
                    _escape_markup(_ellipsize(s.summary, 120))
                    if s.summary
                    else "(no summary)"
                )
            age = _relative_time(s.updated_at or s.created_at)
            values = {
                "minimal": (provider_label, project),
                "compact": (provider_label, project, age),
                "full": (provider_label, project, age, summary),
            }[self._table_column_mode or "full"]
            table.add_row(*values, key=str(idx))
            if s.id == preserve_id:
                target_row = idx
        self._table_sessions = list(self._filtered)
        if self._filtered and target_row:
            table.move_cursor(row=target_row)
        # The detail pane always represents the highlighted row. In particular,
        # filtering must not leave a transcript visible for a session which is
        # no longer in the result list.
        if self._filtered and not selected_is_visible:
            self._select_session("0")
        elif not self._filtered:
            self._clear_selection()
        table.post_message(SessionTable.RebuildSettled(rebuild_generation))

    def on_session_table_columns_changed(
        self, event: SessionTable.ColumnsChanged
    ) -> None:
        if event.mode == self._table_column_mode:
            return
        table = self.query_one("#session-table", DataTable)
        cursor_row = table.cursor_row
        preserve_id = (
            self._table_sessions[cursor_row].id
            if 0 <= cursor_row < len(self._table_sessions)
            else self._selected.id
            if self._selected
            else None
        )
        labels = {
            "minimal": ("Provider", "Project"),
            "compact": ("Provider", "Project", "Age"),
            "full": ("Provider", "Project", "Age", "Summary"),
        }[event.mode]
        self._suppress_table_highlights = True
        table.clear(columns=True)
        table.add_columns(*labels)
        self._table_column_mode = event.mode
        self._rebuild_table(preserve_id, snapshot_cursor=False)

    def _clear_selection(self) -> None:
        self._selected = None
        self._transcript = None
        self._entry_spans = []
        self._entry_widgets = []
        self._rendered_plan = []
        self._highlighted_entry_indices = set()
        self._detail_text = ""
        self._detail_norm = None
        self._entry_norms = None
        self._matches = []
        self._match_idx = -1
        self._transcript_title = "TRANSCRIPT"
        self._update_transcript_title()
        self.query_one("#detail-header", Static).update("")
        self._set_match_counter()
        self._clear_transcript_widgets()
        self.query_one("#detail-content", Static).update(
            "No sessions match your search"
        )

    def on_session_table_rebuild_settled(
        self, event: SessionTable.RebuildSettled
    ) -> None:
        # Only the newest rebuild may lift suppression; a stale settle marker
        # means another rebuild's echoes may still be in flight.
        if event.generation == self._table_rebuild_generation:
            self._suppress_table_highlights = False

    def on_input_changed(self, event: Input.Changed) -> None:
        if event.input.id == self._search_focus_handoff:
            self._search_focus_handoff = None
        if event.input.id == "global-search":
            self._schedule_filter(event.value)
        elif event.input.id == "detail-search":
            self._do_session_search(event.value)

    def on_input_submitted(self, event: Input.Submitted) -> None:
        """Enter in a search box commits the search and returns focus."""
        if event.input.id == "global-search":
            self._search_focus_handoff = None
            self._apply_pane_visibility("sessions")
            self.screen.set_focus(self.query_one("#session-table", DataTable))
        elif event.input.id == "detail-search":
            self._search_focus_handoff = None
            # Enter must always commit: match state can lag the box (e.g.
            # a transcript that finished loading after the last keystroke).
            if self._search_query != event.value:
                self._do_session_search(event.value)
            self._apply_pane_visibility("detail")
            self.screen.set_focus(self.query_one("#detail-scroll", VerticalScroll))

    def on_key(self, event) -> None:
        """Bridge keystrokes queued during a search-focus transition.

        Also disarms a pending `gg` chord: any key other than `g` cancels it,
        matching vim.
        """
        if event.key != "g":
            self._clear_pending_g()
        target_id = self._search_focus_handoff
        if target_id is None:
            return
        if isinstance(self.focused, Input) and self.focused.id == target_id:
            return
        character = event.character
        if character and character.isprintable():
            target = self.query_one(f"#{target_id}", Input)
            target.value = target.value + character
            target.cursor_position = len(target.value)
            event.stop()

    # -- Session selection & detail ----------------------------------------

    def on_data_table_row_selected(self, event: DataTable.RowSelected) -> None:
        # Highlighting already loads the detail (see RowHighlighted below).
        # Enter drills into the transcript. Search remains a contextual `/`.
        self._select_session(str(event.row_key.value))
        self._apply_pane_visibility("detail")
        self.screen.set_focus(self.query_one("#detail-scroll", VerticalScroll))

    def on_data_table_row_highlighted(self, event: DataTable.RowHighlighted) -> None:
        if self._suppress_table_highlights:
            return
        if event.row_key:
            self._select_session(str(event.row_key.value))

    def _select_session(self, row_key: str) -> None:
        try:
            idx = int(row_key)
        except (TypeError, ValueError):
            return
        if idx < 0 or idx >= len(self._filtered):
            return
        session = self._filtered[idx]
        if session == self._selected:
            return
        self._selected = session
        self._transcript = None
        self._entry_spans = []
        self._entry_widgets = []
        self._rendered_plan = []
        self._highlighted_entry_indices = set()
        self._detail_text = ""
        self._detail_norm = None
        self._entry_norms = None
        self._matches = []
        self._match_idx = -1
        self._update_detail_header(session)
        # Load content in worker, tagged with its session so a slow read that
        # lands after the selection moved on can be discarded.
        self._clear_transcript_widgets()
        self.query_one("#detail-content", Static).update("Loading…")
        self.run_worker(
            lambda s=session: (s, self._load_transcript_safe(s)),
            thread=True,
            exclusive=True,
            group="load_content",
            name="_load_content",
        )

    @staticmethod
    def _load_transcript_safe(session: Session) -> Transcript:
        try:
            return load_transcript(session)
        except (TranscriptUnreadable, OSError) as exc:
            return Transcript(
                session,
                [TranscriptEntry("system", f"(could not read session: {exc})")],
            )

    def _on_transcript_loaded(self, transcript: Transcript) -> None:
        """Cache both representations: widgets display entries; search and
        export keep using the same contiguous text buffer as before."""
        self._transcript = transcript
        self._detail_text = render_text(transcript)
        self._entry_spans = self._build_entry_spans(transcript)
        self._detail_norm = None
        self._entry_norms = None
        self._window_start = 0
        self._matches = []
        self._match_idx = -1
        # Marks the match state as belonging to no query, so the sync below
        # knows any populated search box needs a re-run against this
        # transcript — including one the user typed, which is never seeded.
        self._search_query = ""
        self._set_match_counter()
        self._render_detail()
        # A session opened *because* it contains the global query should
        # open with that query active: highlighted, counted, n/N working.
        self._sync_detail_search()

    @staticmethod
    def _build_entry_spans(transcript: Transcript) -> list[tuple[int, int]]:
        """Absolute flat-buffer ranges for each entry's body text."""
        spans: list[tuple[int, int]] = []
        cursor = 0
        for index, entry in enumerate(transcript.entries):
            if index:
                cursor += 2  # render_text's blank line between blocks
            start = cursor + len(entry_label(entry)) + 2  # ``Label: ``
            end = start + len(entry.text)
            spans.append((start, end))
            cursor = end
        return spans

    def _on_content_loaded(self, content: str) -> None:
        """Legacy flat-content hook retained for focused rendering tests."""
        self._transcript = None
        self._entry_spans = []
        self._entry_widgets = []
        self._rendered_plan = []
        self._highlighted_entry_indices = set()
        self._detail_text = content
        self._detail_norm = None
        self._entry_norms = None
        self._window_start = 0
        self._matches = []
        self._match_idx = -1
        self._set_match_counter()
        self._render_detail()

    def _update_detail_header(self, s: Session) -> None:
        header = self.query_one("#detail-header", Static)
        # A hidden compact-pane reports width 0; when opened it will occupy
        # the full viewport, so size the text for that destination width.
        width = header.size.width or max(40, self.size.width - 4)
        summary = _escape_markup(
            _ellipsize(s.summary or "Untitled session", max(24, width - 16))
        )
        project = _escape_markup(_project_name(s.cwd, 28))
        branch = _escape_markup(s.branch or "no branch")
        updated = _relative_time(s.updated_at or s.created_at)
        session_id = _escape_markup(_ellipsize(s.id, 18))
        lines = [
            f"[bold cyan]{s.provider.upper()}[/]  [bold]{summary}[/]",
            f"[dim]{project}  ·  {branch}  ·  {updated}  ·  {session_id}[/]",
        ]
        if self.size.height >= 28 and width >= 56 and s.cwd:
            lines.append(f"[dim]{_escape_markup(_ellipsize(s.cwd, width - 2))}[/]")
        header_text = "\n".join(lines)
        self._transcript_title = f"TRANSCRIPT  ·  {s.provider.upper()}"
        self._update_transcript_title()
        header.update(header_text)

    def _render_detail(self) -> None:
        """Render the visible window of the detail content.

        The match counter is refreshed here, at the end, rather than by each
        caller before it: in the projected view the counter reports how many
        blocks are on screen, which is not known until they have been chosen.
        Updating it first showed the previous render's count. The title is
        refreshed here for the same reason: the projection lapses whenever
        the query goes away, and a pane still announcing MATCHING BLOCKS
        while showing all of them is the one thing the label must never do.
        """
        if not self._detail_text:
            self.query_one("#detail-content", Static).update("(no content)")
        elif self._transcript is not None:
            self._render_structured_entries()
            self._scroll_to_match()
        else:
            self.query_one("#detail-content", Static).update(
                self._compose_visible_text()
            )
            self._scroll_to_match()
        self._update_match_counter()
        self._update_transcript_title()

    def _render_structured_entries(self):
        scroll = self.query_one("#detail-scroll", VerticalScroll)
        window_end = self._window_start + _DISPLAY_WINDOW
        active_span = (
            self._matches[self._match_idx]
            if 0 <= self._match_idx < len(self._matches)
            else None
        )
        visible = [
            (index, entry, span)
            for index, (entry, span) in enumerate(
                zip(self._transcript.entries, self._entry_spans, strict=True)
            )
            if span[1] >= self._window_start and span[0] <= window_end
        ]
        rows = self._project(visible)
        self._projected_blocks = sum(1 for kind, _ in rows if kind == "entry")
        self._window_blocks = len(visible)
        # A query changes highlighting, not structure. Reuse mounted widgets
        # so fast typing never pays a remove/remount cost per keystroke. The
        # plan carries the gap rows as well as the entries, because in the
        # projected view a run of hidden blocks is part of what is on screen
        # and two different runs must not reuse each other's marker.
        if rows == self._rendered_plan:
            widgets_by_index = {w.entry_index: w for w in self._entry_widgets}
            matches_by_index: dict[int, list[tuple[int, int]]] = {}
            for index, _, (start, end) in visible:
                left = bisect_left(self._matches, (start,))
                right = bisect_left(self._matches, (end,))
                if left != right:
                    matches_by_index[index] = self._matches[left:right]
            changed = self._highlighted_entry_indices | set(matches_by_index)
            for index in changed:
                widget = widgets_by_index.get(index)
                if widget is not None:
                    widget.set_matches(
                        self._search_query,
                        matches_by_index.get(index, []),
                        active_span,
                    )
            self._highlighted_entry_indices = set(matches_by_index)
            return None

        self._clear_transcript_widgets()
        self.query_one("#detail-content", Static).update("")
        by_index = {index: (entry, span) for index, entry, span in visible}
        mounted: list[Static] = []
        widgets: list[TranscriptEntryWidget] = []
        for kind, value in rows:
            if kind == "gap":
                mounted.append(TranscriptGapWidget(value))
                continue
            entry, (start, end) = by_index[value]
            widget = TranscriptEntryWidget(entry, value, start)
            left = bisect_left(self._matches, (start,))
            right = bisect_left(self._matches, (end,))
            widget.set_matches(
                self._search_query, self._matches[left:right], active_span
            )
            widgets.append(widget)
            mounted.append(widget)
        self._entry_widgets = widgets
        self._rendered_plan = rows
        self._highlighted_entry_indices = {
            widget.entry_index for widget in widgets if widget._match_spans
        }
        if widgets:
            return scroll.mount(*mounted)
        self.query_one("#detail-content", Static).update(
            "(no matching blocks)" if self._projecting else "(empty session)"
        )
        return None

    @property
    def _projecting(self) -> bool:
        """Whether the matching-only view is in force right now.

        The mode is a view *of a query*. With no query there is nothing to
        project, so the preference is held and simply not applied -- which is
        also what keeps it stable across the moment a session loads, when the
        query is briefly cleared before being re-seeded.
        """
        return self._matches_only and bool(self._search_query)

    def _project(
        self, visible: list[tuple[int, TranscriptEntry, tuple[int, int]]]
    ) -> list[tuple[str, int]]:
        """The rows to render: ``("entry", index)`` and ``("gap", hidden)``.

        Built from the match spans already computed for the flat buffer -- a
        bisect per entry, no parse, no fold, and no second copy of the
        transcript. The full view is the identity case, so a toggle in either
        direction costs one re-render of what is already in memory.
        """
        if not self._projecting:
            return [("entry", index) for index, _, _ in visible]
        rows: list[tuple[str, int]] = []
        hidden = 0
        for index, _, (start, end) in visible:
            if bisect_left(self._matches, (start,)) == bisect_left(
                self._matches, (end,)
            ):
                hidden += 1
                continue
            if hidden:
                rows.append(("gap", hidden))
                hidden = 0
            rows.append(("entry", index))
        if hidden:
            rows.append(("gap", hidden))
        return rows

    def _clear_transcript_widgets(self) -> None:
        """Remove both kinds of mounted row.

        Gaps are not entries, so a query for TranscriptEntryWidget leaves them
        behind -- and a stale marker claiming blocks are hidden that are not
        is worse than no marker at all.
        """
        self.query(TranscriptEntryWidget).remove()
        self.query(TranscriptGapWidget).remove()
        self._rendered_plan = []

    def _compose_visible_text(self) -> str:
        """Bounded display window with omission markers, markup-ready."""
        text = self._detail_text
        start = self._window_start
        end = min(len(text), start + _DISPLAY_WINDOW)
        if self._matches and self._search_query:
            body = self._highlight_matches(start, end)
        else:
            body = _escape_markup(text[start:end])
        prefix = (
            f"… ({start:,} characters before this window omitted)\n\n"
            if start > 0
            else ""
        )
        suffix = (
            f"\n\n… ({len(text) - end:,} characters after this window omitted)"
            if end < len(text)
            else ""
        )
        return prefix + body + suffix

    def _highlight_matches(self, start: int, end: int) -> str:
        """Escape the [start:end) window and wrap in-window matches in markup.

        Match spans are absolute within the full transcript; matches
        outside the window are simply not decorated (they stay reachable
        via n/N, which slides the window first).
        """
        text = self._detail_text
        parts: list[str] = []
        prev = start
        for i, (mstart, mend) in enumerate(self._matches):
            if mstart < prev or mend > end:
                continue
            parts.append(_escape_markup(text[prev:mstart]))
            escaped_match = _escape_markup(text[mstart:mend])
            if i == self._match_idx:
                parts.append(f"[bold reverse yellow]{escaped_match}[/]")
            else:
                # Explicit foreground: entry text is muted white with alpha,
                # and letting it blend over yellow renders the highlight as
                # pale-yellow-on-yellow — a match nobody can read.
                parts.append(f"[black on yellow]{escaped_match}[/]")
            prev = mend
        parts.append(_escape_markup(text[prev:end]))
        return "".join(parts)

    def _ensure_match_visible(self) -> None:
        """Slide the display window so the current match is inside it."""
        if not self._matches or self._match_idx < 0:
            return
        offset, match_end = self._matches[self._match_idx]
        if (
            self._window_start <= offset
            and match_end <= self._window_start + _DISPLAY_WINDOW
        ):
            return
        self._window_start = max(0, offset - _DISPLAY_WINDOW // 2)

    # -- In-session search -------------------------------------------------

    def _sync_detail_search(self) -> None:
        """Carry the global query into the in-session search (and back out).

        Only a query this method itself seeded is ever overwritten or
        cleared — a query the user typed into the detail box stays put.
        """
        box = self.query_one("#detail-search", Input)
        if box.value and box.value != self._detail_seed:
            # Never clobber a user-typed query — but do keep it live: a
            # freshly loaded transcript starts with reset match state, and
            # without a re-run the query silently stops highlighting.
            if self._search_query != box.value:
                self._do_session_search(box.value)
            return
        self._detail_seed = self._filter_query
        if box.value != self._filter_query:
            # Input.Changed → _do_session_search.
            box.value = self._filter_query
        else:
            # Same query, but possibly a freshly loaded transcript whose
            # match state was just reset — re-run against it.
            self._do_session_search(box.value)

    def _do_session_search(self, query: str) -> None:
        """Markdown-insensitive search, matching the global content search —
        a session found globally must never show "0 matches" once opened.

        Structured transcripts are searched per entry: each entry gets the
        C-speed identity fast path, and the per-character offset map is
        built only for entries that both matched and changed under
        normalization. A whole-transcript map would freeze the UI for
        hundreds of milliseconds on multi-megabyte sessions — and this runs
        on every row the cursor visits while a global query is active.
        """
        self._search_query = query
        self._matches = []
        self._match_idx = -1
        if query and self._detail_text:
            if self._transcript is not None:
                needle = normalize_match_text(query.strip())
                if needle:
                    if self._entry_norms is None:
                        self._entry_norms = [None] * len(self._transcript.entries)
                    for i, ((start, _end), entry) in enumerate(
                        zip(
                            self._entry_spans,
                            self._transcript.entries,
                            strict=True,
                        )
                    ):
                        for s, e in self._entry_matches(i, entry, needle, query):
                            self._matches.append((start + s, start + e))
            else:
                # Legacy flat path (no entry structure to lean on).
                if self._detail_norm is None:
                    folded = normalize_match_text(self._detail_text)
                    self._detail_norm = (folded, _offset_map(self._detail_text, folded))
                self._matches = find_text_spans(
                    self._detail_text, query, norm=self._detail_norm
                )
            if self._matches:
                self._match_idx = 0
        self._window_start = 0
        self._ensure_match_visible()
        self._render_detail()

    def _entry_matches(
        self, index: int, entry: TranscriptEntry, needle: str, query: str
    ) -> list[tuple[int, int]]:
        """Spans of `needle` in one entry, with cached fold state.

        The fold is computed once per entry and reused across keystrokes;
        offset recovery is built only for entries that actually match, and a
        full per-character map is reserved for Unicode fold expansions.
        """
        cached = self._entry_norms[index]
        if cached is None:
            folded = normalize_match_text(entry.text)
            cached = [folded, None, False]
            self._entry_norms[index] = cached
        folded, idx_map, map_ready = cached
        if needle not in folded:
            return []
        if not map_ready:
            idx_map = _offset_map(entry.text, folded)
            cached[1] = idx_map
            cached[2] = True
        return find_text_spans(entry.text, query, norm=(folded, idx_map))

    def _set_match_counter(self, text: str = "") -> None:
        counter = self.query_one("#match-counter", Label)
        counter.update(text)
        counter.set_class(bool(text), "-visible")

    def _update_match_counter(self) -> None:
        """Occurrence position, plus the block count when blocks are hidden.

        Both numbers, because in the projected view they answer different
        questions: how far through the hits you are, and how much of the
        session you are being shown at all.
        """
        position = (
            f" {self._match_idx + 1} / {len(self._matches)}" if self._matches else ""
        )
        if self._projecting:
            # Against the blocks the *window* holds, not the whole transcript.
            # The gap markers account for exactly the blocks this number
            # leaves out, so the two have to be drawn from the same set --
            # counting against every entry in the session would leave a
            # remainder the markers never mention.
            blocks = f" · {self._projected_blocks} of {self._window_blocks} blocks "
            self._set_match_counter(f"{position or ' no matches'}{blocks}")
            return
        self._set_match_counter(f"{position} " if position else "")

    def _scroll_to_match(self) -> None:
        """Scroll the detail pane so the current match is visible."""
        if not self._matches or self._match_idx < 0:
            return
        offset = self._matches[self._match_idx][0]
        if self._transcript is not None:
            for widget in self._entry_widgets:
                start, end = self._entry_spans[widget.entry_index]
                if start <= offset < end:
                    # Deliberately no expansion here. Unfolding the entry to
                    # reach a match inside it put the whole block on screen --
                    # up to millions of characters -- to show one line of it,
                    # and it undid an explicit collapse to do that. The widget
                    # has already been handed the new active span and has
                    # moved its preview onto it, so the match is on screen
                    # either way; all that is left is to scroll to it.
                    #
                    # Updating a Static can still change both its wrapped
                    # height and every following widget's position.
                    # Resolve the exact rendered row only after that layout
                    # has settled; scrolling merely to the entry's top leaves
                    # matches deep inside a long entry off-screen.
                    self.call_after_refresh(
                        self._scroll_to_structured_match, widget, offset
                    )
                    return
            return
        start = self._window_start
        if not (start <= offset < start + _DISPLAY_WINDOW):
            return
        # Display line = omission-marker lines + newlines from window start.
        marker_lines = 2 if start > 0 else 0
        line = marker_lines + self._detail_text[start:offset].count("\n")
        scroll = self.query_one("#detail-scroll", VerticalScroll)
        scroll.scroll_to(y=max(0, line - 3), animate=False)

    def _scroll_to_structured_match(
        self, widget: TranscriptEntryWidget, offset: int
    ) -> None:
        """Put the active match row in view with a little leading context."""
        if (
            widget not in self._entry_widgets
            or not self._matches
            or self._match_idx < 0
            or self._matches[self._match_idx][0] != offset
        ):
            return
        row = widget.active_match_row()
        scroll = self.query_one("#detail-scroll", VerticalScroll)
        if row is None:
            scroll.scroll_to_widget(widget, animate=False, top=True, immediate=True)
            return
        scroll.scroll_to(
            y=max(0, widget.virtual_region.y + row - 3),
            animate=False,
            immediate=True,
        )

    # -- Actions -----------------------------------------------------------

    def action_focus_session_list(self) -> None:
        self._search_focus_handoff = None
        self._apply_pane_visibility("sessions")
        self.screen.set_focus(self.query_one("#session-table", DataTable))

    def action_escape(self) -> None:
        """Context-aware Escape: step up one level in the focus hierarchy.

        - On the detail scroll pane → back to the session list.
        - On the session list → up to the main search box (natural for quick
          refiltering without having to grab the mouse or press /).
        - Anywhere else → default to the session list.

        SearchInput has its own Escape binding, so this action only fires when
        focus is outside an input.
        """
        focused = self.focused
        if isinstance(focused, DataTable):
            self._apply_pane_visibility("sessions")
            self.screen.set_focus(self.query_one("#global-search", Input))
        elif isinstance(focused, VerticalScroll):
            self.action_focus_session_list()
        else:
            self.action_focus_session_list()

    def action_focus_global_search(self) -> None:
        self._search_focus_handoff = "global-search"
        self._apply_pane_visibility("sessions")
        self.screen.set_focus(self.query_one("#global-search", Input))

    def action_focus_detail_search(self) -> None:
        if self._selected:
            self._search_focus_handoff = "detail-search"
            self._apply_pane_visibility("detail")
            self.screen.set_focus(self.query_one("#detail-search", Input))

    def action_focus_search(self) -> None:
        """Open the search belonging to the currently active pane."""
        focused = self.focused
        if focused is not None and focused.id in {"detail-search", "detail-scroll"}:
            self.action_focus_detail_search()
        else:
            self.action_focus_global_search()

    def action_focus_left_pane(self) -> None:
        if not isinstance(self.focused, Input):
            self.action_focus_session_list()

    def action_focus_right_pane(self) -> None:
        if self._selected and not isinstance(self.focused, Input):
            self._search_focus_handoff = None
            self._apply_pane_visibility("detail")
            self.screen.set_focus(self.query_one("#detail-scroll", VerticalScroll))

    def action_switch_pane(self) -> None:
        focused = self.focused
        if focused is not None and focused.id in {"detail-search", "detail-scroll"}:
            self.action_focus_session_list()
        elif self._selected:
            self._search_focus_handoff = None
            self._apply_pane_visibility("detail")
            self.screen.set_focus(self.query_one("#detail-scroll", VerticalScroll))

    def action_toggle_focus_mode(self) -> None:
        if self.size.width < 100:
            self.action_switch_pane()
            return
        self._focus_mode = not self._focus_mode
        pane = self._pane_for_widget(self.focused)
        self._apply_pane_visibility(pane)
        self._update_context_bar()
        self._flash_status(f"Focus view: {'on' if self._focus_mode else 'off'}")

    def action_show_help(self) -> None:
        # The multiplexers are probed as the map opens, so the ``t`` entry
        # describes what is running now rather than at launch.
        self.push_screen(ShortcutHelp(multiplexer.available_names()))

    def check_action(self, action: str, parameters: tuple[object, ...]) -> bool | None:
        """Never let printable global shortcuts fire while editing search."""
        editing = self._search_focus_handoff is not None or isinstance(
            self.focused, Input
        )
        return not (editing and action in _EDIT_BLOCKED_ACTIONS)

    def action_copy_resume(self) -> None:
        if not self._selected:
            self._status.update("No session selected")
            return
        cmd = resume_command(
            self._selected.provider, self._selected.id, self._selected.cwd or None
        )
        if copy_to_clipboard(cmd):
            self._status.update(f"Copied: {cmd}")
        else:
            self._status.update(f"Clipboard failed. Command: {cmd}")

    def action_copy_session_id(self) -> None:
        """Copy the selected session's canonical id (provider:id) to the
        clipboard — the exact handle `session-browser get` resolves, so it can
        be pasted into a prompt for an agent to query the session."""
        if not self._selected:
            self._status.update("No session selected")
            return
        sid = canonical_id(self._selected)
        if copy_to_clipboard(sid):
            self._flash_status(f"Copied id: {sid}")
        else:
            self._flash_status(f"Clipboard failed. Id: {sid}", ok=False)

    def action_open_terminal(self) -> None:
        """Open the selected session's folder in a running multiplexer.

        Only multiplexers that are actually up are offered, so on a machine
        living in one of them this key never advertises the other. With exactly
        one running it hands off immediately; with both it asks, since either
        answer is a real destination and guessing would drop the user into the
        wrong one.
        """
        if not self._selected:
            self._status.update("No session selected")
            return
        targets = multiplexer.available_targets()
        if not targets:
            names = ", ".join(t.name for t in multiplexer.TARGETS)
            self._status.update(f"No multiplexer running ({names})")
            return
        if len(targets) == 1:
            self._open_in_multiplexer(targets[0])
            return
        self.push_screen(MultiplexerChoice(targets), self._open_in_multiplexer)

    def _open_in_multiplexer(self, target: multiplexer.Target | None) -> None:
        """Hand the selected session over to one multiplexer.

        Mirrors worktrees.sh for tmux and the same shape for herdr: the folder
        maps to a named session/workspace holding a window/tab that runs the
        resume command. When the browser is already inside that multiplexer we
        switch it (the browser keeps running in the background); otherwise we
        suspend the TUI so the attach can take over the terminal, and resume on
        return. ``target`` is ``None`` when the chooser was dismissed.
        """
        if target is None or not self._selected:
            return
        s = self._selected
        try:
            plan = target.module.prepare_session(
                s.provider, s.id, s.cwd, content_path=s.content_path
            )
        except target.error as exc:
            self._status.update(f"{target.name}: {exc}")
            return
        reused = " (already open)" if plan.reused else ""
        where = f"{target.name} session '{plan.label}'"
        if target.inside():
            result = self._run_handoff(plan.switch_commands())
            # Named for tmux, when tmux was the only destination; it means
            # "this browser was launched as a popup, so close it once the
            # handoff lands" and applies to whichever multiplexer that was.
            if (
                result == 0
                and os.environ.get("SESSION_BROWSER_CLOSE_ON_TMUX_SWITCH") == "1"
            ):
                self.exit()
                return
            self._status.update(f"Switched to {where}{reused}")
        else:
            with self.suspend():
                self._run_handoff(plan.attach_commands())
            self._status.update(f"Returned from {where}{reused}")

    @staticmethod
    def _run_handoff(commands: list[list[str]]) -> int:
        """Run a plan's handoff commands, stopping at the first failure.

        A plan can need more than one command (herdr focuses a tab before
        attaching a client), and the return code reported is the last one run —
        which is what decides whether a launcher-driven browser should close.
        """
        returncode = 0
        for command in commands:
            returncode = subprocess.run(command, check=False).returncode
            if returncode != 0:
                break
        return returncode

    def action_export_chat_clipboard(self) -> None:
        if not self._selected:
            self._status.update("No session selected")
            return
        text = build_chat_export(self._selected, self._detail_text)
        if copy_to_clipboard(text):
            self._status.update("Copied chat export")
        else:
            self._status.update("Clipboard failed for chat export")

    def action_export_chat_file(self) -> None:
        if not self._selected:
            self._status.update("No session selected")
            return
        try:
            path = export_chat_to_file(self._selected, self._detail_text)
        except OSError as exc:
            self._status.update(f"Export failed: {exc}")
            return
        self._status.update(f"Exported chat: {path}")

    def _update_transcript_title(self) -> None:
        """Name the projection in the pane title.

        The match counter carries the numbers, but it is one line high and
        hidden when there is nothing to count. A reader who cannot see why
        half the transcript is missing has been given a broken pane rather
        than a filtered one, so the mode is stated where the pane is named.
        """
        suffix = "  ·  MATCHING BLOCKS" if self._projecting else ""
        self.query_one("#transcript-title", Static).update(
            f"{self._transcript_title}{suffix}"
        )

    def action_toggle_matches_only(self) -> None:
        """Show only the blocks the in-session query hit, or everything.

        Refused without a query rather than silently doing nothing: the mode
        is a view of a search, and a key that appears inert is a worse answer
        than one that says what it needs.
        """
        if self._transcript is None:
            return
        if not self._search_query:
            self._flash_status("Find in session first (s or /) — then m", ok=False)
            return
        self._matches_only = not self._matches_only
        self._render_detail()

    def action_next_match(self) -> None:
        if not self._matches:
            return
        self._match_idx = (self._match_idx + 1) % len(self._matches)
        self._ensure_match_visible()
        self._render_detail()

    def action_prev_match(self) -> None:
        if not self._matches:
            return
        self._match_idx = (self._match_idx - 1) % len(self._matches)
        self._ensure_match_visible()
        self._render_detail()

    def _move_entry_focus(self, delta: int, *, tools_only: bool = False) -> None:
        if not self._entry_widgets:
            return
        candidates = [
            widget
            for widget in self._entry_widgets
            if not tools_only or widget.entry.role == "tool"
        ]
        if not candidates:
            return
        focused = self.focused
        try:
            position = candidates.index(focused)
        except ValueError:
            position = -1 if delta > 0 else 0
        target = candidates[(position + delta) % len(candidates)]
        target.focus()
        self.query_one("#detail-scroll", VerticalScroll).scroll_to_widget(
            target, animate=False, top=True, immediate=True
        )

    def action_next_entry(self) -> None:
        self._move_entry_focus(1)

    def action_prev_entry(self) -> None:
        self._move_entry_focus(-1)

    def action_next_tool(self) -> None:
        self._move_entry_focus(1, tools_only=True)

    def action_prev_tool(self) -> None:
        self._move_entry_focus(-1, tools_only=True)

    def action_refresh_sessions(self) -> None:
        self._load_sessions()

    # -- Vim-style navigation ---------------------------------------------

    def _focused_table_or_scroll(self):
        """Return the DataTable/VerticalScroll the vim keys should drive.

        Focus can also sit on a TranscriptEntryWidget (after J/K/[/] or a
        click on an entry); navigation must keep driving the transcript pane
        then, not fall through to the session table.
        """
        focused = self.focused
        if isinstance(focused, (DataTable, VerticalScroll)):
            return focused
        if isinstance(focused, TranscriptEntryWidget):
            return self.query_one("#detail-scroll", VerticalScroll)
        return None

    def _detail_window_end(self) -> int:
        return min(len(self._detail_text), self._window_start + _DISPLAY_WINDOW)

    def _detail_viewport_anchor(self) -> tuple[int, int] | None:
        """First visible entry and its row offset within the viewport."""
        if self._transcript is None:
            return None
        scroll = self.query_one("#detail-scroll", VerticalScroll)
        top = int(scroll.scroll_y)
        for widget in self._entry_widgets:
            if widget.virtual_region.bottom > top:
                return widget.entry_index, widget.virtual_region.y - top
        return None

    def _render_detail_window(self):
        """Remount one display window without re-targeting an active match."""
        if self._transcript is not None:
            return self._render_structured_entries()
        self.query_one("#detail-content", Static).update(self._compose_visible_text())
        return None

    def _restore_detail_anchor(
        self,
        transcript: Transcript | None,
        window_start: int,
        anchor: tuple[int, int] | None,
        scroll_delta: int,
    ) -> None:
        """Restore a viewport after an overlapping window replacement."""
        if self._transcript is not transcript or self._window_start != window_start:
            return
        scroll = self.query_one("#detail-scroll", VerticalScroll)
        if anchor is not None:
            entry_index, viewport_offset = anchor
            widget = next(
                (
                    candidate
                    for candidate in self._entry_widgets
                    if candidate.entry_index == entry_index
                ),
                None,
            )
            if widget is not None:
                scroll.scroll_to(
                    y=max(0, widget.virtual_region.y - viewport_offset),
                    animate=False,
                    immediate=True,
                )
            elif scroll_delta > 0:
                scroll.scroll_home(animate=False, immediate=True)
            else:
                scroll.scroll_end(animate=False, immediate=True)
        elif scroll_delta > 0:
            scroll.scroll_home(animate=False, immediate=True)
        else:
            scroll.scroll_end(animate=False, immediate=True)
        scroll.scroll_to(
            y=scroll.scroll_y + scroll_delta,
            animate=False,
            immediate=True,
        )
        scroll.focus()

    async def _restore_detail_anchor_after_mount(
        self,
        mount,
        transcript: Transcript | None,
        window_start: int,
        anchor: tuple[int, int] | None,
        scroll_delta: int,
    ) -> None:
        if mount is not None:
            await mount
        if self._transcript is not transcript or self._window_start != window_start:
            return
        self.call_after_refresh(
            self._restore_detail_anchor,
            transcript,
            window_start,
            anchor,
            scroll_delta,
        )

    def _shift_detail_window(self, start: int, scroll_delta: int) -> None:
        """Replace the bounded window while keeping the viewport continuous."""
        anchor = self._detail_viewport_anchor()
        last_start = max(0, len(self._detail_text) - _DISPLAY_WINDOW)
        self._window_start = min(last_start, max(0, start))
        mount = self._render_detail_window()
        self.run_worker(
            self._restore_detail_anchor_after_mount(
                mount,
                self._transcript,
                self._window_start,
                anchor,
                scroll_delta,
            ),
            exclusive=True,
            group="detail_window_navigation",
        )

    def _finish_detail_jump(
        self, transcript: Transcript | None, window_start: int, end: bool
    ) -> None:
        if self._transcript is not transcript or self._window_start != window_start:
            return
        scroll = self.query_one("#detail-scroll", VerticalScroll)
        if end:
            scroll.scroll_end(animate=False, immediate=True)
        else:
            scroll.scroll_home(animate=False, immediate=True)
        scroll.focus()

    async def _finish_detail_jump_after_mount(
        self,
        mount,
        transcript: Transcript | None,
        window_start: int,
        end: bool,
    ) -> None:
        if mount is not None:
            await mount
        if self._transcript is not transcript or self._window_start != window_start:
            return
        self.call_after_refresh(
            self._finish_detail_jump,
            transcript,
            window_start,
            end,
        )

    def _jump_detail_window(self, *, end: bool) -> None:
        """Mount the real head or tail, then complete the requested jump."""
        scroll = self.query_one("#detail-scroll", VerticalScroll)
        start = max(0, len(self._detail_text) - _DISPLAY_WINDOW) if end else 0
        if start != self._window_start:
            self._window_start = start
            mount = self._render_detail_window()
            self.run_worker(
                self._finish_detail_jump_after_mount(
                    mount,
                    self._transcript,
                    self._window_start,
                    end,
                ),
                exclusive=True,
                group="detail_window_navigation",
            )
        elif end:
            scroll.scroll_end(animate=False, immediate=True)
        else:
            scroll.scroll_home(animate=False, immediate=True)
        scroll.focus()

    def _scroll_detail(self, delta: int) -> None:
        """Scroll by rows, sliding the character window at either edge."""
        scroll = self.query_one("#detail-scroll", VerticalScroll)
        if delta > 0:
            crossing = scroll.scroll_y + delta >= scroll.max_scroll_y
            if crossing and self._detail_window_end() < len(self._detail_text):
                self._shift_detail_window(
                    self._window_start + _DISPLAY_WINDOW // 2,
                    delta,
                )
                return
        elif delta < 0:
            crossing = scroll.scroll_y + delta <= 0
            if crossing and self._window_start > 0:
                self._shift_detail_window(
                    self._window_start - _DISPLAY_WINDOW // 2,
                    delta,
                )
                return
        scroll.scroll_relative(y=delta, animate=False)

    def action_nav_down(self) -> None:
        target = self._focused_table_or_scroll()
        if isinstance(target, DataTable):
            target.action_cursor_down()
        elif isinstance(target, VerticalScroll):
            self._scroll_detail(1)
        else:
            self.query_one("#session-table", DataTable).action_cursor_down()

    def action_nav_up(self) -> None:
        target = self._focused_table_or_scroll()
        if isinstance(target, DataTable):
            target.action_cursor_up()
        elif isinstance(target, VerticalScroll):
            self._scroll_detail(-1)
        else:
            self.query_one("#session-table", DataTable).action_cursor_up()

    def _clear_pending_g(self) -> None:
        self._pending_g = False
        if self._pending_g_timer is not None:
            self._pending_g_timer.stop()
            self._pending_g_timer = None

    def action_nav_top(self) -> None:
        """Vim-style `gg`: the first press arms the chord, the second jumps."""
        if not self._pending_g:
            self._pending_g = True
            self._pending_g_timer = self.set_timer(1.0, self._clear_pending_g)
            return
        self._clear_pending_g()
        target = self._focused_table_or_scroll()
        if isinstance(target, VerticalScroll):
            self._jump_detail_window(end=False)
            return
        table = (
            target
            if isinstance(target, DataTable)
            else self.query_one("#session-table", DataTable)
        )
        if table.row_count:
            table.move_cursor(row=0)

    def action_nav_bottom(self) -> None:
        target = self._focused_table_or_scroll()
        if isinstance(target, VerticalScroll):
            self._jump_detail_window(end=True)
            return
        table = (
            target
            if isinstance(target, DataTable)
            else self.query_one("#session-table", DataTable)
        )
        if table.row_count:
            table.move_cursor(row=table.row_count - 1)

    def action_halfpage_down(self) -> None:
        scroll = self.query_one("#detail-scroll", VerticalScroll)
        self._scroll_detail(max(1, scroll.size.height // 2))

    def action_halfpage_up(self) -> None:
        scroll = self.query_one("#detail-scroll", VerticalScroll)
        self._scroll_detail(-max(1, scroll.size.height // 2))


def main() -> None:
    if len(sys.argv) > 1:
        from .cli import run_cli

        raise SystemExit(run_cli(sys.argv[1:]))
    app = SessionBrowser()
    app.run()


if __name__ == "__main__":
    main()
