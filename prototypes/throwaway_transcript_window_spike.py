"""THROWAWAY SPIKE — reference only; never import from session_browser.

Runs the real SessionBrowser with the real transcript parser, search logic,
Textual layout, and TranscriptEntryWidget. Only the bounded-window navigation
is replaced here; production code never imports this module.

    uv run --frozen python prototypes/throwaway_transcript_window_spike.py
"""

from __future__ import annotations

from time import perf_counter

from textual.containers import VerticalScroll
from textual.widgets import DataTable, Static

from session_browser.app import (
    _DISPLAY_WINDOW,
    SessionBrowser,
    TranscriptEntryWidget,
)


WINDOW_STEP = _DISPLAY_WINDOW // 2


class ThrowawayTranscriptWindowSpike(SessionBrowser):
    """Real app plus disposable window navigation and visible measurements."""

    TITLE = "THROWAWAY transcript-window spike"

    def __init__(self) -> None:
        super().__init__()
        self._spike_action = "open session; try G, gg, paging, search"
        self._spike_remount_ms = 0.0
        self._spike_peak_ms = 0.0

    def _render_structured_entries(self) -> None:
        started = perf_counter()
        old_indices = tuple(widget.entry_index for widget in self._entry_widgets)
        super()._render_structured_entries()
        new_indices = tuple(widget.entry_index for widget in self._entry_widgets)
        if old_indices != new_indices:
            self.call_after_refresh(
                self._record_remount,
                started,
                self._window_start,
                len(new_indices),
            )

    def _record_remount(self, started: float, window_start: int, count: int) -> None:
        if window_start != self._window_start:
            return
        self._spike_remount_ms = (perf_counter() - started) * 1000
        self._spike_peak_ms = max(self._spike_peak_ms, self._spike_remount_ms)
        self._show_spike_state(count)

    def _show_spike_state(self, count: int | None = None) -> None:
        if not self._detail_text:
            return
        if count is None:
            count = len(self._entry_widgets)
        end = min(len(self._detail_text), self._window_start + _DISPLAY_WINDOW)
        self.query_one("#transcript-title", Static).update(
            "THROWAWAY SPIKE  ·  real widgets, bounded window"
        )
        self.query_one("#status-bar", Static).update(
            f"SPIKE · {self._spike_action} · chars "
            f"{self._window_start // 1000:,}k–{end // 1000:,}k/"
            f"{len(self._detail_text) // 1000:,}k · {count} widgets · "
            f"{self._spike_remount_ms:.1f} ms (peak {self._spike_peak_ms:.1f}) · "
            "search = full transcript"
        )

    def _viewport_anchor(self) -> tuple[int, int] | None:
        scroll = self.query_one("#detail-scroll", VerticalScroll)
        top = int(scroll.scroll_y)
        for widget in self._entry_widgets:
            if widget.virtual_region.bottom > top:
                return widget.entry_index, widget.virtual_region.y - top
        return None

    def _shift_window(self, start: int, page_delta: int, action: str) -> None:
        anchor = self._viewport_anchor()
        last_start = max(0, len(self._detail_text) - _DISPLAY_WINDOW)
        self._window_start = min(last_start, max(0, start))
        self._spike_action = action
        self._render_detail()
        self.call_after_refresh(self._restore_anchor, anchor, page_delta)

    def _restore_anchor(self, anchor: tuple[int, int] | None, page_delta: int) -> None:
        scroll = self.query_one("#detail-scroll", VerticalScroll)
        if anchor is not None:
            entry_index, screen_offset = anchor
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
                    y=max(0, widget.virtual_region.y - screen_offset),
                    animate=False,
                    immediate=True,
                )
        scroll.scroll_relative(y=page_delta, animate=False)
        self._show_spike_state()

    def action_nav_bottom(self) -> None:
        target = self._focused_table_or_scroll()
        if isinstance(target, VerticalScroll) and self._detail_text:
            self._spike_action = "G → true tail"
            self._window_start = max(0, len(self._detail_text) - _DISPLAY_WINDOW)
            self._render_detail()
            self.call_after_refresh(target.scroll_end, animate=False)
            return
        super().action_nav_bottom()

    def _do_session_search(self, query: str) -> None:
        if query:
            self._spike_action = "search → full transcript"
        super()._do_session_search(query)

    def action_next_match(self) -> None:
        self._spike_action = "n → next full-transcript match"
        super().action_next_match()

    def action_prev_match(self) -> None:
        self._spike_action = "N → previous full-transcript match"
        super().action_prev_match()

    def action_nav_top(self) -> None:
        if not self._pending_g:
            self._pending_g = True
            self._pending_g_timer = self.set_timer(1.0, self._clear_pending_g)
            return
        self._clear_pending_g()
        target = self._focused_table_or_scroll()
        if isinstance(target, VerticalScroll) and self._detail_text:
            self._spike_action = "gg → true head"
            self._window_start = 0
            self._render_detail()
            self.call_after_refresh(target.scroll_home, animate=False)
            return
        if isinstance(target, DataTable):
            target.move_cursor(row=0)

    def action_halfpage_down(self) -> None:
        scroll = self.query_one("#detail-scroll", VerticalScroll)
        amount = max(1, scroll.size.height // 2)
        window_end = self._window_start + _DISPLAY_WINDOW
        crossing_edge = scroll.scroll_y + amount >= scroll.max_scroll_y
        if crossing_edge and window_end < len(self._detail_text):
            self._shift_window(
                self._window_start + WINDOW_STEP,
                amount,
                "ctrl+d → next window (anchored)",
            )
            return
        scroll.scroll_relative(y=amount, animate=False)

    def action_halfpage_up(self) -> None:
        scroll = self.query_one("#detail-scroll", VerticalScroll)
        amount = max(1, scroll.size.height // 2)
        crossing_edge = scroll.scroll_y - amount <= 0
        if crossing_edge and self._window_start > 0:
            self._shift_window(
                self._window_start - WINDOW_STEP,
                -amount,
                "ctrl+u → previous window (anchored)",
            )
            return
        scroll.scroll_relative(y=-amount, animate=False)


def main() -> None:
    ThrowawayTranscriptWindowSpike().run()


if __name__ == "__main__":
    main()
