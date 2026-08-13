"""Capture the three curated session-browser TUI screenshots for the README.

Runs the real TUI headless via Textual's ``run_test``, pointed at a freshly
regenerated demo corpus, drives each curated state with real key presses, and
writes an SVG with a stable filename (no timestamps). Nothing here ever reads
the real HOME: discovery resolves every provider root through ``Path.home()``,
so a fresh fake HOME is generated and installed into the environment before
the app launches.

The three states, matching the shot spec:

    01-list-160x45          default list (160x45), SQLite migration session
                            selected, transcript loaded.
    02-search-sqlite-120x40 global query ``sqlite`` (120x40), migration
                            session selected, search input focused.
    03-detail-focus-wal-120x40  focus mode via ``z`` (120x40), in-session
                            query ``WAL``, detail search focused, first match
                            active.

Invocation (any working directory; the script chdirs to the repo root so the
demo corpus's "here" sessions point at the checkout):

    source .venv/bin/activate
    python scripts/capture_demo.py
        Regenerate a throwaway demo HOME, drive the three states, write
        docs/screenshots/01-list-160x45.svg,
        docs/screenshots/02-search-sqlite-120x40.svg and
        docs/screenshots/03-detail-focus-wal-120x40.svg.
        Existing files are overwritten; that is the point — regeneration.
    python scripts/capture_demo.py --check
        Drive the same three states and verify every curated invariant, but
        write nothing. Exit 0 only if all three states reproduce exactly.
    python scripts/capture_demo.py --home /tmp/demo-home --output-dir /tmp/shots
        Keep the regenerated corpus at a named path and write the SVGs
        somewhere other than docs/screenshots.

``--home`` is always force-regenerated (the corpus timestamps anchor to now),
so every run starts from a fresh, reproducible history. Without ``--home`` a
temporary directory is used and removed on exit.
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
import tempfile
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# The script lives in scripts/; session_browser is only importable once the
# repo root above is on sys.path, so these two imports follow the bootstrap.
from session_browser.app import SessionBrowser
from session_browser.demo import generate

# The session every curated state centres on: the SQLite widgets migration in
# the demo corpus (session_browser.demo). Looked up by id, never by row, so a
# reordering of the newest-first list cannot silently retarget the capture.
MIGRATION_ID = "8f2c4a1b-9d3e-4f5b-8c6a-1a2b3c4d5e6f"

# The global filter is debounced in the app (_FILTER_DEBOUNCE = 0.15s); give
# the timer room to fire and the content-search worker to be registered before
# waiting on it.
_FILTER_SETTLE = 0.3


class ShotError(AssertionError):
    """A curated state did not reproduce."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ShotError(message)


async def _select_migration(app, pilot) -> None:
    """Move the list cursor onto the migration session and wait for load.

    RowHighlighted drives the selection; if the cursor already sits on the
    row (or highlight events were suppressed mid-rebuild) the fallback calls
    ``_select_session`` directly so the state is guaranteed either way.
    """
    table = app.query_one("#session-table")
    idx = next(i for i, s in enumerate(app._table_sessions) if s.id == MIGRATION_ID)
    table.move_cursor(row=idx)
    await pilot.pause()
    if app._selected is None or app._selected.id != MIGRATION_ID:
        app._select_session(str(idx))
        await pilot.pause()
    await app.workers.wait_for_complete()
    await pilot.pause()


async def _drive_default_list(app, pilot) -> None:
    """160x45: the untouched default list with the migration session selected."""
    await app.workers.wait_for_complete()
    await pilot.pause()
    await _select_migration(app, pilot)


async def _drive_global_search(app, pilot) -> None:
    """120x40: global query ``sqlite``, migration selected, search focused."""
    await app.workers.wait_for_complete()
    await pilot.pause()
    await pilot.press("/")
    await pilot.pause()
    await pilot.press("s", "q", "l", "i", "t", "e")
    await pilot.pause(_FILTER_SETTLE)
    await app.workers.wait_for_complete()
    await pilot.pause()


async def _drive_detail_focus(app, pilot) -> None:
    """120x40: focus mode (``z``), in-session query ``WAL``, detail focused."""
    await app.workers.wait_for_complete()
    await pilot.pause()
    await _select_migration(app, pilot)
    await pilot.press("z")
    await pilot.pause()
    await pilot.press("s")
    await pilot.pause()
    await pilot.press("W", "A", "L")
    # Two pauses let the post-refresh scroll-to-match land before capture.
    await pilot.pause()
    await pilot.pause()


def _verify_default_list(app) -> None:
    _require(app._filter_query == "", "global search unexpectedly active")
    _require(not app._focus_mode, "focus mode unexpectedly active")
    selected = app._selected.id if app._selected else None
    _require(selected == MIGRATION_ID, f"selected {selected!r}, want {MIGRATION_ID}")
    _require(app._transcript is not None, "transcript not loaded")
    _require(bool(app._detail_text), "detail pane has no content")


def _verify_global_search(app) -> None:
    box = app.query_one("#global-search")
    _require(box.value == "sqlite", f"global search {box.value!r}, want 'sqlite'")
    _require(app._filter_query == "sqlite", f"filter {app._filter_query!r}")
    _require(
        [s.id for s in app._filtered] == [MIGRATION_ID],
        f"filtered {[s.id for s in app._filtered]}",
    )
    selected = app._selected.id if app._selected else None
    _require(selected == MIGRATION_ID, f"selected {selected!r}, want {MIGRATION_ID}")
    _require(box.has_focus, "global search does not have focus")
    _require(app._transcript is not None, "transcript not loaded")


def _verify_detail_focus(app) -> None:
    box = app.query_one("#detail-search")
    _require(app._focus_mode, "focus mode not active")
    _require(box.value == "WAL", f"detail search {box.value!r}, want 'WAL'")
    _require(app._search_query == "WAL", f"in-session query {app._search_query!r}")
    _require(box.has_focus, "detail search does not have focus")
    _require(len(app._matches) > 0, "no matches for 'WAL'")
    _require(app._match_idx == 0, f"match index {app._match_idx}, want 0")
    _require(app._transcript is not None, "transcript not loaded")


@dataclass(frozen=True)
class Shot:
    """One curated state: how to reach it, how to prove it, what to name it."""

    name: str
    filename: str
    size: tuple[int, int]
    drive: Callable[[SessionBrowser, object], Awaitable[None]]
    verify: Callable[[SessionBrowser], None]


SHOTS = (
    Shot(
        "01-list-160x45",
        "01-list-160x45.svg",
        (160, 45),
        _drive_default_list,
        _verify_default_list,
    ),
    Shot(
        "02-search-sqlite-120x40",
        "02-search-sqlite-120x40.svg",
        (120, 40),
        _drive_global_search,
        _verify_global_search,
    ),
    Shot(
        "03-detail-focus-wal-120x40",
        "03-detail-focus-wal-120x40.svg",
        (120, 40),
        _drive_detail_focus,
        _verify_detail_focus,
    ),
)


async def _capture(home: Path, output_dir: Path, check_only: bool) -> None:
    """Drive every shot; verify each state before writing anything."""
    for shot in SHOTS:
        app = SessionBrowser()
        async with app.run_test(size=shot.size) as pilot:
            await shot.drive(app, pilot)
            shot.verify(app)
            if check_only:
                svg = app.export_screenshot()
                _require(bool(svg.strip()), "empty screenshot export")
                print(f"PASS  {shot.name}  (would write {output_dir / shot.filename})")
            else:
                path = Path(app.save_screenshot(shot.filename, path=str(output_dir)))
                print(f"PASS  {shot.name}  ->  {path}")


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="python scripts/capture_demo.py",
        description=(
            "Regenerate a fake agent-session history, drive the three curated "
            "TUI states headless, and write stable-named SVG screenshots."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "examples:\n"
            "  python scripts/capture_demo.py\n"
            "      Regenerate a throwaway demo HOME and write\n"
            "      docs/screenshots/{01-list-160x45,02-search-sqlite-120x40,\n"
            "      03-detail-focus-wal-120x40}.svg (overwriting existing files).\n"
            "  python scripts/capture_demo.py --check\n"
            "      Drive the three states and verify every invariant, write nothing.\n"
            "  python scripts/capture_demo.py --home /tmp/demo-home \\\n"
            "      --output-dir /tmp/shots\n"
            "      Keep the corpus at a named path, write the SVGs elsewhere.\n"
            "\n"
            "Run from any directory; the script chdirs to the repo root so the\n"
            "corpus's 'here' sessions point at the checkout. Requires the repo's\n"
            "venv (textual is imported at module load)."
        ),
    )
    parser.add_argument(
        "--home",
        type=Path,
        default=None,
        metavar="DIR",
        help=(
            "regenerate the demo corpus under DIR (always overwritten); "
            "defaults to a temporary directory removed on exit"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        metavar="DIR",
        help=(
            "write SVG files here (default: docs/screenshots under the repo "
            "root); created if missing"
        ),
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help=(
            "verify the three curated states reproduce without writing any "
            "screenshots; exit non-zero on the first mismatch"
        ),
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)

    # Anchor the corpus and the TUI at the repo root so "here" sessions and
    # the Project column are stable regardless of where the script is invoked.
    os.chdir(REPO_ROOT)

    temp_owner: tempfile.TemporaryDirectory | None = None
    try:
        if args.home is not None:
            home = args.home.resolve()
        else:
            temp_owner = tempfile.TemporaryDirectory(prefix="session-browser-demo-")
            home = Path(temp_owner.name) / "demo-home"
        output_dir = (
            args.output_dir.resolve()
            if args.output_dir
            else REPO_ROOT / "docs" / "screenshots"
        )

        try:
            result = generate(home, force=True)
        except OSError as exc:
            print(
                f"error: could not regenerate demo HOME at {home}: {exc}",
                file=sys.stderr,
            )
            return 1
        print(
            f"Demo HOME: {home} ({result.total} sessions: {result.claude} claude, "
            f"{result.codex} codex, {result.opencode} opencode)"
        )

        # Discovery reads Path.home() at scan time; install the fake HOME
        # before the app launches so the real one is never touched.
        os.environ["HOME"] = str(home)
        if str(Path.home()) != str(home):
            print(
                f"error: Path.home() resolved to {Path.home()}, not {home}",
                file=sys.stderr,
            )
            return 1

        if args.check:
            print("Check mode: driving the curated states, writing nothing.")
        else:
            output_dir.mkdir(parents=True, exist_ok=True)
            print(f"Writing screenshots to {output_dir}")

        asyncio.run(_capture(home, output_dir, args.check))
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    finally:
        if temp_owner is not None:
            temp_owner.cleanup()
    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
