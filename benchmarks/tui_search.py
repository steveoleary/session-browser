"""Drive the real TUI headlessly against the local corpus and print what a
search shows: how many rows, how long until it settled, the top rows, and
where a session you name landed.

    python benchmarks/tui_search.py "some words" "another query"
    python benchmarks/tui_search.py "some words" --expect claude:1a2b
    python benchmarks/tui_search.py "some words" --repo /tmp/baseline-main

For a change to the TUI's global search -- its waves, ranking or row text --
this is the look at the real thing that a unit test over three fake sessions
cannot give. It judges nothing: read the rows. The settle time includes this
machine's noise and is for comparing two revisions run back to back, not for
quoting.

``--repo`` runs another checkout's code, for a before/after. It is needed
because an editable install answers ``import session_browser`` through an
import hook that outranks ``sys.path``: with ``PYTHONPATH`` alone, the
"baseline" run silently imports this checkout and the two runs agree
perfectly. The path printed first is the one that was imported.

Your own live session is in the corpus and usually quotes the query, so it
tends to sit at the top.
"""

from __future__ import annotations

import argparse
import asyncio
import re
import sys
import time


def _use_checkout(repo: str) -> None:
    sys.meta_path[:] = [
        finder
        for finder in sys.meta_path
        if "editable" not in repr(finder).lower()
        and "editable" not in type(finder).__module__.lower()
    ]
    sys.path.insert(0, repo)


def _plain(cell: object) -> str:
    return re.sub(r"\[/?[^\]]*\]", "", str(cell))


async def _run(args: argparse.Namespace) -> None:
    import session_browser
    from session_browser.app import SessionBrowser

    print(f"code: {session_browser.__file__}")
    app = SessionBrowser()
    async with app.run_test(size=(args.width, 50)) as pilot:
        await app.workers.wait_for_complete()
        await pilot.pause()
        print(f"corpus: {len(app._all_sessions)} sessions")
        box = app.query_one("#global-search")
        table = app.query_one("#session-table")
        for query in args.queries:
            await pilot.press("slash")
            box.value = ""
            await pilot.pause()
            start = time.perf_counter()
            box.value = query
            await pilot.pause(0.3)  # past the keystroke debounce
            instant = len(app._filtered)
            # A search is up to two waves, each a worker that may start the next.
            for _ in range(4):
                await app.workers.wait_for_complete()
                await pilot.pause()
            settled = time.perf_counter() - start
            rows = app._table_sessions
            print(
                f"\n{query!r}: {len(app._filtered)} rows "
                f"({instant} on the keystroke), settled in {settled:.1f}s"
            )
            for i in range(min(args.rows, len(rows))):
                cells = [_plain(c) for c in table.get_row_at(i)]
                print(f"  {i + 1:>3}  " + "  ".join(c[:110] for c in cells))
            if not args.expect:
                continue
            at = next(
                (
                    i
                    for i, s in enumerate(rows)
                    if args.expect in f"{s.provider}:{s.id}"
                ),
                None,
            )
            if at is None:
                print(f"  {args.expect}: not in the results")
                continue
            app._select_session(str(at))
            await app.workers.wait_for_complete()
            await pilot.pause()
            marked = sorted({app._detail_text[s:e].lower() for s, e in app._matches})
            print(
                f"  {args.expect}: row {at + 1}; opened with "
                f"{len(app._matches)} highlights over {marked}"
            )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("queries", nargs="+")
    parser.add_argument(
        "--expect", help="part of a canonical session id; report the row it lands on"
    )
    parser.add_argument("--rows", type=int, default=6, help="top rows to print")
    parser.add_argument(
        "--width",
        type=int,
        default=210,
        help="terminal width; the summary column needs 62",
    )
    parser.add_argument(
        "--repo", help="run this checkout's code instead of the install"
    )
    args = parser.parse_args()
    if args.repo:
        _use_checkout(args.repo)
    asyncio.run(_run(args))


# Search uses worker processes, and each re-imports this file as it starts:
# without the guard every worker launches a TUI of its own.
if __name__ == "__main__":
    main()
