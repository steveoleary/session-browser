"""Compare the Python and ripgrep raw candidate scans on local sessions."""

from __future__ import annotations

import argparse
import statistics
import time
from pathlib import Path
from unittest.mock import patch

from session_browser.discovery import discover_all
from session_browser.transcript import _prefilter_file, search_session_contents


def timed(sessions, query: str, repeats: int, *, python_only: bool):
    times = []
    hits = set()
    context = (
        patch("session_browser.transcript._rg_candidate_paths", return_value=None)
        if python_only
        else _NullContext()
    )
    with context:
        for _ in range(repeats):
            start = time.perf_counter()
            hits = search_session_contents(sessions, query)
            times.append(time.perf_counter() - start)
    return hits, times


class _NullContext:
    def __enter__(self):
        return self

    def __exit__(self, *_exc):
        return False


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "queries",
        nargs="*",
        default=["kookaburra", "ripgrep", "transcript", "zzzzneverpresent"],
    )
    parser.add_argument("--repeats", type=int, default=4)
    args = parser.parse_args()
    sessions = [
        s
        for s in discover_all()
        if (path := _prefilter_file(s)) is not None and path.is_file()
    ]
    total_bytes = sum(Path(_prefilter_file(s)).stat().st_size for s in sessions)
    print(f"corpus: {len(sessions)} files, {total_bytes / 1_000_000:.1f} MB")
    print("query\thits\tparity\tpython median\trg median\tspeedup")
    for query in args.queries:
        old_hits, old = timed(sessions, query, args.repeats, python_only=True)
        new_hits, new = timed(sessions, query, args.repeats, python_only=False)
        old_med, new_med = statistics.median(old), statistics.median(new)
        print(
            f"{query}\t{len(new_hits)}\t{old_hits == new_hits}\t"
            f"{old_med:.3f}s\t{new_med:.3f}s\t{old_med / new_med:.2f}x"
        )


if __name__ == "__main__":
    main()
