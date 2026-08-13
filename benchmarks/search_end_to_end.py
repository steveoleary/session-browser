"""Measure end-to-end transcript search on the locally discovered corpus."""

from __future__ import annotations

import argparse
import statistics
import time

from session_browser.discovery import discover_all
from session_browser.transcript import search_session_contents


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "queries", nargs="*", default=["kookaburra", "zzzzneverpresent", "transcript"]
    )
    parser.add_argument("--provider")
    parser.add_argument("--repeats", type=int, default=7)
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("--repeats must be >= 1")

    sessions = discover_all()
    if args.provider:
        sessions = [s for s in sessions if s.provider == args.provider]
    providers: dict[str, int] = {}
    for session in sessions:
        providers[session.provider] = providers.get(session.provider, 0) + 1

    print(f"corpus: {len(sessions)} sessions {providers}")
    print("query\thits\tmedian\tmin\tmax")
    for query in args.queries:
        samples: list[float] = []
        hits: set[str] = set()
        for _ in range(args.repeats):
            start = time.perf_counter()
            hits = search_session_contents(sessions, query)
            samples.append(time.perf_counter() - start)
        print(
            f"{query}\t{len(hits)}\t{statistics.median(samples):.3f}s\t"
            f"{min(samples):.3f}s\t{max(samples):.3f}s"
        )


if __name__ == "__main__":
    main()
