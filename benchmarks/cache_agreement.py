"""Check, on your own corpus, that the caches answer as the raw files do.

    python benchmarks/cache_agreement.py
    python benchmarks/cache_agreement.py "a phrase" "file.name" --provider claude

The suite holds the entry cache and the scan memo to the stateless path over
fixtures. This asks the same three questions of the history actually on this
machine, which holds record shapes no fixture has thought of yet:

1. discovery returns the same sessions with the memo off, empty, and full;
2. every session's entries, loaded back from the cache, equal a fresh parse,
   warnings included;
3. each query finds the same sessions, counts and offsets cached as stateless.

It builds its caches in a temporary directory and deletes them, so it neither
reads nor disturbs the real ones, and it exits non-zero on any disagreement.
Run it after changing a parser, a scanner, or either cache. The second check
parses the whole history twice over; allow about ten seconds per 5 GB.

A session being written to while this runs can differ between two reads of
it through no fault of the caches. Pass ``--until`` to leave recent ones out.
"""

from __future__ import annotations

import argparse
import dataclasses
import os
import sys
import tempfile
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from datetime import UTC, datetime, timedelta


def _session_view(session) -> dict:
    view = {f.name: getattr(session, f.name) for f in dataclasses.fields(session)}
    view.pop("live", None)  # a process can exit between two discoveries
    return view


def _check_entries(session) -> tuple[str, str, str]:
    from session_browser import entry_cache, transcript

    parsed_warnings: list[str] = []
    try:
        parsed = list(transcript._parse_entries(session, parsed_warnings))
    except Exception as exc:
        return "unparseable", transcript.canonical_id(session), repr(exc)[:80]
    ref = transcript._entry_ref(session)
    if ref is None:
        return "not cacheable", transcript.canonical_id(session), ""
    read_warnings: list[str] = []
    read = list(transcript.iter_entries(session, read_warnings))
    stored = entry_cache.load(ref, transcript.TranscriptEntry)
    if stored is None:
        return "not kept", transcript.canonical_id(session), ""
    same = (
        stored == (parsed, parsed_warnings)
        and read == parsed
        and read_warnings == parsed_warnings
        and all(
            type(a.metadata) is type(b.metadata)
            for a, b in zip(read, parsed, strict=True)
        )
    )
    return ("ok" if same else "MISMATCH"), transcript.canonical_id(session), ""


def _answers(sessions, query) -> dict:
    from session_browser import transcript

    return {
        transcript.canonical_id(r.session): (
            r.match_count,
            r.total_entries,
            [(m.entry_index, m.offsets) for m in r.matches],
        )
        for r in transcript.search_sessions(sessions, query)
        if r.match_count or r.unreadable
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "queries",
        nargs="*",
        default=["session", "transcript.py", "error 404", "zzzzneverpresent"],
    )
    parser.add_argument("--provider")
    parser.add_argument(
        "--until",
        type=float,
        default=30.0,
        metavar="MINUTES",
        help="leave out sessions active in the last MINUTES (default 30)",
    )
    args = parser.parse_args()

    failures = 0
    with tempfile.TemporaryDirectory(prefix="sb-cache-agreement-") as cache_home:
        os.environ["XDG_CACHE_HOME"] = cache_home
        from session_browser.discovery import discover_all

        providers = [args.provider] if args.provider else None
        cutoff = (datetime.now(UTC) - timedelta(minutes=args.until)).isoformat()

        def discover() -> list:
            return [
                s
                for s in discover_all(providers=providers)
                if (s.updated_at or "") < cutoff
            ]

        os.environ["SESSION_BROWSER_CACHE"] = "0"
        sessions = discover()
        reference = [_session_view(s) for s in sessions]
        del os.environ["SESSION_BROWSER_CACHE"]
        for state in ("memo empty", "memo full"):
            agrees = [_session_view(s) for s in discover()] == reference
            failures += not agrees
            print(f"discovery, {state}: {'same' if agrees else 'DIFFERENT'}")

        files = [s for s in sessions if s.provider != "opencode"]
        with ProcessPoolExecutor() as pool:
            outcomes = list(pool.map(_check_entries, files, chunksize=8))
        tally = Counter(kind for kind, _id, _why in outcomes)
        print(f"entries, {len(files)} sessions: {dict(tally)}")
        for kind, ident, why in outcomes:
            if kind == "MISMATCH":
                failures += 1
            if kind != "ok":
                print(f"  {kind}: {ident} {why}")

        for query in args.queries:
            cached = _answers(sessions, query)
            os.environ["SESSION_BROWSER_CACHE"] = "0"
            stateless = _answers(sessions, query)
            del os.environ["SESSION_BROWSER_CACHE"]
            agrees = cached == stateless
            failures += not agrees
            print(
                f"search {query!r}: {len(stateless)} sessions, "
                f"{'same' if agrees else 'DIFFERENT'}"
            )
            if not agrees:
                for ident in sorted(cached.keys() ^ stateless.keys())[:5]:
                    print(f"  only one side: {ident}")
    print("agreed" if not failures else f"{failures} disagreement(s)")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
