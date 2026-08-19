"""Non-interactive CLI for agents: list, search, get.

Output contracts (stable shapes the consuming agent can rely on):
- session_dict everywhere carries "duration_seconds" (created_at →
  updated_at span; null when a timestamp is missing) — with an entry
  count, the triage signal that catches real sessions whose summary
  looks like noise (summaries are often just the first user message).
- "updated_at" is the one name for last activity, on session rows and on
  stats' provider rows alike. Two names for it meant reading the wrong
  one returned null rather than raising, which is unfalsifiable at the
  call site: "no timestamp recorded" and "wrong key" look identical.
- session_dict "branch" is null when a provider has no branch field at all,
  and "" only when the provider supports it but recorded no active branch.
- list:       {"sessions": [session_dict + "total_entries", ...],
               "counts": {"returned", "readable", "empty", "unreadable"},
               "warnings?": [...]}
              (total_entries is null for an unreadable transcript; with
               --around, each session also carries "offset" — signed
               distance from the anchor like "-3h20m"/"+4d02h" — and
               results sort nearest-first)
- get (json): {"session": {...}, "entries": [...], "warnings": [...],
               "total_entries": N, "entry_range?": {"start", "end"},
               "roles?": [...], "clip?": N}
              (--entries windows by absolute position; --head/--tail
               window the *kept* entries — with --role that means e.g.
               the last N user turns, and entry_range is then omitted;
               kept entries carry their absolute "entry_index" — the same
               key, and the same numbering, as a search snippet's.
               "clip" reports the per-entry char cap: stdout defaults to
               4000 with a "… [clipped …]" marker inside the text,
               --output defaults to complete, --clip N overrides either,
               0 disables.
               Several ids batch into {"sessions": [payload, ...],
               "skipped?": [{id, error}, ...]}; a single id keeps the
               flat legacy shape)
- get (text): Markdown document on stdout (or --output file); several
              ids concatenate with "---" separators
- search (json): {"query": "..." | [...], "mode": ..., "filters": {...},
                   "results": [{session_dict plus match_count,
                                total_entries, first_match?, last_match?,
                                summary_matches?, snippets_omitted?,
                                snippets/entries depending on mode}, ...],
                   "skipped?": [{id, error}, ...], "warnings?": [...]}
                  (each snippet is {"role", "entry_index", "text"} plus
                   "query" when several phrases were given. **role is the
                   provenance signal**: "tool" means the phrase was in
                   something the agent read — a file, a diff, a command's
                   output — while "user"/"assistant" mean somebody wrote it.
                   A transcript records everything that passed through the
                   session, so a literal hit proves the text occurred, not
                   that anyone discussed it; --mode ids carries no roles and
                   so nominates candidates rather than confirming them.
                   first_match/last_match are entry indices usable directly
                   with `get --entries`; multiple query phrases are OR'd in
                   one scan, "query" is then a list; with --around each
                   result also carries "offset" as in list. Matching is
                   case-insensitive and markdown-insensitive: backticks and
                   asterisks are stripped from both sides, so "SELECT only"
                   finds "`SELECT` only". Summaries are scanned too — a
                   session can match by summary alone (match_count 0,
                   "summary_matches" lists the phrases). Snippets are
                   capped per result (--max-snippets, default 20), with
                   the overflow counted in "snippets_omitted". When --limit
                   truncates matches, a warning reports how many were
                   dropped and their date range)
- search (text): tab-separated header lines on stdout + skipped diagnostics
                 and advisory warnings on stderr
- search --output-dir (json): {"output_dir": "<dir>", "manifest": "<path>",
                                "files_written": N, "results": N}
- search --output-dir (text): "wrote N file(s) to <dir> (manifest: <path>)"
- stats (json):  {"total": N, "warnings?": [...],
                  "transcript_health": "not_checked",
                  "activity": {"days", "start", "end", "counts": [N, ...]},
                  "providers": [{"provider", "count", "percent",
                                 "updated_at"}, ...],
                  "top_cwds": [{"cwd", "count"}, ...],
                  "oldest", "newest", "filters": {...}}
                 ("transcript_health" is always "not_checked": stats never
                  opens a transcript, so its total counts sessions
                  *discovered*, not sessions readable — corrupt and
                  zero-entry records are in the number, and only `list`,
                  which does open them, names them in its warnings.
                  "top_cwds" decomposes the total by directory, which is
                  also the cheapest check on a --repo/--cwd substring that
                  matched more projects than intended.
                  counts is one bucket per local calendar day, oldest→newest:
                  bucket i is activity.start + i days. It renders on a single
                  line, and the request echo in "filters" comes last, so the
                  blocks that answer a question survive a `| head -N`)
- stats (text):  human dashboard (provider bars, activity sparkline,
                  top working directories)
- output confirmation (json): {"written": "<path>", "id": "prov:id",
                                "warnings": [...]}
- output confirmation (text): "wrote <path>" plus warnings on stderr
- errors:     {"error": {"code", "message", "details"?}} on stderr, exit 1
Text formats are tab-separated one-line-per-record equivalents.
"""

from __future__ import annotations

import argparse
import contextlib
import difflib
import json
import os
import re
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from pathlib import Path

from .discovery import ALL_SCANNERS, Session, discover_all
from .resume import _filename_part
from .transcript import (
    FILTER_ROLES,
    Transcript,
    TranscriptUnreadable,
    canonical_id,
    entry_matches_roles,
    iter_entries,
    load_transcript,
    make_snippet,
    normalize_match_text,
    render_markdown,
    search_sessions,
    session_duration_seconds,
    session_to_dict,
    transcript_to_dict,
)


class CliError(Exception):
    """Expected failure: human message + machine-readable code, exit 1."""

    def __init__(self, message: str, code: str = "error", details: dict | None = None):
        super().__init__(message)
        self.code = code
        self.details = details or {}


# argparse treats any token starting with "-" as an option, so the documented
# relative bounds ("--since -1d") die with "expected one argument" before our
# code runs. Merge a date flag with a following relative-looking value into
# the --flag=value form argparse accepts. Values that merge but aren't valid
# relative bounds (e.g. "-30x") still reach _parse_date, which reports them
# as a structured invalid_date error. --window documents the dashless form
# ("1d") but habits from --since/--until make "-1d" likely, so it merges too.
_DATE_FLAGS = ("--since", "--until", "--window")


def _normalize_argv(argv: list[str]) -> list[str]:
    out: list[str] = []
    i = 0
    while i < len(argv):
        tok = argv[i]
        if tok in _DATE_FLAGS and i + 1 < len(argv) and re.match(r"-\d", argv[i + 1]):
            out.append(f"{tok}={argv[i + 1]}")
            i += 2
        else:
            out.append(tok)
            i += 1
    return out


def run_cli(argv: list[str]) -> int:
    args = build_parser().parse_args(_normalize_argv(argv))
    try:
        return args.handler(args)
    except CliError as exc:
        _print_error(exc, getattr(args, "format", "text"))
        return 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="session-browser",
        description="Browse and retrieve coding-agent sessions. "
        "Run with no arguments to launch the TUI.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_list = sub.add_parser("list", help="discover sessions with metadata filters")
    _add_filter_args(p_list)
    p_list.add_argument(
        "--sort",
        choices=["recent", "oldest"],
        default="recent",
        help="result order: last activity (default), or oldest-first "
        "(for targets buried under newer sessions); ignored with "
        "--around, whose order is nearest-first",
    )
    p_list.add_argument("--format", choices=["json", "text"], default="json")
    p_list.set_defaults(handler=cmd_list)

    p_get = sub.add_parser("get", help="retrieve complete session transcripts")
    p_get.add_argument(
        "session_id",
        nargs="+",
        help="one or more sessions: canonical provider:id, a "
        "raw id, or a unique prefix of either form (like "
        "git short hashes); several ids batch into one "
        'call (JSON wraps them as {"sessions": [...]})',
    )
    p_get.add_argument("--format", choices=["text", "json"], default="text")
    p_get.add_argument(
        "--output", help="write to this file instead of stdout (single session only)"
    )
    p_get.add_argument(
        "--overwrite",
        action="store_true",
        help="allow replacing an existing output file",
    )
    window = p_get.add_mutually_exclusive_group()
    window.add_argument(
        "--entries",
        metavar="A:B",
        help="return only entries A:B (0-based, inclusive; "
        "open ends allowed: '40', '40:', ':40'); indices "
        "match the entry_index carried by search snippets and by "
        "get's own JSON entries, and search's first_match/last_match, "
        "and are always absolute, even with --role",
    )
    window.add_argument(
        "--head",
        type=int,
        metavar="N",
        help="return only the first N entries (with --role: the first N kept entries)",
    )
    window.add_argument(
        "--tail",
        type=int,
        metavar="N",
        help="return only the last N entries (how it ended; "
        "with --role: the last N kept entries, e.g. the "
        "last N user turns)",
    )
    p_get.add_argument(
        "--role",
        action="append",
        metavar="ROLE[,ROLE]",
        help="keep only entries with these roles (repeatable "
        "or comma-separated): user, assistant, tool, "
        "system, or error (failed tool calls); kept "
        "entries carry their absolute indices, usable "
        "with --entries",
    )
    p_get.add_argument(
        "--clip",
        type=int,
        metavar="N",
        help="cap each entry's text at N chars with an "
        "omission marker — entry-count windows don't "
        "bound bytes, and one entry can embed a huge "
        "tool dump. Default: 4000 to stdout, 0 (off) "
        "with --output; 0 disables",
    )
    p_get.set_defaults(handler=cmd_get)

    p_search = sub.add_parser(
        "search",
        help="case-insensitive, markdown-insensitive literal "
        "search over complete transcripts and summaries",
    )
    p_search.add_argument(
        "query",
        nargs="+",
        help="literal phrase(s); multiple phrases are OR'd "
        "in a single scan (see --match-all)",
    )
    _add_filter_args(p_search)
    p_search.add_argument(
        "--mode", choices=["ids", "snippets", "full"], default="snippets"
    )
    p_search.add_argument(
        "--match-all",
        action="store_true",
        help="with multiple phrases, keep only sessions that contain every phrase",
    )
    p_search.add_argument(
        "--sort",
        choices=["recent", "matches", "oldest"],
        default="recent",
        help="result order: last activity (default), "
        "match count, or oldest-first (for targets "
        "buried under newer matches)",
    )
    p_search.add_argument(
        "--context",
        type=int,
        default=200,
        help="snippet context characters (default 200)",
    )
    p_search.add_argument(
        "--max-snippets",
        type=int,
        metavar="N",
        help="cap snippets per result (default 20; 0 = "
        "unlimited); omitted snippets are reported "
        "in snippets_omitted",
    )
    p_search.add_argument("--format", choices=["json", "text"], default="json")
    p_search.add_argument(
        "--output-dir", help="write manifest.json (and full transcripts) here"
    )
    p_search.add_argument(
        "--overwrite",
        action="store_true",
        help="allow replacing existing artifact files",
    )
    p_search.set_defaults(handler=cmd_search)

    p_stats = sub.add_parser(
        "stats",
        help="summarize sessions: provider breakdown, daily "
        "activity, top working directories",
        # An epilog rather than a note in the module docstring, which no
        # agent can reach: this exact field was documented there and a test
        # reader still had to ask what it meant.
        epilog=(
            'JSON output carries "transcript_health": "not_checked", always. '
            "stats never opens a transcript, so its total counts sessions "
            "discovered, not sessions readable -- corrupt and zero-entry "
            "records are inside the number. Only `list` opens them, and it "
            'names them in its "warnings". "top_cwds" decomposes the total '
            "by directory, which is also the cheapest check on a --repo or "
            "--cwd substring that matched more projects than intended."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    _add_filter_args(p_stats)
    p_stats.add_argument(
        "--days",
        type=int,
        default=30,
        help="activity window in days ending today (default 30)",
    )
    p_stats.add_argument(
        "--top",
        type=int,
        default=5,
        help="how many top working directories to show (default 5)",
    )
    p_stats.add_argument("--format", choices=["text", "json"], default="text")
    p_stats.set_defaults(handler=cmd_stats)

    return parser


def _add_filter_args(p: argparse.ArgumentParser) -> None:
    # Validated rather than filtered with. An unknown value used to filter
    # every session out and exit 0, so `--provider claude-code` -- the
    # product's actual name, and a natural guess -- reported that there were
    # no Claude Code sessions at all. Rejecting it names the three valid
    # values instead, which is what the CLI's other enum flags already do.
    # ``type`` runs before the choices check, so CLAUDE still works.
    p.add_argument(
        "--provider",
        type=str.lower,
        choices=sorted(ALL_SCANNERS),
        help="provider to scan (case-insensitive)",
    )
    p.add_argument(
        "--repo",
        help="case-insensitive substring of the project name: the last "
        "segment of the session's project root where the provider records "
        "one, else of its directory. Not a git remote. Only opencode "
        "records a root, so for claude and codex a session run in a git "
        "worktree is named after the worktree directory rather than the "
        "project it belongs to. Read cwd off the hits when worktrees are "
        "in play",
    )
    p.add_argument("--cwd", help="case-insensitive substring of working directory")
    p.add_argument(
        "--exclude-cwd",
        metavar="SUBSTR",
        action="append",
        help="drop sessions whose working directory contains this "
        "case-insensitive substring; repeatable (a session is "
        "dropped if any pattern matches), composes with --cwd/"
        "--here, and is applied before --limit so excluded "
        "sessions do not consume the result budget",
    )
    p.add_argument(
        "--here",
        action="store_true",
        help="scope to the current working directory (exact "
        "path-prefix); excludes sessions with no recorded cwd",
    )
    p.add_argument(
        "--since",
        help="lower bound on last activity (newest message, not "
        "file mtime): ISO 8601, or relative like "
        "-30m/-2h/-1d/-1w; a date-only value means from the "
        "start of that day",
    )
    p.add_argument(
        "--until",
        help="upper bound on last activity (newest message, not "
        "file mtime): ISO 8601, or relative like "
        "-30m/-2h/-1d/-1w; a date-only value includes that "
        "whole day",
    )
    p.add_argument(
        "--around",
        metavar="SESSION",
        help="temporal anchor: keep sessions whose last activity "
        "falls within --window of this session's (same id "
        "forms as `get`: provider:id, raw id, or a unique "
        "prefix); the anchor itself is excluded, results "
        'sort nearest-first and carry a signed "offset" '
        "from the anchor; not combinable with --since/--until",
    )
    p.add_argument(
        "--window",
        metavar="DUR",
        help="half-width of the --around window on each side: "
        "30m/2h/1d/1w (default 1d)",
    )
    p.add_argument(
        "--include-current",
        action="store_true",
        help="include the caller's own live session(s) in results; "
        "by default they are auto-excluded (detected from the "
        "agent's session-id env var, e.g. CLAUDE_CODE_SESSION_ID)",
    )
    p.add_argument(
        "--limit",
        type=int,
        help="maximum number of results; omit it and every match is "
        "returned, unbounded. When it does truncate, a warning reports how "
        "many were dropped and their date range",
    )


def _print_error(exc: CliError, fmt: str) -> None:
    if fmt == "json":
        payload: dict = {"error": {"code": exc.code, "message": str(exc)}}
        if exc.details:
            payload["error"]["details"] = exc.details
        print(json.dumps(payload), file=sys.stderr)
    else:
        print(f"error: {exc}", file=sys.stderr)


# Relative bounds like "-30m" / "-2h" / "-1d" / "-1w" resolve against "now",
# so an agent can say "before this task started" without computing a timestamp.
_RELATIVE_RE = re.compile(r"^-(\d+)([smhdw])$")
_REL_UNITS = {"s": "seconds", "m": "minutes", "h": "hours", "d": "days", "w": "weeks"}


def _relative_delta(value: str) -> timedelta | None:
    """Return the timedelta for a relative bound like '-30m', else None."""
    m = _RELATIVE_RE.match(value)
    if not m:
        return None
    return timedelta(**{_REL_UNITS[m.group(2)]: int(m.group(1))})


def _parse_date(value: str, flag: str) -> datetime:
    delta = _relative_delta(value)
    if delta is not None:
        return datetime.now(UTC) - delta
    try:
        dt = datetime.fromisoformat(value)
    except ValueError:
        raise CliError(
            f"invalid {flag} date: {value!r} "
            f"(expected ISO 8601 or relative like -30m/-2h/-1d)",
            code="invalid_date",
        ) from None
    return dt if dt.tzinfo else dt.replace(tzinfo=UTC)


def _session_time(s: Session) -> datetime | None:
    raw = s.updated_at or s.created_at
    if not raw:
        return None
    try:
        dt = datetime.fromisoformat(raw)
    except ValueError:
        return None
    return dt if dt.tzinfo else dt.replace(tzinfo=UTC)


_DEFAULT_WINDOW = "1d"


def _parse_window(value: str | None) -> timedelta:
    raw = (value or _DEFAULT_WINDOW).strip()
    delta = _relative_delta(raw if raw.startswith("-") else f"-{raw}")
    if delta is None:
        raise CliError(
            f"invalid --window: {value!r} (expected a duration like 30m/2h/1d/1w)",
            code="invalid_window",
        )
    return delta


def _resolve_anchor(sessions: list[Session], args) -> tuple[Session, datetime] | None:
    """Resolve --around into (anchor session, anchor time), or None when the
    flag wasn't given. The anchor is looked up in the *unfiltered* session
    list, so it resolves even when other filters would drop it (the caller's
    own live session is the expected anchor: "what else was I doing around
    this?")."""
    ident = getattr(args, "around", None)
    if not ident:
        if getattr(args, "window", None):
            raise CliError("--window requires --around", code="invalid_filter")
        return None
    if getattr(args, "since", None) or getattr(args, "until", None):
        raise CliError(
            "--around cannot be combined with --since/--until "
            "(it derives both bounds from the anchor; adjust "
            "--window instead)",
            code="invalid_filter",
        )
    anchor = resolve_session(sessions, ident)
    ts = _session_time(anchor)
    if ts is None:
        raise CliError(
            f"session {canonical_id(anchor)} has no parseable "
            f"timestamp to anchor --around on",
            code="invalid_filter",
        )
    return anchor, ts


# Agents export their own session id into the environment of processes they
# spawn, so session-browser reads it from its *own* inherited environment — the
# agent never has to identify or pass anything. The map is additive: drop in a
# new {provider: env var} entry as other agents are confirmed. Empty for an
# agent that exports nothing, in which case nothing is excluded (safe no-op).
_CURRENT_SESSION_ENV = {
    "claude": "CLAUDE_CODE_SESSION_ID",
    "codex": "CODEX_THREAD_ID",
}


def _current_session_ids() -> set[str]:
    """Canonical ids (provider:id) of the caller's own live session(s).

    Env vars accumulate down the spawn chain — a Claude session that launches
    Codex carries both CLAUDE_CODE_SESSION_ID and CODEX_THREAD_ID — so this set
    naturally covers the whole live parent/child chain, not just the innermost
    caller. The values match the ids `discovery` assigns (Claude file stem,
    Codex session_meta id), so canonical-id equality is exact."""
    ids: set[str] = set()
    for provider, var in _CURRENT_SESSION_ENV.items():
        val = os.environ.get(var, "").strip()
        if val:
            ids.add(f"{provider}:{val}")
    return ids


def apply_filters(
    sessions: list[Session],
    args,
    *,
    apply_limit: bool = True,
    warnings: list[str] | None = None,
) -> list[Session]:
    """Metadata filters + recency sort. `search` defers the limit until after
    matching so --limit bounds results, not the candidate set. When *warnings*
    is provided, advisory notes (e.g. sessions --here dropped for missing cwd)
    are appended to it."""
    out = sessions
    if args.provider:
        p = args.provider.lower()
        out = [s for s in out if s.provider.lower() == p]
    if args.repo:
        q = args.repo.lower()
        candidates = out
        blank = sum(1 for s in out if not (s.repository or "").strip())
        out = [s for s in out if q in (s.repository or "").lower()]
        # An empty result and an empty *field* are indistinguishable to the
        # caller, and this flag spent its whole life returning the first while
        # meaning the second. Report the blanks whenever the filter comes up
        # empty, so "no sessions in that repo" cannot be read off a filter
        # that had nothing to match against.
        if not out and blank and warnings is not None:
            warnings.append(
                f"--repo matched nothing, and {blank} session(s) have no "
                f"recorded project name — a session run outside any project "
                f"directory has none. Those can only be reached by --cwd "
                f"<path fragment>, a content search, or --provider."
            )
        if not out and warnings is not None:
            _append_near_miss_warning(
                warnings,
                flag="--repo",
                query=args.repo,
                values=[s.repository for s in candidates if s.repository],
            )
    if args.cwd:
        q = args.cwd.lower()
        candidates = out
        blank = sum(1 for s in out if not (s.cwd or "").strip())
        out = [s for s in out if q in (s.cwd or "").lower()]
        if not out and blank and warnings is not None:
            warnings.append(
                f"--cwd matched nothing, and {blank} session(s) have no "
                f"recorded working directory. Those can only be reached by "
                f"a content search or --provider."
            )
        if not out and warnings is not None:
            _append_near_miss_warning(
                warnings,
                flag="--cwd",
                query=args.cwd,
                values=[s.cwd for s in candidates if s.cwd],
                path_components=True,
            )
    around = _resolve_anchor(sessions, args)
    since = until = until_excl = None
    if around is not None:
        anchor, anchor_ts = around
        half = _parse_window(getattr(args, "window", None))
        since, until = anchor_ts - half, anchor_ts + half
        out = [s for s in out if s is not anchor]
    else:
        if args.since:
            since = _parse_date(args.since, "--since")
        if args.until:
            until = _parse_date(args.until, "--until")
            # date-only (e.g. "2026-06-10") includes the whole day; relative
            # bounds ("-30m") are precise instants, so they must not get the
            # +1 day bump.
            if "T" not in args.until and _relative_delta(args.until) is None:
                until_excl = until + timedelta(days=1)
    if since or until:
        kept = []
        missing_time = 0
        for s in out:
            ts = _session_time(s)
            if ts is None:
                missing_time += 1
                continue  # date filters need a parseable time
            if since and ts < since:
                continue
            if until_excl is not None and ts >= until_excl:
                continue
            if until_excl is None and until is not None and ts > until:
                continue
            kept.append(s)
        out = kept
        if missing_time and warnings is not None:
            warnings.append(
                f"date filters excluded {missing_time} session(s) with no "
                f"parseable last-activity timestamp; they cannot be placed "
                f"inside or outside the requested window"
            )
    if getattr(args, "here", False):
        base = os.getcwd()
        missing = sum(1 for s in out if not (s.cwd or "").strip())
        out = [
            s
            for s in out
            if s.cwd and (s.cwd == base or s.cwd.startswith(base + os.sep))
        ]
        if missing and warnings is not None:
            warnings.append(
                f"--here excluded {missing} session(s) with no recorded cwd; "
                f"they can't be project-scoped. "
                f"To reach them, drop --here and use --cwd <name>, a content "
                f"search, or --provider."
            )
    if not getattr(args, "include_current", False):
        current = _current_session_ids()
        if current:
            dropped = [s for s in out if canonical_id(s) in current]
            if dropped:
                out = [s for s in out if canonical_id(s) not in current]
                if warnings is not None:
                    ids = ", ".join(sorted(canonical_id(s) for s in dropped))
                    warnings.append(
                        f"excluded your own live session(s): {ids} "
                        f"(pass --include-current to keep them)"
                    )
    # Last of the filters, and deliberately so. Membership is unaffected by
    # position — these all AND together — but the reported count is not: run
    # before --since/--here and it counts sessions the caller was never going
    # to see, so `removed 27` lands on a listing that only lost 25. A warning
    # that overstates is the same defect class as one that stays silent.
    #
    # Kept as its own guarded block rather than folded into the --cwd
    # comprehension above: that one runs on every list/search/stats call, and
    # an `and not ...` clause there would charge every caller per-session for
    # a flag almost none of them pass. Unused, this costs one falsy check per
    # invocation; used, it runs over the smallest candidate set there is.
    # Still well ahead of --limit, which is the ordering the feature exists for.
    if excluded := _exclude_patterns(args):
        before = len(out)
        kept = []
        for s in out:
            # Lower once per session, not once per pattern. Measured 1.21x on
            # a 1500-session corpus with 3 patterns; `any(p in (s.cwd or
            # "").lower() ...)` re-lowers for every pattern. No loop opcode
            # probe covers apply_filters, so nothing but this comment and the
            # tests defend it.
            c = (s.cwd or "").lower()
            if not any(p in c for p in excluded):
                kept.append(s)
        out = kept
        # A session with no recorded cwd has nothing to match against, so a
        # non-empty pattern never excludes it. --here is what drops those.
        if (dropped := before - len(out)) and warnings is not None:
            warnings.append(
                f"--exclude-cwd removed {dropped} session(s) matching "
                f"{', '.join(excluded)}"
            )
    out = sorted(out, key=lambda s: s.sort_key, reverse=True)
    if around is not None:
        # Nearest-first is the order --around asks about; the stable sort
        # keeps equal distances in recency order.
        anchor_ts = around[1]
        out.sort(key=lambda s: abs((_session_time(s) or anchor_ts) - anchor_ts))
    if apply_limit and args.limit is not None:
        out = _limited(out, args.limit)
    return out


def _append_near_miss_warning(
    warnings: list[str],
    *,
    flag: str,
    query: str,
    values: list[str],
    path_components: bool = False,
) -> None:
    """Suggest close recorded values after a free-text filter matches nothing.

    The filters remain literal substrings and are never broadened silently.
    For cwd values, compare against path components as well as full paths so
    ``coffe_run`` can suggest ``/work/coffee_run`` instead of being judged
    dissimilar merely because of its parent directories.
    """
    by_key: dict[str, str] = {}
    for value in values:
        key = value.lower()
        by_key.setdefault(key, value)
        if path_components:
            for component in Path(value).parts:
                if component not in (os.sep, ""):
                    by_key.setdefault(component.lower(), value)
    close = difflib.get_close_matches(query.lower(), by_key, n=3, cutoff=0.8)
    suggestions = list(dict.fromkeys(by_key[key] for key in close))
    if suggestions:
        rendered = ", ".join(repr(value) for value in suggestions)
        warnings.append(
            f"{flag} {query!r} matched nothing; did you mean {rendered}? "
            f"The filter was not changed."
        )


def _exclude_patterns(args) -> list[str]:
    """Lowercased --exclude-cwd patterns, or [] when the flag was not passed.

    Blank values are a hard error rather than a no-op: the empty string is a
    substring of every string, so `--exclude-cwd ""` would silently return an
    empty corpus. A filter that silently drops everything is the failure this
    flag is written to avoid, so it fails closed and says why."""
    raw = getattr(args, "exclude_cwd", None)
    if not raw:
        return []
    patterns = []
    for value in raw:
        pattern = value.strip().lower()
        if not pattern:
            raise CliError(
                "--exclude-cwd needs a non-empty substring; an empty value "
                "matches every working directory and would exclude every "
                "session",
                code="invalid_filter",
            )
        patterns.append(pattern)
    return patterns


def _limited(items: list, limit: int) -> list:
    if limit < 0:
        raise CliError("--limit must be >= 0", code="invalid_filter")
    return items[:limit]


# Discovery hints: --provider (or a provider:id handle) names the only
# scanner worth running, so skip the other providers' file/DB scans entirely.
# apply_filters/resolve_session still see a correct (smaller) candidate set.


def _discover(args) -> list[Session]:
    provider = getattr(args, "provider", None)
    if not provider:
        return discover_all()
    providers = [provider]
    # An --around anchor may live under a different provider than the one
    # being filtered to ("codex sessions around this claude session"), so
    # its provider must be scanned too for the anchor to resolve.
    around_provider, sep, _ = (getattr(args, "around", None) or "").partition(":")
    if sep and around_provider in ALL_SCANNERS and around_provider != provider:
        providers.append(around_provider)
    return discover_all(providers=providers)


def _discover_for_idents(idents: list[str]) -> list[Session]:
    providers: set[str] = set()
    for ident in idents:
        provider, sep, _ = ident.partition(":")
        if not (sep and provider in ALL_SCANNERS):
            return discover_all()
        providers.add(provider)
    return discover_all(providers=sorted(providers))


def _entry_count(s: Session) -> int | None:
    """Transcript length without materializing it; None when unreadable."""
    try:
        return sum(1 for _ in iter_entries(s, []))
    except (TranscriptUnreadable, OSError):
        return None


def _human_duration(seconds: int | None) -> str:
    if seconds is None:
        return "-"
    m, s = divmod(seconds, 60)
    h, m = divmod(m, 60)
    if h:
        return f"{h}h{m:02d}m"
    return f"{m}m" if m else f"{s}s"


def _signed_offset(ts: datetime, anchor: datetime) -> str:
    """Signed distance from the --around anchor; negative = before it."""
    seconds = round((ts - anchor).total_seconds())
    sign = "-" if seconds < 0 else "+"
    days, rem = divmod(abs(seconds), 86400)
    if days:
        return f"{sign}{days}d{rem // 3600:02d}h"
    return sign + _human_duration(abs(seconds))


_DEFAULT_CLIP = 4000


def _resolve_clip(args) -> int:
    """Per-entry char cap for get. Explicit --clip always wins; otherwise
    stdout defaults to _DEFAULT_CLIP because --head/--tail/--entries bound
    entry *count*, not bytes — a single entry embedding a giant tool dump
    can dwarf everything else — while --output stays complete (files are
    the documented home for full transcripts)."""
    if args.clip is not None:
        if args.clip < 0:
            raise CliError(
                "--clip must be >= 0 (0 disables clipping)", code="invalid_filter"
            )
        return args.clip
    return 0 if args.output else _DEFAULT_CLIP


def _clip_transcript(t: Transcript, clip: int) -> Transcript:
    if not clip:
        return t
    entries = list(t.entries)
    changed = False
    for i, e in enumerate(entries):
        if len(e.text) > clip:
            marker = (
                f"\n… [clipped {len(e.text) - clip} chars — "
                f"pass --clip 0 for the full entry]"
            )
            entries[i] = replace(e, text=e.text[:clip] + marker)
            changed = True
    return Transcript(t.session, entries, t.warnings) if changed else t


def _id_sample(ids: list[str], cap: int = 5) -> str:
    shown = ", ".join(ids[:cap])
    return shown + (f", … ({len(ids) - cap} more)" if len(ids) > cap else "")


def cmd_list(args) -> int:
    warnings: list[str] = []
    discovered = _discover(args)
    around = _resolve_anchor(discovered, args)
    # Defer the limit so the drop can be counted and reported. It is still
    # applied before the entry counts below, which are the only I/O here —
    # a listing reads exactly as many transcripts as it prints.
    sessions = apply_filters(discovered, args, apply_limit=False, warnings=warnings)
    if around is None and args.sort == "oldest":
        # apply_filters returns recency-descending; --around's nearest-first
        # order answers a different question, so leave it alone.
        sessions.reverse()
    if args.limit is not None and len(sessions) > args.limit >= 0:
        warnings.append(_truncation_warning(args, sessions[args.limit :]))
    if args.limit is not None:
        sessions = _limited(sessions, args.limit)
    # Entry counts open every listed transcript — I/O-bound reads, so
    # overlap them (same rationale as search's per-session scan pool).
    counts: list[int | None] = []
    if sessions:
        with ThreadPoolExecutor(max_workers=8) as ex:
            counts = list(ex.map(_entry_count, sessions))
        unreadable = [
            canonical_id(s) for s, n in zip(sessions, counts, strict=True) if n is None
        ]
        empty = [
            canonical_id(s) for s, n in zip(sessions, counts, strict=True) if n == 0
        ]
        if unreadable:
            warnings.append(
                f"{len(unreadable)} session(s) have unreadable transcripts "
                f"(total_entries: null) — likely corrupt or partial "
                f"records: {_id_sample(unreadable)}"
            )
        if empty:
            warnings.append(
                f"{len(empty)} session(s) parsed to zero entries — "
                f"possibly corrupt records; treat their metadata with "
                f"suspicion: {_id_sample(empty)}"
            )
    if args.format == "json":
        items = []
        for s, n in zip(sessions, counts, strict=True):
            item = session_to_dict(s)
            item["total_entries"] = n
            if around is not None:
                ts = _session_time(s)
                item["offset"] = (
                    _signed_offset(ts, around[1]) if ts is not None else None
                )
            items.append(item)
        payload: dict = {
            "sessions": items,
            "counts": {
                "returned": len(counts),
                "readable": sum(n is not None and n > 0 for n in counts),
                "empty": sum(n == 0 for n in counts),
                "unreadable": sum(n is None for n in counts),
            },
        }
        if warnings:
            payload["warnings"] = warnings
        print(json.dumps(payload, indent=2))
    else:
        for s, n in zip(sessions, counts, strict=True):
            updated = s.updated_at or s.created_at or "-"
            duration = _human_duration(session_duration_seconds(s))
            cols = [canonical_id(s), updated]
            if around is not None:
                ts = _session_time(s)
                cols.append(_signed_offset(ts, around[1]) if ts else "-")
            cols += [
                str(n) if n is not None else "-",
                duration,
                s.cwd or "-",
                s.summary or "-",
            ]
            print("\t".join(cols))
        for w in warnings:
            print(w, file=sys.stderr)
    return 0


def resolve_session(sessions: list[Session], ident: str) -> Session:
    """Resolve provider:id, a raw id, or a unique prefix of either form.

    Exact matches always win; only when nothing matches exactly does a
    git-style prefix lookup run (`claude:7645` or bare `7645`), and it
    resolves only when a single session matches."""
    provider = raw = None
    if ":" in ident:
        provider, _, raw = ident.partition(":")
        for s in sessions:
            if s.provider == provider and s.id == raw:
                return s
    candidates = [s for s in sessions if s.id == ident]
    if not candidates:
        if provider is not None:
            candidates = [
                s
                for s in sessions
                if s.provider == provider and raw and s.id.startswith(raw)
            ]
        else:
            candidates = [s for s in sessions if ident and s.id.startswith(ident)]
    if len(candidates) == 1:
        return candidates[0]
    if not candidates:
        suggestions = _close_ids(sessions, ident)
        msg = f"unknown session id: {ident}"
        if suggestions:
            msg += " — did you mean: " + ", ".join(suggestions) + "?"
        raise CliError(
            msg,
            code="unknown_session",
            details=({"suggestions": suggestions} if suggestions else None),
        )
    handles = sorted(canonical_id(s) for s in candidates)
    raise CliError(
        f"ambiguous session id {ident!r}; matches: " + ", ".join(handles),
        code="ambiguous_session",
        details={"candidates": handles},
    )


def _close_ids(sessions: list[Session], ident: str) -> list[str]:
    """Near-miss candidates for an id that matched nothing exactly — catches
    a mistyped hex group inside an otherwise-correct id, where the prefix
    lookup can't help (the typo sits mid-string). Short idents skip this:
    they were prefix attempts, not typos of a full id."""
    provider, sep, raw = ident.partition(":")
    if not sep:
        provider, raw = None, ident
    if len(raw) < 6:
        return []
    pool = [
        s for s in sessions if provider is None or s.provider == provider
    ] or sessions
    by_raw = {s.id: canonical_id(s) for s in pool}
    close = difflib.get_close_matches(raw, by_raw, n=3, cutoff=0.8)
    return [by_raw[c] for c in close]


def _check_writable(path: Path, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise CliError(
            f"refusing to overwrite existing file: {path} (pass --overwrite)",
            code="exists",
        )


def _write_text_atomic(path: Path, content: str) -> None:
    """Atomically write *content* to *path* via same-directory temp file."""
    fd: int | None = None
    tmp_path: str | None = None
    # Ensure parent directory exists (wrap OSError, including FileExistsError
    # when a regular file occupies the parent path).
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise CliError(
            f"could not write output: {exc}",
            code="write_error",
            details={"path": str(path)},
        ) from exc
    try:
        fd, tmp_path = tempfile.mkstemp(dir=path.parent, prefix=".__tmp_")
        # Transfer fd ownership to the file object so the with-block manages it.
        file_obj = os.fdopen(fd, "w", encoding="utf-8")
        fd = None
        with file_obj as f:
            f.write(content)
        os.replace(tmp_path, path)
    except OSError as exc:
        # Clean up on failure — best-effort.
        if tmp_path is not None:
            with contextlib.suppress(OSError):
                os.unlink(tmp_path)
        if fd is not None:
            with contextlib.suppress(OSError):
                os.close(fd)
        raise CliError(
            f"could not write output: {exc}",
            code="write_error",
            details={"path": str(path)},
        ) from exc


def _entry_window(args, total: int) -> tuple[int, int] | None:
    """Resolve --entries/--head/--tail into a (start, end) inclusive window
    of 0-based entry indices, or None when no windowing was requested. The
    end is clamped to the last entry; a start past the end is an error."""
    if args.head is not None or args.tail is not None:
        n = args.head if args.head is not None else args.tail
        flag = "--head" if args.head is not None else "--tail"
        if n <= 0:
            raise CliError(f"{flag} must be > 0", code="invalid_range")
        if total == 0:
            return None  # empty session: nothing to trim
        if args.head is not None:
            return (0, min(n, total) - 1)
        return (max(0, total - n), total - 1)
    if args.entries is None:
        return None
    spec = args.entries.strip()
    m = re.fullmatch(r"(\d+)|(\d*):(\d*)", spec)
    if not m or (m.group(1) is None and not m.group(2) and not m.group(3)):
        raise CliError(
            f"invalid --entries range: {spec!r} (expected N, A:B, A:, or :B)",
            code="invalid_range",
        )
    if m.group(1) is not None:
        start = end = int(m.group(1))
    else:
        start = int(m.group(2)) if m.group(2) else 0
        end = int(m.group(3)) if m.group(3) else max(total - 1, 0)
    if start >= total:
        raise CliError(
            f"--entries starts at {start} but the session has "
            f"{total} entries (indices 0..{total - 1})",
            code="invalid_range",
            details={"total_entries": total},
        )
    if end < start:
        raise CliError(
            f"invalid --entries range: {spec!r} (start after end)", code="invalid_range"
        )
    return (start, min(end, total - 1))


def _parse_roles(values: list[str] | None) -> list[str] | None:
    """Resolve repeated/comma-separated --role values into an ordered,
    validated role list, or None when the flag wasn't given."""
    if not values:
        return None
    wanted = set()
    for value in values:
        for part in value.split(","):
            part = part.strip().lower()
            if not part:
                continue
            if part not in FILTER_ROLES:
                raise CliError(
                    f"invalid --role {part!r} (choose from: {', '.join(FILTER_ROLES)})",
                    code="invalid_role",
                )
            wanted.add(part)
    return [r for r in FILTER_ROLES if r in wanted] or None


def _session_view(session: Session, args, roles):
    """Load and window one session's transcript. Returns
    (transcript, total, window, indices); raises when unreadable."""
    transcript = load_transcript(session)
    total = len(transcript.entries)
    window = None
    indices: list[int] | None = None
    if roles is not None and (args.head is not None or args.tail is not None):
        # --head/--tail bound the *kept* entries when --role is given —
        # "--role user --tail 5" means the last five user turns, not the
        # user turns among the last five raw entries. Kept entries still
        # carry absolute indices; entry_range is omitted because the kept
        # set is sparse and a contiguous span would mislead.
        role_set = set(roles)
        kept = [
            (i, e)
            for i, e in enumerate(transcript.entries)
            if entry_matches_roles(e, role_set)
        ]
        w = _entry_window(args, len(kept))
        if w is not None:
            kept = kept[w[0] : w[1] + 1]
        indices = [i for i, _ in kept]
        transcript = Transcript(
            transcript.session, [e for _, e in kept], transcript.warnings
        )
    else:
        window = _entry_window(args, total)
        start = 0
        if window is not None:
            start, end = window
            transcript = Transcript(
                transcript.session,
                transcript.entries[start : end + 1],
                transcript.warnings,
            )
        if roles is not None:
            role_set = set(roles)
            kept = [
                (i, e)
                for i, e in enumerate(transcript.entries, start=start)
                if entry_matches_roles(e, role_set)
            ]
            indices = [i for i, _ in kept]
            transcript = Transcript(
                transcript.session, [e for _, e in kept], transcript.warnings
            )
    return transcript, total, window, indices


def cmd_get(args) -> int:
    roles = _parse_roles(args.role)
    idents = args.session_id
    if args.output and len(idents) > 1:
        raise CliError(
            "--output writes a single transcript; pass one "
            "session id (for a batch, use search --output-dir "
            "or one get per file)",
            code="invalid_filter",
        )
    clip = _resolve_clip(args)
    discovered = _discover_for_idents(idents)
    sessions: list[Session] = []
    for ident in idents:
        s = resolve_session(discovered, ident)
        if all(s is not prev for prev in sessions):
            sessions.append(s)
    views = []
    skipped: list[tuple[Session, str]] = []
    for s in sessions:
        try:
            t, total, window, indices = _session_view(s, args, roles)
        except (TranscriptUnreadable, OSError) as exc:
            if len(sessions) == 1:
                raise CliError(
                    f"could not read session {canonical_id(s)}: {exc}",
                    code="unreadable_session",
                ) from exc
            skipped.append((s, str(exc)))
            continue
        views.append((s, _clip_transcript(t, clip), total, window, indices))
    if not views:
        raise CliError(
            "no readable sessions among: "
            + ", ".join(canonical_id(s) for s, _ in skipped),
            code="unreadable_session",
        )
    single = len(idents) == 1
    parse_warnings = [w for _, t, *_ in views for w in t.warnings]
    if args.format == "json":
        payloads = []
        for _s, t, total, window, indices in views:
            payload = transcript_to_dict(t, entry_indices=indices)
            payload["total_entries"] = total
            if window is not None:
                payload["entry_range"] = {"start": window[0], "end": window[1]}
            if roles is not None:
                payload["roles"] = roles
            if clip:
                payload["clip"] = clip
            payloads.append(payload)
        if single:
            content = json.dumps(payloads[0], indent=2)
        else:
            batch: dict = {"sessions": payloads}
            if skipped:
                batch["skipped"] = [
                    {"id": canonical_id(s), "error": err} for s, err in skipped
                ]
            content = json.dumps(batch, indent=2)
    else:
        docs = [
            render_markdown(
                t,
                total_entries=total,
                entry_range=window,
                entry_indices=indices,
                roles=roles,
            )
            for _, t, total, window, indices in views
        ]
        content = "\n\n---\n\n".join(docs)
    if args.output:
        path = Path(args.output)
        _check_writable(path, args.overwrite)
        _write_text_atomic(path, content)
        if args.format == "json":
            print(
                json.dumps(
                    {
                        "written": str(path),
                        "id": canonical_id(views[0][0]),
                        "warnings": parse_warnings,
                    }
                )
            )
        else:
            print(f"wrote {path}")
        if parse_warnings and args.format == "text":
            print(
                f"[{len(parse_warnings)} parse warning(s) — transcript may be partial]",
                file=sys.stderr,
            )
        return 0
    print(content)
    if args.format == "text":
        for s, err in skipped:
            print(f"skipped {canonical_id(s)}: {err}", file=sys.stderr)
        if parse_warnings:
            print(
                f"[{len(parse_warnings)} parse warning(s) — transcript may be partial]",
                file=sys.stderr,
            )
    return 0


def _summary_matches(session: Session, queries: list[str]) -> list[str]:
    """Query phrases whose normalized text appears in the session summary.

    Summaries are the only place a session's *title* lives ("Adapt SQL
    Scripts For Admin App"), and that title often never recurs in the
    transcript body — without this, the most distinctive metadata field
    is invisible to search."""
    summary = normalize_match_text(session.summary or "")
    if not summary:
        return []
    return [q for q in queries if normalize_match_text(q) in summary]


def _truncation_warning(args, dropped) -> str:
    """Make --limit truncation loud: silently dropping results reads as
    "covered everything", and under the default recency sort it is exactly
    the old sessions — the likely target — that vanish first.

    Shared by `search`, whose *dropped* are match results, and `list`, whose
    *dropped* are bare Sessions; hence the `.session` unwrap-if-present."""
    times = sorted(
        t for r in dropped if (t := _session_time(getattr(r, "session", r))) is not None
    )
    span = f" (last activity {times[0].date()} → {times[-1].date()})" if times else ""
    # `--sort matches` exists only on search, so naming it in list's hint
    # would send the reader to an unrecognized argument — the very trap this
    # warning is here to spare them.
    searching = getattr(args, "command", "search") == "search"
    what = "matches" if searching else "sessions"
    alts = "--sort oldest or --sort matches" if searching else "--sort oldest"
    noun = "matched session" if searching else "session"
    hint = {
        "recent": f"the oldest {what} were dropped first — if the target "
        f"may be old, narrow with --since/--until, or use {alts}",
        "oldest": f"the newest {what} were dropped first",
        "matches": "the lowest match counts were dropped first",
    }[args.sort]
    return f"--limit {args.limit} dropped {len(dropped)} {noun}(s){span}; {hint}"


def cmd_search(args) -> int:
    queries = [q.strip() for q in args.query]
    if not all(queries):
        raise CliError("empty search query", code="invalid_query")
    if args.context < 0:
        raise CliError("--context must be >= 0", code="invalid_filter")
    if args.max_snippets is not None and args.max_snippets < 0:
        raise CliError(
            "--max-snippets must be >= 0 (0 disables the cap)", code="invalid_filter"
        )
    warnings: list[str] = []
    discovered = _discover(args)
    around = _resolve_anchor(discovered, args)
    anchor_ts = around[1] if around is not None else None
    candidates = apply_filters(discovered, args, apply_limit=False, warnings=warnings)
    keep = args.mode == "full"
    results = search_sessions(candidates, queries, keep_entries=keep)
    summary_map = {
        canonical_id(r.session): hits
        for r in results
        if not r.unreadable and (hits := _summary_matches(r.session, queries))
    }
    matched = [
        r
        for r in results
        if not r.unreadable
        and (r.match_count > 0 or canonical_id(r.session) in summary_map)
    ]
    skipped = [r for r in results if r.unreadable]
    for r in matched:
        # Summary-only matches were often prefilter-skipped, never parsed:
        # their total_entries would read 0 and look like a corrupt record.
        if r.match_count == 0 and r.total_entries == 0:
            r.total_entries = _entry_count(r.session) or 0
    if args.match_all and len(queries) > 1:
        wanted = {q.casefold() for q in queries}
        matched = [
            r
            for r in matched
            if (
                {m.query.casefold() for m in r.matches}
                | {q.casefold() for q in summary_map.get(canonical_id(r.session), [])}
            )
            >= wanted
        ]
    if args.sort == "matches":
        # Stable: equal match counts keep their recency order.
        matched.sort(key=lambda r: r.match_count, reverse=True)
    elif args.sort == "oldest":
        _floor = datetime.min.replace(tzinfo=UTC)
        matched.sort(key=lambda r: _session_time(r.session) or _floor)
    if args.limit is not None and len(matched) > args.limit >= 0:
        warnings.append(_truncation_warning(args, matched[args.limit :]))
    if args.limit is not None:
        matched = _limited(matched, args.limit)
    if args.output_dir:
        _report_warnings_stderr(warnings)
        _report_skipped_stderr(skipped)
        return _write_search_artifacts(
            args,
            matched,
            skipped,
            warnings,
            anchor_ts=anchor_ts,
            summary_map=summary_map,
        )
    if args.format == "json":
        print(
            json.dumps(
                _search_payload(
                    args,
                    matched,
                    skipped,
                    warnings,
                    anchor_ts=anchor_ts,
                    summary_map=summary_map,
                ),
                indent=2,
            )
        )
    else:
        _print_search_text(args, matched, anchor_ts=anchor_ts, summary_map=summary_map)
        _report_warnings_stderr(warnings)
        _report_skipped_stderr(skipped)
    return 0


def _write_search_artifacts(
    args, matched, skipped, warnings=None, *, anchor_ts=None, summary_map=None
) -> int:
    """Write manifest.json (+ one Markdown transcript per result in full
    mode). All target paths are checked before anything is written."""
    out_dir = Path(args.output_dir)
    planned: list[tuple[Path, str]] = []
    items = []
    for r in matched:
        item = _result_item(
            args, r, include_entries=False, anchor_ts=anchor_ts, summary_map=summary_map
        )
        if args.mode == "full":
            fname = (
                f"{_filename_part(r.session.provider)}-"
                f"{_filename_part(r.session.id)}.md"
            )
            t = Transcript(r.session, r.entries or [], r.warnings)
            planned.append((out_dir / fname, render_markdown(t)))
            item["file"] = fname
        items.append(item)
    manifest = {
        "query": _query_value(args),
        "mode": args.mode,
        "filters": _filters_dict(args),
        "generated_at": datetime.now(UTC).isoformat(),
        "results": items,
    }
    if skipped:
        manifest["skipped"] = _skipped_list(skipped)
    if warnings:
        manifest["warnings"] = warnings
    manifest_path = out_dir / "manifest.json"
    planned.append((manifest_path, json.dumps(manifest, indent=2)))
    if not args.overwrite:
        clashes = [str(p) for p, _ in planned if p.exists()]
        if clashes:
            raise CliError(
                "refusing to overwrite: " + ", ".join(clashes) + " (pass --overwrite)",
                code="exists",
                details={"paths": clashes},
            )
    written: list[Path] = []
    try:
        for path, content in planned:
            _write_text_atomic(path, content)
            written.append(path)
    except CliError:
        # Clean up files written in this operation on failure so artifact
        # sets do not remain partially written.
        for p in written:
            with contextlib.suppress(OSError):
                p.unlink(missing_ok=True)
        raise
    summary = {
        "output_dir": str(out_dir),
        "manifest": str(manifest_path),
        "files_written": len(planned),
        "results": len(matched),
    }
    if args.format == "json":
        print(json.dumps(summary))
    else:
        print(f"wrote {len(planned)} file(s) to {out_dir} (manifest: {manifest_path})")
    return 0


def _filters_dict(args) -> dict:
    return {
        "provider": args.provider,
        "repo": args.repo,
        "cwd": args.cwd,
        "exclude_cwd": getattr(args, "exclude_cwd", None),
        "here": getattr(args, "here", False),
        "include_current": getattr(args, "include_current", False),
        "since": args.since,
        "until": args.until,
        "around": getattr(args, "around", None),
        "window": getattr(args, "window", None),
        "limit": args.limit,
    }


def _skipped_list(skipped) -> list[dict]:
    return [
        {
            "id": canonical_id(r.session),
            "error": r.warnings[0] if r.warnings else "unreadable",
        }
        for r in skipped
    ]


def _report_skipped_stderr(skipped) -> None:
    """Print a skipped-session diagnostic line for each unreadable result."""
    for r in skipped:
        err_msg = r.warnings[0] if r.warnings else "unreadable"
        print(f"skipped {canonical_id(r.session)}: {err_msg}", file=sys.stderr)


def _report_warnings_stderr(warnings) -> None:
    """Print each advisory warning (e.g. --here missing-cwd notes) to stderr."""
    for w in warnings or []:
        print(w, file=sys.stderr)


_DEFAULT_MAX_SNIPPETS = 20


def _snippet_cap(args) -> int:
    """Snippets kept per result; 0 disables. A phrase that hits 500 times
    in one session would otherwise emit 500 snippets — entry-count limits
    don't bound this, so it needs its own cap."""
    if getattr(args, "max_snippets", None) is None:
        return _DEFAULT_MAX_SNIPPETS
    return args.max_snippets


def _result_item(
    args, r, *, include_entries: bool, anchor_ts=None, summary_map=None
) -> dict:
    item = session_to_dict(r.session)
    item["match_count"] = r.match_count
    item["total_entries"] = r.total_entries
    summary_hits = (summary_map or {}).get(canonical_id(r.session))
    if summary_hits:
        # The phrase(s) found in the session's summary — a session can
        # match by summary alone (match_count 0, no snippets).
        item["summary_matches"] = summary_hits
    if anchor_ts is not None:
        ts = _session_time(r.session)
        item["offset"] = _signed_offset(ts, anchor_ts) if ts else None
    if r.matches:
        # Entry indices of the first/last matching entry: where in the
        # session the topic lives, and direct targets for `get --entries`.
        item["first_match"] = r.matches[0].entry_index
        item["last_match"] = r.matches[-1].entry_index
    if r.warnings:
        item["parse_warnings"] = r.warnings
    multi = len(args.query) > 1
    if args.mode in ("snippets", "full"):
        cap = _snippet_cap(args)
        snippets = []
        for m in r.matches:
            for off in m.offsets:
                if cap and len(snippets) >= cap:
                    break
                snippets.append(
                    {
                        "role": m.entry.role,
                        "entry_index": m.entry_index,
                        **({"query": m.query} if multi else {}),
                        "text": make_snippet(
                            m.entry.text, off, len(m.query), args.context
                        ),
                    }
                )
            else:
                continue
            break
        item["snippets"] = snippets
        omitted = r.match_count - len(snippets)
        if omitted > 0:
            item["snippets_omitted"] = omitted
    if include_entries and r.entries is not None:
        t = Transcript(r.session, r.entries, r.warnings)
        item["entries"] = transcript_to_dict(t)["entries"]
    return item


def _query_value(args):
    """Payload/manifest "query": the plain string for a single phrase
    (stable legacy shape), the list when several were OR'd."""
    return args.query[0] if len(args.query) == 1 else args.query


def _search_payload(
    args, matched, skipped, warnings=None, *, anchor_ts=None, summary_map=None
) -> dict:
    payload = {
        "query": _query_value(args),
        "mode": args.mode,
        "filters": _filters_dict(args),
        "results": [
            _result_item(
                args,
                r,
                include_entries=(args.mode == "full"),
                anchor_ts=anchor_ts,
                summary_map=summary_map,
            )
            for r in matched
        ],
    }
    if skipped:
        payload["skipped"] = _skipped_list(skipped)
    if warnings:
        payload["warnings"] = warnings
    return payload


def _print_search_text(args, matched, *, anchor_ts=None, summary_map=None) -> None:
    for r in matched:
        line = (
            f"{canonical_id(r.session)}\t{r.match_count}\t{r.session.updated_at or '-'}"
        )
        if anchor_ts is not None:
            ts = _session_time(r.session)
            line += f"\t{_signed_offset(ts, anchor_ts) if ts else '-'}"
        summary_hits = (summary_map or {}).get(canonical_id(r.session))
        tag = "\t[summary match]" if summary_hits else ""
        print(f"{line}\t{r.session.summary or '-'}{tag}")
        if args.mode == "snippets":
            cap = _snippet_cap(args)
            shown = 0
            for m in r.matches:
                for off in m.offsets:
                    if cap and shown >= cap:
                        break
                    snip = make_snippet(m.entry.text, off, len(m.query), args.context)
                    print(f"  [{m.entry.role}] {snip}")
                    shown += 1
                else:
                    continue
                break
            if r.match_count > shown:
                print(
                    f"  … {r.match_count - shown} more snippet(s) "
                    f"omitted (--max-snippets)"
                )
            print()
        elif args.mode == "full":
            t = Transcript(r.session, r.entries or [], r.warnings)
            print()
            print(render_markdown(t))


# ---------------------------------------------------------------------------
# stats — at-a-glance dashboard over the (filtered) session history
# ---------------------------------------------------------------------------

_SPARK_LEVELS = "▁▂▃▄▅▆▇█"
_BAR_WIDTH = 24


def _sparkline(counts: list[int]) -> str:
    """One glyph per bucket, scaled to the peak. Zero days render as a
    middle dot so quiet stretches stay visually distinct from low activity."""
    peak = max(counts, default=0)
    if peak == 0:
        return "·" * len(counts)
    n = len(_SPARK_LEVELS)
    return "".join(
        "·" if c == 0 else _SPARK_LEVELS[min(n - 1, (c * n - 1) // peak)]
        for c in counts
    )


def _shorten_home(path: str) -> str:
    home = str(Path.home())
    if path == home or path.startswith(home + os.sep):
        return "~" + path[len(home) :]
    return path


_INT_ARRAY_RE = re.compile(r"\[\n\s+-?\d+(?:,\n\s+-?\d+)*\n\s*\]")


def _compact_int_arrays(text: str) -> str:
    """Collapse pretty-printed integer arrays onto one line.

    json.dumps has no per-value formatting, so `activity.counts` printed one
    integer per line: a 45-day window was 45 lines, and `| head -N` — which
    is close to universal agent behaviour on a JSON tool — landed inside the
    array, where the fragment is silently misleading rather than obviously
    cut. Rewriting the rendered text is safe: json escapes newlines inside
    strings, so a literal newline is always indentation and this pattern
    cannot match data.
    """
    return _INT_ARRAY_RE.sub(lambda m: re.sub(r"\s+", "", m.group(0)), text)


def cmd_stats(args) -> int:
    if args.days <= 0:
        raise CliError("--days must be > 0", code="invalid_filter")
    if args.top < 0:
        raise CliError("--top must be >= 0", code="invalid_filter")
    warnings: list[str] = []
    sessions = apply_filters(_discover(args), args, warnings=warnings)

    # Activity buckets over local calendar days so "today" means the user's
    # today, not UTC's. Sessions outside the window still count everywhere else.
    today = datetime.now().astimezone().date()
    start = today - timedelta(days=args.days - 1)
    counts = [0] * args.days

    provider_counts: dict[str, int] = {}
    provider_last: dict[str, datetime] = {}
    cwd_counts: dict[str, int] = {}
    times: list[datetime] = []
    for s in sessions:
        provider_counts[s.provider] = provider_counts.get(s.provider, 0) + 1
        cwd = (s.cwd or "").strip()
        if cwd:
            cwd_counts[cwd] = cwd_counts.get(cwd, 0) + 1
        ts = _session_time(s)
        if ts is None:
            continue
        times.append(ts)
        prev = provider_last.get(s.provider)
        if prev is None or ts > prev:
            provider_last[s.provider] = ts
        day = ts.astimezone().date()
        if start <= day <= today:
            counts[(day - start).days] += 1

    total = len(sessions)
    providers = [
        {
            "provider": p,
            "count": n,
            "percent": round(100 * n / total) if total else 0,
            # Deliberately updated_at, the name session rows use, and not a
            # second spelling of it: one concept had two names across this
            # CLI, and reading the wrong one returned None instead of
            # raising, so a sweep looked like it had worked.
            "updated_at": (
                provider_last[p].isoformat() if p in provider_last else None
            ),
        }
        for p, n in sorted(provider_counts.items(), key=lambda kv: (-kv[1], kv[0]))
    ]
    top_cwds = sorted(cwd_counts.items(), key=lambda kv: (-kv[1], kv[0]))
    # Key order is part of the contract: agents pipe JSON through `head`, so
    # the blocks that answer a question come first and the request echo goes
    # last. See _compact_int_arrays for the other half of that.
    payload: dict = {"total": total}
    if warnings:
        payload["warnings"] = warnings
    # What this total is a count *of*, stated in the payload rather than left
    # to the reader. `list` opens the transcripts it returns and names the
    # unreadable ones in `warnings`; `stats` is forbidden from opening any
    # (see the cli.stats guard in docs/perf_budgets.json), so it aggregates
    # corrupt and empty records into the number silently. That is the correct
    # trade — the fix is not to breach the guard, it is to stop the number
    # claiming more than it knows. A constant string costs no file opens.
    payload["transcript_health"] = "not_checked"
    payload["activity"] = {
        "days": args.days,
        "start": start.isoformat(),
        "end": today.isoformat(),
        "counts": counts,
    }
    payload["providers"] = providers
    payload["top_cwds"] = [{"cwd": c, "count": n} for c, n in top_cwds[: args.top]]
    payload["oldest"] = min(times).isoformat() if times else None
    payload["newest"] = max(times).isoformat() if times else None
    payload["filters"] = _filters_dict(args)
    if args.format == "json":
        print(_compact_int_arrays(json.dumps(payload, indent=2)))
    else:
        _print_stats_text(payload, distinct_cwds=len(cwd_counts))
        _report_warnings_stderr(warnings)
    return 0


def _print_stats_text(payload: dict, *, distinct_cwds: int) -> None:
    total = payload["total"]
    if total == 0:
        print("no sessions match the given filters")
        return
    head = f"{total} sessions · {len(payload['providers'])} providers"
    if distinct_cwds:
        head += f" · {distinct_cwds} directories"
    if payload["oldest"]:
        head += f" · {payload['oldest'][:10]} → {payload['newest'][:10]}"
    print(head)

    print()
    peak = max(p["count"] for p in payload["providers"])
    name_w = max(len(p["provider"]) for p in payload["providers"])
    for p in payload["providers"]:
        bar = "█" * max(1, round(_BAR_WIDTH * p["count"] / peak))
        last = (p["updated_at"] or "")[:10] or "-"
        pct = "<1" if p["percent"] == 0 else str(p["percent"])
        print(
            f"  {p['provider']:<{name_w}}  {bar:<{_BAR_WIDTH}}  "
            f"{p['count']:>5}  {pct:>3}%  last {last}"
        )

    act = payload["activity"]
    day_peak = max(act["counts"], default=0)
    print()
    label = f"activity · last {act['days']} days"
    if day_peak:
        label += f" · peak {day_peak}/day"
    print(label)
    print(f"  {_sparkline(act['counts'])}")
    print(f"  {act['start'][5:]} → {act['end'][5:]}")

    if payload["top_cwds"]:
        print()
        print("top working directories")
        for d in payload["top_cwds"]:
            print(f"  {d['count']:>5}  {_shorten_home(d['cwd'])}")
