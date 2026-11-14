"""Non-interactive CLI for agents: list, search, get, and stats.

Each subcommand's ``--help`` is the public output contract. Keep it beside the
parser so callers can discover the runtime shape without reading source.
"""

from __future__ import annotations

import argparse
import contextlib
import difflib
import json
import os
import re
import shlex
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from datetime import UTC, date, datetime, time, timedelta
from pathlib import Path

from .config import ConfigError, load_ignore
from .discovery import ALL_SCANNERS, Session, claude_root_session, discover_all
from .resume import _filename_part
from .transcript import (
    FILTER_ROLES,
    Transcript,
    TranscriptEntry,
    TranscriptUnreadable,
    canonical_id,
    codex_fork_base,
    codex_prefix_entries,
    codex_thread_of_segment,
    entry_matches_roles,
    is_command_noise,
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

_SESSION_KEYS = (
    "id",
    "provider",
    "model",
    "session_id",
    "summary",
    "cwd",
    "branch",
    "repository",
    "created_at",
    "updated_at",
    "duration_seconds",
    "parent_id",
    "subagent_kind",
    "continued_in",
    "continued_from",
    "live",
)
_ENTRY_KEYS = ("entry_index", "role", "text", "timestamp", "metadata")

# --brief's default windows, in "first N user turns : last M assistant turns".
# Measured, not chosen: across 660 real invocations on this machine the median
# explicit --role user --head was 6 and the median --role assistant --tail was
# 8. The intent half is the softer number -- most intent reads took *every*
# user turn rather than a head window -- but an unbounded default is the wrong
# answer for a flag whose whole point is a bounded read, and it is tunable.
_DEFAULT_BRIEF = "6:8"
_FILTER_KEYS = (
    "provider",
    "repo",
    "cwd",
    "exclude_cwd",
    "here",
    "include_current",
    "no_ignore",
    "no_subagents",
    "model",
    "since",
    "until",
    "around",
    "window",
    "limit",
)
_ERROR_GROUPS = (
    ("error_envelope", ("error",)),
    ("error", ("code", "message", "details")),
)


def _json_key_contract(*groups: tuple[str, tuple[str, ...]]) -> str:
    """Render machine-checkable key sets inside human-readable help."""
    return "\n".join(
        f"JSON keys [{label}]: {', '.join(keys)}" for label, keys in groups
    )


# The tab-separated columns each text format emits, in order. Named here so
# --help can state them and a test can check the names against a real run,
# rather than leaving the shape to be rediscovered by piping and guessing.
#
# Worth stating at all because agents do not know it: across 660 real
# invocations measured on this machine, 222 piped output into jq or an inline
# python reducer to project exactly the fields these columns already carry,
# while --format text was requested 11 times. The data was not missing; the
# contract was, because --help named only the {json,text} enum.
_LIST_TEXT_COLUMNS = ("id", "updated", "entries", "duration", "cwd", "summary")
_SEARCH_TEXT_COLUMNS = ("id", "match_count", "updated", "summary")


# One sentence, in all three commands that emit a session, because the
# distinction is exactly the one a reader gets wrong: null is not "no parent".
_SUBAGENT_CONTRACT = (
    '"parent_id"/"subagent_kind" are three-state: null when the provider '
    'exposes\nno parent-child link at all (Pi), "" when it does and '
    "this session\nis not a subagent, otherwise the spawning session's id and "
    "the kind.\n"
    '"model" is spelled as that tool\'s --model flag takes it, and is null '
    "when it is not\nrecorded where discovery reads: always for Claude and "
    'OpenCode, so null is\n"not known", never "no model".\n'
    '"continued_in"/"continued_from" link one Claude conversation resumed into '
    "a new\nfile: the new session holds a copy of the old one's history, so "
    "read the newer\nand treat the pair as one. Raw ids, null when no link is "
    "recorded.\n"
    '"live" is true when a running Claude Code process holds the session: its\n'
    "transcript may still grow, so counts and endings are as of this read. "
    "null\nwhere the provider records no running processes (Codex, OpenCode, "
    "Pi) and\nfor Claude subagents."
)


def _text_column_contract(label: str, columns: tuple[str, ...], extra: str) -> str:
    return (
        f"Text columns [{label}]: {', '.join(columns)}\n"
        f"Text is tab-separated, one record per line, no header. {extra}"
    )


_LIST_EPILOG = """Output contract:
Default format: json. Text emits one tab-separated session per line. jsonl
emits the envelope on line 1 with "sessions" replaced by "records" (how many
lines follow), then one session object per line, so `| head -N` keeps whole
records and the envelope's counts and warnings.
{keys}
{text}
"warnings" and session "offset" are conditional. "total_entries" is null
when a transcript is unreadable; counts still classifies every returned row.
"preview" is present only with --preview ending: at most the latest user and
assistant entries in chronological order, or null if unreadable. It is an
excerpt, not a summary or evidence of completion. original_chars > len(text)
means clipped; expand retrieves the full entry. Empty readable sessions use [].
{subagents}
""".format(
    subagents=_SUBAGENT_CONTRACT,
    text=_text_column_contract(
        "list",
        _LIST_TEXT_COLUMNS,
        "With --around, an offset column follows updated. Warnings go to "
        "stderr, so stdout stays parseable; a missing value is '-'.",
    ),
    keys=_json_key_contract(
        ("envelope", ("sessions", "counts", "warnings")),
        ("session", _SESSION_KEYS + ("total_entries", "offset", "preview")),
        ("preview_entry", ("entry_index", "role", "text", "original_chars", "expand")),
        ("counts", ("returned", "readable", "empty", "unreadable")),
        *_ERROR_GROUPS,
    ),
)

_GET_EPILOG = """Output contract:
Default format: text (Markdown). JSON must be requested with --format json.
Text puts each entry's time before its role, in UTC ("10:12:35Z Assistant:"),
dated on the first entry and whenever the day changes; untimed entries have none.
Single JSON example: {{"session": {{"id": "claude:abc", ...}},
  "entries": [{{"entry_index": 0, "role": "user", ...}}], ...}}
Batch JSON example: {{"sessions": [{{"session": {{"id": "claude:abc", ...}},
  "entries": [...], ...}}], "skipped": [{{"id": "claude:bad", "error": "..."}}]}}
{keys}
{subagents}
"entry_range", "roles", "brief", "clip", "recent" and "budget" are conditional. Every entry has
all entry keys; entry_index is absolute and matches search snippets. A batch's
"sessions" items have exactly the single payload shape, not list rows.
"forked_from" appears only on a Codex fork, a continuation whose rollout holds
only what came after the fork: its earlier history is the first "entries"
entries of session "id" (null when that history is not on this machine).
It is reported, not spliced in, so entry indices stay the ones search reports.
A rewound Codex thread (Esc-Esc rewind) reads as one session across all its
files. The turns a rewind undid stay where they happened, each with
"rewound": true in its metadata and "(rewound)" after its role in text output:
it happened, but the conversation went on from before it.
"briefed_by" appears when the opening request names a file in another Claude
session's scratchpad (/private/tmp/claude-<uid>/<project>/<id>/scratchpad/), as
work handed to another agent through Herdr usually does: "id" is the session
that wrote the brief, "path" the brief, "present" whether that session is on
this machine. The other direction -- what a session briefed -- is a search for
"<its id>/scratchpad" with --role user.
With --output --format json, stdout is the output_confirmation shape.
""".format(
    subagents=_SUBAGENT_CONTRACT,
    keys=_json_key_contract(
        (
            "single",
            (
                "session",
                "entries",
                "warnings",
                "total_entries",
                "entry_range",
                "roles",
                "brief",
                "clip",
                "recent",
                "budget",
                "forked_from",
                "briefed_by",
            ),
        ),
        ("batch", ("sessions", "skipped")),
        ("forked_from", ("id", "entries")),
        ("briefed_by", ("id", "path", "present")),
        ("session", _SESSION_KEYS),
        ("entry", _ENTRY_KEYS),
        ("entry_range", ("start", "end")),
        ("brief", ("intent", "ending")),
        ("budget", ("limit", "returned_chars", "omitted_chars", "omitted")),
        ("omitted_entry", ("entry_index", "omitted_chars", "expand")),
        ("skipped", ("id", "error")),
        ("output_confirmation", ("written", "id", "warnings")),
        *_ERROR_GROUPS,
    ),
)

_SEARCH_RESULT_KEYS = _SESSION_KEYS + (
    "match_count",
    "total_entries",
    "summary_matches",
    "offset",
    "first_match",
    "last_match",
    "match_roles",
    "first_authored_match",
    "last_authored_match",
    "parse_warnings",
    "snippets",
    "snippets_omitted",
    "entries",
    "file",
)
_SEARCH_EPILOG = """Output contract:
Default format: json; default mode: snippets. Text emits tab-separated result
headers. jsonl emits the envelope on line 1 with "results" replaced by
"records" (how many lines follow), then one result object per line.
ids omits snippets/entries; snippets adds snippets; full adds entries.
{keys}
{text}
{subagents}
Result keys are the union across stdout and manifest variants: "file" exists
only in full-mode manifests; other conditional keys depend on matches, modes,
warnings, and --around. A multi-phrase snippet adds "query". "role" preserves
provenance: tool hits are observed text, not necessarily agent-authored text.
With any match, "match_roles" counts every match by its entry's role (role ->
count, summing to match_count, never capped by --max-snippets), and
"first_authored_match"/"last_authored_match" are the entry indices of the
first and last matching user or assistant entry: null when only tool or
system text matched.
With --role, a match counts only if its entry has one of those roles, and
everything is computed from those matches alone: which sessions are returned
(before --limit), match_count, match_roles, first/last_match, snippets and
--sort matches. A summary match is still reported but no longer qualifies a
session by itself, since a summary has no role. "roles" echoes the flag.
With --output-dir, stdout is artifact_confirmation and manifest.json uses the
artifact_manifest envelope.
""".format(
    subagents=_SUBAGENT_CONTRACT,
    text=_text_column_contract(
        "search",
        _SEARCH_TEXT_COLUMNS,
        "Result headers only: in snippets mode each is followed by INDENTED "
        "snippet lines, '[entry_index] role: text' as get prints them, and a "
        "blank line; full mode follows it with a "
        "Markdown transcript, so a header line is a record but a line is not. "
        "With --around an offset column follows updated; a summary-only hit "
        "appends a [summary match] column.",
    ),
    keys=_json_key_contract(
        (
            "envelope",
            ("query", "mode", "filters", "roles", "results", "skipped", "warnings"),
        ),
        ("filters", _FILTER_KEYS),
        ("result", _SEARCH_RESULT_KEYS),
        ("snippet", ("role", "entry_index", "query", "text")),
        ("entry", _ENTRY_KEYS),
        ("skipped", ("id", "error")),
        (
            "artifact_confirmation",
            ("output_dir", "manifest", "files_written", "results"),
        ),
        (
            "artifact_manifest",
            (
                "query",
                "mode",
                "filters",
                "roles",
                "generated_at",
                "results",
                "skipped",
                "warnings",
            ),
        ),
        *_ERROR_GROUPS,
    ),
)

_STATS_EPILOG = """Output contract:
Default format: text (human dashboard). JSON must be requested with --format json.
{keys}
"warnings" is conditional. transcript_health is always "not_checked": stats
counts discovered sessions without opening transcripts. activity.counts has one
bucket per local calendar day from start through end. top_cwds helps validate
whether a --repo or --cwd substring matched more projects than intended.
projects groups by the inferred project name (session "repository", the
same name --repo matches), so a project's worktree directories count as one;
each carries its total, how many of those are subagents, a count per
provider, and its last activity. --top caps projects and top_cwds alike.
subagents decomposes total: "linked" name a parent, "unlinked" are subagents
whose provider recorded none, and "unknown" are sessions from a provider that
exposes no link, so they are neither known to be subagents nor known not to be.
""".format(
    keys=_json_key_contract(
        (
            "envelope",
            (
                "total",
                "warnings",
                "transcript_health",
                "activity",
                "providers",
                "subagents",
                "projects",
                "top_cwds",
                "oldest",
                "newest",
                "filters",
            ),
        ),
        ("filters", _FILTER_KEYS),
        ("activity", ("days", "start", "end", "counts")),
        ("provider", ("provider", "count", "percent", "updated_at")),
        ("project", ("repository", "count", "subagents", "providers", "updated_at")),
        ("subagents", ("total", "linked", "unlinked", "unknown", "kinds")),
        ("subagent_kind", ("kind", "count")),
        ("top_cwd", ("cwd", "count")),
        *_ERROR_GROUPS,
    )
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


def _output_path(value: str) -> str | None:
    """Normalise --output, treating the conventional '-' as stdout.

    Taken literally, '-' is a legal filename, so the tool used to create a file
    called '-' in the working directory and report 'wrote -'. The next run then
    failed with 'refusing to overwrite existing file: -', which reads as a tool
    error rather than as the caller's own slip. Since omitting --output already
    means stdout, '-' has no other useful meaning, so it maps onto that. A file
    genuinely named '-' is still reachable as './-'.
    """
    return None if value == "-" else value


_CALLS_TEXT_COLUMNS = ("timestamp", "id", "entry_index", "tool", "input")
_CALL_KEYS = (
    "id",
    "entry_index",
    "timestamp",
    "tool",
    "input",
    "input_chars",
    "result_index",
    "result",
    "result_chars",
    "is_error",
    "exit_code",
)

_CALLS_EPILOG = """Output contract:
Default format: json. jsonl emits the envelope on line 1 with "calls" replaced
by "records" (how many lines follow), then one call per line.
{keys}
{text}
One record per tool call whose input contains PATTERN (case-insensitive), with
the output that answered it. "input" is the call as get prints it, tool name
and arguments; "result" is the output entry, null when none was recorded.
Both are clipped to --clip characters; *_chars give the full lengths.
"is_error" is true when the provider marked the output failed, false when it
did not, and null when no output was recorded; "exit_code" is null unless the
provider recorded one (Codex does). A call is paired with its output by
the provider's call id where it records one (Claude), else by order: each
output answers the oldest unanswered call before it, of the same tool. Sessions come newest first, as
list orders them (--sort oldest reverses it); a session's calls are in order.
"counts" appears only with --count-by: the word that follows PATTERN in each
call's input, with how many calls used it, most first -- a subcommand table.
{subagents}
""".format(
    subagents=_SUBAGENT_CONTRACT,
    text=_text_column_contract(
        "calls",
        _CALLS_TEXT_COLUMNS,
        "Each call header is followed by one INDENTED line, '-> ' and the "
        "result on one line (or '-> (no result recorded)'), so a header line "
        "is a record but a line is not. With --count-by, text is instead "
        "'count<TAB>word' per line.",
    ),
    keys=_json_key_contract(
        (
            "envelope",
            (
                "pattern",
                "tool",
                "errors",
                "filters",
                "calls",
                "counts",
                "skipped",
                "warnings",
            ),
        ),
        ("filters", _FILTER_KEYS),
        ("call", _CALL_KEYS),
        ("count", ("word", "count")),
        ("skipped", ("id", "error")),
        *_ERROR_GROUPS,
    ),
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="session-browser",
        description="Browse and retrieve coding-agent sessions. "
        "Run with no arguments to launch the TUI.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_list = sub.add_parser(
        "list",
        help="discover sessions with metadata filters",
        epilog=_LIST_EPILOG,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    _add_filter_args(p_list)
    p_list.add_argument(
        "--preview",
        choices=["ending"],
        help="include the latest user and assistant entries, in order, each capped "
        "at 600 characters with original length and an expansion command. "
        "Reuses the entry-count scan. JSON only; unreadable previews are null",
    )
    p_list.add_argument(
        "--sort",
        choices=["recent", "oldest"],
        default="recent",
        help="result order: last activity (default), or oldest-first "
        "(for targets buried under newer sessions); ignored with "
        "--around, whose order is nearest-first",
    )
    p_list.add_argument(
        "--format",
        choices=["json", "text", "jsonl"],
        default="json",
        help="output format (default: json); jsonl puts one session per "
        "line, so | head -N keeps N whole records",
    )
    p_list.set_defaults(handler=cmd_list)

    p_get = sub.add_parser(
        "get",
        help="retrieve complete session transcripts",
        epilog=_GET_EPILOG,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p_get.add_argument(
        "session_id",
        nargs="+",
        help="one or more sessions: canonical provider:id, a "
        "raw id, or a unique prefix of either form (like "
        "git short hashes); several ids batch into one "
        'call (JSON wraps them as {"sessions": [...]})',
    )
    p_get.add_argument(
        "--format",
        choices=["text", "json"],
        default="text",
        help="output format (default: text)",
    )
    p_get.add_argument(
        "--output",
        type=_output_path,
        help="write to this file instead of stdout (single session only); "
        "'-' means stdout, the same as omitting the flag",
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
    window.add_argument(
        "--recent",
        nargs="?",
        const=8,
        type=int,
        metavar="N",
        help="last N user/assistant entries in conversation order (default 8), "
        "with a shared text budget (default 6000 chars). Includes progress "
        "messages; does not infer final replies. Cannot combine with --role or --clip",
    )
    p_get.add_argument(
        "--budget",
        type=int,
        metavar="N",
        help="with --recent: maximum total entry-text characters per session "
        "(default 6000; 0 disables). Headers, JSON syntax and omission receipts "
        "are additional. Short entries get space first; longer entries share the remainder",
    )
    window.add_argument(
        "--brief",
        nargs="?",
        const=_DEFAULT_BRIEF,
        metavar="N:M",
        help="what was asked for and how it ended, in one call: the "
        "first N user turns and the last M assistant turns "
        f"(default {_DEFAULT_BRIEF}). A slash command with no arguments "
        "(/clear, /effort) and a command's printed output are skipped and "
        "not counted; one with arguments (/goal ...) is kept, as the request. "
        "Both windows are stated in the output. Cannot be combined with "
        "--role, which it sets itself",
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
        epilog=_SEARCH_EPILOG,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p_search.add_argument(
        "query",
        nargs="+",
        help="literal phrase(s); multiple phrases are OR'd "
        "in a single scan (see --match-all)",
    )
    _add_filter_args(p_search)
    p_search.add_argument(
        "--mode",
        choices=["ids", "snippets", "full"],
        default="snippets",
        help="result detail (default: snippets)",
    )
    p_search.add_argument(
        "--match-all",
        action="store_true",
        help="with multiple phrases, keep only sessions that contain every phrase",
    )
    p_search.add_argument(
        "--role",
        action="append",
        metavar="ROLE[,ROLE]",
        help="count only matches in entries with these roles (repeatable or "
        "comma-separated): user, assistant, tool, system, or error. "
        "--role user,assistant keeps what the people and agents in a session "
        "said and drops text a tool merely printed back. Sessions with no such "
        "match are dropped before --limit",
    )
    p_search.add_argument(
        "--sort",
        choices=["recent", "matches", "oldest"],
        default="recent",
        help="result order: last activity, newest first (default); "
        "match count; or oldest, by that same last-activity time, not by "
        "when a session started (for targets buried under newer matches)",
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
        "unlimited); refused with --mode ids, which has none. Omitted "
        "snippets are counted in snippets_omitted, including a match already "
        "visible in a snippet from its entry and a passage repeated within one",
    )
    p_search.add_argument(
        "--format",
        choices=["json", "text", "jsonl"],
        default="json",
        help="output format (default: json); jsonl puts one result per "
        "line, so | head -N keeps N whole records",
    )
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
        epilog=_STATS_EPILOG,
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
        help="how many top projects and working directories to show (default 5)",
    )
    p_stats.add_argument(
        "--format",
        choices=["text", "json"],
        default="text",
        help="output format (default: text)",
    )
    p_stats.set_defaults(handler=cmd_stats)

    p_calls = sub.add_parser(
        "calls",
        help="every tool call whose input contains a pattern, with what came "
        "back: how a command was really used",
        epilog=_CALLS_EPILOG,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p_calls.add_argument(
        "pattern",
        help="literal text to find in a call's input, case-insensitive (a "
        "command name like cue, or a fuller 'cue list')",
    )
    _add_filter_args(p_calls)
    p_calls.add_argument(
        "--tool",
        action=_Once,
        help="only calls of this tool, case-insensitive (Bash, exec_command, "
        "shell, Read, ...)",
    )
    p_calls.add_argument(
        "--errors",
        action="store_true",
        help="only calls whose provider marked them failed",
    )
    p_calls.add_argument(
        "--count-by",
        action="store_true",
        help="count calls by the word after PATTERN in their input instead "
        "of listing them -- for cue, the subcommand table",
    )
    p_calls.add_argument(
        "--clip",
        type=int,
        default=300,
        metavar="N",
        help="cap input and result at N characters each (default 300; 0 disables)",
    )
    p_calls.add_argument(
        "--sort",
        choices=["recent", "oldest"],
        default="recent",
        help="session order: last activity (default), or oldest first",
    )
    p_calls.add_argument(
        "--format",
        choices=["json", "text", "jsonl"],
        default="json",
        help="output format (default: json); text is two lines per call, so "
        "a few hundred fit in one read",
    )
    p_calls.set_defaults(handler=cmd_calls)

    return parser


class _Once(argparse.Action):
    """Store a single-valued option, refusing a second occurrence of it.

    argparse keeps the last of a repeated option and says nothing, so
    ``--until 2026-10-01 ... --until -1d`` -- the shape an edited command line
    takes -- answered for a window nobody asked for, and nothing in the output
    showed which bound had won. A filter given twice is a mistake to report,
    not a choice to make on the caller's behalf. Repeatable flags
    (--exclude-cwd, --role) keep ``append``.
    """

    def __call__(self, parser, namespace, values, option_string=None):
        seen = namespace.__dict__.setdefault("_once_seen", set())
        if self.dest in seen:
            raise argparse.ArgumentError(
                self, "given more than once; it takes a single value"
            )
        seen.add(self.dest)
        setattr(namespace, self.dest, values)


def _add_filter_args(p: argparse.ArgumentParser) -> None:
    # Validated rather than filtered with. An unknown value used to filter
    # every session out and exit 0, so `--provider claude-code` -- the
    # product's actual name, and a natural guess -- reported that there were
    # no Claude Code sessions at all. Rejecting it names the three valid
    # values instead, which is what the CLI's other enum flags already do.
    # ``type`` runs before the choices check, so CLAUDE still works.
    p.add_argument(
        "--provider",
        action=_Once,
        type=str.lower,
        choices=sorted(ALL_SCANNERS),
        help="provider to scan (case-insensitive)",
    )
    p.add_argument(
        "--repo",
        action=_Once,
        help="case-insensitive substring of the inferred project name, not "
        "a git remote URL. OpenCode records a project root. Codex uses the "
        "final segment of its recorded git origin when present, then falls "
        "back to the session directory name. Claude and Pi expose neither "
        "a project root nor origin, so theirs is the directory's name -- "
        "except under a worktrees directory in a known layout "
        "(.../worktrees/<project>/<checkout>, <repo>/.claude/worktrees/"
        "<checkout>, ~/.codex/worktrees/<id>/<project>), which is named for "
        "its project. A worktree elsewhere, and a Codex session without "
        "origin metadata, keeps the checkout directory's name; read cwd off "
        "hits when that uncertainty matters",
    )
    p.add_argument(
        "--cwd",
        action=_Once,
        help="case-insensitive substring of working directory",
    )
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
        action=_Once,
        help="lower bound on last activity (newest message, not "
        "file mtime): ISO 8601, today, yesterday, or relative like "
        "-30m/-2h/-1d/-1w; a date-only value means from the "
        "start of that day (today and yesterday: local midnight)",
    )
    p.add_argument(
        "--until",
        action=_Once,
        help="upper bound on last activity (newest message, not "
        "file mtime): ISO 8601, today, yesterday, or relative like "
        "-30m/-2h/-1d/-1w; a date-only value includes that "
        "whole day",
    )
    p.add_argument(
        "--around",
        action=_Once,
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
        action=_Once,
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
        "--no-ignore",
        action="store_true",
        help="include sessions the user's ignore file hides "
        "(~/.config/session-browser/ignore: gitignore patterns matched "
        "against the working directory, for noise such as loop runs). "
        "Silent otherwise, as ripgrep is; a filter that leaves nothing "
        "while ignored sessions would have matched says so",
    )
    p.add_argument(
        "--model",
        action=_Once,
        metavar="SUBSTR",
        help="case-insensitive substring of the session's model, as that "
        "tool's own --model flag spells it (gpt-6.1-sol, "
        "opencode-go/deepseek-v4.1-flash). Recorded for Codex and Pi; Claude "
        "and OpenCode sessions carry model null and never match",
    )
    p.add_argument(
        "--no-subagents",
        action="store_true",
        help="drop sessions known to be subagents (subagent_kind set), to "
        "count or search only the sessions a person started. Kept: sessions "
        "from a provider that exposes no link (subagent_kind null), since "
        "they are not known to be subagents. Applied before --limit",
    )
    p.add_argument(
        "--limit",
        action=_Once,
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


# "today" and "yesterday" are the commonest date anyone asks about, and an
# agent asked to look at today's sessions otherwise has to work out the date
# and spell it. Local days, because that is the day the person means.
_NAMED_DAYS = {"today": 0, "yesterday": 1}


def _named_day(value: str) -> date | None:
    back = _NAMED_DAYS.get(value.strip().lower())
    today = datetime.now(UTC).astimezone().date()  # the local date
    return None if back is None else today - timedelta(days=back)


def _local_midnight(day: date) -> datetime:
    """The instant *day* begins here, with that day's own UTC offset: a
    naive midnight given to astimezone() is read as local time, DST and all,
    where subtracting a day from an aware time keeps the wrong offset."""
    return datetime.combine(day, time()).astimezone()


def _parse_date(value: str, flag: str) -> datetime:
    delta = _relative_delta(value)
    if delta is not None:
        return datetime.now(UTC) - delta
    day = _named_day(value)
    if day is not None:
        return _local_midnight(day)
    try:
        dt = datetime.fromisoformat(value)
    except ValueError:
        raise CliError(
            f"invalid {flag} date: {value!r} (expected ISO 8601, today, "
            f"yesterday, or relative like -30m/-2h/-1d)",
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
    "pi": "PI_SESSION_ID",
    # OpenCode 2.3 exports the session_v2 id (ses_...) to the commands it
    # runs, which is the id discovery assigns. Checked 2026-11-14.
    "opencode": "OPENCODE_SESSION_ID",
}


def _current_session_ids() -> set[str]:
    """Canonical ids (provider:id) of the caller's own live session(s).

    Env vars accumulate down the spawn chain — a Claude session that launches
    Codex carries both CLAUDE_CODE_SESSION_ID and CODEX_THREAD_ID — so this set
    naturally covers the whole live parent/child chain, not just the innermost
    caller. The values match the ids `discovery` assigns (Claude file stem,
    Codex session_meta id, Pi session-header id, OpenCode session id), so
    canonical-id equality is exact."""
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
    ignored = False
    if not getattr(args, "no_ignore", False):
        try:
            rules = load_ignore()
        except ConfigError as exc:
            raise CliError(str(exc), code="invalid_config") from exc
        if rules is not None:
            before = len(out)
            out = [s for s in out if not rules.ignores(s.cwd)]
            ignored = len(out) != before
    if args.provider:
        p = args.provider.lower()
        out = [s for s in out if s.provider.lower() == p]
    if getattr(args, "model", None):
        q = args.model.lower()
        unknown = sum(1 for s in out if s.model is None)
        out = [s for s in out if q in (s.model or "").lower()]
        # Same reasoning as --repo's blank count: "no session ran on that
        # model" and "no session says which model it ran on" must not read
        # alike, and for Claude and OpenCode it is always the second.
        if not out and unknown and warnings is not None:
            warnings.append(
                f"--model matched nothing, and {unknown} session(s) record no "
                f"model where discovery reads (Claude and OpenCode never do); "
                f"a content search for the model name can still reach them"
            )
    if getattr(args, "no_subagents", False):
        # Opt-in only. Subagents are real work -- a median 49% of a child's
        # vocabulary is absent from its parent -- so nothing hides them by
        # default; this is for counting what people started. Early, so a
        # search never parses a session it was told to leave out.
        out = [s for s in out if not s.subagent_kind]
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
            named = _named_day(args.until)
            if named is not None:
                # The whole of that day, to the next local midnight.
                until_excl = _local_midnight(named + timedelta(days=1))
            elif "T" not in args.until and _relative_delta(args.until) is None:
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
            # A live Claude session's subagents are live too, and a subagent
            # running session-browser inherits its root's id, so it would
            # otherwise find its own brief.
            def live(s: Session) -> bool:
                if canonical_id(s) in current:
                    return True
                root = claude_root_session(s)
                return root is not None and f"claude:{root}" in current

            kept, dropped = [], []
            for s in out:
                (dropped if live(s) else kept).append(s)
            if dropped:
                out = kept
                if warnings is not None:
                    roots = sorted(
                        canonical_id(s) for s in dropped if canonical_id(s) in current
                    )
                    children = len(dropped) - len(roots)
                    ids = ", ".join(roots)
                    if children:
                        ids += " and " if ids else ""
                        ids += f"{children} subagent(s) of it"
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
    if not out and ignored and warnings is not None:
        # ripgrep's "No files were searched, which means ripgrep probably
        # applied a filter you didn't expect", made exact. Only an empty
        # result speaks: a count on every call invites an agent to go and
        # check the ignored sessions each time, and a user who wrote the
        # file already knows. Re-running the filters without the ignore file
        # costs nothing on the path that matters, which is the non-empty one.
        hidden = apply_filters(
            sessions,
            argparse.Namespace(**{**vars(args), "no_ignore": True}),
            apply_limit=False,
        )
        if hidden:
            warnings.append(
                f"no sessions matched, but {len(hidden)} that would have are "
                f"hidden by the ignore file; pass --no-ignore to include them"
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


def _expand_entry(s: Session, index: int) -> str:
    return shlex.join(
        [
            "session-browser",
            "get",
            canonical_id(s),
            "--entries",
            str(index),
            "--clip",
            "0",
        ]
    )


def _ending_preview(s: Session) -> tuple[int | None, list[dict] | None, list[str]]:
    """Count and retain at most two bounded excerpts in one streaming read."""
    warnings: list[str] = []
    latest: dict[str, dict] = {}
    count = 0
    try:
        for i, entry in enumerate(iter_entries(s, warnings)):
            count = i + 1
            if entry.role in {"user", "assistant"}:
                latest[entry.role] = {
                    "entry_index": i,
                    "role": entry.role,
                    "text": entry.text[:600],
                    "original_chars": len(entry.text),
                }
    except (TranscriptUnreadable, OSError):
        return None, None, warnings
    preview = sorted(latest.values(), key=lambda e: e["entry_index"])
    for item in preview:
        item["expand"] = _expand_entry(s, item["entry_index"])
    return count, preview, warnings


def _budget_transcript(t: Transcript, indices: list[int], budget: int):
    """Share a source-text budget without letting a long reply erase user context.

    Allocate shortest entries first. Long entries share the remaining budget;
    ties favour newer entries. Receipts stay outside the source text.
    """
    entries = list(t.entries)
    remaining = budget
    order = sorted(range(len(entries)), key=lambda i: (len(entries[i].text), i))
    omitted = []
    for position, i in enumerate(order):
        entry = entries[i]
        keep = (
            min(len(entry.text), remaining // (len(order) - position))
            if budget
            else len(entry.text)
        )
        remaining -= keep
        if keep < len(entry.text):
            entries[i] = replace(entry, text=entry.text[:keep])
            omitted.append(
                {
                    "entry_index": indices[i],
                    "omitted_chars": len(entry.text) - keep,
                    "expand": _expand_entry(t.session, indices[i]),
                }
            )
    receipt = {
        "limit": budget,
        "returned_chars": sum(len(e.text) for e in entries),
        "omitted_chars": sum(e["omitted_chars"] for e in omitted),
        "omitted": sorted(omitted, key=lambda e: e["entry_index"]),
    }
    return Transcript(t.session, entries, t.warnings), receipt


def _render_budget(receipt: dict) -> str:
    lines = [
        (
            f"Text budget: {receipt['returned_chars']} chars returned; "
            f"{receipt['omitted_chars']} chars omitted from the selected entries. "
            "Metadata and receipts are additional."
        )
    ]
    for item in receipt["omitted"]:
        lines.append(
            f"- [{item['entry_index']}] clipped {item['omitted_chars']} chars; "
            f"expand: `{item['expand']}`"
        )
    return "\n".join(lines)


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
    if args.preview and args.format == "text":
        raise CliError(
            "--preview requires --format json or jsonl", code="invalid_filter"
        )
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
    previews = {}
    if sessions:
        with ThreadPoolExecutor(max_workers=8) as ex:
            if args.preview:
                results = list(ex.map(_ending_preview, sessions))
                for s, (n, preview, parse_warnings) in zip(
                    sessions, results, strict=True
                ):
                    counts.append(n)
                    previews[canonical_id(s)] = preview
                    warnings.extend(f"{canonical_id(s)}: {w}" for w in parse_warnings)
            else:
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
    if args.format != "text":
        items = []
        for s, n in zip(sessions, counts, strict=True):
            item = session_to_dict(s)
            item["total_entries"] = n
            if args.preview:
                item["preview"] = previews[canonical_id(s)]
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
        if args.format == "jsonl":
            _print_jsonl(payload, "sessions")
        else:
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
        if not sessions:
            # Text has no envelope to carry an empty list, and no output at
            # all read as success to agents who then moved on. stderr, so
            # stdout stays one record per line.
            print("no sessions matched", file=sys.stderr)
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


def _parse_brief(spec) -> tuple[int, int] | None:
    """Resolve --brief into (intent turns, ending turns).

    "N:M" mirrors --entries A:B so the CLI has one punctuation for one idea,
    but the numbers are counts rather than positions: the first N user turns
    and the last M assistant turns. Both may be 0, which is how you ask for
    one half without spelling out the two-call form it replaces.
    """
    if spec is None:
        return None
    text = str(spec).strip()
    head, sep, tail = text.partition(":")
    if not sep or not head.strip().isdigit() or not tail.strip().isdigit():
        raise CliError(
            f"invalid --brief {text!r} (expected N:M, e.g. {_DEFAULT_BRIEF})",
            code="invalid_filter",
        )
    intent, ending = int(head), int(tail)
    if intent == 0 and ending == 0:
        raise CliError(
            "--brief 0:0 asks for nothing; give a count on at least one side",
            code="invalid_filter",
        )
    return intent, ending


def _brief_view(transcript: Transcript, brief: tuple[int, int]):
    """The first N user turns and the last M assistant turns, in order.

    One pass over the entries the read already produced -- the whole point of
    the flag is that this costs one transcript parse where the two-call form
    it replaces costs two. The two halves cannot overlap (an entry has one
    role), so no de-duplication is needed, and both are returned in transcript
    order rather than intent-then-ending: a reader who sees absolute indices
    ascending can trust that ordering means what it always means.

    A bare slash command and a command's printed output are not turns of
    intent, and do not count against N. A session opened by /clear, /effort
    and its stdout otherwise spent half the default window on them and pushed
    the actual request out of it. A command that carries arguments stays: it
    is the request.
    """
    intent_n, ending_n = brief
    intent: list[tuple[int, TranscriptEntry]] = []
    ending: list[tuple[int, TranscriptEntry]] = []
    for i, e in enumerate(transcript.entries):
        if e.role == "user":
            if len(intent) < intent_n and not is_command_noise(e):
                intent.append((i, e))
        elif e.role == "assistant":
            ending.append((i, e))
    # ending[-ending_n:], written out: a computed start of len - ending_n
    # goes NEGATIVE when the session has fewer assistant turns than asked
    # for, and the slice then wraps to the last few instead of returning all
    # of them. A 7-turn session asked for 8 came back with 1.
    if ending_n:
        kept = intent + (ending[-ending_n:] if ending_n <= len(ending) else ending)
    else:
        kept = intent
    kept.sort(key=lambda pair: pair[0])
    return kept


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
    (transcript, total, window, indices, briefed); raises when unreadable.
    *briefed* is what _briefed_by found in the whole transcript's opening.

    *indices* are the absolute positions of the entries actually returned,
    always — including the unfiltered case, where they are contiguous but
    still offset by the window's start. They are what "entry_index" reports,
    and a caller that wants to know whether the kept set is *sparse* should
    ask *roles*, not whether indices are present."""
    transcript = load_transcript(session)
    total = len(transcript.entries)
    # Read before any window cuts the opening away.
    briefed = _briefed_by(session, transcript.entries)
    window = None
    brief = _parse_brief(getattr(args, "brief", None))
    if brief is not None:
        kept = _brief_view(transcript, brief)
        indices = [i for i, _ in kept]
        transcript = Transcript(
            transcript.session, [e for _, e in kept], transcript.warnings
        )
        return transcript, total, None, indices, briefed
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
        else:
            indices = list(range(start, start + len(transcript.entries)))
    return transcript, total, window, indices, briefed


def _fork_receipt(
    session: Session, discovered: list[Session], warnings: list[str]
) -> dict | None:
    """Where a Codex fork's earlier history lives, or None if not a fork.

    A fork's rollout holds only what happened after it was forked; the turns
    before are the first part of the session it was forked from. Reported
    rather than spliced in, so entry indices stay the ones search reports
    and --entries addresses: {"id": the source, "entries": how many of the
    source's entries are this fork's history (null when the source is not on
    this machine or its rollout no longer reaches the fork point)}.
    """
    base = codex_fork_base(session)
    if base is None:
        return None
    source_id, byte_end = base
    receipt: dict = {"id": f"codex:{source_id}", "entries": None}
    source = next(
        (s for s in discovered if s.provider == "codex" and s.id == source_id), None
    )
    if (
        source is None
        and (owner := codex_thread_of_segment(source_id, discovered)) is not None
    ):
        # The fork's history ends in a file a rewind made. Which of that
        # thread's entries it inherits is not counted: no fork of a rewound
        # thread has been seen to check the shape against.
        receipt["id"] = f"codex:{owner.id}"
        warnings.append(
            f"forked from codex:{owner.id} after it was rewound; how much of "
            "its history the fork inherits is not counted"
        )
        return receipt
    if source is None:
        warnings.append(
            f"forked from codex:{source_id}, which is not on this machine; "
            "the history before the fork is unavailable"
        )
        return receipt
    prefix = codex_prefix_entries(source, byte_end, warnings)
    if prefix is not None:
        receipt["entries"] = len(prefix)
    return receipt


# A Claude Code session's scratchpad is /private/tmp/claude-<uid>/<project>/
# <session-id>/scratchpad/, so a brief written there names the session that
# wrote it. Another agent handed work through Herdr typically opens "Read the
# brief at <that path> and follow it": the one trace a cross-tool hand-off
# leaves in the session that received it. Measured 2026-11-01: 20 sessions
# here open that way with nothing else linking them to their briefer (Codex,
# Pi and Claude panes), 10 of whose briefers are still on disk.
_SCRATCHPAD_BRIEF = re.compile(
    r"(?P<path>\S*/claude-\d+/[^/\s]+/"
    r"(?P<id>[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12})"
    r"/scratchpad/\S*)"
)


def _briefed_by(session: Session, entries) -> tuple[str, str] | None:
    """(briefer's session id, brief path) when this session's opening request
    names a file in another Claude session's scratchpad, else None.

    Only the opening request is read -- a bare /clear or a command's output
    before it is passed over -- because that is where a hand-off's brief is
    named. A path met later is a file the session used, not who briefed it.
    """
    for e in entries:
        if e.role != "user" or is_command_noise(e):
            continue
        m = _SCRATCHPAD_BRIEF.search(e.text)
        if m is None or m["id"] == session.id:
            return None
        return m["id"], m["path"].rstrip(".,;:)\"'`")
    return None


def _brief_receipt(found: tuple[str, str] | None) -> dict | None:
    if found is None:
        return None
    briefer, path = found
    return {
        "id": f"claude:{briefer}",
        "path": path,
        "present": _claude_transcript_exists(briefer),
    }


def _claude_transcript_exists(session_id: str) -> bool:
    """Whether ``~/.claude/projects/*/<session_id>.jsonl`` exists.

    Asked of the disk rather than of discovery, which `get` scopes to the
    providers its ids name: a Codex session briefed by Claude would otherwise
    report its briefer missing, and discovering every Claude session to ask
    would put that whole scan on every Codex `get`. One stat per project."""
    root = Path.home() / ".claude" / "projects"
    try:
        return any((d / f"{session_id}.jsonl").is_file() for d in root.iterdir())
    except OSError:
        return False


def _render_brief(receipt: dict) -> str:
    where = "" if receipt["present"] else ", which is not on this machine"
    return f"> Briefed by {receipt['id']}{where}, through `{receipt['path']}`."


def _render_fork(fork: dict) -> str:
    n = fork["entries"]
    if n is None:
        return (
            f"> Forked from {fork['id']}; the history before the fork is unavailable."
        )
    return (
        f"> Forked from {fork['id']}: its first {n} entries are this session's "
        f"earlier history (`get {fork['id']} --entries 0:{n - 1}`)."
        if n
        else f"> Forked from {fork['id']} at its start."
    )


def cmd_get(args) -> int:
    roles = _parse_roles(args.role)
    brief = _parse_brief(args.brief)
    if args.recent is not None:
        if args.recent < 1:
            raise CliError("--recent must be positive", code="invalid_filter")
        if roles is not None or args.clip is not None:
            raise CliError(
                "--recent chooses its roles and uses --budget instead of --clip",
                code="invalid_filter",
            )
        roles = ["user", "assistant"]
        args.tail = args.recent
        args.clip = 0
        if args.budget is None:
            args.budget = 6000
        if args.budget < 0:
            raise CliError("--budget must be non-negative", code="invalid_filter")
    elif args.budget is not None:
        raise CliError("--budget requires --recent", code="invalid_filter")
    if brief is not None and roles is not None:
        raise CliError(
            "--brief chooses its own roles (user for the intent, assistant "
            "for the ending); drop --role, or drop --brief and ask for the "
            "two windows separately",
            code="invalid_filter",
        )
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
    budgets = {}
    briefs: dict[str, dict] = {}
    skipped: list[tuple[Session, str]] = []
    for s in sessions:
        try:
            t, total, window, indices, briefed = _session_view(s, args, roles)
        except (TranscriptUnreadable, OSError) as exc:
            if len(sessions) == 1:
                raise CliError(
                    f"could not read session {canonical_id(s)}: {exc}",
                    code="unreadable_session",
                ) from exc
            skipped.append((s, str(exc)))
            continue
        if args.recent is not None:
            t, receipt = _budget_transcript(t, indices, args.budget)
            budgets[canonical_id(s)] = receipt
        views.append((s, _clip_transcript(t, clip), total, window, indices))
        if (receipt := _brief_receipt(briefed)) is not None:
            briefs[canonical_id(s)] = receipt
    if not views:
        raise CliError(
            "no readable sessions among: "
            + ", ".join(canonical_id(s) for s, _ in skipped),
            code="unreadable_session",
        )
    single = len(idents) == 1
    forks = {
        canonical_id(s): fork
        for s, *_ in views
        if (fork := _fork_receipt(s, discovered, views[0][1].warnings)) is not None
    }
    parse_warnings = [w for _, t, *_ in views for w in t.warnings]
    if args.format == "json":
        payloads = []
        for _s, t, total, window, indices in views:
            payload = transcript_to_dict(t, entry_indices=indices)
            payload["total_entries"] = total
            if canonical_id(t.session) in forks:
                payload["forked_from"] = forks[canonical_id(t.session)]
            if canonical_id(t.session) in briefs:
                payload["briefed_by"] = briefs[canonical_id(t.session)]
            if window is not None:
                payload["entry_range"] = {"start": window[0], "end": window[1]}
            if roles is not None:
                payload["roles"] = roles
            if brief is not None:
                payload["brief"] = {"intent": brief[0], "ending": brief[1]}
            if clip:
                payload["clip"] = clip
            if args.recent is not None:
                payload["recent"] = args.recent
                payload["budget"] = budgets[canonical_id(t.session)]
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
                # Label a block with its position only when the kept set is
                # sparse. JSON gains nothing by omitting a key, so it always
                # carries one; a document read by a person does — an unbroken
                # run of "[7] [8] [9]" prefixes is noise the header line
                # already covers with an "Entries: 7 to 9 of 40" range.
                entry_indices=(
                    indices if roles is not None or brief is not None else None
                ),
                roles=roles,
                brief=brief,
                timestamps=True,
            )
            for _, t, total, window, indices in views
        ]
        if args.recent is not None:
            docs = [
                _render_budget(budgets[canonical_id(s)]) + "\n\n" + doc
                for doc, (s, *_) in zip(docs, views, strict=True)
            ]
        docs = [
            _render_fork(forks[canonical_id(s)]) + "\n\n" + doc
            if canonical_id(s) in forks
            else doc
            for doc, (s, *_) in zip(docs, views, strict=True)
        ]
        docs = [
            _render_brief(briefs[canonical_id(s)]) + "\n\n" + doc
            if canonical_id(s) in briefs
            else doc
            for doc, (s, *_) in zip(docs, views, strict=True)
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
    if args.max_snippets is not None and args.mode == "ids":
        # Accepted and ignored, it read as a bound on something; it is not.
        raise CliError(
            "--max-snippets caps snippets, and --mode ids returns none: drop it, "
            "or use --mode snippets",
            code="invalid_filter",
        )
    roles = _parse_roles(getattr(args, "role", None))
    warnings: list[str] = []
    discovered = _discover(args)
    around = _resolve_anchor(discovered, args)
    anchor_ts = around[1] if around is not None else None
    candidates = apply_filters(discovered, args, apply_limit=False, warnings=warnings)
    keep = args.mode == "full"
    results = search_sessions(candidates, queries, keep_entries=keep)
    if roles is not None:
        _keep_matches_in_roles(results, set(roles))
    summary_map = {
        canonical_id(r.session): hits
        for r in results
        if not r.unreadable and (hits := _summary_matches(r.session, queries))
    }
    # A summary has no role, so under --role it is reported but cannot
    # qualify a session by itself: the flag asks who said it, and a summary
    # cannot answer that.
    qualifying_summaries = summary_map if roles is None else {}
    matched = [
        r
        for r in results
        if not r.unreadable
        and (r.match_count > 0 or canonical_id(r.session) in qualifying_summaries)
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
                | {
                    q.casefold()
                    for q in qualifying_summaries.get(canonical_id(r.session), [])
                }
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
    if args.format != "text":
        payload = _search_payload(
            args,
            matched,
            skipped,
            warnings,
            anchor_ts=anchor_ts,
            summary_map=summary_map,
        )
        if args.format == "jsonl":
            _print_jsonl(payload, "results")
        else:
            print(json.dumps(payload, indent=2))
    else:
        _print_search_text(args, matched, anchor_ts=anchor_ts, summary_map=summary_map)
        if not matched:
            print("no sessions matched", file=sys.stderr)
        _report_warnings_stderr(warnings)
        _report_skipped_stderr(skipped)
    return 0


def cmd_calls(args) -> int:
    pattern = args.pattern.strip()
    if not pattern:
        raise CliError("empty calls pattern", code="invalid_query")
    if args.clip < 0:
        raise CliError("--clip must be >= 0 (0 disables)", code="invalid_filter")
    warnings: list[str] = []
    discovered = _discover(args)
    candidates = apply_filters(discovered, args, apply_limit=False, warnings=warnings)
    if args.sort == "oldest":
        _floor = datetime.min.replace(tzinfo=UTC)
        candidates.sort(key=lambda s: _session_time(s) or _floor)
    # Search does the finding: its prefilter rules out every session whose raw
    # bytes lack the pattern, and keep_entries leaves each parsed transcript in
    # hand, so pairing a call with its output reads nothing a second time.
    results = search_sessions(candidates, [pattern], keep_entries=True)
    tool = args.tool.casefold() if args.tool else None
    calls: list[dict] = []
    words: dict[str, int] = {}
    for r in results:
        if r.unreadable or not r.matches or r.entries is None:
            continue
        hits = {
            m.entry_index: m
            for m in r.matches
            if m.entry.metadata and m.entry.metadata.get("kind") == "call"
        }
        if not hits:
            continue
        answers = _pair_tool_outputs(r.entries)
        for index, m in hits.items():
            meta = m.entry.metadata
            name = str(meta.get("tool") or "")
            if tool is not None and name.casefold() != tool:
                continue
            answer = answers.get(index)
            out = r.entries[answer] if answer is not None else None
            out_meta = (out.metadata or {}) if out is not None else {}
            # Every provider marks only failure; an output without the flag
            # is one that succeeded, and no output at all says nothing.
            is_error = bool(out_meta.get("is_error")) if out is not None else None
            if args.errors and not is_error:
                continue
            if args.count_by:
                word = _word_after(m.entry.text, m.offsets[0] + len(pattern))
                words[word] = words.get(word, 0) + 1
                continue
            calls.append(
                {
                    "id": canonical_id(r.session),
                    "entry_index": index,
                    "timestamp": m.entry.timestamp,
                    "tool": name,
                    "input": _clip_to(m.entry.text, args.clip),
                    "input_chars": len(m.entry.text),
                    "result_index": answer,
                    "result": _clip_to(out.text, args.clip) if out else None,
                    "result_chars": len(out.text) if out else None,
                    "is_error": is_error,
                    "exit_code": out_meta.get("exit_code"),
                }
            )
    skipped = [r for r in results if r.unreadable]
    counts = sorted(
        ({"word": w, "count": n} for w, n in words.items()),
        key=lambda c: (-c["count"], c["word"]),
    )
    if args.limit is not None and args.limit >= 0:
        if args.count_by:
            counts = counts[: args.limit]
        elif len(calls) > args.limit:
            warnings.append(
                f"--limit {args.limit} kept {args.limit} of {len(calls)} calls; "
                "omit it for every one"
            )
            calls = calls[: args.limit]
    if args.format == "text":
        if args.count_by:
            for c in counts:
                print(f"{c['count']}\t{c['word']}")
        else:
            for c in calls:
                print(
                    "\t".join(
                        (
                            c["timestamp"],
                            c["id"],
                            str(c["entry_index"]),
                            c["tool"],
                            " ".join(c["input"].split()),
                        )
                    )
                )
                if c["result"] is None:
                    print("  -> (no result recorded)")
                else:
                    flag = " [error]" if c["is_error"] else ""
                    print(f"  ->{flag} {' '.join(c['result'].split())}")
        if not (counts if args.count_by else calls):
            print("no calls matched", file=sys.stderr)
        _report_warnings_stderr(warnings)
        _report_skipped_stderr(skipped)
        return 0
    payload: dict = {
        "pattern": pattern,
        "tool": args.tool,
        "errors": args.errors,
        "filters": _filters_dict(args),
        "calls": calls,
        "skipped": _skipped_list(skipped),
        "warnings": warnings,
    }
    if args.count_by:
        payload["counts"] = counts
    if args.format == "jsonl":
        _print_jsonl(payload, "counts" if args.count_by else "calls")
    else:
        print(json.dumps(payload, indent=2))
    return 0


def _pair_tool_outputs(entries) -> dict[int, int]:
    """{call entry index: the output entry index that answered it}.

    By id where the provider records one on both sides (Claude, which writes
    no entry for an empty result, so order would pair every later call with
    the wrong output). Otherwise by order: a call's output follows it, and
    parallel calls are answered in the order they were made, so an output
    answers the oldest unanswered call before it -- of its own tool, when
    the output names one. Codex writes an output for every call, even a
    silent command's carries its exit header, so order is sound there."""
    pending: list[int] = []
    by_id: dict[str, int] = {}
    answered: dict[int, int] = {}
    for i, e in enumerate(entries):
        meta = e.metadata
        if e.role != "tool" or not meta:
            continue
        kind = meta.get("kind")
        if kind == "call":
            call_id = meta.get("id")
            if call_id:
                by_id[call_id] = i
            else:
                pending.append(i)
        elif kind == "output":
            call_id = meta.get("call_id")
            if call_id:
                call = by_id.pop(call_id, None)
                if call is not None:
                    answered[call] = i
                continue
            if not pending:
                continue
            name = meta.get("tool")
            pick = 0
            if name:
                pick = next(
                    (
                        n
                        for n, c in enumerate(pending)
                        if entries[c].metadata.get("tool") == name
                    ),
                    0,
                )
            answered[pending.pop(pick)] = i
    return answered


def _word_after(text: str, start: int) -> str:
    """The first word after *start*, stripped of quoting; '' at the end."""
    rest = text[start:].lstrip(" \t\"'")
    word = rest.split(None, 1)[0] if rest.split(None, 1) else ""
    return word.strip("\"',;)(}{[]\\") or "(none)"


def _clip_to(text: str, limit: int) -> str:
    if limit == 0 or len(text) <= limit:
        return text
    return text[:limit] + f"… [clipped {len(text) - limit} chars]"


def _keep_matches_in_roles(results, roles: set[str]) -> None:
    """Narrow every result's matches to entries of *roles*, in place.

    Everything downstream reads ``matches`` and ``match_count``, so once this
    has run, selection, --sort matches, first/last_match, match_roles and the
    snippets all count only those entries without each having to be told.
    Tool output dominates raw hits -- agents re-reading a README, a tracker
    listing -- so without this the most-QUOTED session ranked first and
    --sort oldest found the earliest quote: in one run, a pyproject.toml read
    answered "the earliest session mentioning" a project.

    Free: search parses every candidate and keeps every match whatever the
    mode, so the roles are already known and nothing is read again.
    """
    for r in results:
        if r.matches:
            r.matches = [m for m in r.matches if entry_matches_roles(m.entry, roles)]
            r.match_count = sum(len(m.offsets) for m in r.matches)


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
        **_roles_echo(args),
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
    if args.format != "text":
        print(json.dumps(summary))
    else:
        print(f"wrote {len(planned)} file(s) to {out_dir} (manifest: {manifest_path})")
    return 0


def _roles_echo(args) -> dict:
    """{"roles": [...]} when search was narrowed with --role, else {}.

    Its own envelope key, as on `get`, rather than a member of "filters":
    that block is shared with stats, which has no roles to report."""
    roles = _parse_roles(getattr(args, "role", None))
    return {"roles": roles} if roles is not None else {}


def _filters_dict(args) -> dict:
    return {
        "provider": args.provider,
        "repo": args.repo,
        "cwd": args.cwd,
        "exclude_cwd": getattr(args, "exclude_cwd", None),
        "here": getattr(args, "here", False),
        "include_current": getattr(args, "include_current", False),
        "no_ignore": getattr(args, "no_ignore", False),
        "no_subagents": getattr(args, "no_subagents", False),
        "model": getattr(args, "model", None),
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


def _shown_snippets(args, r) -> list[tuple]:
    """(match, snippet text) for each snippet a result shows, in match order.

    At most --max-snippets of them, and never the same passage twice from
    one entry. A tool output that prints a file twice -- AGENTS.md read once
    by name and once through its CLAUDE.md symlink -- matches twice, and the
    two windows around those matches are character-for-character identical:
    the second says nothing the first did not and spends a slot of the cap.
    A match lying inside a window already shown from its entry is skipped
    too: it is on screen already. Both kinds still count in match_count, so
    they land in snippets_omitted. Built per returned result, outside any
    scan loop.
    """
    cap = _snippet_cap(args)
    shown: list[tuple] = []
    seen: set[tuple[int, str]] = set()
    # Windows already shown, per entry. A match lying wholly inside one is
    # already on screen: several phrases OR'd together, or one repeated
    # within a sentence, otherwise each re-showed the same passage, which two
    # agents in a dogfood run reported as near-duplicate snippets.
    windows: dict[int, list[tuple[int, int]]] = {}
    context = args.context
    for m in r.matches:
        width = len(m.query)
        for off in m.offsets:
            if cap and len(shown) >= cap:
                return shown
            spans = windows.setdefault(m.entry_index, [])
            if any(a <= off and off + width <= b for a, b in spans):
                continue
            text = make_snippet(m.entry.text, off, width, context)
            if (m.entry_index, text) in seen:
                continue
            seen.add((m.entry_index, text))
            spans.append((off - context, off + width + context))
            shown.append((m, text))
    return shown


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
        # Who said it, counted from every match rather than the capped
        # snippets, and where the people in the session -- not the tools --
        # first and last said it. Built per returned result from matches
        # search already holds: no extra read, and outside any scan loop.
        roles: dict[str, int] = {}
        authored = []
        for m in r.matches:
            role = m.entry.role
            roles[role] = roles.get(role, 0) + len(m.offsets)
            if role == "user" or role == "assistant":
                authored.append(m.entry_index)
        item["match_roles"] = dict(sorted(roles.items()))
        item["first_authored_match"] = authored[0] if authored else None
        item["last_authored_match"] = authored[-1] if authored else None
    if r.warnings:
        item["parse_warnings"] = r.warnings
    multi = len(args.query) > 1
    if args.mode in ("snippets", "full"):
        snippets = [
            {
                "role": m.entry.role,
                "entry_index": m.entry_index,
                **({"query": m.query} if multi else {}),
                "text": text,
            }
            for m, text in _shown_snippets(args, r)
        ]
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
        **_roles_echo(args),
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
            shown = _shown_snippets(args, r)
            for m, snip in shown:
                # get's own role-filtered shape, so the index can go straight
                # back to get --entries without a JSON detour.
                print(f"  [{m.entry_index}] {m.entry.role}: {snip}")
            if r.match_count > len(shown):
                print(
                    f"  … {r.match_count - len(shown)} more snippet(s) "
                    f"omitted (--max-snippets, or a passage already shown)"
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


def _print_jsonl(payload: dict, records_key: str) -> None:
    """The envelope on one line, then one record per line.

    Pretty-printed JSON is the wrong shape for the line tools agents reach
    for: measured over real usage, 18 of 28 replayed `--format json | head -N`
    calls produced more lines than the cap, so the agent got JSON cut
    mid-object and nothing said so. Here a line is a record, so `| head -N`
    keeps N - 1 whole ones. The envelope comes first, so a cut keeps the
    counts and warnings, and its "records" says how many lines follow it --
    which turns a truncation into something the reader can see.
    """
    records = payload.pop(records_key)
    payload["records"] = len(records)
    print("\n".join([json.dumps(payload), *(json.dumps(r) for r in records)]))


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
    # Subagents are 9.3% of a real corpus and share their parent's cwd
    # exactly, so no other filter separates them and nothing here would
    # otherwise say they exist. Counted rather than hidden: sampling
    # parent/child pairs, a median 49% of a child's vocabulary is absent
    # from its parent, so they are real work and not duplicates of it.
    subagent_kinds: dict[str, int] = {}
    linked = unlinked = unknown = 0
    # Per inferred project name, which is what folds a project's worktree
    # directories together: one project appeared as 9 cwds in top_cwds. The
    # question it answers -- which projects had work, with which providers --
    # used to take a 60 KB `list` and a hand-written group-by.
    projects: dict[str, list] = {}  # name -> [count, subagents, {provider: n}]
    for s in sessions:
        provider_counts[s.provider] = provider_counts.get(s.provider, 0) + 1
        name = (s.repository or "").strip()
        if name:
            proj = projects.get(name)
            if proj is None:
                proj = projects[name] = [0, 0, {}, None]
            proj[0] += 1
            if s.subagent_kind:
                proj[1] += 1
            proj[2][s.provider] = proj[2].get(s.provider, 0) + 1
        if s.subagent_kind is None:
            unknown += 1
        elif s.subagent_kind:
            subagent_kinds[s.subagent_kind] = subagent_kinds.get(s.subagent_kind, 0) + 1
            if s.parent_id:
                linked += 1
            else:
                unlinked += 1
        cwd = (s.cwd or "").strip()
        if cwd:
            cwd_counts[cwd] = cwd_counts.get(cwd, 0) + 1
        ts = _session_time(s)
        if ts is None:
            continue
        times.append(ts)
        if name and (proj[3] is None or ts > proj[3]):
            proj[3] = ts
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
    payload["subagents"] = {
        "total": linked + unlinked,
        # Of those, how many name the session that spawned them. The rest
        # are subagents whose provider recorded no parent -- Codex "review"
        # and "guardian" threads carry none anywhere, so "is a subagent" and
        # "has a known parent" have to stay separate numbers.
        "linked": linked,
        "unlinked": unlinked,
        # Sessions from a provider that exposes no link at all, so they are
        # neither known to be subagents nor known not to be.
        "unknown": unknown,
        "kinds": [
            {"kind": k, "count": n}
            for k, n in sorted(subagent_kinds.items(), key=lambda kv: (-kv[1], kv[0]))
        ],
    }
    ranked = sorted(projects.items(), key=lambda kv: (-kv[1][0], kv[0]))
    payload["projects"] = [
        {
            "repository": name,
            "count": count,
            "subagents": subs,
            "providers": dict(sorted(by.items(), key=lambda kv: (-kv[1], kv[0]))),
            "updated_at": last.isoformat() if last else None,
        }
        for name, (count, subs, by, last) in ranked[: args.top]
    ]
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

    sub = payload["subagents"]
    if sub["total"]:
        print()
        kinds = ", ".join(f"{k['count']} {k['kind']}" for k in sub["kinds"][:4])
        print(f"subagents · {sub['total']} of {total}")
        line = f"  {sub['linked']} linked to a parent"
        if sub["unlinked"]:
            line += f" · {sub['unlinked']} with none recorded"
        print(line)
        if kinds:
            print(f"  {kinds}")
        if sub["unknown"]:
            print(f"  {sub['unknown']} from providers that expose no link")

    act = payload["activity"]
    day_peak = max(act["counts"], default=0)
    print()
    label = f"activity · last {act['days']} days"
    if day_peak:
        label += f" · peak {day_peak}/day"
    print(label)
    print(f"  {_sparkline(act['counts'])}")
    print(f"  {act['start'][5:]} → {act['end'][5:]}")

    if payload["projects"]:
        print()
        print("top projects")
        name_w = max(len(p["repository"]) for p in payload["projects"])
        for p in payload["projects"]:
            by = " · ".join(f"{n} {prov}" for prov, n in p["providers"].items())
            n = p["subagents"]
            subs = f"  ({n} subagent{'' if n == 1 else 's'})" if n else ""
            print(f"  {p['count']:>5}  {p['repository']:<{name_w}}  {by}{subs}")

    if payload["top_cwds"]:
        print()
        print("top working directories")
        for d in payload["top_cwds"]:
            print(f"  {d['count']:>5}  {_shorten_home(d['cwd'])}")
