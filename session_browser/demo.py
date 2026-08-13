"""Demo corpus generator: a believable fake provider history.

The TUI demos poorly against a real session history — real transcripts are
messy, private, and sparse. This module writes a compact but rich fake HOME
tree (Claude JSONL, Codex JSONL, an OpenCode SQLite database) that discovery
reads exactly like the real thing, so every command — list, search, stats,
get, the TUI — runs against it unchanged by pointing HOME at it:

    python -m session_browser.demo                 # writes ./demo-home
    HOME=$PWD/demo-home session-browser            # TUI against the fake corpus
    HOME=$PWD/demo-home session-browser search "sqlite"

The corpus is sized for screenshots: a few sessions per provider, varied
dates over the last month, distinctive searchable terms, one long transcript
for the fold/collapse demo, and scratchpad-ish noise sessions so the list
tells a story before/after --exclude-cwd scratchpad.

Nothing here touches the retrieval path. Generation only writes files, so the
perf budgets that count discovery/parse work on the real corpus are provably
unaffected by this module existing.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sqlite3
import sys
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path

CLAUDE_PROJECTS = ".claude/projects"
CODEX_SESSIONS = ".codex/sessions"
OPENCODE_DB = ".local/share/opencode/opencode.db"

# Scratchpad noise sessions carry this cwd prefix so the --exclude-cwd demo
# has something to drop (mirrors the real /private/tmp scratchpad noise).
SCRATCHPAD = "/private/tmp/scratchpad"

DEFAULT_HOME = "demo-home"


@dataclass(frozen=True)
class _Turn:
    role: str  # user | assistant | tool_call | tool_output
    text: str


@dataclass(frozen=True)
class _Spec:
    provider: str
    id: str
    summary: str
    cwd: str
    branch: str
    repository: str
    days_ago: int
    hour: int
    turns: list[_Turn]


@dataclass(frozen=True)
class DemoResult:
    home: Path
    claude: int
    codex: int
    opencode: int

    @property
    def total(self) -> int:
        return self.claude + self.codex + self.opencode


def _dt(spec: _Spec, now: datetime, offset_minutes: int = 0) -> datetime:
    base = (now - timedelta(days=spec.days_ago)).replace(
        hour=spec.hour % 24, minute=0, second=0, microsecond=0
    )
    return base + timedelta(minutes=offset_minutes)


def _ts(spec: _Spec, now: datetime, offset_minutes: int = 0) -> str:
    return _dt(spec, now, offset_minutes).isoformat()


# ---------------------------------------------------------------------------
# The corpus
# ---------------------------------------------------------------------------
#
# _TURNS and _TURNS2 are spoken scripts; each tool_call/tool_output pair is
# written as a Bash call and its result, which is what real transcripts look
# like around every non-trivial step. Content is invented, deliberately dry,
# and laced with searchable terms so a demo query always lands somewhere.


def _big_migration_turns() -> list[_Turn]:
    """The fold/collapse session: a long SQLite migration, ~60 entries."""
    turns: list[_Turn] = [
        _Turn(
            "user",
            "Let's plan the widgets service migration to SQLite. I want a "
            "phased plan: inventory, schema, data, verification.",
        ),
        _Turn(
            "assistant",
            "Good instinct — migrations that don't start with an inventory "
            "end with a rollback. Phase 1: find every read/write path to "
            "the current JSON store so we know what the schema must cover.",
        ),
        _Turn("tool_call", "rg -l 'load_store|save_store' src/ widgets/"),
        _Turn(
            "tool_output",
            "src/store/reader.py\nsrc/store/writer.py\nwidgets/archive.py\n"
            "widgets/reporting.py",
        ),
        _Turn(
            "user",
            "And WAL mode — should we turn it on from day one?",
        ),
        _Turn(
            "assistant",
            "Yes. WAL gives us concurrent readers during the migration "
            "backfill without blocking writers. Set `PRAGMA journal_mode=WAL` "
            "at open, once, not per query.",
        ),
    ]
    for i in range(1, 13):
        turns.extend(
            [
                _Turn(
                    "user",
                    f"Table {i} of the inventory: what stores widget "
                    f"record fields {i * 10} through {i * 10 + 9}?",
                ),
                _Turn(
                    "assistant",
                    f"widget_records_{i} today, a flat JSON list. In SQLite "
                    f"that becomes a widget_events table with a composite "
                    f"index on (widget_id, ts) and foreign keys to widgets. "
                    f"Rows: on the order of 10^{i % 4 + 3}.",
                ),
                _Turn(
                    "tool_call",
                    f"python -m backfill --table widget_events --batch {i} --dry-run",
                ),
                _Turn(
                    "tool_output",
                    f"backfill dry run: table widget_events batch {i} "
                    f"-> 0 errors, row count matches source",
                ),
            ]
        )
    turns.extend(
        [
            _Turn(
                "user",
                "Verification phase: what gives us confidence the migration "
                "is faithful?",
            ),
            _Turn(
                "assistant",
                "Three things. A row-count assertion per table after "
                "backfill, a checksum over canonical dumps of both stores, "
                "and the existing integration suite pointed at the SQLite "
                "backend behind a flag. Any mismatch aborts the cutover.",
            ),
            _Turn("tool_call", "pytest tests/migration -x -q"),
            _Turn(
                "tool_output",
                "47 passed, 2 skipped in 12.41s",
            ),
            _Turn(
                "user",
                "Cutover plan, then. What flips last?",
            ),
            _Turn(
                "assistant",
                "The read path flips last, after writes have been dual-"
                "written for a day. Rollback is then one config revert, not "
                "a restore.",
            ),
            _Turn(
                "assistant",
                "Summarising the whole plan so far: inventory found 4 "
                "store call sites, schema covers 12 record tables under "
                "WAL with foreign keys, verification is row counts plus "
                "checksums plus the suite, and cutover dual-writes first. "
                "Ready to start Phase 1 for real when you are.",
            ),
        ]
    )
    return turns


def _turns(
    user: str,
    assistant: str,
    *,
    command: str | None = None,
    output: str | None = None,
) -> list[_Turn]:
    out: list[_Turn] = [_Turn("user", user)]
    if command is not None:
        out.append(_Turn("tool_call", command))
        out.append(_Turn("tool_output", output or ""))
    out.append(_Turn("assistant", assistant))
    return out


_TURNS = _big_migration_turns()
_TURNS2 = _turns(
    "The CSV export is mangling every field that contains a quote. "
    "Can we fix the quoting rule once and reuse it?",
    "The writer doubled quotes instead of escaping them, and the exporter "
    "trimmed the same flag off by mistake. One shared `quote_field()` used "
    "by both fixes it — the exporter was the only caller doing its own "
    "quoting, so this is a deletion, not a refactor.",
    command="rg -n quote_field src/ reports/",
    output="src/exporters/csv_writer.py:142:    return f'\"{field}\"'\n"
    "src/exporters/csv_writer.py:157:    value = value.replace('\"', '\"\"')",
)
_TURNS3 = _turns(
    "The scraper's HTTP client is serial: 40 pages at ~300ms each. "
    "Port the fetch loop to asyncio?",
    "Yes — this workload is I/O-bound, so asyncio.gather over a bounded "
    "semaphore (say 8 concurrent) should drop the wall time from ~12s to "
    "under 2s. Keep the parse step sync; it's CPU-bound and the GIL "
    "would serialise it anyway.",
    command="python -m cProfile -s cumtime scraper/fetch.py > profile.txt",
    output="ncalls  tottime  cumtime  filename:lineno(function)\n"
    "    40    0.004   12.310  scraper/fetch.py:17(fetch_page)",
)
_TURNS4 = _turns(
    "Draft the release notes for v2.4.0. Keep them short, one line per "
    "user-visible change.",
    "v2.4.0: faster search (native candidate scan), a --no-scratch "
    "shorthand for --exclude-cwd, and the TUI now folds long tool outputs. "
    "Three bullets, no migrations, no deprecations.",
)
_TURNS5 = _turns(
    "session-browser needs a stats command: provider breakdown, a daily "
    "activity sparkline, top working directories. Where does it live?",
    "A new `stats` subcommand next to list/search/get, sharing the "
    "filter args. The sparkline buckets updated_at by local calendar day "
    "over --days; providers get one row each with count and percent.",
)
_TURNS6 = _turns(
    "The search test went flaky again — it passed on rerun. Suspect?",
    "The test asserts a fixed corpus ordering while sessions are "
    "discovered newest-first from mtimes. Two runs landed the same "
    "second differently. Sort explicitly in the fixture and the flake "
    "dies; the assertion was always about ordering, never about mtime.",
    command="pytest session_browser/test_search.py -q",
    output="41 passed in 7.93s (the failure did not reproduce — rerun)",
)
_TURNS7 = _turns(
    "Quick experiment with embeddings — does clustering scratchpad "
    "sessions by topic help?",
    "Probably not worth pursuing here; the corpus is too small to "
    "cluster meaningfully.",
)
_TURNS8 = _turns(
    "one-off: check whether the token budget fits a bigger context window",
    "Tried it: yes, but the extra context slowed the first query down.",
)
_TURNS9 = _turns(
    "scratch note: remember to bump the parser buffer for the next profile run",
    "Noted in the session; nothing to change yet.",
)

_CODEX1 = _turns(
    "The deploy pipeline is failing at the docker build step. Look at the "
    "latest run and tell me what broke.",
    "The build stage fails on `pip install -r requirements.txt`: the "
    "lockfile pins a version that was yanked. Pinning the patch release "
    "below it and rebuilding fixes the pipeline; the deploy itself was "
    "never reached.",
)
_CODEX2 = _turns(
    "Add a dark mode toggle to the dashboard. The theme hook exists; "
    "wire the switch and persist the choice.",
    "Wired the toggle to the theme hook and persisted the choice in "
    "localStorage under `dashboard.theme`. The chart palette swaps "
    "colour tokens so the toggle flips everything, not just the chrome.",
)
_CODEX3 = _turns(
    "Profile the search index builder — the incremental path is slower "
    "than the full rebuild, which makes no sense.",
    "The incremental path re-reads the manifest on every batch. Caching "
    "it once per run brings the incremental build below the full rebuild, "
    "as expected.",
)
_CODEX4 = _turns(
    "We're seeing a 502 on the checkout endpoint around 14:00 UTC. "
    "Investigate before the next traffic spike.",
    "The gateway's upstream timeout is 500ms and the checkout's "
    "stock-check call regressed to ~700ms after yesterday's release. "
    "Raising the timeout hides it; the fix is the stock-check index.",
)

_OC1 = _turns(
    "Tune the ripgrep candidate batching: batches of 512 files feel "
    "arbitrary now that argv limits are understood.",
    "The 512-per-batch cap predates the byte-based argv budget. "
    "Batching on bytes alone (256 KB) gives fewer, fuller invocations "
    "and removes a magic number nobody can defend.",
)
_OC2 = _turns(
    "Draft the PR CI workflow: what should it actually run?",
    "Lint, the unit suite, and the perf budget tests. Nothing timed — "
    "counts, not stopwatches, are what regress on CI.",
)


def _specs(here: str) -> list[_Spec]:
    return [
        # Claude — one long session for the fold/collapse demo, real-looking
        # project work, the current checkout (for --here), and scratchpad noise.
        _Spec(
            "claude",
            "8f2c4a1b-9d3e-4f5b-8c6a-1a2b3c4d5e6f",
            "Migrate the widgets service to SQLite",
            "/Users/demo/Projects/widgets",
            "feat/sqlite-migration",
            "",
            3,
            14,
            _TURNS,
        ),
        _Spec(
            "claude",
            "3e7b9c2d-4a8f-4c1e-9b6d-2f3a4b5c6d7e",
            "Fix the CSV export quoting bug",
            "/Users/demo/Projects/reports",
            "fix/csv-export",
            "",
            6,
            11,
            _TURNS2,
        ),
        _Spec(
            "claude",
            "6a1d8e3f-7c2b-4d9e-8a4f-5b6c7d8e9f0a",
            "Refactor the scraper onto asyncio",
            "/Users/demo/Projects/scraper",
            "feat/async-scraper",
            "",
            11,
            16,
            _TURNS3,
        ),
        _Spec(
            "claude",
            "1b9c4d2e-8f7a-4e3d-9c5b-7d8e9f0a1b2c",
            "Write release notes for v2.4.0",
            "/Users/demo/Projects/website",
            "main",
            "",
            22,
            10,
            _TURNS4,
        ),
        _Spec(
            "claude",
            "c4d9e1f7-2b8a-4c6d-9f3e-8a0b1c2d3e4f",
            "Add a stats command to session-browser",
            here,
            "feat/stats-command",
            "",
            2,
            17,
            _TURNS5,
        ),
        _Spec(
            "claude",
            "d2f5a8c3-6e9b-4a1d-8c7f-4b5c6d7e8f9a",
            "Squash the flaky search test",
            here,
            "fix/flaky-search-test",
            "",
            5,
            9,
            _TURNS6,
        ),
        _Spec(
            "claude",
            "a7e2b4f9-3c8d-4b5e-9a6c-1d2e3f4a5b6c",
            "Quick experiment with embeddings",
            f"{SCRATCHPAD}/embedding-experiments",
            "main",
            "",
            0,
            8,
            _TURNS7,
        ),
        _Spec(
            "claude",
            "b8f3c5a0-4d9e-4c6f-8b7d-2e3f4a5b6c7d",
            "one-off: token budget check",
            f"{SCRATCHPAD}/token-budget-check",
            "main",
            "",
            1,
            15,
            _TURNS8,
        ),
        _Spec(
            "claude",
            "e9a4d6b1-5e0f-4d7f-9c8e-3f4a5b6c7d8e",
            "scratch note: parser buffer",
            f"{SCRATCHPAD}/parser-buffer-note",
            "main",
            "",
            2,
            12,
            _TURNS9,
        ),
        # Codex — dated rollout tree, the session_meta/event_msg shape.
        _Spec(
            "codex",
            "019d2f4a-8c33-7b2e-9d41-5f6a7b8c9d01",
            "Debug the failing deploy pipeline",
            "/Users/demo/Projects/website",
            "main",
            "",
            1,
            13,
            _CODEX1,
        ),
        _Spec(
            "codex",
            "019d2f4b-9d44-7c3f-8e52-6a7b8c9d0e12",
            "Add a dark mode toggle to the dashboard",
            "/Users/demo/Projects/dashboard",
            "feat/dark-mode",
            "",
            4,
            18,
            _CODEX2,
        ),
        _Spec(
            "codex",
            "019d2f4c-8e55-7d4f-9f63-7b8c9d0e1f23",
            "Profile the search index builder",
            "/Users/demo/Projects/search-service",
            "main",
            "",
            9,
            10,
            _CODEX3,
        ),
        _Spec(
            "codex",
            "019d2f4c-9f66-7e5f-8a74-8c9d0e1f2a34",
            "Investigate the 502 on checkout",
            "/Users/demo/Projects/storefront",
            "fix/checkout-502",
            "",
            13,
            14,
            _CODEX4,
        ),
        # OpenCode — SQLite-backed, with repository names for --repo demos.
        _Spec(
            "opencode",
            "ses_demo_ripgrep",
            "Tune ripgrep candidate batching",
            here,
            "",
            "session-browser",
            3,
            19,
            _OC1,
        ),
        _Spec(
            "opencode",
            "ses_demo_ci_workflow",
            "Draft the PR CI workflow",
            "/Users/demo/Projects/website",
            "",
            "website",
            25,
            9,
            _OC2,
        ),
    ]


# ---------------------------------------------------------------------------
# Writers
# ---------------------------------------------------------------------------


def _claude_project_dir(cwd: str) -> str:
    """Claude encodes the project path as a flat directory name."""
    return "-" + cwd.lstrip("/").replace("/", "-")


def _write_claude(home: Path, spec: _Spec, now: datetime) -> None:
    project = home / CLAUDE_PROJECTS / _claude_project_dir(spec.cwd)
    project.mkdir(parents=True, exist_ok=True)
    lines: list[str] = []
    for i, turn in enumerate(spec.turns):
        ts = _ts(spec, now, i)
        if turn.role == "user":
            lines.append(
                json.dumps(
                    {
                        "type": "user",
                        "message": {"role": "user", "content": turn.text},
                        "sessionId": spec.id,
                        "cwd": spec.cwd,
                        "gitBranch": spec.branch,
                        "timestamp": ts,
                    }
                )
            )
        elif turn.role == "assistant":
            lines.append(
                json.dumps(
                    {
                        "type": "assistant",
                        "message": {
                            "role": "assistant",
                            "content": [{"type": "text", "text": turn.text}],
                            "model": "claude-sonnet-4-5-20250929",
                        },
                        "sessionId": spec.id,
                        "timestamp": ts,
                    }
                )
            )
        elif turn.role == "tool_call":
            lines.append(
                json.dumps(
                    {
                        "type": "assistant",
                        "message": {
                            "role": "assistant",
                            "content": [
                                {
                                    "type": "tool_use",
                                    "name": "Bash",
                                    "input": {"command": turn.text},
                                }
                            ],
                            "model": "claude-sonnet-4-5-20250929",
                        },
                        "sessionId": spec.id,
                        "timestamp": ts,
                    }
                )
            )
        else:  # tool_output
            lines.append(
                json.dumps(
                    {
                        "type": "user",
                        "message": {
                            "role": "user",
                            "content": [
                                {
                                    "type": "tool_result",
                                    "tool_use_id": f"toolu_{i}",
                                    "content": turn.text,
                                }
                            ],
                        },
                        "sessionId": spec.id,
                        "timestamp": ts,
                    }
                )
            )
    (project / f"{spec.id}.jsonl").write_text("\n".join(lines) + "\n")


def _write_codex(home: Path, spec: _Spec, now: datetime) -> None:
    day = _ts(spec, now)[:10]
    sessions = home / CODEX_SESSIONS / day[:4] / day[5:7] / day[8:10]
    sessions.mkdir(parents=True, exist_ok=True)
    filename = f"rollout-{_dt(spec, now).strftime('%Y-%m-%dT%H-%M-%S')}-{spec.id}.jsonl"
    lines = [
        json.dumps(
            {
                "type": "session_meta",
                "timestamp": _ts(spec, now),
                "payload": {
                    "id": spec.id,
                    "cwd": spec.cwd,
                    "git": {"branch": spec.branch},
                },
            }
        )
    ]
    for i, turn in enumerate(spec.turns):
        ts = _ts(spec, now, i + 1)
        if turn.role == "user":
            lines.append(
                json.dumps(
                    {
                        "type": "event_msg",
                        "timestamp": ts,
                        "payload": {"type": "user_message", "message": turn.text},
                    }
                )
            )
        elif turn.role == "assistant":
            # Codex records each answer twice — agent_message and a
            # response_item — and the parser dedupes the adjacent pair.
            lines.append(
                json.dumps(
                    {
                        "type": "event_msg",
                        "timestamp": ts,
                        "payload": {"type": "agent_message", "message": turn.text},
                    }
                )
            )
            lines.append(
                json.dumps(
                    {
                        "type": "event_msg",
                        "timestamp": ts,
                        "payload": {
                            "type": "response_item",
                            "payload": {
                                "type": "message",
                                "role": "assistant",
                                "content": [{"type": "output_text", "text": turn.text}],
                            },
                        },
                    }
                )
            )
        elif turn.role == "tool_output":
            lines.append(
                json.dumps(
                    {
                        "type": "event_msg",
                        "timestamp": ts,
                        "payload": {"type": "patch_apply_end", "stdout": turn.text},
                    }
                )
            )
        # tool_call turns have no codex equivalent in the parser; skip.
    lines.append(
        json.dumps(
            {
                "type": "event_msg",
                "timestamp": _ts(spec, now, len(spec.turns) + 2),
                "payload": {"type": "task_complete"},
            }
        )
    )
    (sessions / filename).write_text("\n".join(lines) + "\n")


_OPENCODE_SCHEMA = [
    """CREATE TABLE project (
        id TEXT PRIMARY KEY, worktree TEXT NOT NULL,
        name TEXT, time_created INTEGER NOT NULL, time_updated INTEGER NOT NULL,
        sandboxes TEXT NOT NULL DEFAULT '[]'
    )""",
    """CREATE TABLE session (
        id TEXT PRIMARY KEY, project_id TEXT NOT NULL,
        slug TEXT NOT NULL DEFAULT '', directory TEXT NOT NULL,
        title TEXT NOT NULL DEFAULT '', version TEXT NOT NULL DEFAULT '1',
        time_created INTEGER NOT NULL, time_updated INTEGER NOT NULL
    )""",
    """CREATE TABLE message (
        id TEXT PRIMARY KEY, session_id TEXT NOT NULL,
        time_created INTEGER NOT NULL, time_updated INTEGER NOT NULL,
        data TEXT NOT NULL
    )""",
    """CREATE TABLE part (
        id TEXT PRIMARY KEY, message_id TEXT NOT NULL,
        session_id TEXT NOT NULL,
        time_created INTEGER NOT NULL, time_updated INTEGER NOT NULL,
        data TEXT NOT NULL
    )""",
]


def _epoch_ms(dt: datetime) -> int:
    return int(dt.timestamp() * 1000)


def _write_opencode(home: Path, specs: list[_Spec], now: datetime) -> None:
    db_path = home / OPENCODE_DB
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path))
    try:
        for ddl in _OPENCODE_SCHEMA:
            conn.execute(ddl)
        for n, spec in enumerate(specs, 1):
            project_id = f"proj_demo_{n}"
            created = _epoch_ms(_dt(spec, now))
            updated = _epoch_ms(_dt(spec, now, max(1, len(spec.turns) - 1)))
            conn.execute(
                "INSERT INTO project VALUES (?, ?, ?, ?, ?, ?)",
                (
                    project_id,
                    spec.cwd,
                    spec.repository,
                    created,
                    updated,
                    "[]",
                ),
            )
            conn.execute(
                "INSERT INTO session (id, project_id, slug, directory, title, "
                "version, time_created, time_updated) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    spec.id,
                    project_id,
                    spec.id.removeprefix("ses_demo_"),
                    spec.cwd,
                    spec.summary,
                    "1",
                    created,
                    updated,
                ),
            )
            for i, turn in enumerate(spec.turns):
                msg_id = f"msg_{spec.id}_{i}"
                msg_ts = _epoch_ms(_dt(spec, now, i))
                role = (
                    "assistant" if turn.role in ("assistant", "tool_output") else "user"
                )
                conn.execute(
                    "INSERT INTO message VALUES (?, ?, ?, ?, ?)",
                    (msg_id, spec.id, msg_ts, msg_ts, json.dumps({"role": role})),
                )
                part_data = (
                    {"type": "tool", "tool": "bash", "state": {"output": turn.text}}
                    if turn.role == "tool_output"
                    else {"type": "text", "text": turn.text}
                )
                conn.execute(
                    "INSERT INTO part VALUES (?, ?, ?, ?, ?, ?)",
                    (
                        f"part_{spec.id}_{i}",
                        msg_id,
                        spec.id,
                        msg_ts,
                        msg_ts,
                        json.dumps(part_data),
                    ),
                )
        conn.commit()
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Entry points
# ---------------------------------------------------------------------------


def generate(
    home: Path, *, now: datetime | None = None, force: bool = False
) -> DemoResult:
    """Write the demo corpus under *home*; returns what was written.

    Refuses to touch a non-empty directory unless *force* is set, so a
    mistyped path can't quietly delete real data. *now* anchors every
    timestamp (defaults to the real present); tests pass a fixed one.
    """
    now = now or datetime.now(UTC)
    if home.exists():
        if not home.is_dir():
            raise FileExistsError(f"{home} exists and is not a directory")
        if force:
            shutil.rmtree(home)
        else:
            raise FileExistsError(
                f"{home} already exists — pass --force to regenerate it"
            )
    specs = _specs(str(Path.cwd().resolve()))
    by_provider: dict[str, list[_Spec]] = {}
    for spec in specs:
        by_provider.setdefault(spec.provider, []).append(spec)
    for spec in by_provider.get("claude", []):
        _write_claude(home, spec, now)
    for spec in by_provider.get("codex", []):
        _write_codex(home, spec, now)
    oc = by_provider.get("opencode", [])
    if oc:
        _write_opencode(home, oc, now)
    return DemoResult(
        home,
        len(by_provider.get("claude", [])),
        len(by_provider.get("codex", [])),
        len(oc),
    )


def _usage(home: Path) -> str:
    return "\n".join(
        [
            "",
            "Run against it (HOME decides every provider path):",
            f"  HOME={home} session-browser",
            f"  HOME={home} session-browser list --json",
            f'  HOME={home} session-browser search "sqlite"',
            f"  HOME={home} session-browser list --exclude-cwd scratchpad",
            "",
            "Regenerate with --force.",
        ]
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m session_browser.demo",
        description="Write a fake agent-session history (Claude, Codex, "
        "OpenCode) under a directory you can point HOME at.",
    )
    parser.add_argument(
        "--home",
        type=Path,
        default=Path(DEFAULT_HOME),
        help=f"target directory (default: ./{DEFAULT_HOME})",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="replace an existing directory instead of refusing",
    )
    args = parser.parse_args(argv)
    try:
        result = generate(args.home, force=args.force)
    except (FileExistsError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(
        f"Wrote {result.total} sessions to {result.home} "
        f"({result.claude} claude, {result.codex} codex, "
        f"{result.opencode} opencode)."
    )
    print(_usage(result.home))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
