"""Deterministic work-accounting for the retrieval hot paths.

Timing cannot police performance in this repository, and that is a measured
conclusion rather than a preference. ``benchmarks/retrieval_compare.py``
already encodes it: within-revision spread on the real corpus exceeds 100% of
the median, which is why that tool reports ``unresolvable`` instead of
pretending a ratio it cannot resolve is a regression. A wall-clock threshold in
the test suite would therefore be two bad things at once -- flaky on changes
that cost nothing, and silent on the changes that matter most, because the
regressions actually worth catching here are individually far below the noise
floor. The per-row ``if progress is not None`` branch that prompted this module
measured +0.18% against a 0.4-1.0% spread. Unmeasurable, real, and exactly the
kind of thing that accumulates.

So this module does not time anything. It counts *work*: transcripts parsed,
bytes read off the corpus, ripgrep invocations, SQL statements, sessions handed
to the process pool, progress callbacks fired. Those are integers. They are
identical on every machine and every run, they are computed in milliseconds,
and they move the instant someone drops the prefilter, reads a file twice,
routes the wrong shape to worker processes, or makes the CLI pay for a readout
only the TUI displays.

Each counter is pinned to an exact number in ``docs/perf_budgets.json``. Any
movement in either direction fails, because a drop is as much a signal as a
rise -- it usually means a workload stopped exercising the path it was written
to guard. Accepting a change is a deliberate act:

    python -m session_browser.perf_budget bless

which rewrites the budgets file so the new numbers land in the diff and get
argued about in review, instead of eroding quietly.

What this does *not* do is tell you whether a change is fast. It tells you
whether a change altered the amount of work done, which is the question you can
actually answer reliably. When a budget legitimately moves, that is the moment
to reach for ``benchmarks/retrieval_compare.py`` and decide whether the extra
work pays for itself.

Those counters work at the grain of a file, a statement, a parse, so CPU added
*inside* a loop doing the same I/O moves none of them. A second layer covers
most of that: the loop probes below count executed Python opcodes over fixed
inputs -- still a count, still identical on every machine, but fine enough to
see a loop body change. Where a structure was already chosen for this reason it
is also pinned directly; ``row_tick_wrappers`` predates the probes and still
guards that the opencode cursor is wrapped at all.

What remains invisible, stated plainly so nobody trusts a green run further
than it deserves: work that stays inside C. Swapping ``str.find`` for a regex
is about one opcode either way. That residue is a job for code review and for
``benchmarks/retrieval_compare.py``, and no green run here is a substitute for
either.
"""

from __future__ import annotations

import argparse
import builtins
import contextlib
import datetime
import io
import json
import os
import shutil
import sqlite3
import sys
import threading
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from pathlib import Path

from . import discovery, transcript

BUDGETS_PATH = Path(__file__).resolve().parent.parent / "docs" / "perf_budgets.json"

# Terms chosen so each exercises a different fate in the prefilter. Their
# selectivity is a property of the synthetic corpus below, not of the machine.
QUERY_ABSENT = "zzqabsentzz"  # in no session: nothing may be parsed at all
QUERY_RARE = "kookaburra"  # in a known handful: the prefilter's whole job
QUERY_COMMON = "session"  # in every session: the unavoidable upper bound


# ---------------------------------------------------------------------------
# The ledger
# ---------------------------------------------------------------------------


@dataclass
class WorkLedger:
    """Thread-safe tallies of work done against the corpus.

    The search path runs probes across a thread pool, so every increment can
    race. A lock rather than an atomic-by-accident idiom keeps that honest:
    these numbers are asserted on exactly, and a lost increment would present
    as a phantom regression on an unrelated change.
    """

    counts: dict[str, int] = field(default_factory=dict)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def add(self, name: str, amount: int = 1) -> None:
        with self._lock:
            self.counts[name] = self.counts.get(name, 0) + amount

    def snapshot(self) -> dict[str, int]:
        """Every counter, including the ones that stayed at zero.

        A counter absent from the report and a counter at zero must not look
        alike: "the CLI fired no progress callbacks" is a guarantee this module
        exists to enforce, and it can only be enforced if the zero is written
        down.
        """
        with self._lock:
            return {name: self.counts.get(name, 0) for name in COUNTERS}


COUNTERS = (
    "corpus_bytes_read",
    "corpus_file_opens",
    "transcripts_parsed",
    "prefilter_file_scans",
    "rg_subprocess_calls",
    "sqlite_connections",
    "sqlite_statements",
    "opencode_sessions_parsed",
    "process_pool_sessions",
    "progress_callbacks",
    "row_tick_wrappers",
)


class _CountingReader:
    """A file object that reports how much of the corpus was consumed.

    Wraps rather than subclasses because the underlying object differs by mode
    (buffered binary reader, text wrapper) and only the read surface needs
    intercepting; everything else delegates untouched so the wrapped code
    cannot tell the difference.
    """

    def __init__(self, fh, ledger: WorkLedger) -> None:
        self._fh = fh
        self._ledger = ledger

    def read(self, *args, **kwargs):
        data = self._fh.read(*args, **kwargs)
        self._ledger.add("corpus_bytes_read", len(data))
        return data

    def readline(self, *args, **kwargs):
        data = self._fh.readline(*args, **kwargs)
        self._ledger.add("corpus_bytes_read", len(data))
        return data

    def readlines(self, *args, **kwargs):
        lines = self._fh.readlines(*args, **kwargs)
        self._ledger.add("corpus_bytes_read", sum(len(x) for x in lines))
        return lines

    def __iter__(self):
        return self

    def __next__(self):
        line = next(self._fh)
        self._ledger.add("corpus_bytes_read", len(line))
        return line

    def __enter__(self):
        self._fh.__enter__()
        return self

    def __exit__(self, *exc):
        return self._fh.__exit__(*exc)

    def __getattr__(self, name):
        return getattr(self._fh, name)


@contextlib.contextmanager
def instrumented(
    ledger: WorkLedger, corpus: Path, *, disable_rg: bool = False
) -> Iterator[None]:
    """Count corpus work for the duration, changing no behaviour.

    Instrumentation lives here and never in the shipped code, which is the
    point: a counter compiled into a hot loop is itself the regression this
    module was written to catch. Everything is patched at the boundary of the
    module under test and restored on exit.

    The one deliberate behavioural override is the process pool, which is
    replaced by a recorder that returns the existing "use threads instead"
    sentinel. Worker processes would take the parse out of this interpreter
    where nothing can see it, and would make the numbers depend on whether the
    host can spawn. Recording the *routing decision* -- how many sessions the
    parent chose to send -- pins the strategy, which is the part a change can
    get wrong; the parsing is then counted in-process as usual.
    """
    real_open = builtins.open
    real_read_bytes = Path.read_bytes
    real_connect = sqlite3.connect
    real_run = transcript.subprocess.run
    real_iter_entries = transcript.iter_entries
    real_file_may_match = transcript._file_may_match
    real_search_entries = transcript._search_entries
    real_probe = transcript._probe_in_processes
    real_ticking_rows = transcript._ticking_rows
    real_rg = transcript._rg_candidate_paths
    corpus_str = str(corpus)

    def in_corpus(target) -> bool:
        try:
            # Containment, not a prefix test: SQLite is opened through a
            # read-only URI (``file:/path/to/db?mode=ro``), so the corpus path
            # sits in the middle of the string rather than at its start.
            return corpus_str in str(target)
        except (TypeError, ValueError):
            return False

    def counting_open(file, *args, **kwargs):
        fh = real_open(file, *args, **kwargs)
        if not in_corpus(file):
            return fh
        ledger.add("corpus_file_opens")
        return _CountingReader(fh, ledger)

    def counting_read_bytes(self):
        data = real_read_bytes(self)
        if in_corpus(self):
            ledger.add("corpus_file_opens")
            ledger.add("corpus_bytes_read", len(data))
        return data

    def counting_connect(database, *args, **kwargs):
        conn = real_connect(database, *args, **kwargs)
        if in_corpus(database):
            ledger.add("sqlite_connections")
            # A trace callback counts statements without proxying the
            # connection, so callers keep the real object and can still set
            # row_factory or reach any attribute a proxy would have to forward.
            conn.set_trace_callback(lambda _sql: ledger.add("sqlite_statements"))
        return conn

    def counting_run(args, *rest, **kwargs):
        if args and str(args[0]).endswith("rg"):
            ledger.add("rg_subprocess_calls")
        return real_run(args, *rest, **kwargs)

    def counting_iter_entries(session, *args, **kwargs):
        # The single gate every canonical parse passes through, whether it was
        # reached by search, by `get`, or by the TUI's transcript cache. A
        # counter on `search_session` would have missed retrieval entirely.
        ledger.add("transcripts_parsed")
        return real_iter_entries(session, *args, **kwargs)

    def counting_file_may_match(path, *args, **kwargs):
        ledger.add("prefilter_file_scans")
        return real_file_may_match(path, *args, **kwargs)

    def counting_search_entries(session, *args, **kwargs):
        if session.provider == "opencode":
            ledger.add("opencode_sessions_parsed")
        return real_search_entries(session, *args, **kwargs)

    def counting_ticking_rows(rows, progress):
        # Counting the wrapper rather than the ticks it emits. The distinction
        # is the whole point: progress reporting has to stay *outside* the row
        # loop, and an inline `if progress is not None` per row would emit
        # identical ticks while costing the CLI a test it can never benefit
        # from. That rewrite is invisible to every other counter here -- it
        # adds no I/O, no parse, no statement -- so the structure gets pinned
        # instead of its output.
        ledger.add("row_tick_wrappers")
        return real_ticking_rows(rows, progress)

    def recording_probe_in_processes(pairs, query, cancelled):
        ledger.add("process_pool_sessions", len(pairs))
        return transcript._PROC_FALLBACK

    def no_rg(sessions, needles):
        return None

    builtins.open = counting_open
    Path.read_bytes = counting_read_bytes
    sqlite3.connect = counting_connect
    transcript.subprocess.run = counting_run
    transcript.iter_entries = counting_iter_entries
    transcript._file_may_match = counting_file_may_match
    transcript._search_entries = counting_search_entries
    transcript._probe_in_processes = recording_probe_in_processes
    transcript._ticking_rows = counting_ticking_rows
    if disable_rg:
        transcript._rg_candidate_paths = no_rg
    try:
        yield
    finally:
        builtins.open = real_open
        Path.read_bytes = real_read_bytes
        sqlite3.connect = real_connect
        transcript.subprocess.run = real_run
        transcript.iter_entries = real_iter_entries
        transcript._file_may_match = real_file_may_match
        transcript._search_entries = real_search_entries
        transcript._probe_in_processes = real_probe
        transcript._ticking_rows = real_ticking_rows
        transcript._rg_candidate_paths = real_rg


# ---------------------------------------------------------------------------
# The corpus
# ---------------------------------------------------------------------------

CLAUDE_SESSIONS = 180  # above _PROC_MIN_CANDIDATES, so the process
CODEX_SESSIONS = 40  # routing decision is reachable and pinnable
OPENCODE_SESSIONS = 40
PI_SESSIONS = 40
RARE_EVERY = 60  # one session in sixty carries the rare term


def _filler(index: int, line: int) -> str:
    """Deterministic prose with no incidental hits for any pinned query.

    Content is ASCII on purpose. Text-mode reads count characters and binary
    reads count bytes, and the two only agree while the corpus stays ASCII --
    which keeps ``corpus_bytes_read`` a single meaningful number instead of a
    sum of two units.
    """
    return (
        f"turn {line} of session {index} covering retries, offsets, "
        f"buffers and the ordering of parsed output"
    )


def _has_rare(index: int) -> bool:
    return index % RARE_EVERY == 0


def build_corpus(root: Path) -> Path:
    """Write a fixed provider-native home tree under *root* and return it.

    Synthetic rather than the author's real sessions: budgets are exact
    integers, so the corpus has to be identical for everyone who runs the
    check. It is built fresh per run rather than committed, because a few
    hundred small files cost milliseconds to write and a committed tree would
    drift out of step with the parsers that read it.
    """
    home = root / "home"
    claude_root = home / ".claude" / "projects"
    for i in range(CLAUDE_SESSIONS):
        project = claude_root / f"-Users-perf-project{i % 3}"
        project.mkdir(parents=True, exist_ok=True)
        lines = [
            json.dumps(
                {
                    "type": "user",
                    "cwd": f"/Users/perf/project{i % 3}",
                    "gitBranch": "main",
                    "timestamp": f"2026-01-{(i % 28) + 1:02d}T09:00:00.000Z",
                    "message": {
                        "role": "user",
                        "content": f"start session {i} {_filler(i, 0)}",
                    },
                }
            )
        ]
        for line in range(1, 20):
            text = _filler(i, line)
            if _has_rare(i) and line == 7:
                text = f"{text} {QUERY_RARE} sighted"
            lines.append(
                json.dumps(
                    {
                        "type": "assistant",
                        "timestamp": f"2026-01-{(i % 28) + 1:02d}T09:0{line % 10}:00.000Z",
                        "message": {
                            "role": "assistant",
                            "content": [{"type": "text", "text": text}],
                        },
                    }
                )
            )
        (project / f"perf-claude-{i:04d}.jsonl").write_text("\n".join(lines) + "\n")

    codex_root = home / ".codex" / "sessions" / "2026" / "01" / "05"
    codex_root.mkdir(parents=True, exist_ok=True)
    for i in range(CODEX_SESSIONS):
        sid = f"perf-codex-{i:04d}"
        lines = [
            json.dumps(
                {
                    "type": "session_meta",
                    "payload": {
                        "id": sid,
                        "cwd": f"/Users/perf/project{i % 3}",
                        "git": {"branch": "main"},
                        "timestamp": "2026-01-05T09:00:00.000Z",
                    },
                }
            ),
            json.dumps(
                {
                    "type": "event_msg",
                    "payload": {
                        "type": "user_message",
                        "message": f"start session {i} {_filler(i, 0)}",
                    },
                }
            ),
        ]
        for line in range(1, 20):
            text = _filler(i, line)
            if _has_rare(i) and line == 7:
                text = f"{text} {QUERY_RARE} sighted"
            lines.append(
                json.dumps(
                    {
                        "type": "event_msg",
                        "payload": {"type": "agent_message", "message": text},
                    }
                )
            )
        (codex_root / f"rollout-2026-01-05T09-00-00-{sid}.jsonl").write_text(
            "\n".join(lines) + "\n"
        )

    codex_db = home / ".codex" / "state_5.sqlite"
    conn = sqlite3.connect(str(codex_db))
    conn.execute(
        "CREATE TABLE threads (id TEXT PRIMARY KEY, rollout_path TEXT NOT NULL, "
        "cwd TEXT NOT NULL, title TEXT NOT NULL DEFAULT '', "
        "git_branch TEXT, first_user_message TEXT NOT NULL DEFAULT '', "
        "created_at_ms INTEGER, updated_at_ms INTEGER, "
        "archived INTEGER NOT NULL DEFAULT 0)"
    )
    for i in range(CODEX_SESSIONS):
        sid = f"perf-codex-{i:04d}"
        rollout = codex_root / f"rollout-2026-01-05T09-00-00-{sid}.jsonl"
        conn.execute(
            "INSERT INTO threads (id, rollout_path, cwd, git_branch, "
            "first_user_message, created_at_ms, updated_at_ms, archived) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, 0)",
            (
                sid,
                str(rollout),
                f"/Users/perf/project{i % 3}",
                "main",
                f"start session {i} {_filler(i, 0)}",
                1767603600000,
                1767603600000,
            ),
        )
    conn.commit()
    conn.close()

    pi_root = home / ".pi" / "agent" / "sessions"
    for i in range(PI_SESSIONS):
        project = pi_root / f"--Users-perf-project{i % 3}--"
        project.mkdir(parents=True, exist_ok=True)
        sid = f"perf-pi-{i:04d}"
        day = f"2026-01-{(i % 28) + 1:02d}"
        lines = [
            json.dumps(
                {
                    "type": "session",
                    "version": 3,
                    "id": sid,
                    "timestamp": f"{day}T09:00:00.000Z",
                    "cwd": f"/Users/perf/project{i % 3}",
                }
            ),
            json.dumps(
                {
                    "type": "message",
                    "id": f"{sid}-m000",
                    "parentId": None,
                    "timestamp": f"{day}T09:00:01.000Z",
                    "message": {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": f"start session {i} {_filler(i, 0)}",
                            }
                        ],
                    },
                }
            ),
        ]
        for line in range(1, 20):
            text = _filler(i, line)
            if _has_rare(i) and line == 7:
                text = f"{text} {QUERY_RARE} sighted"
            lines.append(
                json.dumps(
                    {
                        "type": "message",
                        "id": f"{sid}-m{line:03d}",
                        "parentId": f"{sid}-m{line - 1:03d}",
                        "timestamp": f"{day}T09:0{line % 10}:00.000Z",
                        "message": {
                            "role": "assistant",
                            "content": [{"type": "text", "text": text}],
                        },
                    }
                )
            )
        (project / f"{day}T09-00-00-000Z_{sid}.jsonl").write_text(
            "\n".join(lines) + "\n"
        )

    db_path = home / ".local" / "share" / "opencode" / "opencode.db"
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path))
    conn.execute(
        "CREATE TABLE project (id TEXT PRIMARY KEY, worktree TEXT NOT NULL, "
        "name TEXT, time_created INTEGER NOT NULL, "
        "time_updated INTEGER NOT NULL, sandboxes TEXT NOT NULL DEFAULT '[]')"
    )
    conn.execute(
        "CREATE TABLE session (id TEXT PRIMARY KEY, project_id TEXT NOT NULL, "
        "slug TEXT NOT NULL DEFAULT '', directory TEXT NOT NULL, "
        "title TEXT NOT NULL DEFAULT '', version TEXT NOT NULL DEFAULT '1', "
        "time_created INTEGER NOT NULL, time_updated INTEGER NOT NULL)"
    )
    conn.execute(
        "CREATE TABLE message (id TEXT PRIMARY KEY, session_id TEXT NOT NULL, "
        "time_created INTEGER NOT NULL, time_updated INTEGER NOT NULL, "
        "data TEXT NOT NULL)"
    )
    conn.execute(
        "CREATE TABLE part (id TEXT PRIMARY KEY, message_id TEXT NOT NULL, "
        "session_id TEXT NOT NULL, time_created INTEGER NOT NULL, "
        "time_updated INTEGER NOT NULL, data TEXT NOT NULL)"
    )
    conn.execute(
        "INSERT INTO project VALUES (?, ?, ?, ?, ?, ?)",
        (
            "perf-proj",
            "/Users/perf/project0",
            "project0",
            1700000000000,
            1700000000000,
            "[]",
        ),
    )
    for i in range(OPENCODE_SESSIONS):
        sid = f"perf-oc-{i:04d}"
        conn.execute(
            "INSERT INTO session (id, project_id, slug, directory, title, version, "
            "time_created, time_updated) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (
                sid,
                "perf-proj",
                f"s{i}",
                "/Users/perf/project0",
                f"start session {i}",
                "1",
                1700000000000,
                1700000000000 + i,
            ),
        )
        for line in range(20):
            mid = f"{sid}-m{line:03d}"
            role = "user" if line == 0 else "assistant"
            conn.execute(
                "INSERT INTO message VALUES (?, ?, ?, ?, ?)",
                (
                    mid,
                    sid,
                    1700000000000 + line,
                    1700000000000 + line,
                    json.dumps({"role": role}),
                ),
            )
            text = _filler(i, line)
            if _has_rare(i) and line == 7:
                text = f"{text} {QUERY_RARE} sighted"
            conn.execute(
                "INSERT INTO part VALUES (?, ?, ?, ?, ?, ?)",
                (
                    f"{mid}-p0",
                    mid,
                    sid,
                    1700000000000 + line,
                    1700000000000 + line,
                    json.dumps({"type": "text", "text": text}),
                ),
            )
    conn.commit()
    conn.close()
    return home


# ---------------------------------------------------------------------------
# The workloads
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Workload:
    """One named scenario, and the optimisation its counters exist to defend.

    ``guards`` is not decoration. When a budget breaks, the failure report
    prints it, because the number alone does not tell the next person what
    property they just broke or why anyone cared.
    """

    name: str
    guards: str
    run: Callable[[], None]
    needs_rg: bool = False
    disable_rg: bool = False


def _cli(*argv: str) -> None:
    """Drive the real command, discarding its output.

    Through ``run_cli`` rather than the library functions underneath, so the
    budgets cover argument handling, discovery and rendering as a user meets
    them -- a regression bolted onto the command layer counts just as much as
    one in the parser.
    """
    from .cli import run_cli

    with (
        contextlib.redirect_stdout(io.StringIO()),
        contextlib.redirect_stderr(io.StringIO()),
    ):
        run_cli(list(argv))


def _corpus_home_guard() -> Path:
    """The corpus home, refusing anything that is not one.

    The helpers below move files around inside it; a caller that runs them
    against a real home would hide the user's actual Codex index. The corpus
    home always sits inside a ``sb-perf-`` tempdir, which a real home never
    does."""
    home = Path(os.environ["HOME"])
    parent = home.parent.name
    if not parent.startswith(("perf", "sb-perf")):
        raise AssertionError(f"refusing to touch non-corpus home {home}")
    return home


def _without_codex_db(run: Callable[[], None]) -> None:
    """Run *run* with the corpus Codex index hidden, exercising the fallback.

    The corpus home is the process HOME while workloads run, so the index is
    found and hidden through the environment. Restored afterwards even on
    failure: a rename touches no counters, so this changes nothing except
    which discovery path the workload takes.
    """
    home = _corpus_home_guard()
    db = home / ".codex" / "state_5.sqlite"
    hidden = home / ".codex" / "state_5.sqlite.hidden"
    db.rename(hidden)
    try:
        run()
    finally:
        hidden.rename(db)


def _with_extra_codex_rollout(run: Callable[[], None]) -> None:
    """Run *run* with one unindexed rollout added, exercising the stale path.

    The index-present-but-disagreeing fallback is the one that fires in real
    life (archive, backfill, a just-written session), and it pays the file
    scan *plus* the failed DB query. Removed afterwards even on failure."""
    home = _corpus_home_guard()
    extra = (
        home
        / ".codex"
        / "sessions"
        / "2026"
        / "01"
        / "05"
        / "rollout-2026-01-05T09-00-00-perf-extra.jsonl"
    )
    extra.write_text(
        json.dumps(
            {
                "type": "session_meta",
                "payload": {
                    "id": "perf-codex-extra",
                    "cwd": "/Users/perf/project0",
                    "git": {"branch": "main"},
                    "timestamp": "2026-01-05T09:00:00.000Z",
                },
            }
        )
        + "\n"
        + json.dumps(
            {
                "type": "event_msg",
                "payload": {"type": "user_message", "message": "extra session"},
            }
        )
        + "\n"
    )
    try:
        run()
    finally:
        extra.unlink()


def _tui_search(ledger: WorkLedger, query: str) -> None:
    """The TUI's shape: full entries retained, progress reported throughout."""
    sessions = discovery.discover_all()
    transcript.search_sessions(
        sessions,
        query,
        keep_entries=True,
        progress=lambda *_a: ledger.add("progress_callbacks"),
    )


def workloads(ledger: WorkLedger) -> list[Workload]:
    return [
        Workload(
            "cli.search.absent",
            "A term in no session must cost zero canonical parses: the "
            "prefilter, not the parser, answers a miss.",
            lambda: _cli("search", QUERY_ABSENT),
            needs_rg=True,
        ),
        Workload(
            "cli.search.rare",
            "The prefilter must rule out the corpus and leave only genuine "
            "candidates to parse. This is the number that grows to the whole "
            "corpus if the native scan is dropped.",
            lambda: _cli("search", QUERY_RARE),
            needs_rg=True,
        ),
        Workload(
            "cli.search.rare.no_rg",
            "Same, with ripgrep unavailable: the in-process byte prefilter "
            "must still read each candidate file exactly once, not once to "
            "partition and again to parse.",
            lambda: _cli("search", QUERY_RARE),
            disable_rg=True,
        ),
        Workload(
            "cli.search.common",
            "The unavoidable upper bound -- a term in every session. Nothing "
            "may read the corpus more than once even when nothing is skipped.",
            lambda: _cli("search", QUERY_COMMON),
            needs_rg=True,
        ),
        Workload(
            "cli.search.claude",
            "Provider scoping must confine the work to that provider: no "
            "SQLite connection is opened when opencode was not asked for.",
            lambda: _cli("search", QUERY_RARE, "--provider", "claude"),
            needs_rg=True,
        ),
        Workload(
            "cli.search.opencode",
            "The thin raw-part scan must rule sessions out before the ordered "
            "join, so only survivors reach a canonical parse.",
            lambda: _cli("search", QUERY_RARE, "--provider", "opencode"),
        ),
        Workload(
            "cli.search.pi",
            "Pi transcripts are ordinary JSONL, so they must be ruled out by "
            "the same candidate scan the claude files go through. A parse per "
            "session here means Pi fell out of the prefilter path.",
            lambda: _cli("search", QUERY_RARE, "--provider", "pi"),
            needs_rg=True,
        ),
        Workload(
            "cli.list",
            "Listing is metadata only. It must never parse a transcript, and "
            "its read volume must stay bounded by the discovery window rather "
            "than by session length.",
            lambda: _cli("list", "--limit", "50"),
        ),
        Workload(
            "cli.list.no_codex_db",
            "The file-scan fallback is the safety net for a missing or stale "
            "Codex index. It must stay budgeted so a regression in it cannot "
            "hide behind the DB fast path.",
            lambda: _without_codex_db(lambda: _cli("list", "--limit", "50")),
        ),
        Workload(
            "cli.list.codex_db_stale",
            "The disagreement fallback pays the file scan *plus* the failed "
            "DB query (extra sqlite_statement). Budgeted so a change cannot "
            "make the slowest path silently pay more.",
            lambda: _with_extra_codex_rollout(lambda: _cli("list", "--limit", "50")),
        ),
        Workload(
            "cli.stats",
            "Statistics come from discovery metadata alone; no transcript "
            "may be parsed to produce them.",
            lambda: _cli("stats"),
        ),
        Workload(
            "cli.get",
            "Retrieving one session must read one transcript, on top of the "
            "discovery pass that located it.",
            lambda: _cli("get", "claude:perf-claude-0000"),
        ),
        Workload(
            "tui.search.rare",
            "The progress readout the TUI displays. Callback volume is capped "
            "by design -- per session and per 4096 database rows, never per "
            "row or per entry.",
            lambda: _tui_search(ledger, QUERY_RARE),
            needs_rg=True,
        ),
        Workload(
            "lib.search.process_route",
            "The ids/snippets shape above the candidate threshold must be "
            "routed to worker processes, and the prefiltered-out majority "
            "must not be sent with it.",
            lambda: transcript.search_sessions(
                discovery.discover_all(), QUERY_COMMON, keep_entries=False
            ),
            needs_rg=True,
        ),
        Workload(
            "lib.search.keep_entries",
            "The entries-retaining shape must stay in this process: shipping "
            "whole transcripts back from workers would pickle the corpus.",
            lambda: transcript.search_sessions(
                discovery.discover_all(), QUERY_COMMON, keep_entries=True
            ),
            needs_rg=True,
        ),
    ]


# ---------------------------------------------------------------------------
# The loop probes
# ---------------------------------------------------------------------------
#
# The workloads above count work at the granularity of a file, a statement, a
# parse. That is the right grain for the questions they answer, and it is why
# they are blind to CPU added *inside* a loop that reads the same files and
# runs the same statements. Everything below exists to close that gap.
#
# The measure is executed Python opcodes. It keeps the property that makes this
# module worth having -- it is a count, not a time, so it is identical on every
# machine and every run -- while resolving effects a stopwatch cannot see. The
# two changes that prompted it measure clearly:
#
#     try/except -> contextlib.suppress   9,007 -> 20,507 opcodes  (caught)
#     manual counter -> enumerate()       5,507 ->  4,509 opcodes  (improvement)
#
# The second is the point. This does not merely flag change; it tells you the
# direction, which is what distinguishes a regression from a win and what a
# plausible-sounding argument about CPython internals cannot be trusted to do.
#
# Each probe drives one loop over a fixed input, single-threaded, with no
# corpus and no pool. Small inputs on purpose: the number has to be legible
# enough that a diff of "+9" points at a line, and every probe traces in
# milliseconds despite tracing costing ~22x.
#
# What this still cannot see: work that stays inside C. Swapping ``str.find``
# for a regex is about one opcode either way. It narrows the blind spot from
# "all in-loop CPU" to "in-loop CPU that never returns to the interpreter" --
# smaller, and far more visible in review.
#
# One constraint worth stating: opcode counts are a property of the
# interpreter. They shift between Python versions, so the budgets file records
# the version they were taken on and a mismatch is reported as such rather than
# as a regression.


def count_opcodes(fn: Callable[[], object]) -> int:
    """Executed Python opcodes inside *fn*, counted exactly.

    Every Python frame is counted, stdlib included, because that is where the
    cost being defended against actually lands: ``contextlib.suppress`` is a
    Python context manager, and a probe that only counted this package's own
    frames would score it identical to ``try``/``except``. Work implemented in
    C stays invisible, which is the documented limit of the technique rather
    than an oversight.

    Restores any tracer that was already installed instead of assuming there
    was none, so running under a debugger or a coverage tool leaves the outer
    tool working rather than silently disabled.
    """
    n = 0

    def tracer(frame, event, arg):
        nonlocal n
        if event == "call":
            frame.f_trace_opcodes = True
            frame.f_trace = tracer
            return tracer
        if event == "opcode":
            n += 1
        return tracer

    previous = sys.gettrace()
    sys.settrace(tracer)
    try:
        fn()
    finally:
        sys.settrace(previous)
    return n


@dataclass(frozen=True)
class LoopProbe:
    """One hot loop, its fixed input, and the property its count defends."""

    name: str
    guards: str
    run: Callable[[], None]


def _probe_entries(count: int = 40) -> list:
    """Entries whose text carries a known number of hits for "offset"."""
    return [
        transcript.TranscriptEntry(
            role="user",
            text=(
                f"turn {i} discussing an offset, then another offset, "
                f"with buffers and retries in between"
            ),
        )
        for i in range(count)
    ]


def _probe_jsonl(root: Path) -> Path:
    """A small JSONL transcript, written once and reused by the probes."""
    path = root / "probe.jsonl"
    if not path.is_file():
        path.parent.mkdir(parents=True, exist_ok=True)
        lines = [
            json.dumps(
                {
                    "type": "user",
                    "message": {
                        "role": "user",
                        "content": f"line {i} about offsets and buffers",
                    },
                    "timestamp": "2026-01-01T00:00:00Z",
                }
            )
            for i in range(200)
        ]
        path.write_text("\n".join(lines) + "\n")
    return path


def _probe_codex_jsonl(root: Path) -> Path:
    """A small Codex rollout, in the record mix a real one carries.

    Proportions matter more than volume here: ``reasoning`` and the tool
    call/output pair dominate a real rollout, ``message`` records are a
    minority, and the branch order in the parser is what decides how many
    comparisons the common records pay. A file of nothing but user turns
    would score a change to the rarest branch as though it were the hot one.
    """
    path = root / "probe-codex.jsonl"
    if not path.is_file():
        path.parent.mkdir(parents=True, exist_ok=True)
        ts = "2026-01-01T00:00:00Z"

        def response(payload: dict) -> str:
            return json.dumps(
                {"type": "response_item", "timestamp": ts, "payload": payload}
            )

        lines: list[str] = []
        for i in range(40):
            lines.append(response({"type": "reasoning", "encrypted_content": "opaque"}))
            lines.append(
                response(
                    {
                        "type": "function_call",
                        "name": "shell",
                        "arguments": f'{{"cmd": "ls {i}"}}',
                    }
                )
            )
            lines.append(
                response({"type": "function_call_output", "output": f"file {i}\n"})
            )
            lines.append(
                response(
                    {
                        "type": "message",
                        "role": "user",
                        "content": [
                            {
                                "type": "input_text",
                                "text": f"question {i} about offsets",
                            }
                        ],
                    }
                )
            )
            lines.append(
                json.dumps(
                    {
                        "type": "event_msg",
                        "timestamp": ts,
                        "payload": {
                            "type": "user_message",
                            "message": f"question {i} about offsets",
                        },
                    }
                )
            )
            lines.append(
                response(
                    {
                        "type": "message",
                        "role": "assistant",
                        "content": [{"type": "output_text", "text": f"answer {i}"}],
                    }
                )
            )
            lines.append(
                json.dumps(
                    {
                        "type": "event_msg",
                        "timestamp": ts,
                        "payload": {"type": "token_count", "total": i},
                    }
                )
            )
        path.write_text("\n".join(lines) + "\n")
    return path


def loop_probes(root: Path) -> list[LoopProbe]:
    jsonl = _probe_jsonl(root)
    codex_jsonl = _probe_codex_jsonl(root)

    def parse_jsonl() -> None:
        warnings: list[str] = []
        # The per-line decode loop: the single hottest path in the codebase,
        # run once per line of every transcript ever read.
        for _ in transcript._parse_jsonl(jsonl, warnings, transcript._claude_entries):
            pass

    def codex_entries() -> None:
        warnings: list[str] = []
        # The Codex record loop as production composes it: the per-record
        # branch dispatch *and* the dedupe generator every entry passes
        # through. Both are traversed once per record of every Codex
        # transcript read, and the corpus is majority Codex.
        for _ in transcript._dedupe_codex_turns(
            transcript._parse_jsonl(codex_jsonl, warnings, transcript._codex_entries)
        ):
            pass

    def ticking_rows() -> None:
        rows = [(i, "part", "text") for i in range(500)]
        for _ in transcript._ticking_rows(rows, lambda *_a: None):
            pass

    def entry_matches() -> None:
        for _ in transcript.find_entry_matches(_probe_entries(), "offset"):
            pass

    def count_overlapping() -> None:
        text = " ".join(f"turn {i} offset buffer" for i in range(200))
        transcript._count_overlapping(text, "offset")

    def raw_may_match() -> None:
        raw = " ".join(
            f"line {i} about offsets \\u00e9 and buffers" for i in range(200)
        )
        for _ in range(20):
            transcript._raw_text_may_match(raw, ["kookaburra"])

    def codex_db_rows() -> None:
        rows = [
            {
                "id": f"perf-codex-{i:04d}",
                "rollout_path": (
                    "/Users/perf/.codex/sessions/2026/01/05/"
                    f"rollout-2026-01-05T09-00-00-perf-codex-{i:04d}.jsonl"
                ),
                "cwd": f"/Users/perf/project{i % 3}",
                "git_branch": "main",
                "first_user_message": f"start session {i} {_filler(i, 0)}",
                "created_at_ms": 1767603600000,
                "updated_at_ms": 1767603600000 + i,
            }
            for i in range(40)
        ]
        for r in rows:
            discovery._codex_session_from_row(r)

    return [
        LoopProbe(
            "loop.parse_jsonl",
            "The per-line decode loop. Anything added per line is multiplied "
            "by every line of every transcript read -- this is where a "
            "context manager per line (contextlib.suppress over try/except, "
            "measured at 4.85x) or an extra attribute lookup actually costs.",
            parse_jsonl,
        ),
        LoopProbe(
            "loop.ticking_rows",
            "The opencode row loop, which reads the entire part table. "
            "row_tick_wrappers pins that the wrapper exists; this pins what "
            "one row through it costs.",
            ticking_rows,
        ),
        LoopProbe(
            "loop.find_entry_matches",
            "The offsets loop, including its every-1024 cancellation gate. "
            "The gate must stay cheap: it runs on every hit, and a cancelled() "
            "call moved out of its mask would run on every one.",
            entry_matches,
        ),
        LoopProbe(
            "loop.count_overlapping",
            "The match-counting scan. Its body is a str.find and an integer "
            "add; anything else added here runs once per match in the corpus.",
            count_overlapping,
        ),
        LoopProbe(
            "loop.raw_text_may_match",
            "The byte-prefilter's escape decoding, run over raw text before "
            "any parse. It decides whether a file is read at all, so work "
            "added here is paid on sessions that never match.",
            raw_may_match,
        ),
        LoopProbe(
            "loop.codex_entries",
            "The Codex record loop and the dedupe generator behind it. Codex "
            "is the largest provider in a real corpus, so a comparison added "
            "to the branch dispatch is paid on every record of every rollout "
            "read -- including the reasoning and tool records that never "
            "produce a user-visible entry.",
            codex_entries,
        ),
        LoopProbe(
            "loop.codex_db_rows",
            "Per-row Session construction in the Codex DB fast path. One "
            "dataclass per index row per discovery; anything added inside "
            "this loop is paid once per Codex session, per run.",
            codex_db_rows,
        ),
    ]


def measure_loops(root: Path) -> dict[str, int]:
    """Opcode counts for every loop probe.

    Warmed once before counting. First execution of a code path can differ --
    an import resolved, a cache filled, a regex compiled -- and the budget
    should pin the steady state the corpus actually runs in, not the setup
    that happens once per process.
    """
    counts: dict[str, int] = {}
    for probe in loop_probes(root):
        probe.run()
        counts[probe.name] = count_opcodes(probe.run)
    return counts


def python_tag() -> str:
    """The interpreter the opcode counts belong to."""
    return f"{sys.version_info.major}.{sys.version_info.minor}"


# ---------------------------------------------------------------------------
# Measuring and checking
# ---------------------------------------------------------------------------


def rg_available() -> bool:
    return shutil.which("rg") is not None


@contextlib.contextmanager
def _corpus_home(root: Path) -> Iterator[Path]:
    """Point the process at the synthetic corpus for the duration.

    ``discovery`` resolves every provider root through ``Path.home()``, so
    overriding HOME is enough to redirect all three at once -- the same lever
    ``benchmarks/retrieval_compare.py`` pulls with ``--home``.
    """
    home = build_corpus(root)
    saved = {k: os.environ.get(k) for k in ("HOME", "USERPROFILE")}
    os.environ["HOME"] = str(home)
    os.environ["USERPROFILE"] = str(home)
    try:
        yield home
    finally:
        for key, value in saved.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def measure(root: Path) -> dict[str, dict[str, int]]:
    """Run every runnable workload once and return its counters.

    Once, not repeatedly: these are counts, so a second run would return the
    same integers. Workloads needing ripgrep are omitted from the result when
    it is absent rather than recorded as zero, so a machine without it reports
    a smaller set instead of a corpus of phantom regressions.
    """
    measured: dict[str, dict[str, int]] = {}
    have_rg = rg_available()
    with _corpus_home(root) as home:
        for workload in workloads(WorkLedger()):
            if workload.needs_rg and not have_rg:
                continue
            ledger = WorkLedger()
            # Rebuilt per workload so the closure over `ledger` in tui.search
            # reports into the ledger being measured.
            live = next(w for w in workloads(ledger) if w.name == workload.name)
            with instrumented(ledger, home, disable_rg=live.disable_rg):
                live.run()
            measured[workload.name] = ledger.snapshot()
    return measured


@dataclass(frozen=True)
class Breach:
    workload: str
    counter: str
    expected: int
    actual: int


def compare(
    measured: dict[str, dict[str, int]], budgets: dict[str, dict[str, int]]
) -> tuple[list[Breach], list[str]]:
    """Breaches, and the names of workloads with no budget on either side.

    A workload present in one and missing from the other is reported as
    unbudgeted rather than as a breach: adding or removing a scenario is a
    normal edit, and treating it as a regression would train people to bless
    without reading.
    """
    breaches: list[Breach] = []
    unbudgeted: list[str] = []
    for name in sorted(set(measured) | set(budgets)):
        if name not in budgets or name not in measured:
            unbudgeted.append(name)
            continue
        for counter in COUNTERS:
            expected = budgets[name].get(counter)
            actual = measured[name].get(counter, 0)
            if expected is None:
                unbudgeted.append(f"{name}.{counter}")
            elif expected != actual:
                breaches.append(Breach(name, counter, expected, actual))
    return breaches, unbudgeted


def format_report(
    breaches: list[Breach],
    unbudgeted: list[str],
    guards: dict[str, str],
    *,
    skipped: bool,
) -> str:
    """The message a failing budget prints.

    Written to be read by whoever just broke it and does not yet know what
    they broke: the counters that moved, the property the workload defends,
    and the one command that accepts the change on purpose.
    """
    lines: list[str] = []
    if skipped:
        lines.append(
            "NOTE: ripgrep is not installed; workloads covering the "
            "native candidate scan were not measured."
        )
        lines.append("")
    if unbudgeted:
        lines.append("No budget recorded for: " + ", ".join(unbudgeted))
        lines.append("Run `python -m session_browser.perf_budget bless` to record one.")
        lines.append("")
    if not breaches:
        lines.append("All performance budgets met.")
        return "\n".join(lines)

    by_workload: dict[str, list[Breach]] = {}
    for breach in breaches:
        by_workload.setdefault(breach.workload, []).append(breach)
    for name, items in by_workload.items():
        lines.append(f"PERF BUDGET BREACH  {name}")
        lines.append("")
        width = max(len(b.counter) for b in items)
        for b in items:
            delta = b.actual - b.expected
            lines.append(
                f"  {b.counter:<{width}}  {b.expected:>9,}  ->  "
                f"{b.actual:>9,}   ({delta:+,})"
            )
        lines.append("")
        lines.append(f"  Guards: {guards.get(name, '')}")
        lines.append("")
    lines.append(
        "This is a change in how much work the code does, measured "
        "exactly, not a timing wobble."
    )
    lines.append("")
    lines.append(
        "If the change is wanted, weigh it against "
        "benchmarks/retrieval_compare.py first, then run:"
    )
    lines.append("    python -m session_browser.perf_budget bless")
    lines.append(
        "and say why in the commit message -- the new numbers land in "
        "the diff so the trade-off gets reviewed."
    )
    return "\n".join(lines)


def compare_loops(
    measured: dict[str, int], budgets: dict[str, int]
) -> tuple[list[Breach], list[str]]:
    """Breaches and unbudgeted probes, in the same shape as ``compare``.

    A drop fails as loudly as a rise, for the same reason it does above: the
    usual cause is a probe that stopped exercising its loop, not a loop that
    got cheaper. When a fall is genuine it is a win worth blessing on purpose.
    """
    breaches: list[Breach] = []
    unbudgeted: list[str] = []
    for name in sorted(set(measured) | set(budgets)):
        if name not in budgets or name not in measured:
            unbudgeted.append(name)
            continue
        if budgets[name] != measured[name]:
            breaches.append(Breach(name, "opcodes", budgets[name], measured[name]))
    return breaches, unbudgeted


def format_loop_report(
    breaches: list[Breach],
    unbudgeted: list[str],
    guards: dict[str, str],
    *,
    recorded_python: str | None,
) -> str:
    """The message a failing loop probe prints.

    Deliberately different advice from the workload report. A workload breach
    asks whether the extra work pays for itself; a loop breach is per
    iteration, so the question is narrower and the evidence needed is
    different -- ``retrieval_compare.py`` cannot resolve an effect this small,
    which is the entire reason these probes exist.
    """
    lines: list[str] = []
    actual_python = python_tag()
    if recorded_python and recorded_python != actual_python:
        return (
            f"NOTE: loop opcode budgets were recorded on Python "
            f"{recorded_python}; this is {actual_python}. Opcode counts are a "
            f"property of the interpreter, so these were not checked. Re-bless "
            f"on this version to resume enforcing them."
        )
    if unbudgeted:
        lines.append("No loop budget recorded for: " + ", ".join(unbudgeted))
        lines.append("Run `python -m session_browser.perf_budget bless` to record one.")
        lines.append("")
    if not breaches:
        lines.append("All loop opcode budgets met.")
        return "\n".join(lines)

    for b in breaches:
        delta = b.actual - b.expected
        pct = (delta / b.expected * 100) if b.expected else 0.0
        lines.append(f"LOOP OPCODE BREACH  {b.workload}")
        lines.append("")
        lines.append(
            f"  opcodes  {b.expected:>9,}  ->  {b.actual:>9,}   "
            f"({delta:+,}, {pct:+.1f}%)"
        )
        lines.append("")
        lines.append(f"  Guards: {guards.get(b.workload, '')}")
        lines.append("")
    lines.append(
        "This is Python-level work added or removed *inside* a loop "
        "-- the case the workload counters above cannot see, because "
        "the I/O, the parses and the statements are all unchanged."
    )
    lines.append("")
    lines.append(
        "A rise is paid once per iteration. Before accepting one, "
        "time the two forms directly with timeit on a realistic "
        "input; retrieval_compare.py cannot resolve an effect this "
        "small, which is why these probes exist. A fall is usually a "
        "win -- confirm it is not the probe having stopped covering "
        "the loop, then bless it."
    )
    lines.append("")
    lines.append("    python -m session_browser.perf_budget bless")
    return "\n".join(lines)


def load_budgets() -> dict[str, dict[str, int]]:
    if not BUDGETS_PATH.is_file():
        return {}
    data = json.loads(BUDGETS_PATH.read_text())
    return data.get("workloads", {})


def load_loop_budgets() -> tuple[dict[str, int], str | None]:
    """Recorded opcode counts, and the Python version they were taken on."""
    if not BUDGETS_PATH.is_file():
        return {}, None
    data = json.loads(BUDGETS_PATH.read_text())
    return data.get("loop_opcodes", {}), data.get("loop_python")


def loops_enforceable(recorded_python: str | None) -> bool:
    """Whether recorded opcode counts can be checked on this interpreter.

    Counts taken on another Python are not evidence of anything here, so they
    are reported as unenforceable rather than compared and failed. This is the
    one place the module tolerates going quiet, and it says so out loud.
    """
    return recorded_python is None or recorded_python == python_tag()


def _delta(old: int, new: int) -> str:
    """One counter's movement, as it appears in the diff."""
    if old == 0:
        return f"{old} -> {new}"
    return f"{old} -> {new} ({(new - old) / old * 100:+.1f}%)"


def blessed_delta(
    previous: dict[str, dict[str, int]],
    measured: dict[str, dict[str, int]],
    previous_loops: dict[str, int],
    loops: dict[str, int] | None,
) -> dict[str, object]:
    """What this bless changes, derived rather than asserted by whoever ran it.

    Blessing used to end at a modal asking a human to approve the rewrite. That
    put the question to someone mid-run, with no evidence in front of them, and
    it fired on benign blesses too -- so it trained people to approve unread.

    This replaces the question with the evidence. Every counter that moved is
    computed here and written into the budgets file, which means it lands in
    the diff beside the new numbers and is reviewed at commit time with the
    change that caused it. Consent moved; it did not disappear.

    Derived is the load-bearing word: an agent blessing a regression records
    that it did so whether or not it chooses to mention it.
    """
    moved: dict[str, str] = {}
    for name in sorted(set(previous) & set(measured)):
        for counter in COUNTERS:
            old = previous[name].get(counter)
            new = measured[name].get(counter)
            if old is not None and new is not None and old != new:
                moved[f"{name}.{counter}"] = _delta(old, new)

    # Loops are replaced wholesale rather than merged, so a bless that did not
    # re-measure them (loops is None) has not changed them and must not claim
    # movement it never observed.
    if loops is not None:
        for name in sorted(set(previous_loops) & set(loops)):
            if previous_loops[name] != loops[name]:
                moved[f"loop_opcodes.{name}"] = _delta(
                    previous_loops[name], loops[name]
                )

    added = sorted(set(measured) - set(previous))
    if loops is not None:
        added += [f"loop_opcodes.{n}" for n in sorted(set(loops) - set(previous_loops))]

    return {
        # UTC, not local: the budgets file is committed and read on other
        # machines, where a local-time stamp would be a small lie.
        "date": datetime.datetime.now(datetime.UTC).date().isoformat(),
        "python": python_tag(),
        "moved": moved,
        "added": added,
    }


def format_blessed(blessed: dict[str, object]) -> str:
    """The movement a bless just recorded, printed for whoever ran it.

    Says nothing was blessed away when nothing was: a bless over an unchanged
    baseline is a no-op worth naming out loud, because it is what a bless run
    out of reflex looks like.
    """
    moved = blessed.get("moved") or {}
    added = blessed.get("added") or []
    lines: list[str] = []
    if not moved and not added:
        lines.append("No counter moved. The baseline is unchanged.")
    if moved:
        lines.append(f"{len(moved)} counter(s) moved, now recorded in the diff:")
        lines += [f"    {k}  {v}" for k, v in moved.items()]
        lines.append("")
        lines.append(
            "If any of these is a slowdown rather than a new or removed "
            "workload, say so in the commit message: the numbers above are "
            "now the baseline everything after this is measured against."
        )
    if added:
        lines.append(f"{len(added)} newly budgeted: {', '.join(added)}")
    return "\n".join(lines)


def save_budgets(
    measured: dict[str, dict[str, int]],
    guards: dict[str, str],
    loops: dict[str, int] | None = None,
    loop_guards: dict[str, str] | None = None,
) -> dict[str, object]:
    """Write the budgets file, merging over any workloads not measured here.

    Returns the ``_blessed`` block it recorded, so the caller can print the
    same movement it just wrote rather than computing it a second time.

    Merging matters on a machine without ripgrep: blessing there must not
    silently delete the budgets for the workloads it could not run.

    The loop counts are replaced rather than merged, and stamped with the
    interpreter they were taken on. They are all measured together on any
    machine -- there is no partial case to preserve -- and a merge would let
    counts from two Python versions coexist in one file, each looking valid.
    """
    previous = dict(load_budgets())
    existing_loops, existing_python = load_loop_budgets()
    # Computed before the merge below overwrites the numbers it compares against.
    blessed = blessed_delta(previous, measured, existing_loops, loops)
    workloads_out = dict(previous)
    workloads_out.update(measured)
    BUDGETS_PATH.parent.mkdir(parents=True, exist_ok=True)
    BUDGETS_PATH.write_text(
        json.dumps(
            {
                "_comment": (
                    "Exact work counts for session-browser's retrieval hot paths. "
                    "Generated by `python -m session_browser.perf_budget bless`; do "
                    "not hand-edit. Any movement fails the suite on purpose -- see "
                    "session_browser/perf_budget.py for why these are counts and not "
                    "timings. _blessed records what the last bless changed; it is "
                    "derived from the previous numbers, not written by hand, so it "
                    "is the diff's own account of whether a regression was accepted."
                ),
                "_blessed": blessed,
                "guards": guards,
                "workloads": {k: workloads_out[k] for k in sorted(workloads_out)},
                "loop_guards": loop_guards if loops is not None else _loop_guards(),
                "loop_python": python_tag() if loops is not None else existing_python,
                "loop_opcodes": dict(
                    sorted((loops if loops is not None else existing_loops).items())
                ),
            },
            indent=2,
        )
        + "\n"
    )
    return blessed


def _guards() -> dict[str, str]:
    return {w.name: w.guards for w in workloads(WorkLedger())}


def _loop_guards() -> dict[str, str]:
    import tempfile

    with tempfile.TemporaryDirectory(prefix="sb-perf-guards-") as tmp:
        return {p.name: p.guards for p in loop_probes(Path(tmp))}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m session_browser.perf_budget",
        description="Check or re-record the retrieval work budgets.",
    )
    parser.add_argument(
        "action", choices=("check", "bless", "show"), nargs="?", default="check"
    )
    args = parser.parse_args(argv)

    import tempfile

    with tempfile.TemporaryDirectory(prefix="sb-perf-") as tmp:
        measured = measure(Path(tmp))
        loops = measure_loops(Path(tmp))

    if args.action == "show":
        print(json.dumps(measured, indent=2))
        print(
            json.dumps({"loop_opcodes": loops, "loop_python": python_tag()}, indent=2)
        )
        return 0
    if args.action == "bless":
        blessed = save_budgets(measured, _guards(), loops, _loop_guards())
        print(
            f"Recorded {len(measured)} workloads and {len(loops)} loop "
            f"probes (Python {python_tag()}) to "
            f"{BUDGETS_PATH.relative_to(BUDGETS_PATH.parent.parent)}."
        )
        print(format_blessed(blessed))
        return 0

    breaches, unbudgeted = compare(measured, load_budgets())
    print(format_report(breaches, unbudgeted, _guards(), skipped=not rg_available()))
    loop_budgets, loop_python = load_loop_budgets()
    enforceable = loops_enforceable(loop_python)
    loop_breaches, loop_unbudgeted = (
        compare_loops(loops, loop_budgets) if enforceable else ([], [])
    )
    print()
    print(
        format_loop_report(
            loop_breaches, loop_unbudgeted, _loop_guards(), recorded_python=loop_python
        )
    )
    return 1 if (breaches or loop_breaches) else 0


if __name__ == "__main__":
    sys.exit(main())
