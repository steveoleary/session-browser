"""A scalable, shaped synthetic HOME for the retrieval comparator.

``benchmarks/retrieval_compare.py`` takes a ``--home`` the caller prepares, and
until this module there was nothing to prepare one with: the perf gate's
``session_browser.perf_budget.build_corpus`` is fixed by module constants and
has no CLI, and ``session_browser.demo`` is screenshot-sized. So every
comparator run pointed at a real ``$HOME``, whose sample spread sits on top of
the comparator's 5% limit and therefore returns ``unresolvable``.

**This is deliberately not ``build_corpus`` with parameters.** That corpus is
the *gate's*, and its whole value is being byte-identical for everyone, because
the budgets in ``docs/perf_budgets.json`` are exact integers. A generalisation
bug there would not fail loudly; it would shift counters, and ``bless`` would
then record corpus drift in the ``_blessed`` block as faithfully as it records a
real parser change — which costs the gate the one property that makes its diffs
readable. Two corpora, two purposes:

``build_corpus``   the gate. Fixed, uniform, exact. Never scaled.
this module        the comparator. Scalable, shaped, throwaway. Never consulted
                   by ``test_performance.py``.

Scale is not cosmetic. Most of a comparator sample against a x1 corpus is
interpreter startup and imports, and that fixed cost *dilutes* a
regression before the comparator can see it — a 10% slowdown in search presents
as roughly 6% of the sample, against a 5% limit. Growing the corpus shrinks the
startup share and so restores the tool's sensitivity; cost grows sub-linearly
because startup is paid once per sample either way.

Shape matters for the same reason uniformity is fine for the gate and wrong
here. The gate *counts* work, so a session-length long tail makes no counter
catch anything. The comparator *times* work, and time is dominated by the few
largest sessions, by which provider record vocabulary the parser has to walk,
and by how much of the corpus a query actually opens. So this corpus carries:

* a long tail of session lengths (Pareto, mean about 21 turns, tail to 600),
* a per-provider record mix — Claude tool_use/tool_result blocks and plain
  string content beside the usual list content; Codex rollouts in all three
  eras, legacy ``agent_message``, paginated ``item_completed`` and
  response-item-only; OpenCode tool parts beside text parts; Pi messages split
  across multiple content parts,
* query terms planted at *known* selectivity, from absent through one session in
  a hundred up to every session, so a comparator run can choose how much of the
  corpus it makes the parser read.

Every corpus writes a manifest recording exactly what was built, including the
realised session count for each term, so a run's queries can be chosen from
facts rather than from hope.

A synthetic HOME also contains no live session, so the ``--current-session-env``
exclusion a real-``$HOME`` run needs does not apply to one built here.
"""

from __future__ import annotations

import argparse
import json
import random
import sqlite3
import sys
from dataclasses import dataclass
from pathlib import Path

CLAUDE_PROJECTS = ".claude/projects"
CODEX_SESSIONS = ".codex/sessions"
CODEX_STATE_DB = ".codex/state_5.sqlite"
OPENCODE_DB = ".local/share/opencode/opencode.db"
PI_SESSIONS = ".pi/agent/sessions"

MANIFEST_NAME = "corpus-manifest.json"

# One "scale unit" reproduces the *proportions* of the gate corpus — not its
# contents — so x1 is a familiar size to compare a measurement against. These
# track ``perf_budget``'s constants by intent, not by import: the gate's numbers
# must be free to move for the gate's own reasons without silently rescaling
# every past comparator measurement taken here.
BASE_CLAUDE = 180
BASE_CODEX = 40
BASE_OPENCODE = 40
BASE_PI = 40
BASE_TOTAL = BASE_CLAUDE + BASE_CODEX + BASE_OPENCODE + BASE_PI

DEFAULT_SEED = 20260824

# Session length is drawn from a bounded Pareto: mean MIN_TURNS*a/(a-1) ~= 21
# turns at these constants, median ~12, and a thin tail up to MAX_TURNS. The
# mean is deliberately close to the gate corpus's flat 20 so a x1 corpus is
# comparable in volume while being nothing like it in shape.
MIN_TURNS = 8
MAX_TURNS = 600
TAIL_ALPHA = 1.6

# Words a planted query term can never collide with, so selectivity is exactly
# what the manifest says. ASCII on purpose: text-mode and binary reads only
# agree on a byte count while the corpus stays ASCII.
_WORDS = (
    "retries",
    "offsets",
    "buffers",
    "ordering",
    "parsed",
    "output",
    "cursor",
    "window",
    "rollout",
    "index",
    "digest",
    "checkpoint",
    "queue",
    "batch",
    "stream",
    "decoder",
    "manifest",
    "segment",
)
_FILLER_WORDS = 12


class CorpusError(RuntimeError):
    """The requested corpus cannot be built."""


@dataclass(frozen=True)
class QueryTerm:
    """A search term planted at a known fraction of sessions.

    ``selectivity`` is the fraction of *sessions* that carry the term, not the
    fraction of turns: search reports one result per session, so the session
    count is what decides how much of the corpus a query opens.
    """

    term: str
    selectivity: float

    @property
    def period(self) -> int:
        """Plant in one session out of this many; 0 means never."""
        if self.selectivity <= 0.0:
            return 0
        return max(1, round(1.0 / self.selectivity))


# Spread over three orders of magnitude on purpose. A comparator run picks the
# selectivity it wants: the absent term measures discovery and the prefilter
# with no parsing at all, and the every-session term is the upper bound where
# the parser reads the whole corpus.
TERMS = (
    QueryTerm("zzqabsentzz", 0.0),
    QueryTerm("kookaburra", 0.01),
    QueryTerm("numbat", 0.10),
    QueryTerm("quokka", 0.50),
    QueryTerm("bilby", 1.0),
)

# Twenty slots rather than a random draw, so the era mix is an exact reportable
# fact: half the rollouts legacy, seven paginated, three response-item-only.
_CODEX_ERAS = ("legacy",) * 10 + ("paginated",) * 7 + ("response_item",) * 3


def _project_count(scale: int) -> int:
    """More projects as the corpus grows, but not without bound.

    Claude and Pi both key their directory layout on the working directory, so
    a single project would collapse the whole corpus into one directory listing
    and hide the cost of walking many.
    """
    return min(32, 3 * scale)


def _filler(rng: random.Random) -> str:
    return " ".join(rng.choice(_WORDS) for _ in range(_FILLER_WORDS))


def _turn_count(rng: random.Random) -> int:
    """Draw one session length from the bounded Pareto tail."""
    draw = 1.0 - rng.random()  # (0, 1], so the power is always finite
    return min(MAX_TURNS, int(MIN_TURNS * draw ** (-1.0 / TAIL_ALPHA)))


def planted_terms(index: int) -> list[str]:
    """Which terms session *index* carries, by position rather than by chance.

    Deterministic placement is what makes selectivity a *known* quantity: the
    manifest can state the exact session count for each term without counting
    the corpus back afterwards.
    """
    return [term.term for term in TERMS if term.period and index % term.period == 0]


def planted_session_count(term: QueryTerm, sessions: int) -> int:
    if not term.period:
        return 0
    return (sessions + term.period - 1) // term.period


def _turns_for(rng: random.Random, turns: int, planted: list[str]) -> list[tuple]:
    """Build one session's turns, then salt the planted terms into them."""
    rows = [
        ["user" if position % 2 == 0 else "assistant", _filler(rng)]
        for position in range(turns)
    ]
    for order, term in enumerate(planted):
        slot = 1 + (2 * order) % max(1, turns - 1)
        rows[slot][1] = f"{rows[slot][1]} {term} sighted"
    return [(role, text) for role, text in rows]


def _stamp(day: int, position: int) -> str:
    hour = 9 + position // 60
    return f"2026-01-{day:02d}T{hour:02d}:{position % 60:02d}:00.000Z"


# ---------------------------------------------------------------------------
# Provider writers
# ---------------------------------------------------------------------------


def _write_claude(home: Path, *, index: int, project: int, turns: list[tuple]) -> Path:
    root = home / CLAUDE_PROJECTS / f"-Users-bench-project{project}"
    root.mkdir(parents=True, exist_ok=True)
    day = (index % 28) + 1
    cwd = f"/Users/bench/project{project}"
    # Two shape switches, both by position so the mix is reportable: one
    # session in four carries tool_use/tool_result pairs, and one in seven
    # writes its user turns as a plain string instead of a content list.
    with_tools = index % 4 == 0
    string_content = index % 7 == 0

    lines: list[str] = []
    for position, (role, text) in enumerate(turns):
        stamp = _stamp(day, position)
        if role == "user":
            content = text if string_content else [{"type": "text", "text": text}]
            record = {
                "type": "user",
                "timestamp": stamp,
                "message": {"role": "user", "content": content},
            }
            if position == 0:
                record["cwd"] = cwd
                record["gitBranch"] = "main"
            lines.append(json.dumps(record))
            continue
        lines.append(
            json.dumps(
                {
                    "type": "assistant",
                    "timestamp": stamp,
                    "message": {
                        "role": "assistant",
                        "content": [{"type": "text", "text": text}],
                    },
                }
            )
        )
        if with_tools and position % 5 == 3:
            lines.append(
                json.dumps(
                    {
                        "type": "assistant",
                        "timestamp": stamp,
                        "message": {
                            "role": "assistant",
                            "content": [
                                {
                                    "type": "tool_use",
                                    "name": "Read",
                                    "input": {"file_path": f"{cwd}/segment.py"},
                                }
                            ],
                        },
                    }
                )
            )
            lines.append(
                json.dumps(
                    {
                        "type": "user",
                        "timestamp": stamp,
                        "message": {
                            "role": "user",
                            "content": [
                                {
                                    "type": "tool_result",
                                    "content": f"read {len(text)} bytes of segment",
                                }
                            ],
                        },
                    }
                )
            )
    path = root / f"bench-claude-{index:05d}.jsonl"
    path.write_text("\n".join(lines) + "\n")
    return path


def _codex_lines(era: str, turns: list[tuple], day: int) -> list[str]:
    """One rollout's records in the era's own vocabulary.

    All three eras are represented because the parser walks a different branch
    for each, and a corpus written only in the legacy vocabulary times only that
    branch. This is the comparator's corpus, so the mix costs nothing anyone
    else has to re-bless.
    """
    lines: list[str] = []
    for position, (role, text) in enumerate(turns):
        stamp = _stamp(day, position)
        if role == "user":
            if era == "legacy":
                lines.append(
                    json.dumps(
                        {
                            "type": "event_msg",
                            "timestamp": stamp,
                            "payload": {"type": "user_message", "message": text},
                        }
                    )
                )
            elif era == "paginated":
                lines.append(
                    json.dumps(
                        {
                            "type": "event_msg",
                            "timestamp": stamp,
                            "payload": {
                                "type": "item_completed",
                                "item": {
                                    "type": "UserMessage",
                                    "content": [{"type": "text", "text": text}],
                                },
                            },
                        }
                    )
                )
            else:
                lines.append(
                    json.dumps(
                        {
                            "type": "response_item",
                            "timestamp": stamp,
                            "payload": {
                                "type": "message",
                                "role": "user",
                                "content": [{"type": "input_text", "text": text}],
                            },
                        }
                    )
                )
            continue
        if era == "legacy":
            lines.append(
                json.dumps(
                    {
                        "type": "event_msg",
                        "timestamp": stamp,
                        "payload": {"type": "agent_message", "message": text},
                    }
                )
            )
        else:
            lines.append(
                json.dumps(
                    {
                        "type": "response_item",
                        "timestamp": stamp,
                        "payload": {
                            "type": "message",
                            "role": "assistant",
                            "content": [{"type": "output_text", "text": text}],
                        },
                    }
                )
            )
    return lines


def _write_codex(
    home: Path, *, index: int, era: str, project: int, turns: list[tuple]
) -> tuple[str, Path, str]:
    day = (index % 28) + 1
    root = home / CODEX_SESSIONS / "2026" / "01" / f"{day:02d}"
    root.mkdir(parents=True, exist_ok=True)
    sid = f"bench-codex-{index:05d}"
    cwd = f"/Users/bench/project{project}"
    lines = [
        json.dumps(
            {
                "type": "session_meta",
                "timestamp": _stamp(day, 0),
                "payload": {
                    "id": sid,
                    "cwd": cwd,
                    "git": {"branch": "main"},
                    "timestamp": _stamp(day, 0),
                },
            }
        )
    ]
    lines.extend(_codex_lines(era, turns, day))
    path = root / f"rollout-2026-01-{day:02d}T09-00-00-{sid}.jsonl"
    path.write_text("\n".join(lines) + "\n")
    return sid, path, cwd


def _write_codex_state(home: Path, rows: list[tuple[str, Path, str, str]]) -> Path:
    """Write the Codex thread index the real client keeps beside its rollouts."""
    db_path = home / CODEX_STATE_DB
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path))
    try:
        conn.execute(
            "CREATE TABLE threads (id TEXT PRIMARY KEY, "
            "rollout_path TEXT NOT NULL, cwd TEXT NOT NULL, "
            "title TEXT NOT NULL DEFAULT '', git_branch TEXT, "
            "git_origin_url TEXT, first_user_message TEXT NOT NULL DEFAULT '', "
            "created_at_ms INTEGER, updated_at_ms INTEGER, "
            "archived INTEGER NOT NULL DEFAULT 0)"
        )
        for sid, path, cwd, first in rows:
            conn.execute(
                "INSERT INTO threads (id, rollout_path, cwd, git_branch, "
                "git_origin_url, first_user_message, created_at_ms, "
                "updated_at_ms, archived) VALUES (?, ?, ?, ?, ?, ?, ?, ?, 0)",
                (
                    sid,
                    str(path),
                    cwd,
                    "main",
                    "https://example.test/bench.git",
                    first,
                    1767603600000,
                    1767603600000,
                ),
            )
        conn.commit()
    finally:
        conn.close()
    return db_path


def _write_pi(home: Path, *, index: int, project: int, turns: list[tuple]) -> Path:
    root = home / PI_SESSIONS / f"--Users-bench-project{project}--"
    root.mkdir(parents=True, exist_ok=True)
    sid = f"bench-pi-{index:05d}"
    day = (index % 28) + 1
    stamp_day = f"2026-01-{day:02d}"
    # One session in six splits assistant prose across two content parts, the
    # shape that makes the parser join before it matches.
    split_parts = index % 6 == 0
    lines = [
        json.dumps(
            {
                "type": "session",
                "version": 3,
                "id": sid,
                "timestamp": _stamp(day, 0),
                "cwd": f"/Users/bench/project{project}",
            }
        )
    ]
    for position, (role, text) in enumerate(turns):
        if split_parts and role == "assistant":
            head, _, tail = text.partition(" ")
            content = [
                {"type": "text", "text": head},
                {"type": "text", "text": f" {tail}"},
            ]
        else:
            content = [{"type": "text", "text": text}]
        lines.append(
            json.dumps(
                {
                    "type": "message",
                    "id": f"{sid}-m{position:04d}",
                    "parentId": None if position == 0 else f"{sid}-m{position - 1:04d}",
                    "timestamp": _stamp(day, position),
                    "message": {"role": role, "content": content},
                }
            )
        )
    path = root / f"{stamp_day}T09-00-00-000Z_{sid}.jsonl"
    path.write_text("\n".join(lines) + "\n")
    return path


_OPENCODE_SCHEMA = (
    (
        "CREATE TABLE project (id TEXT PRIMARY KEY, worktree TEXT NOT NULL, "
        "name TEXT, time_created INTEGER NOT NULL, "
        "time_updated INTEGER NOT NULL, sandboxes TEXT NOT NULL DEFAULT '[]')"
    ),
    (
        "CREATE TABLE session (id TEXT PRIMARY KEY, project_id TEXT NOT NULL, "
        "slug TEXT NOT NULL DEFAULT '', directory TEXT NOT NULL, "
        "title TEXT NOT NULL DEFAULT '', version TEXT NOT NULL DEFAULT '1', "
        "time_created INTEGER NOT NULL, time_updated INTEGER NOT NULL)"
    ),
    (
        "CREATE TABLE message (id TEXT PRIMARY KEY, session_id TEXT NOT NULL, "
        "time_created INTEGER NOT NULL, time_updated INTEGER NOT NULL, "
        "data TEXT NOT NULL)"
    ),
    (
        "CREATE TABLE part (id TEXT PRIMARY KEY, message_id TEXT NOT NULL, "
        "session_id TEXT NOT NULL, time_created INTEGER NOT NULL, "
        "time_updated INTEGER NOT NULL, data TEXT NOT NULL)"
    ),
)


def _write_opencode(home: Path, sessions: list[tuple[int, int, list[tuple]]]) -> Path:
    db_path = home / OPENCODE_DB
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path))
    try:
        for ddl in _OPENCODE_SCHEMA:
            conn.execute(ddl)
        projects: set[int] = set()
        for index, project, turns in sessions:
            if project not in projects:
                projects.add(project)
                conn.execute(
                    "INSERT INTO project VALUES (?, ?, ?, ?, ?, ?)",
                    (
                        f"bench-proj-{project}",
                        f"/Users/bench/project{project}",
                        f"project{project}",
                        1700000000000,
                        1700000000000,
                        "[]",
                    ),
                )
            sid = f"bench-oc-{index:05d}"
            conn.execute(
                "INSERT INTO session (id, project_id, slug, directory, title, "
                "version, time_created, time_updated) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    sid,
                    f"bench-proj-{project}",
                    f"s{index}",
                    f"/Users/bench/project{project}",
                    turns[0][1][:40],
                    "1",
                    1700000000000,
                    1700000000000 + index,
                ),
            )
            for position, (role, text) in enumerate(turns):
                mid = f"{sid}-m{position:04d}"
                stamp = 1700000000000 + position
                conn.execute(
                    "INSERT INTO message VALUES (?, ?, ?, ?, ?)",
                    (mid, sid, stamp, stamp, json.dumps({"role": role})),
                )
                # One part in five is a tool part rather than prose, so the
                # part-shape branch is exercised rather than assumed.
                data = (
                    {"type": "tool", "tool": "bash", "state": {"output": text}}
                    if position % 5 == 4
                    else {"type": "text", "text": text}
                )
                conn.execute(
                    "INSERT INTO part VALUES (?, ?, ?, ?, ?, ?)",
                    (f"{mid}-p0", mid, sid, stamp, stamp, json.dumps(data)),
                )
        conn.commit()
    finally:
        conn.close()
    return db_path


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def _tree_bytes(home: Path) -> int:
    return sum(path.stat().st_size for path in home.rglob("*") if path.is_file())


def generate(
    home: Path,
    *,
    scale: int = 1,
    seed: int = DEFAULT_SEED,
    force: bool = False,
) -> dict:
    """Write a corpus under *home* and return its manifest.

    *scale* multiplies each provider's session count; x1 is ``BASE_TOTAL``
    sessions in the gate corpus's proportions. *seed* fixes every length and
    every filler word, so the same arguments reproduce the same corpus — a
    comparator run that has to be repeated later measures the same thing.

    Refuses a non-empty directory unless *force*, because this module's callers
    hand it temporary paths and a mistyped one must not eat a real home.
    """
    if scale < 1:
        raise CorpusError(f"--scale must be >= 1; got {scale}")
    home = Path(home)
    home.mkdir(parents=True, exist_ok=True)
    if any(home.iterdir()) and not force:
        raise CorpusError(
            f"{home} is not empty; pass --force (force=True) to write into it "
            "anyway. Point --home at a directory of its own: this corpus is "
            "throwaway and deleting the wrong tree is the one expensive mistake "
            "here."
        )

    rng = random.Random(seed)
    projects = _project_count(scale)
    counts = {
        "claude": BASE_CLAUDE * scale,
        "codex": BASE_CODEX * scale,
        "opencode": BASE_OPENCODE * scale,
        "pi": BASE_PI * scale,
    }
    total = sum(counts.values())

    # One running index across every provider, so a term's selectivity is a
    # property of the whole corpus rather than of one provider's slice.
    index = 0
    turn_total = 0
    longest = 0
    era_counts = {"legacy": 0, "paginated": 0, "response_item": 0}

    def next_turns() -> list[tuple]:
        nonlocal turn_total, longest
        length = _turn_count(rng)
        turn_total += length
        longest = max(longest, length)
        return _turns_for(rng, length, planted_terms(index))

    for local in range(counts["claude"]):
        _write_claude(home, index=index, project=local % projects, turns=next_turns())
        index += 1

    codex_rows: list[tuple[str, Path, str, str]] = []
    for local in range(counts["codex"]):
        era = _CODEX_ERAS[local % len(_CODEX_ERAS)]
        era_counts[era] += 1
        turns = next_turns()
        sid, path, cwd = _write_codex(
            home, index=index, era=era, project=local % projects, turns=turns
        )
        codex_rows.append((sid, path, cwd, turns[0][1][:80]))
        index += 1
    _write_codex_state(home, codex_rows)

    opencode_sessions: list[tuple[int, int, list[tuple]]] = []
    for local in range(counts["opencode"]):
        opencode_sessions.append((index, local % projects, next_turns()))
        index += 1
    _write_opencode(home, opencode_sessions)

    for local in range(counts["pi"]):
        _write_pi(home, index=index, project=local % projects, turns=next_turns())
        index += 1

    manifest = {
        "scale": scale,
        "seed": seed,
        "home": str(home),
        "sessions": total,
        "sessions_by_provider": counts,
        "projects": projects,
        "turns": {
            "total": turn_total,
            "mean": round(turn_total / total, 2),
            "longest": longest,
            "distribution": "bounded pareto",
            "min": MIN_TURNS,
            "max": MAX_TURNS,
            "alpha": TAIL_ALPHA,
        },
        "codex_eras": era_counts,
        "queries": [
            {
                "term": term.term,
                "selectivity": term.selectivity,
                "sessions": planted_session_count(term, total),
            }
            for term in TERMS
        ],
        "bytes": _tree_bytes(home),
    }
    (home / MANIFEST_NAME).write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build a scalable synthetic provider HOME for "
            "benchmarks/retrieval_compare.py."
        ),
        epilog=(
            "Prints the corpus manifest as JSON on stdout, and writes the same "
            f"manifest to <home>/{MANIFEST_NAME}. The manifest's `queries` list "
            "names each planted term with the exact number of sessions carrying "
            "it, so a comparator run can pick a query by how much of the corpus "
            "it opens rather than by guesswork. This corpus is for the "
            "comparator only; the perf gate builds its own and is not affected "
            "by anything here."
        ),
    )
    parser.add_argument(
        "--home",
        required=True,
        type=Path,
        help="directory to write the corpus into; created if absent",
    )
    parser.add_argument(
        "--scale",
        type=int,
        default=1,
        metavar="N",
        help=(
            f"multiply every provider's session count (default 1 = "
            f"{BASE_TOTAL} sessions). Larger corpora shrink the share of a "
            "comparator sample spent on interpreter startup, which is what "
            "limits the tool's sensitivity to a small regression."
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        metavar="N",
        help=(
            f"fix the drawn session lengths and filler text (default "
            f"{DEFAULT_SEED}); the same seed and scale reproduce the same bytes"
        ),
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="write into a non-empty directory (refused by default)",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        manifest = generate(
            args.home, scale=args.scale, seed=args.seed, force=args.force
        )
    except CorpusError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
