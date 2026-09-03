#!/usr/bin/env python3
"""Check that this repo's documented retrieval workflow still answers.

Two ad-hoc runs on 2026-08-14 handed a fresh agent a research brief against the
real corpus and found more defects than any other check here — a `--repo`
wording error that inflated a count by 35.8%, a `stats`/`list` disagreement,
unstated `--limit` behaviour, help naming the wrong provider as the risky one.
Nothing else catches that class, because it is documentation failing against
its only real consumer.

The consumer is a model, so most of that run cannot be asserted on: this
repository does not gate on stochastic checks, for the same reason its
performance gate counts instead of timing. What *can* be asserted is the half
underneath — that the commands the skill teaches, run literally against a
frozen corpus, still produce the answers the brief expects, and that the traps
they exist to escape are still traps.

So each task below is checked twice. The obvious command must still give the
misleading answer, and the documented one must still give the right answer. A
task where both agree has stopped testing anything: either the CLI changed and
the skill's warning is now stale, or the fixture drifted. Either way it needs a
person, which is why that is an assertion and not a tolerance.

The live half stays in ``brief.md``, to be run by hand and read rather than
scored.
"""

from __future__ import annotations

import collections
import json
import os
import subprocess
import sys
from pathlib import Path

FIXTURE_DIR = Path(__file__).resolve().parent
REPO_ROOT = FIXTURE_DIR.parents[2]
FIXTURE_HOME = FIXTURE_DIR / "home"
TASKS = json.loads((FIXTURE_DIR / "expected.json").read_text())["tasks"]


def sb(*args: str) -> dict:
    """One session-browser command against the fixture's private HOME."""
    env = os.environ.copy()
    env["HOME"] = str(FIXTURE_HOME)
    # A live session id would make the caller's own session part of the
    # corpus, which is the observer effect this fixture exists to be free of.
    env.pop("CLAUDE_CODE_SESSION_ID", None)
    env.pop("CODEX_THREAD_ID", None)
    result = subprocess.run(
        [sys.executable, "-m", "session_browser.app", *args, "--format", "json"],
        cwd=REPO_ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )
    if result.returncode:
        sys.stderr.write(result.stderr)
        raise SystemExit(f"session-browser exited {result.returncode}: {args}")
    try:
        return json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        raise SystemExit(f"not JSON:\n{result.stdout}") from exc


def check_substring_filter() -> None:
    """`--repo` is a substring, so a count off it is an upper bound."""
    task = TASKS["coffee_run_sessions"]
    naive = sb(*task["naive_command"])
    assert naive["counts"]["returned"] == task["naive_answer"], (
        f"--repo {task['cwd']} returned {naive['counts']['returned']}, and the "
        f"brief expects the substring filter to over-count to "
        f"{task['naive_answer']}"
    )
    shadowed = [s for s in naive["sessions"] if s["cwd"] == task["shadow_cwd"]]
    assert shadowed, (
        "the substring filter no longer picks up the shadow project, so this "
        "task tests nothing"
    )
    # The documented escape: group the rows by their own cwd rather than
    # trusting the filter that produced them.
    exact = [s for s in naive["sessions"] if s["cwd"] == task["cwd"]]
    assert len(exact) == task["answer"], (
        f"{len(exact)} sessions have cwd {task['cwd']}, brief says {task['answer']}"
    )
    readable = [s for s in exact if s["total_entries"]]
    assert len(readable) == task["readable_answer"], (
        f"{len(readable)} of those are readable, brief says {task['readable_answer']}"
    )
    # And the receipt the skill points at agrees with the grouping.
    top = {row["cwd"]: row["count"] for row in sb("stats")["top_cwds"]}
    assert top[task["cwd"]] == task["answer"], (
        f"stats top_cwds says {top[task['cwd']]} for {task['cwd']}, "
        f"list grouping says {task['answer']}"
    )
    print("PASS: --repo is a substring, and top_cwds decomposes it")


def check_discovered_versus_readable() -> None:
    """`stats` never opens a transcript, so its total cannot see a bad file."""
    task = TASKS["readable_total"]
    stats = sb("stats")
    assert stats["total"] == task["stats_total"], (
        f"stats total {stats['total']}, brief says {task['stats_total']}"
    )
    assert stats["transcript_health"] == "not_checked", (
        "stats now claims to know transcript health; the brief's task assumes "
        "it does not"
    )
    listing = sb("list")
    counts = listing["counts"]
    assert counts["readable"] == task["readable"], (
        f"counts.readable {counts['readable']}, brief says {task['readable']}"
    )
    assert counts["empty"] == task["empty"], (
        f"counts.empty {counts['empty']}, brief says {task['empty']}"
    )
    assert stats["total"] != counts["readable"], (
        "stats and list agree, so this task no longer distinguishes what was "
        "discovered from what can be read"
    )
    # The bad session is still returned, not filtered out — the warning is the
    # only thing that says so.
    assert any(s["id"] == task["warned_session"] for s in listing["sessions"]), (
        "the unreadable session is no longer returned; the brief's answer "
        "depends on it being present and merely warned about"
    )
    assert any(task["warned_session"] in w for w in listing["warnings"]), (
        "no warning names the unreadable session"
    )
    print("PASS: stats counts what it discovered, list counts what it read")


def check_role_provenance() -> None:
    """A hit is as often a quotation as a statement."""
    task = TASKS["loyalty_tier_provenance"]
    ids = sb("search", task["phrase"], "--mode", "ids")["results"]
    assert len(ids) == 1 and ids[0]["id"] == task["session"], (
        f"expected one id hit in {task['session']}, got {[r['id'] for r in ids]}"
    )
    assert ids[0]["match_count"] == task["match_count"], (
        f"match_count {ids[0]['match_count']}, brief says {task['match_count']}"
    )
    assert "snippets" not in ids[0], (
        "ids mode now carries snippets; the task's point is that ids alone "
        "cannot tell a quotation from a statement"
    )
    # --max-snippets 0 lifts the per-result cap, so the role tally is complete
    # rather than truncated at the default 20.
    snippets = sb(
        "search", task["phrase"], "--mode", "snippets", "--max-snippets", "0"
    )["results"][0]["snippets"]
    roles = collections.Counter(s["role"] for s in snippets)
    assert roles == collections.Counter(task["roles"]), (
        f"role tally {dict(roles)}, brief says {task['roles']}"
    )
    assert roles["tool"] > roles[task["proposed_by_role"]], (
        "most occurrences are no longer material the agent read, so the task "
        "no longer distinguishes reading from saying"
    )
    print("PASS: roles separate what was read from what was proposed")


def check_self_correction() -> None:
    """A confident mid-session claim is often reversed a few entries later."""
    task = TASKS["espresso_decision"]
    hits = sb(
        "search", "espresso machine", task["superseded_answer"], "--mode", "snippets"
    )["results"]
    assert len(hits) == 1 and hits[0]["id"] == task["session"], (
        f"expected one hit in {task['session']}, got {[r['id'] for r in hits]}"
    )
    said = [s for s in hits[0]["snippets"] if s["role"] == "assistant"]
    assert task["superseded_answer"] in said[0]["text"], (
        "the first assistant snippet no longer states the superseded answer, "
        "so quoting a mid-session snippet is no longer a way to be wrong here"
    )
    assert task["answer"] not in said[0]["text"], (
        "the first snippet already carries the final answer; the task needs "
        "the reversal to be invisible from the middle"
    )
    ending = sb("get", task["session"], "--role", "assistant", "--tail", "1")
    last = ending["entries"][0]["text"]
    assert task["answer"] in last, (
        f"the session's last assistant turn does not name {task['answer']}: "
        f"{last[:120]!r}"
    )
    assert task["superseded_answer"] not in last, (
        "the ending still names the superseded answer"
    )
    print("PASS: the ending carries the decision, the middle does not")


def check(state: str) -> None:
    check_substring_filter()
    check_discovered_versus_readable()
    check_role_provenance()
    check_self_correction()
    print(f"PASS: fresh-agent skill brief [{state}]")


def main() -> None:
    if len(sys.argv) != 2 or sys.argv[1] not in {"baseline", "candidate"}:
        raise SystemExit(f"usage: {Path(sys.argv[0]).name} {{baseline,candidate}}")
    check(sys.argv[1])


if __name__ == "__main__":
    main()
