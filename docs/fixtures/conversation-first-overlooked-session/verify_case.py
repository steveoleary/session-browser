#!/usr/bin/env python3
"""Replay the portable conversation-first retrieval-ordering snapshot."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

FIXTURE_DIR = Path(__file__).resolve().parent
REPO_ROOT = FIXTURE_DIR.parents[2]
FIXTURE_HOME = FIXTURE_DIR / "home"
EXPECTED = json.loads((FIXTURE_DIR / "expected.json").read_text())


def run_session_browser(*args: str) -> dict:
    env = os.environ.copy()
    env["HOME"] = str(FIXTURE_HOME)
    env.pop("CLAUDE_CODE_SESSION_ID", None)
    env.pop("CODEX_THREAD_ID", None)
    result = subprocess.run(
        [sys.executable, "-m", "session_browser.app", *args],
        cwd=REPO_ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )
    if result.returncode:
        sys.stderr.write(result.stderr)
        raise SystemExit(
            f"session-browser exited {result.returncode}: {' '.join(args)}"
        )
    try:
        return json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        raise SystemExit(
            f"session-browser did not return JSON:\n{result.stdout}"
        ) from exc


def search(*extra: str) -> dict:
    return run_session_browser(
        "search",
        *EXPECTED["query"],
        "--provider",
        "claude",
        "--mode",
        "snippets",
        "--context",
        "100",
        "--limit",
        "10",
        "--format",
        "json",
        *extra,
    )


def check_baseline() -> None:
    payload = search()
    results = payload["results"]
    ids = [item["id"] for item in results]
    wanted = EXPECTED["baseline"]["ordered_ids"]
    assert ids == wanted, f"baseline order changed: {ids!r} != {wanted!r}"

    by_id = {item["id"]: item for item in results}
    false_lead = EXPECTED["source_incident"]["false_lead_session"]
    desired = EXPECTED["source_incident"]["desired_session"]
    false_roles = {snippet["role"] for snippet in by_id[false_lead]["snippets"]}
    desired_roles = {snippet["role"] for snippet in by_id[desired]["snippets"]}
    assert false_roles == set(EXPECTED["baseline"]["false_lead_roles"]), (
        f"false-lead evidence changed: {sorted(false_roles)!r}"
    )
    required = set(EXPECTED["baseline"]["desired_required_roles"])
    assert required <= desired_roles, (
        f"desired evidence lacks roles: {sorted(required - desired_roles)!r}"
    )
    print("PASS: reproduced retrieval ordering omission")


def check_candidate() -> None:
    payload = search("--sort", EXPECTED["candidate"]["sort"])
    results = payload["results"]
    ids = [item["id"] for item in results]
    wanted = EXPECTED["candidate"]["ordered_ids"]
    assert ids == wanted, f"candidate order is {ids!r}, expected {wanted!r}"

    evidence = EXPECTED["candidate"]["evidence"]
    for item in results:
        expected = evidence[item["id"]]
        actual = item.get("evidence")
        assert actual == expected, (
            f"{item['id']} evidence is {actual!r}, expected {expected!r}"
        )
    print("PASS: conversation-first retrieval candidate fixes the ordering")


def main() -> None:
    modes = {"baseline": check_baseline, "candidate": check_candidate}
    if len(sys.argv) != 2:
        raise SystemExit(f"usage: {Path(sys.argv[0]).name} {{baseline,candidate}}")
    try:
        check = modes[sys.argv[1]]
    except KeyError:
        raise SystemExit(
            f"unknown mode {sys.argv[1]!r}; choose baseline or candidate"
        ) from None
    check()


if __name__ == "__main__":
    main()
