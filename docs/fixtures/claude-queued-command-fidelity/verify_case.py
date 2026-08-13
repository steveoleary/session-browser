#!/usr/bin/env python3
"""Replay the portable Claude queued-command fidelity snapshot."""

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


def user_entries() -> list[str]:
    env = os.environ.copy()
    env["HOME"] = str(FIXTURE_HOME)
    env.pop("CLAUDE_CODE_SESSION_ID", None)
    env.pop("CODEX_THREAD_ID", None)
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "session_browser.app",
            "get",
            EXPECTED["session"],
            "--role",
            "user",
            "--format",
            "json",
        ],
        cwd=REPO_ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )
    if result.returncode:
        sys.stderr.write(result.stderr)
        raise SystemExit(f"session-browser exited {result.returncode}")
    try:
        payload = json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        raise SystemExit(
            f"session-browser did not return JSON:\n{result.stdout}"
        ) from exc
    return [entry["text"] for entry in payload["entries"]]


def check(state: str) -> None:
    queued = EXPECTED["queued_prompt"]
    wanted = EXPECTED[state]["expected_occurrences_in_user_entries"]
    entries = user_entries()
    occurrences = sum(text == queued for text in entries)
    assert occurrences == wanted, (
        f"queued prompt appeared {occurrences} time(s), expected {wanted}"
    )
    # The machine task-notification shares the queued_command record type but
    # is not human speech. It must never surface as a user turn in any state.
    marker = EXPECTED["machine_notification_marker"]
    fabricated = [text for text in entries if marker in text]
    assert not fabricated, (
        f"machine notification surfaced as {len(fabricated)} user turn(s)"
    )
    print(f"PASS: queued-command {state}")


def main() -> None:
    if len(sys.argv) != 2 or sys.argv[1] not in {"baseline", "candidate"}:
        raise SystemExit(f"usage: {Path(sys.argv[0]).name} {{baseline,candidate}}")
    check(sys.argv[1])


if __name__ == "__main__":
    main()
