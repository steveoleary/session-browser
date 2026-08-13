"""Committed fixture-case runner tests."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

from session_browser import case_runner


def _write_committed_case(
    root: Path,
    name: str,
    *,
    accepted_state: str = "baseline",
    failing_state: str = "candidate",
) -> Path:
    """Create a real verifier process whose result records the requested state."""
    case = root / name
    case.mkdir(parents=True)
    (case / "case.json").write_text(
        json.dumps(
            {
                "name": name,
                "accepted_state": accepted_state,
                "verifier": "verify_case.py",
            }
        )
    )
    (case / "verify_case.py").write_text(
        textwrap.dedent(f"""\
        import sys
        from pathlib import Path

        state = sys.argv[1]
        Path(__file__).with_name("states.txt").open("a").write(state + "\\n")
        if state == {failing_state!r}:
            raise SystemExit(f"{{state}} is deliberately red")
        print(f"PASS: {{state}}")
    """)
    )
    return case


def test_committed_case_discovery_rejects_manifest_missing_accepted_state(tmp_path):
    """A manifest without the state to run must be rejected before execution."""
    case = tmp_path / "missing-state"
    case.mkdir()
    (case / "case.json").write_text(
        json.dumps(
            {
                "name": "missing-state",
                "verifier": "verify_case.py",
            }
        )
    )
    (case / "verify_case.py").write_text("raise SystemExit(0)\n")

    with pytest.raises(
        case_runner.CaseError, match="malformed committed case manifest"
    ):
        case_runner.discover_committed_cases(tmp_path)


def test_committed_case_runner_uses_accepted_state_without_running_candidate(tmp_path):
    """The normal suite runs only each manifest's selected green verifier state."""
    case = _write_committed_case(tmp_path, "accepted-baseline")

    results = case_runner.run_committed_cases(
        state="accepted",
        fixtures_root=tmp_path,
    )

    assert [(item.name, item.state, item.returncode) for item in results] == [
        ("accepted-baseline", "baseline", 0),
    ]
    assert (case / "states.txt").read_text().splitlines() == ["baseline"]


def test_committed_case_runner_selects_a_manifest_candidate_as_accepted(tmp_path):
    """Accepted state must come from each manifest, rather than baseline by default."""
    case = _write_committed_case(
        tmp_path,
        "accepted-candidate",
        accepted_state="candidate",
        failing_state="baseline",
    )

    accepted = case_runner.run_committed_cases(
        state="accepted",
        fixtures_root=tmp_path,
    )
    baseline = case_runner.run_committed_cases(
        state="baseline",
        fixtures_root=tmp_path,
    )

    assert [(item.state, item.returncode) for item in accepted] == [("candidate", 0)]
    assert [(item.state, item.returncode) for item in baseline] == [("baseline", 1)]
    assert (case / "states.txt").read_text().splitlines() == [
        "candidate",
        "baseline",
    ]


def test_module_cli_lists_committed_cases_with_accepted_states():
    """The executable module exposes discoverable committed-case state."""
    result = subprocess.run(
        [sys.executable, "-m", "session_browser.case_runner", "cases", "--list"],
        cwd=Path(__file__).resolve().parents[1],
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout.splitlines() == [
        "claude-queued-command-fidelity\taccepted=candidate",
        "conversation-first-overlooked-session\taccepted=baseline",
    ]


def test_module_cli_run_aggregates_synthetic_candidate_results(tmp_path):
    """The module reports every temp candidate and fails if any candidate fails."""
    root = Path(__file__).resolve().parents[1]
    subprocess.run(
        ["git", "init", "--quiet", str(tmp_path)],
        text=True,
        capture_output=True,
        check=True,
    )
    fixtures = tmp_path / "docs" / "fixtures"
    red = _write_committed_case(fixtures, "synthetic-red")
    green = _write_committed_case(
        fixtures,
        "synthetic-green",
        failing_state="baseline",
    )
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join(
        filter(
            None,
            [
                str(root),
                env.get("PYTHONPATH"),
            ],
        )
    )

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "session_browser.case_runner",
            "run",
            "--state",
            "candidate",
        ],
        cwd=tmp_path,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 1
    assert "FAIL: synthetic-red [candidate]" in result.stdout
    assert "PASS: synthetic-green [candidate]" in result.stdout
    assert (red / "states.txt").read_text().splitlines() == ["candidate"]
    assert (green / "states.txt").read_text().splitlines() == ["candidate"]


def test_committed_fixture_cases_have_independent_accepted_baselines():
    """The parser and ordering regressions must be green without coupling phases.

    The parser case is accepted at ``candidate`` since the queued-command fix
    landed, while the ordering case stays at ``baseline`` until Phase 3. Both
    being green in different states is the point of splitting them.
    """
    cases = {case.name: case for case in case_runner.discover_committed_cases()}

    assert set(cases) == {
        "claude-queued-command-fidelity",
        "conversation-first-overlooked-session",
    }
    assert cases["conversation-first-overlooked-session"].accepted_state == "baseline"
    assert cases["claude-queued-command-fidelity"].accepted_state == "candidate"

    results = case_runner.run_committed_cases(state="accepted")

    assert [(item.name, item.state, item.returncode) for item in results] == [
        ("claude-queued-command-fidelity", "candidate", 0),
        ("conversation-first-overlooked-session", "baseline", 0),
    ]
