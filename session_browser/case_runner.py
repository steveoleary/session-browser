"""Runner for the committed regression fixtures under ``docs/fixtures``.

Each fixture directory holds a ``case.json`` manifest, a ``verify_case.py``
verifier and a provider-native ``home/`` tree. A case declares which of its two
states — ``baseline`` or ``candidate`` — is currently accepted, so a fixture
that describes a known-but-unfixed defect can stay green at ``baseline`` while a
fixed one is green at ``candidate``.

Invoke as ``python -m session_browser.case_runner``.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

_COMMITTED_CASE_STATES = ("baseline", "candidate")


class CaseError(Exception):
    """A malformed or missing fixture case that must not be silently ignored."""


@dataclass(frozen=True)
class CommittedCase:
    """One portable fixture with a verifier and its normal accepted state."""

    name: str
    accepted_state: str
    verifier: Path


@dataclass(frozen=True)
class CaseRunResult:
    """The externally observable outcome of one verifier subprocess."""

    name: str
    state: str
    returncode: int


def _repository_root() -> Path:
    result = subprocess.run(
        ["git", "rev-parse", "--show-toplevel"],
        text=True,
        capture_output=True,
        check=False,
    )
    if result.returncode:
        raise CaseError("could not locate Git working tree")
    return Path(result.stdout.strip()).resolve()


def _committed_fixtures_root() -> Path:
    return _repository_root() / "docs" / "fixtures"


def _read_committed_case(manifest_path: Path) -> CommittedCase:
    try:
        manifest = json.loads(manifest_path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise CaseError(f"malformed committed case manifest: {manifest_path}") from exc
    if not isinstance(manifest, dict):
        raise CaseError(f"malformed committed case manifest: {manifest_path}")
    name = manifest.get("name")
    accepted_state = manifest.get("accepted_state")
    verifier_text = manifest.get("verifier")
    if (
        not isinstance(name, str)
        or not name
        or not isinstance(accepted_state, str)
        or accepted_state not in _COMMITTED_CASE_STATES
        or not isinstance(verifier_text, str)
        or not verifier_text
    ):
        raise CaseError(f"malformed committed case manifest: {manifest_path}")
    if name != manifest_path.parent.name:
        raise CaseError(f"malformed committed case manifest: {manifest_path}")
    verifier = Path(verifier_text)
    if verifier.is_absolute() or ".." in verifier.parts:
        raise CaseError(f"malformed committed case manifest: {manifest_path}")
    verifier = manifest_path.parent / verifier
    if not verifier.is_file():
        raise CaseError(f"committed case verifier missing: {verifier}")
    return CommittedCase(
        name=name, accepted_state=accepted_state, verifier=verifier.resolve()
    )


def discover_committed_cases(fixtures_root: Path | None = None) -> list[CommittedCase]:
    """Discover validated committed fixture cases in a stable order."""
    root = (
        Path(fixtures_root) if fixtures_root is not None else _committed_fixtures_root()
    )
    if not root.is_dir():
        raise CaseError(f"committed fixtures directory missing: {root}")
    cases = [_read_committed_case(path) for path in sorted(root.rglob("case.json"))]
    names = [case.name for case in cases]
    if len(names) != len(set(names)):
        raise CaseError("committed case manifests have duplicate names")
    return cases


def run_committed_cases(
    *,
    state: str = "accepted",
    case: str | None = None,
    fixtures_root: Path | None = None,
) -> list[CaseRunResult]:
    """Run selected committed verifiers, preserving their visible output."""
    if state != "accepted" and state not in _COMMITTED_CASE_STATES:
        raise CaseError(f"unknown committed case state: {state}")
    cases = discover_committed_cases(fixtures_root)
    if case is not None:
        cases = [item for item in cases if item.name == case]
        if not cases:
            raise CaseError(f"unknown committed case: {case}")
    results = []
    for item in cases:
        selected_state = item.accepted_state if state == "accepted" else state
        result = subprocess.run(
            [sys.executable, str(item.verifier), selected_state],
            cwd=_repository_root(),
            check=False,
        )
        results.append(
            CaseRunResult(
                name=item.name,
                state=selected_state,
                returncode=result.returncode,
            )
        )
    return results


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m session_browser.case_runner")
    sub = parser.add_subparsers(dest="command", required=True)
    cases = sub.add_parser("cases")
    cases.add_argument("--list", action="store_true", required=True)
    run = sub.add_parser("run")
    run.add_argument(
        "--state", default="accepted", choices=("accepted", *_COMMITTED_CASE_STATES)
    )
    run.add_argument("--case")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "cases":
            for case in discover_committed_cases():
                print(f"{case.name}\taccepted={case.accepted_state}")
        elif args.command == "run":
            results = run_committed_cases(state=args.state, case=args.case)
            failures = [result for result in results if result.returncode]
            for result in results:
                status = "PASS" if result.returncode == 0 else "FAIL"
                print(f"{status}: {result.name} [{result.state}]")
            if failures:
                return 1
    except CaseError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
