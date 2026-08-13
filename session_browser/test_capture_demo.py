"""Focused tests for scripts/capture_demo.py (the curated screenshot capture).

The script drives the real TUI headless to three curated states; these tests
lock those states down so a UI change that breaks a curated frame fails the
suite instead of silently shipping a wrong screenshot.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "capture_demo.py"

SHOT_FILENAMES = [
    "01-list-160x45.svg",
    "02-search-sqlite-120x40.svg",
    "03-detail-focus-wal-120x40.svg",
]


def _run(*extra: str, timeout: int = 180) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(SCRIPT), *extra],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
        timeout=timeout,
    )


def test_check_mode_verifies_all_three_curated_states(tmp_path):
    """--check drives each state, asserts its invariants, and writes nothing."""
    home = tmp_path / "demo-home"
    run = _run("--check", "--home", str(home))
    assert run.returncode == 0, run.stderr
    for name in (
        "01-list-160x45",
        "02-search-sqlite-120x40",
        "03-detail-focus-wal-120x40",
    ):
        assert f"PASS  {name}" in run.stdout
    assert not any(tmp_path.rglob("*.svg")), "check mode must not write screenshots"


def test_write_mode_writes_stable_filenames(tmp_path):
    """A capture run writes exactly the three curated SVGs, no timestamps."""
    home = tmp_path / "demo-home"
    out = tmp_path / "shots"
    run = _run("--home", str(home), "--output-dir", str(out))
    assert run.returncode == 0, run.stderr
    assert sorted(p.name for p in out.glob("*.svg")) == SHOT_FILENAMES
    for svg in out.glob("*.svg"):
        assert svg.stat().st_size > 0
