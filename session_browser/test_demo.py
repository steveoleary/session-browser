"""Tests for the demo corpus generator (session_browser.demo).

The generator writes files and never touches the retrieval path, so these
tests assert corpus shape and end-to-end usability: discovery, search, and
the CLI all read the generated fake HOME exactly as they read a real one.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

import pytest

from session_browser.demo import (
    CODEX_SESSIONS,
    OPENCODE_DB,
    PI_SESSIONS,
    SCRATCHPAD,
    _specs,
    generate,
)
from session_browser.discovery import Session, discover_all
from session_browser.transcript import iter_entries, search_session

REPO_ROOT = Path(__file__).resolve().parents[1]
NOW = datetime(2026, 8, 13, 12, 0, 0, tzinfo=UTC)


@pytest.fixture
def fake_home(tmp_path, monkeypatch):
    home = tmp_path / "demo-home"
    generate(home, now=NOW)
    monkeypatch.setattr("session_browser.discovery.Path.home", lambda: home)
    return home


def _discover() -> list[Session]:
    return discover_all()


def _by_provider(sessions: list[Session]) -> dict[str, list[Session]]:
    out: dict[str, list[Session]] = {}
    for s in sessions:
        out.setdefault(s.provider, []).append(s)
    return out


def test_generate_writes_all_provider_trees(tmp_path):
    home = tmp_path / "home"
    result = generate(home, now=NOW)

    claude_files = list((home / ".claude" / "projects").rglob("*.jsonl"))
    codex_files = list((home / CODEX_SESSIONS).rglob("rollout-*.jsonl"))
    assert len(claude_files) == result.claude
    assert len(codex_files) == result.codex
    pi_files = list((home / PI_SESSIONS).rglob("*.jsonl"))
    assert len(pi_files) == result.pi
    assert (home / OPENCODE_DB).is_file()
    assert result.total == 17


def test_corpus_is_discoverable(fake_home):
    sessions = _discover()
    by_provider = _by_provider(sessions)
    assert len(sessions) == 17
    assert len(by_provider["claude"]) == 9
    assert len(by_provider["codex"]) == 4
    assert len(by_provider["opencode"]) == 2
    assert len(by_provider["pi"]) == 2


def test_search_finds_seeded_terms(fake_home):
    sessions = _discover()
    hits = {
        sid: search_session(s, term).match_count > 0
        for sid, term in {
            "claude:8f2c4a1b-9d3e-4f5b-8c6a-1a2b3c4d5e6f": "sqlite",
            "claude:6a1d8e3f-7c2b-4d9e-8a4f-5b6c7d8e9f0a": "asyncio",
            "codex:019d2f4a-8c33-7b2e-9d41-5f6a7b8c9d01": "docker",
            "opencode:ses_demo_ripgrep": "ripgrep",
            "pi:01a020b0-7ffa-7552-8d62-2aeb60b18db1": "reverse",
        }.items()
        for s in sessions
        if f"{s.provider}:{s.id}" == sid
    }
    assert hits == {sid: True for sid in hits}


def test_big_session_has_many_entries(fake_home):
    big = next(s for s in _discover() if s.id == "8f2c4a1b-9d3e-4f5b-8c6a-1a2b3c4d5e6f")
    entries = list(iter_entries(big, []))
    assert len(entries) >= 60
    assert {e.role for e in entries} == {"user", "assistant", "tool"}


def test_scratchpad_noise_sessions_present(fake_home):
    noise = [s for s in _discover() if SCRATCHPAD in s.cwd]
    assert len(noise) == 3
    assert all(s.provider == "claude" for s in noise)


def test_dates_spread_over_weeks(fake_home):
    sessions = _discover()
    times = [datetime.fromisoformat(s.updated_at) for s in sessions if s.updated_at]
    assert (max(times) - min(times)).days >= 20
    assert max(times) <= NOW


def test_refuses_existing_directory_without_force(tmp_path):
    home = tmp_path / "home"
    generate(home, now=NOW)
    with pytest.raises(FileExistsError):
        generate(home, now=NOW)
    result = generate(home, now=NOW, force=True)
    assert result.total == 17


def test_same_anchor_reproduces_identical_bytes(tmp_path):
    first = tmp_path / "first"
    second = tmp_path / "second"
    generate(first, now=NOW)
    generate(second, now=NOW)
    a = {p.relative_to(first): p.read_bytes() for p in first.rglob("*") if p.is_file()}
    b = {
        p.relative_to(second): p.read_bytes() for p in second.rglob("*") if p.is_file()
    }
    assert a == b


def test_current_checkout_sessions_carry_real_cwd():
    specs = _specs(str(Path.cwd().resolve()))
    here_sessions = [s for s in specs if s.cwd == str(Path.cwd().resolve())]
    assert len(here_sessions) >= 2
    assert {s.provider for s in here_sessions} == {"claude", "opencode"}


def test_cli_generation_and_listing_end_to_end(tmp_path):
    home = tmp_path / "demo-home"
    gen = subprocess.run(
        [sys.executable, "-m", "session_browser.demo", "--home", str(home)],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert gen.returncode == 0, gen.stderr
    assert "Wrote 17 sessions" in gen.stdout

    env = os.environ.copy()
    env["HOME"] = str(home)
    listing = subprocess.run(
        [sys.executable, "-m", "session_browser.app", "list"],
        cwd=REPO_ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )
    assert listing.returncode == 0, listing.stderr
    payload = json.loads(listing.stdout)
    assert len(payload["sessions"]) == 17
    assert {s["provider"] for s in payload["sessions"]} == {
        "claude",
        "codex",
        "opencode",
        "pi",
    }
