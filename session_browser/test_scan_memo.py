"""The scan memo: what discovery read from each transcript, kept between runs.

Tested the way the entry cache is, for the same reason: the memo is only an
optimisation of the scan, so a discovery that used it -- from nothing, and
from a memo already full -- must return exactly the sessions a discovery with
it switched off returns, field for field.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from unittest.mock import patch

import pytest

from session_browser import discovery, perf_budget, scan_memo
from session_browser.discovery import discover_all, scan_claude, scan_pi


@pytest.fixture(autouse=True)
def cache_home(tmp_path, monkeypatch):
    root = tmp_path / "xdg-cache"
    monkeypatch.setenv("XDG_CACHE_HOME", str(root))
    monkeypatch.delenv("SESSION_BROWSER_CACHE", raising=False)
    return root / "session-browser" / "discovery"


@pytest.fixture
def home(tmp_path):
    with patch("session_browser.discovery.Path.home", return_value=tmp_path):
        yield tmp_path


class TestMemo:
    def test_what_is_put_is_got_back_by_the_next_run(self):
        memo = scan_memo.Memo("claude")
        assert memo.get("/a.jsonl", [10, 99]) is scan_memo.MISS
        memo.put("/a.jsonl", [10, 99], {"id": "a", "branch": None})
        memo.save()
        again = scan_memo.Memo("claude")
        assert again.get("/a.jsonl", [10, 99]) == {"id": "a", "branch": None}

    def test_a_changed_signature_is_a_miss(self):
        memo = scan_memo.Memo("claude")
        memo.put("/a.jsonl", [10, 99], {"id": "a"})
        memo.save()
        again = scan_memo.Memo("claude")
        assert again.get("/a.jsonl", [11, 99]) is scan_memo.MISS
        assert again.get("/a.jsonl", [10, 100]) is scan_memo.MISS

    def test_not_a_session_is_remembered_and_is_not_a_miss(self):
        """A resume stub scans to None. Remembered as a miss it would be
        opened and read again by every command, for ever."""
        memo = scan_memo.Memo("claude")
        memo.put("/stub.jsonl", [1, 2], None)
        memo.save()
        assert scan_memo.Memo("claude").get("/stub.jsonl", [1, 2]) is None

    def test_a_file_not_met_this_run_is_forgotten(self):
        memo = scan_memo.Memo("claude")
        memo.put("/kept.jsonl", [1, 2], {"id": "kept"})
        memo.put("/gone.jsonl", [1, 2], {"id": "gone"})
        memo.save()
        second = scan_memo.Memo("claude")
        assert second.get("/kept.jsonl", [1, 2]) == {"id": "kept"}
        second.save()  # /gone.jsonl was not asked about: its transcript is gone
        assert scan_memo.Memo("claude").get("/gone.jsonl", [1, 2]) is scan_memo.MISS

    def test_a_run_that_changed_nothing_writes_nothing(self):
        memo = scan_memo.Memo("claude")
        memo.put("/a.jsonl", [1, 2], {"id": "a"})
        memo.save()
        again = scan_memo.Memo("claude")
        again.get("/a.jsonl", [1, 2])
        with patch("session_browser.scan_memo.os.replace") as replace:
            again.save()
        replace.assert_not_called()

    def test_providers_do_not_share_a_file(self):
        claude = scan_memo.Memo("claude")
        claude.put("/a.jsonl", [1, 2], {"id": "a"})
        claude.save()
        assert scan_memo.Memo("pi").get("/a.jsonl", [1, 2]) is scan_memo.MISS

    def test_a_changed_scanner_starts_from_nothing_and_clears_up(self, cache_home):
        memo = scan_memo.Memo("claude")
        memo.put("/a.jsonl", [1, 2], {"id": "a"})
        memo.save()
        with patch.object(
            scan_memo.entry_cache, "parser_version", return_value="another"
        ):
            later = scan_memo.Memo("claude")
            assert later.get("/a.jsonl", [1, 2]) is scan_memo.MISS
            later.put("/a.jsonl", [1, 2], {"id": "a"})
            later.save()
        assert [p.name for p in cache_home.iterdir()] == ["claude.another.json"]

    def test_a_damaged_file_is_nothing_remembered(self, cache_home):
        memo = scan_memo.Memo("claude")
        memo.put("/a.jsonl", [1, 2], {"id": "a"})
        memo.save()
        next(cache_home.iterdir()).write_text("{ not json")
        assert scan_memo.Memo("claude").get("/a.jsonl", [1, 2]) is scan_memo.MISS

    def test_switched_off_there_is_no_memo(self, monkeypatch):
        monkeypatch.setenv("SESSION_BROWSER_CACHE", "0")
        assert scan_memo.open_memo("claude") is None

    def test_signature_follows_size_mtime_and_a_companion_file(self, tmp_path):
        f = tmp_path / "t.jsonl"
        f.write_text("one\n")
        side = tmp_path / "t.meta.json"
        assert scan_memo.signature(str(tmp_path / "absent")) is None
        alone = scan_memo.signature(str(f), str(side))
        assert alone[2:] == [-1, -1]  # a missing companion is part of the answer
        side.write_text("{}")
        with_side = scan_memo.signature(str(f), str(side))
        assert with_side[:2] == alone[:2] and with_side[2:] != [-1, -1]
        with open(f, "a") as fh:
            fh.write("two\n")
        assert scan_memo.signature(str(f), str(side))[:2] != with_side[:2]


def _turn(role: str, text: str, ts: str, **extra) -> str:
    return json.dumps(
        {
            "type": role,
            "message": {"role": role, "content": text},
            "cwd": "/Users/test/project",
            "gitBranch": "main",
            "timestamp": ts,
            **extra,
        }
    )


def _claude_home(home: Path) -> Path:
    """Two sessions, one continued into the other, a subagent with a sidecar,
    and a resume stub that is not a session at all."""
    project = home / ".claude" / "projects" / "-Users-test-project"
    project.mkdir(parents=True)
    (project / "first.jsonl").write_text(
        _turn("user", "start the wombat survey", "2026-03-01T10:00:00Z")
        + "\n"
        + _turn("assistant", "started", "2026-03-01T10:00:05Z")
        + "\n"
        + json.dumps({"type": "continued-in", "continuedInSessionId": "second"})
        + "\n"
    )
    (project / "second.jsonl").write_text(
        _turn("user", "carry on with the survey", "2026-03-02T09:00:00Z")
        + "\n"
        + _turn("assistant", "carrying on", "2026-03-02T09:00:09Z")
        + "\n"
    )
    (project / "stub.jsonl").write_text(
        json.dumps({"type": "bridge-session", "sessionId": "stub"}) + "\n"
    )
    agents = project / "second" / "subagents"
    agents.mkdir(parents=True)
    (agents / "agent-a1.jsonl").write_text(
        _turn("user", "count the burrows", "2026-03-02T09:01:00Z") + "\n"
    )
    (agents / "agent-a1.meta.json").write_text(
        json.dumps({"agentType": "Explore", "description": "burrow count"})
    )
    return project


def _pi_home(home: Path) -> Path:
    project = home / ".pi" / "agent" / "sessions" / "--Users-test-project--"
    project.mkdir(parents=True)
    for n, text in enumerate(("map the echidna range", "check the quoll notes")):
        (project / f"2026-03-0{n + 1}T10-00-00_pi-{n}.jsonl").write_text(
            json.dumps(
                {
                    "type": "session",
                    "id": f"pi-{n}",
                    "cwd": "/Users/test/project",
                    "timestamp": f"2026-03-0{n + 1}T10:00:00Z",
                }
            )
            + "\n"
            + json.dumps(
                {
                    "type": "message",
                    "timestamp": f"2026-03-0{n + 1}T10:00:01Z",
                    "message": {"role": "user", "content": text},
                }
            )
            + "\n"
        )
    return project


def _three_ways(scan, monkeypatch):
    """*scan*'s sessions with the memo off, then from nothing, then full."""
    monkeypatch.setenv("SESSION_BROWSER_CACHE", "0")
    reference = scan()
    monkeypatch.delenv("SESSION_BROWSER_CACHE")
    return reference, scan(), scan()


class TestClaudeDiscovery:
    def test_cold_and_warm_return_what_a_plain_scan_returns(self, home, monkeypatch):
        _claude_home(home)
        reference, cold, warm = _three_ways(scan_claude, monkeypatch)
        assert {s.id for s in reference} == {"first", "second", "agent-a1"}
        by_id = {s.id: s for s in reference}
        # The links a remembered scan has to carry, or rebuild, to be equal.
        assert by_id["first"].continued_in == "second"
        assert by_id["second"].continued_from == "first"
        assert by_id["agent-a1"].summary == "burrow count"
        assert cold == reference
        assert warm == reference

    def test_a_warm_scan_opens_no_transcript(self, home):
        _claude_home(home)
        scan_claude()
        with patch.object(discovery, "_claude_session") as scanned:
            sessions = scan_claude()
        scanned.assert_not_called()
        assert len(sessions) == 3

    def test_only_a_transcript_that_grew_is_read_again(self, home):
        project = _claude_home(home)
        scan_claude()
        with open(project / "second.jsonl", "a") as fh:
            fh.write(_turn("user", "one more thing", "2026-03-05T12:00:00Z") + "\n")
        with patch.object(
            discovery, "_claude_session", wraps=discovery._claude_session
        ) as scanned:
            sessions = scan_claude()
        assert [c.args[0].name for c in scanned.call_args_list] == ["second.jsonl"]
        second = next(s for s in sessions if s.id == "second")
        assert second.updated_at == "2026-03-05T12:00:00Z"

    def test_a_sidecar_that_changes_relabels_its_subagent(self, home):
        project = _claude_home(home)
        scan_claude()
        sidecar = project / "second" / "subagents" / "agent-a1.meta.json"
        sidecar.write_text(
            json.dumps({"agentType": "Plan", "description": "a longer burrow count"})
        )
        agent = next(s for s in scan_claude() if s.id == "agent-a1")
        assert (agent.subagent_kind, agent.summary) == ("Plan", "a longer burrow count")

    def test_a_deleted_transcript_is_gone_from_the_next_scan(self, home):
        project = _claude_home(home)
        scan_claude()
        os.unlink(project / "first.jsonl")
        sessions = scan_claude()
        assert {s.id for s in sessions} == {"second", "agent-a1"}
        # And the link it carried goes with it, rather than being remembered.
        assert next(s for s in sessions if s.id == "second").continued_from is None

    def test_a_remembered_session_is_not_the_memos_own_object(self, home):
        """Callers mutate sessions -- links, liveness, segments. Two scans
        must not hand back objects that share state through the memo."""
        _claude_home(home)
        scan_claude()
        first = next(s for s in scan_claude() if s.id == "second")
        first.summary = "scribbled on"
        again = next(s for s in scan_claude() if s.id == "second")
        assert again.summary == "carry on with the survey"


class TestPiDiscovery:
    def test_cold_and_warm_return_what_a_plain_scan_returns(self, home, monkeypatch):
        _pi_home(home)
        reference, cold, warm = _three_ways(scan_pi, monkeypatch)
        assert sorted(s.id for s in reference) == ["pi-0", "pi-1"]
        assert reference[0].branch is None  # Pi records none: kept as None
        assert cold == reference
        assert warm == reference

    def test_a_warm_scan_opens_no_transcript(self, home):
        _pi_home(home)
        scan_pi()
        with patch.object(discovery, "_pi_session") as scanned:
            assert len(scan_pi()) == 2
        scanned.assert_not_called()

    def test_only_a_transcript_that_grew_is_read_again(self, home):
        project = _pi_home(home)
        scan_pi()
        target = min(project.iterdir())
        with open(target, "a") as fh:
            fh.write(
                json.dumps(
                    {
                        "type": "message",
                        "timestamp": "2026-03-09T08:00:00Z",
                        "message": {"role": "assistant", "content": "mapped"},
                    }
                )
                + "\n"
            )
        with patch.object(
            discovery, "_pi_session", wraps=discovery._pi_session
        ) as scanned:
            sessions = scan_pi()
        assert [c.args[0] for c in scanned.call_args_list] == [target]
        assert next(s for s in sessions if s.id == "pi-0").updated_at == (
            "2026-03-09T08:00:00Z"
        )


def test_the_whole_synthetic_corpus_discovers_the_same_three_ways(
    tmp_path, monkeypatch
):
    """Every provider and era the perf corpus holds, through discover_all."""
    corpus_home = perf_budget.build_corpus(tmp_path / "perf")
    with patch("session_browser.discovery.Path.home", return_value=corpus_home):
        reference, cold, warm = _three_ways(discover_all, monkeypatch)
    assert len(reference) > 300
    assert cold == reference
    assert warm == reference
