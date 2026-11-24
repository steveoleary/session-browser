"""Grok as a provider: discovery, transcript, search, resume, ending, live.

Every record shape is one grok 1.0.92 wrote on the author's machine
(~/.grok/sessions, 17 sessions, 2026-11-24), cut to the fields read.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from unittest.mock import patch

import pytest

from session_browser import discovery, live
from session_browser.cli import run_cli
from session_browser.discovery import scan_grok
from session_browser.resume import resume_command
from session_browser.transcript import load_transcript, search_session

SID = "01a1d169-6a61-7cc1-b039-9190e4a53e3f"
CWD = "/Users/test/Projects/mystuff"


def _update(kind: str, ts: int = 1794514892, **fields) -> dict:
    return {
        "timestamp": ts,
        "method": "session/update",
        "params": {"sessionId": SID, "update": {"sessionUpdate": kind, **fields}},
    }


def _text(text: str) -> dict:
    return {"type": "text", "text": text}


HOOK = _update(
    "hook_execution", event_name="session_start", runs=[{"name": "plugin:x"}]
)
PROMPT = _update("user_message_chunk", content=_text("Find the slow query"))
THOUGHT = _update("agent_thought_chunk", content=_text("Looking at the log."))
CALL = _update(
    "tool_call",
    toolCallId="call-1",
    title="read_file",
    rawInput={"target_file": "/p/slow.log"},
)
STARTED = _update("tool_call_update", toolCallId="call-1", title="Read slow.log")
DONE = _update(
    "tool_call_update",
    toolCallId="call-1",
    status="completed",
    content=[{"type": "content", "content": _text("SELECT * FROM wombats")}],
)
REPLY = _update("agent_message_chunk", ts=1794514895, content=_text("It is the scan."))
CLOSED = _update("turn_completed", prompt_id="p1", stop_reason="end_turn")


def _session_dir(
    home: Path,
    *,
    sid: str = SID,
    cwd: str = CWD,
    updates: list[dict] | None = None,
    events: list[dict] | None = None,
    **summary,
) -> Path:
    d = home / ".grok" / "sessions" / cwd.replace("/", "%2F") / sid
    d.mkdir(parents=True)
    meta = {
        "info": {"id": sid, "cwd": cwd},
        "session_summary": "Slow query hunt",
        "created_at": "2026-11-12T20:18:29.151436Z",
        "updated_at": "2026-11-12T20:21:43.442449Z",
        "last_active_at": "2026-11-12T20:21:35.514510Z",
        "current_model_id": "grok-4.7",
        "head_branch": "main",
        "git_remotes": ["https://github.com/test/mystuff.git"],
        "generated_title": "Slow query hunt",
        "agent_name": "grok-build-plan",
        **summary,
    }
    (d / "summary.json").write_text(json.dumps(meta))
    (d / "chat_history.jsonl").write_text(
        json.dumps({"type": "system", "content": "You are Grok"}) + "\n"
    )
    if updates is not None:
        (d / "updates.jsonl").write_text("".join(json.dumps(u) + "\n" for u in updates))
    if events is not None:
        (d / "events.jsonl").write_text("".join(json.dumps(e) + "\n" for e in events))
    return d


def _scan(home: Path):
    with patch("session_browser.discovery.Path.home", return_value=home):
        return {s.id: s for s in scan_grok()}


FULL = [HOOK, PROMPT, THOUGHT, CALL, STARTED, DONE, REPLY, CLOSED]


class TestDiscovery:
    def test_a_session_comes_from_its_summary_and_updates(self, tmp_path):
        d = _session_dir(tmp_path, updates=FULL)
        s = _scan(tmp_path)[SID]
        assert (s.provider, s.cwd, s.branch, s.model) == (
            "grok",
            CWD,
            "main",
            "grok-4.7",
        )
        # The person's first words, as everywhere else; the remote's name.
        assert s.summary == "Find the slow query"
        assert s.repository == "mystuff"
        assert s.created_at == "2026-11-12T20:18:29.151436Z"
        assert s.updated_at == "2026-11-12T20:21:35.514510Z"  # last active
        assert s.content_path == str(d / "updates.jsonl")
        assert (s.parent_id, s.subagent_kind) == ("", "")
        assert s.ending == "completed"

    def test_a_session_nothing_was_sent_in_is_not_listed(self, tmp_path):
        _session_dir(tmp_path, updates=None)
        assert _scan(tmp_path) == {}

    def test_the_title_stands_in_for_a_session_with_no_prompt(self, tmp_path):
        _session_dir(tmp_path, updates=[HOOK])
        s = _scan(tmp_path)[SID]
        assert s.summary == "Slow query hunt"
        assert s.ending is None

    def test_the_directory_name_is_the_cwd_when_the_summary_has_none(self, tmp_path):
        _session_dir(tmp_path, updates=FULL, info={"id": SID}, git_remotes=[])
        s = _scan(tmp_path)[SID]
        assert s.cwd == CWD
        assert s.repository == "mystuff"

    def test_a_second_scan_reads_nothing_while_the_files_are_unchanged(self, tmp_path):
        _session_dir(tmp_path, updates=FULL)
        first = _scan(tmp_path)[SID]
        with patch.object(discovery, "_grok_session") as read:
            again = _scan(tmp_path)[SID]
        read.assert_not_called()
        assert (again.summary, again.ending) == (first.summary, first.ending)

    def test_a_subagent_says_so_and_a_fork_does_not(self, tmp_path):
        child = "01a1d169-0000-7000-8000-000000000001"
        fork = "01a1d169-0000-7000-8000-000000000002"
        _session_dir(
            tmp_path,
            sid=child,
            updates=FULL,
            events=_turn(relationship="subagent"),
            parent_session_id=SID,
        )
        _session_dir(
            tmp_path, sid=fork, updates=FULL, events=_turn(), parent_session_id=SID
        )
        found = _scan(tmp_path)
        assert (found[child].subagent_kind, found[child].parent_id) == (
            "subagent",
            SID,
        )
        # parent_session_id is also the source of a fork or a restore.
        assert (found[fork].subagent_kind, found[fork].parent_id) == ("", "")

    def test_a_long_folder_name_is_read_from_its_cwd_file(self, tmp_path):
        d = _session_dir(tmp_path, updates=FULL, info={"id": SID})
        slug = d.parent.with_name("Users-test-a-very-long-path-1a2b3c")
        d.parent.rename(slug)
        (slug / ".cwd").write_text(CWD + "\n")
        assert _scan(tmp_path)[SID].cwd == CWD


def _turn(outcome="completed", *, relationship="primary", yolo=False) -> list[dict]:
    """One turn of events.jsonl, as grok 1.0.92 writes it, cut down."""
    events = [
        {"ts": "2026-11-12T20:18:29.344Z", "type": "mcp_config_resolved"},
        {
            "ts": "2026-11-12T20:21:32.320Z",
            "type": "turn_started",
            "session_id": SID,
            "turn_number": 0,
            "yolo_mode": yolo,
            "session_relationship": relationship,
        },
        {"ts": "2026-11-12T20:21:32.417Z", "type": "phase_changed", "phase": "x"},
    ]
    if outcome:
        events.append(
            {"ts": "2026-11-12T20:21:36Z", "type": "turn_ended", "outcome": outcome}
        )
    return events


def test_a_turn_that_ended_cancelled_is_interrupted(tmp_path):
    # updates.jsonl shows only where it stopped; events.jsonl says why.
    _session_dir(tmp_path, updates=[HOOK, PROMPT, REPLY], events=_turn("cancelled"))
    assert _scan(tmp_path)[SID].ending == "interrupted"


def test_a_turn_still_open_after_a_cancel_is_read_from_the_transcript(tmp_path):
    events = [*_turn("cancelled"), *_turn(None)[1:]]
    _session_dir(tmp_path, updates=[HOOK, PROMPT, CALL], events=events)
    assert _scan(tmp_path)[SID].ending == "mid-action"


class TestLive:
    def _registry(self, home: Path, rows) -> None:
        path = home / ".grok" / "active_sessions.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(rows if isinstance(rows, str) else json.dumps(rows))

    def test_the_open_session_registry_answers(self, tmp_path):
        other = "01a1d169-0000-7000-8000-00000000000f"
        _session_dir(tmp_path, updates=FULL)
        _session_dir(tmp_path, sid=other, updates=FULL)
        self._registry(
            tmp_path,
            [
                {"session_id": SID, "pid": os.getpid(), "cwd": CWD},
                # A pid far above any the OS hands out: a leftover.
                {"session_id": other, "pid": 999999, "cwd": CWD},
            ],
        )
        found = _scan(tmp_path)
        assert (found[SID].live, found[other].live) == (True, False)

    def test_no_registry_means_nothing_is_open(self, tmp_path):
        _session_dir(tmp_path, updates=FULL)
        assert _scan(tmp_path)[SID].live is False

    def test_an_unreadable_registry_says_nothing(self, tmp_path):
        _session_dir(tmp_path, updates=FULL)
        self._registry(tmp_path, "{not json")
        assert _scan(tmp_path)[SID].live is None

    def test_the_process_table_is_not_asked_about_grok(self, tmp_path):
        _session_dir(tmp_path, updates=FULL)
        s = _scan(tmp_path)[SID]
        with patch.object(live, "agent_processes") as probe:
            live.fill_live([s])
        probe.assert_not_called()


@pytest.mark.parametrize(
    ("updates", "ending"),
    [
        pytest.param(FULL, "completed", id="turn-closed"),
        pytest.param([HOOK, PROMPT, THOUGHT, CALL, STARTED], "mid-action", id="call"),
        pytest.param([HOOK, PROMPT, CALL, DONE, THOUGHT], "mid-action", id="output"),
        pytest.param([HOOK, PROMPT, THOUGHT], "unanswered", id="no-reply"),
        pytest.param([HOOK, PROMPT, REPLY], "completed", id="reply-not-closed"),
        pytest.param(
            [HOOK, PROMPT, CALL, _update("turn_completed", stop_reason="cancelled")],
            "interrupted",
            id="cancelled",
        ),
        pytest.param(
            [*FULL, _update("user_message_chunk", content=_text("and the index?"))],
            "unanswered",
            id="next-prompt",
        ),
        pytest.param(
            [*FULL, _update("current_mode_update", currentModeId="plan")],
            "completed",
            id="bookkeeping-after",
        ),
    ],
)
def test_endings(tmp_path, updates, ending):
    _session_dir(tmp_path, updates=updates)
    assert _scan(tmp_path)[SID].ending == ending


class TestTranscript:
    def test_entries_and_their_times(self, tmp_path):
        failed = _update(
            "tool_call_update",
            toolCallId="call-2",
            status="failed",
            content=[{"type": "content", "content": _text("No such file")}],
        )
        edit = _update(
            "tool_call_update",
            toolCallId="call-3",
            status="completed",
            content=[
                {"type": "diff", "path": "/p/fix.sql", "oldText": "", "newText": "X"}
            ],
        )
        _session_dir(tmp_path, updates=[*FULL, failed, edit])
        t = load_transcript(_scan(tmp_path)[SID])
        assert [(e.role, e.metadata) for e in t.entries] == [
            ("user", None),
            ("tool", {"kind": "call", "tool": "read_file", "id": "call-1"}),
            ("tool", {"kind": "output", "call_id": "call-1"}),
            ("assistant", None),
            ("tool", {"kind": "output", "call_id": "call-2", "is_error": True}),
            ("tool", {"kind": "output", "call_id": "call-3"}),
        ]
        assert t.entries[0].text == "Find the slow query"
        assert t.entries[0].timestamp == "2026-11-12T20:21:32Z"
        assert t.entries[1].text == 'read_file({"target_file": "/p/slow.log"})'
        assert t.entries[2].text == "SELECT * FROM wombats"
        assert t.entries[5].text == "/p/fix.sql\nX"
        assert t.warnings == []

    def test_search_reaches_a_tool_output(self, tmp_path):
        _session_dir(tmp_path, updates=FULL)
        r = search_session(_scan(tmp_path)[SID], "wombats")
        assert r.match_count == 1
        assert r.matches[0].entry.role == "tool"


class TestResume:
    def _events(self, yolo: bool) -> list[dict]:
        return [
            {"ts": "2026-11-12T20:18:29.344Z", "type": "mcp_config_resolved"},
            {
                "ts": "2026-11-12T20:21:32.320Z",
                "type": "turn_started",
                "session_id": SID,
                "turn_number": 0,
                "yolo_mode": yolo,
            },
            {"ts": "2026-11-12T20:22:00Z", "type": "yolo_toggled", "enabled": not yolo},
        ]

    def test_a_session_started_in_yolo_mode_resumes_in_it(self, tmp_path):
        _session_dir(tmp_path, updates=FULL, events=self._events(True))
        s = _scan(tmp_path)[SID]
        assert resume_command("grok", s.id, s.cwd, content_path=s.content_path) == (
            f"cd {CWD} && grok --always-approve --resume {SID}"
        )

    def test_one_that_started_asking_resumes_asking(self, tmp_path):
        # Toggled on later: the mode it started in is what resume restores.
        _session_dir(tmp_path, updates=FULL, events=self._events(False))
        s = _scan(tmp_path)[SID]
        assert resume_command("grok", s.id, content_path=s.content_path) == (
            f"grok --resume {SID}"
        )


def test_the_cli_lists_grok_sessions(tmp_path, capsys):
    _session_dir(tmp_path, updates=FULL)
    with patch("session_browser.discovery.Path.home", return_value=tmp_path):
        code = run_cli(["list", "--provider", "grok", "--format", "json"])
    assert code == 0
    rows = json.loads(capsys.readouterr().out)["sessions"]
    assert [(r["id"], r["ending"], r["total_entries"]) for r in rows] == [
        (f"grok:{SID}", "completed", 4)
    ]
