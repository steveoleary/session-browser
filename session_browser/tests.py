"""Tests for session_browser."""

from __future__ import annotations

import json
import os
import re
import sqlite3
import subprocess
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from session_browser import herdr, multiplexer, tmux
from session_browser.discovery import (
    Session,
    _epoch_ms_to_iso,
    _file_mtime_iso,
    _last_activity_iso,
    _repo_name,
    _scan_codex_files,
    discover_all,
    scan_claude,
    scan_codex,
    scan_opencode,
)
from session_browser.herdr import HerdrError, HerdrPane, workspace_label_for_path
from session_browser.resume import (
    build_chat_export,
    copy_to_clipboard,
    export_chat_to_file,
    resume_command,
)
from session_browser.tmux import (
    TmuxError,
    build_plan,
    prepare_session,
    session_name_for_path,
)
from session_browser.transcript import (
    ContentHit,
    Transcript,
    TranscriptEntry,
    load_session_content,
    search_session_contents,
)

# ---------------------------------------------------------------------------
# Resume command tests
# ---------------------------------------------------------------------------


class TestResumeCommand:
    def test_claude(self):
        assert resume_command("claude", "def-456") == (
            "claude --dangerously-skip-permissions --resume def-456"
        )

    def test_codex(self):
        assert resume_command("codex", "ghi-789") == "codex resume ghi-789"

    def test_with_cwd(self):
        result = resume_command("codex", "abc", "/home/user/project")
        assert result == "cd /home/user/project && codex resume abc"

    def test_cwd_with_spaces(self):
        result = resume_command("claude", "x", "/tmp/my project")
        assert "cd '/tmp/my project'" in result
        assert "claude --dangerously-skip-permissions --resume x" in result

    def test_unknown_provider(self):
        result = resume_command("unknown", "id-1")
        assert "unknown provider" in result

    def test_no_cwd(self):
        result = resume_command("codex", "abc")
        assert not result.startswith("cd")


# ---------------------------------------------------------------------------
# Discovery tests
# ---------------------------------------------------------------------------


class TestSessionModel:
    def test_matches(self):
        s = Session(id="abc", provider="codex", summary="Fix CSS bug", branch="main")
        assert s.matches("css")
        assert s.matches("CODEX")
        assert s.matches("main")
        assert not s.matches("python")

    def test_sort_key(self):
        s1 = Session(id="a", provider="x", updated_at="2026-01-02")
        s2 = Session(id="b", provider="x", updated_at="2026-01-01")
        assert s1.sort_key > s2.sort_key

    def test_matches_empty_query(self):
        s = Session(id="abc", provider="codex")
        assert s.matches("")  # empty string matches everything


class TestScanClaude:
    def test_scan_claude_sessions(self, tmp_path):
        projects = tmp_path / ".claude" / "projects" / "-Users-test"
        projects.mkdir(parents=True)
        session_data = [
            json.dumps({"type": "permission-mode", "sessionId": "abc-123"}),
            json.dumps(
                {
                    "type": "user",
                    "message": {"role": "user", "content": "Fix the login page"},
                    "cwd": "/Users/test/project",
                    "gitBranch": "develop",
                    "timestamp": "2026-04-14T10:00:00Z",
                }
            ),
        ]
        (projects / "abc-123.jsonl").write_text("\n".join(session_data))

        with patch("session_browser.discovery.Path.home", return_value=tmp_path):
            sessions = scan_claude()

        assert len(sessions) == 1
        assert sessions[0].id == "abc-123"
        assert sessions[0].provider == "claude"
        assert "Fix the login" in sessions[0].summary
        assert sessions[0].branch == "develop"
        assert sessions[0].repository == "project"

    def test_cwd_falls_back_to_an_earlier_line_not_the_decoded_dir_name(self, tmp_path):
        """The dir-name decode turns "/" back into "-" blindly, so a project
        whose real leaf name has a hyphen decodes into a path that exists
        nowhere. Any cwd already on an earlier line beats it."""
        projects = (
            tmp_path / ".claude" / "projects" / "-Users-test-Projects-session-browser"
        )
        projects.mkdir(parents=True)
        lines = [
            {"type": "last-prompt", "cwd": ""},
            {"type": "attachment", "cwd": "/Users/test/Projects/session-browser"},
            {
                "type": "user",
                "message": {"content": "no cwd on this one"},
                "timestamp": "2026-08-13T10:00:00Z",
            },
        ]
        (projects / "s3.jsonl").write_text(
            "\n".join(json.dumps(x) for x in lines) + "\n"
        )

        with patch("session_browser.discovery.Path.home", return_value=tmp_path):
            sessions = scan_claude()

        assert sessions[0].cwd == "/Users/test/Projects/session-browser"
        assert sessions[0].repository == "session-browser"

    def test_cwd_decodes_dir_name_when_no_line_carries_one(self, tmp_path):
        """The decode stays as the last resort, for a transcript that records
        no cwd anywhere."""
        projects = tmp_path / ".claude" / "projects" / "-Users-test-agentlab"
        projects.mkdir(parents=True)
        lines = [
            {"type": "system", "timestamp": "2026-08-13T09:00:00Z"},
            {
                "type": "user",
                "message": {"content": "still no cwd"},
                "timestamp": "2026-08-13T10:00:00Z",
            },
        ]
        (projects / "s4.jsonl").write_text(
            "\n".join(json.dumps(x) for x in lines) + "\n"
        )

        with patch("session_browser.discovery.Path.home", return_value=tmp_path):
            sessions = scan_claude()

        assert sessions[0].cwd == "/Users/test/agentlab"

    def test_updated_at_uses_last_turn_not_system_or_mtime(self, tmp_path):
        """updated_at must reflect the last real turn, ignoring a later
        session-open 'system' event and a bumped file mtime."""
        projects = tmp_path / ".claude" / "projects" / "-Users-test"
        projects.mkdir(parents=True)
        lines = [
            {
                "type": "user",
                "message": {"content": "start"},
                "cwd": "/p",
                "timestamp": "2026-06-29T14:00:00.000Z",
            },
            {
                "type": "assistant",
                "message": {"content": "working"},
                "timestamp": "2026-06-29T15:28:00.000Z",
            },  # <- last real turn
            {"type": "system", "timestamp": "2026-06-30T08:47:00.000Z"},  # open event
            {"type": "mode"},  # untimestamped trailing meta
            {"type": "permission-mode"},
        ]
        f = projects / "s1.jsonl"
        f.write_text("\n".join(json.dumps(x) for x in lines) + "\n")
        os.utime(f, (4102444800, 4102444800))  # mtime = year 2100, must be ignored

        with patch("session_browser.discovery.Path.home", return_value=tmp_path):
            sessions = scan_claude()

        assert sessions[0].updated_at == "2026-06-29T15:28:00.000Z"

    def test_updated_at_falls_back_to_mtime_when_no_turn(self, tmp_path):
        """A transcript with no user/assistant turn (and no first-message
        created_at) falls back to file mtime, not an empty string."""
        projects = tmp_path / ".claude" / "projects" / "-Users-test"
        projects.mkdir(parents=True)
        f = projects / "s2.jsonl"
        f.write_text(
            "\n".join(
                json.dumps(x)
                for x in [
                    {"type": "system", "timestamp": "2026-06-30T08:00:00.000Z"},
                    {"type": "summary", "summary": "orphan"},
                ]
            )
            + "\n"
        )

        with patch("session_browser.discovery.Path.home", return_value=tmp_path):
            sessions = scan_claude()

        assert sessions[0].updated_at == _file_mtime_iso(f)
        assert sessions[0].updated_at  # non-empty


class TestScanCodex:
    def test_scan_codex_sessions(self, tmp_path):
        sessions_dir = tmp_path / ".codex" / "sessions" / "2026" / "04" / "10"
        sessions_dir.mkdir(parents=True)
        meta = json.dumps(
            {
                "type": "session_meta",
                "timestamp": "2026-04-10T13:00:00Z",
                "payload": {
                    "id": "019d-abc",
                    "cwd": "/Users/test/project",
                    "git": {"branch": "main"},
                },
            }
        )
        (sessions_dir / "rollout-2026-04-10T13-00-00-019d-abc.jsonl").write_text(
            meta + "\n"
        )

        with patch("session_browser.discovery.Path.home", return_value=tmp_path):
            sessions = scan_codex()

        assert len(sessions) == 1
        assert sessions[0].id == "019d-abc"
        assert sessions[0].provider == "codex"
        assert sessions[0].repository == "project"

    def test_updated_at_uses_last_turn_not_lifecycle(self, tmp_path):
        """updated_at must reflect the last user/agent message, ignoring
        trailing token_count / task_complete lifecycle events with later ts."""
        sessions_dir = tmp_path / ".codex" / "sessions" / "2026" / "06" / "29"
        sessions_dir.mkdir(parents=True)
        lines = [
            {
                "type": "session_meta",
                "timestamp": "2026-06-29T14:00:00Z",
                "payload": {"id": "cx-1", "cwd": "/p", "git": {"branch": "main"}},
            },
            {
                "type": "event_msg",
                "timestamp": "2026-06-29T14:00:05Z",
                "payload": {"type": "user_message", "message": "hi"},
            },
            {
                "type": "event_msg",
                "timestamp": "2026-06-29T15:28:00Z",
                "payload": {"type": "agent_message", "message": "done"},
            },  # last turn
            {
                "type": "event_msg",
                "timestamp": "2026-06-30T08:47:00Z",
                "payload": {"type": "token_count"},
            },  # lifecycle, later ts
            {
                "type": "event_msg",
                "timestamp": "2026-06-30T08:47:00Z",
                "payload": {"type": "task_complete"},
            },
        ]
        f = sessions_dir / "rollout-2026-06-29T14-00-00-cx-1.jsonl"
        f.write_text("\n".join(json.dumps(x) for x in lines) + "\n")

        with patch("session_browser.discovery.Path.home", return_value=tmp_path):
            sessions = scan_codex()

        assert sessions[0].updated_at == "2026-06-29T15:28:00Z"


class TestScanCodexDb:
    """The SQLite fast path: same sessions, one query instead of file opens."""

    def _write_db(self, home: Path, rows: list[dict]) -> None:
        db = home / ".codex" / "state_5.sqlite"
        conn = sqlite3.connect(str(db))
        conn.execute(
            "CREATE TABLE threads (id TEXT PRIMARY KEY, rollout_path TEXT NOT NULL, "
            "cwd TEXT NOT NULL, title TEXT NOT NULL DEFAULT '', "
            "git_branch TEXT, first_user_message TEXT NOT NULL DEFAULT '', "
            "created_at_ms INTEGER, updated_at_ms INTEGER, "
            "archived INTEGER NOT NULL DEFAULT 0)"
        )
        for row in rows:
            conn.execute(
                "INSERT INTO threads (id, rollout_path, cwd, git_branch, "
                "first_user_message, created_at_ms, updated_at_ms, archived) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    row["id"],
                    row["rollout_path"],
                    row.get("cwd", ""),
                    row.get("branch"),
                    row.get("summary", ""),
                    row.get("created_ms"),
                    row.get("updated_ms"),
                    row.get("archived", 0),
                ),
            )
        conn.commit()
        conn.close()

    def _write_rollout(
        self, sessions_dir: Path, sid: str, *, summary: str, day: str
    ) -> Path:
        sessions_dir.mkdir(parents=True, exist_ok=True)
        f = sessions_dir / f"rollout-2026-04-{day}T13-00-00-{sid}.jsonl"
        f.write_text(
            json.dumps(
                {
                    "type": "session_meta",
                    "timestamp": "2026-04-10T13:00:00Z",
                    "payload": {"id": sid, "cwd": "/p", "git": {"branch": "main"}},
                }
            )
            + "\n"
            + json.dumps(
                {
                    "type": "event_msg",
                    "timestamp": "2026-04-10T13:00:05Z",
                    "payload": {"type": "user_message", "message": summary},
                }
            )
            + "\n"
        )
        return f

    def test_sessions_come_from_db(self, tmp_path):
        """With an index present, discovery reads threads, not the rollouts."""
        sessions_dir = tmp_path / ".codex" / "sessions" / "2026" / "04" / "10"
        f = self._write_rollout(sessions_dir, "cx-1", summary="hello world", day="10")
        self._write_db(
            tmp_path,
            [
                {
                    "id": "cx-1",
                    "rollout_path": str(f),
                    "cwd": "/p",
                    "branch": "main",
                    "summary": "hello world",
                    "created_ms": 1775826000000,
                    "updated_ms": 1775826005000,
                }
            ],
        )

        with patch("session_browser.discovery.Path.home", return_value=tmp_path):
            sessions = scan_codex()

        assert len(sessions) == 1
        s = sessions[0]
        assert s.id == "cx-1"
        assert s.summary == "hello world"
        assert s.cwd == "/p"
        assert s.branch == "main"
        # Derived from cwd, so the fast path and the file scan agree on it
        # without the index needing a column for it.
        assert s.repository == "p"
        assert s.created_at == "2026-04-10T13:00:00.000Z"
        assert s.updated_at == "2026-04-10T13:00:05.000Z"
        assert s.content_path == str(f)

    def test_summary_matches_file_scan_transformation(self, tmp_path):
        """The index stores the untrimmed message; the scan applies the same
        truncate-and-collapse-newlines rule the file scan uses."""
        sessions_dir = tmp_path / ".codex" / "sessions" / "2026" / "04" / "10"
        f = self._write_rollout(sessions_dir, "cx-1", summary="irrelevant", day="10")
        self._write_db(
            tmp_path,
            [
                {
                    "id": "cx-1",
                    "rollout_path": str(f),
                    "summary": "line one\nline two\n" + ("x" * 200),
                    "created_ms": 1775826000000,
                    "updated_ms": 1775826005000,
                }
            ],
        )

        with patch("session_browser.discovery.Path.home", return_value=tmp_path):
            sessions = scan_codex()

        assert sessions[0].summary == "line one line two " + "x" * 102

    def test_falls_back_to_files_when_index_missing(self, tmp_path):
        """No state_*.sqlite means the file scan runs unchanged."""
        sessions_dir = tmp_path / ".codex" / "sessions" / "2026" / "04" / "10"
        self._write_rollout(sessions_dir, "cx-1", summary="hello", day="10")

        with patch("session_browser.discovery.Path.home", return_value=tmp_path):
            sessions = scan_codex()

        assert len(sessions) == 1
        assert sessions[0].id == "cx-1"
        assert sessions[0].summary == "hello"

    def test_falls_back_when_index_lags_a_just_written_rollout(self, tmp_path):
        """A rollout file with no matching row yet must not be invisible."""
        sessions_dir = tmp_path / ".codex" / "sessions" / "2026" / "04" / "10"
        self._write_rollout(sessions_dir, "cx-1", summary="hello", day="10")
        self._write_rollout(sessions_dir, "cx-2", summary="newest", day="11")
        self._write_db(
            tmp_path,
            [
                {
                    "id": "cx-1",
                    "rollout_path": str(
                        sessions_dir / "rollout-2026-04-10T13-00-00-cx-1.jsonl"
                    ),
                    "summary": "hello",
                    "created_ms": 1775826000000,
                    "updated_ms": 1775826005000,
                }
            ],
        )

        with patch("session_browser.discovery.Path.home", return_value=tmp_path):
            sessions = scan_codex()

        assert {s.id for s in sessions} == {"cx-1", "cx-2"}

    def test_phantom_row_is_dropped_without_fallback(self, tmp_path):
        """A row whose rollout file is gone must not invent a session -- and
        must not cost the fast path: coverage is one-directional, so a
        phantom cannot hide a real session nor turn discovery slow."""
        sessions_dir = tmp_path / ".codex" / "sessions" / "2026" / "04" / "10"
        f = self._write_rollout(sessions_dir, "cx-1", summary="hello", day="10")
        self._write_db(
            tmp_path,
            [
                {
                    "id": "cx-1",
                    "rollout_path": str(f),
                    "summary": "from the index",
                },
                {
                    "id": "cx-gone",
                    "rollout_path": str(
                        sessions_dir / "rollout-2026-04-10T13-00-00-cx-gone.jsonl"
                    ),
                    "summary": "deleted",
                },
            ],
        )

        with patch("session_browser.discovery.Path.home", return_value=tmp_path):
            sessions = scan_codex()

        assert [s.id for s in sessions] == ["cx-1"]
        # The DB summary proves the fast path ran despite the phantom.
        assert sessions[0].summary == "from the index"

    def test_falls_back_when_db_unreadable(self, tmp_path):
        """A locked/corrupt index degrades to the file scan, not to zero."""
        sessions_dir = tmp_path / ".codex" / "sessions" / "2026" / "04" / "10"
        self._write_rollout(sessions_dir, "cx-1", summary="hello", day="10")
        (tmp_path / ".codex" / "state_5.sqlite").write_text("not a database")

        with patch("session_browser.discovery.Path.home", return_value=tmp_path):
            sessions = scan_codex()

        assert {s.id for s in sessions} == {"cx-1"}

    def test_picks_highest_state_version(self, tmp_path):
        """state_10 beats state_9: the two-digit rollover is exactly the case
        a naive lexical max would get wrong, so this test pins the version
        parser, not string ordering."""
        sessions_dir = tmp_path / ".codex" / "sessions" / "2026" / "04" / "10"
        f = self._write_rollout(sessions_dir, "cx-1", summary="hello", day="10")
        self._write_db(
            tmp_path,
            [
                {
                    "id": "cx-1",
                    "rollout_path": str(f),
                    "summary": "from state_9",
                    "created_ms": 1775826000000,
                    "updated_ms": 1775826005000,
                }
            ],
        )
        (tmp_path / ".codex" / "state_5.sqlite").rename(
            tmp_path / ".codex" / "state_9.sqlite"
        )
        db10 = tmp_path / ".codex" / "state_10.sqlite"
        conn = sqlite3.connect(str(db10))
        conn.execute(
            "CREATE TABLE threads (id TEXT PRIMARY KEY, rollout_path TEXT NOT NULL, "
            "cwd TEXT NOT NULL, title TEXT NOT NULL DEFAULT '', "
            "git_branch TEXT, first_user_message TEXT NOT NULL DEFAULT '', "
            "created_at_ms INTEGER, updated_at_ms INTEGER, "
            "archived INTEGER NOT NULL DEFAULT 0)"
        )
        conn.execute(
            "INSERT INTO threads (id, rollout_path, cwd, git_branch, "
            "first_user_message, created_at_ms, updated_at_ms, archived) "
            "VALUES (?, ?, '', NULL, ?, ?, ?, 0)",
            ("cx-1", str(f), "from state_10", 1775826000000, 1775826005000),
        )
        conn.commit()
        conn.close()

        with patch("session_browser.discovery.Path.home", return_value=tmp_path):
            sessions = scan_codex()

        assert sessions[0].summary == "from state_10"

    def test_db_truth_wins_for_new_schema_sessions(self, tmp_path):
        """Codex's current schema emits no user_message events. The file scan
        then reports a blank summary and creation-time activity, while the
        index holds the human's message and true last activity -- the DB path
        must surface the truthful values rather than reproduce the file
        scan's.
        """
        sessions_dir = tmp_path / ".codex" / "sessions" / "2026" / "04" / "10"
        f = sessions_dir / "rollout-2026-04-10T13-00-00-cx-1.jsonl"
        f.parent.mkdir(parents=True)
        f.write_text(
            json.dumps(
                {
                    "type": "session_meta",
                    "timestamp": "2026-04-10T13:00:00Z",
                    "payload": {"id": "cx-1", "cwd": "/p", "git": {"branch": "main"}},
                }
            )
            + "\n"
            + json.dumps(
                {
                    "type": "response_item",
                    "timestamp": "2026-04-10T14:00:00Z",
                    "payload": {
                        "type": "message",
                        "role": "assistant",
                        "content": [{"type": "output_text", "text": "done"}],
                    },
                }
            )
            + "\n"
        )
        self._write_db(
            tmp_path,
            [
                {
                    "id": "cx-1",
                    "rollout_path": str(f),
                    "summary": "fix the login bug",
                    "created_ms": 1775826000000,
                    "updated_ms": 1775829600000,
                }
            ],
        )

        with patch("session_browser.discovery.Path.home", return_value=tmp_path):
            sessions = scan_codex()

        s = sessions[0]
        assert s.summary == "fix the login bug"
        assert s.updated_at == "2026-04-10T14:00:00.000Z"

    def test_archived_rows_are_excluded(self, tmp_path):
        """An archived thread whose rollout file is still in the tree must be
        excluded -- and the exclusion must not cost the fast path (the file
        scan cannot see `archived`, so it would have returned the session).
        """
        sessions_dir = tmp_path / ".codex" / "sessions" / "2026" / "04" / "10"
        f_live = self._write_rollout(sessions_dir, "cx-1", summary="live", day="10")
        f_arch = self._write_rollout(sessions_dir, "cx-arch", summary="old", day="11")
        self._write_db(
            tmp_path,
            [
                {
                    "id": "cx-1",
                    "rollout_path": str(f_live),
                    "summary": "live from db",
                    "created_ms": 1775826000000,
                    "updated_ms": 1775826005000,
                },
                {
                    "id": "cx-arch",
                    "rollout_path": str(f_arch),
                    "summary": "old",
                    "created_ms": 1775826000000,
                    "updated_ms": 1775826005000,
                    "archived": 1,
                },
            ],
        )

        with patch("session_browser.discovery.Path.home", return_value=tmp_path):
            sessions = scan_codex()

        # Both facts at once: the archived session is gone AND the live
        # summary came from the DB, proving the fast path was taken.
        assert [s.id for s in sessions] == ["cx-1"]
        assert sessions[0].summary == "live from db"

    def test_fallback_warns_once(self, tmp_path, caplog):
        """The one user-visible signal that the fast path turned off must
        actually be emitted -- and only once per message per process."""
        from session_browser import discovery

        monkeypatch = pytest.MonkeyPatch()
        monkeypatch.setattr(discovery, "_warned_codex_fallbacks", set())
        sessions_dir = tmp_path / ".codex" / "sessions" / "2026" / "04" / "10"
        self._write_rollout(sessions_dir, "cx-1", summary="hello", day="10")
        self._write_db(tmp_path, [])

        with (
            caplog.at_level("WARNING"),
            patch("session_browser.discovery.Path.home", return_value=tmp_path),
        ):
            assert {s.id for s in scan_codex()} == {"cx-1"}
            assert {s.id for s in scan_codex()} == {"cx-1"}

        misses = [r for r in caplog.records if "codex index is missing" in r.message]
        assert len(misses) == 1
        monkeypatch.undo()

    def test_db_and_file_paths_agree_on_stable_fields(self, tmp_path):
        """The two paths must converge on the fields where agreement is
        claimed: id, cwd, branch, created_at, updated_at, summary, path. A
        fixture carrying Codex's old schema (with a real user_message turn)
        exercises both and asserts equality field by field.
        """
        sessions_dir = tmp_path / ".codex" / "sessions" / "2026" / "04" / "10"
        sessions_dir.mkdir(parents=True)
        f = sessions_dir / "rollout-2026-04-10T13-00-00-cx-1.jsonl"
        f.write_text(
            json.dumps(
                {
                    "type": "session_meta",
                    "timestamp": "2026-04-10T13:00:00.000Z",
                    "payload": {
                        "id": "cx-1",
                        "cwd": "/p",
                        "git": {"branch": "main"},
                        "timestamp": "2026-04-10T13:00:00.000Z",
                    },
                }
            )
            + "\n"
            + json.dumps(
                {
                    "type": "event_msg",
                    "timestamp": "2026-04-10T13:00:05.000Z",
                    "payload": {"type": "user_message", "message": "hello there"},
                }
            )
            + "\n"
            + json.dumps(
                {
                    "type": "event_msg",
                    "timestamp": "2026-04-10T13:05:00.000Z",
                    "payload": {"type": "agent_message", "message": "done"},
                }
            )
            + "\n"
        )
        self._write_db(
            tmp_path,
            [
                {
                    "id": "cx-1",
                    "rollout_path": str(f),
                    "cwd": "/p",
                    "branch": "main",
                    "summary": "hello there",
                    "created_ms": 1775826000000,
                    "updated_ms": 1775826300000,
                }
            ],
        )

        with patch("session_browser.discovery.Path.home", return_value=tmp_path):
            from_db = scan_codex()
            from_files = _scan_codex_files()

        assert len(from_db) == len(from_files) == 1
        db_s, file_s = from_db[0], from_files[0]
        assert db_s.id == file_s.id == "cx-1"
        assert db_s.cwd == file_s.cwd == "/p"
        assert db_s.branch == file_s.branch == "main"
        assert db_s.created_at == file_s.created_at == "2026-04-10T13:00:00.000Z"
        assert db_s.updated_at == file_s.updated_at == "2026-04-10T13:05:00.000Z"
        assert db_s.summary == file_s.summary == "hello there"
        assert db_s.content_path == file_s.content_path == str(f)


class TestEpochMsToIso:
    def test_zulu_shape_matches_rollout_timestamps(self):
        assert _epoch_ms_to_iso(1775826000000, zulu=True) == "2026-04-10T13:00:00.000Z"
        assert _epoch_ms_to_iso(1775826000500, zulu=True) == "2026-04-10T13:00:00.500Z"

    def test_default_shape_is_isoformat(self):
        assert _epoch_ms_to_iso(1775826000000) == "2026-04-10T13:00:00+00:00"

    def test_invalid_inputs_degrade_to_empty(self):
        assert _epoch_ms_to_iso(None) == ""
        assert _epoch_ms_to_iso(0) == ""
        assert _epoch_ms_to_iso("") == ""
        assert _epoch_ms_to_iso("not-a-number") == ""
        assert _epoch_ms_to_iso(10**20) == ""


class TestLastActivity:
    """Direct tests of the reverse-read helper's edge behaviour."""

    def test_window_expands_for_deep_turn(self, tmp_path):
        """The last real turn sitting deeper than the initial 16KB tail must
        still be found via window expansion."""
        f = tmp_path / "big.jsonl"
        lines = [
            json.dumps(
                {
                    "type": "assistant",
                    "message": {"content": "the turn"},
                    "timestamp": "2026-06-29T12:00:00Z",
                }
            )
        ]
        # >16KB of trailing untimestamped metadata after the last real turn
        for _ in range(400):
            lines.append(json.dumps({"type": "last-prompt", "data": "x" * 60}))
        f.write_text("\n".join(lines) + "\n")
        assert f.stat().st_size > 16384
        assert _last_activity_iso(f, "claude") == "2026-06-29T12:00:00Z"

    def test_returns_empty_when_no_turn(self, tmp_path):
        f = tmp_path / "s.jsonl"
        f.write_text(
            json.dumps({"type": "system", "timestamp": "2026-06-30T08:00:00Z"}) + "\n"
        )
        assert _last_activity_iso(f, "claude") == ""

    def test_empty_and_missing_file(self, tmp_path):
        empty = tmp_path / "empty.jsonl"
        empty.write_text("")
        assert _last_activity_iso(empty, "claude") == ""
        assert _last_activity_iso(tmp_path / "nope.jsonl", "claude") == ""

    # Codex writes a turn under a different vocabulary per history mode, and
    # recognising only the legacy one cost more than a wrong timestamp: no
    # turn found means widening the window until the whole file has been read,
    # which on the real corpus was 93 MB read across 147 rollouts to return "".

    def test_codex_legacy_event(self, tmp_path):
        f = tmp_path / "legacy.jsonl"
        f.write_text(
            json.dumps(
                {
                    "type": "event_msg",
                    "timestamp": "2026-06-29T12:00:00Z",
                    "payload": {"type": "user_message", "message": "hi"},
                }
            )
            + "\n"
        )
        assert _last_activity_iso(f, "codex") == "2026-06-29T12:00:00Z"

    def test_codex_paginated_item_completed(self, tmp_path):
        f = tmp_path / "paginated.jsonl"
        f.write_text(
            "\n".join(
                json.dumps(
                    {
                        "type": "event_msg",
                        "timestamp": ts,
                        "payload": {
                            "type": "item_completed",
                            "item": {"type": kind, "id": "i"},
                        },
                    }
                )
                for ts, kind in (
                    ("2026-06-29T12:00:00Z", "UserMessage"),
                    # The assistant side counts too: recognising only the user
                    # side would timestamp a session whose last act was the
                    # reply at its previous user turn.
                    ("2026-06-29T12:00:09Z", "AgentMessage"),
                    ("2026-06-29T12:00:11Z", "CommandExecution"),
                )
            )
            + "\n"
        )
        assert _last_activity_iso(f, "codex") == "2026-06-29T12:00:09Z"

    def test_codex_response_item_when_no_event_vocabulary(self, tmp_path):
        f = tmp_path / "response.jsonl"
        f.write_text(
            "\n".join(
                json.dumps(
                    {
                        "type": "response_item",
                        "timestamp": ts,
                        "payload": {"type": "message", "role": role},
                    }
                )
                for ts, role in (
                    ("2026-06-29T12:00:00Z", "user"),
                    ("2026-06-29T12:00:09Z", "assistant"),
                    # A developer message is the system prompt, not a turn.
                    ("2026-06-29T12:00:11Z", "developer"),
                )
            )
            + "\n"
        )
        assert _last_activity_iso(f, "codex") == "2026-06-29T12:00:09Z"

    def test_codex_non_turn_records_are_still_ignored(self, tmp_path):
        f = tmp_path / "lifecycle.jsonl"
        f.write_text(
            "\n".join(
                json.dumps(record)
                for record in (
                    {
                        "type": "event_msg",
                        "timestamp": "2026-06-29T12:00:00Z",
                        "payload": {"type": "user_message", "message": "hi"},
                    },
                    {
                        "type": "response_item",
                        "timestamp": "2026-06-29T12:00:05Z",
                        "payload": {"type": "reasoning"},
                    },
                    {
                        "type": "turn_context",
                        "timestamp": "2026-06-29T12:00:07Z",
                        "payload": {"cwd": "/tmp"},
                    },
                    {
                        "type": "event_msg",
                        "timestamp": "2026-06-29T12:00:09Z",
                        "payload": {"type": "token_count", "total": 9},
                    },
                )
            )
            + "\n"
        )
        assert _last_activity_iso(f, "codex") == "2026-06-29T12:00:00Z"


class TestRepositoryName:
    """The value ``--repo`` matches, which was empty for every session on
    every provider until it was populated here."""

    @pytest.mark.parametrize(
        ("cwd", "expected"),
        [
            ("/Users/u/Projects/session-browser", "session-browser"),
            ("/Users/u/Projects/session-browser/", "session-browser"),
            ("session-browser", "session-browser"),
            ("", ""),
            ("/", ""),
            ("//", ""),
        ],
    )
    def test_repo_name_is_the_final_path_segment(self, cwd, expected):
        assert _repo_name(cwd) == expected

    def test_repo_name_never_touches_the_filesystem(self, tmp_path):
        """A cwd whose directory is long gone still names its project. 378 of
        1554 real sessions have one, and an existence check would blank them
        while also putting a stat in the discovery path."""
        assert _repo_name(str(tmp_path / "deleted-months-ago")) == "deleted-months-ago"


class TestScanOpencode:
    def _make_opencode_db(self, tmp_path):
        """Create a minimal opencode-style SQLite database."""
        db_path = tmp_path / ".local" / "share" / "opencode" / "opencode.db"
        db_path.parent.mkdir(parents=True)
        conn = sqlite3.connect(str(db_path))
        conn.execute("""CREATE TABLE project (
            id TEXT PRIMARY KEY, worktree TEXT NOT NULL,
            name TEXT, time_created INTEGER NOT NULL, time_updated INTEGER NOT NULL,
            sandboxes TEXT NOT NULL DEFAULT '[]'
        )""")
        conn.execute("""CREATE TABLE session (
            id TEXT PRIMARY KEY, project_id TEXT NOT NULL,
            slug TEXT NOT NULL DEFAULT '', directory TEXT NOT NULL,
            title TEXT NOT NULL DEFAULT '', version TEXT NOT NULL DEFAULT '1',
            time_created INTEGER NOT NULL, time_updated INTEGER NOT NULL
        )""")
        conn.execute("""CREATE TABLE message (
            id TEXT PRIMARY KEY, session_id TEXT NOT NULL,
            time_created INTEGER NOT NULL, time_updated INTEGER NOT NULL,
            data TEXT NOT NULL
        )""")
        conn.execute("""CREATE TABLE part (
            id TEXT PRIMARY KEY, message_id TEXT NOT NULL,
            session_id TEXT NOT NULL,
            time_created INTEGER NOT NULL, time_updated INTEGER NOT NULL,
            data TEXT NOT NULL
        )""")
        conn.execute(
            "INSERT INTO project VALUES (?, ?, ?, ?, ?, ?)",
            (
                "proj1",
                "/Users/test/myproject",
                "myproject",
                1700000000000,
                1700000000000,
                "[]",
            ),
        )
        conn.execute(
            "INSERT INTO session (id, project_id, slug, directory, title, version, "
            "time_created, time_updated) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (
                "ses_abc123",
                "proj1",
                "abc",
                "/Users/test/myproject",
                "Fix authentication bug",
                "1",
                1700000000000,
                1700001000000,
            ),
        )
        conn.commit()
        conn.close()
        return db_path

    def test_scan_opencode_sessions(self, tmp_path):
        self._make_opencode_db(tmp_path)

        with patch("session_browser.discovery.Path.home", return_value=tmp_path):
            sessions = scan_opencode()

        assert len(sessions) == 1
        assert sessions[0].id == "ses_abc123"
        assert sessions[0].provider == "opencode"
        assert sessions[0].summary == "Fix authentication bug"
        assert sessions[0].cwd == "/Users/test/myproject"
        assert sessions[0].repository == "myproject"
        assert sessions[0].branch is None

    def test_repository_falls_back_to_the_worktree_path(self, tmp_path):
        """``project.name`` is NULL for every row in a real install, so the
        path has to answer. The worktree is preferred over the session's own
        directory because it is the project root, and so still names the
        project for a session started in a subdirectory of it."""
        db_path = self._make_opencode_db(tmp_path)
        conn = sqlite3.connect(str(db_path))
        conn.execute("UPDATE project SET name = NULL")
        conn.execute("UPDATE session SET directory = '/Users/test/myproject/src'")
        conn.commit()
        conn.close()

        with patch("session_browser.discovery.Path.home", return_value=tmp_path):
            sessions = scan_opencode()

        assert sessions[0].repository == "myproject"
        assert sessions[0].cwd == "/Users/test/myproject/src"

    def test_repository_falls_past_the_global_projects_root_worktree(self, tmp_path):
        """Opencode files sessions with no project under a catch-all whose
        worktree is "/", which names nothing. Those fall through to their own
        directory rather than going blank -- 115 real sessions do."""
        db_path = self._make_opencode_db(tmp_path)
        conn = sqlite3.connect(str(db_path))
        conn.execute("UPDATE project SET name = NULL, worktree = '/'")
        conn.commit()
        conn.close()

        with patch("session_browser.discovery.Path.home", return_value=tmp_path):
            sessions = scan_opencode()

        assert sessions[0].repository == "myproject"

    def test_scan_opencode_no_db(self, tmp_path):
        with patch("session_browser.discovery.Path.home", return_value=tmp_path):
            sessions = scan_opencode()
        assert sessions == []

    def test_load_opencode_content(self, tmp_path):
        db_path = self._make_opencode_db(tmp_path)
        conn = sqlite3.connect(str(db_path))
        conn.execute(
            "INSERT INTO message VALUES (?, ?, ?, ?, ?)",
            (
                "msg1",
                "ses_abc123",
                1700000000100,
                1700000000100,
                json.dumps({"role": "user"}),
            ),
        )
        conn.execute(
            "INSERT INTO part VALUES (?, ?, ?, ?, ?, ?)",
            (
                "p1",
                "msg1",
                "ses_abc123",
                1700000000100,
                1700000000100,
                json.dumps({"type": "text", "text": "Fix the login page"}),
            ),
        )
        conn.execute(
            "INSERT INTO message VALUES (?, ?, ?, ?, ?)",
            (
                "msg2",
                "ses_abc123",
                1700000000200,
                1700000000200,
                json.dumps({"role": "assistant"}),
            ),
        )
        conn.execute(
            "INSERT INTO part VALUES (?, ?, ?, ?, ?, ?)",
            (
                "p2",
                "msg2",
                "ses_abc123",
                1700000000200,
                1700000000200,
                json.dumps({"type": "text", "text": "I'll fix the login page now."}),
            ),
        )
        conn.commit()
        conn.close()

        session = Session(
            id="ses_abc123", provider="opencode", content_path=str(db_path)
        )
        with patch("session_browser.discovery.Path.home", return_value=tmp_path):
            content = load_session_content(session)

        assert "Fix the login page" in content
        assert "I'll fix the login page" in content

    def test_search_opencode_content(self, tmp_path):
        db_path = self._make_opencode_db(tmp_path)
        conn = sqlite3.connect(str(db_path))
        conn.execute(
            "INSERT INTO message VALUES (?, ?, ?, ?, ?)",
            (
                "msg1",
                "ses_abc123",
                1700000000100,
                1700000000100,
                json.dumps({"role": "user"}),
            ),
        )
        conn.execute(
            "INSERT INTO part VALUES (?, ?, ?, ?, ?, ?)",
            (
                "p1",
                "msg1",
                "ses_abc123",
                1700000000100,
                1700000000100,
                json.dumps({"type": "text", "text": "debug adb shell issue"}),
            ),
        )
        conn.commit()
        conn.close()

        sessions = [
            Session(id="ses_abc123", provider="opencode", content_path=str(db_path))
        ]
        with patch("session_browser.discovery.Path.home", return_value=tmp_path):
            hits = search_session_contents(sessions, "adb")
        assert hits == {"ses_abc123"}

    def test_search_opencode_skips_step_parts(self, tmp_path):
        db_path = self._make_opencode_db(tmp_path)
        conn = sqlite3.connect(str(db_path))
        conn.execute(
            "INSERT INTO message VALUES (?, ?, ?, ?, ?)",
            (
                "msg1",
                "ses_abc123",
                1700000000100,
                1700000000100,
                json.dumps({"role": "assistant"}),
            ),
        )
        conn.execute(
            "INSERT INTO part VALUES (?, ?, ?, ?, ?, ?)",
            (
                "p1",
                "msg1",
                "ses_abc123",
                1700000000100,
                1700000000100,
                json.dumps({"type": "step-start", "snapshot": "adb123"}),
            ),
        )
        conn.commit()
        conn.close()

        sessions = [
            Session(id="ses_abc123", provider="opencode", content_path=str(db_path))
        ]
        with patch("session_browser.discovery.Path.home", return_value=tmp_path):
            hits = search_session_contents(sessions, "adb")
        assert hits == set()

    def test_resume_opencode(self):
        cmd = resume_command("opencode", "ses_abc123")
        assert cmd == "opencode -s ses_abc123"

    def test_resume_opencode_with_cwd(self):
        cmd = resume_command("opencode", "ses_abc123", "/Users/test/project")
        assert cmd == "cd /Users/test/project && opencode -s ses_abc123"


# ---------------------------------------------------------------------------
# Aggregate discovery: concurrent scanners, optional provider restriction
# ---------------------------------------------------------------------------


class TestDiscoverAll:
    def _patched_scanners(self, monkeypatch, calls):
        def scanner(name, sessions):
            def run():
                calls.append(name)
                return sessions

            return run

        from session_browser import discovery

        monkeypatch.setattr(
            discovery,
            "ALL_SCANNERS",
            {
                "claude": scanner(
                    "claude",
                    [
                        Session(
                            id="c1",
                            provider="claude",
                            updated_at="2026-06-02T00:00:00+00:00",
                        )
                    ],
                ),
                "codex": scanner(
                    "codex",
                    [
                        Session(
                            id="x1",
                            provider="codex",
                            updated_at="2026-06-05T00:00:00+00:00",
                        )
                    ],
                ),
            },
        )

    def test_all_scanners_run_and_results_sorted(self, monkeypatch):
        calls: list[str] = []
        self._patched_scanners(monkeypatch, calls)
        sessions = discover_all()
        assert sorted(calls) == ["claude", "codex"]
        assert [s.id for s in sessions] == ["x1", "c1"]  # newest first

    def test_providers_restricts_scan(self, monkeypatch):
        calls: list[str] = []
        self._patched_scanners(monkeypatch, calls)
        sessions = discover_all(providers=["claude"])
        assert calls == ["claude"]
        assert [s.id for s in sessions] == ["c1"]

    def test_provider_names_case_insensitive(self, monkeypatch):
        calls: list[str] = []
        self._patched_scanners(monkeypatch, calls)
        sessions = discover_all(providers=["Codex"])
        assert calls == ["codex"]
        assert [s.id for s in sessions] == ["x1"]

    def test_unknown_provider_scans_nothing(self, monkeypatch):
        calls: list[str] = []
        self._patched_scanners(monkeypatch, calls)
        assert discover_all(providers=["nope"]) == []
        assert calls == []

    def test_one_failing_scanner_does_not_break_the_rest(self, monkeypatch):
        from session_browser import discovery

        def boom():
            raise RuntimeError("scanner exploded")

        monkeypatch.setattr(
            discovery,
            "ALL_SCANNERS",
            {
                "claude": boom,
                "codex": lambda: [Session(id="ok", provider="codex")],
            },
        )
        sessions = discover_all()
        assert [s.id for s in sessions] == ["ok"]


# ---------------------------------------------------------------------------
# Cross-provider service search (replaces the batched-DB helper tests)
# ---------------------------------------------------------------------------


class TestServiceSearchAcrossProviders:
    def _opencode_db_two_sessions(self, tmp_path):
        """opencode DB with two sessions; only ses_b mentions 'kangaroo'."""
        db_path = tmp_path / ".local" / "share" / "opencode" / "opencode.db"
        db_path.parent.mkdir(parents=True)
        conn = sqlite3.connect(str(db_path))
        conn.execute(
            "CREATE TABLE message (id TEXT PRIMARY KEY, session_id TEXT NOT NULL, "
            "time_created INTEGER NOT NULL, time_updated INTEGER NOT NULL, data TEXT NOT NULL)"
        )
        conn.execute(
            "CREATE TABLE part (id TEXT PRIMARY KEY, message_id TEXT NOT NULL, "
            "session_id TEXT NOT NULL, time_created INTEGER NOT NULL, "
            "time_updated INTEGER NOT NULL, data TEXT NOT NULL)"
        )
        rows = [
            (
                "ses_a",
                "ma",
                "pa",
                {"type": "text", "text": "ordinary platypus discussion"},
            ),
            ("ses_b", "mb", "pb", {"type": "text", "text": "the kangaroo hops"}),
            # a step-start part on ses_a containing 'kangaroo' must NOT match —
            # it isn't rendered, so it's unreachable via in-session search.
            (
                "ses_a",
                "ma2",
                "pa2",
                {"type": "step-start", "snapshot": "kangaroo-snap"},
            ),
        ]
        for sid, mid, pid, part in rows:
            conn.execute(
                "INSERT INTO message VALUES (?, ?, ?, ?, ?)",
                (mid, sid, 1, 1, json.dumps({"role": "user"})),
            )
            conn.execute(
                "INSERT INTO part VALUES (?, ?, ?, ?, ?, ?)",
                (pid, mid, sid, 1, 1, json.dumps(part)),
            )
        conn.commit()
        conn.close()
        return db_path

    def test_opencode_search_isolates_sessions(self, tmp_path):
        self._opencode_db_two_sessions(tmp_path)
        sessions = [
            Session(id="ses_a", provider="opencode"),
            Session(id="ses_b", provider="opencode"),
        ]
        with patch("session_browser.discovery.Path.home", return_value=tmp_path):
            hits = search_session_contents(sessions, "kangaroo")
        # Only ses_b has 'kangaroo' in a *rendered* part. ses_a's match is in a
        # step-start snapshot, which is never shown — so it must not match.
        assert hits == {"ses_b"}

    def test_search_unions_across_providers(self, tmp_path):
        """search_session_contents merges hits from DB and file providers."""
        self._opencode_db_two_sessions(tmp_path)  # ses_b -> 'kangaroo'
        claude_file = tmp_path / "claude.jsonl"
        claude_file.write_text(
            json.dumps(
                {"type": "user", "message": {"content": "where do kangaroo live"}}
            )
            + "\n"
        )
        sessions = [
            Session(id="ses_a", provider="opencode"),
            Session(id="ses_b", provider="opencode"),
            Session(id="cl_1", provider="claude", content_path=str(claude_file)),
        ]
        with patch("session_browser.discovery.Path.home", return_value=tmp_path):
            hits = search_session_contents(sessions, "kangaroo")
        assert hits == {"ses_b", "cl_1"}


# ---------------------------------------------------------------------------
# Global search tests
# ---------------------------------------------------------------------------


class TestGlobalSearch:
    def test_empty_query_returns_all(self):
        sessions = [
            Session(id="1", provider="codex", summary="A"),
            Session(id="2", provider="claude", summary="B"),
        ]
        filtered = [s for s in sessions if s.matches("")]
        assert len(filtered) == 2

    def test_query_narrows_results(self):
        sessions = [
            Session(id="1", provider="codex", summary="Fix CSS"),
            Session(id="2", provider="claude", summary="Add auth"),
        ]
        filtered = [s for s in sessions if s.matches("css")]
        assert len(filtered) == 1
        assert filtered[0].summary == "Fix CSS"


class TestContentSearch:
    def _write_claude_jsonl(self, path, user_text):
        path.write_text(
            json.dumps(
                {
                    "type": "user",
                    "message": {"content": user_text},
                }
            )
            + "\n"
        )

    def test_finds_matches_in_rendered_messages(self, tmp_path):
        """A query that appears only inside rendered session messages —
        not metadata — should still return the session. Regression for
        the bug where 'adb' only returned sessions whose title contained it."""
        f1 = tmp_path / "s1.jsonl"
        self._write_claude_jsonl(f1, "connect via adb shell")
        f2 = tmp_path / "s2.jsonl"
        self._write_claude_jsonl(f2, "unrelated log")
        f3 = tmp_path / "s3.jsonl"
        self._write_claude_jsonl(f3, "ADB uppercase works too")
        sessions = [
            Session(id="1", provider="claude", summary="x", content_path=str(f1)),
            Session(id="2", provider="claude", summary="x", content_path=str(f2)),
            Session(id="3", provider="claude", summary="x", content_path=str(f3)),
        ]
        hits = search_session_contents(sessions, "adb")
        assert hits == {"1", "3"}

    def test_empty_query_returns_empty(self, tmp_path):
        f = tmp_path / "s.jsonl"
        self._write_claude_jsonl(f, "anything")
        sessions = [Session(id="1", provider="claude", content_path=str(f))]
        assert search_session_contents(sessions, "") == set()
        assert search_session_contents(sessions, "   ") == set()


# ---------------------------------------------------------------------------
# In-session search navigation
# ---------------------------------------------------------------------------


class TestInSessionSearch:
    def test_find_matches(self):
        text = "hello world hello there hello"
        query = "hello"
        q_lower = query.lower()
        t_lower = text.lower()
        matches = []
        start = 0
        while True:
            idx = t_lower.find(q_lower, start)
            if idx == -1:
                break
            matches.append(idx)
            start = idx + 1
        assert matches == [0, 12, 24]

    def test_navigation_wraps(self):
        matches = [0, 10, 20]
        idx = 2
        idx = (idx + 1) % len(matches)
        assert idx == 0  # wraps to start
        idx = (idx - 1) % len(matches)
        assert idx == 2  # wraps to end


# ---------------------------------------------------------------------------
# Clipboard command formatting
# ---------------------------------------------------------------------------

try:
    from session_browser.app import (
        MultiplexerChoice,
        SearchInput,
        SessionBrowser,
        SessionTable,
        ShortcutHelp,
    )

    _HAS_TEXTUAL = True
except ImportError:
    _HAS_TEXTUAL = False


@pytest.mark.skipif(not _HAS_TEXTUAL, reason="textual not installed")
class TestKeyBindings:
    def test_app_has_vim_navigation(self):
        keys = {b.key for b in SessionBrowser.BINDINGS}
        # vim-style navigation
        assert {"j", "k", "g", "G"}.issubset(keys)
        # half-page scroll
        assert {"ctrl+d", "ctrl+u"}.issubset(keys)
        # uppercase N for prev match (replaces shift+n for display consistency)
        assert "N" in keys
        # core commands still present
        assert {
            "q",
            "/",
            "s",
            "tab",
            "left",
            "right",
            "c",
            "n",
            "r",
            "escape",
            "z",
            "?",
        }.issubset(keys)
        # export commands
        assert {"e", "E"}.issubset(keys)

    def test_copy_session_id_binding_present(self):
        # `i` (mnemonic: id) copies the selected session's canonical id.
        binding = next((b for b in SessionBrowser.BINDINGS if b.key == "i"), None)
        assert binding is not None
        assert binding.action == "copy_session_id"

    def test_search_input_has_flow_bindings(self):
        keys = {b.key for b in SearchInput.BINDINGS}
        # clear/blur
        assert {"escape", "ctrl+u"}.issubset(keys)
        # flow-out and fzf-style result nav
        assert {"down", "ctrl+n", "ctrl+p"}.issubset(keys)

    def test_command_palette_disabled(self):
        # ctrl+p must belong to SearchInput (prev result), not Textual's command palette
        assert SessionBrowser.ENABLE_COMMAND_PALETTE is False

    def test_session_table_overrides_cursor_up(self):
        # Sanity: SessionTable subclasses DataTable and overrides action_cursor_up
        from textual.widgets import DataTable

        assert issubclass(SessionTable, DataTable)
        assert SessionTable.action_cursor_up is not DataTable.action_cursor_up

    @staticmethod
    def _tmux_action_app():
        updates = []
        exits = []
        app = SimpleNamespace(
            _selected=Session(id="abc", provider="claude", cwd="/tmp/proj"),
            _status=SimpleNamespace(update=updates.append),
            exit=lambda: exits.append(True),
            _run_handoff=SessionBrowser._run_handoff,
        )
        app._open_in_multiplexer = lambda target: SessionBrowser._open_in_multiplexer(
            app, target
        )
        plan = SimpleNamespace(
            reused=False,
            label="proj",
            switch_commands=lambda: [["sesh", "connect", "--switch", "proj"]],
            attach_commands=lambda: [["sesh", "connect", "proj"]],
        )
        return app, plan, updates, exits

    @staticmethod
    def _tmux_target(*, inside: bool = True):
        """A tmux target whose "are we inside it" answer is fixed."""
        return multiplexer.Target("tmux", "t", tmux, TmuxError, lambda: inside)

    def test_popup_marker_exits_after_successful_tmux_switch(self, monkeypatch):
        app, plan, updates, exits = self._tmux_action_app()
        monkeypatch.setenv("SESSION_BROWSER_CLOSE_ON_TMUX_SWITCH", "1")
        monkeypatch.setattr(tmux, "prepare_session", lambda *a, **kw: plan)
        monkeypatch.setattr(
            subprocess, "run", lambda cmd, **kw: subprocess.CompletedProcess(cmd, 0)
        )

        SessionBrowser._open_in_multiplexer(app, self._tmux_target())

        assert exits == [True]
        assert updates == []

    def test_normal_launch_stays_open_after_tmux_switch(self, monkeypatch):
        app, plan, updates, exits = self._tmux_action_app()
        monkeypatch.delenv("SESSION_BROWSER_CLOSE_ON_TMUX_SWITCH", raising=False)
        monkeypatch.setattr(tmux, "prepare_session", lambda *a, **kw: plan)
        monkeypatch.setattr(
            subprocess, "run", lambda cmd, **kw: subprocess.CompletedProcess(cmd, 0)
        )

        SessionBrowser._open_in_multiplexer(app, self._tmux_target())

        assert exits == []
        assert updates == ["Switched to tmux session 'proj'"]

    def test_dismissed_chooser_opens_nothing(self, monkeypatch):
        # The chooser's callback fires with None when it was cancelled.
        app, _plan, updates, exits = self._tmux_action_app()
        monkeypatch.setattr(
            tmux, "prepare_session", lambda *a, **kw: pytest.fail("must not prepare")
        )

        SessionBrowser._open_in_multiplexer(app, None)

        assert (updates, exits) == ([], [])

    def test_handoff_failure_is_reported_under_the_target_name(self, monkeypatch):
        app, _plan, updates, _exits = self._tmux_action_app()
        monkeypatch.setattr(
            tmux,
            "prepare_session",
            lambda *a, **kw: (_ for _ in ()).throw(TmuxError("sesh not found")),
        )

        SessionBrowser._open_in_multiplexer(app, self._tmux_target())

        assert updates == ["tmux: sesh not found"]

    def test_handoff_runs_every_command_in_order(self, monkeypatch):
        # herdr's attach is focus-then-attach, so a plan may need two commands.
        calls = []
        monkeypatch.setattr(
            subprocess,
            "run",
            lambda cmd, **kw: (calls.append(cmd), subprocess.CompletedProcess(cmd, 0))[
                1
            ],
        )

        rc = SessionBrowser._run_handoff(
            [["herdr", "tab", "focus", "w1:t2"], ["herdr"]]
        )

        assert rc == 0
        assert calls == [["herdr", "tab", "focus", "w1:t2"], ["herdr"]]

    def test_handoff_stops_at_the_first_failure(self, monkeypatch):
        # Attaching a client to a focus that failed would land somewhere else.
        calls = []
        monkeypatch.setattr(
            subprocess,
            "run",
            lambda cmd, **kw: (calls.append(cmd), subprocess.CompletedProcess(cmd, 1))[
                1
            ],
        )

        rc = SessionBrowser._run_handoff(
            [["herdr", "tab", "focus", "w1:t2"], ["herdr"]]
        )

        assert rc == 1
        assert calls == [["herdr", "tab", "focus", "w1:t2"]]

    def test_no_multiplexer_running_says_so(self, monkeypatch):
        app, _plan, updates, _exits = self._tmux_action_app()
        app.push_screen = lambda *a, **kw: pytest.fail("must not ask")
        monkeypatch.setattr(multiplexer, "available_targets", list)

        SessionBrowser.action_open_terminal(app)

        assert updates == ["No multiplexer running (tmux, herdr)"]

    def test_single_running_multiplexer_needs_no_choice(self, monkeypatch):
        app, plan, updates, _exits = self._tmux_action_app()
        app.push_screen = lambda *a, **kw: pytest.fail("must not ask")
        monkeypatch.setattr(
            multiplexer, "available_targets", lambda: [self._tmux_target()]
        )
        monkeypatch.setattr(tmux, "prepare_session", lambda *a, **kw: plan)
        monkeypatch.setattr(
            subprocess, "run", lambda cmd, **kw: subprocess.CompletedProcess(cmd, 0)
        )

        SessionBrowser.action_open_terminal(app)

        assert updates == ["Switched to tmux session 'proj'"]

    def test_two_running_multiplexers_are_offered_as_a_choice(self, monkeypatch):
        app, _plan, updates, _exits = self._tmux_action_app()
        pushed = []
        app.push_screen = lambda screen, callback: pushed.append((screen, callback))
        targets = [self._tmux_target(), multiplexer.TARGETS[1]]
        monkeypatch.setattr(multiplexer, "available_targets", lambda: targets)
        monkeypatch.setattr(
            tmux,
            "prepare_session",
            lambda *a, **kw: pytest.fail("must not prepare yet"),
        )

        SessionBrowser.action_open_terminal(app)

        assert updates == []
        screen, callback = pushed[0]
        assert isinstance(screen, MultiplexerChoice)
        # The choice is handed straight back to the opener.
        assert callback == app._open_in_multiplexer

    def test_help_names_only_the_running_multiplexers(self):
        entry = ShortcutHelp(["herdr"])._terminal_entry()
        assert "open in herdr" in entry
        assert "tmux" not in entry

    def test_help_names_both_when_both_run(self):
        assert (
            "open in tmux / herdr" in ShortcutHelp(["tmux", "herdr"])._terminal_entry()
        )

    def test_help_reports_when_nothing_is_running(self):
        # Dimmed and explained, not silently dropped: a missing row reads as
        # a bug, the reason reads as the state.
        entry = ShortcutHelp([])._terminal_entry()
        assert "no live multiplexer" in entry
        assert "[dim]" in entry

    def test_help_terminal_entry_keeps_the_second_column_aligned(self):
        for names in ([], ["tmux"], ["herdr"], ["tmux", "herdr"]):
            entry = ShortcutHelp(names)._terminal_entry()
            visible = re.sub(r"\[/?[a-z ]*\]", "", entry)
            assert len(visible) == ShortcutHelp._USE_COLUMN, names


def _make_hit(
    count: int = 1,
    role: str = "user",
    before: str = "",
    match: str = "auth",
    after: str = "",
) -> ContentHit:
    return ContentHit(count, role, before, match, after)


def _make_app_with_rows(n: int = 5):
    app = SessionBrowser()
    fake = [
        Session(
            id=f"s{i}",
            provider="claude",
            summary=f"session {i}",
            updated_at=f"2026-01-{i:02d}",
        )
        for i in range(1, n + 1)
    ]
    return app, fake


async def _install_fake_sessions(app, pilot, fake):
    """Populate the app with fake sessions and let the UI settle.

    Wait for the on-mount discovery worker (which scans the real machine for
    sessions) to finish *before* installing fakes — otherwise it can complete
    mid-test, rebuild the table and reset the cursor, swallowing keypresses.
    """
    await app.workers.wait_for_complete()
    app._all_sessions = fake
    app._apply_filter("")
    await pilot.pause()


@pytest.mark.skipif(not _HAS_TEXTUAL, reason="textual not installed")
@pytest.mark.asyncio
class TestKeyboardFlow:
    """End-to-end flow tests using Textual's Pilot."""

    async def test_down_arrow_from_search_focuses_list(self):
        app, fake = _make_app_with_rows()
        async with app.run_test() as pilot:
            await _install_fake_sessions(app, pilot, fake)
            await pilot.press("/")
            await pilot.press("down")
            await pilot.pause()
            assert app.query_one("#session-table").has_focus

    async def test_up_at_first_row_focuses_search(self):
        app, fake = _make_app_with_rows()
        async with app.run_test() as pilot:
            await _install_fake_sessions(app, pilot, fake)
            table = app.query_one("#session-table")
            table.focus()
            await pilot.pause()
            assert table.cursor_row == 0
            await pilot.press("up")
            await pilot.pause()
            assert app.query_one("#global-search").has_focus
            assert table.cursor_row == 0

    async def test_ctrl_n_p_browses_from_search(self):
        app, fake = _make_app_with_rows()
        async with app.run_test() as pilot:
            await _install_fake_sessions(app, pilot, fake)
            search = app.query_one("#global-search")
            table = app.query_one("#session-table")
            await pilot.press("/")
            for _ in range(3):
                await pilot.press("ctrl+n")
            await pilot.pause()
            assert search.has_focus, "search stays focused during ^N browsing"
            assert table.cursor_row == 3
            await pilot.press("ctrl+p")
            await pilot.pause()
            assert table.cursor_row == 2

    async def test_escape_clears_global_search_then_toggles_to_list(self):
        """Escape clears a query first; only an empty search loses focus."""
        app, fake = _make_app_with_rows()
        async with app.run_test() as pilot:
            await _install_fake_sessions(app, pilot, fake)
            search = app.query_one("#global-search")
            table = app.query_one("#session-table")
            await pilot.press("/")
            await pilot.press("a", "b", "c")
            await pilot.pause()
            assert search.value == "abc"
            await pilot.press("escape")
            await pilot.pause()
            assert search.has_focus
            assert search.value == ""
            assert app._filter_query == ""
            assert app._filtered == fake
            await pilot.press("escape")
            await pilot.pause()
            assert table.has_focus

    async def test_escape_cancels_in_progress_global_search(self):
        import asyncio
        import threading
        import time

        app, fake = _make_app_with_rows()
        started = threading.Event()
        stopped = threading.Event()

        def slow_search(sessions, query, *, cache, cancelled, progress=None):
            started.set()
            while not cancelled():
                time.sleep(0.001)
            stopped.set()
            return {}

        async with app.run_test() as pilot:
            await _install_fake_sessions(app, pilot, fake)
            search = app.query_one("#global-search")
            search.focus()
            search.value = "needle"
            with patch(
                "session_browser.app.search_session_hits", side_effect=slow_search
            ):
                app._apply_filter("needle")
                assert await asyncio.to_thread(started.wait, 1)
                await pilot.press("escape")
                assert await asyncio.to_thread(stopped.wait, 1)
                await pilot.pause()

            assert search.value == ""
            assert search.has_focus
            assert search.spinner_glyph is None
            assert app._filter_query == ""

    async def test_escape_in_detail_search_still_blurs(self):
        """Detail search is not the root; empty-Escape should still blur to list."""
        app, fake = _make_app_with_rows()
        async with app.run_test() as pilot:
            await _install_fake_sessions(app, pilot, fake)
            table = app.query_one("#session-table")
            detail_search = app.query_one("#detail-search")
            table.focus()
            await pilot.pause()
            await pilot.press("s")
            await pilot.pause()
            assert detail_search.has_focus
            await pilot.press("escape")  # empty → blur to table
            await pilot.pause()
            assert table.has_focus

    async def test_enter_on_row_focuses_detail_scroll(self):
        app, fake = _make_app_with_rows()
        async with app.run_test() as pilot:
            await _install_fake_sessions(app, pilot, fake)
            app.query_one("#session-table").focus()
            await pilot.pause()
            await pilot.press("enter")
            await pilot.pause()
            assert app.query_one("#detail-scroll").has_focus

    async def test_escape_from_list_focuses_global_search(self):
        app, fake = _make_app_with_rows()
        async with app.run_test() as pilot:
            await _install_fake_sessions(app, pilot, fake)
            table = app.query_one("#session-table")
            table.focus()
            await pilot.pause()
            await pilot.press("escape")
            await pilot.pause()
            assert app.query_one("#global-search").has_focus

    async def test_escape_unwinds_detail_to_list_to_search(self):
        """Escape should step up one level each press: detail → list → search."""
        app, fake = _make_app_with_rows()
        async with app.run_test() as pilot:
            await _install_fake_sessions(app, pilot, fake)
            table = app.query_one("#session-table")
            table.focus()
            await pilot.pause()
            # Simulate being in the detail pane (Enter lands on detail-search, not
            # scroll, so drive focus directly to mimic scrolling through content)
            app.query_one("#detail-scroll").focus()
            await pilot.pause()
            await pilot.press("escape")
            await pilot.pause()
            assert table.has_focus, "first escape: detail → list"
            await pilot.press("escape")
            await pilot.pause()
            assert app.query_one("#global-search").has_focus, (
                "second escape: list → search"
            )

    async def test_enter_then_escape_then_enter_repeats(self):
        """Guard against regressions: Enter on the same row must keep working."""
        app, fake = _make_app_with_rows()
        async with app.run_test() as pilot:
            await _install_fake_sessions(app, pilot, fake)
            table = app.query_one("#session-table")
            table.focus()
            await pilot.pause()
            detail_scroll = app.query_one("#detail-scroll")
            for _ in range(3):
                await pilot.press("enter")
                await pilot.pause()
                assert detail_scroll.has_focus, "Enter should focus transcript"
                await pilot.press("escape")
                await pilot.pause()
                assert table.has_focus, "Escape should hop back to table"

    async def test_contextual_slash_and_fast_typing_are_safe(self):
        app, fake = _make_app_with_rows()
        async with app.run_test() as pilot:
            await _install_fake_sessions(app, pilot, fake)
            table = app.query_one("#session-table")
            table.focus()
            await pilot.press("/", "s", "p", "e", "a", "k", "e", "r")
            await pilot.pause()
            assert app.query_one("#global-search").value == "speaker"
            assert app.is_running

            app.query_one("#global-search").value = ""
            app._apply_filter("")
            app._select_session("0")
            app.query_one("#detail-scroll").focus()
            await pilot.press("/", "n", "e", "e", "d", "l", "e")
            await pilot.pause()
            assert app.query_one("#detail-search").value == "needle"

    async def test_stale_debounce_does_not_clobber_direct_filter(self):
        """A pending debounce must not resurrect a query the box no longer shows.

        Typing schedules a debounced filter; if the search box is then reset
        and a filter applied directly, the old timer can still fire before the
        Changed("") message (which would reschedule it) is processed. Firing
        the callback by hand emulates that loaded-machine interleaving — it
        used to apply the stale query, empty the table, and clear the
        selection, so the contextual `/` above found nothing to search.
        """
        app, fake = _make_app_with_rows()
        async with app.run_test() as pilot:
            await _install_fake_sessions(app, pilot, fake)
            app.query_one("#session-table").focus()
            await pilot.press("/", "s", "p", "e", "a", "k", "e", "r")
            await pilot.pause()
            app.query_one("#global-search").value = ""
            app._apply_filter("")
            app._select_session("0")
            assert app._pending_query == "speaker", "debounce still pending"
            app._run_pending_filter()
            assert app._filter_query == "", "stale query must not be applied"
            assert app._selected is not None, "stale query must not clear selection"

    async def test_gg_chord_jumps_list_to_top(self):
        from textual.widgets import DataTable

        app, fake = _make_app_with_rows(5)
        async with app.run_test() as pilot:
            await _install_fake_sessions(app, pilot, fake)
            table = app.query_one("#session-table", DataTable)
            app.screen.set_focus(table)
            table.move_cursor(row=4)
            await pilot.press("g")
            assert table.cursor_row == 4, "single g only arms the chord"
            await pilot.press("j")  # any other key cancels the pending chord
            await pilot.press("g")
            assert table.cursor_row == 4, "chord was cancelled, g re-arms"
            await pilot.press("g")
            assert table.cursor_row == 0

    async def test_tab_and_arrows_switch_master_detail_panels(self):
        app, fake = _make_app_with_rows()
        async with app.run_test() as pilot:
            await _install_fake_sessions(app, pilot, fake)
            table = app.query_one("#session-table")
            table.focus()
            await pilot.press("tab")
            assert app.query_one("#detail-scroll").has_focus
            await pilot.press("left")
            assert table.has_focus
            await pilot.press("right")
            assert app.query_one("#detail-scroll").has_focus


@pytest.mark.skipif(not _HAS_TEXTUAL, reason="textual not installed")
@pytest.mark.asyncio
class TestResponsiveLayout:
    """Breakpoint composition should follow the terminal, not crush panes."""

    @staticmethod
    def _column_labels(app):
        table = app.query_one("#session-table")
        return [str(column.label) for column in table.columns.values()]

    async def test_compact_layout_drills_between_full_width_panes(self):
        app, fake = _make_app_with_rows()
        async with app.run_test(size=(88, 28)) as pilot:
            await _install_fake_sessions(app, pilot, fake)
            left = app.query_one("#left-pane")
            right = app.query_one("#right-pane")

            assert app.screen.has_class("-compact")
            assert left.display and not right.display
            assert left.region.width >= 84
            assert self._column_labels(app) == ["Provider", "Project", "Age", "Summary"]

            await pilot.press("enter")
            await pilot.pause()
            assert not left.display and right.display
            assert app.query_one("#detail-scroll").has_focus

            await pilot.press("escape")
            await pilot.pause()
            assert left.display and not right.display
            assert app.query_one("#session-table").has_focus

    async def test_standard_layout_uses_two_panes_and_compact_columns(self):
        app, fake = _make_app_with_rows()
        async with app.run_test(size=(120, 34)) as pilot:
            await _install_fake_sessions(app, pilot, fake)
            assert app.screen.has_class("-standard")
            assert app.query_one("#left-pane").display
            assert app.query_one("#right-pane").display
            assert self._column_labels(app) == ["Provider", "Project", "Age"]

    async def test_focus_mode_expands_and_switches_the_active_pane(self):
        app, fake = _make_app_with_rows()
        async with app.run_test(size=(180, 48)) as pilot:
            await _install_fake_sessions(app, pilot, fake)
            left = app.query_one("#left-pane")
            right = app.query_one("#right-pane")
            assert app.screen.has_class("-wide")
            assert left.display and right.display
            assert left.region.width <= 76

            await pilot.press("z")
            await pilot.pause()
            assert left.display and not right.display
            assert left.region.width >= 176

            await pilot.press("tab")
            await pilot.pause()
            assert not left.display and right.display
            assert app.query_one("#detail-scroll").has_focus

            await pilot.press("z")
            await pilot.pause()
            assert left.display and right.display

    async def test_micro_short_layout_removes_nonessential_chrome(self):
        from textual.widgets import Footer

        app, fake = _make_app_with_rows()
        async with app.run_test(size=(68, 22)) as pilot:
            await _install_fake_sessions(app, pilot, fake)
            assert app.screen.has_class("-micro")
            assert app.screen.has_class("-short")
            assert app.query_one("#app-bar").region.height == 1
            assert not app.query_one("#sessions-title").display
            assert not app.query_one("#sessions-header").display
            assert not app.query_one(Footer).display
            assert app.query_one("#status-bar").display

    async def test_live_resize_reflows_active_pane_and_context_hints(self):
        app, fake = _make_app_with_rows()
        async with app.run_test(size=(180, 48)) as pilot:
            await _install_fake_sessions(app, pilot, fake)
            await pilot.press("tab")
            await pilot.pause()

            await pilot.resize_terminal(88, 28)
            await pilot.pause()
            assert app.screen.has_class("-compact")
            assert not app.query_one("#left-pane").display
            assert app.query_one("#right-pane").display
            compact_status = str(app._status.render())
            assert "tab switch" in compact_status
            assert "z focus" not in compact_status

            await pilot.resize_terminal(180, 48)
            await pilot.pause()
            assert app.screen.has_class("-wide")
            assert app.query_one("#left-pane").display
            assert app.query_one("#right-pane").display
            assert "z focus" in str(app._status.render())

    async def test_help_overlay_opens_and_closes(self):
        from session_browser.app import ShortcutHelp

        app, fake = _make_app_with_rows()
        async with app.run_test(size=(88, 28)) as pilot:
            await _install_fake_sessions(app, pilot, fake)
            await pilot.press("?")
            await pilot.pause()
            assert isinstance(app.screen, ShortcutHelp)
            await pilot.press("escape")
            await pilot.pause()
            assert not isinstance(app.screen, ShortcutHelp)


@pytest.mark.skipif(not _HAS_TEXTUAL, reason="textual not installed")
@pytest.mark.asyncio
class TestCopySessionId:
    """`i` copies the selected session's canonical id for handing to an agent."""

    async def test_i_copies_canonical_id_to_clipboard(self):
        app, fake = _make_app_with_rows()
        async with app.run_test() as pilot:
            await _install_fake_sessions(app, pilot, fake)
            table = app.query_one("#session-table")
            table.focus()
            await pilot.pause()
            table.move_cursor(row=1)  # second session: id "s2"
            await pilot.pause()
            assert app._selected is not None and app._selected.id == "s2"
            captured: list[str] = []
            with patch(
                "session_browser.app.copy_to_clipboard",
                side_effect=lambda t: captured.append(t) or True,
            ):
                await pilot.press("i")
                await pilot.pause()
            # Canonical id is the exact handle `session-browser get` resolves.
            assert captured == ["claude:s2"]
            assert "claude:s2" in str(app._status.render())
            # Visual confirmation: the status bar pulses success-styled.
            assert app._status.has_class("-flash-ok")
            assert not app._status.has_class("-flash-err")

    async def test_copy_id_failure_flashes_error(self):
        app, fake = _make_app_with_rows()
        async with app.run_test() as pilot:
            await _install_fake_sessions(app, pilot, fake)
            table = app.query_one("#session-table")
            table.focus()
            await pilot.pause()
            table.move_cursor(row=1)
            await pilot.pause()
            assert app._selected is not None
            with patch("session_browser.app.copy_to_clipboard", return_value=False):
                await pilot.press("i")
                await pilot.pause()
            # Clipboard failed → error-styled pulse, id still shown to copy by hand.
            assert app._status.has_class("-flash-err")
            assert not app._status.has_class("-flash-ok")
            assert "Clipboard failed" in str(app._status.render())

    async def test_copy_id_without_selection_reports_no_session(self):
        app, fake = _make_app_with_rows()
        async with app.run_test() as pilot:
            await _install_fake_sessions(app, pilot, fake)
            app._selected = None
            with patch("session_browser.app.copy_to_clipboard") as mock_copy:
                app.action_copy_session_id()
            mock_copy.assert_not_called()
            assert "No session selected" in str(app._status.render())


@pytest.mark.skipif(not _HAS_TEXTUAL, reason="textual not installed")
@pytest.mark.asyncio
class TestSearchUX:
    """Cursor stability and debounce when search results merge in."""

    async def test_results_start_directly_below_search_input(self):
        app = SessionBrowser()
        async with app.run_test(size=(96, 24)) as pilot:
            await pilot.pause()
            search = app.query_one("#global-search")
            table = app.query_one("#session-table")

            assert table.region.y == search.region.bottom

    async def test_rebuild_keeps_cursor_on_same_session(self):
        app, fake = _make_app_with_rows(5)
        async with app.run_test() as pilot:
            await _install_fake_sessions(app, pilot, fake)
            table = app.query_one("#session-table")
            table.focus()
            table.move_cursor(row=2)
            await pilot.pause()
            before = app._filtered[table.cursor_row].id
            # The content-search second wave rebuilds the whole table.
            app._rebuild_table()
            await pilot.pause()
            assert app._filtered[table.cursor_row].id == before
            assert table.cursor_row == 2

    async def test_cursor_follows_session_when_rows_inserted_above(self):
        app, fake = _make_app_with_rows(3)  # s1, s2, s3
        async with app.run_test() as pilot:
            await _install_fake_sessions(app, pilot, fake)
            table = app.query_one("#session-table")
            table.focus()
            table.move_cursor(row=2)  # parked on s3
            await pilot.pause()
            # Second wave adds two content-match rows ahead of the current list.
            extra = [
                Session(
                    id="x1", provider="claude", summary="x1", updated_at="2026-02-01"
                ),
                Session(
                    id="x2", provider="claude", summary="x2", updated_at="2026-02-02"
                ),
            ]
            app._filtered = extra + fake
            app._rebuild_table()
            await pilot.pause()
            # Cursor tracks the session, not the row index, so it lands on s3 (now row 4).
            assert app._filtered[table.cursor_row].id == "s3"

    async def test_full_search_results_preserve_visible_cursor_when_selection_lags(
        self,
    ):
        app, fake = _make_app_with_rows(5)
        async with app.run_test() as pilot:
            await _install_fake_sessions(app, pilot, fake)
            table = app.query_one("#session-table")
            table.focus()

            # Model the brief gap between DataTable moving its visible cursor
            # and the corresponding RowHighlighted message reaching the app.
            table.move_cursor(row=3)
            app._selected = fake[0]

            # The full transcript search inserts another match above it.
            extra = Session(
                id="x1", provider="claude", summary="x1", updated_at="2026-02-01"
            )
            app._filtered = [extra, *fake]
            app._rebuild_table()
            await pilot.pause()

            assert table.cursor_row == 4
            assert app._filtered[table.cursor_row].id == "s4"

    def _auth_fake(self):
        return [
            Session(
                id="a1", provider="claude", summary="auth flow", updated_at="2026-01-01"
            ),
            Session(
                id="a2", provider="claude", summary="authn bug", updated_at="2026-01-02"
            ),
            Session(
                id="o1",
                provider="claude",
                summary="other thing",
                updated_at="2026-01-03",
            ),
        ]

    async def test_global_search_filters_metadata_immediately(self):
        app = SessionBrowser()
        fake = self._auth_fake()
        async with app.run_test() as pilot:
            await _install_fake_sessions(app, pilot, fake)
            assert len(app._filtered) == 3
            app._apply_filter("auth")
            # Metadata matches should appear immediately; otherwise slow
            # transcript scans make the search field look broken.
            assert {s.id for s in app._filtered} == {"a1", "a2"}
            await app.workers.wait_for_complete()
            await pilot.pause()
            assert {s.id for s in app._filtered} == {"a1", "a2"}

    async def test_filter_keeps_detail_synchronized_with_visible_selection(self):
        app = SessionBrowser()
        fake = self._auth_fake()
        async with app.run_test() as pilot:
            await _install_fake_sessions(app, pilot, fake)
            # Load the transcript through the real worker path (patched to a
            # known string) and drain its completion message; injecting via
            # _on_content_loaded would race the queued real worker result.
            transcript = Transcript(
                fake[2], [TranscriptEntry("assistant", "original selected transcript")]
            )
            with patch("session_browser.app.load_transcript", return_value=transcript):
                app._select_session("2")
                await app.workers.wait_for_complete()
                await pilot.pause()
            assert app._selected is not None
            assert app._selected.id == "o1"
            assert app._detail_text == "Assistant: original selected transcript"

            with patch(
                "session_browser.app.search_session_hits",
                return_value={"o1": _make_hit()},
            ):
                app._apply_filter("auth")
                assert {s.id for s in app._filtered} == {"a1", "a2"}
                assert app._selected is not None
                assert app._selected.id == "a1"
                assert app._detail_text == ""

                await app.workers.wait_for_complete()
                await pilot.pause()

            assert {s.id for s in app._filtered} == {"a1", "a2", "o1"}
            assert app._selected is not None
            assert app._selected.id == "a1"
            table = app.query_one("#session-table")
            assert app._filtered[table.cursor_row].id == "a1"

    async def test_deep_search_spinner_falls_back_into_input_when_short(self):
        """A short terminal hides the pane titles (`Screen.-short
        .pane-title`), so the progress readout has nowhere to go. The bare
        in-box glyph is the fallback there — without it, short terminals
        would show no sign that a search is still running at all.
        """
        app = SessionBrowser()
        fake = self._auth_fake()
        async with app.run_test(size=(80, 24)) as pilot:
            await _install_fake_sessions(app, pilot, fake)
            app._status.update("General status stays independent")
            app._apply_filter("auth")
            search = app.query_one("#global-search")
            assert search.spinner_glyph is not None
            visible_width = search.scrollable_content_region.width
            visible_line = search.render_line(0).crop(0, visible_width)
            assert visible_line.text.endswith(search.spinner_glyph)
            assert "General status stays independent" in str(app._status.render())

            await app.workers.wait_for_complete()
            await pilot.pause()

            assert search.spinner_glyph is None
            assert "General status stays independent" in str(app._status.render())

    async def test_progress_readout_degrades_instead_of_truncating(self):
        """At 100 columns the pane is 41 wide, leaving 31 for the readout —
        one character more than the full phrase. Three-digit counts push it
        over, and the fallback must be a shorter *wording*, never a phrase
        chopped mid-word.
        """
        app = SessionBrowser()
        fake = self._auth_fake()
        async with app.run_test(size=(100, 34)) as pilot:
            await _install_fake_sessions(app, pilot, fake)
            title = app.query_one("#sessions-title")
            status = app.query_one("#sessions-status")
            app._apply_filter("auth")

            rendered = []
            for done, total in ((9, 64), (120, 640), (1200, 6400)):
                app._note_search_progress(
                    app._search_generation, "reading", done, total
                )
                app._render_spinner()
                rendered.append(str(status.render()))

            for text in rendered:
                assert "…" not in text.replace("transcripts", "")
            # The widest form survives at two digits and is dropped as the
            # numbers grow, rather than the line overflowing the pane.
            assert "reading 9 of 64 transcripts" in rendered[0]
            assert "transcripts" not in rendered[-1]
            # The title itself never moves; the readout hangs off its right.
            assert str(title.render()).strip() == "SESSIONS"

            await app.workers.wait_for_complete()
            await pilot.pause()

    async def test_progress_readout_hangs_off_the_right_of_the_pane_title(self):
        app = SessionBrowser()
        fake = self._auth_fake()
        async with app.run_test(size=(120, 34)) as pilot:
            await _install_fake_sessions(app, pilot, fake)
            title = app.query_one("#sessions-title")
            status = app.query_one("#sessions-status")
            assert str(title.render()).strip() == "SESSIONS"
            assert str(status.render()).strip() == ""

            app._apply_filter("auth")
            app._note_search_progress(app._search_generation, "reading", 12, 64)
            app._render_spinner()
            # The readout sits in its own slot to the right of the title —
            # right-aligned, never pushing the title itself.
            assert "reading 12 of 64 transcripts" in str(status.render())
            assert str(title.render()).strip() == "SESSIONS"
            assert status.region.x >= title.region.x + title.region.width
            # The pane title carried it, so the in-box glyph stays out of the
            # way rather than animating twice in the same pane.
            assert app.query_one("#global-search").spinner_glyph is None

            await app.workers.wait_for_complete()
            await pilot.pause()

    async def test_progress_readout_reports_the_scanning_phase_first(self):
        app = SessionBrowser()
        fake = self._auth_fake()
        async with app.run_test(size=(120, 34)) as pilot:
            await _install_fake_sessions(app, pilot, fake)
            status = app.query_one("#sessions-status")
            app._apply_filter("auth")
            app._note_search_progress(app._search_generation, "scanning", 0, 1345)
            app._render_spinner()
            assert "scanning 1,345 sessions" in str(status.render())
            await app.workers.wait_for_complete()
            await pilot.pause()

    async def test_superseded_search_progress_is_ignored(self):
        """Cancellation is cooperative and the conversation scan has no check
        inside it, so an abandoned search keeps reporting for a while. Its
        counters must not interleave with the live search's — a readout that
        jumps backwards looks more broken than none at all.
        """
        app = SessionBrowser()
        fake = self._auth_fake()
        async with app.run_test(size=(120, 34)) as pilot:
            await _install_fake_sessions(app, pilot, fake)
            status = app.query_one("#sessions-status")
            app._apply_filter("auth")
            stale = app._search_generation

            app._apply_filter("authn")  # supersedes the first search
            app._note_search_progress(app._search_generation, "reading", 40, 90)
            app._note_search_progress(stale, "reading", 3, 7)
            app._render_spinner()
            assert "reading 40 of 90 transcripts" in str(status.render())

            # Superseding a worker cancels it, and waiting on a cancelled
            # worker raises; stop the group and let teardown proceed.
            app.workers.cancel_group(app, "content_search")
            await pilot.pause()

    async def test_conversation_scan_keeps_the_readout_moving(self):
        """The database scan routinely outlives the file probes. Once the
        file phase is complete the display must hand over to it rather than
        sit on a finished "N of N", which is what read as a hang.
        """
        app = SessionBrowser()
        fake = self._auth_fake()
        async with app.run_test(size=(120, 34)) as pilot:
            await _install_fake_sessions(app, pilot, fake)
            status = app.query_one("#sessions-status")
            app._apply_filter("auth")
            gen = app._search_generation

            app._note_search_progress(gen, "reading", 5, 20)
            app._note_search_progress(gen, "conversations", 8192, 0)
            app._render_spinner()
            # Reading is still in flight, so it keeps the line.
            assert "reading 5 of 20 transcripts" in str(status.render())

            app._note_search_progress(gen, "reading", 20, 20)
            app._render_spinner()
            assert "scanning conversations… 8,192" in str(status.render())

            app._note_search_progress(gen, "conversations", 12, 62)
            app._render_spinner()
            assert "reading 12 of 62 conversations" in str(status.render())

            app._note_search_progress(gen, "indexing", 3, 9)
            app._render_spinner()
            assert "indexing 3 of 9 results" in str(status.render())

            await app.workers.wait_for_complete()
            await pilot.pause()

    async def test_landed_results_are_named_in_the_pane_title(self):
        """The second wave must announce itself. The whole point of the
        readout is that the extra rows arrive expected, not mysterious.
        """
        app = SessionBrowser()
        fake = self._auth_fake()
        async with app.run_test(size=(120, 34)) as pilot:
            await _install_fake_sessions(app, pilot, fake)
            title = app.query_one("#sessions-title")
            status = app.query_one("#sessions-status")
            app._stop_deep_search(found=41)
            assert "+41 from transcripts" in str(status.render())

            app._clear_done_flash()
            assert str(status.render()).strip() == ""
            assert str(title.render()).strip() == "SESSIONS"

    async def test_pane_title_reverts_when_nothing_was_added(self):
        app = SessionBrowser()
        fake = self._auth_fake()
        async with app.run_test(size=(120, 34)) as pilot:
            await _install_fake_sessions(app, pilot, fake)
            title = app.query_one("#sessions-title")
            status = app.query_one("#sessions-status")
            app._apply_filter("auth")
            await app.workers.wait_for_complete()
            await pilot.pause()
            # Fake sessions have no transcripts, so nothing is added and the
            # title must not be left advertising a wave that never came.
            assert str(title.render()).strip() == "SESSIONS"
            assert str(status.render()).strip() == ""
            assert app.query_one("#global-search").spinner_glyph is None

    async def test_clearing_global_search_clears_spinner(self):
        app = SessionBrowser()
        fake = self._auth_fake()
        async with app.run_test() as pilot:
            await _install_fake_sessions(app, pilot, fake)
            app._apply_filter("auth")
            search = app.query_one("#global-search")
            assert search.spinner_glyph is not None

            app._apply_filter("")

            assert search.spinner_glyph is None

    async def test_global_search_debounces_keystrokes(self):
        import asyncio

        app, fake = _make_app_with_rows()
        async with app.run_test() as pilot:
            await _install_fake_sessions(app, pilot, fake)
            calls: list[str] = []
            orig = app._apply_filter

            def spy(q: str) -> None:
                calls.append(q)
                orig(q)

            app._apply_filter = spy
            app.query_one("#global-search").focus()
            await pilot.press("a", "b", "c")
            # Within the debounce window, no filter has run yet.
            assert calls == []
            await asyncio.sleep(0.3)
            await pilot.pause()
            # One coalesced filter with the final value.
            assert calls == ["abc"]

    async def test_content_hits_rank_conversation_above_tool_echo(self):
        """A session that merely quoted the phrase in tool output must not
        outrank older sessions where someone actually said it — but user
        and assistant evidence sort together by recency, so a fresh
        assistant-prose match is not buried under stale user matches."""
        app = SessionBrowser()
        fake = [
            Session(
                id="new-tool",
                provider="claude",
                summary="quoting run",
                updated_at="2026-03-01",
            ),
            Session(
                id="mid-assist",
                provider="claude",
                summary="a mention",
                updated_at="2026-02-01",
            ),
            Session(
                id="old-user",
                provider="claude",
                summary="tmux install",
                updated_at="2026-01-01",
            ),
        ]
        async with app.run_test() as pilot:
            await _install_fake_sessions(app, pilot, fake)
            app._filter_query = "prompt grab"
            app._content_hits = {
                "new-tool": _make_hit(role="tool", match="prompt grab"),
                "mid-assist": _make_hit(role="assistant", match="prompt grab"),
                "old-user": _make_hit(role="user", match="prompt grab"),
            }
            app._rebuild_filtered("prompt grab")
            assert [s.id for s in app._filtered] == [
                "mid-assist",
                "old-user",
                "new-tool",
            ]

    async def test_async_callbacks_survive_unmounted_widgets(self):
        """Timer ticks and worker completions can be delivered while the app
        is tearing down, after their widgets are unmounted. Historically this
        crashed on exit with NoMatches (e.g. quitting mid-deep-search); every
        async re-entry point must absorb it."""
        app = SessionBrowser()
        fake = self._auth_fake()
        async with app.run_test() as pilot:
            await _install_fake_sessions(app, pilot, fake)
            app._pending_query = "auth"
            await app.query_one("#global-search").remove()
            await app.query_one("#status-bar").remove()
            await pilot.pause()
            # None of these may raise against the gutted DOM.
            app._render_spinner()
            app._run_pending_filter()
            app._restore_context_bar()
            app._sync_responsive_chrome()

    async def test_header_count_reflects_active_filter(self):
        """While a query is live the header must say how many sessions
        matched, not keep showing the unfiltered total."""
        app = SessionBrowser()
        fake = [
            Session(
                id="s1",
                provider="claude",
                summary="prompt grab notes",
                updated_at="2026-03-01",
            ),
            Session(
                id="s2", provider="claude", summary="unrelated", updated_at="2026-02-01"
            ),
            Session(
                id="s3",
                provider="claude",
                summary="also unrelated",
                updated_at="2026-01-01",
            ),
        ]
        async with app.run_test() as pilot:
            await _install_fake_sessions(app, pilot, fake)
            app._filter_query = "prompt grab"
            app._content_hits = {}
            app._rebuild_filtered("prompt grab")
            text = str(app._app_meta.render())
            assert "1" in text and "of 3" in text
            # Clearing the filter restores the plain total.
            app._apply_filter("")
            assert "of 3" not in str(app._app_meta.render())

    async def test_content_hit_row_shows_count_and_snippet(self):
        """The summary cell shows *why* the row matched, not the opening
        prompt of an unrelated-looking session."""
        app = SessionBrowser()
        fake = [
            Session(
                id="s1",
                provider="claude",
                summary="let's install tmux-thumbs",
                updated_at="2026-01-01",
            )
        ]
        async with app.run_test() as pilot:
            await _install_fake_sessions(app, pilot, fake)
            app._filter_query = "prompt grab"
            app._content_hits = {
                "s1": _make_hit(
                    count=13,
                    role="user",
                    before="the ",
                    match="prompt grab",
                    after=" command",
                )
            }
            app._rebuild_filtered("prompt grab")
            await pilot.pause()
            table = app.query_one("#session-table")
            summary_cell = table.get_row_at(0)[-1]
            assert "13×" in summary_cell
            assert "prompt grab" in summary_cell
            assert "tmux-thumbs" not in summary_cell

    async def test_global_query_carries_into_opened_transcript(self):
        app = SessionBrowser()
        fake = self._auth_fake()
        async with app.run_test() as pilot:
            transcript = Transcript(
                fake[0],
                [
                    TranscriptEntry("user", "please fix the auth flow"),
                    TranscriptEntry("assistant", "the `auth` module is fixed"),
                ],
            )
            with patch("session_browser.app.load_transcript", return_value=transcript):
                await _install_fake_sessions(app, pilot, fake)
                await app.workers.wait_for_complete()
                await pilot.pause()
                app._apply_filter("auth")
                await app.workers.wait_for_complete()
                # Drain the worker StateChanged message, the transcript-loaded
                # sync, and the seeded Input.Changed it triggers.
                for _ in range(3):
                    await pilot.pause()
            box = app.query_one("#detail-search")
            assert box.value == "auth"
            # Both occurrences count — including the backticked one, because
            # the in-session search shares the global markdown-insensitive
            # matching.
            assert len(app._matches) == 2
            assert app._match_idx == 0
            assert "1 / 2" in str(app.query_one("#match-counter").render())

    async def test_user_detail_query_survives_global_search_sync(self):
        app = SessionBrowser()
        fake = self._auth_fake()
        async with app.run_test() as pilot:
            await _install_fake_sessions(app, pilot, fake)
            box = app.query_one("#detail-search")
            box.value = "my own query"
            await pilot.pause()
            app._apply_filter("auth")
            await app.workers.wait_for_complete()
            await pilot.pause()
            assert box.value == "my own query"

    async def test_user_detail_query_rehighlights_on_session_switch(self):
        # A user-typed in-session query must stay live when the selection
        # moves to another session: the new transcript resets match state,
        # and the sync must re-run the query rather than leave the box
        # populated but the matches empty (regression: only retyping the
        # query re-triggered highlighting).
        app = SessionBrowser()
        fake = self._auth_fake()
        async with app.run_test() as pilot:
            transcript = Transcript(
                fake[0],
                [
                    TranscriptEntry("user", "please fix the auth flow"),
                ],
            )
            with patch("session_browser.app.load_transcript", return_value=transcript):
                await _install_fake_sessions(app, pilot, fake)
                await app.workers.wait_for_complete()
                await pilot.pause()
                box = app.query_one("#detail-search")
                box.value = "auth"  # user-typed: no global filter active
                await pilot.pause()
                assert len(app._matches) == 1
                app._select_session("1")
                await app.workers.wait_for_complete()
                for _ in range(3):
                    await pilot.pause()
            assert box.value == "auth"
            assert len(app._matches) == 1
            assert app._match_idx == 0
            assert "1 / 1" in str(app.query_one("#match-counter").render())

    async def test_clearing_global_search_clears_seeded_detail_query(self):
        app = SessionBrowser()
        fake = self._auth_fake()
        async with app.run_test() as pilot:
            await _install_fake_sessions(app, pilot, fake)
            app._apply_filter("auth")
            await app.workers.wait_for_complete()
            await pilot.pause()
            box = app.query_one("#detail-search")
            assert box.value == "auth"
            app._apply_filter("")
            await app.workers.wait_for_complete()
            await pilot.pause()
            assert box.value == ""

    async def test_detail_search_is_markdown_insensitive(self):
        app, fake = _make_app_with_rows()
        async with app.run_test() as pilot:
            await _install_fake_sessions(app, pilot, fake)
            app._on_content_loaded("We should use `SELECT` only for this")
            app._do_session_search("select only")
            assert len(app._matches) == 1
            start, end = app._matches[0]
            assert app._detail_text[start:end] == "SELECT` only"


@pytest.mark.skipif(not _HAS_TEXTUAL, reason="textual not installed")
class TestMarkupSafety:
    """Session content is adversarial markup: brackets, tags, backslashes."""

    def test_content_hit_cell_with_trailing_backslash_renders(self):
        # A snippet part ending in "\" must not escape the tag appended
        # after it — that MarkupError crashed the whole app. Both markup
        # dialects must parse: DataTable cells go through Rich,
        # Static widgets through Textual's (stricter-to-please) parser.
        from rich.text import Text
        from textual.content import Content

        from session_browser.app import _format_content_hit

        hit = _make_hit(
            count=2120,
            role="tool",
            before="C:\\router\\cfg\\",
            match="pg",
            after=" tail [/] and \\",
        )
        cell = _format_content_hit(hit)
        Text.from_markup(cell)  # must not raise
        Content.from_markup(cell)  # must not raise

    def test_escape_markup_neutralizes_both_markup_dialects(self):
        from rich.text import Text
        from textual.content import Content

        from session_browser.app import _escape_markup

        # `[Session(...)]` is a valid *Textual* tag though not a Rich one;
        # a trailing backslash gets a guard space so a following tag
        # survives in both dialects.
        for raw in (
            'sessions = [Session(id="s")]',
            "[bold]not a tag[/] ends with backslash \\",
            "\\",
            "C:\\router\\cfg\\",
        ):
            markup = f"[on yellow]{_escape_markup(raw)}[/]tail"
            expected = raw + (" tail" if raw.endswith("\\") else "tail")
            assert Text.from_markup(markup).plain == expected
            assert Content.from_markup(markup).plain == expected


@pytest.mark.skipif(not _HAS_TEXTUAL, reason="textual not installed")
class TestProjectName:
    def test_basename_of_cwd(self):
        from session_browser.app import _project_name

        assert _project_name("/home/user/proj") == "proj"

    def test_trailing_separator_ignored(self):
        from session_browser.app import _project_name

        assert _project_name("/home/user/proj" + os.sep) == "proj"

    def test_missing_cwd_shows_dash(self):
        from session_browser.app import _project_name

        assert _project_name("") == "—"

    def test_long_name_truncated_with_ellipsis(self):
        from session_browser.app import _project_name

        name = _project_name("/home/user/a-really-long-project-directory-name")
        assert len(name) == 18
        assert name.endswith("…")


@pytest.mark.skipif(not _HAS_TEXTUAL, reason="textual not installed")
@pytest.mark.asyncio
class TestProjectColumn:
    """The session list identifies each session's project by cwd basename."""

    async def test_table_shows_project_column(self):
        app = SessionBrowser()
        fake = [
            Session(
                id="s1",
                provider="claude",
                summary="with cwd",
                cwd="/home/user/myproj",
                updated_at="2026-01-02",
            ),
            Session(
                id="s2",
                provider="claude",
                summary="without cwd",
                updated_at="2026-01-01",
            ),
        ]
        async with app.run_test() as pilot:
            await _install_fake_sessions(app, pilot, fake)
            table = app.query_one("#session-table")
            labels = [str(col.label) for col in table.columns.values()]
            # Summary is deliberately last — it absorbs clipping in narrow
            # panes so the short metadata columns stay visible.
            assert labels == ["Provider", "Project", "Age", "Summary"]
            assert "myproj" in str(table.get_row_at(0)[1])
            assert "—" in str(table.get_row_at(1)[1])


@pytest.mark.skipif(not _HAS_TEXTUAL, reason="textual not installed")
@pytest.mark.asyncio
class TestDisplayWindow:
    """The 200K display bound is a movable window, not a truncation."""

    def _long_text(self) -> str:
        # ~210K chars of 1000-char lines, then a needle on the last line.
        return "\n".join(["x" * 1000] * 210) + "\nthe NEEDLE sentence"

    async def test_default_window_starts_at_beginning(self):
        app, fake = _make_app_with_rows(1)
        async with app.run_test() as pilot:
            await _install_fake_sessions(app, pilot, fake)
            app._on_content_loaded(self._long_text())
            visible = app._compose_visible_text()
            assert visible.startswith("x")
            assert "characters after this window omitted" in visible
            assert "NEEDLE" not in visible

    async def test_search_moves_window_to_match_beyond_boundary(self):
        app, fake = _make_app_with_rows(1)
        async with app.run_test() as pilot:
            await _install_fake_sessions(app, pilot, fake)
            text = self._long_text()
            app._on_content_loaded(text)
            app._do_session_search("needle")
            assert len(app._matches) == 1
            assert app._window_start > 0
            visible = app._compose_visible_text()
            assert "NEEDLE" in visible
            assert "characters before this window omitted" in visible
            # Source text untouched: the window is purely presentational.
            assert app._detail_text == text

    async def test_navigation_slides_window_and_wraps_back(self):
        app, fake = _make_app_with_rows(1)
        async with app.run_test() as pilot:
            await _install_fake_sessions(app, pilot, fake)
            text = (
                "first NEEDLE here\n"
                + "\n".join(["x" * 1000] * 210)
                + "\nlast NEEDLE there"
            )
            app._on_content_loaded(text)
            app._do_session_search("needle")
            assert len(app._matches) == 2
            assert app._window_start == 0  # first match is in view
            app.action_next_match()
            assert app._window_start > 0  # slid to the far match
            # The match itself is wrapped in highlight markup, so assert on
            # neighbouring text that only exists near the far needle.
            visible = app._compose_visible_text()
            assert "there" in visible and "first" not in visible
            app.action_next_match()  # wraps to the first match
            assert app._window_start == 0

    async def test_match_counter_counts_beyond_window(self):
        app, fake = _make_app_with_rows(1)
        async with app.run_test() as pilot:
            await _install_fake_sessions(app, pilot, fake)
            text = "first NEEDLE\n" + "\n".join(["x" * 1000] * 210) + "\nlast NEEDLE"
            app._on_content_loaded(text)
            app._do_session_search("needle")
            from textual.widgets import Label

            counter = str(app.query_one("#match-counter", Label).render())
            assert "1 / 2" in counter


@pytest.mark.skipif(not _HAS_TEXTUAL, reason="textual not installed")
@pytest.mark.asyncio
class TestStructuredTranscript:
    def _transcript(self, session: Session) -> Transcript:
        return Transcript(
            session,
            [
                TranscriptEntry(
                    "user", "Please inspect the auth flow", "2026-01-01T10:00:00Z"
                ),
                TranscriptEntry(
                    "assistant", "I will inspect it now", "2026-01-01T10:00:01Z"
                ),
                TranscriptEntry(
                    "tool",
                    'Read({"file": "auth.py"})',
                    metadata={"kind": "call", "tool": "Read"},
                ),
                TranscriptEntry(
                    "tool",
                    "first line\n" + "x" * 700 + " searchable-tail",
                    metadata={"kind": "output"},
                ),
            ],
        )

    def _windowed_transcript(self, session: Session) -> Transcript:
        return Transcript(
            session,
            [
                TranscriptEntry(
                    "assistant",
                    f"entry {index:02d}\n" + (chr(97 + index % 26) * 180),
                )
                for index in range(30)
            ],
        )

    async def test_entries_render_as_role_widgets_and_tool_output_collapses(self):
        from session_browser.app import TranscriptEntryWidget

        app, fake = _make_app_with_rows(1)
        async with app.run_test() as pilot:
            await _install_fake_sessions(app, pilot, fake)
            app._on_transcript_loaded(self._transcript(fake[0]))
            await pilot.pause()

            widgets = list(app.query(TranscriptEntryWidget))
            assert [widget.entry.role for widget in widgets] == [
                "user",
                "assistant",
                "tool",
                "tool",
            ]
            output = widgets[-1]
            assert output.collapsed is True
            assert "chars hidden" in str(output.render())
            output.action_toggle_collapsed()
            assert output.collapsed is False
            assert "searchable-tail" in str(output.render())

    async def test_drag_selection_copies_without_collapsing_or_jumping(
        self, monkeypatch
    ):
        """Mouse release after a drag is selection, not a collapse click."""
        from textual import events

        import session_browser.app as app_module
        from session_browser.app import TranscriptEntryWidget

        app, fake = _make_app_with_rows(1)
        copied = []
        monkeypatch.setattr(
            app_module,
            "copy_to_clipboard",
            lambda text: (copied.append(text), True)[1],
        )
        async with app.run_test(size=(120, 30)) as pilot:
            await _install_fake_sessions(app, pilot, fake)
            app._on_transcript_loaded(self._transcript(fake[0]))
            await pilot.pause()

            output = list(app.query(TranscriptEntryWidget))[-1]
            output.action_toggle_collapsed()
            await pilot.pause()
            output.scroll_visible(top=True, animate=False)
            await pilot.pause()

            scroll = app.query_one("#detail-scroll")
            scroll_before = scroll.scroll_y

            def mouse_event(event_type, x):
                screen_x = output.content_region.x + x
                screen_y = output.content_region.y + 1
                return event_type(
                    output,
                    screen_x,
                    screen_y,
                    0,
                    0,
                    1,
                    False,
                    False,
                    False,
                    screen_x,
                    screen_y,
                )

            # Drive events through App.on_event: unlike Pilot.mouse_up, this
            # exercises Textual's synthesized Click after a same-widget drag.
            await app.on_event(mouse_event(events.MouseDown, 0))
            await pilot.pause()
            await app.on_event(mouse_event(events.MouseMove, 4))
            await pilot.pause()
            await app.on_event(mouse_event(events.MouseUp, 4))
            await pilot.pause()

            assert copied == ["first"]
            assert output.collapsed is False
            assert scroll.scroll_y == scroll_before
            assert app.query_one("#status-bar").has_class("-flash-ok")

            await pilot.click(output, offset=(2, 1))
            await pilot.pause()
            assert output.collapsed is True

    async def test_collapsed_entry_previews_its_match_not_its_first_line(self):
        """A hit behind the fold has to render, not merely be counted.

        Only the *active* match expands its own entry, so every other match
        inside a collapsed tool output reaches the screen through the
        preview. Pinned to line one, that preview showed none of them — and
        a broad query looked like it had missed occurrences that a narrower
        one found, because narrowing eventually made one of them active.
        """
        from session_browser.app import TranscriptEntryWidget

        app, fake = _make_app_with_rows(1)
        async with app.run_test() as pilot:
            await _install_fake_sessions(app, pilot, fake)
            app._on_transcript_loaded(
                Transcript(
                    fake[0],
                    [
                        TranscriptEntry("user", "please find goal"),
                        TranscriptEntry(
                            "tool",
                            "first line\n"
                            + "x" * 700
                            + "\nthen ran /goal on it\n"
                            + "y" * 700,
                            metadata={"kind": "output"},
                        ),
                    ],
                )
            )
            await pilot.pause()

            app._do_session_search("goal")
            await pilot.pause()

            # The first match is in the opening entry, so the tool output
            # never expands: whatever it shows, it shows while collapsed.
            output = list(app.query(TranscriptEntryWidget))[-1]
            assert output.collapsed is True
            rendered = str(output.render())
            assert "then ran /goal on it" in rendered
            assert "1 match here" in rendered
            assert "first line" not in rendered

    async def test_collapsed_entry_without_matches_previews_first_line(self):
        """No query, no reason to move the preview off the opening line."""
        from session_browser.app import TranscriptEntryWidget

        app, fake = _make_app_with_rows(1)
        async with app.run_test() as pilot:
            await _install_fake_sessions(app, pilot, fake)
            app._on_transcript_loaded(self._transcript(fake[0]))
            await pilot.pause()

            app._do_session_search("nothing-matches-this")
            await pilot.pause()

            output = list(app.query(TranscriptEntryWidget))[-1]
            assert output.collapsed is True
            rendered = str(output.render())
            assert "first line" in rendered
            assert "match here" not in rendered

    async def test_search_keeps_flat_buffer_and_reuses_widgets(self):
        from session_browser.app import TranscriptEntryWidget

        app, fake = _make_app_with_rows(1)
        async with app.run_test() as pilot:
            await _install_fake_sessions(app, pilot, fake)
            transcript = self._transcript(fake[0])
            app._on_transcript_loaded(transcript)
            await pilot.pause()
            original_widgets = list(app.query(TranscriptEntryWidget))

            app._do_session_search("searchable-tail")
            await pilot.pause()

            assert app._detail_text.endswith("searchable-tail")
            assert len(app._matches) == 1
            # Typing a query updates existing widgets; it does not rebuild the
            # structured transcript or rescan each entry independently.
            assert list(app.query(TranscriptEntryWidget)) == original_widgets
            assert original_widgets[-1].collapsed is False

    async def test_match_navigation_scrolls_to_match_inside_long_entry(self):
        """n/N target the rendered match row, not merely its entry widget."""
        from textual.containers import VerticalScroll

        from session_browser.app import TranscriptEntryWidget

        app, fake = _make_app_with_rows(1)
        async with app.run_test(size=(120, 30)) as pilot:
            await _install_fake_sessions(app, pilot, fake)
            transcript = Transcript(
                fake[0],
                [
                    TranscriptEntry("user", "A short opening entry"),
                    TranscriptEntry(
                        "assistant",
                        "\n".join(
                            [f"line {index} " + "x" * 100 for index in range(40)]
                            + ["first co-pilot match"]
                            + [f"line {index} " + "y" * 100 for index in range(40)]
                            + ["second co-pilot match"]
                        ),
                    ),
                ],
            )
            app._on_transcript_loaded(transcript)
            await pilot.pause()
            app._do_session_search("co-pilot")
            await pilot.pause()

            scroll = app.query_one("#detail-scroll", VerticalScroll)
            long_entry = list(app.query(TranscriptEntryWidget))[1]

            def active_match_is_visible() -> bool:
                row = long_entry.active_match_row()
                assert row is not None
                match_y = long_entry.virtual_region.y + row
                top = int(scroll.scroll_y)
                return top <= match_y < top + scroll.scrollable_content_region.height

            assert app._match_idx == 0
            assert active_match_is_visible()

            app.action_next_match()
            await pilot.pause()
            assert app._match_idx == 1
            assert active_match_is_visible()

            app.action_prev_match()
            await pilot.pause()
            assert app._match_idx == 0
            assert active_match_is_visible()

    async def test_structural_navigation_jumps_entries_and_tools(self):
        from session_browser.app import TranscriptEntryWidget

        app, fake = _make_app_with_rows(1)
        async with app.run_test() as pilot:
            await _install_fake_sessions(app, pilot, fake)
            app._on_transcript_loaded(self._transcript(fake[0]))
            await pilot.pause()
            widgets = list(app.query(TranscriptEntryWidget))

            app.action_next_entry()
            await pilot.pause()
            assert app.focused is widgets[0]
            app.action_next_entry()
            await pilot.pause()
            assert app.focused is widgets[1]
            app.action_next_tool()
            await pilot.pause()
            assert app.focused is widgets[2]
            app.action_next_tool()
            await pilot.pause()
            assert app.focused is widgets[3]

    async def test_vim_nav_drives_transcript_when_entry_focused(self):
        """j/k/gg/G must keep acting on the transcript pane when focus sits on
        an entry widget (after J/K/[/] or a click), not fall back to the
        session table."""
        from textual.containers import VerticalScroll
        from textual.widgets import DataTable

        from session_browser.app import TranscriptEntryWidget

        app, fake = _make_app_with_rows(3)
        async with app.run_test() as pilot:
            await _install_fake_sessions(app, pilot, fake)
            app._on_transcript_loaded(self._transcript(fake[0]))
            await pilot.pause()

            app.action_next_entry()
            await pilot.pause()
            assert isinstance(app.focused, TranscriptEntryWidget)

            scroll = app.query_one("#detail-scroll", VerticalScroll)
            assert app._focused_table_or_scroll() is scroll

            table = app.query_one("#session-table", DataTable)
            row_before = table.cursor_row
            app.action_nav_bottom()
            await pilot.pause()
            assert table.cursor_row == row_before, (
                "G with an entry focused must not move the session list"
            )

    async def test_vim_head_and_tail_remount_the_true_transcript_ends(self):
        from textual.containers import VerticalScroll

        from session_browser.app import TranscriptEntryWidget

        app, fake = _make_app_with_rows(1)
        with patch("session_browser.app._DISPLAY_WINDOW", 500):
            async with app.run_test(size=(120, 18)) as pilot:
                await _install_fake_sessions(app, pilot, fake)
                app._on_transcript_loaded(self._windowed_transcript(fake[0]))
                scroll = app.query_one("#detail-scroll", VerticalScroll)
                scroll.focus()
                await pilot.pause()

                assert app._window_start == 0
                assert next(iter(app.query(TranscriptEntryWidget))).entry_index == 0

                app.action_nav_bottom()
                await app.workers.wait_for_complete()
                await pilot.pause()
                await pilot.pause()
                assert app._window_start == len(app._detail_text) - 500
                assert list(app.query(TranscriptEntryWidget))[-1].entry_index == 29
                assert int(scroll.scroll_y) == scroll.max_scroll_y

                app.action_nav_top()
                app.action_nav_top()
                await app.workers.wait_for_complete()
                await pilot.pause()
                await pilot.pause()
                assert app._window_start == 0
                assert next(iter(app.query(TranscriptEntryWidget))).entry_index == 0
                assert int(scroll.scroll_y) == 0

    async def test_page_navigation_crosses_windows_with_overlap(self):
        from textual.containers import VerticalScroll

        from session_browser.app import TranscriptEntryWidget

        def first_visible_index(app, scroll) -> int:
            top = int(scroll.scroll_y)
            return next(
                widget.entry_index
                for widget in app.query(TranscriptEntryWidget)
                if widget.virtual_region.bottom > top
            )

        app, fake = _make_app_with_rows(1)
        with patch("session_browser.app._DISPLAY_WINDOW", 500):
            async with app.run_test(size=(120, 18)) as pilot:
                await _install_fake_sessions(app, pilot, fake)
                app._on_transcript_loaded(self._windowed_transcript(fake[0]))
                scroll = app.query_one("#detail-scroll", VerticalScroll)
                scroll.focus()
                await pilot.pause()

                scroll.scroll_end(animate=False)
                await pilot.pause()
                before_down = first_visible_index(app, scroll)
                app.action_halfpage_down()
                await app.workers.wait_for_complete()
                await pilot.pause()
                after_down = first_visible_index(app, scroll)
                assert app._window_start == 250
                assert after_down >= before_down

                scroll.scroll_home(animate=False)
                await pilot.pause()
                before_up = first_visible_index(app, scroll)
                app.action_halfpage_up()
                await app.workers.wait_for_complete()
                await pilot.pause()
                after_up = first_visible_index(app, scroll)
                assert app._window_start == 0
                assert after_up <= before_up

    async def test_line_navigation_crosses_window_boundary(self):
        from textual.containers import VerticalScroll

        app, fake = _make_app_with_rows(1)
        with patch("session_browser.app._DISPLAY_WINDOW", 500):
            async with app.run_test(size=(120, 18)) as pilot:
                await _install_fake_sessions(app, pilot, fake)
                app._on_transcript_loaded(self._windowed_transcript(fake[0]))
                scroll = app.query_one("#detail-scroll", VerticalScroll)
                scroll.focus()
                await pilot.pause()

                scroll.scroll_end(animate=False)
                await pilot.pause()
                app.action_nav_down()
                await app.workers.wait_for_complete()
                await pilot.pause()
                assert app._window_start == 250

                scroll.scroll_home(animate=False)
                await pilot.pause()
                app.action_nav_up()
                await app.workers.wait_for_complete()
                await pilot.pause()
                assert app._window_start == 0


class TestCopyToClipboard:
    def test_linux_tries_clip_exe_first(self):
        """On Linux (including WSL), clip.exe should be tried before xclip/xsel."""
        with (
            patch("platform.system", return_value="Linux"),
            patch("subprocess.run") as mock_run,
        ):
            # clip.exe succeeds
            call_order = []

            def side_effect(args, **kwargs):
                call_order.append(args[0])
                if args[0] == "clip.exe":
                    return subprocess.CompletedProcess(args, 0)
                raise FileNotFoundError

            mock_run.side_effect = side_effect
            result = copy_to_clipboard("test text")
            assert result is True
            assert call_order[0] == "clip.exe"

    def test_linux_falls_back_when_clip_exe_missing(self):
        """When clip.exe not found, fall back to xclip."""
        with (
            patch("platform.system", return_value="Linux"),
            patch("subprocess.run") as mock_run,
        ):
            call_order = []

            def side_effect(args, **kwargs):
                call_order.append(args[0])
                if args[0] == "xclip":
                    return subprocess.CompletedProcess(args, 0)
                raise FileNotFoundError

            mock_run.side_effect = side_effect
            result = copy_to_clipboard("test text")
            assert result is True
            assert call_order == ["clip.exe", "xclip"]

    def test_darwin_still_uses_pbcopy(self):
        with (
            patch("platform.system", return_value="Darwin"),
            patch("subprocess.run") as mock_run,
        ):
            mock_run.return_value = subprocess.CompletedProcess(["pbcopy"], 0)
            result = copy_to_clipboard("test")
            assert result is True
            assert mock_run.call_args[0][0] == ["pbcopy"]

    def test_linux_returns_false_when_all_fail(self):
        with (
            patch("platform.system", return_value="Linux"),
            patch("subprocess.run", side_effect=FileNotFoundError),
        ):
            result = copy_to_clipboard("test")
            assert result is False


class TestClipboardFormatting:
    def test_cwd_wrapping(self):
        cmd = resume_command("codex", "abc", "/home/user/my project")
        assert cmd.startswith("cd ")
        assert "codex resume " in cmd

    def test_no_cwd_no_wrapping(self):
        cmd = resume_command("claude", "abc")
        assert not cmd.startswith("cd ")


class TestChatExport:
    def test_build_chat_export_uses_title_and_content_only(self):
        session = Session(
            id="abc-123",
            provider="codex",
            summary="Add export flow",
            cwd="/tmp/project",
            branch="main",
        )

        result = build_chat_export(session, "User: hello\nAssistant: hi")

        assert result == "Add export flow\n\nUser: hello\nAssistant: hi\n"
        assert "codex" not in result
        assert "/tmp/project" not in result
        assert "abc-123" not in result

    def test_build_chat_export_falls_back_to_session_id_title(self):
        session = Session(id="abc-123", provider="claude")

        result = build_chat_export(session, "User: hello")

        assert result == "abc-123\n\nUser: hello\n"

    def test_export_chat_to_file_writes_sanitized_txt_name(self, tmp_path):
        session = Session(
            id="abc/123",
            provider="claude",
            summary="Fix: export chat!",
        )

        path = export_chat_to_file(session, "User: hello", output_dir=tmp_path)

        assert path == tmp_path / "session-export-claude-abc-123.txt"
        assert path.read_text() == "Fix: export chat!\n\nUser: hello\n"


# ---------------------------------------------------------------------------
# tmux integration tests
# ---------------------------------------------------------------------------


class TestSessionNameForPath:
    def test_basename_only(self):
        assert session_name_for_path("/home/user/my-project") == "my-project"

    def test_sanitizes_dots_and_spaces(self):
        # tmux forbids '.' and ':' in session names.
        assert session_name_for_path("/tmp/my project.v2") == "my_project_v2"

    def test_trailing_slash(self):
        assert session_name_for_path("/home/user/proj/") == "proj"

    def test_empty_path_falls_back(self):
        assert session_name_for_path("") == "session"


class TestBuildPlan:
    def test_new_session_opens_shell_then_types_resume(self):
        plan = build_plan("claude", "def-456", "/home/user/proj", session_exists=False)
        assert plan.session == "proj"
        # The window starts as a plain shell (no command, no -n) so it keeps the
        # user's normal prompt/config; the agent is typed in via send-keys.
        assert plan.create_commands() == [
            ["tmux", "new-session", "-d", "-s", "proj", "-c", "/home/user/proj"],
            [
                "tmux",
                "send-keys",
                "-t",
                "proj",
                "-l",
                "claude --dangerously-skip-permissions --resume def-456",
            ],
            ["tmux", "send-keys", "-t", "proj", "Enter"],
            ["tmux", "set-option", "-t", "proj", "-w", "@sb_session_id", "def-456"],
        ]
        assert "-n" not in plan.create_commands()[0]

    def test_existing_session_adds_window(self):
        plan = build_plan("codex", "ghi-789", "/home/user/proj", session_exists=True)
        assert plan.create_commands() == [
            ["tmux", "new-window", "-t", "proj", "-c", "/home/user/proj"],
            ["tmux", "send-keys", "-t", "proj", "-l", "codex resume ghi-789"],
            ["tmux", "send-keys", "-t", "proj", "Enter"],
            ["tmux", "set-option", "-t", "proj", "-w", "@sb_session_id", "ghi-789"],
        ]

    def test_existing_window_is_reused_not_duplicated(self):
        # A prior visit tagged a window with this session_id; jumping back in
        # should just select it, not open another window running resume again.
        plan = build_plan(
            "codex",
            "ghi-789",
            "/home/user/proj",
            session_exists=True,
            existing_window="@3",
            existing_tag="ghi-789",
        )
        assert plan.create_commands() == [
            ["tmux", "select-window", "-t", "@3"],
        ]

    def test_reused_window_learns_the_forked_id(self):
        # Claude's resume forks: the window was tagged "abc-123" but the user
        # picked the conversation's newer id. Reuse must also append that id
        # to the tag so the next visit exact-matches without a lineage walk.
        plan = build_plan(
            "claude",
            "def-456",
            "/home/user/proj",
            session_exists=True,
            existing_window="@3",
            existing_tag="abc-123",
        )
        assert plan.create_commands() == [
            ["tmux", "select-window", "-t", "@3"],
            [
                "tmux",
                "set-option",
                "-t",
                "@3",
                "-w",
                "@sb_session_id",
                "abc-123 def-456",
            ],
        ]

    def test_resume_typed_literally_so_window_survives_agent_exit(self):
        # The agent runs inside the shell; when it exits, that same configured
        # shell remains, so no command-wrapping / `exec $SHELL` is needed.
        plan = build_plan("claude", "x", "/home/user/proj", session_exists=False)
        send = plan.create_commands()[1]
        assert send[:5] == ["tmux", "send-keys", "-t", "proj", "-l"]
        assert send[-1] == "claude --dangerously-skip-permissions --resume x"
        assert "exec" not in send[-1]

    def test_connect_switch_matches_worktrees_sh(self):
        # Inside tmux: switch the client, like ensure_repo_session.
        plan = build_plan("claude", "x", "/home/user/proj", session_exists=True)
        assert plan.connect_command(switch=True) == [
            "sesh",
            "connect",
            "--switch",
            "proj",
        ]

    def test_connect_attaches_without_switch(self):
        # Plain terminal: attach (no --switch, which would no-op and flicker).
        plan = build_plan("claude", "x", "/home/user/proj", session_exists=True)
        assert plan.connect_command(switch=False) == ["sesh", "connect", "proj"]

    def test_cwd_passed_via_flag_not_cd(self):
        # cwd is handled by tmux -c, so the resume command itself has no `cd`.
        plan = build_plan("codex", "z", "/tmp/my project", session_exists=False)
        assert "cd " not in plan.resume
        open_window = plan.create_commands()[0]
        assert open_window[-2:] == ["-c", "/tmp/my project"]


class TestPrepareSession:
    def test_requires_cwd(self):
        with pytest.raises(TmuxError, match="no folder"):
            prepare_session("claude", "x", "")

    def test_rejects_unknown_provider(self):
        with (
            patch.object(tmux.shutil, "which", return_value="/usr/bin/tool"),
            pytest.raises(TmuxError, match="unknown provider"),
        ):
            prepare_session("bogus", "x", "/tmp/proj")

    def test_requires_tmux_on_path(self):
        with (
            patch.object(
                tmux.shutil,
                "which",
                side_effect=lambda c: None if c == "tmux" else "/x",
            ),
            pytest.raises(TmuxError, match="tmux not found"),
        ):
            prepare_session("claude", "x", "/tmp/proj")

    def test_requires_sesh_on_path(self):
        with (
            patch.object(
                tmux.shutil,
                "which",
                side_effect=lambda c: None if c == "sesh" else "/x",
            ),
            pytest.raises(TmuxError, match="sesh not found"),
        ):
            prepare_session("claude", "x", "/tmp/proj")

    def test_creates_detached_session_when_missing(self):
        calls = []

        def fake_run(cmd, *a, **kw):
            calls.append(cmd)
            # has-session check fails (session missing); create succeeds.
            rc = 1 if cmd[:2] == ["tmux", "has-session"] else 0
            return subprocess.CompletedProcess(cmd, rc, stdout="", stderr="")

        with (
            patch.object(tmux.shutil, "which", return_value="/usr/bin/tool"),
            patch.object(tmux.subprocess, "run", side_effect=fake_run),
        ):
            plan = prepare_session("claude", "def-456", "/home/user/proj")

        assert plan.session == "proj"
        assert ["tmux", "has-session", "-t", "=proj"] in calls
        assert any(c[:2] == ["tmux", "new-session"] for c in calls)

    def test_adds_window_when_session_exists(self):
        def fake_run(cmd, *a, **kw):
            rc = 0  # has-session returns 0 => exists
            return subprocess.CompletedProcess(cmd, rc, stdout="", stderr="")

        with (
            patch.object(tmux.shutil, "which", return_value="/usr/bin/tool"),
            patch.object(tmux.subprocess, "run", side_effect=fake_run),
        ):
            plan = prepare_session("claude", "x", "/home/user/proj")

        assert plan.session_exists is True
        assert plan.create_commands()[0][:2] == ["tmux", "new-window"]

    def test_reuses_window_already_running_this_session(self):
        calls = []

        def fake_run(cmd, *a, **kw):
            calls.append(cmd)
            if cmd[:2] == ["tmux", "has-session"]:
                return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")
            if cmd[:2] == ["tmux", "list-windows"]:
                out = "@1 other-session\n@3 x\n"
                return subprocess.CompletedProcess(cmd, 0, stdout=out, stderr="")
            return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

        with (
            patch.object(tmux.shutil, "which", return_value="/usr/bin/tool"),
            patch.object(tmux.subprocess, "run", side_effect=fake_run),
        ):
            plan = prepare_session("claude", "x", "/home/user/proj")

        assert plan.existing_window == "@3"
        assert plan.create_commands() == [["tmux", "select-window", "-t", "@3"]]
        # Only the has-session/list-windows lookups and the select ran — no
        # duplicate new-window/send-keys for a session already open.
        assert not any(c[:2] == ["tmux", "new-window"] for c in calls)
        assert not any(c[:2] == ["tmux", "send-keys"] for c in calls)

    def test_reuses_window_tagged_with_an_ancestor_id(self, tmp_path):
        # The window was opened for "old-id"; claude --resume forked the
        # conversation to "new-id", which is the row the user picks next time.
        # The fork's transcript still carries the ancestor id in top-level
        # session_id fields, so the old window must be found and retagged.
        transcript = tmp_path / "new-id.jsonl"
        transcript.write_text(
            '{"type":"assistant","sessionId":"new-id","session_id":"old-id"}\n'
            '{"type":"user","sessionId":"new-id"}\n'
        )
        calls = []

        def fake_run(cmd, *a, **kw):
            calls.append(cmd)
            if cmd[:2] == ["tmux", "list-windows"]:
                out = "@1 unrelated\n@3 old-id\n"
                return subprocess.CompletedProcess(cmd, 0, stdout=out, stderr="")
            return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

        with (
            patch.object(tmux.shutil, "which", return_value="/usr/bin/tool"),
            patch.object(tmux.subprocess, "run", side_effect=fake_run),
        ):
            plan = prepare_session(
                "claude", "new-id", "/home/user/proj", content_path=str(transcript)
            )

        assert plan.existing_window == "@3"
        assert not any(c[:2] == ["tmux", "new-window"] for c in calls)
        assert not any(c[:2] == ["tmux", "send-keys"] for c in calls)
        assert [
            "tmux",
            "set-option",
            "-t",
            "@3",
            "-w",
            "@sb_session_id",
            "old-id new-id",
        ] in calls

    def test_raises_when_tmux_create_fails(self):
        def fake_run(cmd, *a, **kw):
            if cmd[:2] == ["tmux", "has-session"]:
                return subprocess.CompletedProcess(cmd, 1, stdout="", stderr="")
            return subprocess.CompletedProcess(cmd, 1, stdout="", stderr="boom")

        with (
            patch.object(tmux.shutil, "which", return_value="/usr/bin/tool"),
            patch.object(tmux.subprocess, "run", side_effect=fake_run),
            pytest.raises(TmuxError, match="boom"),
        ):
            prepare_session("claude", "x", "/home/user/proj")


# ---------------------------------------------------------------------------
# herdr integration tests
# ---------------------------------------------------------------------------


def _herdr_pane(
    pane_id="w1:p1",
    cwd="/home/user/proj",
    agent=None,
    session_id=None,
    tab_id=None,
    workspace_id=None,
):
    """One entry as `herdr pane list` prints it."""
    pane = {
        "pane_id": pane_id,
        "tab_id": tab_id or pane_id.replace(":p", ":t"),
        "workspace_id": workspace_id or pane_id.split(":")[0],
        "cwd": cwd,
    }
    if agent:
        pane["agent"] = agent
        pane["agent_session"] = {
            "agent": agent,
            "kind": "id",
            "source": f"herdr:{agent}",
            "value": session_id or "",
        }
    return pane


def _herdr_stub(panes=(), created=None, fail=None):
    """A fake `herdr` CLI. Returns the calls made and the runner to patch in.

    ``fail`` names a command prefix (e.g. ``["pane", "run"]``) that should exit
    non-zero with a herdr-shaped JSON error on stderr.
    """
    calls = []
    default_created = {
        "workspace": {"workspace_id": "w9"},
        "tab": {"tab_id": "w9:t1"},
        "root_pane": {"pane_id": "w9:p1"},
    }

    def fake_run(cmd, *a, **kw):
        calls.append(cmd)
        args = cmd[1:]
        if fail and args[: len(fail)] == fail:
            error = json.dumps(
                {"error": {"code": "pane_busy", "message": "pane is not at a prompt"}}
            )
            return subprocess.CompletedProcess(cmd, 1, stdout="", stderr=error)
        if args[:2] == ["pane", "list"]:
            body = {"panes": list(panes)}
        elif args[:2] in (["workspace", "create"], ["tab", "create"]):
            body = created if created is not None else default_created
        else:
            # Input-sending commands such as `pane run` print nothing at all
            # and report success through the exit status alone.
            return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")
        out = json.dumps({"id": "cli:test", "result": body})
        return subprocess.CompletedProcess(cmd, 0, stdout=out, stderr="")

    return calls, fake_run


class TestWorkspaceLabelForPath:
    def test_basename_only(self):
        assert workspace_label_for_path("/home/user/my-project") == "my-project"

    def test_keeps_characters_tmux_would_reject(self):
        # herdr labels are display text, not tmux session names.
        assert workspace_label_for_path("/tmp/my project.v2") == "my project.v2"

    def test_trailing_slash(self):
        assert workspace_label_for_path("/home/user/proj/") == "proj"

    def test_empty_path_falls_back(self):
        assert workspace_label_for_path("") == "session"


class TestHerdrPaneLookup:
    def test_finds_the_pane_running_this_conversation(self):
        panes = [
            HerdrPane("w1:p1", "w1:t1", "w1", "/a", "", ""),
            HerdrPane("w1:p2", "w1:t2", "w1", "/a", "claude", "abc"),
        ]
        found = herdr.find_pane(panes, "claude", {"abc"})
        assert found is not None and found.pane_id == "w1:p2"

    def test_ignores_a_matching_id_under_a_different_agent(self):
        # Session ids are per-provider; a codex pane on the same string is a
        # different conversation.
        panes = [HerdrPane("w1:p1", "w1:t1", "w1", "/a", "codex", "abc")]
        assert herdr.find_pane(panes, "claude", {"abc"}) is None

    def test_ignores_panes_without_an_agent(self):
        panes = [HerdrPane("w1:p1", "w1:t1", "w1", "/a", "", "")]
        assert herdr.find_pane(panes, "claude", {""}) is None

    def test_workspace_found_through_a_pane_in_that_folder(self):
        panes = [
            HerdrPane("w1:p1", "w1:t1", "w1", "/other", "", ""),
            HerdrPane("w2:p1", "w2:t1", "w2", "/home/user/proj", "", ""),
        ]
        assert herdr.workspace_for_cwd(panes, "/home/user/proj") == "w2"

    def test_workspace_match_survives_a_trailing_slash(self):
        # Provider-recorded cwds are not normalised the way herdr's are.
        panes = [HerdrPane("w2:p1", "w2:t1", "w2", "/home/user/proj", "", "")]
        assert herdr.workspace_for_cwd(panes, "/home/user/proj/") == "w2"

    def test_unknown_folder_has_no_workspace(self):
        panes = [HerdrPane("w1:p1", "w1:t1", "w1", "/other", "", "")]
        assert herdr.workspace_for_cwd(panes, "/home/user/proj") is None


class TestHerdrPrepareSession:
    def test_requires_cwd(self):
        with pytest.raises(HerdrError, match="no folder"):
            herdr.prepare_session("claude", "x", "")

    def test_requires_herdr_on_path(self):
        with (
            patch.object(herdr.shutil, "which", return_value=None),
            pytest.raises(HerdrError, match="herdr not found"),
        ):
            herdr.prepare_session("claude", "x", "/home/user/proj")

    def test_rejects_unknown_provider(self):
        with (
            patch.object(herdr.shutil, "which", return_value="/usr/bin/herdr"),
            pytest.raises(HerdrError, match="unknown provider"),
        ):
            herdr.prepare_session("bogus", "x", "/home/user/proj")

    def test_creates_workspace_for_an_unknown_folder(self):
        calls, fake_run = _herdr_stub()
        with (
            patch.object(herdr.shutil, "which", return_value="/usr/bin/herdr"),
            patch.object(herdr.subprocess, "run", side_effect=fake_run),
        ):
            plan = herdr.prepare_session("claude", "def-456", "/home/user/proj")

        assert calls == [
            ["herdr", "pane", "list"],
            [
                "herdr",
                "workspace",
                "create",
                "--cwd",
                "/home/user/proj",
                "--label",
                "proj",
                "--no-focus",
            ],
            [
                "herdr",
                "pane",
                "run",
                "w9:p1",
                "claude --dangerously-skip-permissions --resume def-456",
            ],
        ]
        assert (plan.tab, plan.pane, plan.reused) == ("w9:t1", "w9:p1", False)
        assert plan.label == "proj"

    def test_adds_a_tab_to_the_folders_existing_workspace(self):
        calls, fake_run = _herdr_stub(
            panes=[_herdr_pane("w4:p1", cwd="/home/user/proj")],
            created={"tab": {"tab_id": "w4:t2"}, "root_pane": {"pane_id": "w4:p2"}},
        )
        with (
            patch.object(herdr.shutil, "which", return_value="/usr/bin/herdr"),
            patch.object(herdr.subprocess, "run", side_effect=fake_run),
        ):
            plan = herdr.prepare_session("codex", "ghi-789", "/home/user/proj")

        assert [
            "herdr",
            "tab",
            "create",
            "--workspace",
            "w4",
            "--cwd",
            "/home/user/proj",
            "--no-focus",
        ] in calls
        assert not any(c[1:3] == ["workspace", "create"] for c in calls)
        assert ["herdr", "pane", "run", "w4:p2", "codex resume ghi-789"] in calls
        assert (plan.tab, plan.pane) == ("w4:t2", "w4:p2")

    def test_reuses_the_pane_already_running_this_conversation(self):
        calls, fake_run = _herdr_stub(
            panes=[
                _herdr_pane("w4:p1", cwd="/home/user/proj"),
                _herdr_pane(
                    "w4:p2", cwd="/home/user/proj", agent="claude", session_id="abc-123"
                ),
            ]
        )
        with (
            patch.object(herdr.shutil, "which", return_value="/usr/bin/herdr"),
            patch.object(herdr.subprocess, "run", side_effect=fake_run),
        ):
            plan = herdr.prepare_session("claude", "abc-123", "/home/user/proj")

        assert plan.reused is True
        assert (plan.pane, plan.tab) == ("w4:p2", "w4:t2")
        # Nothing was created and resume was not typed a second time.
        assert calls == [["herdr", "pane", "list"]]

    def test_reuses_a_pane_sitting_on_an_ancestor_id(self, tmp_path):
        # herdr reports the id the pane's agent is on; claude --resume forked
        # the conversation, so that is an ancestor of the row the user picked.
        transcript = tmp_path / "new-id.jsonl"
        transcript.write_text(
            '{"type":"assistant","sessionId":"new-id","session_id":"old-id"}\n'
        )
        calls, fake_run = _herdr_stub(
            panes=[
                _herdr_pane(
                    "w4:p2", cwd="/home/user/proj", agent="claude", session_id="old-id"
                ),
            ]
        )
        with (
            patch.object(herdr.shutil, "which", return_value="/usr/bin/herdr"),
            patch.object(herdr.subprocess, "run", side_effect=fake_run),
        ):
            plan = herdr.prepare_session(
                "claude", "new-id", "/home/user/proj", content_path=str(transcript)
            )

        assert (plan.reused, plan.pane) == (True, "w4:p2")
        assert calls == [["herdr", "pane", "list"]]

    def test_reused_pane_is_focused_through_its_agent(self):
        # A tab can hold several panes; agent focus targets the right one.
        plan = herdr.HerdrPlan(
            label="proj",
            cwd="/home/user/proj",
            resume="x",
            session_id="abc",
            tab="w4:t2",
            pane="w4:p2",
            reused=True,
        )
        assert plan.switch_commands() == [["herdr", "agent", "focus", "w4:p2"]]

    def test_new_tab_is_focused_by_tab(self):
        # A pane created here has no agent yet to focus by name.
        plan = herdr.HerdrPlan(
            label="proj",
            cwd="/home/user/proj",
            resume="x",
            session_id="abc",
            tab="w9:t1",
            pane="w9:p1",
            reused=False,
        )
        assert plan.switch_commands() == [["herdr", "tab", "focus", "w9:t1"]]

    def test_attach_focuses_before_taking_the_terminal(self):
        # Focusing alone only rearranges a UI the plain terminal cannot see.
        plan = herdr.HerdrPlan(
            label="proj",
            cwd="/home/user/proj",
            resume="x",
            session_id="abc",
            tab="w9:t1",
            pane="w9:p1",
            reused=False,
        )
        assert plan.attach_commands() == [["herdr", "tab", "focus", "w9:t1"], ["herdr"]]

    def test_reports_the_message_from_a_failed_command(self):
        _calls, fake_run = _herdr_stub(fail=["pane", "run"])
        with (
            patch.object(herdr.shutil, "which", return_value="/usr/bin/herdr"),
            patch.object(herdr.subprocess, "run", side_effect=fake_run),
            pytest.raises(HerdrError, match="pane is not at a prompt"),
        ):
            herdr.prepare_session("claude", "x", "/home/user/proj")

    def test_rejects_a_create_that_reported_no_pane(self):
        _calls, fake_run = _herdr_stub(created={"tab": {"tab_id": "w9:t1"}})
        with (
            patch.object(herdr.shutil, "which", return_value="/usr/bin/herdr"),
            patch.object(herdr.subprocess, "run", side_effect=fake_run),
            pytest.raises(HerdrError, match="root pane"),
        ):
            herdr.prepare_session("claude", "x", "/home/user/proj")

    def test_rejects_non_json_output(self):
        def fake_run(cmd, *a, **kw):
            return subprocess.CompletedProcess(cmd, 0, stdout="not json", stderr="")

        with (
            patch.object(herdr.shutil, "which", return_value="/usr/bin/herdr"),
            patch.object(herdr.subprocess, "run", side_effect=fake_run),
            pytest.raises(HerdrError, match="no JSON"),
        ):
            herdr.prepare_session("claude", "x", "/home/user/proj")

    def test_accepts_a_silent_success_from_pane_run(self):
        # `pane run` only sends input: it prints nothing and answers through
        # its exit status, so demanding a result body would fail every handoff.
        calls, fake_run = _herdr_stub()
        with (
            patch.object(herdr.shutil, "which", return_value="/usr/bin/herdr"),
            patch.object(herdr.subprocess, "run", side_effect=fake_run),
        ):
            plan = herdr.prepare_session("claude", "x", "/home/user/proj")

        assert calls[-1][1:3] == ["pane", "run"]
        assert plan.pane == "w9:p1"


# ---------------------------------------------------------------------------
# Multiplexer availability — which handoff destinations are live
# ---------------------------------------------------------------------------


class TestTmuxAvailable:
    def test_not_installed(self):
        with patch.object(tmux.shutil, "which", return_value=None):
            assert tmux.available() is False

    def test_inside_tmux_needs_no_probe(self, monkeypatch):
        monkeypatch.setenv("TMUX", "/tmp/tmux-501/default,123,0")
        with (
            patch.object(tmux.shutil, "which", return_value="/usr/bin/tmux"),
            patch.object(
                tmux.subprocess, "run", side_effect=AssertionError("must not probe")
            ),
        ):
            assert tmux.available() is True

    def test_server_running(self, monkeypatch):
        monkeypatch.delenv("TMUX", raising=False)
        with (
            patch.object(tmux.shutil, "which", return_value="/usr/bin/tmux"),
            patch.object(
                tmux.subprocess, "run", return_value=subprocess.CompletedProcess([], 0)
            ),
        ):
            assert tmux.available() is True

    def test_installed_but_no_server(self, monkeypatch):
        # The case that hides `t` for someone living in another multiplexer.
        monkeypatch.delenv("TMUX", raising=False)
        with (
            patch.object(tmux.shutil, "which", return_value="/usr/bin/tmux"),
            patch.object(
                tmux.subprocess, "run", return_value=subprocess.CompletedProcess([], 1)
            ),
        ):
            assert tmux.available() is False

    def test_a_wedged_probe_does_not_hang_the_browser(self, monkeypatch):
        monkeypatch.delenv("TMUX", raising=False)
        with (
            patch.object(tmux.shutil, "which", return_value="/usr/bin/tmux"),
            patch.object(
                tmux.subprocess, "run", side_effect=subprocess.TimeoutExpired("tmux", 2)
            ),
        ):
            assert tmux.available() is False


class TestHerdrAvailable:
    @staticmethod
    def _status(**fields):
        payload = json.dumps(fields)
        return lambda *a, **kw: subprocess.CompletedProcess(
            [], 0, stdout=payload, stderr=""
        )

    def test_not_installed(self):
        with patch.object(herdr.shutil, "which", return_value=None):
            assert herdr.available() is False

    def test_server_running_and_compatible(self):
        with (
            patch.object(herdr.shutil, "which", return_value="/usr/bin/herdr"),
            patch.object(
                herdr.subprocess,
                "run",
                side_effect=self._status(running=True, compatible=True),
            ),
        ):
            assert herdr.available() is True

    def test_server_not_running(self):
        # `status server` exits 0 either way, so the JSON is what decides.
        with (
            patch.object(herdr.shutil, "which", return_value="/usr/bin/herdr"),
            patch.object(
                herdr.subprocess, "run", side_effect=self._status(running=False)
            ),
        ):
            assert herdr.available() is False

    def test_incompatible_protocol_is_not_available(self):
        # Every command below goes through the socket; an unspoken protocol
        # fails them just as a stopped server does.
        with (
            patch.object(herdr.shutil, "which", return_value="/usr/bin/herdr"),
            patch.object(
                herdr.subprocess,
                "run",
                side_effect=self._status(running=True, compatible=False),
            ),
        ):
            assert herdr.available() is False

    def test_unreadable_status_is_not_available(self):
        with (
            patch.object(herdr.shutil, "which", return_value="/usr/bin/herdr"),
            patch.object(
                herdr.subprocess,
                "run",
                return_value=subprocess.CompletedProcess([], 0, stdout="", stderr=""),
            ),
        ):
            assert herdr.available() is False

    def test_a_wedged_socket_does_not_hang_the_browser(self):
        with (
            patch.object(herdr.shutil, "which", return_value="/usr/bin/herdr"),
            patch.object(
                herdr.subprocess,
                "run",
                side_effect=subprocess.TimeoutExpired("herdr", 2),
            ),
        ):
            assert herdr.available() is False


class TestAvailableTargets:
    def test_keys_are_distinct(self):
        keys = [t.key for t in multiplexer.TARGETS]
        assert len(set(keys)) == len(keys)

    def test_only_running_multiplexers_are_offered(self):
        with (
            patch.object(tmux, "available", return_value=False),
            patch.object(herdr, "available", return_value=True),
        ):
            assert multiplexer.available_names() == ["herdr"]

    def test_both_running_keeps_tmux_first(self):
        with (
            patch.object(tmux, "available", return_value=True),
            patch.object(herdr, "available", return_value=True),
        ):
            assert multiplexer.available_names() == ["tmux", "herdr"]

    def test_neither_running(self):
        with (
            patch.object(tmux, "available", return_value=False),
            patch.object(herdr, "available", return_value=False),
        ):
            assert multiplexer.available_targets() == []

    def test_each_target_carries_its_own_error_type(self):
        # _open_in_multiplexer catches through the target, so a mismatch here
        # would surface as an unhandled exception in the TUI.
        for target in multiplexer.TARGETS:
            assert issubclass(target.error, Exception)
            assert target.error.__module__ == target.module.__name__


@pytest.mark.skipif(not _HAS_TEXTUAL, reason="textual not installed")
@pytest.mark.asyncio
class TestSelectionCopy:
    """Copying a mouse text selection (ctrl+c / cmd+c) must reach the system
    clipboard helper, not just Textual's OSC 52 escape, and confirm in the
    status bar."""

    async def test_selection_copy_routes_through_system_clipboard(self, monkeypatch):
        import session_browser.app as app_module

        app, fake = _make_app_with_rows()
        copied = []
        monkeypatch.setattr(
            app_module,
            "copy_to_clipboard",
            lambda text: (copied.append(text), True)[1],
        )
        async with app.run_test() as pilot:
            await _install_fake_sessions(app, pilot, fake)
            # Screen.action_copy_text (the built-in ctrl+c/cmd+c binding)
            # funnels the selected text through App.copy_to_clipboard.
            app.copy_to_clipboard("selected text")
            await pilot.pause()
            assert copied == ["selected text"]
            assert app.query_one("#status-bar").has_class("-flash-ok")

    async def test_selection_copy_flags_missing_clipboard_tool(self, monkeypatch):
        import session_browser.app as app_module

        app, fake = _make_app_with_rows()
        monkeypatch.setattr(app_module, "copy_to_clipboard", lambda text: False)
        async with app.run_test() as pilot:
            await _install_fake_sessions(app, pilot, fake)
            app.copy_to_clipboard("selected text")
            await pilot.pause()
            assert app.query_one("#status-bar").has_class("-flash-err")

    async def test_status_bar_is_visible_not_under_footer(self):
        """#status-bar must occupy its own row: with dock:bottom it overlapped
        the Footer (same-edge docks stack on the same row) and every copy
        confirmation was silently invisible."""
        from textual.widgets import Footer

        app, fake = _make_app_with_rows()
        async with app.run_test() as pilot:
            await _install_fake_sessions(app, pilot, fake)
            status = app.query_one("#status-bar")
            footer = app.query_one(Footer)
            assert status.region.height == 1
            assert not status.region.overlaps(footer.region)
