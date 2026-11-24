"""Tests for session_browser."""

from __future__ import annotations

import asyncio
import contextlib
import gc
import json
import os
import re
import sqlite3
import subprocess
import sys
import textwrap
import time
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from session_browser import compressed, herdr, herdr_core, multiplexer, tmux
from session_browser.discovery import (
    Session,
    _codex_repo_name,
    _epoch_ms_to_iso,
    _file_mtime_iso,
    _last_activity_iso,
    _repo_name,
    _scan_codex_files,
    discover_all,
    scan_claude,
    scan_codex,
    scan_opencode,
    scan_pi,
)
from session_browser.herdr import HerdrError, HerdrPane, workspace_label_for_path
from session_browser.resume import (
    build_chat_export,
    copy_to_clipboard,
    export_chat_to_file,
    resume_command,
    resume_flags,
    resume_target,
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
    search_sessions,
)

# ---------------------------------------------------------------------------
# Resume command tests
# ---------------------------------------------------------------------------


class TestResumeCommand:
    def test_claude(self):
        # No transcript, so no mode: the CLI starts as its config says.
        assert resume_command("claude", "def-456") == "claude --resume def-456"

    def test_codex(self):
        assert resume_command("codex", "ghi-789") == "codex resume ghi-789"

    def test_with_cwd(self):
        result = resume_command("codex", "abc", "/home/user/project")
        assert result == "cd /home/user/project && codex resume abc"

    def test_cwd_with_spaces(self):
        result = resume_command("claude", "x", "/tmp/my project")
        assert "cd '/tmp/my project'" in result
        assert "claude --resume x" in result

    def test_unknown_provider(self):
        result = resume_command("unknown", "id-1")
        assert "unknown provider" in result

    def test_no_cwd(self):
        result = resume_command("codex", "abc")
        assert not result.startswith("cd")


def _claude_mode_transcript(path: Path, *records: dict) -> str:
    """A Claude transcript opening the way Claude Code 2.1.331 writes one."""
    lines = [
        {"type": "mode", "mode": "normal", "sessionId": "s1"},
        *records,
        {
            "type": "user",
            "message": {"role": "user", "content": "hello"},
            "sessionId": "s1",
            "timestamp": "2026-11-20T10:00:00.000Z",
        },
    ]
    path.write_text("".join(json.dumps(r) + "\n" for r in lines))
    return str(path)


def _permission_mode(mode: str) -> dict:
    return {"type": "permission-mode", "permissionMode": mode, "sessionId": "s1"}


def _codex_mode_rollout(path: Path, approval, sandbox) -> str:
    """A rollout whose first turn_context carries these policies, as 0.172.0
    writes it: session_meta, then the instruction items, then the context."""
    lines = [
        {"type": "session_meta", "payload": {"id": "t1", "cwd": "/w"}},
        {
            "type": "response_item",
            "payload": {"type": "message", "role": "developer", "content": []},
        },
        {
            "type": "turn_context",
            "payload": {
                "cwd": "/w",
                "approval_policy": approval,
                "approvals_reviewer": "user",
                "sandbox_policy": sandbox,
                "model": "gpt-6.1-sol",
            },
        },
        {
            "type": "turn_context",
            "payload": {
                "approval_policy": "on-request",
                "sandbox_policy": {"type": "read-only"},
            },
        },
    ]
    path.write_text("".join(json.dumps(r) + "\n" for r in lines))
    return str(path)


class TestResumeMode:
    """A session resumes in the mode it was started in, read off its file."""

    def test_claude_plan_mode_is_kept(self, tmp_path):
        path = _claude_mode_transcript(tmp_path / "s1.jsonl", _permission_mode("plan"))
        assert resume_command("claude", "s1", content_path=path) == (
            "claude --permission-mode plan --resume s1"
        )

    @pytest.mark.parametrize("mode", ["default", "acceptEdits", "auto", "manual"])
    def test_claude_modes_pass_through_as_recorded(self, tmp_path, mode):
        path = _claude_mode_transcript(tmp_path / "s1.jsonl", _permission_mode(mode))
        assert resume_flags("claude", path) == ("--permission-mode", mode)

    def test_claude_bypass_keeps_the_skip_permissions_flag(self, tmp_path):
        path = _claude_mode_transcript(
            tmp_path / "s1.jsonl", _permission_mode("bypassPermissions")
        )
        assert resume_command("claude", "s1", content_path=path) == (
            "claude --dangerously-skip-permissions --resume s1"
        )

    def test_the_first_recorded_mode_is_the_one_it_started_in(self, tmp_path):
        path = _claude_mode_transcript(
            tmp_path / "s1.jsonl",
            _permission_mode("default"),
            _permission_mode("bypassPermissions"),
        )
        assert resume_flags("claude", path) == ("--permission-mode", "default")

    def test_a_mode_carried_only_on_a_user_record_counts(self, tmp_path):
        # A session /clear opened records no permission-mode record; a later
        # user record can still carry the mode.
        path = _claude_mode_transcript(
            tmp_path / "s1.jsonl",
            {
                "type": "user",
                "permissionMode": "acceptEdits",
                "message": {"role": "user", "content": "go"},
            },
        )
        assert resume_flags("claude", path) == ("--permission-mode", "acceptEdits")

    def test_no_recorded_mode_means_no_mode_flag(self, tmp_path):
        path = _claude_mode_transcript(tmp_path / "s1.jsonl")
        assert resume_command("claude", "s1", content_path=path) == (
            "claude --resume s1"
        )

    def test_a_quoted_or_nested_mode_is_not_the_sessions(self, tmp_path):
        # A tool result that shows another transcript holds the key escaped,
        # and a structured result can nest it; neither is this session's mode.
        quoted = {
            "type": "user",
            "message": {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "content": '{"permissionMode":"bypassPermissions"}',
                    }
                ],
            },
            "toolUseResult": {"permissionMode": "bypassPermissions"},
        }
        path = _claude_mode_transcript(
            tmp_path / "s1.jsonl", quoted, _permission_mode("plan")
        )
        assert resume_flags("claude", path) == ("--permission-mode", "plan")

    def test_a_mode_this_cli_does_not_take_is_left_out(self, tmp_path):
        path = _claude_mode_transcript(
            tmp_path / "s1.jsonl", _permission_mode("someFutureMode")
        )
        assert resume_flags("claude", path) == ()

    def test_an_unreadable_transcript_means_no_mode_flag(self, tmp_path):
        missing = str(tmp_path / "gone.jsonl")
        assert resume_command("claude", "s1", content_path=missing) == (
            "claude --resume s1"
        )

    def test_codex_full_access_resumes_bypassed(self, tmp_path):
        path = _codex_mode_rollout(
            tmp_path / "r.jsonl", "never", {"type": "danger-full-access"}
        )
        assert resume_command("codex", "t1", content_path=path) == (
            "codex resume --dangerously-bypass-approvals-and-sandbox t1"
        )

    def test_codex_sandboxed_session_keeps_its_policies(self, tmp_path):
        path = _codex_mode_rollout(
            tmp_path / "r.jsonl",
            "on-request",
            {"type": "workspace-write", "network_access": False},
        )
        assert resume_command("codex", "t1", content_path=path) == (
            "codex resume --sandbox workspace-write --ask-for-approval on-request t1"
        )

    def test_codex_policy_this_cli_dropped_is_left_out(self, tmp_path):
        path = _codex_mode_rollout(
            tmp_path / "r.jsonl", "untrusted", {"type": "read-only"}
        )
        assert resume_flags("codex", path) == ("--sandbox", "read-only")

    def test_codex_compressed_rollout_is_read(self, tmp_path):
        plain = tmp_path / "r.jsonl"
        _codex_mode_rollout(plain, "never", {"type": "danger-full-access"})
        packed = tmp_path / ("r.jsonl" + compressed.SUFFIX)
        packed.write_bytes(compressed.zstd.compress(plain.read_bytes()))
        plain.unlink()
        assert resume_flags("codex", str(packed)) == (
            "--dangerously-bypass-approvals-and-sandbox",
        )

    @pytest.mark.parametrize("provider", ["opencode", "pi"])
    def test_providers_with_no_mode_get_no_flags(self, tmp_path, provider):
        path = _claude_mode_transcript(tmp_path / "s.jsonl", _permission_mode("plan"))
        assert resume_flags(provider, path) == ()

    def test_subagent_resumes_its_root_in_the_roots_mode(self, tmp_path):
        project = tmp_path / "proj"
        (project / "root-1" / "subagents").mkdir(parents=True)
        root = _claude_mode_transcript(
            project / "root-1.jsonl", _permission_mode("plan")
        )
        child = Session(
            id="agent-a1",
            provider="claude",
            content_path=str(project / "root-1" / "subagents" / "agent-a1.jsonl"),
            parent_id="root-1",
            subagent_kind="Explore",
        )
        assert resume_target(child) == ("root-1", root)
        plain = Session(id="s9", provider="codex", content_path="/r.jsonl")
        assert resume_target(plain) == ("s9", "/r.jsonl")

    def test_tmux_plan_carries_the_mode(self, tmp_path):
        path = _claude_mode_transcript(tmp_path / "s1.jsonl", _permission_mode("plan"))
        plan = build_plan(
            "claude", "s1", "/home/user/proj", session_exists=True, content_path=path
        )
        assert plan.resume == "claude --permission-mode plan --resume s1"


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


class TestClaudeContinuationAndLiveness:
    """One conversation resumed into a new file is linked both ways, and a
    session a running Claude Code process holds says so."""

    @staticmethod
    def _transcript(path, sid, *, tail=()):
        records = [
            {
                "type": "user",
                "sessionId": sid,
                "uuid": "u-1",
                "message": {"role": "user", "content": "plan the blockside work"},
                "cwd": "/Users/test/blockside",
                "timestamp": "2026-10-15T18:57:20Z",
            },
            {
                "type": "assistant",
                "sessionId": sid,
                "message": {"content": [{"type": "text", "text": "ok"}]},
                "timestamp": "2026-10-15T19:00:00Z",
            },
            *tail,
        ]
        path.write_text("".join(json.dumps(r) + "\n" for r in records))

    def test_a_continued_session_is_linked_both_ways(self, tmp_path):
        """The record exists only at the end of the OLD file, after its last
        turn. The new file -- a copy of the old history plus more -- learns
        its predecessor by inversion. A successor no longer on disk is still
        named, with nothing to point back."""
        project = tmp_path / ".claude" / "projects" / "-Users-test-blockside"
        project.mkdir(parents=True)

        def continued(target):
            return {
                "type": "continued-in",
                "timestamp": "2026-10-15T21:52:42Z",
                "sessionId": "x",
                "continuedInSessionId": target,
            }

        self._transcript(project / "old.jsonl", "old", tail=[continued("new")])
        self._transcript(project / "new.jsonl", "new")
        self._transcript(project / "orphan.jsonl", "orphan", tail=[continued("gone")])

        with patch("session_browser.discovery.Path.home", return_value=tmp_path):
            by_id = {s.id: s for s in scan_claude()}

        links = {sid: (s.continued_in, s.continued_from) for sid, s in by_id.items()}
        assert links == {
            "old": ("new", None),
            "new": (None, "old"),
            "orphan": ("gone", None),
        }
        # Still the last TURN, not the continued-in record after it.
        assert by_id["old"].updated_at == "2026-10-15T19:00:00Z"

    def test_live_comes_from_the_running_process_registry(self, tmp_path):
        """~/.claude/sessions/<pid>.json names the session each running
        process holds. A file left by a process that has gone does not make
        its session live, and a subagent's liveness is its root's, so it is
        not claimed either way."""
        project = tmp_path / ".claude" / "projects" / "-Users-test-blockside"
        project.mkdir(parents=True)
        for sid in ("running", "finished", "crashed"):
            self._transcript(project / f"{sid}.jsonl", sid)
        sub = project / "running" / "subagents"
        sub.mkdir(parents=True)
        self._transcript(sub / "agent-a1.jsonl", "running")

        registry = tmp_path / ".claude" / "sessions"
        registry.mkdir()
        (registry / f"{os.getpid()}.json").write_text(
            json.dumps({"pid": os.getpid(), "sessionId": "running"})
        )
        # A pid far above any the OS hands out: its process is gone.
        (registry / "999999.json").write_text(
            json.dumps({"pid": 999999, "sessionId": "crashed"})
        )
        (registry / "garbage.json").write_text("{not json")

        with patch("session_browser.discovery.Path.home", return_value=tmp_path):
            live = {s.id: s.live for s in scan_claude()}

        assert live == {
            "running": True,
            "finished": False,
            "crashed": False,
            "agent-a1": None,
        }


def _away_summary(text: str, ts: str = "2026-11-22T20:03:48.618Z") -> dict:
    """A recap as Claude Code 2.1.330 writes one."""
    return {
        "parentUuid": "p-1",
        "isSidechain": False,
        "type": "system",
        "subtype": "away_summary",
        "content": text,
        "timestamp": ts,
        "uuid": "r-1",
        "isMeta": False,
        "userType": "external",
        "entrypoint": "cli",
        "cwd": "/Users/test/blockside",
        "sessionId": "s1",
    }


class TestClaudeRecap:
    """A session whose newest record is Claude Code's own recap is named by
    the recap; anything said after the recap puts the first prompt back."""

    RECAP = (
        "You wanted the blockside plan written up, so it is drafted and "
        "committed. Next, read it and say which option to build."
    )

    @staticmethod
    def _scan(tmp_path, *tail, subagent=False):
        project = tmp_path / ".claude" / "projects" / "-Users-test-blockside"
        project.mkdir(parents=True, exist_ok=True)
        path = project / "s1.jsonl"
        if subagent:
            (project / "s1.jsonl").write_text("")
            path = project / "s1" / "subagents" / "agent-a1.jsonl"
            path.parent.mkdir(parents=True)
            path.with_name("agent-a1.meta.json").write_text(
                json.dumps({"agentType": "Explore", "description": "Find it"})
            )
        TestClaudeContinuationAndLiveness._transcript(path, "s1", tail=tail)
        with patch("session_browser.discovery.Path.home", return_value=tmp_path):
            return {s.id: s for s in scan_claude()}

    def test_a_recap_nothing_followed_is_the_summary(self, tmp_path):
        found = self._scan(
            tmp_path,
            _away_summary(self.RECAP + " (disable recaps in /config)"),
            {"type": "ai-title", "aiTitle": "Blockside plan", "sessionId": "s1"},
        )
        session = found["s1"]
        assert session.summary_is_recap
        # The settings pointer is chrome, not part of the recap.
        assert session.summary == self.RECAP[:120]
        # A recap is not a turn: the session's activity is still its last one.
        assert session.updated_at == "2026-10-15T19:00:00Z"

    def test_the_newest_recap_wins(self, tmp_path):
        found = self._scan(
            tmp_path,
            _away_summary("An older recap. (turn off recaps in /config)"),
            _away_summary(self.RECAP),
        )
        assert found["s1"].summary == self.RECAP[:120]

    def test_a_prompt_after_the_recap_keeps_the_first_prompt(self, tmp_path):
        prompt = {
            "type": "user",
            "message": {"role": "user", "content": "ok, now build option two"},
            "timestamp": "2026-10-16T09:00:00Z",
        }
        session = self._scan(tmp_path, _away_summary(self.RECAP), prompt)["s1"]
        assert not session.summary_is_recap
        assert session.summary == "plan the blockside work"

    def test_work_after_the_recap_keeps_the_first_prompt(self, tmp_path):
        # The agent went on (a background task, a goal loop) after the recap,
        # so it no longer says where things stand.
        reply = {
            "type": "assistant",
            "message": {"content": [{"type": "text", "text": "task done"}]},
            "timestamp": "2026-10-16T09:00:00Z",
        }
        session = self._scan(tmp_path, _away_summary(self.RECAP), reply)["s1"]
        assert not session.summary_is_recap
        assert session.summary == "plan the blockside work"

    def test_a_subagent_keeps_its_spawners_label(self, tmp_path):
        found = self._scan(tmp_path, _away_summary(self.RECAP), subagent=True)
        child = found["agent-a1"]
        assert not child.summary_is_recap
        assert child.summary == "Find it"

    def test_the_recap_survives_the_scan_memo(self, tmp_path):
        first = self._scan(tmp_path, _away_summary(self.RECAP))["s1"]
        with patch("session_browser.discovery.Path.home", return_value=tmp_path):
            again = {s.id: s for s in scan_claude()}["s1"]
        assert first.summary_is_recap and again.summary_is_recap
        assert again.summary == first.summary

    def test_a_malformed_recap_is_ignored(self, tmp_path):
        record = _away_summary(self.RECAP)
        record["content"] = ["not", "text"]
        session = self._scan(tmp_path, record)["s1"]
        assert not session.summary_is_recap
        assert session.summary == "plan the blockside work"


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
        """A transcript with no timestamped user/assistant turn (and no
        first-message created_at) falls back to file mtime, not an empty
        string."""
        projects = tmp_path / ".claude" / "projects" / "-Users-test"
        projects.mkdir(parents=True)
        f = projects / "s2.jsonl"
        f.write_text(
            "\n".join(
                json.dumps(x)
                for x in [
                    {"type": "system", "timestamp": "2026-06-30T08:00:00.000Z"},
                    {"type": "user", "message": {"content": "no timestamp"}},
                ]
            )
            + "\n"
        )

        with patch("session_browser.discovery.Path.home", return_value=tmp_path):
            sessions = scan_claude()

        assert sessions[0].updated_at == _file_mtime_iso(f)
        assert sessions[0].updated_at  # non-empty

    @staticmethod
    def _turn(text, **extra):
        return json.dumps(
            {
                "type": "user",
                "message": {"role": "user", "content": text},
                "cwd": "/Users/test/project",
                "timestamp": "2026-09-25T10:00:00Z",
                **extra,
            }
        )

    def _subagent(self, where, agent_id, meta):
        where.mkdir(parents=True, exist_ok=True)
        # sessionId names the ROOT session on every subagent record; a scanner
        # that took the id from it would file the child as its parent.
        (where / f"agent-{agent_id}.jsonl").write_text(
            self._turn("the brief", sessionId="root", isSidechain=True) + "\n"
        )
        if meta is not None:
            (where / f"agent-{agent_id}.meta.json").write_text(meta)

    def test_subagents_are_discovered_and_linked(self, tmp_path):
        """Children sit under the root's own directory: flat at any depth for
        the Agent tool, a level further down for the Workflow tool. The
        sidecar names the spawning agent of a grandchild and the kind."""
        project = tmp_path / ".claude" / "projects" / "-Users-test"
        project.mkdir(parents=True)
        (project / "root.jsonl").write_text(self._turn("top") + "\n")
        subagents = project / "root" / "subagents"
        self._subagent(
            subagents,
            "a1",
            json.dumps({"agentType": "Explore", "description": "map the repo"}),
        )
        self._subagent(
            subagents,
            "a2",
            json.dumps(
                {"agentType": "general-purpose", "parentAgentId": "a1", "spawnDepth": 2}
            ),
        )
        self._subagent(
            subagents / "workflows" / "wf_1",
            "w1",
            json.dumps({"agentType": "workflow-subagent", "description": "zeta"}),
        )
        # A directory with no subagents/ in it -- tool results only -- is not
        # a session and holds none.
        (project / "other" / "tool-results").mkdir(parents=True)

        with patch("session_browser.discovery.Path.home", return_value=tmp_path):
            got = {s.id: s for s in scan_claude()}

        assert set(got) == {"root", "agent-a1", "agent-a2", "agent-w1"}
        link = {sid: (s.parent_id, s.subagent_kind) for sid, s in got.items()}
        assert link == {
            "root": ("", ""),
            "agent-a1": ("root", "Explore"),
            "agent-a2": ("agent-a1", "general-purpose"),
            "agent-w1": ("root", "workflow-subagent"),
        }
        # The spawner's description labels the row; without one, the brief.
        assert got["agent-a1"].summary == "map the repo"
        assert got["agent-a2"].summary == "the brief"
        assert got["agent-a1"].cwd == "/Users/test/project"

    def test_subagent_without_a_readable_sidecar_is_still_a_subagent(
        self, tmp_path, caplog
    ):
        """The directory alone says it is a child of the root session."""
        project = tmp_path / ".claude" / "projects" / "-Users-test"
        project.mkdir(parents=True)
        (project / "root.jsonl").write_text(self._turn("top") + "\n")
        self._subagent(project / "root" / "subagents", "gone", None)
        self._subagent(project / "root" / "subagents", "bad", "{not json")

        with patch("session_browser.discovery.Path.home", return_value=tmp_path):
            got = {s.id: s for s in scan_claude()}

        for sid in ("agent-gone", "agent-bad"):
            assert (got[sid].parent_id, got[sid].subagent_kind) == ("root", "subagent")
        assert "no readable sidecar" in caplog.text

    def _write(self, tmp_path, name, lines):
        projects = tmp_path / ".claude" / "projects" / "-Users-test-driver"
        projects.mkdir(parents=True, exist_ok=True)
        (projects / f"{name}.jsonl").write_text("\n".join(lines) + "\n")

    def _ids(self, tmp_path):
        with patch("session_browser.discovery.Path.home", return_value=tmp_path):
            return {s.id for s in scan_claude()}

    def test_resume_stub_with_no_conversation_is_not_a_session(self, tmp_path):
        """Opening Claude Code only to run /resume leaves a one-line
        bridge-session file behind. Listed, it sits beside the session it was
        opened to reach -- same project, same age -- and resuming it fails
        with "No conversation found"."""
        bridge = {"type": "bridge-session", "sessionId": "stub", "lastSequenceNum": 0}
        self._write(tmp_path, "stub", [json.dumps(bridge), json.dumps(bridge)])

        assert self._ids(tmp_path) == set()

    def test_file_of_only_local_commands_and_bookkeeping_is_not_a_session(
        self, tmp_path
    ):
        """The same shape with a slash command run in it: system and
        bookkeeping records, still no turn."""
        lines = [
            {"type": "mode", "mode": "normal"},
            {"type": "permission-mode", "permissionMode": "default"},
            {"type": "system", "subtype": "local_command", "content": "/skills"},
            {"type": "cost-state", "totalCostUSD": 0},
            {"type": "last-prompt", "leafUuid": "x"},
        ]
        self._write(tmp_path, "cmds", [json.dumps(x) for x in lines])

        assert self._ids(tmp_path) == set()

    def test_file_with_a_turn_anywhere_is_kept(self, tmp_path):
        """Only a first *non-meta* user message feeds the summary, but any
        user or assistant record proves there is a conversation to show."""
        self._write(
            tmp_path,
            "assistant-only",
            [json.dumps({"type": "assistant", "message": {"content": "hi"}})],
        )
        self._write(
            tmp_path,
            "meta-only",
            [json.dumps({"type": "user", "isMeta": True, "message": {"content": "x"}})],
        )

        assert self._ids(tmp_path) == {"assistant-only", "meta-only"}

    def test_file_with_an_undecodable_line_is_kept(self, tmp_path):
        """A damaged line may have been a turn. Dropping the file would hide
        the corruption the zero-entries warning exists to report."""
        self._write(
            tmp_path,
            "damaged",
            [json.dumps({"type": "bridge-session"}), '{"type": "user", "mess'],
        )

        assert self._ids(tmp_path) == {"damaged"}

    def test_file_with_no_records_at_all_is_kept(self, tmp_path):
        """An empty file is as likely a failed write as an empty session, so
        it stays visible and gets reported as zero entries, not dropped."""
        self._write(tmp_path, "blank", [""])

        assert self._ids(tmp_path) == {"blank"}


class TestClaudeSummary:
    """Claude Code records a slash command as a user message of markup and a
    paste wrapped in <pasted_content> tags. Neither may reach the summary as
    XML: it is what the list, the TUI table and the delete confirmation show.
    """

    CMD = (
        "<command-name>/goal</command-name>\n"
        "            <command-message>goal</command-message>\n"
        "            <command-args>work the open tickets yolo style</command-args>"
    )
    BARE = (
        "<command-name>/clear</command-name>\n"
        "            <command-message>clear</command-message>\n"
        "            <command-args></command-args>"
    )

    def _summary(self, tmp_path, *contents, meta_first=False):
        projects = tmp_path / ".claude" / "projects" / "-Users-test"
        projects.mkdir(parents=True, exist_ok=True)
        lines = []
        if meta_first:
            lines.append(
                {"type": "user", "isMeta": True, "message": {"content": "caveat"}}
            )
        for content in contents:
            lines.append(
                {
                    "type": "user",
                    "message": {"role": "user", "content": content},
                    "cwd": "/Users/test/project",
                    "timestamp": "2026-10-15T10:00:00Z",
                }
            )
        (projects / "s1.jsonl").write_text(
            "\n".join(json.dumps(x) for x in lines) + "\n"
        )
        with patch("session_browser.discovery.Path.home", return_value=tmp_path):
            (session,) = scan_claude()
        return session.summary

    def test_a_command_with_arguments_reads_as_typed(self, tmp_path):
        assert (
            self._summary(tmp_path, self.CMD, "a later prompt")
            == "/goal work the open tickets yolo style"
        )

    def test_tag_order_does_not_matter(self, tmp_path):
        content = (
            "<command-message>walkthrough</command-message> "
            "<command-name>/walkthrough</command-name> "
            "<command-args>build me one</command-args>"
        )
        assert self._summary(tmp_path, content) == "/walkthrough build me one"

    def test_a_bare_command_yields_to_the_prompt_after_it(self, tmp_path):
        """/clear and /model say nothing about the session they opened; the
        first real prompt does. The command's own output is not a prompt."""
        assert (
            self._summary(
                tmp_path,
                self.BARE,
                "<local-command-stdout>Set model to opus</local-command-stdout>",
                self.BARE,
                "why does the delete view wrap?",
                meta_first=True,
            )
            == "why does the delete view wrap?"
        )

    def test_a_bare_command_with_nothing_after_it_names_itself(self, tmp_path):
        assert self._summary(tmp_path, self.BARE) == "/clear"

    def test_cleaning_happens_before_the_cut(self, tmp_path):
        """Cutting first left summaries ending '<command-args></com'."""
        long_args = "x " * 200
        content = self.CMD.replace("work the open tickets yolo style", long_args)
        summary = self._summary(tmp_path, content)
        assert summary.startswith("/goal x x")
        assert len(summary) == 120
        assert "<" not in summary

    def test_a_paste_loses_its_wrapper(self, tmp_path):
        content = (
            '  <pasted_content id="3540">\nTraceback (most recent call last)\n'
            "</pasted_content>\n\nwhat is this"
        )
        assert (
            self._summary(tmp_path, content)
            == "Traceback (most recent call last) what is this"
        )

    def test_an_ordinary_prompt_is_untouched(self, tmp_path):
        """Prose that merely mentions a tag is the user speaking."""
        content = "why does <command-name> show up\nin the list?"
        assert self._summary(tmp_path, content) == content.replace("\n", " ")


class TestScanCodex:
    def test_the_file_scan_reads_a_model_the_head_already_passes(self, tmp_path):
        """A turn_context above the first prompt names the model for free; one
        below it would cost reading past the prompt, so it is not read."""
        day = tmp_path / ".codex" / "sessions" / "2026" / "04" / "10"
        day.mkdir(parents=True)

        def rollout(sid, context_first):
            meta = {"type": "session_meta", "payload": {"id": sid, "cwd": "/p"}}
            context = {"type": "turn_context", "payload": {"model": "gpt-6.1-sol"}}
            prompt = {
                "type": "event_msg",
                "payload": {"type": "user_message", "message": "hello"},
            }
            records = (
                [meta, context, prompt] if context_first else [meta, prompt, context]
            )
            path = day / f"rollout-2026-04-10T13-00-00-{sid}.jsonl"
            path.write_text("\n".join(json.dumps(r) for r in records) + "\n")

        rollout("above", True)
        rollout("below", False)
        with patch("session_browser.discovery.Path.home", return_value=tmp_path):
            found = {s.id: (s.model, s.summary) for s in scan_codex()}
        assert found == {"above": ("gpt-6.1-sol", "hello"), "below": (None, "hello")}

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
                    "git": {
                        "branch": "main",
                        "repository_url": "https://example.test/org/true-project.git",
                    },
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
        assert sessions[0].repository == "true-project"

    def test_a_subagent_keeps_its_own_id_not_its_parents(self, tmp_path):
        """payload.session_id names the PARENT on a subagent rollout.

        Measured 2026-09-08: of 74 Codex subagent rollouts on this machine,
        71 carried session_id != id, and every one of those named the parent.
        Deriving the session id from it would file the child under its
        parent's identity — `codex resume` on the child would resume the
        parent, and one session would be reported twice. agent-sessions
        shipped exactly that bug; this test is why we cannot.
        """
        sessions_dir = tmp_path / ".codex" / "sessions" / "2026" / "04" / "10"
        sessions_dir.mkdir(parents=True)
        meta = json.dumps(
            {
                "type": "session_meta",
                "timestamp": "2026-04-10T13:00:00Z",
                "payload": {
                    "id": "child-own-id",
                    "session_id": "parent-uuid",
                    "cwd": "/Users/test/project",
                    "source": {
                        "subagent": {
                            "thread_spawn": {"parent_thread_id": "parent-uuid"}
                        }
                    },
                },
            }
        )
        (sessions_dir / "rollout-2026-04-10T13-00-00-child-own-id.jsonl").write_text(
            meta + "\n"
        )

        with patch("session_browser.discovery.Path.home", return_value=tmp_path):
            sessions = scan_codex()

        assert [s.id for s in sessions] == ["child-own-id"]

    def test_file_fallback_uses_origin_for_a_worktree(self, tmp_path):
        sessions_dir = tmp_path / ".codex" / "sessions" / "2026" / "04" / "10"
        sessions_dir.mkdir(parents=True)
        meta = {
            "type": "session_meta",
            "timestamp": "2026-04-10T13:00:00Z",
            "payload": {
                "id": "worktree-session",
                "cwd": "/Users/test/Projects/worktrees/codex",
                "git": {
                    "branch": "main",
                    "repository_url": "git@example.test:org/actual-project.git",
                },
            },
        }
        (sessions_dir / "rollout-worktree-session.jsonl").write_text(
            json.dumps(meta) + "\n"
        )

        with patch("session_browser.discovery.Path.home", return_value=tmp_path):
            sessions = scan_codex()

        assert sessions[0].repository == "actual-project"

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
            "git_branch TEXT, git_origin_url TEXT, source TEXT, "
            "first_user_message TEXT NOT NULL DEFAULT '', "
            "created_at_ms INTEGER, updated_at_ms INTEGER, "
            "archived INTEGER NOT NULL DEFAULT 0)"
        )
        for row in rows:
            conn.execute(
                "INSERT INTO threads (id, rollout_path, cwd, git_branch, git_origin_url, "
                "first_user_message, created_at_ms, updated_at_ms, archived) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    row["id"],
                    row["rollout_path"],
                    row.get("cwd", ""),
                    row.get("branch"),
                    row.get("origin_url"),
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
        # No origin in this row: the fallback remains the cwd name.
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

    def test_db_uses_origin_for_a_worktree(self, tmp_path):
        sessions_dir = tmp_path / ".codex" / "sessions" / "2026" / "04" / "10"
        f = self._write_rollout(sessions_dir, "cx-worktree", summary="hello", day="10")
        self._write_db(
            tmp_path,
            [
                {
                    "id": "cx-worktree",
                    "rollout_path": str(f),
                    "cwd": "/Users/test/Projects/worktrees/codex",
                    "origin_url": "git@example.test:org/actual-project.git",
                }
            ],
        )

        with patch("session_browser.discovery.Path.home", return_value=tmp_path):
            sessions = scan_codex()

        assert sessions[0].repository == "actual-project"

    def test_falls_back_to_files_when_index_missing(self, tmp_path):
        """No state_*.sqlite means the file scan runs unchanged."""
        sessions_dir = tmp_path / ".codex" / "sessions" / "2026" / "04" / "10"
        self._write_rollout(sessions_dir, "cx-1", summary="hello", day="10")

        with patch("session_browser.discovery.Path.home", return_value=tmp_path):
            sessions = scan_codex()

        assert len(sessions) == 1
        assert sessions[0].id == "cx-1"
        assert sessions[0].summary == "hello"

    def test_unindexed_rollout_is_read_from_disk_alone(self, tmp_path):
        """A rollout file with no matching row must not be invisible -- and
        must cost only its own read, not a file scan of the whole tree. Codex
        leaves some rollouts out of its index for good, and treating that as
        "fall back to the file scan" put ~1.5 s on every call on a real
        machine for as long as the gap lasted."""
        sessions_dir = tmp_path / ".codex" / "sessions" / "2026" / "04" / "10"
        self._write_rollout(sessions_dir, "cx-1", summary="hello", day="10")
        unindexed = self._write_rollout(
            sessions_dir, "cx-2", summary="newest", day="11"
        )
        self._write_db(
            tmp_path,
            [
                {
                    "id": "cx-1",
                    "rollout_path": str(
                        sessions_dir / "rollout-2026-04-10T13-00-00-cx-1.jsonl"
                    ),
                    "summary": "hello from the index",
                    "created_ms": 1775826000000,
                    "updated_ms": 1775826005000,
                }
            ],
        )

        from session_browser import discovery

        read: list[str] = []
        real = discovery._codex_session_from_file

        def recording(f):
            read.append(f.name)
            return real(f)

        with (
            patch("session_browser.discovery.Path.home", return_value=tmp_path),
            patch.object(discovery, "_codex_session_from_file", recording),
        ):
            sessions = scan_codex()

        by_id = {s.id: s for s in sessions}
        assert set(by_id) == {"cx-1", "cx-2"}
        # The indexed session still came from its row, so the fast path ran...
        assert by_id["cx-1"].summary == "hello from the index"
        # ...and the unindexed one from its own file, the only one opened.
        assert by_id["cx-2"].summary == "newest"
        assert by_id["cx-2"].content_path == str(unindexed)
        assert read == [unindexed.name]

    def test_unindexed_twin_of_an_indexed_thread_is_not_listed_twice(self, tmp_path):
        """A file the index has no row for, carrying a thread id the index
        does have under another path, is the same session: the row wins."""
        sessions_dir = tmp_path / ".codex" / "sessions" / "2026" / "04" / "10"
        f = self._write_rollout(sessions_dir, "cx-1", summary="hello", day="10")
        twin = sessions_dir / "rollout-2026-04-12T13-00-00-cx-1.jsonl"
        twin.write_bytes(f.read_bytes())
        self._write_db(
            tmp_path,
            [{"id": "cx-1", "rollout_path": str(f), "summary": "from the index"}],
        )

        with patch("session_browser.discovery.Path.home", return_value=tmp_path):
            sessions = scan_codex()

        assert [(s.id, s.summary) for s in sessions] == [("cx-1", "from the index")]

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
            "git_branch TEXT, git_origin_url TEXT, source TEXT, "
            "first_user_message TEXT NOT NULL DEFAULT '', "
            "created_at_ms INTEGER, updated_at_ms INTEGER, "
            "archived INTEGER NOT NULL DEFAULT 0)"
        )
        conn.execute(
            "INSERT INTO threads (id, rollout_path, cwd, git_branch, git_origin_url, "
            "first_user_message, created_at_ms, updated_at_ms, archived) "
            "VALUES (?, ?, '', NULL, NULL, ?, ?, ?, 0)",
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

    def test_model_comes_from_the_index(self, tmp_path):
        """threads.model, migration-added: its value when present, None when
        an older Codex left it empty -- and an index from before the column
        existed still serves every other field from the fast path."""
        sessions_dir = tmp_path / ".codex" / "sessions" / "2026" / "04" / "10"
        f1 = self._write_rollout(sessions_dir, "cx-1", summary="hi", day="10")
        f2 = self._write_rollout(sessions_dir, "cx-2", summary="hi", day="11")
        rows = [
            {"id": "cx-1", "rollout_path": str(f1), "summary": "one from db"},
            {"id": "cx-2", "rollout_path": str(f2), "summary": "two from db"},
        ]
        self._write_db(tmp_path, rows)

        with patch("session_browser.discovery.Path.home", return_value=tmp_path):
            before = {s.id: (s.summary, s.model) for s in scan_codex()}
        assert before == {"cx-1": ("one from db", None), "cx-2": ("two from db", None)}

        conn = sqlite3.connect(str(tmp_path / ".codex" / "state_5.sqlite"))
        conn.execute("ALTER TABLE threads ADD COLUMN model TEXT")
        conn.execute("UPDATE threads SET model = 'gpt-6.1-sol' WHERE id = 'cx-1'")
        conn.execute("UPDATE threads SET model = '' WHERE id = 'cx-2'")
        conn.commit()
        conn.close()

        with patch("session_browser.discovery.Path.home", return_value=tmp_path):
            after = {s.id: (s.summary, s.model) for s in scan_codex()}
        assert after == {
            "cx-1": ("one from db", "gpt-6.1-sol"),
            "cx-2": ("two from db", None),
        }

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

    def test_large_index_gap_warns_once(self, tmp_path, caplog, monkeypatch):
        """An index missing a large share of the tree is probably broken, and
        reading that share from disk is slow in its own right: said, and only
        once per message per process."""
        from session_browser import discovery

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

        gaps = [r for r in caplog.records if "codex index has no row" in r.message]
        assert len(gaps) == 1
        assert "rollout-2026-04-10T13-00-00-cx-1.jsonl" in gaps[0].message
        assert "100% of the tree" in gaps[0].message

    def test_small_index_gap_is_quiet(self, tmp_path, caplog, monkeypatch):
        """A handful of unindexed rollouts costs a handful of reads, so it is
        not worth a line on stderr: four fresh agents in a row read the old
        per-call warning as noise and were right to."""
        from session_browser import discovery

        monkeypatch.setattr(discovery, "_warned_codex_fallbacks", set())
        sessions_dir = tmp_path / ".codex" / "sessions" / "2026" / "04" / "10"
        rows = []
        for i in range(10):
            sid = f"cx-{i}"
            f = self._write_rollout(sessions_dir, sid, summary="hi", day=f"{i + 10}")
            rows.append({"id": sid, "rollout_path": str(f), "summary": "hi"})
        self._write_rollout(sessions_dir, "cx-new", summary="newest", day="28")
        self._write_db(tmp_path, rows)

        with (
            caplog.at_level("WARNING"),
            patch("session_browser.discovery.Path.home", return_value=tmp_path),
        ):
            assert len(scan_codex()) == 11

        assert not [r for r in caplog.records if "codex index" in r.message]

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


class TestCompressedCodexRollouts:
    """Codex's background worker rewrites old rollouts as ``.jsonl.zst`` and
    leaves its index naming the ``.jsonl``. Measured 2026-10-26 on codex-cli
    0.164.0: 684 of 709 rollouts on one machine had been compressed, and every
    one of them was invisible -- dropped by the index path as a row whose file
    had gone, and never matched by the file scan's glob."""

    WORD = "kookaburra"

    def _rollout(self, home: Path, sid: str = "cx-z") -> Path:
        day = home / ".codex" / "sessions" / "2026" / "07" / "31"
        day.mkdir(parents=True, exist_ok=True)
        f = day / f"rollout-2026-07-31T13-51-37-{sid}.jsonl"
        lines = [
            {
                "type": "session_meta",
                "timestamp": "2026-07-31T13:51:37Z",
                "payload": {"id": sid, "cwd": "/p", "git": {"branch": "main"}},
            },
            {
                "type": "event_msg",
                "timestamp": "2026-07-31T13:51:40Z",
                "payload": {"type": "user_message", "message": "read the handoff"},
            },
            {
                "type": "event_msg",
                "timestamp": "2026-07-31T13:52:00Z",
                "payload": {
                    "type": "agent_message",
                    "message": f"a {self.WORD} was sighted",
                },
            },
        ]
        f.write_text("".join(json.dumps(x) + "\n" for x in lines))
        return f

    def _compress(self, f: Path) -> Path:
        z = f.with_name(f.name + compressed.SUFFIX)
        z.write_bytes(compressed.zstd.compress(f.read_bytes()))
        f.unlink()
        return z

    def _index(self, home: Path, rollout: Path, sid: str = "cx-z") -> None:
        TestScanCodexDb()._write_db(
            home,
            [
                {
                    "id": sid,
                    "rollout_path": str(rollout),
                    "cwd": "/p",
                    "summary": "read the handoff",
                    "created_ms": 1785505897000,
                    "updated_ms": 1785505920000,
                }
            ],
        )

    def test_index_row_finds_its_compressed_rollout(self, tmp_path):
        plain = self._rollout(tmp_path)
        z = self._compress(plain)
        self._index(tmp_path, plain)

        with patch("session_browser.discovery.Path.home", return_value=tmp_path):
            sessions = scan_codex()

            assert [s.id for s in sessions] == ["cx-z"]
            # The index path ran (its summary), pointed at the file on disk.
            assert sessions[0].summary == "read the handoff"
            assert sessions[0].content_path == str(z)
            assert search_session_contents(sessions, self.WORD) == {"cx-z"}
            assert self.WORD in load_session_content(sessions[0])

    def test_file_scan_reads_a_compressed_rollout(self, tmp_path):
        self._compress(self._rollout(tmp_path))

        with patch("session_browser.discovery.Path.home", return_value=tmp_path):
            sessions = scan_codex()

        assert [s.id for s in sessions] == ["cx-z"]
        assert sessions[0].summary == "read the handoff"
        assert sessions[0].updated_at == "2026-07-31T13:52:00Z"

    def test_plain_rollout_wins_over_a_compressed_twin(self, tmp_path):
        plain = self._rollout(tmp_path)
        twin = plain.with_name(plain.name + compressed.SUFFIX)
        twin.write_bytes(compressed.zstd.compress(plain.read_bytes()))
        self._index(tmp_path, plain)

        with patch("session_browser.discovery.Path.home", return_value=tmp_path):
            sessions = scan_codex()

        assert [s.content_path for s in sessions] == [str(plain)]

    def test_decompresses_once_then_reads_the_copy(self, tmp_path):
        z = self._compress(self._rollout(tmp_path))
        real = compressed._decompress
        calls = []

        def counting(path, target):
            calls.append(path)
            real(path, target)

        with patch.object(compressed, "_decompress", counting):
            first = compressed.readable(z)
            compressed._known.clear()  # a fresh process: the copy on disk
            second = compressed.readable(z)

        assert first == second
        assert calls == [z]
        assert self.WORD in first.read_text()

    def test_a_changed_source_replaces_its_stale_copy(self, tmp_path):
        z = self._compress(self._rollout(tmp_path))
        old = compressed.readable(z)
        z.write_bytes(compressed.zstd.compress(b'{"type":"session_meta"}\n'))
        os.utime(z, ns=(1, 1))

        new = compressed.readable(z)

        assert new != old
        assert not old.exists()
        assert new.read_text() == '{"type":"session_meta"}\n'

    def test_a_deleted_cache_is_rebuilt(self, tmp_path):
        z = self._compress(self._rollout(tmp_path))
        compressed.readable(z).unlink()

        assert self.WORD in compressed.readable(z).read_text()

    def test_a_corrupt_rollout_is_reported_not_skipped(self, tmp_path):
        """A file that will not decompress must surface as unreadable. Were it
        silently dropped from the candidate scan, search would answer "no
        match" for a session it never read."""
        plain = self._rollout(tmp_path)
        z = plain.with_name(plain.name + compressed.SUFFIX)
        z.write_bytes(b"not zstd at all")
        plain.unlink()
        session = Session(id="cx-z", provider="codex", content_path=str(z))

        [result] = search_sessions([session], self.WORD)

        assert result.unreadable
        assert any("decompress" in w for w in result.warnings)
        assert list(compressed.cache_dir().glob(".partial-*")) == []

    # -- a parse reads the rollout itself; only scans keep copies ----------

    @pytest.fixture
    def cache_home(self, tmp_path, monkeypatch):
        """A cache of this test's own, so the copies counted are its own."""
        monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))
        return tmp_path / "cache" / "session-browser"

    @staticmethod
    def _copies() -> list[Path]:
        return sorted(compressed.cache_dir().glob("*.jsonl"))

    def test_a_parse_reads_the_rollout_itself_and_keeps_no_copy(
        self, tmp_path, cache_home
    ):
        """With the entry cache on, a compressed rollout's parse is the last
        read it gets while it stays the same, so a copy made for that read is
        five times the file kept for nothing: 1,400 of them, 1.8 GB, on the
        machine this was measured on."""
        z = self._compress(self._rollout(tmp_path))
        session = Session(id="cx-z", provider="codex", content_path=str(z))
        with patch.object(compressed, "_decompress") as copy:
            [first] = search_sessions([session], self.WORD)
            [again] = search_sessions([session], self.WORD)
        assert first.match_count == again.match_count == 1
        copy.assert_not_called()
        assert self._copies() == []

    def test_with_the_cache_off_the_scan_copies_once_and_the_parse_reads_it(
        self, tmp_path, cache_home, monkeypatch
    ):
        monkeypatch.setenv("SESSION_BROWSER_CACHE", "0")
        z = self._compress(self._rollout(tmp_path))
        session = Session(id="cx-z", provider="codex", content_path=str(z))
        with (
            patch.object(
                compressed, "_decompress", wraps=compressed._decompress
            ) as copy,
            patch.object(compressed, "_stream", wraps=compressed._stream) as stream,
        ):
            [result] = search_sessions([session], self.WORD)
        assert result.match_count == 1
        assert copy.call_count == 1
        stream.assert_not_called()  # the copy the scan made is what was parsed
        assert len(self._copies()) == 1

    @pytest.mark.parametrize("cache", ["on", "off"])
    @pytest.mark.parametrize("damage", ["truncated", "empty"])
    def test_a_rollout_cut_short_is_unreadable_and_kept_nowhere(
        self, tmp_path, cache_home, monkeypatch, cache, damage
    ):
        """zstd reports a file that ends early with EOFError, which is
        neither its own error nor an OSError, so no handler caught it and one
        such rollout took every search down. It must read as unreadable, as a
        corrupt one does, and leave nothing behind: no copy, and no stored
        parse of the part that did decompress -- which, streamed, is most of
        it."""
        if cache == "off":
            monkeypatch.setenv("SESSION_BROWSER_CACHE", "0")
        f = self._rollout(tmp_path)
        with f.open("a") as fh:
            for _ in range(4000):  # incompressible, so the cut falls late
                record = {"type": "event_msg", "payload": {"type": "agent_message"}}
                record["payload"]["message"] = os.urandom(48).hex()
                fh.write(json.dumps(record) + "\n")
        z = self._compress(f)
        data = z.read_bytes()
        z.write_bytes(data[: len(data) - 1000] if damage == "truncated" else b"")
        if damage == "truncated":
            read = 0
            with (
                pytest.raises(OSError, match="decompress"),
                compressed.open_binary(z) as fh,
            ):
                for _line in fh:
                    read += 1
            assert read > 1000  # it failed part-way, not at the start
        session = Session(id="cx-z", provider="codex", content_path=str(z))

        [result] = search_sessions([session], self.WORD)

        assert result.unreadable
        assert any("decompress" in w for w in result.warnings)
        assert self._copies() == []
        assert list(compressed.cache_dir().glob(".partial-*")) == []
        entries = cache_home / "entries"
        assert not entries.exists() or not any(entries.rglob("*.txt"))

    def test_head_and_lineage_reads_make_no_copy(self, tmp_path, cache_home):
        from session_browser.transcript import codex_fork_base, lineage_ids

        f = self._rollout(tmp_path)
        with f.open("a") as fh:
            fh.write(json.dumps({"type": "event_msg", "session_id": "cx-older"}) + "\n")
        z = self._compress(f)
        session = Session(id="cx-z", provider="codex", content_path=str(z))

        assert lineage_ids("codex", "cx-z", str(z)) == {"cx-z", "cx-older"}
        assert codex_fork_base(session) is None  # read its first line to say so
        assert self._copies() == []

    def test_a_copy_swept_mid_read_falls_back_to_the_rollout(
        self, tmp_path, cache_home
    ):
        z = self._compress(self._rollout(tmp_path))
        gone = compressed.cache_dir() / "swept-away.jsonl"
        looked = (gone, time.time(), os.stat(z))
        with (
            patch.object(compressed, "_copy", return_value=looked),
            compressed.open_binary(z) as fh,
        ):
            assert self.WORD.encode() in fh.read()

    def test_the_sweep_keeps_what_scans_use_and_ages_out_the_rest(
        self, tmp_path, cache_home
    ):
        """A copy's mtime is when a scan last wanted it. A scan refreshes it
        through readable(); a parse through open_binary() does not, so what
        only a parse read -- every copy, before the entry cache -- expires."""
        scanned = self._compress(self._rollout(tmp_path, "cx-scanned"))
        parsed = self._compress(self._rollout(tmp_path, "cx-parsed"))
        copies = [compressed.readable(scanned), compressed.readable(parsed)]
        partial = compressed.cache_dir() / ".partial-stale"
        partial.write_bytes(b"half a file")
        fresh = compressed.cache_dir() / ".partial-fresh"
        fresh.write_bytes(b"still being written")
        now = time.time()
        long_ago = now - 2 * compressed.SWEEP_EVERY
        for p in (*copies, partial):
            os.utime(p, (long_ago, long_ago))
        compressed._known.clear()

        assert compressed.readable(scanned) == copies[0]  # a scan wants it
        with compressed.open_binary(parsed) as fh:  # a parse reads, wants nothing
            assert self.WORD.encode() in fh.read()

        assert compressed.sweep(now) == 1
        assert copies[0].exists() and not copies[1].exists()
        assert fresh.exists() and not partial.exists()

    def test_a_rollout_that_will_not_open_leaves_nothing_to_finalize(
        self, tmp_path, cache_home
    ):
        """An open that fails is an OSError and nothing more. A reader built
        around a file it never managed to open used to be left half-made, and
        its finalizer then failed too, printing a traceback to stderr -- which
        agents read and the TUI captures -- for a permission error, a lost
        race with a delete, or a directory with a rollout's name."""
        not_a_file = tmp_path / "rollout-x.jsonl.zst"
        not_a_file.mkdir()
        unraisable: list = []
        with patch.object(sys, "unraisablehook", unraisable.append):
            with pytest.raises(OSError):
                compressed._stream(not_a_file)
            gc.collect()
        assert unraisable == []

    def test_the_sweep_is_rationed_and_makes_nothing_where_nothing_was(
        self, tmp_path, cache_home
    ):
        assert compressed.sweep_if_due() == 0
        assert not compressed.cache_dir().exists()

        z = self._compress(self._rollout(tmp_path))
        long_ago = time.time() - 2 * compressed.SWEEP_EVERY
        os.utime(compressed.readable(z), (long_ago, long_ago))
        assert compressed.sweep_if_due() == 1
        os.utime(compressed.readable(z), (long_ago, long_ago))
        assert compressed.sweep_if_due() == 0  # once a day, not once a search
        later = time.time() + compressed.SWEEP_EVERY + 1
        assert compressed.sweep_if_due(later) == 1


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

    @pytest.mark.parametrize(
        ("origin_url", "expected"),
        [
            ("https://example.test/org/project.git", "project"),
            ("ssh://git@example.test/org/project", "project"),
            ("git@example.test:project.git", "project"),
            ("", "worktree-name"),
        ],
    )
    def test_codex_origin_name_with_cwd_fallback(self, origin_url, expected):
        assert (
            _codex_repo_name("/Projects/worktrees/worktree-name", origin_url)
            == expected
        )


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
            time_created INTEGER NOT NULL, time_updated INTEGER NOT NULL,
            parent_id TEXT, agent TEXT
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

    def test_resume_pi(self):
        cmd = resume_command("pi", "01a020b0-7ffa")
        assert cmd == "pi --session 01a020b0-7ffa"

    def test_resume_opencode(self):
        cmd = resume_command("opencode", "ses_abc123")
        assert cmd == "opencode -s ses_abc123"

    def test_resume_opencode_with_cwd(self):
        cmd = resume_command("opencode", "ses_abc123", "/Users/test/project")
        assert cmd == "cd /Users/test/project && opencode -s ses_abc123"


# ---------------------------------------------------------------------------
# pi discovery: per-file JSONL with a cwd-carrying header line
# ---------------------------------------------------------------------------


class TestScanPi:
    def _write(
        self,
        tmp_path,
        lines,
        *,
        project="--Users-test-Projects-game--",
        name="2026-08-18T17-52-48-324Z_01a01601-22c4-7973-bf04-2698dd8d19ca.jsonl",
    ):
        d = tmp_path / ".pi" / "agent" / "sessions" / project
        d.mkdir(parents=True, exist_ok=True)
        path = d / name
        path.write_text("\n".join(json.dumps(x) for x in lines) + "\n")
        return path

    def _session_lines(self):
        return [
            {
                "type": "session",
                "version": 3,
                "id": "01a01601-22c4-7973-bf04-2698dd8d19ca",
                "timestamp": "2026-08-18T17:52:48.324Z",
                "cwd": "/Users/test/Projects/game",
            },
            {
                "type": "model_change",
                "timestamp": "2026-08-18T17:52:49.030Z",
                "provider": "qwen-lab",
                "modelId": "qwen38-local",
            },
            {
                "type": "message",
                "timestamp": "2026-08-18T17:52:55.000Z",
                "message": {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Bootstrap the game skeleton"}
                    ],
                },
            },
            {
                "type": "message",
                "timestamp": "2026-08-18T17:53:10.000Z",
                "message": {
                    "role": "assistant",
                    "content": [{"type": "text", "text": "Scaffolding it now"}],
                },
            },
        ]

    def test_scan_pi_sessions(self, tmp_path):
        self._write(tmp_path, self._session_lines())

        with patch("session_browser.discovery.Path.home", return_value=tmp_path):
            sessions = scan_pi()

        assert len(sessions) == 1
        s = sessions[0]
        assert s.id == "01a01601-22c4-7973-bf04-2698dd8d19ca"
        assert s.provider == "pi"
        assert s.summary == "Bootstrap the game skeleton"
        assert s.cwd == "/Users/test/Projects/game"
        assert s.repository == "game"
        assert s.created_at == "2026-08-18T17:52:48.324Z"
        assert s.updated_at == "2026-08-18T17:53:10.000Z"

    def test_model_comes_from_the_first_model_change(self, tmp_path):
        """Spelled provider/model, as pi's own --model flag takes it. A later
        switch is past the head this scan reads, so the first one names the
        session; a session with no model_change at all is not known, not
        modelless."""
        lines = self._session_lines()
        lines.append({"type": "model_change", "provider": "x", "modelId": "later"})
        self._write(tmp_path, lines)
        unrecorded = [
            {**line, "id": "02b"} if line["type"] == "session" else line
            for line in self._session_lines()
            if line["type"] != "model_change"
        ]
        self._write(tmp_path, unrecorded, name="2026-08-18T18-00-00-000Z_02b.jsonl")

        with patch("session_browser.discovery.Path.home", return_value=tmp_path):
            models = {s.id: s.model for s in scan_pi()}

        assert models == {
            "01a01601-22c4-7973-bf04-2698dd8d19ca": "qwen-lab/qwen38-local",
            "02b": None,
        }

    def test_branch_is_absent_not_empty(self, tmp_path):
        """pi records no branch anywhere in its format, which is a different
        fact from a session that ran on no branch."""
        self._write(tmp_path, self._session_lines())

        with patch("session_browser.discovery.Path.home", return_value=tmp_path):
            sessions = scan_pi()

        assert sessions[0].branch is None

    def test_cwd_comes_from_the_header_not_the_decoded_dir_name(self, tmp_path):
        """The directory name encodes "/" as "-" exactly as Claude's does, so
        decoding a hyphenated project yields a path that exists nowhere. Every
        pi transcript carries the real cwd on its first line instead."""
        lines = self._session_lines()
        lines[0]["cwd"] = "/Users/test/Projects/agent-loadout"
        self._write(tmp_path, lines, project="--Users-test-Projects-agent-loadout--")

        with patch("session_browser.discovery.Path.home", return_value=tmp_path):
            sessions = scan_pi()

        assert sessions[0].cwd == "/Users/test/Projects/agent-loadout"
        assert sessions[0].repository == "agent-loadout"

    def test_id_falls_back_to_the_filename_uuid(self, tmp_path):
        """The stem's uuid suffix is the same id the header carries, so a
        truncated or unparseable header still leaves the session resumable."""
        self._write(tmp_path, [{"type": "model_change", "provider": "x"}])

        with patch("session_browser.discovery.Path.home", return_value=tmp_path):
            sessions = scan_pi()

        assert sessions[0].id == "01a01601-22c4-7973-bf04-2698dd8d19ca"

    def test_tool_traffic_is_not_activity(self, tmp_path):
        """A tool result is the harness answering itself. Last activity is the
        last thing a participant said -- the same line claude and codex draw."""
        lines = self._session_lines()
        lines.append(
            {
                "type": "message",
                "timestamp": "2026-08-18T19:00:00.000Z",
                "message": {
                    "role": "toolResult",
                    "toolName": "bash",
                    "content": [{"type": "text", "text": "ok"}],
                },
            }
        )
        self._write(tmp_path, lines)

        with patch("session_browser.discovery.Path.home", return_value=tmp_path):
            sessions = scan_pi()

        assert sessions[0].updated_at == "2026-08-18T17:53:10.000Z"

    def test_summary_reads_the_first_user_turn_only(self, tmp_path):
        lines = self._session_lines()
        lines.append(
            {
                "type": "message",
                "timestamp": "2026-08-18T18:00:00.000Z",
                "message": {
                    "role": "user",
                    "content": [{"type": "text", "text": "second thing I asked"}],
                },
            }
        )
        self._write(tmp_path, lines)

        with patch("session_browser.discovery.Path.home", return_value=tmp_path):
            sessions = scan_pi()

        assert sessions[0].summary == "Bootstrap the game skeleton"

    def test_scan_pi_no_sessions_dir(self, tmp_path):
        with patch("session_browser.discovery.Path.home", return_value=tmp_path):
            assert scan_pi() == []

    def test_one_unreadable_file_does_not_lose_the_others(self, tmp_path):
        self._write(tmp_path, self._session_lines())
        d = tmp_path / ".pi" / "agent" / "sessions" / "--Users-test-Projects-game--"
        (d / "2026-08-18T18-00-00-000Z_broken.jsonl").write_bytes(
            b"\xff\xfe not json\n"
        )

        with patch("session_browser.discovery.Path.home", return_value=tmp_path):
            sessions = scan_pi()

        assert "01a01601-22c4-7973-bf04-2698dd8d19ca" in {s.id for s in sessions}


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
    from textual.widgets import DataTable
    from textual.worker import WorkerCancelled

    from session_browser.app import (
        MultiplexerChoice,
        SearchInput,
        SessionBrowser,
        SessionTable,
        ShortcutHelp,
        build_hierarchy,
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
            _discovered=[],
        )
        app._open_in_multiplexer = lambda target: SessionBrowser._open_in_multiplexer(
            app, target
        )
        app._resume_session = lambda s, sid: SessionBrowser._resume_session(app, s, sid)
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

    def test_a_claude_subagent_hands_off_its_root_session(self, monkeypatch):
        """Claude Code resumes a conversation, not an agent inside one."""
        app, plan, _updates, _exits = self._tmux_action_app()
        app._selected = Session(
            id="agent-a2",
            provider="claude",
            cwd="/tmp/proj",
            content_path="/h/.claude/projects/-p/root/subagents/workflows/wf_1/agent-a2.jsonl",
            parent_id="root",
            subagent_kind="workflow-subagent",
        )
        calls = []
        monkeypatch.setattr(
            tmux, "prepare_session", lambda *a, **kw: calls.append((a, kw)) or plan
        )
        monkeypatch.setattr(
            subprocess, "run", lambda cmd, **kw: subprocess.CompletedProcess(cmd, 0)
        )

        SessionBrowser._open_in_multiplexer(app, self._tmux_target())

        assert calls == [
            (
                ("claude", "root", "/tmp/proj"),
                {"content_path": "/h/.claude/projects/-p/root.jsonl"},
            )
        ]

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


async def _cross_detail_window(app, pilot, cross) -> None:
    """Run *cross*, which slides the transcript window, and return only once the
    viewport restore that slide deferred has landed.

    The restore runs in a worker that awaits the remount and then schedules
    itself with call_after_refresh, so neither wait_for_complete() nor a
    pause() waits for it. A test that moved the viewport next -- scroll_home
    before a wheel-up -- used to lose to it about half the time on a busy
    machine: the restore landed afterwards and put the viewport back at its
    anchor, so the wheel-up no longer crossed. call_after_refresh captures the
    bound method when it schedules, which is why the wrapper must be in place
    before *cross* runs."""
    landed = asyncio.Event()
    restore = app._restore_detail_anchor

    def traced(*args):
        restore(*args)
        landed.set()

    app._restore_detail_anchor = traced
    try:
        cross()
        await asyncio.wait_for(landed.wait(), timeout=5)
    finally:
        del app._restore_detail_anchor
    await pilot.pause()


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
class TestPointerFlow:
    """A click must leave the keyboard where the pointer went.

    Two paths used to fail that. Textual focuses the focusable widget under the
    pointer, and a pane's title strip, header and border are none of them
    focusable, so clicking one did nothing at all. And a click on the cell the
    cursor already occupies posts RowSelected -- the same message Enter sends,
    and the app answers that by handing the reader to the transcript.

    What a click on a row *does* is settled by the layout, not by the click: a
    row clicked beside a visible detail pane has already been shown, and there
    the keyboard stays in the list. In a one-pane flow the transcript for it is
    behind the pane switch, so the same click brings that pane up -- otherwise
    the row takes two clicks, and the first has no consequence a reader can
    see beyond the highlight bar.
    """

    async def test_click_on_pane_chrome_focuses_that_pane(self):
        app, fake = _make_app_with_rows()
        async with app.run_test(size=(160, 45)) as pilot:
            await _install_fake_sessions(app, pilot, fake)
            app.query_one("#detail-scroll").focus()
            await pilot.pause()

            await pilot.click("#sessions-status")
            await pilot.pause()
            assert app.query_one("#session-table").has_focus

            await pilot.click("#transcript-title")
            await pilot.pause()
            assert app.query_one("#detail-scroll").has_focus

    async def test_click_on_highlighted_row_keeps_the_list_focused(self):
        """Beside a visible detail pane, the transcript has already appeared.

        The click that moved the highlight showed it, and moving the keyboard
        as well would be a bounce for a reader whose pointer is in the list.
        """
        app, fake = _make_app_with_rows()
        async with app.run_test(size=(160, 45)) as pilot:
            await _install_fake_sessions(app, pilot, fake)
            app.action_focus_right_pane()
            await pilot.pause()

            # The first click moves the cursor, and the detailed transcript for
            # that row is on screen. The second lands on the row the cursor
            # already sits on -- Textual's activate, the message Enter sends.
            await pilot.click("#session-table", offset=(6, 4))
            await pilot.pause()
            assert app._selected is not None and app._selected.id == "s4"
            await pilot.click("#session-table", offset=(6, 4))
            await pilot.pause()

            assert app.query_one("#session-table").has_focus
            assert not app.query_one("#detail-scroll").has_focus

    async def test_enter_still_hands_the_reader_to_the_transcript(self):
        app, fake = _make_app_with_rows()
        async with app.run_test(size=(160, 45)) as pilot:
            await _install_fake_sessions(app, pilot, fake)
            app.query_one("#session-table").focus()
            await pilot.pause()
            await pilot.press("enter")
            await pilot.pause()
            assert app.query_one("#detail-scroll").has_focus

    async def test_one_click_opens_when_the_detail_is_behind_a_pane_switch(self):
        """The click that moves the highlight is the click that opens.

        There is no pane on screen for the selection to appear in, so the
        switch is the other half of the same click -- otherwise a reader picks
        a row and the only thing that moves is the highlight bar.
        """
        app, fake = _make_app_with_rows()
        async with app.run_test(size=(88, 28)) as pilot:
            await _install_fake_sessions(app, pilot, fake)
            assert app.screen.has_class("-compact")

            await pilot.click("#session-table", offset=(6, 4))
            await pilot.pause()

            assert app._selected is not None and app._selected.id == "s4"
            assert not app.query_one("#left-pane").display
            assert app.query_one("#right-pane").display
            assert app.query_one("#detail-scroll").has_focus

    async def test_click_on_the_highlighted_row_opens_in_a_one_pane_flow(self):
        """The other click Textual posts: the row the cursor already sits on.

        Reading and coming back leaves the cursor where it was, so the next
        click on that row arrives as RowSelected rather than RowHighlighted.
        Same gesture, same row, so it must not answer differently.
        """
        app, fake = _make_app_with_rows()
        async with app.run_test(size=(88, 28)) as pilot:
            await _install_fake_sessions(app, pilot, fake)
            await pilot.click("#session-table", offset=(6, 4))
            await pilot.pause()
            await pilot.press("escape")
            await pilot.pause()
            assert app.query_one("#left-pane").display

            await pilot.click("#session-table", offset=(6, 4))
            await pilot.pause()

            assert not app.query_one("#left-pane").display
            assert app.query_one("#detail-scroll").has_focus

    async def test_focus_view_counts_as_one_pane(self):
        """`z` is the same drill-down by hand: the detail is one switch away."""
        app, fake = _make_app_with_rows()
        async with app.run_test(size=(160, 45)) as pilot:
            await _install_fake_sessions(app, pilot, fake)
            app.query_one("#session-table").focus()
            await pilot.press("z")
            await pilot.pause()
            assert app.screen.has_class("-focus-left")
            assert not app.query_one("#right-pane").display

            await pilot.click("#session-table", offset=(6, 4))
            await pilot.pause()

            assert app.screen.has_class("-focus-right")
            assert app.query_one("#detail-scroll").has_focus

    async def test_a_keyboard_move_after_a_click_only_moves_the_highlight(self):
        """Only the message a click posted may open; arrows never inherit it.

        In the one-pane flow the arrows are how the list is read, so a stamp
        left behind by an earlier click would turn browsing into opening.
        """
        app, fake = _make_app_with_rows()
        async with app.run_test(size=(88, 28)) as pilot:
            await _install_fake_sessions(app, pilot, fake)
            await pilot.click("#session-table", offset=(6, 4))
            await pilot.pause()
            await pilot.press("escape")
            await pilot.pause()

            await pilot.press("down")
            await pilot.pause()

            table = app.query_one("#session-table")
            assert table.cursor_row == 4
            assert app.query_one("#left-pane").display
            assert table.has_focus

    async def test_a_click_that_posts_no_row_message_leaves_no_stamp(self):
        """A header click picks no row, and Textual returns from its own
        handler before moving the cursor: were it to stamp, the next arrow key
        would read that stamp as its own."""
        app, fake = _make_app_with_rows()
        async with app.run_test(size=(88, 28)) as pilot:
            await _install_fake_sessions(app, pilot, fake)
            table = app.query_one("#session-table")

            await pilot.click("#session-table", offset=(6, 0))
            await pilot.pause()
            assert table.cursor_row == 0

            await pilot.press("down")
            await pilot.pause()

            assert table.cursor_row == 1
            assert app.query_one("#left-pane").display
            assert not app.query_one("#right-pane").display


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
        app, fake = _make_app_with_rows()
        async with app.run_test() as pilot:
            await _install_fake_sessions(app, pilot, fake)
            calls: list[str] = []
            orig = app._apply_filter
            timers = []

            def spy(q: str) -> None:
                calls.append(q)
                orig(q)

            def set_fake_timer(_delay, callback):
                timer = SimpleNamespace(stopped=False, callback=callback)
                timer.stop = lambda: setattr(timer, "stopped", True)
                timers.append(timer)
                return timer

            with patch.object(app, "set_timer", side_effect=set_fake_timer):
                app._apply_filter = spy
                app.query_one("#global-search").focus()
                await pilot.press("a", "b", "c")

                assert calls == []
                assert [timer.stopped for timer in timers] == [True, True, False]

                timers[-1].callback()
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

    async def test_ignore_file_hides_sessions_until_toggled(
        self, tmp_path, monkeypatch
    ):
        """Discovery drops what the ignore file names; `.` brings it back, and
        the header says which view is showing."""
        monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
        (tmp_path / "session-browser").mkdir()
        (tmp_path / "session-browser" / "ignore").write_text("dogfood/\n")
        fake = [
            Session(id="k1", provider="claude", cwd="/p/keep", updated_at="2026-03-02"),
            Session(
                id="n1", provider="codex", cwd="/p/dogfood/a", updated_at="2026-03-01"
            ),
            Session(
                id="n2", provider="codex", cwd="/p/dogfood/b", updated_at="2026-02-01"
            ),
        ]
        monkeypatch.setattr("session_browser.app.discover_all", lambda: fake)
        app = SessionBrowser()
        async with app.run_test() as pilot:
            await app.workers.wait_for_complete()
            await pilot.pause()
            assert [s.id for s in app._filtered] == ["k1"]
            assert "2 ignored" in str(app._app_meta.render())
            app.query_one("#session-table").focus()
            await pilot.press("full_stop")
            await pilot.pause()
            assert [s.id for s in app._filtered] == ["k1", "n1", "n2"]
            assert "incl. 2 ignored" in str(app._app_meta.render())
            await pilot.press("full_stop")
            await pilot.pause()
            assert [s.id for s in app._filtered] == ["k1"]

    async def test_ignored_notice_can_be_turned_off(self, tmp_path, monkeypatch):
        monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
        (tmp_path / "session-browser").mkdir()
        (tmp_path / "session-browser" / "ignore").write_text("dogfood/\n")
        (tmp_path / "session-browser" / "config.toml").write_text(
            "[tui]\nignored_notice = false\n"
        )
        fake = [
            Session(id="k1", provider="claude", cwd="/p/keep", updated_at="2026-03-02"),
            Session(
                id="n1", provider="codex", cwd="/p/dogfood", updated_at="2026-03-01"
            ),
        ]
        monkeypatch.setattr("session_browser.app.discover_all", lambda: fake)
        app = SessionBrowser()
        async with app.run_test() as pilot:
            await app.workers.wait_for_complete()
            await pilot.pause()
            assert [s.id for s in app._filtered] == ["k1"]
            assert "ignored" not in str(app._app_meta.render())

    async def test_rows_mark_unfinished_endings_and_recaps(self):
        """A row whose last turn did not finish says how; a finished one is
        unmarked. A recap summary carries its own mark, and the detail header
        names the ending in words."""
        app = SessionBrowser()
        fake = []
        for sid, ending in (
            ("s1", "interrupted"),
            ("s2", "mid-action"),
            ("s3", "unanswered"),
            ("s4", "completed"),
            ("s5", None),
        ):
            s = Session(
                id=sid,
                provider="claude",
                summary=f"work {sid}",
                updated_at=f"2026-01-0{sid[1]}",
            )
            s.ending = ending
            fake.append(s)
        fake[3].summary_is_recap = True
        async with app.run_test() as pilot:
            await _install_fake_sessions(app, pilot, fake)
            table = app.query_one("#session-table")
            cells = {
                str(table.get_row_at(r)[-1]).split("work ")[1]: str(
                    table.get_row_at(r)[-1]
                )
                for r in range(table.row_count)
            }
            assert cells["s1"].startswith("[red]✗[/] ")
            assert cells["s2"].startswith("[yellow]…[/] ")
            assert cells["s3"].startswith("[yellow]?[/] ")
            assert cells["s4"] == "[dim]↺[/] work s4"
            assert cells["s5"] == "work s5"
            app._update_detail_header(fake[0])
            header = str(app.query_one("#detail-header").render())
            assert "ended interrupted" in header

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

    def test_the_column_names_a_worktree_session_for_its_project(self):
        from session_browser.app import _session_project

        wt = Session(
            id="w",
            provider="claude",
            cwd="/Users/u/Projects/worktrees/Shop/fix-login",
            repository="Shop",
        )
        assert _session_project(wt) == "Shop"
        assert _session_project(Session(id="n", provider="pi", cwd="/a/b")) == "b"

    def test_this_project_scope_takes_the_projects_worktrees(self):
        app = SessionBrowser()
        app._here_only = True
        app._cwd_base = "/Users/u/Projects/Shop"

        def s(sid, cwd):
            return Session(id=sid, provider="claude", cwd=cwd)

        sessions = [
            s("own", "/Users/u/Projects/Shop"),
            s("sub", "/Users/u/Projects/Shop/api"),
            s("wt", "/Users/u/Projects/worktrees/Shop/fix-login"),
            s("herdr", "/Users/u/.herdr/worktrees/Shop/worktree-calm-river-1841"),
            s("other-wt", "/Users/u/Projects/worktrees/Bank/fix-login"),
            s("sibling", "/Users/u/Projects/ShopTools"),
            s("none", ""),
        ]
        assert [x.id for x in app._cwd_scope(sessions)] == [
            "own",
            "sub",
            "wt",
            "herdr",
        ]


class TestWorktreeProject:
    @pytest.mark.parametrize(
        ("cwd", "expected"),
        [
            ("/Users/u/Projects/worktrees/Shop/fix-login", "Shop"),
            ("/Users/u/Projects/worktrees/Shop/fix-login/src", "Shop"),
            ("/Users/u/.herdr/worktrees/shop/worktree-quiet-cloud-12cd", "shop"),
            ("/Users/u/Projects/shop/.claude/worktrees/brave-otter", "shop"),
            ("/Users/u/.codex/worktrees/a1b2/shop", "shop"),
            ("/Users/u/.codex/worktrees/a1b2/shop/web", "shop"),
            # A worktrees directory, or one project's folder of them, is not
            # a checkout: the layout's checkout directory must be there.
            ("/Users/u/Projects/worktrees/codex", ""),
            ("/Users/u/Projects/worktrees", ""),
            ("/Users/u/.codex/worktrees/a1b2", ""),
            ("/Users/u/Projects/shop", ""),
            ("", ""),
        ],
    )
    def test_layouts(self, cwd, expected):
        from session_browser.discovery import worktree_project

        assert worktree_project(cwd) == expected

    def test_claude_and_pi_fall_back_to_it(self):
        from session_browser.discovery import _cwd_project

        assert _cwd_project("/Users/u/Projects/worktrees/Shop/fix-login") == "Shop"
        assert _cwd_project("/Users/u/Projects/shop") == "shop"


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


class TestCodexForkReading:
    def test_fork_inherits_only_the_source_prefix(self, tmp_path):
        from session_browser.app import _codex_fork_transcript

        source_path = tmp_path / "source.jsonl"
        inherited = [
            {"type": "session_meta", "payload": {"session_id": "source"}},
            {
                "type": "event_msg",
                "payload": {"type": "user_message", "message": "first"},
            },
            {
                "type": "response_item",
                "payload": {
                    "type": "message",
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": "first answer"}],
                },
            },
        ]
        prefix = "".join(json.dumps(record) + "\n" for record in inherited).encode()
        source_path.write_bytes(
            prefix
            + (
                json.dumps(
                    {
                        "type": "event_msg",
                        "payload": {"type": "user_message", "message": "later"},
                    }
                )
                + "\n"
            ).encode()
        )
        fork_path = tmp_path / "fork.jsonl"
        fork_path.write_text(
            json.dumps(
                {
                    "type": "session_meta",
                    "payload": {
                        "session_id": "fork",
                        "history_base": {
                            "thread_id": "source",
                            "end_byte_offset": len(prefix),
                        },
                    },
                }
            )
            + "\n"
            + json.dumps(
                {
                    "type": "event_msg",
                    "payload": {"type": "user_message", "message": "new"},
                }
            )
            + "\n"
        )
        source = Session(id="source", provider="codex", content_path=str(source_path))
        fork = Session(id="fork", provider="codex", content_path=str(fork_path))
        transcript = _codex_fork_transcript(fork, {source.id: source, fork.id: fork})
        assert [(e.role, e.text) for e in transcript.entries] == [
            ("user", "first"),
            ("assistant", "first answer"),
            ("user", "new"),
        ]
        assert transcript.warnings == []


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
            assert "1 match" in rendered
            assert "showing match 1 of 1" in rendered
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
            assert "showing match" not in rendered

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
            # And the match being the active one is not a reason to unfold the
            # block it sits in: the preview moves onto it instead.
            assert original_widgets[-1].collapsed is True
            assert "searchable-tail" in str(original_widgets[-1].render())

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

    def _tool_block_transcript(
        self, session: Session, kind: str = "output", copies: int = 3
    ) -> Transcript:
        """A tool block far past the fold with *copies* matches spread through it."""
        chunk = "\n".join(f"line {index} " + "z" * 90 for index in range(12))
        body = chunk.join(
            f"\noccurrence {number} of quokka here\n" for number in range(copies)
        )
        return Transcript(
            session,
            [
                TranscriptEntry("user", "look for quokka"),
                TranscriptEntry(
                    "tool",
                    f"opening line of the block\n{chunk}{body}{chunk}",
                    metadata={"kind": kind, "tool": "Bash"},
                ),
            ],
        )

    async def test_long_tool_calls_collapse_like_long_outputs(self):
        """A call is as unbounded as an output, and used not to collapse.

        Only ``kind == "output"`` was collapsible, so a 90,000-character
        command arrived fully expanded however little of it anyone wanted.
        """
        from session_browser.app import TranscriptEntryWidget

        app, fake = _make_app_with_rows(1)
        async with app.run_test() as pilot:
            await _install_fake_sessions(app, pilot, fake)
            app._on_transcript_loaded(self._tool_block_transcript(fake[0], kind="call"))
            await pilot.pause()

            call = list(app.query(TranscriptEntryWidget))[-1]
            assert call.collapsible is True
            assert call.collapsed is True
            assert "chars hidden" in str(call.render())

            call.action_toggle_collapsed()
            assert call.collapsed is False
            assert "occurrence 2 of quokka here" in str(call.render())

    async def test_navigating_matches_in_a_tool_block_moves_the_preview(self):
        """n/N inside one collapsed block: the preview follows, the fold holds.

        The old behaviour expanded the entry the moment its match became
        active, which put the entire block on screen to show one line of it --
        and did so by discarding a collapse the reader had not asked to undo.
        """
        from session_browser.app import TranscriptEntryWidget

        app, fake = _make_app_with_rows(1)
        async with app.run_test(size=(120, 30)) as pilot:
            await _install_fake_sessions(app, pilot, fake)
            app._on_transcript_loaded(self._tool_block_transcript(fake[0]))
            await pilot.pause()
            app._do_session_search("quokka")
            await pilot.pause()

            block = list(app.query(TranscriptEntryWidget))[-1]
            # Four matches: the user entry's, then three in the block.
            assert len(app._matches) == 4
            assert app._match_idx == 0

            def preview() -> str:
                assert block.collapsed is True, "navigation unfolded the block"
                return str(block.render())

            for occurrence in (0, 1, 2):
                app.action_next_match()
                await pilot.pause()
                rendered = preview()
                assert f"showing match {occurrence + 1} of 3" in rendered
                assert f"occurrence {occurrence} of quokka here" in rendered

            for occurrence in (1, 0):
                app.action_prev_match()
                await pilot.pause()
                rendered = preview()
                assert f"showing match {occurrence + 1} of 3" in rendered
                assert f"occurrence {occurrence} of quokka here" in rendered

            # Wrapping back past the block, onto the match in the user entry:
            # no active match here any more, so the preview falls back to the
            # first one rather than to the opening line.
            app.action_prev_match()
            await pilot.pause()
            assert app._match_idx == 0
            rendered = preview()
            assert "showing match 1 of 3" in rendered
            assert "opening line of the block" not in rendered

    async def test_an_explicit_expansion_survives_navigation(self):
        """Enter is a decision; n/N must not overrule it in either direction."""
        from session_browser.app import TranscriptEntryWidget

        app, fake = _make_app_with_rows(1)
        async with app.run_test(size=(120, 30)) as pilot:
            await _install_fake_sessions(app, pilot, fake)
            app._on_transcript_loaded(self._tool_block_transcript(fake[0]))
            await pilot.pause()
            app._do_session_search("quokka")
            await pilot.pause()

            block = list(app.query(TranscriptEntryWidget))[-1]
            block.action_toggle_collapsed()
            assert block.collapsed is False
            for _ in range(4):
                app.action_next_match()
                await pilot.pause()
                assert block.collapsed is False

            block.action_toggle_collapsed()
            assert block.collapsed is True
            for _ in range(4):
                app.action_prev_match()
                await pilot.pause()
                assert block.collapsed is True

    async def test_the_header_names_the_block_and_counts_its_matches(self):
        from session_browser.app import TranscriptEntryWidget

        app, fake = _make_app_with_rows(1)
        async with app.run_test() as pilot:
            await _install_fake_sessions(app, pilot, fake)
            app._on_transcript_loaded(self._tool_block_transcript(fake[0]))
            await pilot.pause()
            app._do_session_search("quokka")
            await pilot.pause()

            rendered = str(list(app.query(TranscriptEntryWidget))[-1].render())
            assert "Tool output: Bash" in rendered
            assert "3 matches" in rendered
            assert "chars hidden" in rendered
            assert "Enter/click to expand" in rendered

    async def test_navigating_a_megabyte_block_builds_only_the_preview(self):
        """The bound the whole feature rests on, asserted on a real one.

        Rendered length is the proxy that matters: the widget builds the
        markup it is about to display, so a step that stayed proportional to
        the block would show up here as megabytes of it. The block below holds
        a thousand matches, which is also the number the highlighter would
        walk per step if it walked all of them.
        """
        from session_browser.app import TranscriptEntryWidget

        app, fake = _make_app_with_rows(1)
        filler = "q" * 2000
        block = "".join(f"{filler}\nnumbat sighting {index}\n" for index in range(1000))
        assert len(block) > 2_000_000
        async with app.run_test(size=(120, 30)) as pilot:
            await _install_fake_sessions(app, pilot, fake)
            app._on_transcript_loaded(
                Transcript(
                    fake[0],
                    [
                        TranscriptEntry("user", "count the numbat sightings"),
                        TranscriptEntry(
                            "tool",
                            block,
                            metadata={"kind": "output", "tool": "Bash"},
                        ),
                    ],
                )
            )
            await pilot.pause()
            app._do_session_search("numbat")
            await pilot.pause()

            widget = list(app.query(TranscriptEntryWidget))[-1]
            for _ in range(5):
                app.action_next_match()
                await pilot.pause()
                assert widget.collapsed is True
                assert len(str(widget.render())) < 1000
                # And the scroll target is measured over the preview too.
                assert widget.active_match_row() is not None

    def _mixed_transcript(self, session: Session) -> Transcript:
        """Matching blocks separated by runs of non-matching ones."""
        entries = [TranscriptEntry("user", "start of the session")]
        for index in range(3):
            entries += [
                TranscriptEntry("assistant", f"filler answer {index}"),
                TranscriptEntry(
                    "tool",
                    f'Bash({{"cmd": "echo {index}"}})',
                    metadata={"kind": "call", "tool": "Bash"},
                ),
                TranscriptEntry(
                    "tool",
                    f"unremarkable output {index}\n" + "w" * 600,
                    metadata={"kind": "output", "tool": "Bash"},
                ),
            ]
            entries.append(
                TranscriptEntry("user", f"and now the bandicoot question {index}")
            )
        return Transcript(session, entries)

    async def test_matching_only_keeps_whole_blocks_and_marks_the_gaps(self):
        """The projection is blocks, in order, with the omissions named."""
        from session_browser.app import TranscriptEntryWidget, TranscriptGapWidget

        app, fake = _make_app_with_rows(1)
        async with app.run_test(size=(120, 40)) as pilot:
            await _install_fake_sessions(app, pilot, fake)
            app._on_transcript_loaded(self._mixed_transcript(fake[0]))
            await pilot.pause()
            app._do_session_search("bandicoot")
            await pilot.pause()
            assert len(list(app.query(TranscriptEntryWidget))) == 13

            app.action_toggle_matches_only()
            await pilot.pause()

            widgets = list(app.query(TranscriptEntryWidget))
            assert [w.entry_index for w in widgets] == [4, 8, 12]
            # Whole blocks, not snippets: the entry text is the entry's own.
            assert all("bandicoot question" in w.entry.text for w in widgets)
            gaps = [str(g.render()) for g in app.query(TranscriptGapWidget)]
            assert len(gaps) == 3
            assert "4 non-matching blocks hidden" in gaps[0]
            assert all("3 non-matching blocks hidden" in g for g in gaps[1:])

            # The counter and the markers are drawn from one set: what is
            # shown plus what the gaps account for is what the full view had.
            counter = str(app.query_one("#match-counter").render())
            assert "3 of 13 blocks" in counter
            hidden = sum(int(g.split()[1]) for g in gaps)
            assert 3 + hidden == 13

            app.action_toggle_matches_only()
            await pilot.pause()
            assert len(list(app.query(TranscriptEntryWidget))) == 13
            assert not list(app.query(TranscriptGapWidget))

    async def test_matching_only_reports_its_state_and_its_counts(self):
        from textual.widgets import Label, Static

        app, fake = _make_app_with_rows(1)
        async with app.run_test(size=(120, 40)) as pilot:
            await _install_fake_sessions(app, pilot, fake)
            app._on_transcript_loaded(self._mixed_transcript(fake[0]))
            await pilot.pause()
            app._do_session_search("bandicoot")
            await pilot.pause()
            title = app.query_one("#transcript-title", Static)
            assert "MATCHING BLOCKS" not in str(title.render())

            app.action_toggle_matches_only()
            await pilot.pause()
            assert "MATCHING BLOCKS" in str(title.render())
            counter = str(app.query_one("#match-counter", Label).render())
            assert "1 / 3" in counter
            assert "3 of 13 blocks" in counter

            app.action_toggle_matches_only()
            await pilot.pause()
            assert "MATCHING BLOCKS" not in str(title.render())
            assert "blocks" not in str(app.query_one("#match-counter", Label).render())

    async def test_matches_only_needs_a_query_and_says_so(self):
        app, fake = _make_app_with_rows(1)
        async with app.run_test(size=(120, 40)) as pilot:
            await _install_fake_sessions(app, pilot, fake)
            app._on_transcript_loaded(self._mixed_transcript(fake[0]))
            await pilot.pause()

            app.action_toggle_matches_only()
            await pilot.pause()
            assert app._matches_only is False
            status = str(app.query_one("#status-bar").render())
            assert "Find in session first" in status
            assert "MATCHING BLOCKS" not in str(
                app.query_one("#transcript-title").render()
            )

    async def test_a_query_with_no_hits_projects_to_an_explicit_empty_state(self):
        from session_browser.app import TranscriptEntryWidget

        app, fake = _make_app_with_rows(1)
        async with app.run_test(size=(120, 40)) as pilot:
            await _install_fake_sessions(app, pilot, fake)
            app._on_transcript_loaded(self._mixed_transcript(fake[0]))
            await pilot.pause()
            app._do_session_search("bandicoot")
            await pilot.pause()
            app.action_toggle_matches_only()
            await pilot.pause()

            app._do_session_search("nothing-in-here-at-all")
            await pilot.pause()
            assert not list(app.query(TranscriptEntryWidget))
            content = str(app.query_one("#detail-content").render())
            assert "no matching blocks" in content

    async def test_clearing_the_query_restores_the_full_transcript(self):
        """The mode is a view of a query, so it lapses when the query does."""
        from session_browser.app import TranscriptEntryWidget

        app, fake = _make_app_with_rows(1)
        async with app.run_test(size=(120, 40)) as pilot:
            await _install_fake_sessions(app, pilot, fake)
            app._on_transcript_loaded(self._mixed_transcript(fake[0]))
            await pilot.pause()
            app._do_session_search("bandicoot")
            await pilot.pause()
            app.action_toggle_matches_only()
            await pilot.pause()
            assert len(list(app.query(TranscriptEntryWidget))) == 3

            app._do_session_search("")
            await pilot.pause()
            assert app._matches_only is True
            assert app._projecting is False
            assert len(list(app.query(TranscriptEntryWidget))) == 13
            assert "MATCHING BLOCKS" not in str(
                app.query_one("#transcript-title").render()
            )

            app._do_session_search("bandicoot")
            await pilot.pause()
            assert len(list(app.query(TranscriptEntryWidget))) == 3

    async def test_navigation_and_identity_survive_the_projection(self):
        """n/N, J/K, tool jumps and entry indices all still mean the same."""
        from session_browser.app import TranscriptEntryWidget

        app, fake = _make_app_with_rows(1)
        async with app.run_test(size=(120, 40)) as pilot:
            await _install_fake_sessions(app, pilot, fake)
            app._on_transcript_loaded(self._mixed_transcript(fake[0]))
            await pilot.pause()
            app._do_session_search("bandicoot")
            await pilot.pause()
            app.action_toggle_matches_only()
            await pilot.pause()

            widgets = list(app.query(TranscriptEntryWidget))
            # Absolute identity: the second retained block is still entry 8.
            assert widgets[1].entry_index == 8
            assert app._entry_spans[8][0] == widgets[1].text_start

            app.action_next_match()
            await pilot.pause()
            assert app._match_idx == 1
            app.action_prev_match()
            await pilot.pause()
            assert app._match_idx == 0

            app.action_next_entry()
            await pilot.pause()
            assert app.focused is widgets[0]
            app.action_next_entry()
            await pilot.pause()
            assert app.focused is widgets[1]
            # No tool block survived this projection, so ] is a no-op rather
            # than a jump into something the reader cannot see.
            app.action_next_tool()
            await pilot.pause()
            assert app.focused is widgets[1]

    async def test_the_projection_copies_and_exports_the_whole_session(self):
        """Display-only: the canonical buffer is untouched by the view."""
        app, fake = _make_app_with_rows(1)
        async with app.run_test(size=(120, 40)) as pilot:
            await _install_fake_sessions(app, pilot, fake)
            app._on_transcript_loaded(self._mixed_transcript(fake[0]))
            await pilot.pause()
            app._do_session_search("bandicoot")
            await pilot.pause()
            before = app._detail_text

            app.action_toggle_matches_only()
            await pilot.pause()
            assert app._detail_text == before
            assert len(app._transcript.entries) == 13
            assert len(app._matches) == 3
            assert app._entry_spans == app._build_entry_spans(app._transcript)

    async def test_the_projection_is_bounded_on_a_big_noisy_transcript(self):
        """Many matches and a megabyte block: rows stay windowed, not mounted
        one per match."""
        from session_browser.app import (
            _DISPLAY_WINDOW,
            TranscriptEntryWidget,
            TranscriptGapWidget,
        )

        app, fake = _make_app_with_rows(1)
        entries = [
            TranscriptEntry("assistant", f"potoroo mention {index}\n" + "p" * 400)
            for index in range(400)
        ]
        entries.insert(
            200,
            TranscriptEntry(
                "tool",
                "potoroo " + ("v" * 2_000_000),
                metadata={"kind": "output", "tool": "Bash"},
            ),
        )
        async with app.run_test(size=(120, 40)) as pilot:
            await _install_fake_sessions(app, pilot, fake)
            app._on_transcript_loaded(Transcript(fake[0], entries))
            await pilot.pause()
            app._do_session_search("potoroo")
            await pilot.pause()
            app.action_toggle_matches_only()
            await pilot.pause()

            widgets = list(app.query(TranscriptEntryWidget))
            assert widgets, "the projection mounted nothing at all"
            # The display window still governs. Every match is in a matching
            # block here, so an unwindowed projection would mount all 401 of
            # them; what is mounted is the windowful the full view would show.
            assert len(widgets) < len(entries)
            assert all(
                app._entry_spans[widget.entry_index][0]
                <= app._window_start + _DISPLAY_WINDOW
                for widget in widgets
            )
            # And the megabyte block inside it is still folded to a preview.
            megabyte = [w for w in widgets if len(w.entry.text) > 1_000_000]
            assert len(megabyte) == 1
            assert megabyte[0].collapsed is True
            assert len(str(megabyte[0].render())) < 1000
            for gap in app.query(TranscriptGapWidget):
                assert "hidden" in str(gap.render())

    async def test_the_projection_follows_a_session_switch(self):
        """The mode is a preference; the query is re-seeded per session."""
        from session_browser.app import TranscriptEntryWidget

        app, fake = _make_app_with_rows(2)
        async with app.run_test(size=(120, 40)) as pilot:
            await _install_fake_sessions(app, pilot, fake)
            app._on_transcript_loaded(self._mixed_transcript(fake[0]))
            await pilot.pause()
            app._do_session_search("bandicoot")
            await pilot.pause()
            app.action_toggle_matches_only()
            await pilot.pause()
            assert len(list(app.query(TranscriptEntryWidget))) == 3

            # A second session, loaded the way the worker delivers one.
            app._on_transcript_loaded(
                Transcript(
                    fake[1],
                    [
                        TranscriptEntry("user", "a bandicoot again"),
                        TranscriptEntry("assistant", "nothing to see"),
                        TranscriptEntry("assistant", "still nothing"),
                    ],
                )
            )
            await pilot.pause()
            app._do_session_search("bandicoot")
            await pilot.pause()

            assert app._matches_only is True
            widgets = list(app.query(TranscriptEntryWidget))
            assert [w.entry_index for w in widgets] == [0]
            assert "1 of 3 blocks" in str(app.query_one("#match-counter").render())

    async def test_the_projection_survives_focus_mode_and_a_narrow_screen(self):
        from session_browser.app import TranscriptEntryWidget, TranscriptGapWidget

        app, fake = _make_app_with_rows(1)
        async with app.run_test(size=(88, 28)) as pilot:
            await _install_fake_sessions(app, pilot, fake)
            assert app.screen.has_class("-compact")
            app._on_transcript_loaded(self._mixed_transcript(fake[0]))
            await pilot.pause()
            app._do_session_search("bandicoot")
            await pilot.pause()
            app.action_toggle_matches_only()
            await pilot.pause()
            assert len(list(app.query(TranscriptEntryWidget))) == 3
            assert len(list(app.query(TranscriptGapWidget))) == 3

            app.action_toggle_focus_mode()
            await pilot.pause()
            assert len(list(app.query(TranscriptEntryWidget))) == 3
            assert "MATCHING BLOCKS" in str(app.query_one("#transcript-title").render())

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
                await _cross_detail_window(app, pilot, app.action_nav_down)
                assert app._window_start == 250

                scroll.scroll_home(animate=False)
                await pilot.pause()
                app.action_nav_up()
                await app.workers.wait_for_complete()
                await pilot.pause()
                assert app._window_start == 0

    async def test_mouse_wheel_crosses_both_window_boundaries(self):
        from textual import events
        from textual.containers import VerticalScroll

        app, fake = _make_app_with_rows(1)
        with patch("session_browser.app._DISPLAY_WINDOW", 500):
            async with app.run_test(size=(120, 18)) as pilot:
                await _install_fake_sessions(app, pilot, fake)
                app._on_transcript_loaded(self._windowed_transcript(fake[0]))
                scroll = app.query_one("#detail-scroll", VerticalScroll)
                await pilot.pause()

                scroll.scroll_end(animate=False, immediate=True)
                await pilot.pause()
                down = events.MouseScrollDown(
                    scroll, 1, 1, 0, 1, 0, False, False, False
                )
                await _cross_detail_window(
                    app, pilot, lambda: scroll._on_mouse_scroll_down(down)
                )
                assert app._window_start == 250

                scroll.scroll_home(animate=False, immediate=True)
                await pilot.pause()
                up = events.MouseScrollUp(scroll, 1, 1, 0, -1, 0, False, False, False)
                scroll._on_mouse_scroll_up(up)
                await app.workers.wait_for_complete()
                await pilot.pause()
                assert app._window_start == 0


@pytest.mark.skipif(not _HAS_TEXTUAL, reason="textual not installed")
class TestBuildHierarchy:
    """The flattening: parents first, children under them, nothing lost."""

    def _s(self, sid, parent=None, kind=None):
        return Session(id=sid, provider="codex", parent_id=parent, subagent_kind=kind)

    def test_children_follow_their_parent_and_carry_a_depth(self):
        rows = [
            self._s("a"),
            self._s("b", "a", "thread_spawn"),
            self._s("c", "b", "review"),
            self._s("d"),
        ]
        ordered, meta = build_hierarchy(rows)
        assert [s.id for s in ordered] == ["a", "b", "c", "d"]
        assert [meta[s.id].depth for s in ordered] == [0, 1, 2, 0]
        assert meta["a"].child_count == 1 and meta["a"].has_children
        assert meta["c"].child_count == 0 and not meta["c"].has_children

    def test_a_subagent_with_no_visible_parent_is_marked_not_hidden(self):
        """The row still has to say what it is.

        11 of 74 Codex subagents on a real machine record no parent at all,
        and a filter can hide the parent of any of the rest. Rendering those
        as ordinary sessions is the duplicate-looking-row confusion this
        whole feature exists to remove.
        """
        rows = [self._s("a"), self._s("lost", "not-here", "guardian")]
        ordered, meta = build_hierarchy(rows)
        assert [s.id for s in ordered] == ["a", "lost"]
        assert meta["lost"].depth == 0
        assert meta["lost"].orphaned is True
        assert meta["a"].orphaned is False

    def test_collapsing_hides_the_whole_subtree_and_nothing_else(self):
        rows = [
            self._s("a"),
            self._s("b", "a", "review"),
            self._s("c", "b", "review"),
            self._s("d"),
        ]
        ordered, _ = build_hierarchy(rows, {"a"})
        assert [s.id for s in ordered] == ["a", "d"]
        ordered, _ = build_hierarchy(rows, {"b"})
        assert [s.id for s in ordered] == ["a", "b", "d"]

    def test_a_collapsed_child_is_hidden_rather_than_moved(self):
        """Regression: the cycle sweep re-added collapsed rows at the end.

        Marking the hidden subtree seen is what stops that, and without it
        collapsing a parent *relocated* its children to the bottom of the
        list instead of hiding them — worse than not collapsing at all.
        """
        rows = [self._s("a"), self._s("b", "a", "review"), self._s("d")]
        ordered, _ = build_hierarchy(rows, {"a"})
        assert "b" not in [s.id for s in ordered]

    def test_a_cycle_costs_the_nesting_not_the_sessions(self):
        """Parent ids come from another tool's metadata, so a cycle can
        arrive on disk. A session missing from the list is the one failure a
        browser must never have."""
        rows = [self._s("x", "y", "review"), self._s("y", "x", "review")]
        ordered, meta = build_hierarchy(rows)
        assert sorted(s.id for s in ordered) == ["x", "y"]
        assert all(meta[s.id].depth == 0 for s in ordered)

    def test_a_session_parented_to_itself_stays_a_root(self):
        rows = [self._s("self", "self", "review")]
        ordered, meta = build_hierarchy(rows)
        assert [s.id for s in ordered] == ["self"]
        assert meta["self"].depth == 0

    def test_disabled_returns_the_input_order_but_keeps_the_markers(self):
        rows = [self._s("a"), self._s("b", "a", "review")]
        ordered, meta = build_hierarchy(rows, enabled=False)
        assert [s.id for s in ordered] == ["a", "b"]
        assert [meta[s.id].depth for s in ordered] == [0, 0]
        # Flat is not the same as unlabelled: b is still a subagent.
        assert meta["b"].orphaned is True

    def test_top_level_order_is_whatever_the_sort_produced(self):
        rows = [self._s("z"), self._s("y"), self._s("zz", "z", "review")]
        ordered, _ = build_hierarchy(rows)
        assert [s.id for s in ordered] == ["z", "zz", "y"]


@pytest.mark.skipif(not _HAS_TEXTUAL, reason="textual not installed")
@pytest.mark.asyncio
class TestSessionTree:
    """The tree in the table: markers, expand/collapse, and the flat switch."""

    def _rows(self):
        return [
            Session(
                id="parent",
                provider="codex",
                summary="the orchestrating session",
                updated_at="2026-01-03",
            ),
            Session(
                id="kid1",
                provider="codex",
                summary="explorer",
                updated_at="2026-01-02",
                parent_id="parent",
                subagent_kind="thread_spawn",
            ),
            Session(
                id="kid2",
                provider="codex",
                summary="reviewer",
                updated_at="2026-01-01",
                parent_id="parent",
                subagent_kind="review",
            ),
            Session(
                id="lonely",
                provider="codex",
                summary="a subagent whose parent is gone",
                updated_at="2026-01-04",
                parent_id="vanished",
                subagent_kind="guardian",
            ),
        ]

    async def test_children_nest_under_their_parent_with_a_count(self):
        app = SessionBrowser()
        async with app.run_test(size=(120, 40)) as pilot:
            await _install_fake_sessions(app, pilot, self._rows())
            assert [s.id for s in app._table_sessions] == [
                "parent",
                "kid1",
                "kid2",
                "lonely",
            ]
            assert app._row_meta["kid1"].depth == 1
            assert app._row_meta["parent"].child_count == 2

    async def test_space_collapses_and_expands_the_highlighted_parent(self):
        app = SessionBrowser()
        async with app.run_test(size=(120, 40)) as pilot:
            await _install_fake_sessions(app, pilot, self._rows())
            table = app.query_one("#session-table", DataTable)
            table.move_cursor(row=0)
            await pilot.pause()

            app.action_toggle_children()
            await pilot.pause()
            assert [s.id for s in app._table_sessions] == ["parent", "lonely"]

            app.action_toggle_children()
            await pilot.pause()
            assert [s.id for s in app._table_sessions] == [
                "parent",
                "kid1",
                "kid2",
                "lonely",
            ]

    async def test_collapsing_keeps_the_cursor_on_the_parent(self):
        app = SessionBrowser()
        async with app.run_test(size=(120, 40)) as pilot:
            await _install_fake_sessions(app, pilot, self._rows())
            table = app.query_one("#session-table", DataTable)
            table.move_cursor(row=0)
            await pilot.pause()
            app.action_toggle_children()
            await pilot.pause()
            assert app._table_sessions[table.cursor_row].id == "parent"

    async def test_a_leaf_says_so_rather_than_doing_nothing(self):
        app = SessionBrowser()
        async with app.run_test(size=(120, 40)) as pilot:
            await _install_fake_sessions(app, pilot, self._rows())
            table = app.query_one("#session-table", DataTable)
            table.move_cursor(row=1)
            await pilot.pause()
            app.action_toggle_children()
            await pilot.pause()
            assert "No subagents" in str(app.query_one("#status-bar").render())

    async def test_the_flat_switch_returns_the_plain_list(self):
        app = SessionBrowser()
        async with app.run_test(size=(120, 40)) as pilot:
            await _install_fake_sessions(app, pilot, self._rows())
            app.action_toggle_tree()
            await pilot.pause()
            # The result list in its own order, untouched — flat is the
            # identity case, not a second ordering.
            assert [s.id for s in app._table_sessions] == [
                "parent",
                "kid1",
                "kid2",
                "lonely",
            ]
            assert all(m.depth == 0 for m in app._row_meta.values())
            assert "Flat session list" in str(app.query_one("#status-bar").render())

    async def test_selecting_a_nested_row_opens_that_session(self):
        """The row key indexes the rows on screen, which the tree reorders."""
        app = SessionBrowser()
        async with app.run_test(size=(120, 40)) as pilot:
            await _install_fake_sessions(app, pilot, self._rows())
            table = app.query_one("#session-table", DataTable)
            table.move_cursor(row=2)
            await pilot.pause()
            assert app._selected is not None
            assert app._selected.id == "kid2"

    async def test_the_row_shows_its_marker(self):
        app = SessionBrowser()
        async with app.run_test(size=(120, 40)) as pilot:
            await _install_fake_sessions(app, pilot, self._rows())
            table = app.query_one("#session-table", DataTable)
            cells = [str(table.get_cell_at((r, 0))) for r in range(table.row_count)]
            assert "▾2" in cells[0]
            assert "sub" in cells[3]


@pytest.mark.skipif(not _HAS_TEXTUAL, reason="textual not installed")
@pytest.mark.asyncio
class TestReadingFilter:
    """The f presets: role narrowing plus harness noise, display only."""

    def _transcript(self, session: Session) -> Transcript:
        """One of everything a Claude session actually records.

        Entry 7 is the false positive the predicate has to survive: a human
        message that quotes a slash-command marker without being one.
        """
        return Transcript(
            session,
            [
                TranscriptEntry("user", "so where are we with the plan?"),
                TranscriptEntry("assistant", "here is the plan"),
                TranscriptEntry(
                    "tool",
                    'Bash({"cmd": "ls"})',
                    metadata={"kind": "call", "tool": "Bash"},
                ),
                TranscriptEntry(
                    "tool",
                    "a.py b.py",
                    metadata={"kind": "output", "tool": "Bash"},
                ),
                TranscriptEntry(
                    "user",
                    "<command-name>/clear</command-name>\n"
                    "<command-message>clear</command-message>",
                ),
                TranscriptEntry(
                    "user",
                    "<local-command-stdout>Set model to Opus 5</local-command-stdout>",
                ),
                TranscriptEntry(
                    "system",
                    "<task-notification>\n<task-id>abc</task-id>",
                    metadata={"kind": "injected", "subtype": "task_notification"},
                ),
                TranscriptEntry(
                    "user", "note that <command-name> shows up as a user turn"
                ),
                TranscriptEntry("assistant", "understood"),
                TranscriptEntry(
                    "tool",
                    "boom",
                    metadata={"kind": "output", "tool": "Bash", "is_error": True},
                ),
            ],
        )

    async def _loaded(self, app, pilot, fake):
        await _install_fake_sessions(app, pilot, fake)
        app._on_transcript_loaded(self._transcript(fake[0]))
        await pilot.pause()

    async def test_each_preset_keeps_exactly_what_it_names(self):
        from session_browser.app import TranscriptEntryWidget

        app, fake = _make_app_with_rows(1)
        async with app.run_test(size=(120, 40)) as pilot:
            await self._loaded(app, pilot, fake)

            def shown():
                return [w.entry_index for w in app.query(TranscriptEntryWidget)]

            assert shown() == list(range(10))

            app.action_cycle_entry_filter()
            await pilot.pause()
            # Conversation: both voices, no tool traffic, no harness bookkeeping.
            assert shown() == [0, 1, 7, 8]

            app.action_cycle_entry_filter()
            await pilot.pause()
            assert shown() == [0, 7]

            app.action_cycle_entry_filter()
            await pilot.pause()
            assert shown() == [1, 8]

            # And back round to the whole thing.
            app.action_cycle_entry_filter()
            await pilot.pause()
            assert shown() == list(range(10))

    async def test_a_quoted_marker_is_still_a_message(self):
        """Leading marker only: quoting one of these is speaking, not noise."""
        from session_browser.app import TranscriptEntryWidget

        app, fake = _make_app_with_rows(1)
        async with app.run_test(size=(120, 40)) as pilot:
            await self._loaded(app, pilot, fake)
            app.action_cycle_entry_filter()
            app.action_cycle_entry_filter()
            await pilot.pause()

            texts = [w.entry.text for w in app.query(TranscriptEntryWidget)]
            assert any("shows up as a user turn" in t for t in texts)
            assert not any(t.startswith("<command-name>") for t in texts)
            assert not any(t.startswith("<local-command-stdout>") for t in texts)

    async def test_the_title_and_gaps_name_the_preset(self):
        from session_browser.app import TranscriptGapWidget

        app, fake = _make_app_with_rows(1)
        async with app.run_test(size=(120, 40)) as pilot:
            await self._loaded(app, pilot, fake)
            title = app.query_one("#transcript-title")
            assert "CONVERSATION" not in str(title.render())

            app.action_cycle_entry_filter()
            await pilot.pause()
            assert "CONVERSATION" in str(title.render())
            gaps = [str(g.render()) for g in app.query(TranscriptGapWidget)]
            assert "5 filtered blocks hidden" in gaps[0]
            assert "1 filtered block hidden" in gaps[1]
            # Shown plus accounted-for is the whole window, as for m.
            assert 4 + 5 + 1 == 10
            assert "4 of 10 blocks" in str(app.query_one("#match-counter").render())

    async def test_the_filter_and_matching_blocks_stack(self):
        """Two projections, one set of gaps, and a title naming both."""
        from session_browser.app import TranscriptEntryWidget, TranscriptGapWidget

        app, fake = _make_app_with_rows(1)
        async with app.run_test(size=(120, 40)) as pilot:
            await self._loaded(app, pilot, fake)
            app._do_session_search("plan")
            await pilot.pause()
            app.action_toggle_matches_only()
            await pilot.pause()
            # The query alone reaches both voices.
            assert [w.entry_index for w in app.query(TranscriptEntryWidget)] == [0, 1]

            app.action_cycle_entry_filter()
            await pilot.pause()
            assert [w.entry_index for w in app.query(TranscriptEntryWidget)] == [0, 1]

            app.action_cycle_entry_filter()
            await pilot.pause()
            assert [w.entry_index for w in app.query(TranscriptEntryWidget)] == [0]

            title = str(app.query_one("#transcript-title").render())
            assert "YOUR TURNS" in title and "MATCHING BLOCKS" in title
            # With both on a run can hold blocks dropped by either, so the
            # marker claims neither cause.
            gaps = [str(g.render()) for g in app.query(TranscriptGapWidget)]
            assert gaps == [g for g in gaps if "9 blocks hidden" in g]
            assert len(gaps) == 1

    async def test_a_preset_that_leaves_nothing_says_which_key_widens_it(self):
        from session_browser.app import TranscriptEntryWidget

        app, fake = _make_app_with_rows(1)
        async with app.run_test(size=(120, 40)) as pilot:
            await _install_fake_sessions(app, pilot, fake)
            app._on_transcript_loaded(
                Transcript(
                    fake[0],
                    [TranscriptEntry("user", "one thing, then I left")],
                )
            )
            await pilot.pause()
            for _ in range(3):
                app.action_cycle_entry_filter()
            await pilot.pause()

            assert not list(app.query(TranscriptEntryWidget))
            content = str(app.query_one("#detail-content").render())
            assert "no agent turns in this session" in content
            assert "f to widen" in content

    async def test_an_empty_stack_names_both_ways_out(self):
        app, fake = _make_app_with_rows(1)
        async with app.run_test(size=(120, 40)) as pilot:
            await self._loaded(app, pilot, fake)
            # A phrase only the human used, so narrowing to the agent's turns
            # empties the pane for a reason neither key owns alone.
            app._do_session_search("where are we")
            await pilot.pause()
            app.action_toggle_matches_only()
            app.action_cycle_entry_filter()
            app.action_cycle_entry_filter()
            app.action_cycle_entry_filter()
            await pilot.pause()

            content = str(app.query_one("#detail-content").render())
            assert "no agent turns blocks matched" in content
            assert "f or m to widen" in content

    async def test_the_canonical_session_is_untouched(self):
        """Indices, matches, n/N, the flat buffer and the export are the same."""
        from session_browser.app import TranscriptEntryWidget

        app, fake = _make_app_with_rows(1)
        async with app.run_test(size=(120, 40)) as pilot:
            await self._loaded(app, pilot, fake)
            app._do_session_search("plan")
            await pilot.pause()
            before_text = app._detail_text
            before_matches = list(app._matches)

            app.action_cycle_entry_filter()
            await pilot.pause()

            assert app._detail_text == before_text
            assert app._matches == before_matches
            assert "1 / 2" in str(app.query_one("#match-counter").render())
            # Absolute positions, not positions within the projection.
            assert [w.entry_index for w in app.query(TranscriptEntryWidget)] == [
                0,
                1,
                7,
                8,
            ]
            # The export reads the flat buffer, so it still carries the
            # harness noise the reading view drops.
            export = build_chat_export(fake[0], app._detail_text)
            assert "<command-name>/clear</command-name>" in export

            app.action_next_match()
            await pilot.pause()
            assert "2 / 2" in str(app.query_one("#match-counter").render())

    async def test_the_preset_survives_a_session_switch(self):
        """Unlike m, it is a reading preference rather than a view of a query."""
        from session_browser.app import TranscriptEntryWidget

        app, fake = _make_app_with_rows(2)
        async with app.run_test(size=(120, 40)) as pilot:
            await self._loaded(app, pilot, fake)
            app.action_cycle_entry_filter()
            await pilot.pause()
            assert app._filter_idx == 1

            app._on_transcript_loaded(
                Transcript(
                    fake[1],
                    [
                        TranscriptEntry("user", "second session"),
                        TranscriptEntry(
                            "tool", "ls", metadata={"kind": "call", "tool": "Bash"}
                        ),
                        TranscriptEntry("assistant", "answered"),
                    ],
                )
            )
            await pilot.pause()

            assert app._filter_idx == 1
            assert [w.entry_index for w in app.query(TranscriptEntryWidget)] == [0, 2]
            assert "CONVERSATION" in str(app.query_one("#transcript-title").render())

    async def test_cycling_is_never_refused_and_says_what_it_did(self):
        """m needs a query; a preset is a legitimate view of any transcript."""
        app, fake = _make_app_with_rows(1)
        async with app.run_test(size=(120, 40)) as pilot:
            await self._loaded(app, pilot, fake)
            app.action_cycle_entry_filter()
            await pilot.pause()
            assert "Showing conversation" in str(app.query_one("#status-bar").render())

            for _ in range(3):
                app.action_cycle_entry_filter()
            await pilot.pause()
            assert app._filter_idx == 0
            assert "Showing everything" in str(app.query_one("#status-bar").render())

    async def test_f_does_not_fire_while_typing_in_a_search_box(self):
        app, fake = _make_app_with_rows(1)
        async with app.run_test(size=(120, 40)) as pilot:
            await self._loaded(app, pilot, fake)
            from session_browser.app import SearchInput

            search = app.query_one("#global-search", SearchInput)
            search.focus()
            await pilot.pause()
            await pilot.press("f")
            await pilot.pause()

            assert app._filter_idx == 0
            assert search.value == "f"

    def _commands_transcript(self, session: Session) -> Transcript:
        """Commands that carry a request beside ones that only drive the
        harness, in both shapes: Claude's tag block and Pi's skill expansion."""
        skill_body = "# goodbye\n\n" + "End the session cleanly.\n" * 200
        return Transcript(
            session,
            [
                TranscriptEntry(
                    "user",
                    "<command-message>goal</command-message>\n"
                    "<command-name>/goal</command-name>\n"
                    "<command-args>fix the login bug</command-args>",
                ),
                TranscriptEntry(
                    "user",
                    "<command-name>/clear</command-name>\n<command-args></command-args>",
                ),
                TranscriptEntry(
                    "user",
                    f'<skill name="goodbye" location="/s/SKILL.md">\n{skill_body}'
                    "</skill>\n\nwe'll pick this up tomorrow",
                ),
                TranscriptEntry(
                    "user",
                    f'<skill name="goodbye" location="/s/SKILL.md">\n{skill_body}</skill>',
                ),
                TranscriptEntry("assistant", "done"),
            ],
        )

    async def test_a_command_carrying_a_request_is_conversation(self):
        """/goal fix the login bug is the person speaking; a bare /clear and a
        bare skill are not, so CONVERSATION and YOUR TURNS keep only the first
        kind."""
        from session_browser.app import TranscriptEntryWidget

        app, fake = _make_app_with_rows(1)
        async with app.run_test(size=(120, 40)) as pilot:
            await _install_fake_sessions(app, pilot, fake)
            app._on_transcript_loaded(self._commands_transcript(fake[0]))
            await pilot.pause()

            def shown():
                return [w.entry_index for w in app.query(TranscriptEntryWidget)]

            app.action_cycle_entry_filter()
            await pilot.pause()
            assert shown() == [0, 2, 4]

            app.action_cycle_entry_filter()
            await pilot.pause()
            assert shown() == [0, 2]

    async def test_a_pi_skill_folds_to_what_was_typed(self):
        """The skill's own file is folded away, the preview is the line the
        person typed after the command, and the header names the skill."""
        from session_browser.app import TranscriptEntryWidget

        app, fake = _make_app_with_rows(1)
        async with app.run_test(size=(120, 40)) as pilot:
            await _install_fake_sessions(app, pilot, fake)
            app._on_transcript_loaded(self._commands_transcript(fake[0]))
            await pilot.pause()

            widgets = {w.entry_index: w for w in app.query(TranscriptEntryWidget)}
            typed, bare = widgets[2], widgets[3]
            assert typed.collapsed and bare.collapsed
            text, _offset, footer = typed._visible_text()
            assert text == "we'll pick this up tomorrow"
            assert "chars hidden" in footer
            assert "/skill:goodbye" in typed._header_markup()
            # Nothing typed: the opening tag at least names what ran.
            assert bare._visible_text()[0].startswith('<skill name="goodbye"')
            # A Claude command is short and stays as it was.
            assert not widgets[0].collapsible

            typed.action_toggle_collapsed()
            assert typed._visible_text()[0] == typed.entry.text

    async def test_the_footer_offers_the_conversation_view(self):
        """It shipped behind ? and was then asked for as a new feature."""
        from textual.widgets import Footer

        app, fake = _make_app_with_rows(1)
        async with app.run_test(size=(160, 40)) as pilot:
            await self._loaded(app, pilot, fake)
            shown = {key for key, b in app.active_bindings.items() if b.binding.show}
            assert "f" in shown
            assert app.query_one(Footer)


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
                "claude --resume def-456",
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
        assert send[-1] == "claude --resume x"
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
    kind="id",
):
    """One entry as `herdr pane list` prints it.

    ``kind`` is how herdr identified the session: ``"id"`` for claude and
    codex, ``"path"`` for pi, whose detector reports the transcript file."""
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
            "kind": kind,
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

    def test_pi_pane_is_matched_by_transcript_path(self):
        """Herdr identifies a pi pane by the transcript file it has open, not
        by an id, and says so in ``kind``. The path is compared against the
        session's content_path, normalised on both sides."""
        path = "/Users/test/.pi/agent/sessions/--Users-test-game--/s.jsonl"
        panes = [
            HerdrPane("w1:p1", "w1:t1", "w1", "/a", "pi", path, "path"),
        ]
        found = herdr.find_pane(
            panes,
            "pi",
            {"01a01601"},
            "/Users/test/.pi/./agent/sessions/--Users-test-game--/s.jsonl",
        )
        assert found is not None and found.pane_id == "w1:p1"

    def test_pi_pane_on_another_transcript_is_not_a_match(self):
        panes = [
            HerdrPane("w1:p1", "w1:t1", "w1", "/a", "pi", "/other/s.jsonl", "path"),
        ]
        assert herdr.find_pane(panes, "pi", {"01a01601"}, "/mine/s.jsonl") is None

    def test_a_path_pane_is_never_matched_against_the_id_set(self):
        """Ids and paths are different namespaces; a hit across them could
        only ever be an accident."""
        panes = [HerdrPane("w1:p1", "w1:t1", "w1", "/a", "pi", "abc", "path")]
        assert herdr.find_pane(panes, "pi", {"abc"}) is None

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
                "claude --resume def-456",
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

    def test_reuses_a_pi_pane_reported_by_transcript_path(self, tmp_path):
        # pi's detector names the session by the file it has open, so reuse
        # has to compare against content_path rather than the id.
        transcript = tmp_path / "2026-08-18T17-52-48-324Z_01a01601.jsonl"
        transcript.write_text("{}\n")
        calls, fake_run = _herdr_stub(
            panes=[
                _herdr_pane(
                    "w4:p2",
                    cwd="/home/user/proj",
                    agent="pi",
                    session_id=str(transcript),
                    kind="path",
                ),
            ]
        )
        with (
            patch.object(herdr.shutil, "which", return_value="/usr/bin/herdr"),
            patch.object(herdr.subprocess, "run", side_effect=fake_run),
        ):
            plan = herdr.prepare_session(
                "pi", "01a01601", "/home/user/proj", content_path=str(transcript)
            )

        assert (plan.reused, plan.pane) == (True, "w4:p2")
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
        # Two targets may share a name -- the original herdr and herdr-core
        # are one "herdr" choice -- and then they share a key too; different
        # names never do.
        key_for = {t.name: t.key for t in multiplexer.TARGETS}
        assert len(set(key_for.values())) == len(key_for)
        for target in multiplexer.TARGETS:
            assert target.key == key_for[target.name]

    def test_only_running_multiplexers_are_offered(self):
        with (
            patch.object(tmux, "available", return_value=False),
            patch.object(herdr, "available", return_value=True),
            patch.object(herdr_core, "available", return_value=False),
        ):
            assert multiplexer.available_names() == ["herdr"]

    def test_both_running_keeps_tmux_first(self):
        with (
            patch.object(tmux, "available", return_value=True),
            patch.object(herdr, "available", return_value=True),
            patch.object(herdr_core, "available", return_value=False),
        ):
            assert multiplexer.available_names() == ["tmux", "herdr"]

    def test_neither_running(self):
        with (
            patch.object(tmux, "available", return_value=False),
            patch.object(herdr, "available", return_value=False),
            patch.object(herdr_core, "available", return_value=False),
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


# ---------------------------------------------------------------------------
# Word matching (TUI global search fallback)
# ---------------------------------------------------------------------------

TARGET_TURN = "there is a sibling codex pane you can ask for a second opinion or review"


def _claude_file(path, *turns):
    """A Claude transcript of (role, text) turns; role "tool" writes *text*
    as a tool result, which is how something the agent read is recorded."""
    lines = []
    for role, text in turns:
        if role == "tool":
            content = [{"type": "tool_result", "content": text}]
            record = {"type": "user", "message": {"role": "user", "content": content}}
        else:
            record = {"type": role, "message": {"role": role, "content": text}}
        lines.append(json.dumps(record) + "\n")
    path.write_text("".join(lines))
    return str(path)


class TestUnorderedTerms:
    def test_terms_are_normalized_words_without_repeats_or_single_letters(self):
        from session_browser.transcript import unordered_terms

        assert unordered_terms("Codex  PANE review pane") == ["codex", "pane", "review"]
        assert unordered_terms("a pane") == []  # one usable term is no fallback
        assert unordered_terms('"codex pane" review') == []  # quoted = literal

    def test_reordered_words_in_one_entry_match(self):
        from session_browser.transcript import (
            normalize_match_text,
            unordered_terms,
            unordered_window,
        )

        folded = normalize_match_text(TARGET_TURN)
        assert unordered_window(folded, unordered_terms("review codex pane"))

    def test_words_too_far_apart_do_not(self):
        from session_browser.transcript import unordered_terms, unordered_window

        far = "codex pane " + "x " * 120 + "review"
        assert unordered_window(far, unordered_terms("codex pane review")) is None

    def test_the_tightest_window_wins(self):
        from session_browser.transcript import unordered_terms, unordered_window

        text = "review this. " + "filler " * 5 + "then codex pane review"
        start, end, spans = unordered_window(text, unordered_terms("codex pane review"))
        assert text[start:end] == "codex pane review"
        assert [text[s:e] for s, e in spans] == ["codex", "pane", "review"]


class TestWordSearch:
    """search_session_word_hits: every word, somewhere in the conversation."""

    def _session(self, tmp_path, name, *turns, **fields):
        return Session(
            id=name,
            provider="claude",
            content_path=_claude_file(tmp_path / f"{name}.jsonl", *turns),
            **fields,
        )

    def _hits(self, sessions, query):
        from session_browser.transcript import search_session_word_hits

        return search_session_word_hits(sessions, query)

    def test_words_in_one_entry_match_and_show_the_window(self, tmp_path):
        from session_browser.transcript import WORDS_YOURS

        target = self._session(tmp_path, "target", ("user", TARGET_TURN))
        found = self._hits([target], "review codex pane")["target"]
        assert found.strength == WORDS_YOURS
        hit = found.hit
        assert hit.role == "user" and hit.count == 1
        assert hit.match.startswith("codex") and hit.match.endswith("review")

    def test_words_scattered_across_turns_match(self, tmp_path):
        """The reported miss: a cue typed 75 entries after the others used to
        lose the session, because every word had to share one entry."""
        from session_browser.transcript import WORDS_YOURS

        scattered = self._session(
            tmp_path,
            "scattered",
            ("user", "close the ticket unless it has a release watch"),
            ("assistant", "closing them now"),
            ("user", "/yolo"),
        )
        missing_one = self._session(
            tmp_path,
            "missing-one",
            ("user", "close the ticket unless it has a release watch"),
        )
        hits = self._hits([scattered, missing_one], "watch ticket yolo")
        assert set(hits) == {"scattered"}
        assert hits["scattered"].strength == WORDS_YOURS
        # The row shows a turn a person wrote, with one of the words marked.
        assert hits["scattered"].hit.role == "user"
        assert hits["scattered"].hit.match.lower() in ("watch", "ticket", "yolo")

    def test_grades_by_who_said_the_words(self, tmp_path):
        from session_browser.transcript import (
            WORDS_CLOSE,
            WORDS_QUOTED,
            WORDS_SAID,
            WORDS_YOURS,
        )

        apart = "the watch " + "filler " * 60 + "on that ticket"
        sessions = [
            self._session(
                tmp_path, "yours", ("user", "set a watch"), ("user", "on the ticket")
            ),
            self._session(
                tmp_path, "close", ("assistant", "the watch on that ticket is green")
            ),
            self._session(
                tmp_path,
                "said",
                ("user", "set a watch"),
                ("assistant", "ticket closed"),
            ),
            self._session(tmp_path, "quoted", ("tool", "watch: ticket T-1 GREEN")),
            # A subagent's "user" turn is the prompt its parent wrote.
            self._session(
                tmp_path,
                "subagent",
                ("user", "set a watch"),
                ("user", "on the ticket"),
                parent_id="yours",
            ),
            # Apart in tool text, or apart in ONE entry of it: not a match.
            self._session(
                tmp_path, "tool-scatter", ("tool", "watch"), ("tool", "ticket")
            ),
            self._session(tmp_path, "tool-apart", ("tool", apart)),
        ]
        hits = self._hits(sessions, "watch ticket")
        assert {sid: h.strength for sid, h in hits.items()} == {
            "yours": WORDS_YOURS,
            "close": WORDS_CLOSE,
            "said": WORDS_SAID,
            "quoted": WORDS_QUOTED,
            "subagent": WORDS_SAID,
        }
        assert hits["quoted"].hit.role == "tool"

    def test_metadata_holds_a_word_like_any_other_cue(self, tmp_path, monkeypatch):
        from session_browser.transcript import (
            WORDS_SAID,
            WORDS_YOURS,
            metadata_word_strengths,
        )

        monkeypatch.setenv("HOME", "/home/sam")
        turn = ("user", "close the ticket")
        sessions = [
            # The project is a directory ABOVE the working one: a worktree.
            self._session(
                tmp_path,
                "worktree",
                turn,
                cwd="/home/sam/worktrees/Acme_New/reset-minutes",
            ),
            self._session(tmp_path, "elsewhere", turn, cwd="/home/sam/other"),
            # An id is hexadecimal, and "cafe" is a hexadecimal word.
            Session(
                id="00cafe00",
                provider="claude",
                cwd="/home/sam/Acme_New",
                content_path=_claude_file(tmp_path / "hex.jsonl", ("user", "hello")),
            ),
            # The home prefix is in every path, so it is not a cue.
            self._session(tmp_path, "home", turn, cwd="/home/sam/other"),
        ]
        hits = self._hits(sessions, "acme ticket")
        assert {sid: h.strength for sid, h in hits.items()} == {"worktree": WORDS_YOURS}
        assert self._hits(sessions, "sam ticket") == {}
        assert self._hits(sessions, "acme cafe") == {}
        # The directory projects are kept in admits a session, and is not
        # a word anyone typed: "worktrees" here, ~/Projects on most machines.
        shelved = self._hits(sessions, "worktrees ticket")
        assert {sid: h.strength for sid, h in shelved.items()} == {
            "worktree": WORDS_SAID
        }

        # Provider and branch admit a session without counting as typed.
        on_branch = self._session(
            tmp_path, "branch", ("assistant", "ticket closed"), branch="fix/watch"
        )
        assert self._hits([on_branch], "watch ticket")["branch"].strength == WORDS_SAID

        # Metadata alone, with no transcript read at all.
        titled = Session(
            id="titled",
            provider="claude",
            summary="/yolo grab an open ticket",
            cwd="/home/sam/Acme_New",
        )
        delegated = Session(
            id="delegated",
            provider="claude",
            summary="/yolo grab an open ticket",
            cwd="/home/sam/other",
            parent_id="titled",
        )
        assert metadata_word_strengths([titled, delegated, *sessions], "acme yolo") == {
            "titled": WORDS_YOURS
        }
        assert metadata_word_strengths([titled, delegated], "ticket yolo") == {
            "titled": WORDS_YOURS,
            "delegated": WORDS_SAID,
        }
        assert metadata_word_strengths([titled], '"ticket yolo"') == {}


@pytest.mark.skipif(not _HAS_TEXTUAL, reason="textual not installed")
@pytest.mark.asyncio
class TestUnorderedGlobalSearch:
    async def _search(self, app, pilot, query):
        search = app.query_one("#global-search")
        search.value = query
        app._apply_filter(query)
        # The literal wave, then the unordered wave it launches. Setting the
        # value has already started a search of its own, which the call above
        # supersedes; waiting on that cancelled worker raises, about one run
        # in eight, depending on whether it had started.
        for _ in range(3):
            with contextlib.suppress(WorkerCancelled):
                await app.workers.wait_for_complete()
            await pilot.pause()

    async def test_literal_hits_rank_first_and_unordered_ones_open_with_matches(
        self, tmp_path
    ):
        app = SessionBrowser()
        literal = Session(
            id="literal",
            provider="claude",
            summary="older",
            updated_at="2026-01-01",
            content_path=_claude_file(
                tmp_path / "l.jsonl", ("user", "the codex pane review step")
            ),
        )
        loose = Session(
            id="loose",
            provider="claude",
            summary="newer",
            updated_at="2026-02-01",
            content_path=_claude_file(tmp_path / "u.jsonl", ("user", TARGET_TURN)),
        )
        neither = Session(
            id="neither",
            provider="claude",
            summary="unrelated",
            updated_at="2026-03-01",
            content_path=_claude_file(tmp_path / "n.jsonl", ("user", "codex only")),
        )
        async with app.run_test(size=(160, 40)) as pilot:
            await _install_fake_sessions(app, pilot, [neither, loose, literal])
            await self._search(app, pilot, "codex pane review")
            # The phrase hit outranks the newer unordered one.
            assert [s.id for s in app._filtered] == ["literal", "loose"]
            assert "loose" in app._word_hits
            assert "literal" not in app._word_hits

            # Reordered: the phrase is in neither, so both come from the
            # fallback, and recency orders them as it does any one tier.
            await self._search(app, pilot, "review codex pane")
            assert [s.id for s in app._filtered] == ["loose", "literal"]
            assert set(app._word_hits) == {"loose", "literal"}

            # Opening the unordered hit never shows "0 matches": each term
            # is highlighted inside the window that matched.
            app._select_session(str([s.id for s in app._table_sessions].index("loose")))
            await app.workers.wait_for_complete()
            await pilot.pause()
            assert app._selected.id == "loose"
            words = [app._detail_text[s:e].lower() for s, e in app._matches]
            assert sorted(words) == ["codex", "pane", "review"]

    async def test_a_quoted_query_gets_no_unordered_wave(self, tmp_path):
        app = SessionBrowser()
        loose = Session(
            id="loose",
            provider="claude",
            content_path=_claude_file(tmp_path / "u.jsonl", ("user", TARGET_TURN)),
        )
        async with app.run_test() as pilot:
            await _install_fake_sessions(app, pilot, [loose])
            await self._search(app, pilot, '"review codex pane"')
            assert app._filtered == [] and app._word_hits == {}

    async def test_another_cue_narrows_to_the_session_instead_of_losing_it(
        self, tmp_path
    ):
        """Words said in different turns find the session; the one where a
        person typed them outranks a newer one where the agent supplied
        some; and opening it marks each word where it was said."""
        app = SessionBrowser()
        wanted = Session(
            id="wanted",
            provider="claude",
            summary="is this still an issue?",
            updated_at="2026-01-01",
            content_path=_claude_file(
                tmp_path / "w.jsonl",
                ("user", "close the ticket unless it has a release watch"),
                ("tool", "ticket T-1 closed: watch GREEN, yolo mode off"),
                ("assistant", "done"),
                ("user", "/yolo"),
            ),
        )
        agent_said = Session(
            id="agent-said",
            provider="claude",
            summary="tidy up",
            updated_at="2026-02-01",
            content_path=_claude_file(
                tmp_path / "a.jsonl",
                ("user", "grab a ticket"),
                ("assistant", "I will watch the build"),
                ("assistant", "running in yolo mode"),
            ),
        )
        two_of_three = Session(
            id="two-of-three",
            provider="claude",
            summary="other",
            updated_at="2026-03-01",
            content_path=_claude_file(
                tmp_path / "t.jsonl", ("user", "watch the ticket")
            ),
        )
        async with app.run_test(size=(220, 40)) as pilot:
            await _install_fake_sessions(app, pilot, [two_of_three, agent_said, wanted])
            await self._search(app, pilot, "watch ticket")
            assert {s.id for s in app._filtered} == {
                "two-of-three",
                "agent-said",
                "wanted",
            }
            await self._search(app, pilot, "watch ticket yolo")
            assert [s.id for s in app._filtered] == ["wanted", "agent-said"]

            table = app.query_one("#session-table")
            assert app._table_column_mode == "full"
            summary = str(table.get_row_at(0)[-1])
            # Found by its words, and showing a turn a person wrote.
            assert "≈" in summary and "release" in summary

            app._select_session("0")
            await app.workers.wait_for_complete()
            await pilot.pause()
            assert app._selected.id == "wanted"
            words = [app._detail_text[s:e].lower() for s, e in app._matches]
            # Each word where a person or the agent said it. The tool listing
            # holds all three together, and is not what gets marked: the row
            # was found, and ranked, by what was said.
            assert words == ["ticket", "watch", "yolo"]

            # Two of three words is not a match, so nothing is marked in a
            # session that never reached this query's results.
            app._select_session(
                str([s.id for s in app._table_sessions].index("agent-said"))
            )
            await app.workers.wait_for_complete()
            await pilot.pause()
            app.query_one("#detail-search").value = "watch ticket pineapple"
            await pilot.pause()
            assert app._matches == []

    async def test_words_held_by_metadata_show_before_any_transcript_is_read(
        self, tmp_path, monkeypatch
    ):
        monkeypatch.setenv("HOME", "/home/sam")
        app = SessionBrowser()
        titled = Session(
            id="titled",
            provider="claude",
            summary="/yolo grab an open ticket",
            cwd="/home/sam/worktrees/Acme_New/reset-minutes",
            content_path=_claude_file(tmp_path / "m.jsonl", ("user", "hello")),
        )
        other = Session(
            id="other",
            provider="claude",
            summary="/yolo grab an open ticket",
            cwd="/home/sam/elsewhere",
            content_path=_claude_file(tmp_path / "o.jsonl", ("user", "hello")),
        )
        async with app.run_test(size=(160, 40)) as pilot:
            await _install_fake_sessions(app, pilot, [titled, other])
            app.query_one("#global-search").value = "acme yolo"
            app._apply_filter("acme yolo")
            # No worker has run: the row is there on the keystroke.
            assert [s.id for s in app._filtered] == ["titled"]
            for _ in range(3):
                await app.workers.wait_for_complete()
                await pilot.pause()
            assert [s.id for s in app._filtered] == ["titled"]


@pytest.mark.skipif(not _HAS_TEXTUAL, reason="textual not installed")
def test_a_search_inside_the_running_app_can_use_worker_processes(tmp_path):
    """Textual's print capture gives stderr a fileno() of -1, and the
    multiprocessing resource tracker passes that fd to the child it spawns, so
    a pool first started inside the app failed and every TUI search parsed on
    threads: the words wave ran 3.9x slower in the app than standalone. The
    app now starts the tracker before it runs.

    In a fresh interpreter, because in this one an earlier test has usually
    started the tracker already, and the test would pass without the fix."""
    script = tmp_path / "probe_in_app.py"
    script.write_text(
        textwrap.dedent(
            """
            import asyncio, json, sys
            from pathlib import Path
            from session_browser import transcript as t
            from session_browser.app import SessionBrowser
            from session_browser.discovery import Session

            def parse():
                path = Path(sys.argv[1])
                path.write_text(json.dumps(
                    {"type": "user", "message": {"content": "a wombat"}}) + "\\n")
                s = Session(id="s", provider="claude", content_path=str(path))
                out = t._probe_in_processes([(0, s)], "wombat", None)
                return "fallback" if out is t._PROC_FALLBACK else out[0].match_count

            async def main():
                app = SessionBrowser()
                async with app.run_test() as pilot:
                    worker = app.run_worker(parse, thread=True)
                    print("RESULT", await worker.wait())

            if __name__ == "__main__":
                asyncio.run(main())
            """
        )
    )
    done = subprocess.run(
        [sys.executable, str(script), str(tmp_path / "t.jsonl")],
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
        env={**os.environ, "HOME": str(tmp_path)},
    )
    assert "RESULT 1" in done.stdout, done.stdout + done.stderr
