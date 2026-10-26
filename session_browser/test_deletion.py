"""Deleting a session from its provider's store, and the TUI path to it."""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path
from unittest.mock import patch

import pytest

from session_browser import compressed, deletion
from session_browser.deletion import DeleteFailed, DeleteRefused
from session_browser.discovery import Session, scan_claude

CODEX_ID = "01a13ec3-2559-7c02-bf0a-4c0ad05c0177"


@pytest.fixture
def home(tmp_path):
    with patch("session_browser.discovery.Path.home", return_value=tmp_path):
        yield tmp_path


def _claude_session(home: Path, sid: str = "abc-123") -> Session:
    """A top-level Claude session with everything a real interactive one
    leaves behind: subagents and tool output beside the transcript, plus
    session-env and file-history under ~/.claude."""
    project = home / ".claude" / "projects" / "-Users-test"
    project.mkdir(parents=True, exist_ok=True)
    line = {
        "type": "user",
        "message": {"role": "user", "content": "Fix the login page"},
        "cwd": "/Users/test/project",
        "timestamp": "2026-04-14T10:00:00Z",
    }
    (project / f"{sid}.jsonl").write_text(json.dumps(line) + "\n")
    agents = project / sid / "subagents"
    agents.mkdir(parents=True)
    (agents / "agent-a1.jsonl").write_text(json.dumps(line) + "\n")
    (agents / "agent-a1.meta.json").write_text('{"agentType": "Explore"}')
    (project / sid / "tool-results").mkdir()
    (home / ".claude" / "session-env" / sid).mkdir(parents=True)
    (home / ".claude" / "file-history" / sid).mkdir(parents=True)
    (home / ".claude" / "file-history" / sid / "x@v1").write_text("old")
    return next(s for s in scan_claude() if s.id == sid)


class TestClaudeDelete:
    def test_removes_every_per_session_trace_and_nothing_shared(self, home):
        session = _claude_session(home)
        other = _claude_session(home, "keep-me")
        history = home / ".claude" / "history.jsonl"
        history.write_text(json.dumps({"sessionId": session.id}) + "\n")

        plan = deletion.plan_delete(session, scan_claude())
        assert plan.subagents == 1
        # Transcript last, so a failure part-way leaves the session listed.
        assert plan.paths[-1] == Path(session.content_path)
        deletion.delete_session(plan)

        claude = home / ".claude"
        project = Path(session.content_path).parent
        assert not (project / f"{session.id}.jsonl").exists()
        assert not (project / session.id).exists()
        assert not (claude / "session-env" / session.id).exists()
        assert not (claude / "file-history" / session.id).exists()
        # Shared state another running claude appends to is not ours to edit.
        assert history.read_text().strip()
        assert Path(other.content_path).exists()
        assert (project / other.id / "subagents").is_dir()
        assert [s.id for s in scan_claude() if not s.subagent_kind] == ["keep-me"]

    def test_a_session_with_no_auxiliary_state_is_just_its_transcript(self, home):
        project = home / ".claude" / "projects" / "-Users-test"
        project.mkdir(parents=True)
        (project / "bare.jsonl").write_text(
            json.dumps({"type": "user", "message": {"content": "hi"}}) + "\n"
        )
        session = next(iter(scan_claude()))

        plan = deletion.plan_delete(session, [session])

        assert plan.paths == (project / "bare.jsonl",)
        assert plan.subagents == 0

    def test_a_subagent_is_refused(self, home):
        _claude_session(home)
        child = next(s for s in scan_claude() if s.subagent_kind)
        with pytest.raises(DeleteRefused, match="subagent"):
            deletion.plan_delete(child, [])

    def test_a_running_session_is_refused(self, home):
        session = _claude_session(home)
        markers = home / ".claude" / "sessions"
        markers.mkdir()
        (markers / f"{os.getpid()}.json").write_text(
            json.dumps({"pid": os.getpid(), "sessionId": session.id})
        )
        with pytest.raises(DeleteRefused, match="still running"):
            deletion.plan_delete(session, [])

    def test_a_marker_left_by_a_dead_process_does_not_block(self, home):
        session = _claude_session(home)
        dead = subprocess.Popen(["true"])
        dead.wait()
        markers = home / ".claude" / "sessions"
        markers.mkdir()
        (markers / f"{dead.pid}.json").write_text(
            json.dumps({"pid": dead.pid, "sessionId": session.id})
        )
        assert deletion.plan_delete(session, []).paths

    def test_a_transcript_outside_the_projects_tree_is_refused(self, home):
        stray = Session(
            id="abc", provider="claude", content_path=str(home / "elsewhere/abc.jsonl")
        )
        with pytest.raises(DeleteRefused, match="Unexpected transcript location"):
            deletion.plan_delete(stray, [])


@pytest.fixture
def fake_codex(tmp_path, monkeypatch):
    """A ``codex`` on PATH that records its argv and does what the real one
    was measured to do to the rollout file, steered by environment."""
    bindir = tmp_path / "bin"
    bindir.mkdir()
    log = tmp_path / "codex-argv"
    script = bindir / "codex"
    script.write_text(
        "#!/bin/sh\n"
        f'echo "$@" >> "{log}"\n'
        '[ -n "$FAKE_CODEX_FAIL" ] && { echo "no such thread" >&2; exit 1; }\n'
        '[ -z "$FAKE_CODEX_KEEP" ] && /bin/rm -f "$FAKE_CODEX_ROLLOUT"\n'
        'echo "Deleted session $3."\n'
    )
    script.chmod(0o755)
    monkeypatch.setenv("PATH", str(bindir))
    rollout = tmp_path / "rollout.jsonl"
    rollout.write_text("{}\n")
    monkeypatch.setenv("FAKE_CODEX_ROLLOUT", str(rollout))
    session = Session(id=CODEX_ID, provider="codex", content_path=str(rollout))
    return session, log


class TestCodexDelete:
    def test_hands_the_whole_job_to_codex_delete(self, fake_codex):
        session, log = fake_codex
        plan = deletion.plan_delete(session, [session])

        assert plan.paths == ()
        message = deletion.delete_session(plan)

        assert log.read_text().split() == ["delete", "--force", CODEX_ID]
        assert not Path(session.content_path).exists()
        assert CODEX_ID in message

    def test_a_failed_codex_delete_says_why(self, fake_codex, monkeypatch):
        session, _log = fake_codex
        monkeypatch.setenv("FAKE_CODEX_FAIL", "1")
        with pytest.raises(DeleteFailed, match="exited 1: no such thread"):
            deletion.delete_session(deletion.plan_delete(session, []))

    def test_success_with_the_rollout_still_on_disk_is_a_failure(
        self, fake_codex, monkeypatch
    ):
        session, _log = fake_codex
        monkeypatch.setenv("FAKE_CODEX_KEEP", "1")
        with pytest.raises(DeleteFailed, match="remains"):
            deletion.delete_session(deletion.plan_delete(session, []))

    def test_a_compressed_rollout_takes_its_decompressed_copy_with_it(
        self, fake_codex, tmp_path, monkeypatch
    ):
        session, _log = fake_codex
        rollout = tmp_path / "rollout.jsonl.zst"
        rollout.write_bytes(compressed.zstd.compress(b"{}\n"))
        monkeypatch.setenv("FAKE_CODEX_ROLLOUT", str(rollout))
        session.content_path = str(rollout)
        copy = compressed.readable(rollout)

        deletion.delete_session(deletion.plan_delete(session, []))

        assert not rollout.exists()
        assert not copy.exists()

    def test_no_codex_on_path_is_refused(self, tmp_path, monkeypatch):
        monkeypatch.setenv("PATH", str(tmp_path))
        session = Session(id=CODEX_ID, provider="codex")
        with pytest.raises(DeleteRefused, match="codex CLI"):
            deletion.plan_delete(session, [])

    def test_an_id_that_is_not_a_uuid_is_never_passed_as_a_name(self, fake_codex):
        _session, log = fake_codex
        with pytest.raises(DeleteRefused, match="Not a Codex thread id"):
            deletion.plan_delete(Session(id="my-thread", provider="codex"), [])
        assert not log.exists()

    def test_counts_spawned_subagents_at_every_depth(self, fake_codex):
        session, _log = fake_codex
        child = Session(id="c1", provider="codex", parent_id=CODEX_ID)
        grandchild = Session(id="c2", provider="codex", parent_id="c1")
        unrelated = Session(id="c3", provider="codex", parent_id="other")
        plan = deletion.plan_delete(session, [session, child, grandchild, unrelated])
        assert plan.subagents == 2

    def test_a_spawned_subagent_is_refused(self, fake_codex):
        child = Session(id=CODEX_ID, provider="codex", parent_id="p")
        with pytest.raises(DeleteRefused, match="subagent"):
            deletion.plan_delete(child, [])


@pytest.mark.parametrize("provider", ["opencode", "pi"])
def test_unproven_providers_are_refused(provider):
    with pytest.raises(DeleteRefused, match="not supported yet"):
        deletion.plan_delete(Session(id="x", provider=provider), [])


# ---------------------------------------------------------------------------
# TUI
# ---------------------------------------------------------------------------

try:
    from session_browser.app import ConfirmDelete, SessionBrowser

    _HAS_TEXTUAL = True
except ImportError:
    _HAS_TEXTUAL = False


@pytest.mark.skipif(not _HAS_TEXTUAL, reason="textual not installed")
@pytest.mark.asyncio
class TestDeleteKey:
    async def _open_confirm(self, app, pilot):
        await app.workers.wait_for_complete()
        await pilot.pause()
        app.query_one("#session-table").focus()
        await pilot.pause()
        await pilot.press("D")
        await pilot.pause()
        assert isinstance(app.screen, ConfirmDelete)

    async def test_any_key_but_y_cancels_and_deletes_nothing(self, home):
        session = _claude_session(home)
        app = SessionBrowser()
        async with app.run_test() as pilot:
            await self._open_confirm(app, pilot)
            await pilot.press("enter")
            await pilot.pause()
            assert not isinstance(app.screen, ConfirmDelete)
        assert Path(session.content_path).exists()

    async def test_y_deletes_and_the_list_no_longer_has_it(self, home):
        session = _claude_session(home)
        app = SessionBrowser()
        async with app.run_test() as pilot:
            await self._open_confirm(app, pilot)
            await pilot.press("y")
            await app.workers.wait_for_complete()
            await pilot.pause()
            await app.workers.wait_for_complete()
            await pilot.pause()
            assert not Path(session.content_path).exists()
            assert app._discovered == []
            # The confirmation outlives the reload the delete triggers.
            assert "Deleted claude session abc-123" in str(app._status.render())

    async def test_a_refusal_is_reported_without_a_prompt(self, home):
        _claude_session(home)
        markers = home / ".claude" / "sessions"
        markers.mkdir()
        (markers / "1.json").write_text(
            json.dumps({"pid": os.getpid(), "sessionId": "abc-123"})
        )
        app = SessionBrowser()
        async with app.run_test() as pilot:
            await app.workers.wait_for_complete()
            await pilot.pause()
            app.query_one("#session-table").focus()
            await pilot.press("D")
            await pilot.pause()
            assert not isinstance(app.screen, ConfirmDelete)
            assert "still running" in str(app._status.render())
