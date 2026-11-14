"""Deleting a session from its provider's store, and the TUI path to it."""

from __future__ import annotations

import json
import os
import sqlite3
import subprocess
import sys
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

    def test_a_rewound_thread_is_every_one_of_its_files(self, fake_codex, tmp_path):
        # codex delete removes all of a rewound thread's rollouts; one left
        # behind would come back as a session on the next discovery.
        session, _log = fake_codex
        earlier = tmp_path / "rollout-earlier.jsonl"
        earlier.write_text("{}\n")
        session.segments = (str(earlier),)
        plan = deletion.plan_delete(session, [])
        assert [r.value for r in plan.removes if r.label == "rollout"] == [
            str(earlier),
            session.content_path,
        ]
        # The fake removes only the newest file, so the earlier one remains.
        with pytest.raises(DeleteFailed, match=r"rollout-earlier\.jsonl remains"):
            deletion.delete_session(plan)

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


OPENCODE_ID = "ses_e6912efd2ffe3UsWZKvr7Xx1Km"


@pytest.fixture
def fake_opencode(tmp_path, monkeypatch):
    """An ``opencode`` on PATH that records its argv and does what the real
    2.x one was measured to do -- remove the session row and its children --
    steered by environment, over a v2 database in a fake home."""
    from session_browser.test_transcript import make_opencode_v2_db

    db = make_opencode_v2_db(
        tmp_path,
        {
            OPENCODE_ID: {"messages": [("user", {"text": "hello"})]},
            "ses_child": {"parent_id": OPENCODE_ID},
            "ses_keep": {},
        },
    )
    bindir = tmp_path / "bin"
    bindir.mkdir()
    log = tmp_path / "opencode-argv"
    script = bindir / "opencode"
    script.write_text(
        f"#!{sys.executable}\n"
        "import os, sqlite3, sys\n"
        f"open({str(log)!r}, 'a').write(' '.join(sys.argv[1:]))\n"
        "if os.environ.get('FAKE_OPENCODE_FAIL'):\n"
        "    print('Session not found: ' + sys.argv[3], file=sys.stderr)\n"
        "    sys.exit(1)\n"
        "if not os.environ.get('FAKE_OPENCODE_KEEP'):\n"
        f"    c = sqlite3.connect({str(db)!r})\n"
        "    c.execute('DELETE FROM session_v2 WHERE id = ? OR parent_id = ?',\n"
        "              (sys.argv[3], sys.argv[3]))\n"
        "    c.commit()\n"
        "print('Session ' + sys.argv[3] + ' deleted')\n"
    )
    script.chmod(0o755)
    monkeypatch.setenv("PATH", str(bindir))
    with patch("session_browser.discovery.Path.home", return_value=tmp_path):
        yield Session(id=OPENCODE_ID, provider="opencode"), log, db


class TestOpencodeDelete:
    def test_hands_the_whole_job_to_opencode_session_delete(self, fake_opencode):
        session, log, db = fake_opencode
        child = Session(id="ses_child", provider="opencode", parent_id=OPENCODE_ID)
        plan = deletion.plan_delete(session, [session, child])

        assert (plan.paths, plan.subagents) == ((), 1)
        assert plan.command[1:] == ("session", "delete", OPENCODE_ID)
        message = deletion.delete_session(plan)

        assert log.read_text() == f"session delete {OPENCODE_ID}"
        assert OPENCODE_ID in message
        left = sqlite3.connect(db).execute("SELECT id FROM session_v2").fetchall()
        assert left == [("ses_keep",)]

    def test_a_failed_delete_says_why(self, fake_opencode, monkeypatch):
        session, _log, _db = fake_opencode
        monkeypatch.setenv("FAKE_OPENCODE_FAIL", "1")
        with pytest.raises(DeleteFailed, match="opencode session exited 1: Session"):
            deletion.delete_session(deletion.plan_delete(session, []))

    def test_success_with_the_session_still_there_is_a_failure(
        self, fake_opencode, monkeypatch
    ):
        session, _log, _db = fake_opencode
        monkeypatch.setenv("FAKE_OPENCODE_KEEP", "1")
        with pytest.raises(DeleteFailed, match="still there"):
            deletion.delete_session(deletion.plan_delete(session, []))

    def test_an_id_that_is_not_a_session_id_never_reaches_argv(self, fake_opencode):
        _session, log, _db = fake_opencode
        for bad in ("--help", "ses_../x", "not-a-session"):
            with pytest.raises(DeleteRefused, match="Not an OpenCode session id"):
                deletion.plan_delete(Session(id=bad, provider="opencode"), [])
        assert not log.exists()

    def test_a_session_opencode_does_not_hold_is_refused(self, fake_opencode):
        with pytest.raises(DeleteRefused, match="no session ses_gone"):
            deletion.plan_delete(Session(id="ses_gone", provider="opencode"), [])

    def test_a_1x_database_is_refused(self, fake_opencode):
        session, log, db = fake_opencode
        db.unlink()
        conn = sqlite3.connect(db)
        conn.execute("CREATE TABLE session (id TEXT PRIMARY KEY)")
        conn.execute("INSERT INTO session VALUES (?)", (OPENCODE_ID,))
        conn.commit()
        conn.close()
        with pytest.raises(DeleteRefused, match=r"predates 2\.x"):
            deletion.plan_delete(session, [])
        assert not log.exists()

    def test_no_opencode_on_path_is_refused(self, tmp_path, monkeypatch):
        monkeypatch.setenv("PATH", str(tmp_path))
        session = Session(id=OPENCODE_ID, provider="opencode")
        with pytest.raises(DeleteRefused, match="opencode CLI"):
            deletion.plan_delete(session, [])


PI_ID = "01a196f8-a776-76f7-9f00-4a7342c8d2cb"


def _pi_session(home: Path, sid: str = PI_ID) -> Session:
    from session_browser.discovery import scan_pi

    project = home / ".pi" / "agent" / "sessions" / "--Users-test-project--"
    project.mkdir(parents=True, exist_ok=True)
    lines = [
        {
            "type": "session",
            "id": sid,
            "cwd": "/Users/test/project",
            "timestamp": "2026-11-01T11:57:20.630Z",
        },
        {"type": "message", "message": {"role": "user", "content": "hello"}},
    ]
    (project / f"2026-11-01T11-57-20-630Z_{sid}.jsonl").write_text(
        "\n".join(json.dumps(x) for x in lines) + "\n"
    )
    return next(s for s in scan_pi() if s.id == sid)


class TestPiDelete:
    @pytest.fixture(autouse=True)
    def _nothing_running(self, monkeypatch):
        monkeypatch.setattr(deletion, "_pi_running_in", lambda cwd: False)

    def test_removes_the_transcript_and_extension_state_named_for_it(self, home):
        session = _pi_session(home)
        other = _pi_session(home, "01a196f8-f2f9-742d-8b88-c8635725ebb4")
        ext = home / ".pi" / "agent" / "pi-blackhole"
        ext.mkdir(parents=True)
        mine = [ext / f"{PI_ID}-pending.json", ext / f"{PI_ID}-pending.stale.json"]
        theirs = ext / f"{other.id}-pending.json"
        shared = ext / "pi-blackhole-config.json"
        for f in (*mine, theirs, shared):
            f.write_text("{}")

        plan = deletion.plan_delete(session, [session, other])

        assert plan.paths[-1] == Path(session.content_path)  # transcript last
        assert sorted(plan.paths[:-1]) == sorted(mine)
        assert [r.label for r in plan.removes] == [
            "transcript",
            "extension",
            "extension",
        ]
        deletion.delete_session(plan)
        assert not Path(session.content_path).exists()
        assert not any(f.exists() for f in mine)
        assert Path(other.content_path).exists()
        assert theirs.exists() and shared.exists()

    def test_a_running_pi_in_its_directory_refuses(self, home, monkeypatch):
        session = _pi_session(home)
        seen = []
        monkeypatch.setattr(
            deletion, "_pi_running_in", lambda cwd: seen.append(cwd) or True
        )
        with pytest.raises(DeleteRefused, match="pi is running"):
            deletion.plan_delete(session, [])
        assert seen == ["/Users/test/project"]

    def test_a_transcript_outside_the_sessions_tree_is_refused(self, home):
        stray = Session(
            id=PI_ID,
            provider="pi",
            content_path=str(home / "elsewhere" / f"x_{PI_ID}.jsonl"),
        )
        with pytest.raises(DeleteRefused, match="Unexpected transcript location"):
            deletion.plan_delete(stray, [])

    def test_an_id_that_is_not_a_uuid_is_refused(self, home):
        """The extension sweep matches names by the id's prefix, so an id
        that could be a prefix of something else never gets that far."""
        with pytest.raises(DeleteRefused, match="Not a Pi session id"):
            deletion.plan_delete(Session(id="01", provider="pi"), [])


def test_pi_running_in_reads_ps_then_lsof(monkeypatch, tmp_path):
    """Only processes whose program is pi are asked about, and a cwd match
    is judged on resolved paths (/tmp vs /private/tmp on macOS)."""
    calls = []

    def fake_run(argv, **_kw):
        calls.append(argv)
        out = (
            "  101 node /opt/homebrew/bin/pi -p hello\n"
            "  102 node /usr/lib/other/server.js\n"
            "  103 /usr/bin/vim pi\n"
            if argv[0] == "ps"
            else f"p101\nn{tmp_path}\n"
        )
        return subprocess.CompletedProcess(argv, 0, out, "")

    monkeypatch.setattr(deletion.subprocess, "run", fake_run)
    assert deletion._pi_running_in(str(tmp_path))
    assert calls[1][-1] == "101"
    assert not deletion._pi_running_in(str(tmp_path / "elsewhere"))


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


@pytest.mark.skipif(not _HAS_TEXTUAL, reason="textual not installed")
def test_the_confirmation_names_each_folder_once_and_never_wraps_a_name(home):
    """A wall of wrapped absolute paths was the complaint: the long part of
    every path is its folder, so each folder is written once and the entries
    under it carry only their names, which then fit on a line."""
    import io

    from rich.console import Console

    session = _claude_session(home, "3f2a8c1e-9b7d-4e2f-a1c3-5d6e7f8a9b0c")
    plan = deletion.plan_delete(session, scan_claude())
    out = io.StringIO()
    # The modal is 100 wide less border and padding; 94 is what is left.
    Console(file=out, width=94).print(ConfirmDelete(plan)._removals())
    lines = out.getvalue().splitlines()

    project = "~/.claude/projects/-Users-test/"
    assert [ln for ln in lines if not ln.startswith(" ")] == [
        project,
        "~/.claude/session-env/",
        "~/.claude/file-history/",
        "and its 1 subagent, which goes with it",
    ]
    entries = [ln.split() for ln in lines if ln.startswith("  ")]
    assert [e[:2] for e in entries] == [
        ["transcript", f"{session.id}.jsonl"],
        ["session", "dir"],
        ["state", session.id],
        ["state", session.id],
    ]
    assert entries[1][2:] == [session.id, "subagents,", "tool", "output"]
