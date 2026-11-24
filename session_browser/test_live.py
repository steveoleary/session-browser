"""Which sessions a running agent process still holds.

The process table is faked everywhere but one smoke test: what is running on
the machine that runs the suite is not the suite's business.
"""

from __future__ import annotations

import json
import os
import sys
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from session_browser import deletion, live, multiplexer, tmux
from session_browser.cli import run_cli
from session_browser.deletion import DeleteRefused
from session_browser.discovery import Session
from session_browser.live import AgentProcess, fill_live
from session_browser.tmux import TmuxError
from session_browser.transcript import Transcript, render_markdown

ROLLOUT = "/h/.codex/sessions/2026/11/20/rollout-2026-11-20T10-00-00-cx-1.jsonl"


def _procs(*procs: tuple[str, tuple[str, ...]]):
    return [AgentProcess(100 + i, agent, argv) for i, (agent, argv) in enumerate(procs)]


def _running(*procs, open_paths=frozenset()):
    """Patch the probe: these agent processes, holding these files open."""
    return (
        patch.object(live, "agent_processes", return_value=_procs(*procs)),
        patch.object(live, "_open_transcripts", return_value=set(open_paths)),
    )


def _fill(sessions, *procs, open_paths=frozenset()):
    a, b = _running(*procs, open_paths=open_paths)
    with a, b:
        fill_live(sessions)
    return [s.live for s in sessions]


class TestProgram:
    @pytest.mark.parametrize(
        ("argv", "agent"),
        [
            (["/opt/homebrew/bin/codex", "resume", "x"], "codex"),
            (["node", "/opt/homebrew/bin/pi", "--session", "x"], "pi"),
            (["/usr/local/bin/bun", "/x/pi"], "pi"),
            (["/Users/u/.local/bin/opencode", "-s", "ses_1"], "opencode"),
            (["/opt/homebrew/bin/grok"], ""),  # so is Grok's
            (["node", "/x/chrome-devtools-mcp/main.js"], ""),
            (["vim", "pi"], ""),
            (["claude", "--resume", "x"], ""),  # Claude's registry answers
            ([], ""),
        ],
    )
    def test_the_program_is_the_first_argument_or_the_interpreted_one(
        self, argv, agent
    ):
        assert live._program(argv) == agent


class TestFillLive:
    def test_no_process_of_an_agent_means_none_of_its_sessions_is_live(self):
        sessions = [
            Session(id="cx-1", provider="codex", content_path=ROLLOUT),
            Session(id="p1", provider="pi", content_path="/p.jsonl"),
            Session(id="ses_1", provider="opencode", content_path="/oc.db"),
        ]
        assert _fill(sessions) == [False, False, False]

    def test_a_session_named_in_a_processes_arguments_is_live(self):
        sessions = [
            Session(id="cx-1", provider="codex", content_path=ROLLOUT),
            Session(id="cx-2", provider="codex", content_path="/r2.jsonl"),
        ]
        found = _fill(sessions, ("codex", ("codex", "resume", "cx-1")))
        # cx-2 might be what another codex runs, or nothing: not said.
        assert found == [True, None]

    def test_a_rollout_a_codex_process_holds_open_is_live(self, tmp_path):
        rollout = tmp_path / "rollout.jsonl"
        rollout.write_text("{}\n")
        sessions = [Session(id="cx-1", provider="codex", content_path=str(rollout))]
        found = _fill(
            sessions,
            ("codex", ("codex",)),
            open_paths={os.path.realpath(rollout)},
        )
        assert found == [True]

    def test_pi_is_named_by_id_or_by_its_transcripts_path(self):
        by_id = Session(id="0199-aa", provider="pi", content_path="/a.jsonl")
        by_path = Session(id="0199-bb", provider="pi", content_path="/b.jsonl")
        other = Session(id="0199-cc", provider="pi", content_path="/c.jsonl")
        found = _fill(
            [by_id, by_path, other],
            ("pi", ("node", "/bin/pi", "--session", "0199-aa")),
            ("pi", ("node", "/bin/pi", "--session", "/b.jsonl")),
        )
        assert found == [True, True, None]

    def test_an_open_file_says_nothing_for_an_agent_that_does_not_hold_one(self):
        # OpenCode writes one shared database: holding it open names no
        # session in particular.
        s = Session(id="ses_1", provider="opencode", content_path="/oc.db")
        assert _fill([s], ("opencode", ("opencode",)), open_paths={"/oc.db"}) == [None]

    def test_an_unreadable_process_table_marks_nothing(self):
        s = Session(id="cx-1", provider="codex", content_path=ROLLOUT)
        with patch.object(live, "agent_processes", return_value=None):
            fill_live([s])
        assert s.live is None

    def test_a_failed_lsof_still_lets_the_arguments_answer(self):
        sessions = [
            Session(id="cx-1", provider="codex", content_path=ROLLOUT),
            Session(id="cx-2", provider="codex", content_path="/r2.jsonl"),
        ]
        with (
            patch.object(
                live,
                "agent_processes",
                return_value=_procs(("codex", ("codex", "resume", "cx-1"))),
            ),
            patch.object(live, "_open_transcripts", return_value=None),
        ):
            fill_live(sessions)
        assert [s.live for s in sessions] == [True, None]

    def test_claude_keeps_discoverys_answer_unless_asked_afresh(self, tmp_path):
        s = Session(id="c1", provider="claude")
        s.live = True
        with patch.object(live, "agent_processes") as probe:
            fill_live([s])
        probe.assert_not_called()  # nothing but Claude: no probe at all
        assert s.live is True
        with patch.object(live, "_claude_live_session_ids", return_value=set()):
            assert live.is_live(s) is False

    def test_lsof_is_run_only_for_processes_that_hold_transcripts(self):
        s = Session(id="p1", provider="pi", content_path="/p.jsonl")
        with (
            patch.object(
                live, "agent_processes", return_value=_procs(("pi", ("node", "pi")))
            ),
            patch.object(live.subprocess, "run") as run,
        ):
            fill_live([s])
        run.assert_not_called()


@pytest.mark.skipif(
    sys.platform != "darwin" and not sys.platform.startswith("linux"),
    reason="the process table is read on macOS and Linux only",
)
def test_the_real_process_table_can_be_read():
    procs = live.agent_processes()
    assert isinstance(procs, list)
    assert all(p.agent in live._AGENTS for p in procs)


# -- Guards -------------------------------------------------------------------


def test_delete_refuses_a_live_codex_session():
    s = Session(
        id="0199a213-81c0-7800-8aa1-bbab2a035a53",
        provider="codex",
        content_path=ROLLOUT,
    )
    a, b = _running(("codex", ("codex", "resume", s.id)))
    with a, b, pytest.raises(DeleteRefused, match="still running"):
        deletion.plan_delete(s, [])


def test_delete_goes_ahead_when_the_probe_cannot_say(monkeypatch):
    s = Session(
        id="0199a213-81c0-7800-8aa1-bbab2a035a53",
        provider="codex",
        content_path=ROLLOUT,
    )
    monkeypatch.setattr(deletion.shutil, "which", lambda name: f"/bin/{name}")
    with patch.object(live, "agent_processes", return_value=None):
        assert deletion.plan_delete(s, []).command


# -- CLI ----------------------------------------------------------------------


@pytest.fixture
def cli(tmp_path, monkeypatch, capsys):
    rollout = tmp_path / "rollout.jsonl"
    rollout.write_text(
        json.dumps(
            {
                "type": "event_msg",
                "payload": {"type": "user_message", "message": "hello"},
            }
        )
        + "\n"
    )
    sessions = [
        Session(
            id="cx-1",
            provider="codex",
            summary="running",
            cwd="/p",
            content_path=str(rollout),
            updated_at="2026-11-20T10:00:00+00:00",
        ),
        Session(
            id="cx-2",
            provider="codex",
            summary="idle",
            cwd="/p",
            content_path=str(rollout),
            updated_at="2026-11-19T10:00:00+00:00",
        ),
    ]
    for s in sessions:
        s.ending = "completed"
    monkeypatch.setattr("session_browser.cli.discover_all", lambda *a, **k: sessions)

    def run(*argv):
        a, b = _running(("codex", ("codex", "resume", "cx-1")))
        with a, b:
            code = run_cli(list(argv))
        out = capsys.readouterr()
        return code, out.out

    return run


def test_list_json_says_which_sessions_are_live(cli):
    code, out = cli("list", "--format", "json")
    assert code == 0
    assert {r["session_id"]: r["live"] for r in json.loads(out)["sessions"]} == {
        "cx-1": True,
        "cx-2": None,
    }


def test_list_text_marks_the_live_session_in_its_last_column(cli):
    code, out = cli("list", "--format", "text")
    assert code == 0
    assert [line.split("\t")[-1] for line in out.splitlines()] == ["live", "-"]


def test_get_header_says_it_is_live():
    s = Session(id="cx-1", provider="codex")
    s.live = True
    assert "- Live: a running process still holds it" in render_markdown(
        Transcript(s, [])
    )
    s.live = None
    assert "- Live:" not in render_markdown(Transcript(s, []))


# -- TUI ----------------------------------------------------------------------

textual = pytest.importorskip("textual")

from session_browser.app import (
    ConfirmResumeLive,
    SessionBrowser,
)


def _handoff_app(selected: Session):
    pushed, updates = [], []
    app = SimpleNamespace(
        _selected=selected,
        _discovered=[selected],
        _status=SimpleNamespace(update=updates.append),
        push_screen=lambda screen, callback=None: pushed.append(screen),
        exit=lambda: None,
        _run_handoff=lambda commands: 0,
    )
    app._resume_session = lambda s, sid: SessionBrowser._resume_session(app, s, sid)
    return app, pushed, updates


def _tmux_target():
    return multiplexer.Target("tmux", "t", tmux, TmuxError, lambda: True)


def test_handing_off_a_live_session_asks_first(monkeypatch):
    s = Session(id="cx-1", provider="codex", cwd="/p", content_path=ROLLOUT)
    app, pushed, _ = _handoff_app(s)
    monkeypatch.setattr(tmux, "already_open", lambda *a, **k: False)
    monkeypatch.setattr(
        tmux, "prepare_session", lambda *a, **k: pytest.fail("asked first")
    )
    a, b = _running(("codex", ("codex", "resume", "cx-1")))
    with a, b:
        SessionBrowser._open_in_multiplexer(app, _tmux_target())
    assert [type(screen) for screen in pushed] == [ConfirmResumeLive]


def test_a_live_session_the_multiplexer_already_hosts_opens_without_asking(
    monkeypatch,
):
    s = Session(id="cx-1", provider="codex", cwd="/p", content_path=ROLLOUT)
    app, pushed, updates = _handoff_app(s)
    plan = SimpleNamespace(
        reused=True, label="p", switch_commands=list, attach_commands=list
    )
    monkeypatch.setattr(tmux, "already_open", lambda *a, **k: True)
    monkeypatch.setattr(tmux, "prepare_session", lambda *a, **k: plan)
    a, b = _running(("codex", ("codex", "resume", "cx-1")))
    with a, b:
        SessionBrowser._open_in_multiplexer(app, _tmux_target())
    assert pushed == []
    assert updates == ["Switched to tmux session 'p' (already open)"]


@pytest.mark.asyncio
async def test_copying_the_resume_command_of_a_live_session_warns(monkeypatch):
    from session_browser import app as app_module

    copied = []

    def copy(text: str) -> bool:
        copied.append(text)
        return True

    monkeypatch.setattr(app_module, "copy_to_clipboard", copy)
    s = Session(id="cx-1", provider="codex", cwd="/p", content_path=ROLLOUT)
    browser = SessionBrowser()
    async with browser.run_test() as pilot:
        await browser.workers.wait_for_complete()
        browser._selected = s
        a, b = _running(("codex", ("codex", "resume", "cx-1")))
        with a, b:
            browser.action_copy_resume()
        await pilot.pause()
        status = str(browser._status.render())
    assert copied == ["cd /p && codex resume cx-1"]
    assert "still running" in status and "open in a codex process" in status


@pytest.mark.asyncio
async def test_a_live_row_is_marked():
    from session_browser.tests import _install_fake_sessions

    browser = SessionBrowser()
    running = Session(
        id="s1", provider="claude", summary="work s1", updated_at="2026-01-02"
    )
    running.live = True
    idle = Session(
        id="s2", provider="claude", summary="work s2", updated_at="2026-01-01"
    )
    async with browser.run_test() as pilot:
        await _install_fake_sessions(browser, pilot, [running, idle])
        table = browser.query_one("#session-table")
        cells = [str(table.get_row_at(r)[-1]) for r in range(table.row_count)]
    assert cells == ["[green]▶[/] work s1", "work s2"]
