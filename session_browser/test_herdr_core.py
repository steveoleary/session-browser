"""The Herdr app / herdr-core handoff, and choosing between the two herdrs."""

from __future__ import annotations

import json
import subprocess
from unittest.mock import patch

import pytest

from session_browser import herdr, herdr_core, multiplexer
from session_browser.herdr_core import (
    AppSpace,
    CoreTerminal,
    HerdrCoreError,
    HerdrCorePlan,
)

SPACE = "sp1"
TAB = "tb1"


def _terminal(
    address="t1",
    argv=("/bin/zsh", "-l"),
    cwd="/home/user/proj",
    pid=None,
    space=SPACE,
    tab=TAB,
):
    return CoreTerminal(address, tuple(argv), cwd, pid, space, tab)


def _get_body(terminal: CoreTerminal) -> dict:
    """``hc get --json`` for one terminal, cut to the fields read."""
    labels = {"where": "x / 1"}
    if terminal.space:
        labels["herdr.space"] = terminal.space
        labels["herdr.tab"] = terminal.tab
    return {
        "address": terminal.address,
        "argv": ["/bin/zsh", "-l"],
        "cwd": "/launch/folder",
        "foreground": {
            "argv": list(terminal.argv),
            "cwd": terminal.cwd,
            "pid": terminal.pid,
        },
        "labels": labels,
        "state": "live",
    }


def _stub(terminals=(), spaces=(), created=None, fail=None):
    """Fake ``hc`` and ``herdr-app``. Returns the calls made and the runner.

    ``fail`` names a command prefix that should fail the way that CLI does:
    ``herdr-app do`` prints ``{"error": ...}`` on stdout and exits 1.
    """
    calls = []
    by_address = {t.address: t for t in terminals}

    def fake_run(cmd, *a, **kw):
        calls.append(cmd)
        if fail and cmd[: len(fail)] == fail:
            out = json.dumps({"error": "no such space"})
            return subprocess.CompletedProcess(cmd, 1, stdout=out, stderr="")
        if cmd[:2] == ["hc", "ls"]:
            body = {
                "hidden_finished": 0,
                "terminals": [{"address": t.address} for t in terminals]
                + [{"address": "box/t9"}],
            }
        elif cmd[:2] == ["hc", "get"]:
            body = _get_body(by_address[cmd[2]])
        elif cmd[:2] == ["herdr-app", "ls"]:
            body = {
                "spaces": [
                    {"id": s.space_id, "name": s.name, "tabs": []} for s in spaces
                ]
                + [{"id": "remote", "name": "vps", "machine": "vps", "tabs": []}]
            }
        elif cmd[:3] == ["herdr-app", "do", "tab.new"]:
            body = created or {"terminal": "t8", "tab": "tb8"}
        elif cmd[:3] == ["herdr-app", "do", "space.new"]:
            body = created or {"terminal": "t9", "tab": "tb9", "space": "sp9"}
        else:
            raise AssertionError(f"unexpected command {cmd}")
        return subprocess.CompletedProcess(cmd, 0, stdout=json.dumps(body), stderr="")

    return calls, fake_run


def _prepare(*args, calls_and_run, **kw):
    _calls, fake_run = calls_and_run
    with (
        patch.object(herdr_core.shutil, "which", return_value="/bin/x"),
        patch.object(herdr_core.subprocess, "run", side_effect=fake_run),
    ):
        return herdr_core.prepare_session(*args, **kw)


class TestFindTerminal:
    def test_matches_a_resume_command_in_argv(self):
        found = _terminal("t2", ("claude", "--resume", "abc"))
        terminals = [_terminal("t1"), found]
        assert herdr_core.find_terminal(terminals, "claude", {"abc"}) == found

    def test_an_id_in_another_programs_argv_is_not_the_conversation(self):
        terminals = [_terminal(argv=("session-browser", "get", "abc"))]
        assert herdr_core.find_terminal(terminals, "claude", {"abc"}) is None

    def test_the_agent_may_be_run_through_an_interpreter(self):
        found = _terminal(argv=("node", "/opt/bin/codex", "resume", "abc"))
        assert herdr_core.find_terminal([found], "codex", {"abc"}) == found

    def test_a_different_providers_terminal_never_matches(self):
        terminals = [_terminal(argv=("codex", "resume", "abc"))]
        assert herdr_core.find_terminal(terminals, "claude", {"abc"}) is None

    def test_a_terminal_the_app_does_not_show_is_skipped(self):
        terminals = [_terminal(argv=("claude", "--resume", "abc"), space="", tab="")]
        assert herdr_core.find_terminal(terminals, "claude", {"abc"}) is None

    def test_an_empty_id_matches_nothing(self):
        terminals = [_terminal(argv=("claude", ""))]
        assert herdr_core.find_terminal(terminals, "claude", {""}) is None

    def test_a_bare_claude_is_found_through_its_live_session_record(self, tmp_path):
        sessions = tmp_path / ".claude" / "sessions"
        sessions.mkdir(parents=True)
        (sessions / "4242.json").write_text(
            json.dumps({"pid": 4242, "sessionId": "forked-id"})
        )
        found = _terminal(argv=("claude",), pid=4242)
        assert (
            herdr_core.find_terminal([found], "claude", {"forked-id"}, home=tmp_path)
            == found
        )
        # The record is Claude's own; another provider's pid is never read as one.
        assert (
            herdr_core.find_terminal([found], "codex", {"forked-id"}, home=tmp_path)
            is None
        )

    def test_a_missing_or_unreadable_record_is_no_match(self, tmp_path):
        sessions = tmp_path / ".claude" / "sessions"
        sessions.mkdir(parents=True)
        (sessions / "7.json").write_text("{not json")
        assert herdr_core.claude_live_session(7, tmp_path) == ""
        assert herdr_core.claude_live_session(8, tmp_path) == ""
        assert herdr_core.claude_live_session(None, tmp_path) == ""


class TestSpaceForCwd:
    def test_the_space_of_a_terminal_in_that_folder(self):
        spaces = [AppSpace("sp1", "one"), AppSpace("sp2", "two")]
        terminals = [
            _terminal("t1", cwd="/elsewhere", space="sp1"),
            _terminal("t2", cwd="/home/user/proj", space="sp2"),
        ]
        found = herdr_core.space_for_cwd(terminals, spaces, "/home/user/proj/")
        assert found == spaces[1]

    def test_no_terminal_in_the_folder_means_no_space(self):
        spaces = [AppSpace("sp1", "one")]
        terminals = [_terminal(cwd="/elsewhere", space="sp1")]
        assert herdr_core.space_for_cwd(terminals, spaces, "/home/user/proj") is None

    def test_a_hidden_terminal_does_not_place_a_space(self):
        spaces = [AppSpace("sp1", "one")]
        terminals = [_terminal(space="", tab="")]
        assert herdr_core.space_for_cwd(terminals, spaces, "/home/user/proj") is None


class TestPrepareSession:
    def test_requires_a_cwd(self):
        with pytest.raises(HerdrCoreError, match="no folder"):
            herdr_core.prepare_session("claude", "x", "")

    def test_requires_both_clis(self):
        with (
            patch.object(herdr_core.shutil, "which", return_value=None),
            pytest.raises(HerdrCoreError, match="not both on PATH"),
        ):
            herdr_core.prepare_session("claude", "x", "/home/user/proj")

    def test_unknown_provider(self):
        with pytest.raises(HerdrCoreError, match="unknown provider"):
            _prepare("bogus", "x", "/home/user/proj", calls_and_run=_stub())

    def test_reuses_the_terminal_already_running_it(self):
        running = _terminal("t5", ("claude", "--resume", "abc-123"))
        stub = _stub([running], [AppSpace(SPACE, "proj")])
        plan = _prepare("claude", "abc-123", "/home/user/proj", calls_and_run=stub)
        assert plan.reused
        assert (plan.terminal, plan.space, plan.tab, plan.label) == (
            "t5",
            SPACE,
            TAB,
            "proj",
        )
        assert not [c for c in stub[0] if c[:2] == ["herdr-app", "do"]]
        # Another machine's terminal is never read.
        assert ["hc", "get", "box/t9", "--json"] not in stub[0]

    def test_adds_a_tab_to_the_folders_space(self):
        shell = _terminal("t1", cwd="/home/user/proj")
        stub = _stub([shell], [AppSpace(SPACE, "my proj")])
        plan = _prepare("codex", "ghi-789", "/home/user/proj", calls_and_run=stub)
        assert stub[0][-1] == [
            *("herdr-app", "do", "tab.new", "--space", SPACE),
            *("--folder", "/home/user/proj"),
            *("--argv", json.dumps(["codex", "resume", "ghi-789"])),
        ]
        assert not plan.reused
        assert (plan.terminal, plan.tab, plan.space, plan.label) == (
            "t8",
            "tb8",
            SPACE,
            "my proj",
        )

    def test_creates_a_space_for_a_new_folder(self):
        stub = _stub([_terminal(cwd="/elsewhere")], [AppSpace(SPACE, "x")])
        plan = _prepare("claude", "def-456", "/home/user/proj", calls_and_run=stub)
        assert stub[0][-1] == [
            *("herdr-app", "do", "space.new", "--name", "proj"),
            *("--folder", "/home/user/proj"),
            *(
                "--argv",
                json.dumps(
                    ["claude", "--dangerously-skip-permissions", "--resume", "def-456"]
                ),
            ),
        ]
        assert (plan.terminal, plan.tab, plan.space, plan.label) == (
            "t9",
            "tb9",
            "sp9",
            "proj",
        )

    def test_the_apps_error_is_reported(self):
        stub = _stub(fail=["herdr-app", "do", "space.new"])
        with pytest.raises(HerdrCoreError, match="no such space"):
            _prepare("claude", "x", "/home/user/proj", calls_and_run=stub)

    def test_a_create_without_a_terminal_is_an_error(self):
        stub = _stub(created={"space": "sp9"})
        with pytest.raises(HerdrCoreError, match="new tab's terminal"):
            _prepare("claude", "x", "/home/user/proj", calls_and_run=stub)


class TestPlan:
    def _plan(self, **kw):
        fields = {
            "label": "proj",
            "cwd": "/p",
            "resume": "r",
            "session_id": "s",
            "space": "sp1",
            "tab": "tb1",
            "terminal": "t5",
            "reused": True,
        }
        fields.update(kw)
        return HerdrCorePlan(**fields)

    def test_switch_shows_space_tab_and_terminal(self):
        assert self._plan().switch_commands() == [
            ["herdr-app", "do", "space.select", "--id", "sp1"],
            ["herdr-app", "do", "tab.select", "--id", "tb1"],
            ["herdr-app", "do", "pane.focus", "--terminal", "t5"],
        ]

    def test_unknown_levels_are_left_out(self):
        assert self._plan(space="", tab="").switch_commands() == [
            ["herdr-app", "do", "pane.focus", "--terminal", "t5"]
        ]

    def test_attach_ends_by_attaching_this_terminal(self):
        commands = self._plan().attach_commands()
        assert commands[:-1] == self._plan().switch_commands()
        assert commands[-1] == ["hc", "attach", "t5"]


class TestAvailable:
    def _run(self, returncode=0, stdout="{}"):
        return subprocess.CompletedProcess([], returncode, stdout=stdout, stderr="")

    def test_needs_both_clis(self):
        with patch.object(
            herdr_core.shutil, "which", side_effect=lambda n: n == "hc" or None
        ):
            assert not herdr_core.available()

    @pytest.mark.parametrize(
        ("returncode", "stdout", "expected"),
        [
            (0, json.dumps({"space": "sp1"}), True),
            (1, "", False),
            (0, json.dumps({"error": "no window"}), False),
            (0, "not json", False),
        ],
    )
    def test_reads_the_windows_status(self, returncode, stdout, expected):
        with (
            patch.object(herdr_core.shutil, "which", return_value="/bin/x"),
            patch.object(
                herdr_core.subprocess,
                "run",
                return_value=self._run(returncode, stdout),
            ) as run,
        ):
            assert herdr_core.available() is expected
        assert run.call_args.args[0] == ["herdr-app", "status"]

    def test_inside_is_read_from_the_terminal_address(self, monkeypatch):
        monkeypatch.setenv("HC_TERMINAL_ID", "t2k")
        assert herdr_core.in_herdr_core()
        monkeypatch.delenv("HC_TERMINAL_ID")
        assert not herdr_core.in_herdr_core()


class TestChoosingAHerdr:
    """Both herdrs are one "herdr" choice; the live one is used."""

    @pytest.fixture
    def state(self, monkeypatch):
        state = {
            "tmux": False,
            "core": False,
            "legacy": False,
            "in_core": False,
            "in_legacy": False,
        }
        probed = []

        def probe(name):
            def available():
                probed.append(name)
                return state[name]

            return available

        monkeypatch.setattr(multiplexer.tmux, "available", probe("tmux"))
        monkeypatch.setattr(herdr_core, "available", probe("core"))
        monkeypatch.setattr(herdr, "available", probe("legacy"))
        inside_key = {herdr_core: "in_core", herdr: "in_legacy"}

        def inside(module):
            key = inside_key.get(module, "tmux")
            return lambda: state[key]

        targets = tuple(
            multiplexer.Target(t.name, t.key, t.module, t.error, inside(t.module))
            for t in multiplexer.TARGETS
        )
        monkeypatch.setattr(multiplexer, "TARGETS", targets)
        state["probed"] = probed
        return state

    def _modules(self):
        return [t.module for t in multiplexer.available_targets()]

    def test_the_new_herdr_alone(self, state):
        state["core"] = True
        assert self._modules() == [herdr_core]

    def test_the_old_herdr_alone(self, state):
        state["legacy"] = True
        assert self._modules() == [herdr]

    def test_both_running_prefers_the_new_one(self, state):
        state.update(core=True, legacy=True)
        assert self._modules() == [herdr_core]
        # The old one is not even asked once the new one answered.
        assert "legacy" not in state["probed"]

    def test_both_running_from_inside_the_old_one_uses_the_old_one(self, state):
        state.update(core=True, legacy=True, in_legacy=True)
        assert self._modules() == [herdr]

    def test_inside_the_new_one_it_wins_even_if_inside_the_old_too(self, state):
        state.update(core=True, legacy=True, in_core=True, in_legacy=True)
        assert self._modules() == [herdr_core]

    def test_tmux_and_a_herdr_are_two_choices(self, state):
        state.update(tmux=True, legacy=True)
        assert [t.name for t in multiplexer.available_targets()] == ["tmux", "herdr"]

    def test_names_are_listed_once(self):
        assert multiplexer.target_names() == ["tmux", "herdr"]
