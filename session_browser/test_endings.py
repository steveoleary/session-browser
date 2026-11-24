"""How a session ended, read off the end of its transcript.

Every record shape here is one measured on the author's machine (2026-11-24):
Claude Code 2.1.331, codex-cli 0.172.0, Pi, OpenCode 2.x.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from unittest.mock import patch

import pytest

from session_browser import compressed, discovery
from session_browser.cli import run_cli
from session_browser.discovery import (
    Session,
    _last_activity_iso,
    fill_endings,
    scan_claude,
    scan_codex,
    scan_opencode,
    scan_pi,
)
from session_browser.test_transcript import make_opencode_v2_db
from session_browser.transcript import load_transcript, render_markdown


def _write(path: Path, records: list[dict]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r) + "\n" for r in records))
    return path


def _ending(path: Path, provider: str) -> str | None:
    seen: dict = {}
    _last_activity_iso(path, provider, seen=seen)
    return seen.get("ending")


# -- Claude Code ------------------------------------------------------------

TS = "2026-11-20T10:00:00.000Z"


def _user(content, **extra) -> dict:
    return {
        "type": "user",
        "message": {"role": "user", "content": content},
        "timestamp": TS,
        **extra,
    }


def _assistant(*blocks: dict) -> dict:
    return {"type": "assistant", "message": {"content": list(blocks)}, "timestamp": TS}


TEXT = {"type": "text", "text": "Done: the plan is written."}
THINKING = {"type": "thinking", "thinking": "hmm"}
TOOL_USE = {
    "type": "tool_use",
    "id": "toolu_1",
    "name": "Bash",
    "input": {"command": "ls"},
}
TOOL_RESULT = {"type": "tool_result", "tool_use_id": "toolu_1", "content": "a b"}
INTERRUPT = [{"type": "text", "text": "[Request interrupted by user]"}]
INTERRUPT_TOOL = [
    {"type": "text", "text": "[Request interrupted by user for tool use]"}
]
CAVEAT = _user("<local-command-caveat>Caveat: the messages below...", isMeta=True)
CLEAR = _user(
    "<command-name>/clear</command-name>\n            "
    "<command-message>clear</command-message>\n"
    "            <command-args></command-args>"
)
MODEL_OUT = _user("<local-command-stdout>Set model to Opus</local-command-stdout>")
BOOKKEEPING = [
    {"type": "system", "subtype": "turn_duration", "durationMs": 9, "timestamp": TS},
    {"type": "last-prompt", "lastPrompt": "x", "sessionId": "s1"},
    {"type": "ai-title", "aiTitle": "A title", "sessionId": "s1"},
]


@pytest.mark.parametrize(
    ("tail", "expected"),
    [
        pytest.param([_assistant(TEXT)], "completed", id="reply"),
        pytest.param(
            [_assistant(TEXT), _assistant(THINKING)],
            "completed",
            id="reply-then-thinking-block",
        ),
        pytest.param([_assistant(TEXT, TOOL_USE)], "mid-action", id="tool-call"),
        pytest.param(
            [_assistant(TOOL_USE), _user([TOOL_RESULT])],
            "mid-action",
            id="tool-result-never-answered",
        ),
        pytest.param(
            [_assistant(TOOL_USE), _user(INTERRUPT_TOOL)],
            "interrupted",
            id="cancelled-during-tool",
        ),
        pytest.param([_user(INTERRUPT)], "interrupted", id="cancelled"),
        pytest.param(
            [_assistant(TEXT), _user("and now the tests?")],
            "unanswered",
            id="prompt-with-no-reply",
        ),
        pytest.param(
            [
                _assistant(TEXT),
                _user(
                    "<command-name>/goal</command-name>\n<command-message>goal"
                    "</command-message>\n<command-args>fix the login bug"
                    "</command-args>"
                ),
            ],
            "unanswered",
            id="command-with-arguments-is-a-request",
        ),
        pytest.param(
            [_assistant(TEXT), CAVEAT, CLEAR],
            "completed",
            id="bare-command-is-not-a-turn",
        ),
        pytest.param(
            [
                _assistant(TEXT),
                CAVEAT,
                _user("<command-name>/model</command-name>"),
                MODEL_OUT,
            ],
            "completed",
            id="command-output-is-not-a-turn",
        ),
        pytest.param(
            [_assistant(TEXT), _user("<task-notification>\n<task-id>b1</task-id>")],
            "completed",
            id="task-notification-is-machine-traffic",
        ),
        pytest.param(
            [_assistant(TEXT), *BOOKKEEPING],
            "completed",
            id="bookkeeping-after-the-turn",
        ),
        pytest.param(
            [
                _assistant(TOOL_USE),
                _user(
                    [
                        {
                            **TOOL_RESULT,
                            "is_error": True,
                            "content": "The user doesn't want to proceed with this "
                            "tool use. The tool use was rejected.",
                        }
                    ]
                ),
            ],
            "interrupted",
            id="tool-rejected",
        ),
        pytest.param(
            [
                _assistant(TEXT),
                {
                    "type": "attachment",
                    "timestamp": TS,
                    "attachment": {
                        "type": "queued_command",
                        "prompt": "next?",
                        "origin": {"kind": "human"},
                    },
                },
            ],
            "unanswered",
            id="queued-prompt",
        ),
    ],
)
def test_claude_endings(tmp_path, tail, expected):
    path = _write(tmp_path / "s1.jsonl", [_user("plan the work"), *tail])
    assert _ending(path, "claude") == expected


def test_claude_session_with_only_commands_has_no_ending(tmp_path):
    # Opened, then /clear: nothing was asked or answered.
    path = _write(tmp_path / "s1.jsonl", [CAVEAT, CLEAR])
    assert _ending(path, "claude") is None


def test_claude_ending_comes_through_discovery_and_its_memo(tmp_path):
    project = tmp_path / ".claude" / "projects" / "-Users-test-p"
    _write(project / "done.jsonl", [_user("plan it", cwd="/p"), _assistant(TEXT)])
    _write(project / "cut.jsonl", [_user("plan it", cwd="/p"), _user(INTERRUPT)])
    with patch("session_browser.discovery.Path.home", return_value=tmp_path):
        first = {s.id: s.ending for s in scan_claude()}
        again = {s.id: s.ending for s in scan_claude()}
    assert first == again == {"done": "completed", "cut": "interrupted"}


# -- Pi ---------------------------------------------------------------------


def _pi(role: str, content, **extra) -> dict:
    return {
        "type": "message",
        "id": "m",
        "parentId": None,
        "timestamp": TS,
        "message": {"role": role, "content": content, **extra},
    }


PI_TEXT = [{"type": "text", "text": "done"}]
PI_CALL = [{"type": "toolCall", "id": "c1", "name": "bash", "arguments": {}}]


@pytest.mark.parametrize(
    ("tail", "expected"),
    [
        pytest.param(
            [_pi("assistant", PI_TEXT, stopReason="stop")], "completed", id="stop"
        ),
        pytest.param(
            [_pi("assistant", PI_CALL, stopReason="toolUse")],
            "mid-action",
            id="tool-call",
        ),
        pytest.param(
            [
                _pi("assistant", PI_CALL, stopReason="toolUse"),
                _pi(
                    "toolResult",
                    PI_TEXT,
                    toolCallId="c1",
                    toolName="bash",
                    isError=False,
                ),
            ],
            "mid-action",
            id="tool-result",
        ),
        pytest.param(
            [
                _pi(
                    "assistant",
                    [],
                    stopReason="aborted",
                    errorMessage="Operation aborted",
                )
            ],
            "interrupted",
            id="aborted",
        ),
        pytest.param(
            [
                _pi(
                    "assistant",
                    [],
                    stopReason="error",
                    errorMessage="This operation was aborted",
                )
            ],
            "interrupted",
            id="abort-reported-as-error",
        ),
        pytest.param(
            [
                _pi(
                    "assistant",
                    [],
                    stopReason="error",
                    errorMessage="Request timed out.",
                )
            ],
            "unanswered",
            id="failed",
        ),
        pytest.param(
            [
                _pi("assistant", PI_TEXT, stopReason="stop"),
                _pi("user", [{"type": "text", "text": "more?"}]),
            ],
            "unanswered",
            id="prompt-with-no-reply",
        ),
        pytest.param(
            [
                _pi("assistant", PI_TEXT, stopReason="stop"),
                {"type": "compaction", "timestamp": TS, "summary": "s"},
            ],
            "completed",
            id="compaction-after",
        ),
    ],
)
def test_pi_endings(tmp_path, tail, expected):
    head = [
        {"type": "session", "id": "pi-1", "cwd": "/p", "timestamp": TS},
        _pi("user", [{"type": "text", "text": "hello"}]),
    ]
    path = _write(tmp_path / "x_pi-1.jsonl", [*head, *tail])
    assert _ending(path, "pi") == expected


def test_pi_ending_comes_through_discovery(tmp_path):
    path = tmp_path / ".pi" / "agent" / "sessions" / "--p--" / "t_pi-1.jsonl"
    _write(
        path,
        [
            {"type": "session", "id": "pi-1", "cwd": "/p", "timestamp": TS},
            _pi("user", [{"type": "text", "text": "hello"}]),
            _pi("assistant", [], stopReason="aborted"),
        ],
    )
    with patch("session_browser.discovery.Path.home", return_value=tmp_path):
        assert [s.ending for s in scan_pi()] == ["interrupted"]


# -- Codex ------------------------------------------------------------------


def _ev(ptype: str, **payload) -> dict:
    return {"timestamp": TS, "type": "event_msg", "payload": {"type": ptype, **payload}}


def _ri(ptype: str, **payload) -> dict:
    return {
        "timestamp": TS,
        "type": "response_item",
        "payload": {"type": ptype, **payload},
    }


def _msg(role: str, text: str) -> dict:
    kind = "output_text" if role == "assistant" else "input_text"
    return _ri("message", role=role, content=[{"type": kind, "text": text}])


CODEX_HEAD = [
    {
        "timestamp": TS,
        "type": "session_meta",
        "payload": {"id": "cx-1", "cwd": "/p", "timestamp": TS},
    },
    _msg("user", "# AGENTS.md instructions for /p\n..."),
    {"timestamp": TS, "type": "turn_context", "payload": {"model": "gpt-6.1-sol"}},
    _ev("task_started", turn_id="t1"),
    _msg("user", "fix the build"),
    _ev(
        "item_completed",
        item={"type": "UserMessage", "content": [{"type": "text", "text": "fix"}]},
    ),
]
CALL = _ri("custom_tool_call", call_id="c1", name="exec", input="ls")
OUTPUT = _ri("custom_tool_call_output", call_id="c1", output="a b")
REPLY = [
    _ev("item_completed", item={"type": "AgentMessage"}),
    _msg("assistant", "Fixed."),
]
COMPLETE = _ev("task_complete", turn_id="t1", last_agent_message="Fixed.")


@pytest.mark.parametrize(
    ("tail", "expected"),
    [
        pytest.param(
            [CALL, OUTPUT, *REPLY, {"type": "token_usage_record"}, COMPLETE],
            "completed",
            id="turn-complete",
        ),
        pytest.param(
            [
                CALL,
                OUTPUT,
                *REPLY,
                COMPLETE,
                _ev("item_completed", item={"type": "CommandExecution"}),
            ],
            "completed",
            id="shell-command-after-the-turn",
        ),
        pytest.param(
            [CALL, OUTPUT, _ev("task_complete", turn_id="t1", last_agent_message=None)],
            "completed",
            id="turn-closed-after-a-tool-with-no-reply",
        ),
        pytest.param(
            [_ev("task_complete", turn_id="t1", last_agent_message=None)],
            "unanswered",
            id="turn-closed-with-no-reply",
        ),
        pytest.param(
            [
                CALL,
                OUTPUT,
                _msg("developer", "<turn_aborted>"),
                _ev("turn_aborted", reason="interrupted"),
            ],
            "interrupted",
            id="aborted",
        ),
        pytest.param(
            [_ri("reasoning", summary=[]), CALL], "mid-action", id="call-with-no-output"
        ),
        pytest.param(
            [CALL, OUTPUT, {"type": "token_usage_record"}],
            "mid-action",
            id="output-never-answered",
        ),
        pytest.param(
            [*REPLY, COMPLETE, _msg("user", "and the docs?")],
            "unanswered",
            id="next-prompt-not-started",
        ),
        pytest.param([*REPLY], "completed", id="reply-with-no-close"),
    ],
)
def test_codex_endings(tmp_path, tail, expected):
    path = _write(tmp_path / "rollout.jsonl", [*CODEX_HEAD, *tail])
    assert _ending(path, "codex") == expected


def test_codex_only_injected_context_has_no_ending(tmp_path):
    path = _write(tmp_path / "rollout.jsonl", CODEX_HEAD[:3])
    assert _ending(path, "codex") is None


def _codex_home(tmp_path: Path, *, compress: bool) -> Path:
    """A Codex home with an index row for one rollout ending interrupted."""
    day = tmp_path / ".codex" / "sessions" / "2026" / "11" / "20"
    name = "rollout-2026-11-20T10-00-00-0199a213-81c0-7800-8aa1-bbab2a035a53.jsonl"
    path = _write(
        day / name,
        [*CODEX_HEAD, CALL, OUTPUT, _ev("turn_aborted", reason="interrupted")],
    )
    if compress:
        packed = path.with_name(name + compressed.SUFFIX)
        packed.write_bytes(compressed.zstd.compress(path.read_bytes()))
        path.unlink()
    conn = sqlite3.connect(str(tmp_path / ".codex" / "state_5.sqlite"))
    conn.execute(
        "CREATE TABLE threads (id TEXT PRIMARY KEY, rollout_path TEXT NOT NULL, "
        "cwd TEXT NOT NULL, git_branch TEXT, git_origin_url TEXT, source TEXT, "
        "first_user_message TEXT NOT NULL DEFAULT '', created_at_ms INTEGER, "
        "updated_at_ms INTEGER, archived INTEGER NOT NULL DEFAULT 0)"
    )
    conn.execute(
        "INSERT INTO threads (id, rollout_path, cwd, first_user_message, "
        "created_at_ms, updated_at_ms) VALUES (?, ?, '/p', 'fix the build', 1, 2)",
        ("0199a213-81c0-7800-8aa1-bbab2a035a53", str(day / name)),
    )
    conn.commit()
    conn.close()
    return tmp_path


@pytest.mark.parametrize("compress", [False, True], ids=["plain", "zst"])
def test_codex_index_sessions_get_their_ending_when_asked(tmp_path, compress):
    home = _codex_home(tmp_path, compress=compress)
    with patch("session_browser.discovery.Path.home", return_value=home):
        sessions = scan_codex()
    # The index path reads no rollout, so discovery cannot say yet.
    assert [s.ending for s in sessions] == [None]
    fill_endings(sessions)
    assert [s.ending for s in sessions] == ["interrupted"]


def test_codex_endings_are_remembered_while_the_rollout_is_unchanged(tmp_path):
    home = _codex_home(tmp_path, compress=True)
    with patch("session_browser.discovery.Path.home", return_value=home):
        fill_endings(scan_codex())
        sessions = scan_codex()
    with patch.object(discovery, "_codex_rollout_ending") as read:
        fill_endings(sessions)
    read.assert_not_called()
    assert [s.ending for s in sessions] == ["interrupted"]


def test_codex_memo_keeps_sessions_a_later_call_did_not_ask_about(tmp_path):
    a = _write(tmp_path / "a.jsonl", [*CODEX_HEAD, *REPLY, COMPLETE])
    b = _write(tmp_path / "b.jsonl", [*CODEX_HEAD, CALL])
    sa = Session(id="a", provider="codex", content_path=str(a))
    sb = Session(id="b", provider="codex", content_path=str(b))
    fill_endings([sa, sb])
    fill_endings([Session(id="a", provider="codex", content_path=str(a))])
    again = Session(id="b", provider="codex", content_path=str(b))
    with patch.object(discovery, "_codex_rollout_ending") as read:
        fill_endings([again])
    read.assert_not_called()
    assert again.ending == "mid-action"


def test_an_unreadable_rollout_costs_only_its_own_ending(tmp_path):
    good = _write(tmp_path / "a.jsonl", [*CODEX_HEAD, *REPLY, COMPLETE])
    bad = tmp_path / "b.jsonl.zst"
    bad.write_bytes(b"not zstd at all")
    sessions = [
        Session(id="b", provider="codex", content_path=str(bad)),
        Session(id="a", provider="codex", content_path=str(good)),
    ]
    fill_endings(sessions)
    assert [s.ending for s in sessions] == [None, "completed"]


# -- OpenCode ---------------------------------------------------------------


def test_opencode_2_records_its_own_outcome(tmp_path):
    db = make_opencode_v2_db(tmp_path, {"ses_a": {}, "ses_b": {}, "ses_c": {}})
    conn = sqlite3.connect(str(db))
    conn.execute("ALTER TABLE session_v2 ADD COLUMN idle_outcome TEXT")
    conn.execute("UPDATE session_v2 SET idle_outcome = 'succeeded' WHERE id='ses_a'")
    conn.execute("UPDATE session_v2 SET idle_outcome = 'interrupted' WHERE id='ses_b'")
    conn.commit()
    conn.close()
    with patch("session_browser.discovery.Path.home", return_value=tmp_path):
        found = {s.id: s.ending for s in scan_opencode()}
    # ses_c is busy, or never went idle: OpenCode has not said.
    assert found == {"ses_a": "completed", "ses_b": "interrupted", "ses_c": None}


def test_opencode_2_without_the_outcome_column_keeps_everything_else(tmp_path):
    db = make_opencode_v2_db(tmp_path, {"ses_a": {}})
    conn = sqlite3.connect(str(db))
    conn.execute("ALTER TABLE session_v2 ADD COLUMN model TEXT")
    conn.execute(
        "UPDATE session_v2 SET model = ?",
        (json.dumps({"id": "m1", "providerID": "local"}),),
    )
    conn.commit()
    conn.close()
    with patch("session_browser.discovery.Path.home", return_value=tmp_path):
        found = [(s.id, s.model, s.ending) for s in scan_opencode()]
    assert found == [("ses_a", "local/m1", None)]


def _opencode_v1(home: Path, messages: dict[str, list[dict]]) -> None:
    """A 1.x database: a message table, each row's data a JSON object."""
    db = home / ".local" / "share" / "opencode" / "opencode.db"
    db.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db))
    conn.execute(
        "CREATE TABLE message (id TEXT PRIMARY KEY, session_id TEXT, "
        "time_created INTEGER, time_updated INTEGER, data TEXT)"
    )
    for sid, rows in messages.items():
        for n, data in enumerate(rows):
            conn.execute(
                "INSERT INTO message VALUES (?, ?, ?, ?, ?)",
                (f"{sid}-{n}", sid, 1000 + n, 1000 + n, json.dumps(data)),
            )
    conn.commit()
    conn.close()


def test_opencode_1_endings_come_from_each_sessions_last_message(tmp_path):
    asked = {"role": "user"}
    _opencode_v1(
        tmp_path,
        {
            "ses_done": [asked, {"role": "assistant", "finish": "stop"}],
            "ses_tools": [asked, {"role": "assistant", "finish": "tool-calls"}],
            "ses_cut": [
                asked,
                {"role": "assistant", "error": {"name": "MessageAbortedError"}},
            ],
            "ses_failed": [
                asked,
                {"role": "assistant", "error": {"name": "APIError"}},
            ],
            "ses_asked": [{"role": "assistant", "finish": "stop"}, asked],
            "ses_streaming": [asked, {"role": "assistant"}],
        },
    )
    sessions = [
        Session(id=sid, provider="opencode")
        for sid in (
            "ses_done",
            "ses_tools",
            "ses_cut",
            "ses_failed",
            "ses_asked",
            "ses_streaming",
            "ses_unknown",
        )
    ]
    with patch("session_browser.discovery.Path.home", return_value=tmp_path):
        filled = fill_endings(sessions)
    assert {s.id: s.ending for s in sessions} == {
        "ses_done": "completed",
        "ses_tools": "mid-action",
        "ses_cut": "interrupted",
        "ses_failed": "unanswered",
        "ses_asked": "unanswered",
        "ses_streaming": "mid-action",
        "ses_unknown": None,
    }
    assert len(filled) == 6


def test_opencode_2_sessions_are_not_asked_again(tmp_path):
    # A 2.x session with no ending is busy; its database holds no message
    # table, and nothing is read for it.
    make_opencode_v2_db(tmp_path, {"ses_a": {}})
    busy = Session(id="ses_a", provider="opencode")
    with patch("session_browser.discovery.Path.home", return_value=tmp_path):
        assert fill_endings([busy]) == []
    assert busy.ending is None


# -- CLI and rendering --------------------------------------------------------


@pytest.fixture
def ended(tmp_path, monkeypatch, capsys):
    """Four Claude sessions, one per ending, plus one that cannot say."""
    tails = {
        "done": [_assistant(TEXT)],
        "asked": [_assistant(TEXT), _user("and the docs?")],
        "busy": [_assistant(TOOL_USE)],
        "cut": [_user(INTERRUPT)],
        "blank": [],
    }
    sessions = []
    for n, (sid, tail) in enumerate(tails.items()):
        path = _write(tmp_path / f"{sid}.jsonl", [_user("plan it"), *tail])
        s = Session(
            id=sid,
            provider="claude",
            summary=sid,
            cwd="/p",
            content_path=str(path),
            updated_at=f"2026-11-2{n}T10:00:00+00:00",
        )
        s.ending = _ending(path, "claude") if tail else None
        sessions.append(s)
    monkeypatch.setattr("session_browser.cli.discover_all", lambda *a, **k: sessions)

    def run(*argv):
        code = run_cli(list(argv))
        out = capsys.readouterr()
        return code, out.out, out.err

    return run, sessions


def test_list_json_carries_the_ending(ended):
    run, _ = ended
    code, out, _ = run("list", "--format", "json")
    assert code == 0
    assert {r["session_id"]: r["ending"] for r in json.loads(out)["sessions"]} == {
        "done": "completed",
        "asked": "unanswered",
        "busy": "mid-action",
        "cut": "interrupted",
        "blank": None,
    }


def test_list_text_carries_the_ending_after_the_summary(ended):
    run, _ = ended
    code, out, _ = run("list", "--format", "text")
    assert code == 0
    # Seventh of eight: live follows it.
    last = {line.split("\t")[0]: line.split("\t")[6] for line in out.splitlines()}
    assert last == {
        "claude:done": "completed",
        "claude:asked": "unanswered",
        "claude:busy": "mid-action",
        "claude:cut": "interrupted",
        "claude:blank": "-",
    }


def test_ending_filter_and_its_unfinished_shorthand(ended):
    run, _ = ended
    _, out, _ = run("list", "--ending", "unfinished", "--format", "json")
    assert sorted(r["session_id"] for r in json.loads(out)["sessions"]) == [
        "asked",
        "busy",
        "cut",
    ]
    _, out, _ = run("list", "--ending", "completed,interrupted", "--format", "json")
    payload = json.loads(out)
    assert sorted(r["session_id"] for r in payload["sessions"]) == ["cut", "done"]


def test_ending_filter_applies_to_search_too(ended):
    run, _ = ended
    _, out, _ = run("search", "plan it", "--ending", "mid-action", "--format", "json")
    assert [r["session_id"] for r in json.loads(out)["results"]] == ["busy"]


def test_an_unknown_ending_is_refused_with_the_choices(ended, capsys):
    run, _ = ended
    with pytest.raises(SystemExit) as exc:
        run("list", "--ending", "finished")
    assert exc.value.code == 2
    assert "completed, unanswered, mid-action, interrupted, unfinished" in (
        capsys.readouterr().err
    )


def test_get_header_says_how_it_ended(ended):
    _, sessions = ended
    assert "- Ended: interrupted" in render_markdown(load_transcript(sessions[3]))
    assert "- Ended: —" in render_markdown(load_transcript(sessions[4]))
