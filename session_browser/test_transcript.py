"""Tests for the shared transcript service."""

from __future__ import annotations

import json
import shutil
import sqlite3
from pathlib import Path
from unittest.mock import patch

import pytest

import session_browser.transcript as transcript_mod
from session_browser.discovery import Session
from session_browser.transcript import (
    ContentSearchCache,
    SessionSearchResult,
    Transcript,
    TranscriptEntry,
    TranscriptUnreadable,
    _opencode_part_entry,
    canonical_id,
    entry_matches_roles,
    find_entry_matches,
    find_text_spans,
    lineage_ids,
    load_session_content,
    load_transcript,
    make_snippet,
    render_markdown,
    render_text,
    search_session,
    search_session_contents,
    search_session_hits,
    search_sessions,
    session_to_dict,
    transcript_to_dict,
)


def write_claude_jsonl(path: Path, lines: list[dict | str]) -> None:
    out = []
    for line in lines:
        out.append(line if isinstance(line, str) else json.dumps(line))
    path.write_text("\n".join(out) + "\n")


def claude_session(path: Path) -> Session:
    return Session(id="cl-1", provider="claude", content_path=str(path))


class TestClaudeParser:
    def test_string_user_and_assistant_messages(self, tmp_path):
        f = tmp_path / "s.jsonl"
        write_claude_jsonl(
            f,
            [
                {
                    "type": "user",
                    "message": {"content": "fix the login page"},
                    "timestamp": "2026-06-01T10:00:00Z",
                },
                {
                    "type": "assistant",
                    "message": {"content": "on it"},
                    "timestamp": "2026-06-01T10:00:05Z",
                },
            ],
        )
        t = load_transcript(claude_session(f))
        assert [(e.role, e.text) for e in t.entries] == [
            ("user", "fix the login page"),
            ("assistant", "on it"),
        ]
        assert t.entries[0].timestamp == "2026-06-01T10:00:00Z"
        assert t.warnings == []

    def test_assistant_list_content_text_and_tool_use(self, tmp_path):
        f = tmp_path / "s.jsonl"
        write_claude_jsonl(
            f,
            [
                {
                    "type": "assistant",
                    "message": {
                        "content": [
                            {"type": "text", "text": "let me check"},
                            {
                                "type": "tool_use",
                                "name": "Bash",
                                "input": {"command": "ls -la /tmp"},
                            },
                        ]
                    },
                },
            ],
        )
        t = load_transcript(claude_session(f))
        assert t.entries[0] == TranscriptEntry("assistant", "let me check")
        tool = t.entries[1]
        assert tool.role == "tool"
        assert tool.metadata == {"kind": "call", "tool": "Bash"}
        assert "ls -la /tmp" in tool.text and tool.text.startswith("Bash(")

    def test_user_tool_result_blocks_become_tool_entries(self, tmp_path):
        """Previously-unrendered tool_result content must now be recoverable."""
        f = tmp_path / "s.jsonl"
        write_claude_jsonl(
            f,
            [
                {
                    "type": "user",
                    "message": {
                        "content": [
                            {"type": "tool_result", "content": "total 0 drwxr-xr-x"},
                            {
                                "type": "tool_result",
                                "content": [
                                    {"type": "text", "text": "part one"},
                                    {"type": "text", "text": "part two"},
                                ],
                            },
                        ]
                    },
                },
            ],
        )
        t = load_transcript(claude_session(f))
        assert [e.role for e in t.entries] == ["tool", "tool"]
        assert t.entries[0].text == "total 0 drwxr-xr-x"
        assert t.entries[0].metadata == {"kind": "output"}
        assert t.entries[1].text == "part one\npart two"

    def test_tool_result_error_flag_captured(self, tmp_path):
        """is_error on a tool_result must survive into entry metadata so
        the error pseudo-role filter can find failed calls."""
        f = tmp_path / "s.jsonl"
        write_claude_jsonl(
            f,
            [
                {
                    "type": "user",
                    "message": {
                        "content": [
                            {
                                "type": "tool_result",
                                "content": "boom",
                                "is_error": True,
                            },
                            {"type": "tool_result", "content": "fine"},
                        ]
                    },
                },
            ],
        )
        t = load_transcript(claude_session(f))
        assert t.entries[0].metadata == {"kind": "output", "is_error": True}
        assert t.entries[1].metadata == {"kind": "output"}

    def test_meta_and_protocol_lines_skipped(self, tmp_path):
        f = tmp_path / "s.jsonl"
        write_claude_jsonl(
            f,
            [
                {"type": "user", "isMeta": True, "message": {"content": "meta noise"}},
                {"type": "summary", "summary": "a summary line"},
                {"type": "user", "message": {"content": "real"}},
            ],
        )
        t = load_transcript(claude_session(f))
        assert [e.text for e in t.entries] == ["real"]

    def test_malformed_line_appends_warning_and_continues(self, tmp_path):
        f = tmp_path / "s.jsonl"
        write_claude_jsonl(
            f,
            [
                {"type": "user", "message": {"content": "before"}},
                "{not json",
                {"type": "assistant", "message": {"content": "after"}},
            ],
        )
        t = load_transcript(claude_session(f))
        assert [e.text for e in t.entries] == ["before", "after"]
        assert len(t.warnings) == 1 and "line 2" in t.warnings[0]

    def test_jsonl_stays_bytes_until_native_json_decode(self, tmp_path):
        """Avoid a redundant TextIOWrapper decode of large JSONL records.

        The file must be opened in binary mode and each line decoded exactly
        once, by the parser itself, so a multi-megabyte record is not scanned
        twice. The seam is the JSON decoder: it must receive one already
        decoded ``str`` per line. (json.loads is no longer on the happy path
        at all -- ``_parse_jsonl`` defers to it only for the BOM and
        non-UTF-8 lines whose semantics it must reproduce exactly, and this
        fixture contains neither.)
        """
        f = tmp_path / "s.jsonl"
        write_claude_jsonl(
            f,
            [
                {"type": "user", "message": {"content": "café"}},
                {"type": "assistant", "message": {"content": "done"}},
            ],
        )
        modes: list[str] = []
        seen: list[type] = []
        real_open = open
        real_raw_decode = transcript_mod._JSON_DECODER.raw_decode

        def recording_open(file, mode="r", *args, **kwargs):
            modes.append(mode)
            return real_open(file, mode, *args, **kwargs)

        def recording_raw_decode(s, idx=0):
            seen.append(type(s))
            return real_raw_decode(s, idx)

        with (
            patch.object(transcript_mod, "open", recording_open, create=True),
            patch.object(
                transcript_mod._JSON_DECODER, "raw_decode", recording_raw_decode
            ),
        ):
            transcript = load_transcript(claude_session(f))
        assert [e.text for e in transcript.entries] == ["café", "done"]
        assert modes and all(m == "rb" for m in modes)
        assert seen == [str, str]

    def test_non_dict_jsonl_value_appends_warning_and_continues(self, tmp_path):
        """A syntactically valid JSONL line that is not a dict must be
        treated as a recoverable warning, not crash with AttributeError."""
        f = tmp_path / "s.jsonl"
        write_claude_jsonl(
            f,
            [
                {"type": "user", "message": {"content": "before"}},
                [1, 2],  # valid JSON but not a dict
                {"type": "assistant", "message": {"content": "after"}},
            ],
        )
        t = load_transcript(claude_session(f))
        assert [e.text for e in t.entries] == ["before", "after"]
        assert len(t.warnings) == 1 and "line 2" in t.warnings[0]

    def test_missing_content_path_raises_unreadable(self, tmp_path):
        s = Session(
            id="x", provider="claude", content_path=str(tmp_path / "nope.jsonl")
        )
        with pytest.raises(TranscriptUnreadable):
            load_transcript(s)
        s2 = Session(id="y", provider="claude", content_path="")
        with pytest.raises(TranscriptUnreadable):
            load_transcript(s2)

    def test_unsupported_provider_raises_unreadable(self, tmp_path):
        f = tmp_path / "s.jsonl"
        f.write_text("{}\n")
        with pytest.raises(TranscriptUnreadable):
            load_transcript(Session(id="x", provider="mystery", content_path=str(f)))

    # ------------------------------------------------------------------
    # Timestamp preservation
    # ------------------------------------------------------------------

    def test_tool_use_preserves_timestamp(self, tmp_path):
        """tool_use entry must preserve the assistant event timestamp."""
        f = tmp_path / "s.jsonl"
        write_claude_jsonl(
            f,
            [
                {
                    "type": "assistant",
                    "message": {
                        "content": [
                            {
                                "type": "tool_use",
                                "name": "Bash",
                                "input": {"command": "ls"},
                            },
                        ]
                    },
                    "timestamp": "2026-06-01T10:00:05Z",
                },
            ],
        )
        t = load_transcript(claude_session(f))
        assert t.entries[0].timestamp == "2026-06-01T10:00:05Z"

    def test_tool_result_preserves_timestamp(self, tmp_path):
        """tool_result entry must preserve the user event timestamp."""
        f = tmp_path / "s.jsonl"
        write_claude_jsonl(
            f,
            [
                {
                    "type": "user",
                    "message": {
                        "content": [
                            {"type": "tool_result", "content": "some output"},
                        ]
                    },
                    "timestamp": "2026-06-01T10:00:10Z",
                },
            ],
        )
        t = load_transcript(claude_session(f))
        assert t.entries[0].timestamp == "2026-06-01T10:00:10Z"

    # ------------------------------------------------------------------
    # Empty-content filtering
    # ------------------------------------------------------------------

    def test_empty_string_content_skipped(self, tmp_path):
        """Empty string user/assistant content should not yield entries."""
        f = tmp_path / "s.jsonl"
        write_claude_jsonl(
            f,
            [
                {
                    "type": "user",
                    "message": {"content": ""},
                    "timestamp": "2026-06-01T10:00:00Z",
                },
                {
                    "type": "assistant",
                    "message": {"content": ""},
                    "timestamp": "2026-06-01T10:00:01Z",
                },
                {
                    "type": "user",
                    "message": {"content": "real"},
                    "timestamp": "2026-06-01T10:00:02Z",
                },
            ],
        )
        t = load_transcript(claude_session(f))
        assert [e.text for e in t.entries] == ["real"]

    def test_empty_list_text_blocks_skipped(self, tmp_path):
        """Empty text blocks in list content should not yield entries."""
        f = tmp_path / "s.jsonl"
        write_claude_jsonl(
            f,
            [
                {
                    "type": "user",
                    "message": {
                        "content": [
                            {"type": "text", "text": ""},
                            {"type": "text", "text": "real user text"},
                        ]
                    },
                    "timestamp": "t1",
                },
                {
                    "type": "assistant",
                    "message": {
                        "content": [
                            {"type": "text", "text": ""},
                            {"type": "text", "text": "real assistant text"},
                        ]
                    },
                    "timestamp": "t2",
                },
            ],
        )
        t = load_transcript(claude_session(f))
        assert [e.text for e in t.entries] == ["real user text", "real assistant text"]

    def test_empty_tool_result_content_skipped(self, tmp_path):
        """Empty tool_result string/list content should not yield entries."""
        f = tmp_path / "s.jsonl"
        write_claude_jsonl(
            f,
            [
                {
                    "type": "user",
                    "message": {
                        "content": [
                            {"type": "tool_result", "content": ""},
                            {"type": "tool_result", "content": []},
                            {"type": "tool_result", "content": "real output"},
                        ]
                    },
                    "timestamp": "t1",
                },
            ],
        )
        t = load_transcript(claude_session(f))
        assert [e.text for e in t.entries] == ["real output"]

    # ------------------------------------------------------------------
    # Missing tool_use name
    # ------------------------------------------------------------------

    def test_tool_use_missing_name_defaults_to_question_mark(self, tmp_path):
        """Tool_use without a name should default to '?'."""
        f = tmp_path / "s.jsonl"
        write_claude_jsonl(
            f,
            [
                {
                    "type": "assistant",
                    "message": {
                        "content": [
                            {"type": "tool_use", "input": {"command": "ls"}},
                        ]
                    },
                    "timestamp": "t1",
                },
            ],
        )
        t = load_transcript(claude_session(f))
        assert t.entries[0].text.startswith("?(")
        assert t.entries[0].metadata == {"kind": "call", "tool": "?"}

    # ------------------------------------------------------------------
    # Nested-field malformation hardening (Task 6)
    # ------------------------------------------------------------------

    def test_malformed_message_field_skipped_without_crash(self, tmp_path):
        """A non-dict 'message' field must be skipped, not crash."""
        f = tmp_path / "s.jsonl"
        write_claude_jsonl(
            f,
            [
                {"type": "user", "message": {"content": "before"}},
                {"type": "assistant", "message": None},
                {"type": "assistant", "message": {"content": "after"}},
            ],
        )
        t = load_transcript(claude_session(f))
        assert [e.text for e in t.entries] == ["before", "after"]

    # ------------------------------------------------------------------
    # Delivered queued commands
    # ------------------------------------------------------------------

    def test_queued_human_command_becomes_a_user_entry(self, tmp_path):
        """A prompt typed while the agent worked is a normal user turn.

        Claude records it as an ``attachment``, not a ``user`` message, so
        without this the prompt is absent from the conversation entirely.
        """
        f = tmp_path / "s.jsonl"
        write_claude_jsonl(
            f,
            [
                {
                    "type": "user",
                    "message": {"content": "start the work"},
                    "timestamp": "t1",
                },
                {
                    "type": "queue-operation",
                    "operation": "enqueue",
                    "content": "also check the logs",
                    "timestamp": "t2",
                },
                {
                    "type": "queue-operation",
                    "operation": "remove",
                    "content": "also check the logs",
                    "timestamp": "t3",
                },
                {
                    "type": "attachment",
                    "attachment": {
                        "type": "queued_command",
                        "prompt": "also check the logs",
                        "commandMode": "prompt",
                        "origin": {"kind": "human"},
                        "timestamp": "t2",
                    },
                    "timestamp": "t2",
                },
                {
                    "type": "assistant",
                    "message": {"content": "checking"},
                    "timestamp": "t4",
                },
            ],
        )
        t = load_transcript(claude_session(f))
        assert [(e.role, e.text) for e in t.entries] == [
            ("user", "start the work"),
            ("user", "also check the logs"),
            ("assistant", "checking"),
        ]

    def test_queued_command_emitted_once_not_per_queue_record(self, tmp_path):
        """Enqueue/remove records carry the same text; they must not duplicate it."""
        f = tmp_path / "s.jsonl"
        write_claude_jsonl(
            f,
            [
                {
                    "type": "queue-operation",
                    "operation": "enqueue",
                    "content": "one prompt",
                    "timestamp": "t1",
                },
                {
                    "type": "queue-operation",
                    "operation": "remove",
                    "content": "one prompt",
                    "timestamp": "t2",
                },
                {
                    "type": "attachment",
                    "attachment": {
                        "type": "queued_command",
                        "prompt": "one prompt",
                        "commandMode": "prompt",
                        "origin": {"kind": "human"},
                    },
                    "timestamp": "t1",
                },
            ],
        )
        t = load_transcript(claude_session(f))
        assert [e.text for e in t.entries] == ["one prompt"]

    def test_queued_command_without_queue_records_still_parsed(self, tmp_path):
        """Roughly a quarter of real delivered prompts have no enqueue/remove pair."""
        f = tmp_path / "s.jsonl"
        write_claude_jsonl(
            f,
            [
                {
                    "type": "attachment",
                    "attachment": {
                        "type": "queued_command",
                        "prompt": "unpaired prompt",
                        "commandMode": "prompt",
                        "origin": {"kind": "human"},
                    },
                    "timestamp": "t1",
                },
            ],
        )
        t = load_transcript(claude_session(f))
        assert [(e.role, e.text) for e in t.entries] == [
            ("user", "unpaired prompt"),
        ]

    def test_machine_task_notification_is_not_a_user_entry(self, tmp_path):
        """Over half of real queued_command records are machine notifications.

        They have no ``origin`` and ``commandMode == "task-notification"``.
        Emitting them would fabricate user turns the human never typed.
        """
        f = tmp_path / "s.jsonl"
        write_claude_jsonl(
            f,
            [
                {
                    "type": "attachment",
                    "attachment": {
                        "type": "queued_command",
                        "prompt": "<task-notification>\n<task-id>abc</task-id>",
                        "commandMode": "task-notification",
                        "timestamp": "t1",
                    },
                    "timestamp": "t1",
                },
                {"type": "user", "message": {"content": "a real turn"}},
            ],
        )
        t = load_transcript(claude_session(f))
        assert [e.text for e in t.entries] == ["a real turn"]

    def test_queued_command_image_paste_keeps_text_drops_base64(self, tmp_path):
        """An image-paste prompt is a block list; base64 must never reach search."""
        f = tmp_path / "s.jsonl"
        write_claude_jsonl(
            f,
            [
                {
                    "type": "attachment",
                    "attachment": {
                        "type": "queued_command",
                        "commandMode": "prompt",
                        "origin": {"kind": "human"},
                        "imagePasteIds": ["p1"],
                        "prompt": [
                            {"type": "text", "text": "[Image #1]\n\nmargin looks off"},
                            {
                                "type": "image",
                                "source": {
                                    "type": "base64",
                                    "media_type": "image/png",
                                    "data": "iVBORw0KGgoAAAANSUhEUg",
                                },
                            },
                        ],
                    },
                    "timestamp": "t1",
                },
            ],
        )
        t = load_transcript(claude_session(f))
        assert [(e.role, e.text) for e in t.entries] == [
            ("user", "[Image #1]\n\nmargin looks off"),
        ]
        assert "iVBORw0KGgo" not in t.entries[0].text

    def test_other_attachment_kinds_are_ignored(self, tmp_path):
        """Skill listings, reminders and permission blobs are not user speech."""
        f = tmp_path / "s.jsonl"
        write_claude_jsonl(
            f,
            [
                {
                    "type": "attachment",
                    "attachment": {
                        "type": "task_reminder",
                        "content": "remember the tasks",
                    },
                    "timestamp": "t1",
                },
                {
                    "type": "attachment",
                    "attachment": {"type": "skill_listing", "skills": ["a", "b"]},
                    "timestamp": "t2",
                },
                {"type": "user", "message": {"content": "a real turn"}},
            ],
        )
        t = load_transcript(claude_session(f))
        assert [e.text for e in t.entries] == ["a real turn"]

    # ------------------------------------------------------------------
    # Injected content recorded as user turns
    # ------------------------------------------------------------------

    @pytest.mark.parametrize(
        "text,subtype",
        [
            (
                "<task-notification>\n<task-id>abc</task-id>\n</task-notification>",
                "task_notification",
            ),
            ("[Request interrupted by user]", "interrupted"),
            (
                (
                    "This session is being continued from a previous conversation that "
                    "ran out of context."
                ),
                "continuation",
            ),
        ],
    )
    def test_injected_user_records_become_system_entries(self, tmp_path, text, subtype):
        """Claude records harness-injected text as a user turn; it is not speech.

        16% of non-meta user records in a real corpus are these. Left as user
        entries they answer `get --role user` and literal search as though the
        human had typed them.
        """
        f = tmp_path / "s.jsonl"
        write_claude_jsonl(
            f,
            [
                {"type": "user", "message": {"content": text}, "timestamp": "t1"},
                {
                    "type": "user",
                    "message": {"content": "what I actually said"},
                    "timestamp": "t2",
                },
            ],
        )
        t = load_transcript(claude_session(f))
        assert [(e.role, e.metadata) for e in t.entries] == [
            ("system", {"kind": "injected", "subtype": subtype}),
            ("user", None),
        ]
        assert [e.text for e in t.entries if entry_matches_roles(e, {"user"})] == [
            "what I actually said"
        ]

    def test_injected_marker_mid_message_is_still_a_user_turn(self, tmp_path):
        """Only a leading marker reclassifies; quoting one must not."""
        f = tmp_path / "s.jsonl"
        write_claude_jsonl(
            f,
            [
                {
                    "type": "user",
                    "message": {
                        "content": "why did this print [Request interrupted by user] twice?"
                    },
                },
            ],
        )
        t = load_transcript(claude_session(f))
        assert t.entries[0].role == "user"
        assert t.entries[0].metadata is None

    def test_injected_text_block_in_list_content_is_classified(self, tmp_path):
        """The marker check applies to block-list content, not just strings."""
        f = tmp_path / "s.jsonl"
        write_claude_jsonl(
            f,
            [
                {
                    "type": "user",
                    "message": {
                        "content": [
                            {"type": "text", "text": "[Request interrupted by user]"},
                        ]
                    },
                    "timestamp": "t1",
                },
            ],
        )
        t = load_transcript(claude_session(f))
        assert [(e.role, e.metadata) for e in t.entries] == [
            ("system", {"kind": "injected", "subtype": "interrupted"}),
        ]

    def test_malformed_queued_command_shapes_skipped_without_crash(self, tmp_path):
        """Unknown or broken attachment shapes must be skipped, never guessed."""
        f = tmp_path / "s.jsonl"
        write_claude_jsonl(
            f,
            [
                {"type": "attachment", "attachment": None, "timestamp": "t1"},
                {
                    "type": "attachment",
                    "attachment": {
                        "type": "queued_command",
                        "prompt": "no origin at all",
                    },
                    "timestamp": "t2",
                },
                {
                    "type": "attachment",
                    "attachment": {
                        "type": "queued_command",
                        "prompt": "bad origin",
                        "origin": "human",
                    },
                    "timestamp": "t3",
                },
                {
                    "type": "attachment",
                    "attachment": {
                        "type": "queued_command",
                        "prompt": 42,
                        "origin": {"kind": "human"},
                    },
                    "timestamp": "t4",
                },
                {
                    "type": "attachment",
                    "attachment": {
                        "type": "queued_command",
                        "prompt": "",
                        "origin": {"kind": "human"},
                    },
                    "timestamp": "t5",
                },
                {"type": "user", "message": {"content": "survivor"}},
            ],
        )
        t = load_transcript(claude_session(f))
        assert [e.text for e in t.entries] == ["survivor"]


class TestCodexParser:
    def codex_session(self, path: Path) -> Session:
        return Session(id="cx-1", provider="codex", content_path=str(path))

    def test_event_msg_envelope_and_bare_messages(self, tmp_path):
        f = tmp_path / "r.jsonl"
        write_claude_jsonl(
            f,
            [
                {
                    "type": "event_msg",
                    "timestamp": "2026-06-01T09:00:00Z",
                    "payload": {"type": "user_message", "message": "hello codex"},
                },
                {
                    "type": "event_msg",
                    "payload": {"type": "agent_message", "message": "hello user"},
                },
                {
                    "type": "user_message",
                    "content": [{"type": "input_text", "text": "bare event"}],
                },
                {"type": "agent_message", "message": "bare reply"},
            ],
        )
        t = load_transcript(self.codex_session(f))
        assert [(e.role, e.text) for e in t.entries] == [
            ("user", "hello codex"),
            ("assistant", "hello user"),
            ("user", "bare event"),
            ("assistant", "bare reply"),
        ]
        assert t.entries[0].timestamp == "2026-06-01T09:00:00Z"

    def test_response_item_assistant_message_and_tools(self, tmp_path):
        f = tmp_path / "r.jsonl"
        long_args = json.dumps({"cmd": "x" * 600})
        long_output = "y" * 1200 + " NEEDLE-OUT"
        write_claude_jsonl(
            f,
            [
                {
                    "type": "response_item",
                    "payload": {
                        "type": "message",
                        "role": "assistant",
                        "content": [{"type": "output_text", "text": "answer text"}],
                    },
                },
                {
                    "type": "response_item",
                    "payload": {
                        "type": "function_call",
                        "name": "shell",
                        "arguments": long_args,
                    },
                },
                {
                    "type": "response_item",
                    "payload": {"type": "function_call_output", "output": long_output},
                },
                {
                    "type": "response_item",
                    "payload": {
                        "type": "custom_tool_call",
                        "name": "apply_patch",
                        "input": "patch body",
                    },
                },
                {
                    "type": "response_item",
                    "payload": {
                        "type": "custom_tool_call_output",
                        "output": "patched ok",
                    },
                },
                {
                    "type": "response_item",
                    "payload": {"type": "reasoning", "encrypted_content": "opaque"},
                },
            ],
        )
        t = load_transcript(self.codex_session(f))
        roles = [e.role for e in t.entries]
        assert roles == ["assistant", "tool", "tool", "tool", "tool"]
        # Completeness regression: the old renderer cut args at 300 and
        # output at 500 chars — the service must keep everything.
        assert t.entries[1].text == f"shell({long_args})"
        assert t.entries[2].text == long_output
        assert t.entries[1].metadata["kind"] == "call"
        assert t.entries[2].metadata == {"kind": "output"}
        assert t.entries[3].text == "apply_patch(patch body)"
        assert t.entries[4].text == "patched ok"

    def test_agent_message_response_item_pair_deduped(self, tmp_path):
        """Codex records each assistant turn twice (event_msg agent_message
        + response_item message, ms apart); only one entry must survive."""
        f = tmp_path / "r.jsonl"
        write_claude_jsonl(
            f,
            [
                {
                    "type": "event_msg",
                    "timestamp": "2026-06-08T15:51:59.689Z",
                    "payload": {"type": "agent_message", "message": "the HUD was real"},
                },
                {
                    "type": "response_item",
                    "timestamp": "2026-06-08T15:51:59.695Z",
                    "payload": {
                        "type": "message",
                        "role": "assistant",
                        "content": [
                            {"type": "output_text", "text": "the HUD was real"}
                        ],
                    },
                },
                {
                    "type": "event_msg",
                    "payload": {"type": "user_message", "message": "thanks"},
                },
            ],
        )
        t = load_transcript(self.codex_session(f))
        assert [(e.role, e.text) for e in t.entries] == [
            ("assistant", "the HUD was real"),
            ("user", "thanks"),
        ]
        # The kept entry is the first of the pair (its timestamp wins).
        assert t.entries[0].timestamp == "2026-06-08T15:51:59.689Z"

    def test_identical_answers_across_turns_survive_dedup(self, tmp_path):
        """Only *adjacent* assistant repeats are duplicates; the same
        answer re-given after an intervening turn must be kept."""
        f = tmp_path / "r.jsonl"
        write_claude_jsonl(
            f,
            [
                {
                    "type": "event_msg",
                    "payload": {"type": "agent_message", "message": "Implemented."},
                },
                {
                    "type": "event_msg",
                    "payload": {"type": "user_message", "message": "again please"},
                },
                {
                    "type": "event_msg",
                    "payload": {"type": "agent_message", "message": "Implemented."},
                },
            ],
        )
        t = load_transcript(self.codex_session(f))
        assert [(e.role, e.text) for e in t.entries] == [
            ("assistant", "Implemented."),
            ("user", "again please"),
            ("assistant", "Implemented."),
        ]

    def test_dedup_applies_to_search_counts_and_indices(self, tmp_path):
        """search goes through the same entry stream: the duplicated turn
        must count once, and entry indices must match get's."""
        f = tmp_path / "r.jsonl"
        write_claude_jsonl(
            f,
            [
                {
                    "type": "event_msg",
                    "payload": {"type": "agent_message", "message": "rare gemsbok"},
                },
                {
                    "type": "response_item",
                    "payload": {
                        "type": "message",
                        "role": "assistant",
                        "content": [{"type": "output_text", "text": "rare gemsbok"}],
                    },
                },
            ],
        )
        r = search_session(self.codex_session(f), "gemsbok")
        assert r.match_count == 1
        assert r.total_entries == 1
        assert r.matches[0].entry_index == 0

    def test_legacy_output_item_done_and_patch_apply(self, tmp_path):
        f = tmp_path / "r.jsonl"
        write_claude_jsonl(
            f,
            [
                {
                    "type": "response.output_item.done",
                    "item": {
                        "type": "message",
                        "content": [{"type": "output_text", "text": "legacy text"}],
                    },
                },
                {"type": "patch_apply_end", "stdout": "Done it\n"},
            ],
        )
        t = load_transcript(self.codex_session(f))
        assert [(e.role, e.text) for e in t.entries] == [
            ("assistant", "legacy text"),
            ("tool", "Done it"),
        ]
        assert t.entries[1].metadata == {"kind": "output"}

    def test_malformed_event_msg_payload_skipped_without_crash(self, tmp_path):
        """Null or non-dict event_msg.payload must be skipped, not crash."""
        f = tmp_path / "r.jsonl"
        write_claude_jsonl(
            f,
            [
                {
                    "type": "event_msg",
                    "timestamp": "t1",
                    "payload": {"type": "user_message", "message": "first good"},
                },
                {"type": "event_msg", "payload": None},
                {"type": "event_msg", "payload": "just a string"},
                {"type": "event_msg", "payload": 42},
                {
                    "type": "event_msg",
                    "timestamp": "t2",
                    "payload": {"type": "agent_message", "message": "second good"},
                },
            ],
        )
        t = load_transcript(self.codex_session(f))
        assert [(e.role, e.text) for e in t.entries] == [
            ("user", "first good"),
            ("assistant", "second good"),
        ]
        assert t.warnings == []
        assert t.entries[0].timestamp == "t1"
        assert t.entries[1].timestamp == "t2"

    # ------------------------------------------------------------------
    # The three eras of a Codex user turn. Which vocabulary a rollout uses
    # depends on its thread history mode, and no single one appears in every
    # file, so each is covered here on its own and in the pairings that occur.
    # ------------------------------------------------------------------

    def response_user(self, text: str) -> dict:
        return {
            "type": "response_item",
            "timestamp": "2026-06-01T09:00:00Z",
            "payload": {
                "type": "message",
                "role": "user",
                "content": [{"type": "input_text", "text": text}],
            },
        }

    def item_completed_user(self, text: str) -> dict:
        return {
            "type": "event_msg",
            "timestamp": "2026-06-01T09:00:01Z",
            "payload": {
                "type": "item_completed",
                "item": {
                    "type": "UserMessage",
                    "id": "item-1",
                    "content": [{"type": "text", "text": text}],
                },
            },
        }

    def event_user(self, text: str) -> dict:
        return {
            "type": "event_msg",
            "timestamp": "2026-06-01T09:00:01Z",
            "payload": {"type": "user_message", "message": text},
        }

    def test_response_item_user_turn_is_the_only_record_in_a_paginated_rollout(
        self, tmp_path
    ):
        """A rollout with neither canonical vocabulary still yields its turns.

        128 of 784 real rollouts look like this. Before the response item was
        parsed, searching them for anything the user typed returned nothing.
        """
        f = tmp_path / "r.jsonl"
        write_claude_jsonl(
            f,
            [
                self.response_user("clean out any duplication from the fallback"),
                {
                    "type": "response_item",
                    "payload": {
                        "type": "message",
                        "role": "assistant",
                        "content": [{"type": "output_text", "text": "will do"}],
                    },
                },
            ],
        )
        t = load_transcript(self.codex_session(f))
        assert [(e.role, e.text) for e in t.entries] == [
            ("user", "clean out any duplication from the fallback"),
            ("assistant", "will do"),
        ]
        assert t.entries[0].timestamp == "2026-06-01T09:00:00Z"

    def test_item_completed_user_turn(self, tmp_path):
        """The paginated vocabulary: a UserMessage TurnItem. Its ``skill``
        parts name an invoked skill and carry no prose, so they add nothing."""
        f = tmp_path / "r.jsonl"
        write_claude_jsonl(
            f,
            [
                {
                    "type": "event_msg",
                    "timestamp": "2026-06-01T09:00:01Z",
                    "payload": {
                        "type": "item_completed",
                        "item": {
                            "type": "UserMessage",
                            "id": "item-1",
                            "content": [
                                {"type": "text", "text": "use the herdr skill"},
                                {"type": "skill", "name": "herdr", "path": "/x/y.md"},
                            ],
                        },
                    },
                },
                # Only UserMessage is taken: an AgentMessage item repeats the
                # response item the model returned.
                {
                    "type": "event_msg",
                    "payload": {
                        "type": "item_completed",
                        "item": {"type": "AgentMessage", "id": "item-2"},
                    },
                },
            ],
        )
        t = load_transcript(self.codex_session(f))
        assert [(e.role, e.text) for e in t.entries] == [
            ("user", "use the herdr skill")
        ]

    def test_paired_response_item_is_dropped_by_source_not_by_text(self, tmp_path):
        """The legacy pairing, with texts that differ.

        19 real pairs are not text-identical — the response item carries extra
        model-context wrapping — so pairing on equality would leak them. The
        canonical event wins, and its text is the one kept.
        """
        f = tmp_path / "r.jsonl"
        write_claude_jsonl(
            f,
            [
                self.response_user("# Context from my IDE setup:\n\nship it"),
                self.event_user("ship it"),
            ],
        )
        t = load_transcript(self.codex_session(f))
        assert [(e.role, e.text) for e in t.entries] == [("user", "ship it")]

    def test_paired_response_item_is_dropped_in_the_paginated_era_too(self, tmp_path):
        f = tmp_path / "r.jsonl"
        write_claude_jsonl(
            f,
            [
                self.response_user("ship it"),
                self.item_completed_user("ship it"),
            ],
        )
        t = load_transcript(self.codex_session(f))
        assert [(e.role, e.text) for e in t.entries] == [("user", "ship it")]

    def test_pairing_survives_an_intervening_lifecycle_record(self, tmp_path):
        """Upstream does not promise the two records are adjacent, and a
        record that yields no entry must not break the pair."""
        f = tmp_path / "r.jsonl"
        write_claude_jsonl(
            f,
            [
                self.response_user("ship it"),
                {"type": "event_msg", "payload": {"type": "token_count", "total": 7}},
                {"type": "turn_context", "payload": {"cwd": "/tmp"}},
                self.event_user("ship it"),
            ],
        )
        t = load_transcript(self.codex_session(f))
        assert [(e.role, e.text) for e in t.entries] == [("user", "ship it")]

    def test_repeated_canonical_user_turns_are_never_collapsed(self, tmp_path):
        """A retry after an aborted turn repeats the text verbatim, and 60 of
        those exist in the real corpus. Only response→canonical collapses."""
        f = tmp_path / "r.jsonl"
        write_claude_jsonl(
            f,
            [
                self.event_user("try again"),
                self.event_user("try again"),
                self.response_user("and again"),
                self.response_user("and again"),
            ],
        )
        t = load_transcript(self.codex_session(f))
        assert [e.text for e in t.entries] == [
            "try again",
            "try again",
            "and again",
            "and again",
        ]

    def test_injected_context_is_not_indexed(self, tmp_path):
        """Codex presents its own injected context as role=user. Indexing it
        would match every session in a repository on its AGENTS.md."""
        f = tmp_path / "r.jsonl"
        write_claude_jsonl(
            f,
            [
                self.response_user("# AGENTS.md instructions for /repo\n\nbe careful"),
                self.response_user("<environment_context>\n <cwd>/repo</cwd>"),
                self.response_user('<codex_internal_context source="goal">\ngoal'),
                # Not injected: the image prefix opens a real question, and
                # dropping it would lose the question with it.
                self.response_user("<image name=[Image #1]></image>why the dupes?"),
            ],
        )
        t = load_transcript(self.codex_session(f))
        assert [e.text for e in t.entries] == [
            "<image name=[Image #1]></image>why the dupes?"
        ]

    def test_injected_context_still_loses_to_a_canonical_record(self, tmp_path):
        """The filter is a fallback-path rule. A human who genuinely opens a
        message that way is recorded canonically, and that record wins."""
        f = tmp_path / "r.jsonl"
        write_claude_jsonl(
            f,
            [
                self.response_user("<environment_context>\n <cwd>/repo</cwd>"),
                self.event_user("<environment_context> is what I want to discuss"),
            ],
        )
        t = load_transcript(self.codex_session(f))
        assert [e.text for e in t.entries] == [
            "<environment_context> is what I want to discuss"
        ]

    def test_response_item_user_image_only_yields_no_text(self, tmp_path):
        """An image part carries no searchable text. The record still counts
        as activity for discovery, which is tested there, not here."""
        f = tmp_path / "r.jsonl"
        write_claude_jsonl(
            f,
            [
                {
                    "type": "response_item",
                    "payload": {
                        "type": "message",
                        "role": "user",
                        "content": [{"type": "input_image", "image_url": "data:x"}],
                    },
                },
            ],
        )
        t = load_transcript(self.codex_session(f))
        assert t.entries == []

    def test_developer_response_item_is_skipped(self, tmp_path):
        """role=developer carries the system prompt, not a turn."""
        f = tmp_path / "r.jsonl"
        write_claude_jsonl(
            f,
            [
                {
                    "type": "response_item",
                    "payload": {
                        "type": "message",
                        "role": "developer",
                        "content": [{"type": "input_text", "text": "<permissions>"}],
                    },
                },
                {
                    "type": "response_item",
                    "payload": {
                        "type": "message",
                        "role": "assistant",
                        "content": [{"type": "output_text", "text": "valid assistant"}],
                    },
                },
            ],
        )
        t = load_transcript(self.codex_session(f))
        assert [e.role for e in t.entries] == ["assistant"]
        assert t.entries[0].text == "valid assistant"

    # ------------------------------------------------------------------
    # Edge-case hardening (Task 2)
    # ------------------------------------------------------------------

    def test_response_item_null_and_scalar_payload_skipped(self, tmp_path):
        """response_item with null or non-dict payload must not crash."""
        f = tmp_path / "r.jsonl"
        write_claude_jsonl(
            f,
            [
                {
                    "type": "response_item",
                    "payload": {
                        "type": "message",
                        "role": "assistant",
                        "content": [{"type": "output_text", "text": "before"}],
                    },
                },
                {"type": "response_item", "payload": None},
                {"type": "response_item", "payload": "scalar string"},
                {"type": "response_item", "payload": 99},
                {
                    "type": "response_item",
                    "payload": {
                        "type": "message",
                        "role": "assistant",
                        "content": [{"type": "output_text", "text": "after"}],
                    },
                },
            ],
        )
        t = load_transcript(self.codex_session(f))
        assert [e.text for e in t.entries] == ["before", "after"]

    def test_patch_apply_end_non_string_stdout(self, tmp_path):
        """patch_apply_end with non-string stdout must not crash and must
        skip entries for non-string truthy values."""
        f = tmp_path / "r.jsonl"
        write_claude_jsonl(
            f,
            [
                {"type": "patch_apply_end", "stdout": "valid output\n"},
                {"type": "patch_apply_end", "stdout": 42},
                {"type": "patch_apply_end", "stdout": ["list", "output"]},
                {"type": "patch_apply_end", "stdout": {"key": "val"}},
                {"type": "patch_apply_end", "stdout": ""},
                {
                    "type": "user_message",
                    "content": [{"type": "input_text", "text": "still alive"}],
                },
            ],
        )
        t = load_transcript(self.codex_session(f))
        assert [e.text for e in t.entries] == ["valid output", "still alive"]

    def test_function_call_output_null_output_skipped(self, tmp_path):
        """function_call_output / custom_tool_call_output with null output
        must not yield a bogus 'null' string entry."""
        f = tmp_path / "r.jsonl"
        write_claude_jsonl(
            f,
            [
                {
                    "type": "response_item",
                    "payload": {
                        "type": "function_call",
                        "name": "fn1",
                        "arguments": "{}",
                    },
                },
                {
                    "type": "response_item",
                    "payload": {"type": "function_call_output", "output": None},
                },
                {
                    "type": "response_item",
                    "payload": {
                        "type": "function_call_output",
                        "output": "real result",
                    },
                },
                {
                    "type": "response_item",
                    "payload": {
                        "type": "custom_tool_call",
                        "name": "tool1",
                        "input": "{}",
                    },
                },
                {
                    "type": "response_item",
                    "payload": {"type": "custom_tool_call_output", "output": None},
                },
                {
                    "type": "response_item",
                    "payload": {
                        "type": "custom_tool_call_output",
                        "output": "real output",
                    },
                },
            ],
        )
        t = load_transcript(self.codex_session(f))
        # Should have tool calls and their real outputs, but no bogus "null" entries
        texts = [e.text for e in t.entries]
        assert "null" not in texts
        assert texts == ["fn1({})", "real result", "tool1({})", "real output"]

    # ------------------------------------------------------------------
    # Nested-field malformation hardening (Task 6)
    # ------------------------------------------------------------------

    def test_legacy_output_item_done_null_item_skipped(self, tmp_path):
        """response.output_item.done with null or scalar item must not crash."""
        f = tmp_path / "r.jsonl"
        write_claude_jsonl(
            f,
            [
                {
                    "type": "response.output_item.done",
                    "item": {
                        "type": "message",
                        "content": [{"type": "output_text", "text": "before"}],
                    },
                },
                {"type": "response.output_item.done", "item": None},
                {"type": "response.output_item.done", "item": "just a string"},
                {"type": "response.output_item.done", "item": 42},
                {
                    "type": "response.output_item.done",
                    "item": {
                        "type": "message",
                        "content": [{"type": "output_text", "text": "after"}],
                    },
                },
            ],
        )
        t = load_transcript(self.codex_session(f))
        assert [e.text for e in t.entries] == ["before", "after"]

    def test_response_item_assistant_null_content_skipped(self, tmp_path):
        """response_item assistant message with null/scalar content must
        not crash, and surrounding valid events still parse."""
        f = tmp_path / "r.jsonl"
        write_claude_jsonl(
            f,
            [
                {
                    "type": "response_item",
                    "payload": {
                        "type": "message",
                        "role": "assistant",
                        "content": [{"type": "output_text", "text": "before"}],
                    },
                },
                {
                    "type": "response_item",
                    "payload": {
                        "type": "message",
                        "role": "assistant",
                        "content": None,
                    },
                },
                {
                    "type": "response_item",
                    "payload": {
                        "type": "message",
                        "role": "assistant",
                        "content": "a scalar string",
                    },
                },
                {
                    "type": "response_item",
                    "payload": {
                        "type": "message",
                        "role": "assistant",
                        "content": [{"type": "output_text", "text": "after"}],
                    },
                },
            ],
        )
        t = load_transcript(self.codex_session(f))
        assert [e.text for e in t.entries] == ["before", "after"]


def make_opencode_db(home: Path) -> Path:
    """Create a minimal opencode DB with one session and rich parts."""
    import sqlite3

    db_path = home / ".local" / "share" / "opencode" / "opencode.db"
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path))
    conn.execute(
        "CREATE TABLE message (id TEXT PRIMARY KEY, session_id TEXT, "
        "time_created INTEGER, time_updated INTEGER, data TEXT)"
    )
    conn.execute(
        "CREATE TABLE part (id TEXT PRIMARY KEY, message_id TEXT, "
        "session_id TEXT, time_created INTEGER, time_updated INTEGER, "
        "data TEXT)"
    )
    long_out = "v" * 800 + " END"
    rows = [
        (
            "m1",
            "ses_1",
            1,
            json.dumps({"role": "user"}),
            "p1",
            json.dumps({"type": "text", "text": "please fix auth"}),
        ),
        (
            "m2",
            "ses_1",
            2,
            json.dumps({"role": "assistant"}),
            "p2",
            json.dumps({"type": "text", "text": "fixing now"}),
        ),
        (
            "m3",
            "ses_1",
            3,
            json.dumps({"role": "assistant"}),
            "p3",
            json.dumps(
                {
                    "type": "tool",
                    "tool": "bash",
                    "state": {
                        "status": "completed",
                        "input": {"command": "pytest -q"},
                        "output": long_out,
                    },
                }
            ),
        ),
        (
            "m4",
            "ses_1",
            4,
            json.dumps({"role": "assistant"}),
            "p4",
            json.dumps({"type": "step-start", "snapshot": "SECRET-SNAP"}),
        ),
    ]
    for mid, sid, ts, mdata, pid, pdata in rows:
        conn.execute(
            "INSERT INTO message VALUES (?, ?, ?, ?, ?)", (mid, sid, ts, ts, mdata)
        )
        conn.execute(
            "INSERT INTO part VALUES (?, ?, ?, ?, ?, ?)", (pid, mid, sid, ts, ts, pdata)
        )
    conn.commit()
    conn.close()
    return db_path


class TestOpencodeParser:
    def test_text_and_tool_parts_complete(self, tmp_path):
        make_opencode_db(tmp_path)
        s = Session(id="ses_1", provider="opencode")
        with (
            patch("session_browser.transcript.Path.home", return_value=tmp_path),
            patch("session_browser.discovery.Path.home", return_value=tmp_path),
        ):
            t = load_transcript(s)
        assert [(e.role, e.text) for e in t.entries[:2]] == [
            ("user", "please fix auth"),
            ("assistant", "fixing now"),
        ]
        tool = t.entries[2]
        assert tool.role == "tool"
        assert tool.text.startswith('bash({"command": "pytest -q"}) [completed]')
        assert tool.text.endswith("v" * 800 + " END")  # old code cut at 500
        # step-start parts are internal state: never an entry.
        assert len(t.entries) == 3
        assert all("SECRET-SNAP" not in e.text for e in t.entries)

    def test_missing_db_is_unreadable(self, tmp_path):
        s = Session(id="ses_1", provider="opencode")
        with (
            patch(
                "session_browser.discovery.Path.home", return_value=tmp_path / "void"
            ),
            pytest.raises(TranscriptUnreadable),
        ):
            load_transcript(s)

    def test_tool_part_error_status_captured(self):
        """A tool part with state.status == 'error' must carry is_error
        metadata; other statuses must not."""
        failed = _opencode_part_entry(
            json.dumps(
                {
                    "type": "tool",
                    "tool": "bash",
                    "state": {
                        "status": "error",
                        "input": {"command": "make"},
                        "output": "exit 2",
                    },
                }
            ),
            "assistant",
        )
        assert failed.metadata == {"tool": "bash", "is_error": True}
        ok = _opencode_part_entry(
            json.dumps(
                {
                    "type": "tool",
                    "tool": "bash",
                    "state": {"status": "completed", "input": {}},
                }
            ),
            "assistant",
        )
        assert ok.metadata == {"tool": "bash"}

    # ------------------------------------------------------------------
    # Non-dict JSON guard gap (Task 6)
    # ------------------------------------------------------------------

    def test_non_dict_msg_data_falls_back_to_system_role(self, tmp_path):
        """message.data that is valid JSON but not a dict must not crash
        and should fall back to role='system'."""
        db_path = tmp_path / ".local" / "share" / "opencode" / "opencode.db"
        db_path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(str(db_path))
        conn.execute(
            "CREATE TABLE message (id TEXT PRIMARY KEY, session_id TEXT, "
            "time_created INTEGER, time_updated INTEGER, data TEXT)"
        )
        conn.execute(
            "CREATE TABLE part (id TEXT PRIMARY KEY, message_id TEXT, "
            "session_id TEXT, time_created INTEGER, time_updated INTEGER, "
            "data TEXT)"
        )
        # msg_data is a JSON string (valid JSON but not a dict)
        conn.execute(
            "INSERT INTO message VALUES ('m1', 'ses_nd1', 1, 1, ?)",
            (json.dumps("not a dict"),),
        )
        conn.execute(
            "INSERT INTO part VALUES ('p1', 'm1', 'ses_nd1', 1, 1, ?)",
            (json.dumps({"type": "text", "text": "hello from non-dict msg"}),),
        )
        conn.commit()
        conn.close()
        s = Session(id="ses_nd1", provider="opencode")
        with (
            patch("session_browser.transcript.Path.home", return_value=tmp_path),
            patch("session_browser.discovery.Path.home", return_value=tmp_path),
        ):
            t = load_transcript(s)
        assert len(t.entries) == 1
        assert t.entries[0].role == "system"
        assert t.entries[0].text == "hello from non-dict msg"

    def test_unparseable_msg_data_falls_back_to_system_role(self, tmp_path):
        """message.data that cannot be parsed at all must fall back to
        role='system', never to a speaking role.

        Covers the ``except (JSONDecodeError, TypeError)`` arm, which the
        non-dict test above does not reach: that one parses successfully and
        fails the isinstance check. Both arms exist, and both must land on
        'system'.

        The role matters more than the recovery. ``--role user`` is documented
        as human speech only, so a corrupt message row surfacing as a user turn
        would put words that nobody typed into the intent trail — the exact
        misclassification shape behind 3bc9bc7 and 529f179.
        """
        db_path = tmp_path / ".local" / "share" / "opencode" / "opencode.db"
        db_path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(str(db_path))
        conn.execute(
            "CREATE TABLE message (id TEXT PRIMARY KEY, session_id TEXT, "
            "time_created INTEGER, time_updated INTEGER, data TEXT)"
        )
        conn.execute(
            "CREATE TABLE part (id TEXT PRIMARY KEY, message_id TEXT, "
            "session_id TEXT, time_created INTEGER, time_updated INTEGER, "
            "data TEXT)"
        )
        # Truncated JSON — json.loads raises JSONDecodeError.
        conn.execute(
            "INSERT INTO message VALUES ('m1', 'ses_bad', 1, 1, ?)",
            ('{"role": "user"',),
        )
        conn.execute(
            "INSERT INTO part VALUES ('p1', 'm1', 'ses_bad', 1, 1, ?)",
            (json.dumps({"type": "text", "text": "from truncated msg"}),),
        )
        # NULL data — json.loads(None) raises TypeError.
        conn.execute("INSERT INTO message VALUES ('m2', 'ses_bad', 2, 2, NULL)")
        conn.execute(
            "INSERT INTO part VALUES ('p2', 'm2', 'ses_bad', 2, 2, ?)",
            (json.dumps({"type": "text", "text": "from null msg"}),),
        )
        conn.commit()
        conn.close()
        s = Session(id="ses_bad", provider="opencode")
        with (
            patch("session_browser.transcript.Path.home", return_value=tmp_path),
            patch("session_browser.discovery.Path.home", return_value=tmp_path),
        ):
            t = load_transcript(s)
        assert [(e.role, e.text) for e in t.entries] == [
            ("system", "from truncated msg"),
            ("system", "from null msg"),
        ]

    def test_non_dict_part_data_skipped_gracefully(self, tmp_path):
        """part.data that is valid JSON but not a dict must be skipped
        without crashing; surrounding valid content still parsed."""
        db_path = tmp_path / ".local" / "share" / "opencode" / "opencode.db"
        db_path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(str(db_path))
        conn.execute(
            "CREATE TABLE message (id TEXT PRIMARY KEY, session_id TEXT, "
            "time_created INTEGER, time_updated INTEGER, data TEXT)"
        )
        conn.execute(
            "CREATE TABLE part (id TEXT PRIMARY KEY, message_id TEXT, "
            "session_id TEXT, time_created INTEGER, time_updated INTEGER, "
            "data TEXT)"
        )
        conn.execute(
            "INSERT INTO message VALUES ('m1', 'ses_nd2', 1, 1, ?)",
            (json.dumps({"role": "user"}),),
        )
        # valid part before the bad one
        conn.execute(
            "INSERT INTO part VALUES ('p1', 'm1', 'ses_nd2', 1, 1, ?)",
            (json.dumps({"type": "text", "text": "before"}),),
        )
        # part.data is a JSON array (valid JSON but not a dict)
        conn.execute(
            "INSERT INTO part VALUES ('p2', 'm1', 'ses_nd2', 2, 2, ?)",
            (json.dumps(["not", "a", "dict"]),),
        )
        # valid part after the bad one
        conn.execute(
            "INSERT INTO part VALUES ('p3', 'm1', 'ses_nd2', 3, 3, ?)",
            (json.dumps({"type": "text", "text": "after"}),),
        )
        conn.commit()
        conn.close()
        s = Session(id="ses_nd2", provider="opencode")
        with (
            patch("session_browser.transcript.Path.home", return_value=tmp_path),
            patch("session_browser.discovery.Path.home", return_value=tmp_path),
        ):
            t = load_transcript(s)
        assert [e.text for e in t.entries] == ["before", "after"]
        assert t.warnings == []


class TestRendering:
    def _transcript(self) -> Transcript:
        s = Session(
            id="abc",
            provider="claude",
            summary="Fix auth",
            cwd="/home/u/proj",
            branch="main",
            created_at="2026-06-01T10:00:00Z",
            updated_at="2026-06-02T10:00:00Z",
        )
        return Transcript(
            s,
            [
                TranscriptEntry("user", "please fix"),
                TranscriptEntry("assistant", "done"),
                TranscriptEntry(
                    "tool", "Bash({})", metadata={"kind": "call", "tool": "Bash"}
                ),
                TranscriptEntry("tool", "ok", metadata={"kind": "output"}),
                TranscriptEntry("tool", "merged", metadata={"tool": "x"}),
                TranscriptEntry("system", "note"),
            ],
            warnings=["line 9: invalid JSON, skipped"],
        )

    def test_render_text_labels_and_separation(self):
        text = render_text(self._transcript())
        assert text.split("\n\n") == [
            "User: please fix",
            "Assistant: done",
            "Tool call: Bash({})",
            "Tool output: ok",
            "Tool: merged",
            "System: note",
        ]

    def test_render_text_empty(self):
        t = Transcript(Session(id="x", provider="claude"), [])
        assert render_text(t) == "(empty session)"

    def test_render_markdown_has_header_and_body(self):
        md = render_markdown(self._transcript())
        assert md.startswith("# Session claude:abc\n")
        assert "- Provider: claude" in md
        assert "- Summary: Fix auth" in md
        assert "- Parse warnings: 1" in md
        assert "\n---\n" in md
        assert "User: please fix" in md

    def test_serialization_shapes(self):
        t = self._transcript()
        d = transcript_to_dict(t)
        assert d["session"] == session_to_dict(t.session)
        assert d["session"]["id"] == "claude:abc"
        assert d["session"]["session_id"] == "abc"
        assert d["entries"][0] == {
            "role": "user",
            "text": "please fix",
            "timestamp": None,
            "metadata": None,
        }
        assert d["entries"][2]["metadata"] == {"kind": "call", "tool": "Bash"}
        assert d["warnings"] == ["line 9: invalid JSON, skipped"]
        json.dumps(d)  # must be JSON-serializable

    def test_error_entries_labelled(self):
        t = Transcript(
            Session(id="x", provider="claude"),
            [
                TranscriptEntry(
                    "tool", "boom", metadata={"kind": "output", "is_error": True}
                ),
                TranscriptEntry(
                    "tool", "make: exit 2", metadata={"tool": "bash", "is_error": True}
                ),
            ],
        )
        assert render_text(t).split("\n\n") == [
            "Tool output (error): boom",
            "Tool (error): make: exit 2",
        ]

    def test_entry_matches_roles_with_error_pseudo_role(self):
        err = TranscriptEntry(
            "tool", "boom", metadata={"kind": "output", "is_error": True}
        )
        ok = TranscriptEntry("tool", "fine", metadata={"kind": "output"})
        plain = TranscriptEntry("assistant", "hi")
        assert entry_matches_roles(err, {"error"})
        assert not entry_matches_roles(ok, {"error"})
        assert not entry_matches_roles(plain, {"error"})
        assert entry_matches_roles(ok, {"tool"})
        assert entry_matches_roles(plain, {"assistant", "error"})

    def test_render_text_with_indices_prefixes_blocks(self):
        t = Transcript(
            Session(id="x", provider="claude"),
            [
                TranscriptEntry("user", "please fix"),
                TranscriptEntry("user", "still broken"),
            ],
        )
        assert render_text(t, entry_indices=[0, 7]).split("\n\n") == [
            "[0] User: please fix",
            "[7] User: still broken",
        ]

    def test_render_markdown_role_filtered_header_and_empty_body(self):
        t = self._transcript()
        kept = Transcript(t.session, [t.entries[0]], t.warnings)
        md = render_markdown(kept, total_entries=6, entry_indices=[0], roles=["user"])
        assert "- Entries: 1 of 6 (roles: user)" in md
        assert "[0] User: please fix" in md
        empty = Transcript(t.session, [], t.warnings)
        md = render_markdown(
            empty,
            total_entries=6,
            entry_indices=[],
            roles=["system"],
            entry_range=(0, 2),
        )
        assert "- Entries: 0 of 6 (roles: system), within 0–2" in md
        assert "(no entries with roles: system)" in md

    def test_transcript_to_dict_with_indices(self):
        t = self._transcript()
        kept = Transcript(t.session, [t.entries[0], t.entries[5]], t.warnings)
        d = transcript_to_dict(kept, entry_indices=[0, 5])
        assert [(e["index"], e["role"]) for e in d["entries"]] == [
            (0, "user"),
            (5, "system"),
        ]

    def test_canonical_id(self):
        assert canonical_id(Session(id="a-1", provider="codex")) == "codex:a-1"


class TestLoadSessionContent:
    def test_renders_readable_text(self, tmp_path):
        f = tmp_path / "s.jsonl"
        write_claude_jsonl(f, [{"type": "user", "message": {"content": "hi"}}])
        s = Session(id="x", provider="claude", content_path=str(f))
        assert load_session_content(s) == "User: hi"

    def test_unreadable_session_returns_message(self):
        s = Session(id="x", provider="claude", content_path="/nope/missing")
        out = load_session_content(s)
        assert out.startswith("(could not read session:")


class TestEntryMatching:
    def test_case_insensitive_offsets_per_entry(self):
        entries = [
            TranscriptEntry("user", "Wombat says wombat"),
            TranscriptEntry("assistant", "no marsupials here"),
            TranscriptEntry("tool", "WOMBAT"),
        ]
        matches = list(find_entry_matches(entries, "wombat"))
        assert [(m.entry_index, m.offsets) for m in matches] == [(0, [0, 12]), (2, [0])]
        assert matches[0].entry.role == "user"

    def test_blank_query_matches_nothing(self):
        entries = [TranscriptEntry("user", "anything")]
        assert list(find_entry_matches(entries, "")) == []
        assert list(find_entry_matches(entries, "   ")) == []

    def test_unicode_casefold_matching(self):
        """Unicode casefold matching: 'strasse' matches 'Straße'."""
        entries = [TranscriptEntry("user", "Straße")]
        matches = list(find_entry_matches(entries, "strasse"))
        assert len(matches) == 1
        assert matches[0].offsets == [0]

    def test_multiple_queries_or_in_one_pass(self):
        entries = [
            TranscriptEntry("user", "wombat here"),
            TranscriptEntry("assistant", "quokka there"),
            TranscriptEntry("tool", "both wombat and quokka"),
        ]
        matches = list(find_entry_matches(entries, ["wombat", "quokka"]))
        assert [(m.entry_index, m.query) for m in matches] == [
            (0, "wombat"),
            (1, "quokka"),
            (2, "wombat"),
            (2, "quokka"),
        ]

    def test_blank_queries_dropped_from_list(self):
        entries = [TranscriptEntry("user", "wombat")]
        matches = list(find_entry_matches(entries, ["  ", "wombat"]))
        assert [(m.entry_index, m.query) for m in matches] == [(0, "wombat")]


class TestMarkdownInsensitiveMatching:
    """Backticks/asterisks are formatting, not content: 'SELECT only'
    must find '`SELECT` only' (the exact miss that lost the May session
    in the 2026-07-17 retrieval experiment)."""

    def test_backticked_haystack_matches_plain_query(self):
        entries = [
            TranscriptEntry("assistant", "every SQL statement is `SELECT` only.")
        ]
        matches = list(find_entry_matches(entries, "select only"))
        assert len(matches) == 1
        # Offset maps back to the original text: the match starts at the
        # backtick-stripped "S", i.e. the character after the backtick.
        off = matches[0].offsets[0]
        assert entries[0].text[off] == "S"

    def test_bold_haystack_matches_plain_query(self):
        entries = [TranscriptEntry("user", "this is a **bold** claim")]
        matches = list(find_entry_matches(entries, "bold claim"))
        assert len(matches) == 1

    def test_markdown_in_query_matches_plain_haystack(self):
        entries = [TranscriptEntry("user", "select only statements")]
        matches = list(find_entry_matches(entries, "`select` only"))
        assert len(matches) == 1
        assert matches[0].offsets == [0]

    def test_fold_and_strip_length_cancellation(self):
        """One 'ß' (+1 char) and one '`' (-1 char) keep the length equal;
        identity must be decided by equality, not length, or offsets
        would silently be wrong."""
        entries = [TranscriptEntry("user", "ß`marker here")]
        matches = list(find_entry_matches(entries, "ssmarker"))
        assert len(matches) == 1
        assert matches[0].offsets == [0]

    def test_normalize_map_agrees_with_fast_path(self):
        cases = [
            "plain text",
            "`code` and **bold**",
            "Straße`x*",
            "ß`cancel",
            "",
            "```fence```",
            "*",
            "İstanbul `mix`",
        ]
        for text in cases:
            normalized, idx_map = transcript_mod._normalize_map(text)
            assert normalized == transcript_mod.normalize_match_text(text), text
            assert len(idx_map) == len(normalized), text

    def test_compact_offset_map_matches_full_map(self):
        """Compact maps must return every offset the canonical per-character
        map would, including markers and Unicode fold expansions."""
        cases = [
            "Alpha BETA",
            "`Alpha` and **BETA**",
            "***leading markers",
            "middle `**markers**` then tail",
            "Straße` expansion falls back",
            "ß`length cancellation falls back",
            "ﬃ` multiple **İ** expansions 🎉",
        ]
        for text in cases:
            folded, full = transcript_mod._normalize_map(text)
            compact = transcript_mod._offset_map(text, folded)
            actual = [
                transcript_mod._original_index(text, i, compact)
                for i in range(len(folded))
            ]
            assert actual == full, text

    def test_unicode_expansion_avoids_full_character_map(self):
        """One fold expansion in a large entry should remain a sparse map."""
        text = "A" * 20_000 + " Straße `Need**le` tail"
        folded = transcript_mod.normalize_match_text(text)
        with patch.object(
            transcript_mod,
            "_normalize_map",
            side_effect=AssertionError("full map built"),
        ):
            compact = transcript_mod._offset_map(text, folded)
            spans = find_text_spans(text, "needle", norm=(folded, compact))
        assert isinstance(compact, transcript_mod._SparseOffsetMap)
        assert text[slice(*spans[0])] == "Need**le"

    def test_marker_only_match_avoids_full_character_map(self):
        """Case changes and markdown deletion preserve a compact mapping;
        a large matched entry must not allocate an index per character."""
        text = "A" * 20_000 + " `Need**le` tail"
        with patch.object(
            transcript_mod,
            "_normalize_map",
            side_effect=AssertionError("full map built"),
        ):
            match = next(
                iter(find_entry_matches([TranscriptEntry("tool", text)], "needle"))
            )
            spans = find_text_spans(text, "needle")
        assert text[match.offsets[0] :].startswith("Need**le")
        assert text[slice(*spans[0])] == "Need**le"

    def test_marker_only_offsets_use_compact_position_map(self):
        """Frequent hits must not rescan the whole entry for each offset."""
        text = ("`g` " * 20_000) + "tail"
        folded = transcript_mod.normalize_match_text(text)
        compact = transcript_mod._offset_map(text, folded)

        assert isinstance(compact, transcript_mod._StrippedOffsetMap)
        assert transcript_mod._original_index(text, 0, compact) == 1
        assert (
            transcript_mod._original_index(text, len(folded) - 1, compact)
            == len(text) - 1
        )

    def test_cached_hit_maps_only_first_snippet_span(self):
        """A snippet must not map every occurrence in a broad-prefix hit."""
        text = "`g` " * 20_000
        cached = transcript_mod._CachedTranscript(
            fingerprint=None,
            texts=(text,),
            roles=("tool",),
            folded_texts=(transcript_mod.normalize_match_text(text),),
            size_bytes=0,
        )
        with patch.object(
            transcript_mod, "_original_index", wraps=transcript_mod._original_index
        ) as original:
            hit = transcript_mod._hit_from_cached(cached, "g", "g")

        assert hit is not None and hit.count == 20_000
        assert original.call_count == 2

    def test_snippet_from_mapped_offset_reads_naturally(self):
        text = "the review confirmed `SELECT` only across all files"
        entries = [TranscriptEntry("assistant", text)]
        m = next(iter(find_entry_matches(entries, "select only")))
        snip = make_snippet(text, m.offsets[0], len("select only"), 10)
        assert "SELECT` only" in snip


class TestSnippets:
    def test_window_with_ellipses(self):
        text = "a" * 50 + "NEEDLE" + "b" * 50
        snip = make_snippet(text, 50, 6, context=10)
        assert snip == "…" + "a" * 10 + "NEEDLE" + "b" * 10 + "…"

    def test_no_ellipses_at_text_edges(self):
        assert make_snippet("NEEDLE tail", 0, 6, context=20) == "NEEDLE tail"


class TestSearchSession:
    def test_counts_occurrences_and_keeps_matches(self, tmp_path):
        f = tmp_path / "s.jsonl"
        write_claude_jsonl(
            f,
            [
                {"type": "user", "message": {"content": "adb devices then adb shell"}},
                {"type": "assistant", "message": {"content": "ok"}},
            ],
        )
        s = Session(id="x", provider="claude", content_path=str(f))
        r = search_session(s, "adb")
        assert r.match_count == 2 and not r.unreadable
        assert len(r.matches) == 1 and r.matches[0].entry.role == "user"
        assert r.entries is None  # not kept unless asked
        assert r.total_entries == 2  # counted even without keeping entries

    def test_keep_entries_only_when_matched(self, tmp_path):
        f = tmp_path / "s.jsonl"
        write_claude_jsonl(f, [{"type": "user", "message": {"content": "hello"}}])
        s = Session(id="x", provider="claude", content_path=str(f))
        hit = search_session(s, "hello", keep_entries=True)
        assert hit.entries is not None and hit.entries[0].text == "hello"
        miss = search_session(s, "absent", keep_entries=True)
        assert miss.match_count == 0 and miss.entries is None

    def test_unreadable_marks_result(self):
        s = Session(id="x", provider="claude", content_path="/nope")
        r = search_session(s, "q")
        assert r.unreadable and r.match_count == 0 and r.warnings


class TestSearchSessionContents:
    def test_matches_full_content_beyond_old_3mb_guard(self, tmp_path):
        """The old _jsonl_matches stopped reading after 3MB; the service
        must search the complete transcript."""
        f = tmp_path / "big.jsonl"
        filler = json.dumps({"type": "user", "message": {"content": "x" * 3_200_000}})
        tail = json.dumps(
            {"type": "user", "message": {"content": "the platypus appears"}}
        )
        f.write_text(filler + "\n" + tail + "\n")
        sessions = [Session(id="big", provider="claude", content_path=str(f))]
        assert search_session_contents(sessions, "platypus") == {"big"}

    def test_matches_previously_truncated_tool_output(self, tmp_path):
        f = tmp_path / "s.jsonl"
        write_claude_jsonl(
            f,
            [
                {
                    "type": "user",
                    "message": {
                        "content": [
                            {
                                "type": "tool_result",
                                "content": "pad " * 200 + "kookaburra",
                            }
                        ]
                    },
                },
            ],
        )
        sessions = [Session(id="t1", provider="claude", content_path=str(f))]
        assert search_session_contents(sessions, "kookaburra") == {"t1"}

    def test_blank_query_and_unreadable_sessions(self, tmp_path):
        sessions = [Session(id="x", provider="claude", content_path="/nope")]
        assert search_session_contents(sessions, "") == set()
        assert search_session_contents(sessions, "q") == set()

    def test_cache_reuses_parse_and_preserves_hit_parity(self, tmp_path):
        f = tmp_path / "s.jsonl"
        write_claude_jsonl(
            f,
            [
                {"type": "user", "message": {"content": "Alpha platypus"}},
                {"type": "assistant", "message": {"content": "Straße"}},
            ],
        )
        sessions = [Session(id="s", provider="claude", content_path=str(f))]
        cache = ContentSearchCache()
        expected = {
            query: search_session_contents(sessions, query)
            for query in ("alpha", "platypus", "STRASSE", "absent")
        }
        with patch.object(
            transcript_mod, "iter_entries", wraps=transcript_mod.iter_entries
        ) as spy:
            for query, baseline in expected.items():
                cached = search_session_contents(sessions, query, cache=cache)
                assert cached == baseline
            assert spy.call_count == 1

    def test_cache_invalidates_when_transcript_changes(self, tmp_path):
        f = tmp_path / "s.jsonl"
        write_claude_jsonl(
            f,
            [
                {"type": "user", "message": {"content": "first phrase"}},
            ],
        )
        sessions = [Session(id="s", provider="claude", content_path=str(f))]
        cache = ContentSearchCache()
        assert search_session_contents(sessions, "second", cache=cache) == set()
        with f.open("a") as out:
            out.write(
                json.dumps({"type": "user", "message": {"content": "second phrase"}})
                + "\n"
            )
        assert search_session_contents(sessions, "second", cache=cache) == {"s"}

    def test_cache_stays_within_memory_budget(self, tmp_path):
        sessions = []
        for i in range(3):
            f = tmp_path / f"s{i}.jsonl"
            write_claude_jsonl(
                f, [{"type": "user", "message": {"content": "needle " + "x" * 500}}]
            )
            sessions.append(Session(id=f"s{i}", provider="claude", content_path=str(f)))
        cache = ContentSearchCache(max_bytes=700)
        assert search_session_contents(sessions, "needle", cache=cache) == {
            "s0",
            "s1",
            "s2",
        }
        assert cache.size_bytes <= 700
        assert cache.entry_count < len(sessions)

    def test_cache_declines_oversize_entry_before_folding(self):
        """A full cache must not normalize text it cannot possibly admit."""
        cache = ContentSearchCache(max_bytes=64)
        session = Session(id="large", provider="claude")
        with patch.object(
            transcript_mod,
            "normalize_match_text",
            wraps=transcript_mod.normalize_match_text,
        ) as fold:
            admitted = cache._admit(session, ("needle " + "x" * 500,), ("user",))
        assert admitted is None
        assert fold.call_count == 0
        assert cache.entry_count == 0


class TestFindTextSpans:
    def test_plain_spans(self):
        assert find_text_spans("alpha beta alpha", "alpha") == [(0, 5), (11, 16)]

    def test_case_insensitive(self):
        assert find_text_spans("Alpha", "ALPHA") == [(0, 5)]

    def test_markdown_span_covers_original_extent(self):
        text = "We should use `SELECT` only here"
        spans = find_text_spans(text, "select only")
        assert len(spans) == 1
        start, end = spans[0]
        assert text[start:end] == "SELECT` only"

    def test_blank_query_or_text(self):
        assert find_text_spans("text", "   ") == []
        assert find_text_spans("", "q") == []

    def test_precomputed_norm_gives_same_spans(self):
        text = "use `prompt grab` twice: prompt grab"
        norm = transcript_mod._normalize_map(text)
        assert find_text_spans(text, "prompt grab", norm) == find_text_spans(
            text, "prompt grab"
        )


class TestSearchSessionHits:
    def _sessions(self, tmp_path):
        f = tmp_path / "s.jsonl"
        write_claude_jsonl(
            f,
            [
                {"type": "user", "message": {"content": "run the prompt grab command"}},
                {
                    "type": "assistant",
                    "message": {"content": "prompt grab noted; prompt grab again"},
                },
            ],
        )
        return [Session(id="s", provider="claude", content_path=str(f))]

    def test_counts_all_matches_and_prefers_user_evidence(self, tmp_path):
        hit = search_session_hits(self._sessions(tmp_path), "prompt grab")["s"]
        assert hit.count == 3
        assert hit.role == "user"
        assert hit.match == "prompt grab"
        assert hit.before.endswith("run the ")
        assert hit.after.startswith(" command")

    def test_absent_or_blank_query_yields_no_hits(self, tmp_path):
        sessions = self._sessions(tmp_path)
        assert search_session_hits(sessions, "wombat") == {}
        assert search_session_hits(sessions, "   ") == {}

    def test_cached_hits_agree_with_uncached(self, tmp_path):
        sessions = self._sessions(tmp_path)
        cold = search_session_hits(sessions, "prompt grab")
        cache = ContentSearchCache()
        first = search_session_hits(sessions, "prompt grab", cache=cache)
        with patch.object(
            transcript_mod, "iter_entries", wraps=transcript_mod.iter_entries
        ) as spy:
            warm = search_session_hits(sessions, "prompt grab", cache=cache)
            assert spy.call_count == 0
        assert cold == first == warm

    def test_markdown_formatted_match_keeps_original_snippet(self, tmp_path):
        f = tmp_path / "md.jsonl"
        write_claude_jsonl(
            f,
            [
                {
                    "type": "assistant",
                    "message": {"content": "use the **prompt grab** helper"},
                },
            ],
        )
        sessions = [Session(id="md", provider="claude", content_path=str(f))]
        hit = search_session_hits(sessions, "prompt grab")["md"]
        assert hit.count == 1
        assert hit.role == "assistant"
        assert "prompt grab" in hit.match


# ---------------------------------------------------------------------------
# Bulk search (search_sessions): provider-aware retrieval + raw prefilter
# ---------------------------------------------------------------------------


def make_opencode_db_multi(home: Path, texts_by_session: dict[str, list[str]]) -> None:
    """opencode DB with one user text part per string, per session."""
    db_path = home / ".local" / "share" / "opencode" / "opencode.db"
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path))
    conn.execute(
        "CREATE TABLE message (id TEXT PRIMARY KEY, session_id TEXT, "
        "time_created INTEGER, time_updated INTEGER, data TEXT)"
    )
    conn.execute(
        "CREATE TABLE part (id TEXT PRIMARY KEY, message_id TEXT, "
        "session_id TEXT, time_created INTEGER, time_updated INTEGER, "
        "data TEXT)"
    )
    n = 0
    for sid, texts in texts_by_session.items():
        for text in texts:
            n += 1
            conn.execute(
                "INSERT INTO message VALUES (?, ?, ?, ?, ?)",
                (f"m{n}", sid, n, n, json.dumps({"role": "user"})),
            )
            conn.execute(
                "INSERT INTO part VALUES (?, ?, ?, ?, ?, ?)",
                (
                    f"p{n}",
                    f"m{n}",
                    sid,
                    n,
                    n,
                    json.dumps({"type": "text", "text": text}),
                ),
            )
    conn.commit()
    conn.close()


def _result_view(r: SessionSearchResult):
    return (
        r.session.id,
        r.unreadable,
        r.match_count,
        r.total_entries,
        [(m.entry_index, m.query, m.offsets) for m in r.matches],
        None if r.entries is None else [e.text for e in r.entries],
    )


class TestPrefilterSafety:
    def test_safe_phrases_fold_to_needles(self):
        pf = transcript_mod._prefilter_needles
        assert pf("Hello World") == ["hello world"]
        assert pf(["alpha", " Beta Gamma "]) == ["alpha", "beta gamma"]
        # Markdown markers are stripped before the safety check, matching
        # canonical normalization.
        assert pf("semi;colons_and*stars") == ["semi;colons_andstars"]
        assert pf("`select` only") == ["select only"]

    def test_raw_scan_tolerates_markdown_interruptions(self):
        rm = transcript_mod._raw_text_may_match
        assert rm('{"t": "is `SELECT` only"}', ["select only"])
        assert rm('{"t": "**bold** claim"}', ["bold claim"])
        assert not rm('{"t": "select nothing"}', ["select only"])

    def test_file_scan_tolerates_markdown_interruptions(self, tmp_path):
        f = tmp_path / "raw.jsonl"
        f.write_text('{"t": "statement is `SELECT` only here"}\n')
        assert transcript_mod._file_may_match(f, ["select only"], "claude")
        f2 = tmp_path / "raw2.jsonl"
        f2.write_text('{"t": "select something else entirely"}\n')
        assert not transcript_mod._file_may_match(f2, ["select only"], "claude")

    def test_unsafe_phrases_force_full_scan(self):
        pf = transcript_mod._prefilter_needles
        unsafe = [
            "error 404",  # digits: json.dumps reformats numbers (1e2 -> 100.0)
            "a.b",
            "x+y",
            "v,w",
            "k: v",  # float repr / dumps separators
            'quo"te',
            "back\\slash",  # JSON string escaping
            "call(",
            "state[",
            "obj{",  # formatter/structural synthesis
            "who?",  # "?" placeholder for missing names
            "café",  # non-ASCII: may be \uXXXX-escaped raw
            "tab\tchar",
            "new\nline",  # control chars are escaped raw
            "init",
            "inf",
            "e",  # inside "Infinity" / float repr "e"
        ]
        for phrase in unsafe:
            assert pf(phrase) is None, phrase
        # one unsafe phrase disables prefiltering for the whole set
        assert pf(["fine", "err 12"]) is None
        assert pf(["", "  "]) is None

    def test_fold_risk_chars_exhaustive(self):
        """_FOLD_RISK_CHARS must be exactly the codepoints whose casefold
        contains ASCII, for the Unicode tables this Python ships."""
        risky = {
            chr(cp)
            for cp in range(0x80, 0x110000)
            if any(c.isascii() for c in chr(cp).casefold())
        }
        assert set(transcript_mod._FOLD_RISK_CHARS) == risky


class TestSearchSessionsBulk:
    def _mixed_sessions(self, tmp_path) -> list[Session]:
        """One searchable corpus per provider path (file, DB, unreadable)."""
        claude_hit = tmp_path / "c1.jsonl"
        write_claude_jsonl(
            claude_hit,
            [
                {"type": "user", "message": {"content": "the kangaroo hops"}},
                {"type": "assistant", "message": {"content": "noted"}},
            ],
        )
        claude_miss = tmp_path / "c2.jsonl"
        write_claude_jsonl(
            claude_miss,
            [
                {"type": "user", "message": {"content": "nothing relevant"}},
            ],
        )
        codex = tmp_path / "x1.jsonl"
        write_claude_jsonl(
            codex,
            [
                {
                    "type": "event_msg",
                    "payload": {
                        "type": "user_message",
                        "message": "codex kangaroo question",
                    },
                },
            ],
        )
        make_opencode_db_multi(
            tmp_path,
            {
                "oc1": ["opencode kangaroo kangaroo", "second entry"],
                "oc2": ["no marsupials here"],
            },
        )
        return [
            Session(id="cl-hit", provider="claude", content_path=str(claude_hit)),
            Session(id="oc1", provider="opencode"),
            Session(id="cl-miss", provider="claude", content_path=str(claude_miss)),
            Session(id="xc1", provider="codex", content_path=str(codex)),
            Session(id="oc2", provider="opencode"),
            Session(id="oc-absent", provider="opencode"),
            Session(id="cl-gone", provider="claude", content_path="/nope"),
            Session(id="odd", provider="mystery", content_path=""),
        ]

    @pytest.mark.parametrize(
        "query",
        [
            "kangaroo",  # prefilter-safe hit
            "zz-never-present",  # prefilter-safe miss
            ["kangaroo", "marsupials"],  # multi-phrase OR
            "error 404",  # unsafe -> full canonical scan
        ],
    )
    def test_matches_per_session_semantics(self, tmp_path, query):
        """search_sessions == search_session applied per session, in order."""
        sessions = self._mixed_sessions(tmp_path)
        with (
            patch("session_browser.transcript.Path.home", return_value=tmp_path),
            patch("session_browser.discovery.Path.home", return_value=tmp_path),
        ):
            for keep in (False, True):
                bulk = search_sessions(sessions, query, keep_entries=keep)
                single = [search_session(s, query, keep_entries=keep) for s in sessions]
                bulk_view = [_result_view(r) for r in bulk]
                single_view = [_result_view(r) for r in single]
                # Prefiltered-out sessions may skip the entry count, but only
                # when they have no matches at all (they never reach output).
                for b, s in zip(bulk_view, single_view, strict=True):
                    assert b[:3] == s[:3]
                    if b[2] > 0 or b[1]:
                        assert b == s
        assert [r.session.id for r in bulk] == [s.id for s in sessions]

    def test_progress_reports_scanning_then_reading(self, tmp_path):
        """Two phases, and the reading denominator counts only the sessions
        the prefilter could not rule out — a bar over the whole corpus would
        jump to nearly full instantly and then look stuck.
        """
        sessions = self._mixed_sessions(tmp_path)
        seen: list[tuple[str, int, int]] = []
        with (
            patch("session_browser.transcript.Path.home", return_value=tmp_path),
            patch("session_browser.discovery.Path.home", return_value=tmp_path),
        ):
            cache = ContentSearchCache()
            hits = cache.search_hits(
                sessions, "kangaroo", progress=lambda *a: seen.append(a)
            )
        assert hits, "fixture must produce hits for the phases to mean anything"
        assert seen[0] == ("scanning", 0, len(sessions))
        reading = [s for s in seen if s[0] == "reading"]
        assert reading, "the reading phase must be reported"
        total = reading[0][2]
        assert reading[0][1] == 0
        # Monotonic, never past the denominator.
        counts = [done for _, done, _ in reading]
        assert counts == sorted(counts)
        assert counts[-1] <= total

    def test_progress_is_off_by_default(self, tmp_path):
        """The CLI must pay nothing for a display it does not have. With no
        callback the search takes exactly the same path and answer.
        """
        sessions = self._mixed_sessions(tmp_path)
        with (
            patch("session_browser.transcript.Path.home", return_value=tmp_path),
            patch("session_browser.discovery.Path.home", return_value=tmp_path),
        ):
            plain = search_sessions(sessions, "kangaroo")
            noted: list = []
            watched = search_sessions(
                sessions, "kangaroo", progress=lambda *a: noted.append(a)
            )
        assert [_result_view(r) for r in plain] == [_result_view(r) for r in watched]
        assert noted, "the watched run should have reported something"

    def test_opencode_uses_single_connection(self, tmp_path):
        make_opencode_db_multi(
            tmp_path,
            {"oc1": ["alpha wombat"], "oc2": ["beta"], "oc3": ["gamma wombat"]},
        )
        sessions = [Session(id=f"oc{i}", provider="opencode") for i in (1, 2, 3)]
        real_connect = sqlite3.connect
        calls: list = []

        def counting(*a, **k):
            calls.append(a)
            return real_connect(*a, **k)

        with (
            patch("session_browser.discovery.Path.home", return_value=tmp_path),
            patch("session_browser.transcript.sqlite3.connect", side_effect=counting),
        ):
            results = search_sessions(sessions, "wombat")
        assert len(calls) == 1
        assert [(r.session.id, r.match_count) for r in results] == [
            ("oc1", 1),
            ("oc2", 0),
            ("oc3", 1),
        ]

    def test_opencode_prefilter_keeps_database_text_as_bytes(self, tmp_path):
        """The broad part scan must not decode every JSON row to str before
        it can reject an ASCII miss."""
        make_opencode_db_multi(
            tmp_path, {"oc1": ["alpha wombat"], "oc2": ["plain miss"]}
        )
        sessions = [
            Session(id="oc1", provider="opencode"),
            Session(id="oc2", provider="opencode"),
        ]
        seen: list[type] = []
        real_scan = transcript_mod._raw_bytes_may_match

        def checking(raw, needles):
            seen.append(type(raw))
            return real_scan(raw, needles)

        with (
            patch("session_browser.discovery.Path.home", return_value=tmp_path),
            patch.object(transcript_mod, "_raw_bytes_may_match", side_effect=checking),
        ):
            results = search_sessions(sessions, "notpresent")
        assert seen and set(seen) == {bytes}
        assert [r.match_count for r in results] == [0, 0]

    def test_missing_databases_reported_unreadable(self, tmp_path):
        sessions = [Session(id="oc1", provider="opencode")]
        with (
            patch(
                "session_browser.transcript.Path.home", return_value=tmp_path / "void"
            ),
            patch(
                "session_browser.discovery.Path.home", return_value=tmp_path / "void"
            ),
        ):
            results = search_sessions(sessions, "anything")
        assert all(r.unreadable for r in results)
        assert results[0].warnings == ["opencode database not found"]

    def test_prefilter_skips_canonical_parse_on_definite_miss(self, tmp_path):
        f = tmp_path / "s.jsonl"
        write_claude_jsonl(
            f, [{"type": "user", "message": {"content": "only pangolins here"}}]
        )
        s = Session(id="cl", provider="claude", content_path=str(f))
        with patch.object(
            transcript_mod, "search_session", wraps=transcript_mod.search_session
        ) as spy:
            miss = search_sessions([s], "kangaroo")
            assert spy.call_count == 0  # ruled out from raw bytes
            hit = search_sessions([s], "pangolins")
            assert spy.call_count == 1  # candidate -> canonical parse
        assert miss[0].match_count == 0 and not miss[0].unreadable
        assert hit[0].match_count == 1

    @pytest.mark.skipif(shutil.which("rg") is None, reason="ripgrep unavailable")
    def test_ripgrep_candidates_preserve_escape_and_casefold_hits(self, tmp_path):
        files = []
        for name, raw in [
            ("plain.jsonl", '{"message":{"content":"needle"}}\n'),
            ("escaped.jsonl", '{"message":{"content":"\\u006eeedle"}}\n'),
            ("folded.jsonl", '{"message":{"content":"NEEDLE and Straße"}}\n'),
            ("miss.jsonl", '{"message":{"content":"haystack"}}\n'),
        ]:
            path = tmp_path / name
            path.write_text(raw)
            files.append(Session(id=name, provider="claude", content_path=str(path)))

        candidates = transcript_mod._rg_candidate_paths(files, ["needle"])
        assert candidates is not None
        assert {p.name for p in candidates} == {
            "plain.jsonl",
            "escaped.jsonl",
            "folded.jsonl",
        }

    @pytest.mark.skipif(shutil.which("rg") is None, reason="ripgrep unavailable")
    def test_ripgrep_ignores_irrelevant_escapes(self, tmp_path):
        """Escapes decoding to characters that cannot take part in a match
        (ANSI, emoji surrogates, accented letters) must not flag a file;
        escapes decoding to needle or markdown-marker characters must."""
        files = []
        for name, raw in [
            ("ansi.jsonl", '{"message":{"content":"\\u001b[31mred herring"}}\n'),
            ("emoji.jsonl", '{"message":{"content":"\\ud83c\\udf89 party"}}\n'),
            ("accent.jsonl", '{"message":{"content":"caf\\u00e9 stop"}}\n'),
            ("escaped-hit.jsonl", '{"message":{"content":"\\u004eEEDLE"}}\n'),
            ("escaped-tick.jsonl", '{"message":{"content":"nee\\u0060dle"}}\n'),
        ]:
            path = tmp_path / name
            path.write_text(raw)
            files.append(Session(id=name, provider="claude", content_path=str(path)))
        candidates = transcript_mod._rg_candidate_paths(files, ["needle"])
        assert candidates is not None
        assert {p.name for p in candidates} == {
            "escaped-hit.jsonl",
            "escaped-tick.jsonl",
        }

    def test_irrelevant_escapes_skip_parse_without_ripgrep(self, tmp_path):
        """The in-process prefilter must rule out a file whose only escapes
        are irrelevant without a canonical parse, and still find matches
        hidden behind needle-character or marker-character escapes."""
        ansi = tmp_path / "ansi.jsonl"
        ansi.write_text(
            '{"type": "user", "message": {"content": "\\u001b[0m plain miss"}}\n'
        )
        ticked = tmp_path / "ticked.jsonl"
        ticked.write_text(
            '{"type": "user", "message": {"content": "the nee\\u0060dle hides"}}\n'
        )
        sessions = [
            Session(id="ansi", provider="claude", content_path=str(ansi)),
            Session(id="ticked", provider="claude", content_path=str(ticked)),
        ]
        with (
            patch.object(transcript_mod, "_rg_candidate_paths", return_value=None),
            patch.object(
                transcript_mod, "search_session", wraps=transcript_mod.search_session
            ) as spy,
        ):
            results = search_sessions(sessions, "needle")
        assert [(r.session.id, r.match_count) for r in results] == [
            ("ansi", 0),
            ("ticked", 1),
        ]
        assert spy.call_count == 1

    def test_escaped_fold_risk_char_still_matches(self, tmp_path):
        """'strasse' must match raw 'Stra\\u00dfe': the ß exists only after
        escape decoding, so the escape marker must keep the file — on both
        the ripgrep path and the in-process fallback."""
        f = tmp_path / "s.jsonl"
        f.write_text('{"type": "user", "message": {"content": "Stra\\u00dfe statt"}}\n')
        s = Session(id="cl", provider="claude", content_path=str(f))
        results = search_sessions([s], "strasse")
        assert results[0].match_count == 1
        with patch.object(transcript_mod, "_rg_candidate_paths", return_value=None):
            results = search_sessions([s], "strasse")
        assert results[0].match_count == 1

    def test_ripgrep_failure_falls_back_to_python_prefilter(self, tmp_path):
        path = tmp_path / "hit.jsonl"
        write_claude_jsonl(
            path, [{"type": "user", "message": {"content": "a wombat appears"}}]
        )
        session = Session(id="hit", provider="claude", content_path=str(path))
        with (
            patch.object(transcript_mod, "_rg_candidate_paths", return_value=None),
            patch.object(
                transcript_mod, "_file_may_match", wraps=transcript_mod._file_may_match
            ) as fallback,
        ):
            assert search_sessions([session], "wombat")[0].match_count == 1
            assert fallback.call_count == 1

    def test_unsafe_query_always_scans_canonically(self, tmp_path):
        f = tmp_path / "s.jsonl"
        write_claude_jsonl(
            f, [{"type": "user", "message": {"content": "only pangolins here"}}]
        )
        s = Session(id="cl", provider="claude", content_path=str(f))
        with patch.object(
            transcript_mod, "search_session", wraps=transcript_mod.search_session
        ) as spy:
            search_sessions([s], "kangaroo 42")  # digits: prefilter unsafe
            assert spy.call_count == 1

    def test_dumps_respacing_found_despite_compact_raw(self, tmp_path):
        """Tool-call text is json.dumps-formatted ('": "'), while the raw file
        is compact ('":"'); an unsafe query matching only the formatted text
        must still be found via the full-scan fallback."""
        f = tmp_path / "s.jsonl"
        f.write_text(
            '{"type":"assistant","message":{"content":[{"type":'
            '"tool_use","name":"Bash","input":{"command":"ls"}}]}}\n'
        )
        s = Session(id="cl", provider="claude", content_path=str(f))
        results = search_sessions([s], '"command": "ls"')
        assert results[0].match_count == 1
        assert results[0].matches[0].entry.text == 'Bash({"command": "ls"})'

    def test_escaped_unicode_raw_still_matches(self, tmp_path):
        """\\uXXXX-escaped raw text must not be prefiltered away."""
        f = tmp_path / "s.jsonl"
        f.write_text(
            '{"type": "user", "message": {"content": "the z\\u0065bra runs"}}\n'
        )
        s = Session(id="cl", provider="claude", content_path=str(f))
        results = search_sessions([s], "zebra runs")
        assert results[0].match_count == 1

    def test_literal_backslash_u_text_still_matches(self, tmp_path):
        """Text containing a literal backslash-u sequence must still be
        found: the escape-decoded haystack alone would destroy it."""
        f = tmp_path / "s.jsonl"
        write_claude_jsonl(
            f, [{"type": "user", "message": {"content": "see the \\uface marker"}}]
        )
        s = Session(id="cl", provider="claude", content_path=str(f))
        results = search_sessions([s], "uface marker")
        assert results[0].match_count == 1

    def test_codex_output_text_join_spans_parts(self, tmp_path):
        """Codex joins a message's output_text parts with no separator, so a
        phrase spanning two parts never appears contiguously in the raw."""
        f = tmp_path / "x.jsonl"
        write_claude_jsonl(
            f,
            [
                {
                    "type": "response_item",
                    "payload": {
                        "type": "message",
                        "role": "assistant",
                        "content": [
                            {"type": "output_text", "text": "KANGA"},
                            {"type": "output_text", "text": "roo"},
                        ],
                    },
                },
            ],
        )
        s = Session(id="xc", provider="codex", content_path=str(f))
        results = search_sessions([s], "kangaroo")
        assert results[0].match_count == 1
        assert results[0].matches[0].entry.text == "KANGAroo"

    def test_fold_risk_chars_survive_prefilter(self, tmp_path):
        """'strasse' must match raw 'Straße': ß is invisible to the ASCII
        byte scan and needs the fold-risk fallback."""
        f = tmp_path / "s.jsonl"
        write_claude_jsonl(
            f, [{"type": "user", "message": {"content": "Straße statt"}}]
        )
        s = Session(id="cl", provider="claude", content_path=str(f))
        results = search_sessions([s], "strasse")
        assert results[0].match_count == 1
        assert results[0].matches[0].offsets == [0]

    def test_opencode_part_prefilter_skips_parse_but_not_hits(self, tmp_path):
        make_opencode_db_multi(
            tmp_path, {"oc1": ["one Straße part"], "oc2": ["plain miss"]}
        )
        sessions = [
            Session(id="oc1", provider="opencode"),
            Session(id="oc2", provider="opencode"),
        ]
        with patch("session_browser.discovery.Path.home", return_value=tmp_path):
            results = search_sessions(sessions, "strasse")
        assert [(r.session.id, r.match_count) for r in results] == [
            ("oc1", 1),
            ("oc2", 0),
        ]

    def test_opencode_markdown_interrupted_match_survives_prefilter(self, tmp_path):
        """DB-backed rows go through _raw_text_may_match: a backticked
        phrase must not be ruled out from the raw part JSON."""
        make_opencode_db_multi(
            tmp_path,
            {
                "oc1": ["statement is `SELECT` only here"],
                "oc2": ["select nothing relevant"],
            },
        )
        sessions = [
            Session(id="oc1", provider="opencode"),
            Session(id="oc2", provider="opencode"),
        ]
        with patch("session_browser.discovery.Path.home", return_value=tmp_path):
            results = search_sessions(sessions, "select only")
        assert [(r.session.id, r.match_count) for r in results] == [
            ("oc1", 1),
            ("oc2", 0),
        ]

    @pytest.mark.skipif(shutil.which("rg") is None, reason="ripgrep unavailable")
    def test_ripgrep_candidates_tolerate_markdown_interruptions(self, tmp_path):
        """The rg query pattern must keep files where markdown markers
        interrupt the phrase, and still exclude genuine misses."""
        files = []
        for name, raw in [
            ("ticked.jsonl", '{"message":{"content":"is `SELECT` only"}}\n'),
            ("bold.jsonl", '{"message":{"content":"**SELECT** only"}}\n'),
            ("plain.jsonl", '{"message":{"content":"select only"}}\n'),
            ("miss.jsonl", '{"message":{"content":"select nothing"}}\n'),
        ]:
            path = tmp_path / name
            path.write_text(raw)
            files.append(Session(id=name, provider="claude", content_path=str(path)))
        candidates = transcript_mod._rg_candidate_paths(files, ["select only"])
        assert candidates is not None
        assert {p.name for p in candidates} == {
            "ticked.jsonl",
            "bold.jsonl",
            "plain.jsonl",
        }

    def test_markdown_interrupted_match_without_ripgrep(self, tmp_path):
        """Force the in-process fallback: a backticked phrase must still be
        found end-to-end when rg is unavailable."""
        f = tmp_path / "md.jsonl"
        write_claude_jsonl(
            f, [{"type": "user", "message": {"content": "statement is `SELECT` only."}}]
        )
        s = Session(id="md", provider="claude", content_path=str(f))
        with patch.object(transcript_mod, "_rg_candidate_paths", return_value=None):
            results = search_sessions([s], "select only")
        assert results[0].match_count == 1


# ---------------------------------------------------------------------------
# Parse phase in worker processes
# ---------------------------------------------------------------------------


class TestProcessPoolParse:
    """The process path must be an optimisation only: identical results, the
    same cancellation contract, and a silent fall back to threads wherever
    worker processes cannot be used."""

    def _corpus(self, tmp_path, n=6) -> list[Session]:
        sessions = []
        for i in range(n):
            path = tmp_path / f"p{i}.jsonl"
            write_claude_jsonl(
                path,
                [
                    {
                        "type": "user",
                        "message": {
                            "content": (
                                "a wombat appears" if i % 2 == 0 else "nothing relevant"
                            )
                        },
                    },
                    {"type": "assistant", "message": {"content": f"reply {i}"}},
                ],
            )
            sessions.append(
                Session(id=f"s{i}", provider="claude", content_path=str(path))
            )
        return sessions

    def test_process_results_identical_to_threads(self, tmp_path):
        """Real worker processes, compared against the thread path."""
        sessions = self._corpus(tmp_path)
        real = transcript_mod._probe_in_processes
        seen: list[object] = []

        def recording(*a, **k):
            out = real(*a, **k)
            seen.append(out)
            return out

        with (
            patch.object(transcript_mod, "_PROC_MIN_CANDIDATES", 2),
            patch.object(transcript_mod, "_probe_in_processes", recording),
        ):
            via_procs = search_sessions(sessions, "wombat")
        assert len(seen) == 1, "process path was not taken"
        assert seen[0] is not transcript_mod._PROC_FALLBACK, (
            "worker processes were unusable here, so this test compared the "
            "thread path against itself and proved nothing"
        )
        via_threads = search_sessions(sessions, "wombat")
        assert [_result_view(r) for r in via_procs] == [
            _result_view(r) for r in via_threads
        ]
        assert [r.session is s for r, s in zip(via_procs, sessions, strict=True)] == [
            True
        ] * len(sessions), "caller's Session object must survive"

    def test_entries_never_returned_by_process_path(self, tmp_path):
        """keep_entries=False is the only shape sent to processes, so no
        result may come back carrying entries."""
        sessions = self._corpus(tmp_path)
        with patch.object(transcript_mod, "_PROC_MIN_CANDIDATES", 2):
            results = search_sessions(sessions, "wombat")
        assert all(r.entries is None for r in results)

    def test_keep_entries_stays_on_threads(self, tmp_path):
        sessions = self._corpus(tmp_path)
        with (
            patch.object(transcript_mod, "_PROC_MIN_CANDIDATES", 2),
            patch.object(transcript_mod, "_probe_in_processes") as spy,
        ):
            results = search_sessions(sessions, "wombat", keep_entries=True)
        spy.assert_not_called()
        assert any(r.entries for r in results)

    def test_below_threshold_stays_on_threads(self, tmp_path):
        sessions = self._corpus(tmp_path)
        with (
            patch.object(transcript_mod, "_PROC_MIN_CANDIDATES", len(sessions) + 1),
            patch.object(transcript_mod, "_probe_in_processes") as spy,
        ):
            assert search_sessions(sessions, "wombat")[0].match_count == 1
        spy.assert_not_called()

    def test_missing_candidate_scan_stays_on_threads(self, tmp_path):
        """Without ripgrep the parent would have to run the expensive
        in-process prefilter serially to partition, so threads keep the job."""
        sessions = self._corpus(tmp_path)
        with (
            patch.object(transcript_mod, "_PROC_MIN_CANDIDATES", 2),
            patch.object(transcript_mod, "_rg_candidate_paths", return_value=None),
            patch.object(transcript_mod, "_probe_in_processes") as spy,
        ):
            results = search_sessions(sessions, "wombat")
        spy.assert_not_called()
        assert sum(r.match_count for r in results) == 3

    def test_unusable_pool_falls_back_silently(self, tmp_path):
        """An environment that cannot spawn must still answer, via threads."""
        sessions = self._corpus(tmp_path)
        expected = [_result_view(r) for r in search_sessions(sessions, "wombat")]
        with (
            patch.object(transcript_mod, "_PROC_MIN_CANDIDATES", 2),
            patch.object(transcript_mod, "_process_pool_usable", return_value=False),
        ):
            got = search_sessions(sessions, "wombat")
        assert [_result_view(r) for r in got] == expected

    def test_broken_pool_falls_back_to_threads(self, tmp_path):
        sessions = self._corpus(tmp_path)
        expected = [_result_view(r) for r in search_sessions(sessions, "wombat")]
        with (
            patch.object(transcript_mod, "_PROC_MIN_CANDIDATES", 2),
            patch.object(
                transcript_mod,
                "ProcessPoolExecutor",
                side_effect=OSError("no processes for you"),
            ),
        ):
            got = search_sessions(sessions, "wombat")
        assert [_result_view(r) for r in got] == expected

    def test_pool_rejected_when_main_module_is_not_a_file(self):
        """Both start methods re-import __main__ in the child, so a __main__
        that is not a readable file would kill every worker."""
        import sys

        main = sys.modules["__main__"]
        original = getattr(main, "__file__", None)
        try:
            main.__file__ = "<stdin>"
            assert transcript_mod._process_pool_usable() is False
            del main.__file__
            assert transcript_mod._process_pool_usable() is False
        finally:
            if original is None:
                if hasattr(main, "__file__"):
                    del main.__file__
            else:
                main.__file__ = original

    def test_cancelled_process_search_reports_every_session(self, tmp_path):
        """Cancellation after classification must still yield one result per
        input session — the thread path's contract — not a short list."""
        sessions = self._corpus(tmp_path)
        calls = {"n": 0}

        def cancelled() -> bool:
            # False through the classification loop, True once parsing starts.
            calls["n"] += 1
            return calls["n"] > len(sessions)

        with patch.object(transcript_mod, "_PROC_MIN_CANDIDATES", 2):
            results = search_sessions(sessions, "wombat", cancelled=cancelled)
        assert len(results) == len(sessions)
        assert [r.session.id for r in results] == [s.id for s in sessions]
        # Whatever did complete must agree with the uncancelled answer; a
        # session is parsed whole or reported empty, never partially counted.
        truth = {
            r.session.id: r.match_count for r in search_sessions(sessions, "wombat")
        }
        for r in results:
            assert r.match_count in (0, truth[r.session.id])

    def test_unreadable_sessions_report_identically_via_processes(self, tmp_path):
        """A worker must reproduce the thread path's unreadable warnings
        verbatim: same flag, same message text."""
        sessions = self._corpus(tmp_path)
        broken = [
            Session(id="gone", provider="claude", content_path="/nope.jsonl"),
            Session(id="nopath", provider="claude", content_path=""),
            Session(id="dir", provider="codex", content_path=str(tmp_path)),
        ]
        corpus = sessions + broken
        with patch.object(transcript_mod, "_PROC_MIN_CANDIDATES", 2):
            via_procs = search_sessions(corpus, "wombat")
        via_threads = search_sessions(corpus, "wombat")
        assert [(r.session.id, r.unreadable, r.warnings) for r in via_procs] == [
            (r.session.id, r.unreadable, r.warnings) for r in via_threads
        ]
        assert [r.unreadable for r in via_procs][-3:] == [True, True, True]


class TestLineageIds:
    """A conversation's ids across resume forks. Both multiplexer
    integrations recognise a live terminal by these, so they live here rather
    than beside either one."""

    def test_no_transcript_returns_just_the_selected_id(self):
        assert lineage_ids("claude", "abc", "") == {"abc"}
        assert lineage_ids("claude", "abc", "/nonexistent/file.jsonl") == {"abc"}

    def test_collects_top_level_ancestor_ids(self, tmp_path):
        f = tmp_path / "s.jsonl"
        f.write_text(
            '{"type":"assistant","sessionId":"new","session_id":"old"}\n'
            '{"type":"assistant","sessionId":"new","session_id":"older"}\n'
            '{"type":"user","sessionId":"new"}\n'
        )
        assert lineage_ids("claude", "new", str(f)) == {"new", "old", "older"}

    def test_ignores_ids_mentioned_inside_message_content(self, tmp_path):
        # A conversation *talking about* another session id (e.g. a pasted
        # resume command) must not make that conversation look related.
        f = tmp_path / "s.jsonl"
        f.write_text(
            '{"type":"user","sessionId":"new","message":'
            '{"content":"run claude --resume other, '
            'its \\"session_id\\" is other"}}\n'
        )
        assert lineage_ids("claude", "new", str(f)) == {"new"}

    def test_tolerates_malformed_lines(self, tmp_path):
        f = tmp_path / "s.jsonl"
        f.write_text(
            'not json "session_id" here\n'
            '{"session_id":"old"}\n'
            '["session_id","list-not-dict"]\n'
        )
        assert lineage_ids("claude", "new", str(f)) == {"new", "old"}

    def test_opencode_content_path_is_never_read(self, tmp_path, monkeypatch):
        """Opencode's ``content_path`` is the shared multi-gigabyte
        ``opencode.db``, not this session's transcript. Reading it froze the
        ``t`` handoff for seconds, so the file must not be opened at all —
        asserted by making any open fail rather than by checking the result,
        which a short fixture would satisfy either way."""
        db = tmp_path / "opencode.db"
        db.write_text('{"session_id":"old"}\n')

        def explode(*args, **kwargs):
            raise AssertionError("opencode content_path must not be opened")

        monkeypatch.setattr("builtins.open", explode)
        assert lineage_ids("opencode", "new", str(db)) == {"new"}

    def test_codex_transcript_is_still_scanned(self, tmp_path):
        """Codex keeps a per-session JSONL, so it stays on the scanned side of
        the guard and its behaviour is unchanged."""
        f = tmp_path / "rollout.jsonl"
        f.write_text('{"session_id":"old"}\n')
        assert lineage_ids("codex", "new", str(f)) == {"new", "old"}
