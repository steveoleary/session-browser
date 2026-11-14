"""A Codex thread that was rewound, read as one session across its files.

Codex's Esc-Esc rewind starts a new rollout for the same thread,
``rollout-<ts>-<thread>_<segment>.jsonl``, holding only what was written
after the rewind. Its first line's ``history_base`` names the previous file
and the ordinal and byte offset where the history it continues from ends; the
previous file keeps the turns the rewind undid. Every text here is invented.
"""

from __future__ import annotations

import json
import sqlite3
from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import patch

import pytest

import session_browser.transcript as transcript_mod
from session_browser import discovery
from session_browser.discovery import Session, codex_segment_key, scan_codex
from session_browser.transcript import (
    codex_fork_base,
    entry_label,
    load_transcript,
    render_text,
    search_sessions,
)

THREAD = "019d7a00-0000-7000-8000-000000000001"
SEG1 = "019d7a05-0000-7000-8000-000000000002"
SEG2 = "019d7a0a-0000-7000-8000-000000000003"


def _record(ordinal: int | None, kind: str, text: str = "", **payload) -> dict:
    obj: dict = {"timestamp": f"2026-04-10T13:00:{ordinal or 0:02d}Z"}
    if ordinal is not None:
        obj["ordinal"] = ordinal
    if kind == "session_meta":
        obj.update(type="session_meta", payload={"id": THREAD, "cwd": "/p", **payload})
    else:
        obj.update(type="event_msg", payload={"type": kind, "message": text})
    return obj


def _write(path: Path, records: list[dict]) -> list[int]:
    """Write *records* as JSONL; return each record's byte offset."""
    offsets, data = [], b""
    for r in records:
        offsets.append(len(data))
        data += json.dumps(r).encode() + b"\n"
    path.write_bytes(data)
    return offsets


def _link(previous: str, ordinal: int, offset: int) -> dict:
    return {
        "history_base": {
            "thread_id": previous,
            "end_ordinal_exclusive": ordinal,
            "end_byte_offset": offset,
        }
    }


def write_rewound_thread(day_dir: Path, *, seg2_from: str = SEG1) -> list[Path]:
    """Three files: a first file, a rewind from it, a rewind from that.

    Each earlier file keeps an undone turn past its boundary. *seg2_from*
    names the file the second rewind continues from."""
    day_dir.mkdir(parents=True, exist_ok=True)
    base = day_dir / f"rollout-2026-04-10T14-00-00-{THREAD}.jsonl"
    offsets = _write(
        base,
        [
            _record(0, "session_meta"),
            _record(1, "user_message", "plan the kookaburra survey"),
            _record(2, "agent_message", "a survey plan"),
            _record(3, "user_message", "count the wombats twice"),
            _record(4, "agent_message", "counting wombats twice"),
            _record(5, "turn_aborted"),
        ],
    )
    seg1 = day_dir / f"rollout-2026-04-10T14-05-00-{THREAD}_{SEG1}.jsonl"
    seg1_offsets = _write(
        seg1,
        [
            _record(3, "session_meta", **_link(THREAD, 3, offsets[3])),
            _record(4, "user_message", "count the wombats once"),
            _record(5, "agent_message", "one wombat"),
            _record(6, "user_message", "an abandoned numbat idea"),
            _record(7, "turn_aborted"),
        ],
    )
    # Named an hour EARLIER than seg1, as the local-time name of a file made
    # just after the clocks go back would be. Its segment id still sorts last.
    seg2 = day_dir / f"rollout-2026-04-10T13-10-00-{THREAD}_{SEG2}.jsonl"
    if seg2_from == SEG1:
        link = _link(SEG1, 6, seg1_offsets[3])
    else:
        link = _link(THREAD, 3, offsets[3])
    _write(
        seg2,
        [
            _record(6, "session_meta", **link),
            _record(7, "user_message", "the final echidna question"),
            _record(8, "agent_message", "an echidna answer"),
        ],
    )
    return [base, seg1, seg2]


def rewound_session(files: list[Path]) -> Session:
    s = Session(id=THREAD, provider="codex", content_path=str(files[-1]))
    s.segments = tuple(str(p) for p in files[:-1])
    return s


def uuid7_start(value: str) -> str:
    ms = int(value[:8] + value[9:13], 16)
    return (
        datetime.fromtimestamp(ms / 1000, tz=UTC).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3]
        + "Z"
    )


@pytest.fixture
def home(tmp_path):
    with patch("session_browser.discovery.Path.home", return_value=tmp_path):
        yield tmp_path


def _day(home: Path) -> Path:
    return home / ".codex" / "sessions" / "2026" / "04" / "10"


def _write_index(home: Path, rows: list[dict]) -> None:
    conn = sqlite3.connect(str(home / ".codex" / "state_5.sqlite"))
    conn.execute(
        "CREATE TABLE threads (id TEXT PRIMARY KEY, rollout_path TEXT NOT NULL, "
        "cwd TEXT NOT NULL, git_branch TEXT, git_origin_url TEXT, source TEXT, "
        "first_user_message TEXT NOT NULL DEFAULT '', created_at_ms INTEGER, "
        "updated_at_ms INTEGER, archived INTEGER NOT NULL DEFAULT 0)"
    )
    for r in rows:
        conn.execute(
            "INSERT INTO threads (id, rollout_path, cwd, first_user_message, "
            "created_at_ms, updated_at_ms, archived) VALUES (?, ?, '/p', ?, ?, ?, ?)",
            (
                r["id"],
                r["rollout_path"],
                r.get("summary", ""),
                1775829000000,
                1775829600000,
                r.get("archived", 0),
            ),
        )
    conn.commit()
    conn.close()


class TestDiscovery:
    def test_segment_key_orders_by_creation_not_by_local_time_name(self, tmp_path):
        files = write_rewound_thread(tmp_path)
        by_name = sorted(files, key=lambda p: p.name)
        assert by_name[0] == files[2]  # the misleading local-time name
        assert sorted(files, key=codex_segment_key) == files

    def test_file_scan_reads_a_rewound_thread_as_one_session(self, home):
        files = write_rewound_thread(_day(home))
        sessions = scan_codex()
        assert [s.id for s in sessions] == [THREAD]
        s = sessions[0]
        assert s.content_path == str(files[2])
        assert s.segments == (str(files[0]), str(files[1]))
        # The thread's own first prompt, not the one sent after a rewind.
        assert s.summary == "plan the kookaburra survey"
        assert s.created_at == uuid7_start(THREAD)

    def test_index_path_attaches_earlier_files_without_opening_them(
        self, home, monkeypatch
    ):
        files = write_rewound_thread(_day(home))
        _write_index(
            home,
            [{"id": THREAD, "rollout_path": str(files[2]), "summary": "from index"}],
        )

        def no_reads(*_a, **_k):
            raise AssertionError("an earlier file was read during discovery")

        monkeypatch.setattr(discovery, "_codex_session_from_file", no_reads)
        sessions = scan_codex()
        assert [s.id for s in sessions] == [THREAD]
        s = sessions[0]
        assert s.summary == "from index"
        assert s.segments == (str(files[0]), str(files[1]))
        # The index dates a rewound thread from its newest file.
        assert s.created_at == uuid7_start(THREAD)

    def test_an_index_behind_the_latest_rewind_still_reads_the_newest_file(self, home):
        files = write_rewound_thread(_day(home))
        _write_index(home, [{"id": THREAD, "rollout_path": str(files[1])}])
        (s,) = scan_codex()
        assert s.content_path == str(files[2])
        assert s.segments == (str(files[0]), str(files[1]))

    def test_an_index_behind_the_first_rewind_still_finds_it(self, home):
        files = write_rewound_thread(_day(home))
        _write_index(home, [{"id": THREAD, "rollout_path": str(files[0])}])
        (s,) = scan_codex()
        assert s.content_path == str(files[2])

    def test_an_archived_rewound_thread_leaves_no_session_behind(self, home):
        files = write_rewound_thread(_day(home))
        _write_index(
            home, [{"id": THREAD, "rollout_path": str(files[2]), "archived": 1}]
        )
        assert scan_codex() == []


class TestParse:
    def _rows(self, session: Session) -> list[tuple[str, str, bool]]:
        t = load_transcript(session)
        return [
            (e.role, e.text, bool(e.metadata and e.metadata.get("rewound")))
            for e in t.entries
        ]

    def test_every_file_in_the_order_it_happened_with_undone_turns_marked(
        self, tmp_path
    ):
        session = rewound_session(write_rewound_thread(tmp_path))
        assert self._rows(session) == [
            ("user", "plan the kookaburra survey", False),
            ("assistant", "a survey plan", False),
            ("user", "count the wombats twice", True),
            ("assistant", "counting wombats twice", True),
            ("user", "count the wombats once", False),
            ("assistant", "one wombat", False),
            ("user", "an abandoned numbat idea", True),
            ("user", "the final echidna question", False),
            ("assistant", "an echidna answer", False),
        ]
        assert load_transcript(session).warnings == []

    def test_an_undone_turn_says_so_wherever_it_is_rendered(self, tmp_path):
        t = load_transcript(rewound_session(write_rewound_thread(tmp_path)))
        undone = t.entries[2]
        assert entry_label(undone) == "User (rewound)"
        assert "User (rewound): count the wombats twice" in render_text(t)
        # Nothing is added to the text itself: the raw file must still
        # contain every phrase a parse can match, or the prefilter is unsound.
        assert undone.text == "count the wombats twice"

    def test_a_record_past_the_boundary_without_an_ordinal_is_still_undone(
        self, tmp_path
    ):
        files = write_rewound_thread(tmp_path)
        lines = files[0].read_text().splitlines()
        record = json.loads(lines[4])
        del record["ordinal"]
        lines[4] = json.dumps(record)
        files[0].write_text("\n".join(lines) + "\n")
        rows = self._rows(rewound_session(files))
        assert ("assistant", "counting wombats twice", True) in rows

    def test_the_byte_offset_wins_over_a_stated_ordinal(self, tmp_path):
        files = write_rewound_thread(tmp_path)
        lines = files[1].read_text().splitlines()
        meta = json.loads(lines[0])
        meta["payload"]["history_base"]["end_ordinal_exclusive"] = 2
        lines[0] = json.dumps(meta)
        files[1].write_text("\n".join(lines) + "\n")
        session = rewound_session(files)
        rows = self._rows(session)
        assert ("assistant", "a survey plan", False) in rows
        assert any("disagrees" in w for w in load_transcript(session).warnings)

    def test_a_file_no_rewind_link_reaches_is_read_as_undone_entire(self, tmp_path):
        # The second rewind went back past the start of the first's file.
        session = rewound_session(write_rewound_thread(tmp_path, seg2_from=THREAD))
        rows = self._rows(session)
        assert ("user", "count the wombats once", True) in rows
        assert ("assistant", "one wombat", True) in rows
        assert any("no rewind link" in w for w in load_transcript(session).warnings)

    def test_a_missing_earlier_file_costs_only_its_own_history(self, tmp_path):
        files = write_rewound_thread(tmp_path)
        session = rewound_session(files)
        files[0].unlink()
        session.segments = (str(files[1]),)
        t = load_transcript(session)
        assert [e.text for e in t.entries][-2:] == [
            "the final echidna question",
            "an echidna answer",
        ]
        assert any("unavailable" in w for w in t.warnings)

    def test_a_rewind_is_not_a_fork(self, tmp_path):
        session = rewound_session(write_rewound_thread(tmp_path))
        assert codex_fork_base(session) is None
        # Not even when only the newest file is known.
        alone = Session(id=THREAD, provider="codex", content_path=session.content_path)
        assert codex_fork_base(alone) is None

    def test_a_fork_that_was_later_rewound_is_still_a_fork(self, tmp_path):
        files = write_rewound_thread(tmp_path)
        lines = files[0].read_text().splitlines()
        meta = json.loads(lines[0])
        meta["payload"]["history_base"] = {
            "thread_id": "019d0000-0000-7000-8000-00000000000f",
            "end_byte_offset": 120,
        }
        lines[0] = json.dumps(meta)
        files[0].write_text("\n".join(lines) + "\n")
        session = rewound_session(files)
        assert codex_fork_base(session) == ("019d0000-0000-7000-8000-00000000000f", 120)


class TestSearch:
    def test_a_phrase_said_only_before_the_rewind_is_found(self, tmp_path):
        session = rewound_session(write_rewound_thread(tmp_path))
        (r,) = search_sessions([session], "kookaburra survey")
        assert r.match_count == 1

    def test_found_without_ripgrep_too(self, tmp_path, monkeypatch):
        session = rewound_session(write_rewound_thread(tmp_path))
        monkeypatch.setattr(transcript_mod, "_rg_candidate_paths", lambda *_a: None)
        (r,) = search_sessions([session], "kookaburra survey")
        assert r.match_count == 1

    def test_a_phrase_in_no_file_rules_the_thread_out_unparsed(self, tmp_path):
        session = rewound_session(write_rewound_thread(tmp_path))
        (r,) = search_sessions([session], "platypus")
        assert r.match_count == 0
        assert r.total_entries == 0  # ruled out by the prefilter, never parsed

    def test_a_worker_process_reads_the_whole_chain(self, tmp_path):
        session = rewound_session(write_rewound_thread(tmp_path))
        count, matches, _w, unreadable, total = transcript_mod._parse_worker(
            (session, "kookaburra survey")
        )
        assert (count, unreadable, total) == (1, False, 9)
        assert matches[0].entry_index == 0
