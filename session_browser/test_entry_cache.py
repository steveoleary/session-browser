"""The entry cache: parsed transcripts kept on disk, and search over them.

Two things are tested here and they fail differently. The cache itself must
give back exactly what it was handed or nothing at all, because a wrong entry
is served silently where a missing one only costs a parse. And everything
built on it -- retrieval, search, the words wave, deletion -- must answer as
it does with the cache switched off, which is the path that existed first and
stays the reference.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from unittest.mock import patch

import pytest

import session_browser.transcript as transcript_mod
from session_browser import deletion, entry_cache
from session_browser.cli import run_cli
from session_browser.discovery import Session, scan_claude
from session_browser.test_transcript import (
    _result_view,
    make_opencode_db_multi,
    write_claude_jsonl,
)
from session_browser.transcript import (
    TranscriptEntry,
    count_entries,
    iter_entries,
    search_each,
    search_session_word_hits,
    search_sessions,
)


@pytest.fixture(autouse=True)
def cache_home(tmp_path, monkeypatch):
    """A cache of this test's own, so nothing another test parsed is found."""
    root = tmp_path / "xdg-cache"
    monkeypatch.setenv("XDG_CACHE_HOME", str(root))
    monkeypatch.delenv("SESSION_BROWSER_CACHE", raising=False)
    return root / "session-browser" / "entries"


def _files(cache_home: Path, suffix: str = "") -> list[Path]:
    return sorted(p for p in cache_home.rglob(f"*{suffix}") if p.is_file())


def _ref(tmp_path, name="s.jsonl", body="x\n") -> entry_cache.Ref:
    source = tmp_path / name
    source.write_text(body)
    ref = entry_cache.locate("claude", "sid", str(source))
    assert ref is not None
    return ref


# Text chosen to break a careless encoding: a newline (the separator between
# entries), a NUL, markdown markers, non-ASCII, a fold-risk character, a lone
# surrogate that strict UTF-8 refuses, and nothing at all.
AWKWARD = [
    TranscriptEntry("user", "first line\nsecond line", "2026-01-01T00:00:00Z"),
    TranscriptEntry("assistant", "nul \x00 inside, `code` and **bold**"),
    TranscriptEntry("tool", "Straße → ﬁn", "", {"kind": "call", "tool": "Bash"}),
    TranscriptEntry("tool", "lone \ud83d surrogate", "", {"kind": "output", "n": 3}),
    TranscriptEntry("system", "", "2026-01-01T00:00:01Z", {"rewound": True}),
]


class TestRoundTrip:
    def test_what_is_stored_is_what_is_loaded(self, tmp_path):
        ref = _ref(tmp_path)
        assert entry_cache.store(ref, AWKWARD, ["line 3: invalid JSON, skipped"])
        entries, warnings = entry_cache.load(ref, TranscriptEntry)
        assert entries == AWKWARD
        assert warnings == ["line 3: invalid JSON, skipped"]

    def test_an_empty_transcript_round_trips(self, tmp_path):
        ref = _ref(tmp_path)
        assert entry_cache.store(ref, [], [])
        assert entry_cache.load(ref, TranscriptEntry) == ([], [])
        assert entry_cache.count(ref) == 0

    def test_count_reads_the_header_not_the_entries(self, tmp_path):
        ref = _ref(tmp_path)
        entry_cache.store(ref, AWKWARD, [])
        os.unlink(ref.text)  # the count must not need it
        assert entry_cache.count(ref) == len(AWKWARD)

    def test_the_text_file_holds_entry_text_and_nothing_else(self, tmp_path):
        """ripgrep scans this file. A role, a timestamp or a byte count in it
        would match queries nobody typed and make every search load every
        session to find that out."""
        ref = _ref(tmp_path)
        entry_cache.store(ref, AWKWARD, [])
        expected = b"".join(
            e.text.encode("utf-8", "surrogatepass") + b"\n" for e in AWKWARD
        )
        assert Path(ref.text).read_bytes() == expected

    def test_nothing_stored_is_a_miss(self, tmp_path):
        ref = _ref(tmp_path)
        assert entry_cache.load(ref, TranscriptEntry) is None
        assert entry_cache.count(ref) is None

    @pytest.mark.parametrize("damage", ["truncate-text", "drop-index", "junk-index"])
    def test_a_damaged_pair_is_a_miss_not_a_wrong_answer(self, tmp_path, damage):
        ref = _ref(tmp_path)
        entry_cache.store(ref, AWKWARD, [])
        if damage == "truncate-text":
            Path(ref.text).write_bytes(Path(ref.text).read_bytes()[:-3])
        elif damage == "drop-index":
            os.unlink(ref.index)
        else:
            Path(ref.index).write_text("{}\nnot json")
        assert entry_cache.load(ref, TranscriptEntry) is None

    def test_metadata_json_cannot_carry_is_refused_not_mangled(self, tmp_path):
        """JSON gives a tuple back as a list. Stored, that entry would load
        unequal to the one a parse yields, and nothing would say so."""
        ref = _ref(tmp_path)
        odd = [TranscriptEntry("tool", "x", "", {"pair": (1, 2)})]
        assert entry_cache.store(ref, odd, []) is False
        assert entry_cache.load(ref, TranscriptEntry) is None

    def test_an_unwritable_cache_is_not_an_error(self, tmp_path, monkeypatch):
        blocker = tmp_path / "not-a-directory"
        blocker.write_text("in the way")
        monkeypatch.setenv("XDG_CACHE_HOME", str(blocker))
        ref = _ref(tmp_path)
        assert entry_cache.store(ref, AWKWARD, []) is False
        assert entry_cache.load(ref, TranscriptEntry) is None


class TestFingerprint:
    def test_no_file_means_no_ref(self, tmp_path):
        assert entry_cache.locate("claude", "s", "") is None
        assert entry_cache.locate("claude", "s", str(tmp_path / "gone.jsonl")) is None

    def test_an_appended_file_is_a_different_version(self, tmp_path):
        source = tmp_path / "s.jsonl"
        source.write_text("one\n")
        before = entry_cache.locate("claude", "s", str(source))
        with open(source, "a") as fh:
            fh.write("two\n")
        after = entry_cache.locate("claude", "s", str(source))
        assert before.text != after.text
        # Same session, so the same stem: that is how the old version is found
        # and removed when the new one is written.
        assert before.names[0].split(".")[0] == after.names[0].split(".")[0]

    def test_a_touched_file_is_a_different_version(self, tmp_path):
        source = tmp_path / "s.jsonl"
        source.write_text("one\n")
        before = entry_cache.locate("claude", "s", str(source))
        st = source.stat()
        os.utime(source, ns=(st.st_atime_ns, st.st_mtime_ns + 1_000_000_000))
        assert entry_cache.locate("claude", "s", str(source)).text != before.text

    def test_how_a_file_is_read_is_part_of_its_name(self, tmp_path):
        """The same rollout parses differently as a guardian, as a spawned
        child and as part of a rewound chain; one cached parse must never be
        served for another."""
        source = tmp_path / "rollout.jsonl"
        source.write_text("{}\n")
        earlier = tmp_path / "earlier.jsonl"
        earlier.write_text("{}\n")
        path = str(source)
        refs = {
            entry_cache.locate("codex", "t", path).text,
            entry_cache.locate("codex", "t", path, "guardian").text,
            entry_cache.locate("codex", "t", path, "thread_spawn").text,
            entry_cache.locate("codex", "t", path, None, (str(earlier),)).text,
            entry_cache.locate("codex", "other", path).text,
        }
        assert len(refs) == 5

    def test_a_segment_that_changes_changes_the_version(self, tmp_path):
        source = tmp_path / "rollout.jsonl"
        source.write_text("{}\n")
        earlier = tmp_path / "earlier.jsonl"
        earlier.write_text("{}\n")
        before = entry_cache.locate("codex", "t", str(source), None, (str(earlier),))
        earlier.write_text("{}\n{}\n")
        after = entry_cache.locate("codex", "t", str(source), None, (str(earlier),))
        assert before.text != after.text

    def test_a_changed_parser_is_a_different_version(self, tmp_path):
        """The hazard a hand-kept version number invites: yesterday's parse
        served by today's parser. The version is the parser's own source."""
        source = tmp_path / "s.jsonl"
        source.write_text("one\n")
        before = entry_cache.locate("claude", "s", str(source))
        with patch.object(entry_cache, "parser_version", return_value="another"):
            after = entry_cache.locate("claude", "s", str(source))
        assert before.text != after.text

    def test_parser_version_covers_the_parser_and_is_stable(self):
        assert "transcript.py" in entry_cache._PARSER_SOURCES
        assert "discovery.py" in entry_cache._PARSER_SOURCES
        entry_cache.parser_version.cache_clear()
        first = entry_cache.parser_version()
        entry_cache.parser_version.cache_clear()
        assert entry_cache.parser_version() == first

    def test_storing_a_new_version_removes_the_old_one(self, tmp_path, cache_home):
        source = tmp_path / "s.jsonl"
        source.write_text("one\n")
        old = entry_cache.locate("claude", "s", str(source))
        entry_cache.store(old, AWKWARD[:1], [])
        with open(source, "a") as fh:
            fh.write("two\n")
        new = entry_cache.locate("claude", "s", str(source))
        entry_cache.store(new, AWKWARD[:2], [])
        assert {str(p) for p in _files(cache_home)} == {new.text, new.index}


class TestSweep:
    def test_entries_whose_source_is_gone_are_removed(self, tmp_path, cache_home):
        kept = _ref(tmp_path, "kept.jsonl")
        gone = _ref(tmp_path, "gone.jsonl")
        entry_cache.store(kept, AWKWARD, [])
        entry_cache.store(gone, AWKWARD, [])
        os.unlink(tmp_path / "gone.jsonl")
        assert entry_cache.sweep() == 1
        assert {str(p) for p in _files(cache_home)} == {kept.text, kept.index}

    def test_a_folded_file_goes_with_the_rest(self, tmp_path, cache_home):
        gone = _ref(tmp_path, "gone.jsonl")
        entry_cache.store(gone, AWKWARD, [], b"strasse\n")
        assert len(_files(cache_home)) == 3
        os.unlink(tmp_path / "gone.jsonl")
        assert entry_cache.sweep() == 1
        assert _files(cache_home) == []

    def test_the_rationed_sweep_runs_once_per_period(self, tmp_path):
        gone = _ref(tmp_path, "gone.jsonl")
        entry_cache.store(gone, AWKWARD, [])
        os.unlink(tmp_path / "gone.jsonl")
        assert entry_cache.sweep_if_due(now=1_000_000.0) == 1
        again = _ref(tmp_path, "again.jsonl")
        entry_cache.store(again, AWKWARD, [])
        os.unlink(tmp_path / "again.jsonl")
        soon = 1_000_000.0 + entry_cache.SWEEP_EVERY - 1
        assert entry_cache.sweep_if_due(now=soon) == 0
        assert entry_cache.sweep_if_due(now=soon + 2) == 1

    def test_deleting_a_session_deletes_what_was_cached_from_it(
        self, tmp_path, cache_home
    ):
        """'Permanently delete' has to include this package's own copy of the
        session's words, and its subagents' -- the transcript being gone from
        the provider's store is not the end of it otherwise."""
        project = tmp_path / ".claude" / "projects" / "-Users-test"
        for sid in ("doomed", "kept"):
            agents = project / sid / "subagents"
            agents.mkdir(parents=True)
            line = json.dumps(
                {
                    "type": "user",
                    "message": {"role": "user", "content": f"said in {sid}"},
                    "cwd": "/Users/test/project",
                    "timestamp": "2026-04-14T10:00:00Z",
                }
            )
            (project / f"{sid}.jsonl").write_text(line + "\n")
            (agents / "agent-a1.jsonl").write_text(line + "\n")
            (agents / "agent-a1.meta.json").write_text('{"agentType": "Explore"}')
        with patch("session_browser.discovery.Path.home", return_value=tmp_path):
            sessions = scan_claude()
            for s in sessions:
                list(iter_entries(s, []))
            assert len(_files(cache_home, ".txt")) == 4
            doomed = next(s for s in sessions if s.id == "doomed")
            deletion.delete_session(deletion.plan_delete(doomed, sessions))
        left = b"".join(p.read_bytes() for p in _files(cache_home, ".txt"))
        assert b"said in doomed" not in left
        assert left.count(b"said in kept") == 2


def _claude(tmp_path, name: str, *turns: str) -> Session:
    path = tmp_path / f"{name}.jsonl"
    write_claude_jsonl(
        path,
        [
            {"type": "user" if i % 2 == 0 else "assistant", "message": {"content": t}}
            for i, t in enumerate(turns)
        ],
    )
    return Session(id=name, provider="claude", content_path=str(path))


class TestIterEntries:
    def test_a_second_read_parses_nothing_and_yields_the_same(self, tmp_path):
        path = tmp_path / "s.jsonl"
        write_claude_jsonl(
            path,
            [
                {"type": "user", "message": {"content": "hello"}},
                "this line is not json",
                {"type": "assistant", "message": {"content": "hi **there**"}},
            ],
        )
        session = Session(id="s", provider="claude", content_path=str(path))
        first_warnings: list[str] = []
        first = list(iter_entries(session, first_warnings))
        assert first_warnings, "the fixture must raise a warning to carry"
        with patch.object(transcript_mod, "_parse_entries") as parse:
            again_warnings: list[str] = []
            again = list(iter_entries(session, again_warnings))
        parse.assert_not_called()
        assert again == first
        assert again_warnings == first_warnings

    def test_a_read_abandoned_part_way_keeps_nothing(self, tmp_path, cache_home):
        """Only a reader that reached the end saw every entry there is."""
        session = _claude(tmp_path, "s", "one", "two", "three")
        stream = iter_entries(session, [])
        next(stream)
        stream.close()
        assert _files(cache_home) == []

    def test_a_transcript_that_grew_is_parsed_again(self, tmp_path):
        session = _claude(tmp_path, "s", "one", "two")
        assert len(list(iter_entries(session, []))) == 2
        with open(session.content_path, "a") as fh:
            fh.write(json.dumps({"type": "user", "message": {"content": "three"}}))
            fh.write("\n")
        assert [e.text for e in iter_entries(session, [])] == ["one", "two", "three"]

    def test_switched_off_writes_nothing(self, tmp_path, cache_home, monkeypatch):
        monkeypatch.setenv("SESSION_BROWSER_CACHE", "0")
        session = _claude(tmp_path, "s", "one", "two")
        assert len(list(iter_entries(session, []))) == 2
        search_sessions([session], "one")
        assert not cache_home.exists()

    def test_an_unreadable_session_still_says_so(self, tmp_path):
        gone = Session(id="g", provider="claude", content_path=str(tmp_path / "nope"))
        with pytest.raises(transcript_mod.TranscriptUnreadable):
            list(iter_entries(gone, []))

    def test_count_entries_agrees_with_the_entries(self, tmp_path):
        session = _claude(tmp_path, "s", "one", "two", "three")
        assert count_entries(session) == 3  # cold: reads, and caches
        with (
            patch.object(transcript_mod, "_parse_entries") as parse,
            patch.object(entry_cache, "load") as load,
        ):
            assert count_entries(session) == 3  # warm: the header alone
        parse.assert_not_called()
        load.assert_not_called()


class TestFoldedText:
    """The one kind of entry a case-insensitive scan of original text cannot
    decide, kept a second time as the matcher sees it."""

    def test_only_a_session_that_needs_one_gets_a_folded_file(
        self, tmp_path, cache_home
    ):
        plain = _claude(tmp_path, "plain", "nothing unusual", "caf\u00e9 and \u2192")
        risky = _claude(tmp_path, "risky", "Stra\u00dfe `one`", "plain reply")
        for s in (plain, risky):
            list(iter_entries(s, []))
        folds = _files(cache_home, entry_cache.FOLD_SUFFIX)
        assert len(folds) == 1
        # Normalized as the matcher normalizes -- casefolded, markers gone --
        # and only the entry that needed it.
        assert folds[0].read_bytes() == b"strasse one\n"

    def test_a_match_only_the_folded_text_holds_is_found_and_nothing_else_loaded(
        self, tmp_path
    ):
        sessions = [
            _claude(tmp_path, "risky", "Stra\u00dfe", "of\ufb01ce hours"),
            _claude(tmp_path, "other", "a street", "an office"),
        ]
        search_sessions(sessions, "warm")
        for query, expected in (("strasse", ["risky"]), ("office", ["risky", "other"])):
            with patch.object(entry_cache, "load", wraps=entry_cache.load) as load:
                results = search_sessions(sessions, query)
            assert [r.session.id for r in results if r.match_count] == expected
            assert load.call_count == len(expected)
        # A phrase neither holds: the session with the awkward characters is
        # no longer a candidate for it merely by holding them.
        with patch.object(entry_cache, "load", wraps=entry_cache.load) as load:
            assert not any(r.match_count for r in search_sessions(sessions, "wombat"))
        load.assert_not_called()


class TestWhatIsNotKept:
    def test_a_rewound_thread_read_with_a_file_missing_is_not_kept(
        self, tmp_path, cache_home
    ):
        """A fingerprint says the files have not changed, not that the read
        went well. An earlier file that could not be opened yields a warning
        and a shorter transcript; stored, that would be served for as long as
        the files stayed the same."""
        from session_browser.test_codex_rewind import (
            rewound_session,
            write_rewound_thread,
        )

        files = write_rewound_thread(tmp_path / "day")
        whole = rewound_session(files)
        full = [e.text for e in iter_entries(whole, [])]
        assert len(_files(cache_home, ".txt")) == 1  # read cleanly: kept

        # The same thread, now pointing at an earlier file that has no link.
        short = rewound_session(
            write_rewound_thread(tmp_path / "other", seg2_from=whole.id)
        )
        warnings: list[str] = []
        first = [e.text for e in iter_entries(short, warnings)]
        assert warnings, "the fixture must warn for this to test anything"
        assert len(_files(cache_home, ".txt")) == 1  # the warned read: not kept
        with patch.object(
            transcript_mod, "_parse_entries", wraps=transcript_mod._parse_entries
        ) as parse:
            assert [e.text for e in iter_entries(short, [])] == first
        assert parse.call_count == 1
        assert full  # and the clean one still loads
        with patch.object(transcript_mod, "_parse_entries") as parse:
            assert [e.text for e in iter_entries(whole, [])] == full
        parse.assert_not_called()


def _codex(tmp_path, name: str, *records: dict) -> Session:
    path = tmp_path / f"{name}.jsonl"
    path.write_text("".join(json.dumps(r) + "\n" for r in records))
    return Session(id=name, provider="codex", content_path=str(path))


def _corpus(tmp_path) -> list[Session]:
    """Sessions that each exercise one way cached text differs from raw."""
    pi = tmp_path / "pi.jsonl"
    pi.write_text(
        json.dumps(
            {
                "type": "message",
                "timestamp": "2026-01-01T00:00:00Z",
                "message": {"role": "user", "content": "pi says version 3.14.10"},
            }
        )
        + "\n"
    )
    recap = tmp_path / "recap.jsonl"
    # Claude Code's own recap: a system record the parser keeps, its settings
    # pointer stripped. Every other system subtype is still skipped.
    write_claude_jsonl(
        recap,
        [
            {"type": "user", "message": {"content": "go"}},
            {"type": "system", "subtype": "turn_duration", "durationMs": 7},
            {
                "type": "system",
                "subtype": "away_summary",
                "content": "The quokka census is done. (disable recaps in /config)",
            },
        ],
    )
    escaped = tmp_path / "escaped.jsonl"
    # Raw \\uXXXX escapes spelling "wombat", which only decoding reveals.
    escaped.write_text(
        '{"type":"user","message":{"content":"an \\u0077\\u006f\\u006d\\u0062'
        '\\u0061\\u0074 in escapes, path a\\/b"}}\n'
    )
    return [
        _claude(tmp_path, "plain", "the wombat appears", "noted, perf_budget.py"),
        _claude(tmp_path, "marked", "a `wom`**bat** in markers", "commit 3fbb4d0"),
        _claude(tmp_path, "folds", "Straße and Kelvin K", "ﬁnal ofﬁce"),
        _claude(tmp_path, "upper", "WOMBAT SHOUTED", "Error 404: Not Found"),
        _claude(tmp_path, "miss", "nothing relevant", "still nothing"),
        _claude(tmp_path, "tool", "run it", '{"command": "ls -la"}'),
        Session(id="escaped", provider="claude", content_path=str(escaped)),
        Session(id="recap", provider="claude", content_path=str(recap)),
        Session(id="pi", provider="pi", content_path=str(pi)),
        _codex(
            tmp_path,
            "joined",
            {
                "type": "response_item",
                "payload": {
                    "type": "message",
                    "role": "assistant",
                    "content": [
                        {"type": "output_text", "text": "half a wom"},
                        {"type": "output_text", "text": "bat in two parts"},
                    ],
                },
            },
        ),
        Session(id="gone", provider="claude", content_path=str(tmp_path / "nope")),
        Session(id="odd", provider="mystery", content_path=""),
    ]


QUERIES = [
    "wombat",  # plain, marked, upper-cased, escaped and joined across parts
    "WOMBAT",
    "zz-never-present",
    "perf_budget.py",  # a dot: the raw prefilter cannot take it
    "3fbb4d0",  # digits: likewise
    "error 404",
    "3.14.10",
    '"command": "ls -la"',  # quotes, a colon and a dash
    "a/b",  # a slash, escaped in the raw file
    "strasse",  # ß casefolds to ss
    "kelvin k",  # the Kelvin sign casefolds to k
    "final office",  # ligatures casefold to fi and ffi
    "straße",  # a needle no scan can take
    ["wombat", "nothing"],  # alternates
    ["3fbb4d0", "straße"],  # one scannable, one not
    "it",  # in nearly everything
    "quokka census",  # only in a recap
    "recaps in /config",  # in the raw file, stripped from the entry
]


def _answer(r):
    """What a search result says, leaving out what it is free not to know.

    A session a scan ruled out is never read, so its ``total_entries`` is 0;
    one that was read and held no match reports its real length. Which of the
    two a non-matching session gets depends on how sharp the scan was -- the
    raw prefilter lets through sessions the scan of cached text rules out --
    and search_sessions documents that no output surfaces it. Everything
    about a session that did match, or could not be read, must agree.
    """
    view = _result_view(r)
    if r.match_count == 0 and not r.unreadable:
        return (*view[:3], None, *view[4:5], None)
    return view


class TestSearchAgreesWithTheStatelessPath:
    """The cache is an optimisation of search and nothing else. Whatever the
    raw files answer with the cache off is the answer; cold, warm and without
    ripgrep must all give it, for every kind of phrase."""

    @pytest.mark.parametrize("query", QUERIES, ids=str)
    @pytest.mark.parametrize("rg", [True, False], ids=["rg", "no-rg"])
    def test_cold_and_warm_match_the_reference(self, tmp_path, monkeypatch, query, rg):
        sessions = _corpus(tmp_path)
        monkeypatch.setenv("SESSION_BROWSER_CACHE", "0")
        reference = [_answer(r) for r in search_sessions(sessions, query)]
        monkeypatch.delenv("SESSION_BROWSER_CACHE")
        patches = []
        if not rg:
            patches = [
                patch.object(
                    transcript_mod, "_rg_entry_text_candidates", return_value=None
                ),
                patch.object(transcript_mod, "_rg_candidate_paths", return_value=None),
            ]
        for p in patches:
            p.start()
        try:
            cold = [_answer(r) for r in search_sessions(sessions, query)]
            warm = [_answer(r) for r in search_sessions(sessions, query)]
            kept = [
                _answer(r) for r in search_sessions(sessions, query, keep_entries=True)
            ]
        finally:
            for p in patches:
                p.stop()
        assert cold == reference
        assert warm == reference
        monkeypatch.setenv("SESSION_BROWSER_CACHE", "0")
        assert kept == [
            _answer(r) for r in search_sessions(sessions, query, keep_entries=True)
        ]

    @pytest.mark.parametrize("query", QUERIES, ids=str)
    def test_what_a_deferred_search_defers_settles_to_the_reference(
        self, tmp_path, monkeypatch, query
    ):
        """defer=True takes the scan's word that a cached session matches
        and loads nothing. Loading what it deferred must give back the
        reference exactly, and nothing may be deferred that holds no match:
        a limited search shows a deferred session's date as a match without
        ever reading it."""
        sessions = _corpus(tmp_path)
        monkeypatch.setenv("SESSION_BROWSER_CACHE", "0")
        reference = [_answer(r) for r in search_sessions(sessions, query)]
        monkeypatch.delenv("SESSION_BROWSER_CACHE")
        search_sessions(sessions, "anything")  # the first search fills the cache
        results = search_sessions(sessions, query, defer=True)
        deferred = [i for i, r in enumerate(results) if r.deferred]
        loaded = search_each([results[i].session for i in deferred], query)
        assert all(r.match_count for r in loaded)
        for i, r in zip(deferred, loaded, strict=True):
            results[i] = r
        assert [_answer(r) for r in results] == reference
        if query == "wombat":  # and it does defer, or this proves nothing
            assert {sessions[i].id for i in deferred} == {
                "plain",
                "marked",
                "upper",
                "escaped",
                "joined",
            }

    def test_the_reference_actually_finds_what_these_queries_are_for(
        self, tmp_path, monkeypatch
    ):
        """Agreement proves nothing if both sides find nothing."""
        monkeypatch.setenv("SESSION_BROWSER_CACHE", "0")
        sessions = _corpus(tmp_path)

        def found(query):
            return {
                r.session.id for r in search_sessions(sessions, query) if r.match_count
            }

        assert found("wombat") == {"plain", "marked", "upper", "escaped", "joined"}
        assert found("strasse") == {"folds"}
        assert found("final office") == {"folds"}
        assert found("3fbb4d0") == {"marked"}
        assert found("a/b") == {"escaped"}
        assert found("3.14.10") == {"pi"}

    def test_the_words_wave_agrees_too(self, tmp_path, monkeypatch):
        sessions = _corpus(tmp_path)
        queries = ["wombat appears", "commit markers", "error found", "absent words"]

        def hits(query):
            return {
                sid: (h.strength, h.hit)
                for sid, h in search_session_word_hits(sessions, query).items()
            }

        monkeypatch.setenv("SESSION_BROWSER_CACHE", "0")
        reference = [hits(q) for q in queries]
        assert any(reference), "the fixture must produce word hits"
        monkeypatch.delenv("SESSION_BROWSER_CACHE")
        assert [hits(q) for q in queries] == reference  # cold
        assert [hits(q) for q in queries] == reference  # warm


class TestWhatAWarmSearchCosts:
    def test_a_phrase_the_raw_prefilter_cannot_take_reads_only_its_matches(
        self, tmp_path
    ):
        """The case the cache changes most. A digit or a dot used to send
        every session to the parser; over cached text they are ordinary
        characters, so the scan rules sessions out as it does for any word."""
        sessions = [s for s in _corpus(tmp_path) if s.id not in ("gone", "odd")]
        search_sessions(sessions, "anything")  # the first search fills the cache
        with (
            patch.object(
                transcript_mod, "_parse_entries", wraps=transcript_mod._parse_entries
            ) as parse,
            patch.object(entry_cache, "load", wraps=entry_cache.load) as load,
        ):
            results = search_sessions(sessions, "3fbb4d0")
        assert [r.session.id for r in results if r.match_count] == ["marked"]
        parse.assert_not_called()
        # The match and nothing else: not the corpus, and not the session
        # whose sharp s and ligatures once made it a candidate for any phrase.
        assert load.call_count == 1

    def test_a_search_reads_no_raw_transcript_once_warm(self, tmp_path):
        sessions = [s for s in _corpus(tmp_path) if s.id not in ("gone", "odd")]
        search_sessions(sessions, "anything")
        with patch.object(transcript_mod, "_rg_candidate_paths") as raw_scan:
            search_sessions(sessions, "wombat")
        raw_scan.assert_not_called()

    def test_only_a_changed_session_is_parsed_again(self, tmp_path):
        sessions = [s for s in _corpus(tmp_path) if s.id not in ("gone", "odd")]
        search_sessions(sessions, "anything")
        with open(sessions[0].content_path, "a") as fh:
            fh.write(json.dumps({"type": "user", "message": {"content": "an emu"}}))
            fh.write("\n")
        with patch.object(
            transcript_mod, "_parse_entries", wraps=transcript_mod._parse_entries
        ) as parse:
            results = search_sessions(sessions, "emu")
        assert [c.args[0].id for c in parse.call_args_list] == [sessions[0].id]
        assert [r.session.id for r in results if r.match_count] == [sessions[0].id]

    def test_a_session_that_cannot_be_cached_is_still_searched(
        self, tmp_path, monkeypatch
    ):
        """An unwritable cache must cost speed, never an answer."""
        blocker = tmp_path / "not-a-directory"
        blocker.write_text("in the way")
        monkeypatch.setenv("XDG_CACHE_HOME", str(blocker))
        sessions = _corpus(tmp_path)
        found = {r.session.id for r in search_sessions(sessions, "wombat") if r.matches}
        assert found == {"plain", "marked", "upper", "escaped", "joined"}

    def test_a_cancelled_search_builds_nothing(self, tmp_path, cache_home):
        sessions = _corpus(tmp_path)
        assert search_sessions(sessions, "wombat", cancelled=lambda: True) == []
        assert (
            transcript_mod._ensure_entry_texts(
                sessions, list(range(len(sessions))), cancelled=lambda: True
            )
            == {}
        )
        assert _files(cache_home, ".txt") == []
        assert _files(cache_home, ".idx") == []


def _on(session: Session, day: int, summary: str = "") -> Session:
    """*session*, last active on 2026-06-<day>, under *summary*."""
    session.updated_at = f"2026-06-{day:02d}T10:00:00+00:00"
    session.summary = summary
    return session


def _dated_corpus(tmp_path) -> list[Session]:
    """Sessions in a known recency order, each reaching a match another way.

    Newest first: a miss, then a match from every path a limited search can
    take -- cached Claude, Pi and Codex sessions, OpenCode's database, one
    the cache will not hold, one matching on its summary alone, one that
    cannot be read -- so a window of any size cuts across all of them.
    """
    pi = tmp_path / "pi10.jsonl"
    pi.write_text(
        json.dumps(
            {
                "type": "message",
                "timestamp": "2026-06-10T10:00:00Z",
                "message": {"role": "user", "content": "pi says wombat"},
            }
        )
        + "\n"
    )
    make_opencode_db_multi(tmp_path, {"oc8": ["an opencode wombat", "and more"]})
    codex = _codex(
        tmp_path,
        "codex5",
        {
            "type": "response_item",
            "payload": {
                "type": "message",
                "role": "assistant",
                "content": [{"type": "output_text", "text": "codex saw a wombat"}],
            },
        },
    )
    return [
        _on(_claude(tmp_path, "newest-miss", "nothing to see", "still nothing"), 12),
        _on(_claude(tmp_path, "c11", "a wombat at dawn", "noted"), 11),
        _on(Session(id="pi10", provider="pi", content_path=str(pi)), 10),
        _on(_claude(tmp_path, "census", "no animals", "none"), 9, "Wombat census"),
        _on(Session(id="oc8", provider="opencode"), 8),
        _on(_claude(tmp_path, "c7", "a wombat, a wombat", "two of them"), 7),
        _on(_claude(tmp_path, "uncached6", "a wombat it will not keep", "ok"), 6),
        _on(codex, 5),
        _on(
            Session(
                id="gone4", provider="claude", content_path=str(tmp_path / "gone4")
            ),
            4,
        ),
        _on(_claude(tmp_path, "c3", "a `wom`**bat** in markers", "a quokka"), 3),
        _on(_claude(tmp_path, "c2", "wombat", "quokka"), 2),
        _on(_claude(tmp_path, "c1", "only a quokka", "fine"), 1),
    ]


# Each is run with the cache off, which is the reference, then cold, warm and
# without ripgrep, and all four must print the same thing.
LIMITED = [
    ("wombat", "--limit", "0"),
    ("wombat", "--limit", "1"),
    ("wombat", "--limit", "3"),
    ("wombat", "--limit", "50"),
    ("wombat", "--sort", "oldest", "--limit", "2"),
    ("wombat", "--mode", "full", "--limit", "2"),
    ("wombat", "--mode", "ids", "--format", "text", "--limit", "4"),
    ("wombat", "--format", "jsonl", "--limit", "2"),
    ("wombat", "quokka", "--limit", "3"),  # alternates
    ("census", "--limit", "1"),  # a summary and nothing else
    ("wombat", "--around", "claude:c7", "--limit", "3"),  # nearest first
    ("wombat", "--provider", "claude", "--limit", "2"),
]


class TestALimitedSearchLoadsOnlyWhatItShows:
    """--limit under a metadata sort bounds the work, not just the output.

    The cached-text scan says which sessions match, and discovery says when
    each was last active, so the rows a limit keeps are known before any is
    read and only those are loaded. Whatever that skips, the output must be
    the stateless path's, byte for byte, in every shape search prints.
    """

    @pytest.fixture
    def run(self, tmp_path, monkeypatch, capsys):
        sessions = _dated_corpus(tmp_path)
        refused = next(s for s in sessions if s.id == "uncached6").content_path
        real_store = entry_cache.store

        def store(ref, *args, **kwargs):
            # One session the cache will not hold, which is then searched the
            # stateless way inside an otherwise cached search.
            if ref.source == refused:
                return False
            return real_store(ref, *args, **kwargs)

        monkeypatch.setattr(entry_cache, "store", store)
        monkeypatch.setattr(
            "session_browser.cli.discover_all", lambda *a, **k: sessions
        )
        monkeypatch.setattr(Path, "home", staticmethod(lambda: tmp_path))

        def run(*argv, cache=True, rg=True):
            with pytest.MonkeyPatch.context() as mp:
                if not cache:
                    mp.setenv("SESSION_BROWSER_CACHE", "0")
                if not rg:
                    mp.setattr(transcript_mod, "_rg_entry_text_candidates", _none)
                    mp.setattr(transcript_mod, "_rg_candidate_paths", _none)
                code = run_cli(["search", *argv])
            captured = capsys.readouterr()
            return code, captured.out, captured.err

        run.sessions = sessions
        return run

    @pytest.mark.parametrize("argv", LIMITED, ids=" ".join)
    def test_prints_what_the_stateless_path_prints(self, run, argv):
        reference = run(*argv, cache=False)
        assert reference[0] == 0
        assert run(*argv) == reference  # cold
        assert run(*argv) == reference  # warm
        assert run(*argv, rg=False) == reference

    def test_the_reference_is_worth_agreeing_with(self, run):
        _, out, _ = run("wombat", "--limit", "3", cache=False)
        data = json.loads(out)
        assert [r["id"] for r in data["results"]] == [
            "claude:c11",
            "pi:pi10",
            "claude:census",
        ]
        assert data["results"][2]["summary_matches"] == ["wombat"]
        assert [s["id"] for s in data["skipped"]] == ["claude:gone4"]
        assert any("--limit 3 dropped 6 matched" in w for w in data["warnings"])

    def test_only_the_rows_shown_are_loaded(self, tmp_path, run):
        run("wombat")  # the first search fills the cache
        with (
            patch.object(entry_cache, "load", wraps=entry_cache.load) as load,
            patch.object(
                transcript_mod, "_parse_entries", wraps=transcript_mod._parse_entries
            ) as parse,
        ):
            code, out, _ = run("wombat", "--limit", "2")
        assert code == 0
        assert [r["id"] for r in json.loads(out)["results"]] == [
            "claude:c11",
            "pi:pi10",
        ]
        hits = {Path(c.args[0].source).stem for c in load.call_args_list}
        # The two shown, and the one session the cache refuses, which is
        # searched statelessly: a load that misses, then what this process
        # parsed of it in the first search.
        assert hits == {"c11", "pi10", "uncached6"}
        # Nothing is parsed but what could not be read, limit or none.
        assert {c.args[0].id for c in parse.call_args_list} == {"gone4"}

    def test_a_session_the_cache_refuses_is_parsed_once_per_search(self, run):
        """Trying to cache it and then reading it statelessly used to
        parse it twice, on every search."""
        transcript_mod._unstored.clear()
        with patch.object(
            transcript_mod, "_parse_entries", wraps=transcript_mod._parse_entries
        ) as parse:
            run("wombat")
            first = [c.args[0].id for c in parse.call_args_list]
            parse.reset_mock()
            run("wombat", "--limit", "2")
            later = [c.args[0].id for c in parse.call_args_list]
        assert first.count("uncached6") == 1
        assert "uncached6" not in later

    def test_a_false_candidate_is_dropped_and_the_next_one_shown(self, run):
        """The scan is exact, so this cannot happen -- which is why a shown
        row is still checked rather than trusted."""
        reference = run("wombat", "--limit", "2", cache=False)
        run("wombat")
        miss = next(s for s in run.sessions if s.id == "newest-miss")
        planted = entry_cache.locate(miss.provider, miss.id, miss.content_path).text
        real = transcript_mod._rg_entry_text_candidates

        def scan(texts, needles):
            found = real(texts, needles)
            return None if found is None else found | {planted}

        with patch.object(transcript_mod, "_rg_entry_text_candidates", scan):
            assert run("wombat", "--limit", "2") == reference

    def test_a_session_gone_before_its_load_is_skipped_as_ever(self, run):
        """Removed after the scan named it and before the load: reported as
        unreadable, in the place it would have been, and the next row shown."""
        run("wombat")
        victim = next(s for s in run.sessions if s.id == "c11").content_path
        real = transcript_mod._rg_entry_text_candidates

        def scan(texts, needles):
            found = real(texts, needles)
            os.unlink(victim)
            return found

        with patch.object(transcript_mod, "_rg_entry_text_candidates", scan):
            got = run("wombat", "--limit", "2")
        reference = run("wombat", "--limit", "2", cache=False)  # gone from the start
        assert got == reference
        assert [s["id"] for s in json.loads(got[1])["skipped"]] == [
            "claude:c11",
            "claude:gone4",
        ]

    def test_many_rows_are_loaded_on_worker_processes(self, run):
        run("wombat")
        with (
            patch.object(transcript_mod, "_PROC_MIN_CANDIDATES", 3),
            patch.object(
                transcript_mod,
                "_probe_in_processes",
                return_value=transcript_mod._PROC_FALLBACK,
            ) as route,
        ):
            code, out, _ = run("wombat", "--provider", "claude", "--limit", "5")
        assert code == 0
        # The deferred rows among the five shown, in one batch. The summary
        # match needs no load, and the session the cache refuses was searched
        # with the rest, on threads: two raw sessions are below the threshold.
        assert route.call_count == 1
        assert [s.id for _, s in route.call_args.args[0]] == ["c11", "c7", "c3"]
        # The pool was unusable here, so threads loaded the same rows.
        assert [r["id"] for r in json.loads(out)["results"]] == [
            "claude:c11",
            "claude:census",
            "claude:c7",
            "claude:uncached6",
            "claude:c3",
        ]

    def test_output_files_match_too(self, tmp_path, run):
        def written(out_dir, **kw):
            code, _, _ = run(
                "wombat",
                "--mode",
                "full",
                "--limit",
                "2",
                "--output-dir",
                str(out_dir),
                **kw,
            )
            assert code == 0
            files = {p.name: p.read_text() for p in sorted(out_dir.iterdir())}
            manifest = json.loads(files.pop("manifest.json"))
            manifest.pop("generated_at")
            return manifest, files

        reference = written(tmp_path / "reference", cache=False)
        run("wombat")
        assert written(tmp_path / "cached") == reference
        assert len(reference[1]) == 2  # a transcript per row, and not empty
        assert all("wombat" in text for text in reference[1].values())


def _none(*_args, **_kwargs):
    return None


class TestBuildingInWorkerProcesses:
    def _many(self, tmp_path, n=8) -> list[Session]:
        return [
            _claude(tmp_path, f"s{i}", "a wombat" if i % 2 else "nothing", f"reply {i}")
            for i in range(n)
        ]

    def test_a_large_first_build_is_routed_to_processes(self, tmp_path):
        sessions = self._many(tmp_path)
        with (
            patch.object(transcript_mod, "_PROC_MIN_CANDIDATES", 4),
            patch.object(
                transcript_mod,
                "_store_in_processes",
                return_value=transcript_mod._PROC_FALLBACK,
            ) as route,
        ):
            results = search_sessions(sessions, "wombat")
        assert len(route.call_args.args[0]) == len(sessions)
        # The pool was unusable, so threads built the same files.
        assert sum(r.match_count for r in results) == 4

    def test_a_handful_of_changed_sessions_stays_on_threads(self, tmp_path):
        sessions = self._many(tmp_path, n=3)
        with (
            patch.object(transcript_mod, "_PROC_MIN_CANDIDATES", 4),
            patch.object(transcript_mod, "_store_in_processes") as route,
        ):
            search_sessions(sessions, "wombat")
        route.assert_not_called()

    def test_real_worker_processes_write_a_cache_the_parent_can_read(
        self, tmp_path, cache_home
    ):
        sessions = self._many(tmp_path)
        jobs = [
            (
                i,
                s,
                entry_cache.locate(s.provider, s.id, s.content_path),
            )
            for i, s in enumerate(sessions)
        ]
        stored = transcript_mod._store_in_processes(jobs, None)
        if stored is transcript_mod._PROC_FALLBACK:
            pytest.skip("worker processes unavailable here")
        assert stored == set(range(len(sessions)))
        assert len(_files(cache_home, ".txt")) == len(sessions)
        with patch.object(transcript_mod, "_parse_entries") as parse:
            assert [e.text for e in iter_entries(sessions[1], [])] == [
                "a wombat",
                "reply 1",
            ]
        parse.assert_not_called()


class TestProgress:
    def test_parsing_is_reported_only_when_there_is_something_to_parse(self, tmp_path):
        sessions = [s for s in _corpus(tmp_path) if s.id not in ("gone", "odd")]
        seen: list[tuple[str, int, int]] = []
        search_sessions(sessions, "wombat", progress=lambda *a: seen.append(a))
        parsing = [s for s in seen if s[0] == "parsing"]
        assert parsing[0] == ("parsing", 0, len(sessions))
        assert sorted(done for _, done, _ in parsing) == list(range(len(sessions) + 1))
        seen.clear()
        search_sessions(sessions, "wombat", progress=lambda *a: seen.append(a))
        assert not [s for s in seen if s[0] == "parsing"]
