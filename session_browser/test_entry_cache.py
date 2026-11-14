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
from session_browser.discovery import Session, scan_claude
from session_browser.test_transcript import _result_view, write_claude_jsonl
from session_browser.transcript import (
    TranscriptEntry,
    count_entries,
    iter_entries,
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
