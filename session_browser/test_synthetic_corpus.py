"""Behaviour tests for the comparator's scalable synthetic corpus.

These tests keep the corpora tiny — the largest builds one scale unit — because
the point of each is a *property* (selectivity is exact, the shape is present,
the build reproduces) and none of them get truer at 2,000 sessions. Anything
large enough to time belongs in a temporary directory a benchmark deletes, not
in the suite.
"""

from __future__ import annotations

import ast
import json
import sqlite3
from pathlib import Path

import pytest

from benchmarks import synthetic_corpus as sc
from session_browser import discovery, transcript


def _corpus(tmp_path, **kwargs):
    return sc.generate(tmp_path / "home", **kwargs)


def _sessions(monkeypatch, home):
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setattr(
        "pathlib.Path.home", lambda: home, raising=False
    )  # discovery reads Path.home() directly
    return discovery.discover_all()


class TestManifest:
    def test_scale_multiplies_every_provider(self, tmp_path):
        one = _corpus(tmp_path / "a", scale=1)
        three = _corpus(tmp_path / "b", scale=3)
        assert three["sessions"] == 3 * one["sessions"]
        for provider, count in one["sessions_by_provider"].items():
            assert three["sessions_by_provider"][provider] == 3 * count

    def test_manifest_is_written_beside_the_tree(self, tmp_path):
        manifest = _corpus(tmp_path, scale=1)
        written = json.loads((tmp_path / "home" / sc.MANIFEST_NAME).read_text())
        # The manifest on disk is what a later run reads to choose its
        # queries, so it has to be the same object the builder returned.
        assert written == manifest

    def test_reported_bytes_match_the_tree(self, tmp_path):
        manifest = _corpus(tmp_path, scale=1)
        home = tmp_path / "home"
        on_disk = sum(
            path.stat().st_size
            for path in home.rglob("*")
            if path.is_file() and path.name != sc.MANIFEST_NAME
        )
        assert manifest["bytes"] == on_disk

    def test_scale_below_one_is_refused(self, tmp_path):
        with pytest.raises(sc.CorpusError):
            _corpus(tmp_path, scale=0)

    def test_non_empty_directory_is_refused_without_force(self, tmp_path):
        home = tmp_path / "home"
        home.mkdir()
        (home / "please-do-not-delete").write_text("real data\n")
        with pytest.raises(sc.CorpusError):
            sc.generate(home, scale=1)
        assert (home / "please-do-not-delete").exists()


class TestReproducibility:
    def test_same_seed_writes_the_same_jsonl(self, tmp_path):
        sc.generate(tmp_path / "one", scale=1)
        sc.generate(tmp_path / "two", scale=1)
        for left in sorted((tmp_path / "one").rglob("*.jsonl")):
            right = tmp_path / "two" / left.relative_to(tmp_path / "one")
            assert left.read_bytes() == right.read_bytes()

    def test_a_different_seed_writes_different_text(self, tmp_path):
        first = sc.generate(tmp_path / "one", scale=1)
        second = sc.generate(tmp_path / "two", scale=1, seed=sc.DEFAULT_SEED + 1)
        # Session counts are structural and must not move with the seed; the
        # lengths drawn from the tail are exactly what the seed decides.
        assert first["sessions"] == second["sessions"]
        assert first["turns"]["total"] != second["turns"]["total"]


class TestSelectivity:
    """The manifest's promise, checked against the product's own search."""

    @pytest.fixture(scope="class")
    def built(self, tmp_path_factory):
        home = tmp_path_factory.mktemp("selectivity") / "home"
        return sc.generate(home, scale=1), home

    @pytest.mark.parametrize("term", [term.term for term in sc.TERMS])
    def test_planted_count_is_what_the_manifest_says(self, built, monkeypatch, term):
        manifest, home = built
        promised = next(
            entry["sessions"] for entry in manifest["queries"] if entry["term"] == term
        )
        sessions = _sessions(monkeypatch, home)
        results = transcript.search_sessions(sessions, term, keep_entries=False)
        matched = [result for result in results if result.match_count]
        assert len(matched) == promised

    def test_the_absent_term_is_absent(self, built):
        manifest, _ = built
        absent = next(
            entry for entry in manifest["queries"] if entry["selectivity"] == 0.0
        )
        assert absent["sessions"] == 0

    def test_selectivity_is_a_property_of_the_whole_corpus(self, built, monkeypatch):
        """Not of one provider's slice.

        Terms are planted against a running index that spans every provider, so
        a query at 10% opens a tenth of the *corpus*. If planting restarted per
        provider the rare term would land in every provider's first session and
        a rare query would touch all four parsers, which is not what "rare"
        should cost.
        """
        _, home = built
        sessions = _sessions(monkeypatch, home)
        results = transcript.search_sessions(sessions, "kookaburra", keep_entries=False)
        providers = {
            result.session.provider for result in results if result.match_count
        }
        assert len(providers) < 4


class TestShape:
    def test_session_lengths_have_a_long_tail(self, tmp_path):
        manifest = _corpus(tmp_path, scale=1)
        turns = manifest["turns"]
        # A flat corpus would put the longest session on the mean. The tail is
        # the whole reason this corpus is not build_corpus.
        assert turns["longest"] > 4 * turns["mean"]
        assert turns["mean"] > sc.MIN_TURNS

    def test_all_three_codex_eras_are_written(self, tmp_path):
        manifest = _corpus(tmp_path, scale=1)
        assert all(count > 0 for count in manifest["codex_eras"].values())

    def test_codex_rollouts_carry_their_era_vocabulary(self, tmp_path):
        _corpus(tmp_path, scale=1)
        home = tmp_path / "home"
        text = "\n".join(
            path.read_text() for path in (home / sc.CODEX_SESSIONS).rglob("*.jsonl")
        )
        assert '"agent_message"' in text  # legacy
        assert '"item_completed"' in text  # paginated
        assert '"input_text"' in text  # response-item-only user turn

    def test_claude_sessions_carry_tool_blocks_and_string_content(self, tmp_path):
        _corpus(tmp_path, scale=1)
        home = tmp_path / "home"
        text = "\n".join(
            path.read_text() for path in (home / sc.CLAUDE_PROJECTS).rglob("*.jsonl")
        )
        assert '"tool_use"' in text
        assert '"tool_result"' in text
        assert '"content": "' in text  # the plain-string variant

    def test_opencode_rows_mix_tool_parts_with_text(self, tmp_path):
        _corpus(tmp_path, scale=1)
        db = tmp_path / "home" / sc.OPENCODE_DB
        conn = sqlite3.connect(str(db))
        try:
            kinds = {
                json.loads(row[0]).get("type")
                for row in conn.execute("SELECT data FROM part")
            }
        finally:
            conn.close()
        assert kinds == {"text", "tool"}


class TestDiscovery:
    """Provider-native means the product reads it without knowing it is fake."""

    def test_every_written_session_is_discovered_and_readable(
        self, tmp_path, monkeypatch
    ):
        manifest = _corpus(tmp_path, scale=1)
        sessions = _sessions(monkeypatch, tmp_path / "home")
        assert len(sessions) == manifest["sessions"]
        by_provider: dict[str, int] = {}
        for session in sessions:
            by_provider[session.provider] = by_provider.get(session.provider, 0) + 1
        assert by_provider == manifest["sessions_by_provider"]

    def test_no_session_parses_empty(self, tmp_path, monkeypatch):
        _corpus(tmp_path, scale=1)
        sessions = _sessions(monkeypatch, tmp_path / "home")
        for session in sessions:
            warnings: list[str] = []
            assert list(transcript.iter_entries(session, warnings)), session.session_id
            assert not warnings, session.session_id

    def test_the_manifest_file_is_not_mistaken_for_a_session(
        self, tmp_path, monkeypatch
    ):
        manifest = _corpus(tmp_path, scale=1)
        sessions = _sessions(monkeypatch, tmp_path / "home")
        assert len(sessions) == manifest["sessions"]


class TestGateCorpusIsUntouched:
    """The one thing this module must never become.

    ``build_corpus`` is the perf gate's, and its value is being byte-identical
    for everyone because the budgets are exact integers. If this module ever
    grew an import of it — or worse, a call that passed a size through — a
    generalisation bug would not fail loudly here; it would shift the gate's
    counters and ``bless`` would record corpus drift as though it were a parser
    change. The separation is the design, so it gets a test rather than a
    comment.
    """

    def test_no_reference_to_the_gate_corpus(self):
        # Prose is allowed to name it — the separation has to be explained
        # somewhere. Code is not: an import or a call is how the two corpora
        # would start sharing a fate.
        tree = ast.parse(Path(sc.__file__).read_text())
        imported = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                imported.add(node.module or "")
            if isinstance(node, ast.Import | ast.ImportFrom):
                imported.update(alias.name for alias in node.names)
        assert not any("perf_budget" in name for name in imported)
        called = {
            node.func.id
            for node in ast.walk(tree)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
        }
        assert "build_corpus" not in called


class TestCli:
    """A corpus can be built at a chosen scale without writing a script."""

    def test_prints_the_manifest_it_wrote(self, tmp_path, capsys):
        home = tmp_path / "home"
        assert sc.main(["--home", str(home), "--scale", "2"]) == 0
        printed = json.loads(capsys.readouterr().out)
        assert printed == json.loads((home / sc.MANIFEST_NAME).read_text())
        assert printed["scale"] == 2
        assert printed["sessions"] == 2 * sc.BASE_TOTAL

    def test_seed_is_selectable_from_the_command_line(self, tmp_path, capsys):
        sc.main(["--home", str(tmp_path / "a"), "--seed", "1"])
        first = json.loads(capsys.readouterr().out)
        sc.main(["--home", str(tmp_path / "b"), "--seed", "2"])
        second = json.loads(capsys.readouterr().out)
        assert first["turns"]["total"] != second["turns"]["total"]

    def test_a_bad_scale_is_an_error_not_a_traceback(self, tmp_path, capsys):
        assert sc.main(["--home", str(tmp_path / "home"), "--scale", "0"]) == 1
        assert capsys.readouterr().err.startswith("error: ")

    def test_a_non_empty_home_is_refused_and_says_how_to_override(
        self, tmp_path, capsys
    ):
        home = tmp_path / "home"
        home.mkdir()
        (home / "keep-me").write_text("real data\n")
        assert sc.main(["--home", str(home)]) == 1
        # The message has to name the flag, not the keyword argument: this
        # error is reached from the command line far more often than from code.
        assert "--force" in capsys.readouterr().err
        assert (home / "keep-me").exists()

    def test_force_writes_into_a_non_empty_home(self, tmp_path, capsys):
        home = tmp_path / "home"
        home.mkdir()
        (home / "stale").write_text("left over\n")
        assert sc.main(["--home", str(home), "--force"]) == 0
        assert json.loads(capsys.readouterr().out)["sessions"] == sc.BASE_TOTAL

    def test_help_states_what_one_scale_unit_is(self, capsys):
        with pytest.raises(SystemExit):
            sc.parse_args(["--help"])
        assert str(sc.BASE_TOTAL) in capsys.readouterr().out
