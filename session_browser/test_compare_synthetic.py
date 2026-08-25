"""Behaviour tests for the one-shot build-and-compare wrapper.

The comparator itself is faked throughout. What this wrapper owns is the corpus
lifecycle and the refusals around it, and running two real revisions for real
timings would test the comparator instead, slowly.
"""

from __future__ import annotations

import json
import pathlib
import shutil

import pytest

from benchmarks import compare_synthetic, retrieval_compare, synthetic_corpus


@pytest.fixture
def comparator(monkeypatch):
    """Record how the comparator was called; write a plausible report."""
    calls = []

    def fake(**kwargs):
        calls.append(kwargs)
        report = {"passed": True, "aggregate": {"ratio": 1.0, "verdict": "ok"}}
        path = kwargs["report_path"]
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(report))
        return report

    monkeypatch.setattr(retrieval_compare, "run_comparison", fake)
    return calls


def _run(tmp_path, comparator_kwargs=None, **kwargs):
    return compare_synthetic.run(
        baseline_repo=tmp_path / "base",
        candidate_repo=tmp_path / "cand",
        report_path=tmp_path / "report.json",
        **kwargs,
    )


class TestCorpusLifecycle:
    def test_builds_a_corpus_and_points_the_comparison_at_it(
        self, tmp_path, comparator
    ):
        _run(tmp_path, scale=1)
        (call,) = comparator
        home = call["home"]
        assert (home / synthetic_corpus.MANIFEST_NAME).exists() is False
        # The manifest existed while the comparison ran; the corpus is gone now.
        assert not home.exists()

    def test_temporary_corpus_is_removed_even_when_the_run_fails(
        self, tmp_path, monkeypatch
    ):
        seen = {}

        def boom(**kwargs):
            seen["home"] = kwargs["home"]
            raise retrieval_compare.ComparatorError("candidate is slower")

        monkeypatch.setattr(retrieval_compare, "run_comparison", boom)
        with pytest.raises(retrieval_compare.ComparatorError):
            _run(tmp_path, scale=1)
        # A failing run is exactly when a forgotten corpus would survive.
        assert not seen["home"].exists()

    def test_keep_corpus_leaves_it_behind(self, tmp_path, comparator):
        _run(tmp_path, scale=1, keep_corpus=True)
        (call,) = comparator
        assert (call["home"] / synthetic_corpus.MANIFEST_NAME).exists()

    def test_a_named_corpus_is_never_deleted(self, tmp_path, comparator):
        named = tmp_path / "mine"
        _run(tmp_path, scale=1, corpus=named)
        assert (named / synthetic_corpus.MANIFEST_NAME).exists()

    def test_a_named_corpus_must_be_empty(self, tmp_path, comparator):
        named = tmp_path / "mine"
        named.mkdir()
        (named / "keep-me").write_text("real data\n")
        with pytest.raises(synthetic_corpus.CorpusError):
            _run(tmp_path, scale=1, corpus=named)
        assert (named / "keep-me").exists()
        assert not comparator


class TestRefusals:
    def test_thin_sampling_is_refused_before_a_corpus_is_built(
        self, tmp_path, comparator
    ):
        with pytest.raises(compare_synthetic.WrapperError) as caught:
            _run(tmp_path, repeats=retrieval_compare.MIN_REPEATS - 1)
        # The point of refusing here rather than letting the comparator refuse
        # is that nothing was written first.
        assert not comparator
        assert str(retrieval_compare.MIN_REPEATS) in str(caught.value)

    def test_the_minimum_is_the_comparator_s_own(self, tmp_path, comparator):
        _run(tmp_path, scale=1, repeats=retrieval_compare.MIN_REPEATS)
        (call,) = comparator
        assert call["repeats"] == retrieval_compare.MIN_REPEATS

    def test_an_unplanted_selectivity_is_refused_with_the_options(self, tmp_path):
        with pytest.raises(compare_synthetic.WrapperError) as caught:
            compare_synthetic.queries_for([0.33])
        assert "0.01" in str(caught.value)


class TestQueries:
    def test_default_queries_span_the_cost_curve(self, tmp_path, comparator):
        _run(tmp_path, scale=1)
        (call,) = comparator
        planted = {term.term: term.selectivity for term in synthetic_corpus.TERMS}
        chosen = sorted(planted[query] for query in call["queries"])
        assert chosen == sorted(compare_synthetic.DEFAULT_SELECTIVITIES)

    def test_explicit_queries_replace_the_defaults(self, tmp_path, comparator):
        _run(tmp_path, scale=1, queries=["kookaburra"])
        (call,) = comparator
        assert call["queries"] == ["kookaburra"]

    def test_no_live_session_exclusion_is_passed(self, tmp_path, comparator):
        """A synthetic HOME has no live session, so there is nothing to exclude.

        The comparator's ``--current-session-env`` exists because a real
        ``$HOME`` contains the session doing the measuring, which grows mid-run
        and makes every query matching it ``volatile``. Nothing here is being
        written to, so passing an exclusion would be cargo cult.
        """
        _run(tmp_path, scale=1)
        (call,) = comparator
        assert not call.get("current_session_env")
        assert not call.get("include_current")


class TestReport:
    def test_the_report_records_the_corpus_it_measured(self, tmp_path, comparator):
        report = _run(tmp_path, scale=2, seed=7)
        assert report["corpus"]["scale"] == 2
        assert report["corpus"]["seed"] == 7
        assert report["corpus"]["sessions"] == 2 * synthetic_corpus.BASE_TOTAL
        # And it survives on disk, after the corpus itself is gone.
        on_disk = json.loads((tmp_path / "report.json").read_text())
        assert on_disk["corpus"] == report["corpus"]

    def test_a_failing_run_still_records_the_corpus(self, tmp_path, monkeypatch):
        def boom(**kwargs):
            kwargs["report_path"].write_text(json.dumps({"passed": False}))
            raise retrieval_compare.ComparatorError("candidate is slower")

        monkeypatch.setattr(retrieval_compare, "run_comparison", boom)
        with pytest.raises(retrieval_compare.ComparatorError):
            _run(tmp_path, scale=1)
        written = json.loads((tmp_path / "report.json").read_text())
        assert written["passed"] is False
        assert written["corpus"]["sessions"] == synthetic_corpus.BASE_TOTAL


class TestCli:
    def test_prints_a_summary_naming_the_corpus(self, tmp_path, comparator, capsys):
        code = compare_synthetic.main(
            [
                "--baseline-repo",
                str(tmp_path / "base"),
                "--candidate-repo",
                str(tmp_path / "cand"),
                "--report",
                str(tmp_path / "report.json"),
                "--scale",
                "1",
            ]
        )
        assert code == 0
        printed = json.loads(capsys.readouterr().out)
        assert printed["passed"] is True
        assert printed["sessions"] == synthetic_corpus.BASE_TOTAL
        # A deleted corpus has no path worth printing.
        assert "corpus" not in printed

    def test_a_kept_corpus_is_named_so_it_can_be_found_again(
        self, tmp_path, comparator, capsys
    ):
        code = compare_synthetic.main(
            [
                "--baseline-repo",
                str(tmp_path / "base"),
                "--candidate-repo",
                str(tmp_path / "cand"),
                "--report",
                str(tmp_path / "report.json"),
                "--keep-corpus",
            ]
        )
        assert code == 0
        printed = json.loads(capsys.readouterr().out)
        # --keep-corpus writes to a temporary directory with a generated name,
        # so a caller that is not told the path cannot find what it kept.
        kept = pathlib.Path(printed["corpus"])
        try:
            assert kept.is_dir()
            assert (kept / synthetic_corpus.MANIFEST_NAME).exists()
        finally:
            # --keep-corpus deliberately outlives the run, and this one is
            # outside tmp_path, so the test is what has to remove it.
            shutil.rmtree(kept, ignore_errors=True)

    def test_a_refused_run_is_an_error_not_a_traceback(
        self, tmp_path, comparator, capsys
    ):
        code = compare_synthetic.main(
            [
                "--baseline-repo",
                str(tmp_path / "base"),
                "--candidate-repo",
                str(tmp_path / "cand"),
                "--report",
                str(tmp_path / "report.json"),
                "--repeats",
                "3",
            ]
        )
        assert code == 1
        assert capsys.readouterr().err.startswith("error: ")
        assert not comparator

    def test_a_conviction_says_where_the_report_is(self, tmp_path, monkeypatch, capsys):
        def boom(**kwargs):
            kwargs["report_path"].write_text(json.dumps({"passed": False}))
            raise retrieval_compare.ComparatorError("candidate is slower")

        monkeypatch.setattr(retrieval_compare, "run_comparison", boom)
        code = compare_synthetic.main(
            [
                "--baseline-repo",
                str(tmp_path / "base"),
                "--candidate-repo",
                str(tmp_path / "cand"),
                "--report",
                str(tmp_path / "report.json"),
            ]
        )
        assert code == 1
        assert "report.json" in capsys.readouterr().err
