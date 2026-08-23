"""Behaviour tests for the cross-revision retrieval comparator."""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from benchmarks import retrieval_compare


def _payload(
    *,
    match_count: int = 2,
    ids: tuple[str, ...] = ("claude:one",),
    unreadable: tuple[str, ...] = (),
) -> dict:
    results = []
    for index, session_id in enumerate(ids):
        results.append(
            {
                "id": session_id,
                "match_count": match_count + index,
                "summary_matches": ["needle"],
                "first_match": 3,
                "last_match": 7,
                "total_entries": 10,
            }
        )
    return {
        "query": "needle",
        "results": results,
        "skipped": [
            {"id": session_id, "error": "unreadable"} for session_id in unreadable
        ],
    }


def _runner(payloads, timings, calls):
    """Return a product-boundary fake with independently supplied facts."""
    positions = {repo: 0 for repo in payloads}

    def run(repo, query, env, search_args):
        calls.append((repo.name, query, tuple(search_args), env["HOME"]))
        pos = positions[repo]
        positions[repo] += 1
        return retrieval_compare.SearchRun(
            payload=payloads[repo][pos],
            seconds=timings[repo][pos],
        )

    return run


def test_signature_keeps_only_stable_ordered_search_facts():
    signature = retrieval_compare.canonical_signature(
        {
            "generated_at": "volatile",
            "results": [
                {
                    "id": "claude:two",
                    "summary": "volatile text",
                    "match_count": 4,
                    "summary_matches": ["needle"],
                    "first_match": 2,
                    "last_match": 9,
                    "total_entries": 12,
                }
            ],
            "skipped": [{"id": "claude:broken", "error": "volatile error"}],
        }
    )

    assert signature == {
        "result_ids": ["claude:two"],
        "results": [
            {
                "id": "claude:two",
                "match_count": 4,
                "summary_matches": ["needle"],
                "first_match": 2,
                "last_match": 9,
                "total_entries": 12,
            }
        ],
        "unreadable_ids": ["claude:broken"],
    }


def test_comparison_alternates_measured_revisions_after_warming(tmp_path, monkeypatch):
    baseline, candidate = tmp_path / "baseline", tmp_path / "candidate"
    baseline.mkdir()
    candidate.mkdir()
    (tmp_path / "home").mkdir()
    calls = []
    payload = _payload()
    monkeypatch.setattr(
        retrieval_compare,
        "_run_search",
        _runner(
            {baseline: [payload] * 5, candidate: [payload] * 5},
            {baseline: [1.0] * 5, candidate: [0.9] * 5},
            calls,
        ),
    )

    report = retrieval_compare.run_comparison(
        baseline_repo=baseline,
        candidate_repo=candidate,
        home=tmp_path / "home",
        queries=["needle"],
        repeats=3,
        warmup=0,
    )

    assert [call[0] for call in calls] == [
        # stability probe: each revision reproduces its own result first
        "baseline",
        "candidate",
        "baseline",
        "candidate",
        # then alternating measured runs
        "baseline",
        "candidate",
        "candidate",
        "baseline",
        "baseline",
        "candidate",
    ]
    assert report["queries"]["needle"]["baseline_samples"] == [1.0, 1.0, 1.0]
    assert report["queries"]["needle"]["candidate_samples"] == [0.9, 0.9, 0.9]
    assert all("--limit" not in call[2] for call in calls)


def test_comparison_rejects_a_run_where_every_query_was_unstable(tmp_path, monkeypatch):
    """With nothing stable left, the run must refuse to conclude anything."""
    baseline, candidate = tmp_path / "baseline", tmp_path / "candidate"
    baseline.mkdir()
    candidate.mkdir()
    (tmp_path / "home").mkdir()
    stable = _payload()
    changed = _payload(match_count=3)
    monkeypatch.setattr(
        retrieval_compare,
        "_run_search",
        _runner(
            {baseline: [stable, changed, stable], candidate: [stable] * 3},
            {baseline: [1.0] * 3, candidate: [1.0] * 3},
            [],
        ),
    )

    with pytest.raises(
        retrieval_compare.ComparatorError,
        match="every query touched a session that changed",
    ):
        retrieval_compare.run_comparison(
            baseline_repo=baseline,
            candidate_repo=candidate,
            home=tmp_path / "home",
            queries=["needle"],
            repeats=2,
            warmup=0,
        )


def test_a_volatile_query_is_excluded_without_failing_the_whole_run(
    tmp_path, monkeypatch, capsys
):
    """One query matching a live session must not discard the other queries.

    A moving session file also makes two identical revisions disagree, so
    stability is established per revision before equivalence is claimed.
    """
    baseline, candidate = tmp_path / "baseline", tmp_path / "candidate"
    baseline.mkdir()
    candidate.mkdir()
    (tmp_path / "home").mkdir()
    stable = _payload()
    grown = _payload(match_count=3)
    timings = {
        (baseline, "moving"): [1.0, 1.0],
        (candidate, "moving"): [1.0, 1.0],
        (baseline, "still"): [1.0] * 4,
        (candidate, "still"): [1.0] * 4,
    }
    payloads = {
        # the moving query's baseline grows between the two probe runs
        (baseline, "moving"): [stable, grown],
        (candidate, "moving"): [stable] * 2,
        (baseline, "still"): [stable] * 4,
        (candidate, "still"): [stable] * 4,
    }
    positions = {key: 0 for key in timings}

    def run(repo, query, _env, _search_args):
        key = (repo, query)
        position = positions[key]
        positions[key] += 1
        return retrieval_compare.SearchRun(
            payload=payloads[key][position], seconds=timings[key][position]
        )

    monkeypatch.setattr(retrieval_compare, "_run_search", run)

    report = retrieval_compare.run_comparison(
        baseline_repo=baseline,
        candidate_repo=candidate,
        home=tmp_path / "home",
        queries=["moving", "still"],
        repeats=2,
        warmup=0,
    )

    assert report["volatile_queries"] == ["moving"]
    assert report["queries"]["moving"]["verdict"] == "volatile"
    assert report["queries"]["moving"]["unstable_revision"] == "baseline"
    assert report["queries"]["still"]["verdict"] == "ok"
    assert report["aggregate"]["per_query_ratios"] == [1.0]
    assert report["passed"] is True
    assert "changed mid-run" in capsys.readouterr().err


def test_comparison_reports_strict_mismatch_diagnostics(tmp_path, monkeypatch):
    baseline, candidate = tmp_path / "baseline", tmp_path / "candidate"
    baseline.mkdir()
    candidate.mkdir()
    (tmp_path / "home").mkdir()
    monkeypatch.setattr(
        retrieval_compare,
        "_run_search",
        _runner(
            {baseline: [_payload()] * 2, candidate: [_payload(match_count=9)] * 2},
            {baseline: [1.0] * 2, candidate: [1.0] * 2},
            [],
        ),
    )

    with pytest.raises(
        retrieval_compare.ComparatorError, match="signature mismatch for query 'needle'"
    ) as exc:
        retrieval_compare.run_comparison(
            baseline_repo=baseline,
            candidate_repo=candidate,
            home=tmp_path / "home",
            queries=["needle"],
            repeats=1,
            warmup=0,
        )

    assert '"match_count": 2' in str(exc.value)
    assert '"match_count": 9' in str(exc.value)


@pytest.mark.parametrize(
    ("candidate_seconds", "passes"),
    [
        (1.05, True),
        (1.051, False),
    ],
)
def test_comparison_uses_a_strict_five_percent_median_threshold(
    tmp_path, monkeypatch, candidate_seconds, passes
):
    baseline, candidate = tmp_path / "baseline", tmp_path / "candidate"
    baseline.mkdir()
    candidate.mkdir()
    (tmp_path / "home").mkdir()
    payload = _payload()
    monkeypatch.setattr(
        retrieval_compare,
        "_run_search",
        _runner(
            {baseline: [payload] * 3, candidate: [payload] * 3},
            {baseline: [1.0] * 3, candidate: [candidate_seconds] * 3},
            [],
        ),
    )

    call = lambda: retrieval_compare.run_comparison(
        baseline_repo=baseline,
        candidate_repo=candidate,
        home=tmp_path / "home",
        queries=["needle"],
        repeats=1,
        warmup=0,
    )
    if passes:
        assert call()["passed"] is True
    else:
        with pytest.raises(retrieval_compare.ComparatorError, match="slower than 5%"):
            call()


def test_comparison_writes_raw_samples_medians_and_aggregate_report(
    tmp_path, monkeypatch
):
    baseline, candidate = tmp_path / "baseline", tmp_path / "candidate"
    baseline.mkdir()
    candidate.mkdir()
    (tmp_path / "home").mkdir()
    payload = _payload(unreadable=("claude:broken",))
    monkeypatch.setattr(
        retrieval_compare,
        "_run_search",
        _runner(
            {baseline: [payload] * 4, candidate: [payload] * 4},
            {baseline: [1.0, 9.0, 2.0, 3.0], candidate: [0.5, 9.0, 1.5, 2.5]},
            [],
        ),
    )
    report_path = tmp_path / "reports" / "comparison.json"

    returned = retrieval_compare.run_comparison(
        baseline_repo=baseline,
        candidate_repo=candidate,
        home=tmp_path / "home",
        queries=["needle"],
        repeats=2,
        report_path=report_path,
        warmup=0,
    )

    written = json.loads(report_path.read_text())
    assert written == returned
    assert written["queries"]["needle"] == {
        "baseline_signature": retrieval_compare.canonical_signature(payload),
        "candidate_signature": retrieval_compare.canonical_signature(payload),
        "baseline_samples": [2.0, 3.0],
        "candidate_samples": [1.5, 2.5],
        "baseline_median": 2.5,
        "candidate_median": 2.0,
        "baseline_trimmed_median": 2.5,
        "candidate_trimmed_median": 2.0,
        "baseline_relative_spread": 0.4,
        "candidate_relative_spread": 0.5,
        "noise_floor": 0.5,
        "ratio": 0.8,
        "verdict": "ok",
    }
    assert written["aggregate"] == {
        "statistic": "median_per_query_trimmed_ratio",
        "per_query_ratios": [0.8],
        "ratio": 0.8,
        "noise_floor": 0.5,
        "verdict": "ok",
    }
    assert written["equivalent"] is True
    assert written["slow_queries"] == []
    assert written["unresolvable_queries"] == []


def test_aggregate_uses_per_query_ratios_not_pooled_raw_samples(tmp_path, monkeypatch):
    """Equal query medians stay equal despite unlike absolute scales/outliers."""
    baseline, candidate = tmp_path / "baseline", tmp_path / "candidate"
    baseline.mkdir()
    candidate.mkdir()
    (tmp_path / "home").mkdir()
    payload = _payload()
    timings = {
        (baseline, "small"): [0.0, 0.0, 1.0, 1.0, 1.0],
        (candidate, "small"): [0.0, 0.0, 0.0, 1.0, 100.0],
        (baseline, "large"): [0.0, 0.0, 100.0, 100.0, 100.0],
        (candidate, "large"): [0.0, 0.0, 100.0, 100.0, 1_000.0],
    }
    positions = {key: 0 for key in timings}

    def run(repo, query, _env, _search_args):
        key = (repo, query)
        position = positions[key]
        positions[key] += 1
        return retrieval_compare.SearchRun(
            payload=payload, seconds=timings[key][position]
        )

    monkeypatch.setattr(retrieval_compare, "_run_search", run)

    report = retrieval_compare.run_comparison(
        baseline_repo=baseline,
        candidate_repo=candidate,
        home=tmp_path / "home",
        queries=["small", "large"],
        repeats=3,
        warmup=0,
    )

    assert report["queries"]["small"]["ratio"] == 1.0
    assert report["queries"]["large"]["ratio"] == 1.0
    assert report["aggregate"] == {
        "statistic": "median_per_query_trimmed_ratio",
        "per_query_ratios": [1.0, 1.0],
        "ratio": 1.0,
        "noise_floor": 54.5,
        "verdict": "ok",
    }
    assert report["passed"] is True


def test_missing_query_file_exits_cleanly_without_a_traceback(tmp_path, capsys):
    baseline, candidate = tmp_path / "baseline", tmp_path / "candidate"
    baseline.mkdir()
    candidate.mkdir()

    exit_code = retrieval_compare.main(
        [
            "--baseline-repo",
            str(baseline),
            "--candidate-repo",
            str(candidate),
            "--home",
            str(tmp_path / "home"),
            "--query-file",
            str(tmp_path / "missing-private-queries.txt"),
            "--report",
            str(tmp_path / "report.json"),
        ]
    )

    captured = capsys.readouterr()
    assert exit_code == 1
    assert captured.err.startswith("error: cannot read query file")
    assert "Traceback" not in captured.err


def test_comparison_rejects_missing_home_before_product(tmp_path, monkeypatch):
    """Both revisions must receive one real, pre-existing source HOME."""
    baseline, candidate = tmp_path / "baseline", tmp_path / "candidate"
    baseline.mkdir()
    candidate.mkdir()
    monkeypatch.setattr(
        retrieval_compare,
        "_run_search",
        lambda *_args, **_kwargs: pytest.fail("product invoked"),
    )

    with pytest.raises(
        retrieval_compare.ComparatorError,
        match="--home must be an existing directory",
    ):
        retrieval_compare.run_comparison(
            baseline_repo=baseline,
            candidate_repo=candidate,
            home=tmp_path / "missing-home",
            queries=["needle"],
            repeats=1,
            warmup=0,
        )


def test_comparison_rejects_non_discovery_environment_override(tmp_path, monkeypatch):
    """Environment overrides must not replace HOME or supply unrelated state."""
    baseline, candidate, home = (
        tmp_path / "baseline",
        tmp_path / "candidate",
        tmp_path / "home",
    )
    baseline.mkdir()
    candidate.mkdir()
    home.mkdir()
    monkeypatch.setattr(
        retrieval_compare,
        "_run_search",
        lambda *_args, **_kwargs: pytest.fail("product invoked"),
    )

    with pytest.raises(
        retrieval_compare.ComparatorError,
        match=r"unsupported --current-session-env 'HOME'",
    ):
        retrieval_compare.run_comparison(
            baseline_repo=baseline,
            candidate_repo=candidate,
            home=home,
            queries=["needle"],
            repeats=1,
            warmup=0,
            current_session_env=["HOME=/not-the-source-home"],
        )


def test_comparison_preserves_known_current_session_overrides(tmp_path, monkeypatch):
    """The two discovery-affecting exclusion variables remain configurable."""
    baseline, candidate, home = (
        tmp_path / "baseline",
        tmp_path / "candidate",
        tmp_path / "home",
    )
    baseline.mkdir()
    candidate.mkdir()
    home.mkdir()
    environments = []
    payload = _payload()

    def run(_repo, _query, env, _search_args):
        environments.append(env)
        return retrieval_compare.SearchRun(payload=payload, seconds=1.0)

    monkeypatch.setattr(retrieval_compare, "_run_search", run)

    retrieval_compare.run_comparison(
        baseline_repo=baseline,
        candidate_repo=candidate,
        home=home,
        queries=["needle"],
        repeats=1,
        warmup=0,
        current_session_env=[
            "CLAUDE_CODE_SESSION_ID=claude-caller",
            "CODEX_THREAD_ID=codex-caller",
        ],
    )

    assert len(environments) == 6
    assert all(env["HOME"] == str(home) for env in environments)
    assert all(env["CLAUDE_CODE_SESSION_ID"] == "claude-caller" for env in environments)
    assert all(env["CODEX_THREAD_ID"] == "codex-caller" for env in environments)


def test_comparator_runs_only_product_commands_and_never_git(tmp_path, monkeypatch):
    baseline, candidate = tmp_path / "baseline", tmp_path / "candidate"
    baseline.mkdir()
    candidate.mkdir()
    (tmp_path / "home").mkdir()
    commands = []

    def product_command(command, **kwargs):
        commands.append((command, kwargs))
        return SimpleNamespace(returncode=0, stdout=json.dumps(_payload()), stderr="")

    monkeypatch.setattr(retrieval_compare.subprocess, "run", product_command)
    clock = iter(float(value) for value in range(64))
    monkeypatch.setattr(
        retrieval_compare.time,
        "perf_counter",
        lambda: next(clock),
    )

    retrieval_compare.run_comparison(
        baseline_repo=baseline,
        candidate_repo=candidate,
        home=tmp_path / "home",
        queries=["needle"],
        repeats=1,
        warmup=0,
    )

    assert len(commands) == 6
    assert all(
        command
        == [
            retrieval_compare.sys.executable,
            "-m",
            "session_browser.app",
            "search",
            "needle",
            "--mode",
            "ids",
            "--format",
            "json",
        ]
        for command, _kwargs in commands
    )
    assert {kwargs["cwd"] for _command, kwargs in commands} == {baseline, candidate}
    assert {kwargs["env"]["HOME"] for _command, kwargs in commands} == {
        str(tmp_path / "home")
    }


def test_warmup_runs_are_discarded_before_measuring(tmp_path, monkeypatch):
    """Cold runs must not reach the samples that decide ship or no-ship."""
    baseline, candidate = tmp_path / "baseline", tmp_path / "candidate"
    baseline.mkdir()
    candidate.mkdir()
    (tmp_path / "home").mkdir()
    payload = _payload()
    monkeypatch.setattr(
        retrieval_compare,
        "_run_search",
        _runner(
            # first entry is the signature run, next two are warmup, then measured
            {baseline: [payload] * 7, candidate: [payload] * 7},
            {
                baseline: [99.0] * 4 + [1.0, 1.0, 1.0],
                candidate: [99.0] * 4 + [1.0, 1.0, 1.0],
            },
            [],
        ),
    )

    report = retrieval_compare.run_comparison(
        baseline_repo=baseline,
        candidate_repo=candidate,
        home=tmp_path / "home",
        queries=["needle"],
        repeats=3,
        warmup=2,
    )

    assert report["queries"]["needle"]["baseline_samples"] == [1.0, 1.0, 1.0]
    assert report["queries"]["needle"]["candidate_samples"] == [1.0, 1.0, 1.0]
    assert report["warmup"] == 2


def test_a_single_stalled_run_does_not_move_the_ratio(tmp_path, monkeypatch):
    """One outlier sample is trimmed away instead of convicting the candidate."""
    baseline, candidate = tmp_path / "baseline", tmp_path / "candidate"
    baseline.mkdir()
    candidate.mkdir()
    (tmp_path / "home").mkdir()
    payload = _payload()
    monkeypatch.setattr(
        retrieval_compare,
        "_run_search",
        _runner(
            {baseline: [payload] * 7, candidate: [payload] * 7},
            {baseline: [1.0] * 7, candidate: [1.0] * 6 + [50.0]},
            [],
        ),
    )

    report = retrieval_compare.run_comparison(
        baseline_repo=baseline,
        candidate_repo=candidate,
        home=tmp_path / "home",
        queries=["needle"],
        repeats=5,
        warmup=0,
    )

    assert report["queries"]["needle"]["candidate_trimmed_median"] == 1.0
    assert report["queries"]["needle"]["ratio"] == 1.0
    assert report["passed"] is True


def test_slowdown_inside_the_noise_floor_is_unresolvable_not_a_failure(
    tmp_path, monkeypatch
):
    """A machine too noisy to measure the effect must not convict the candidate.

    Real corpus runs measured 2026-08-23 scatter by 1-21% of the median at the
    comparator's default sampling, and by 44-67% when given fewer than four
    samples, while the aggregate ratio sits on 1.00 -- so a fixed threshold on
    raw medians reports pure variance as a regression. The samples below swing
    harder still (relative_spread 1.17) to keep the branch under test unmissable
    on any machine this suite runs on; they are a deliberate exaggeration of the
    measured band, not a claim about it.
    """
    baseline, candidate = tmp_path / "baseline", tmp_path / "candidate"
    baseline.mkdir()
    candidate.mkdir()
    (tmp_path / "home").mkdir()
    payload = _payload()
    monkeypatch.setattr(
        retrieval_compare,
        "_run_search",
        _runner(
            {baseline: [payload] * 7, candidate: [payload] * 7},
            # both revisions swing wildly; the candidate's centre is 8% higher
            {
                baseline: [0.0, 0.0, 1.0, 1.5, 2.0, 2.5, 3.0],
                candidate: [0.0, 0.0, 1.1, 1.6, 2.16, 2.7, 3.2],
            },
            [],
        ),
    )

    report = retrieval_compare.run_comparison(
        baseline_repo=baseline,
        candidate_repo=candidate,
        home=tmp_path / "home",
        queries=["needle"],
        repeats=5,
        warmup=0,
    )

    result = report["queries"]["needle"]
    assert result["ratio"] > retrieval_compare.SLOWDOWN_LIMIT
    assert result["noise_floor"] > result["ratio"] - 1.0
    assert result["verdict"] == "unresolvable"
    assert report["unresolvable_queries"] == ["needle"]
    assert report["slow_queries"] == []


def test_slowdown_beyond_the_noise_floor_still_fails(tmp_path, monkeypatch):
    """A clean machine must still convict a real regression."""
    baseline, candidate = tmp_path / "baseline", tmp_path / "candidate"
    baseline.mkdir()
    candidate.mkdir()
    (tmp_path / "home").mkdir()
    payload = _payload()
    monkeypatch.setattr(
        retrieval_compare,
        "_run_search",
        _runner(
            {baseline: [payload] * 7, candidate: [payload] * 7},
            {baseline: [0.0, 0.0] + [1.0] * 5, candidate: [0.0, 0.0] + [1.2] * 5},
            [],
        ),
    )

    with pytest.raises(
        retrieval_compare.ComparatorError,
        match="slower than 5% beyond measurement noise",
    ):
        retrieval_compare.run_comparison(
            baseline_repo=baseline,
            candidate_repo=candidate,
            home=tmp_path / "home",
            queries=["needle"],
            repeats=5,
            warmup=0,
        )


def test_unresolvable_queries_are_reported_loudly(tmp_path, monkeypatch, capsys):
    """A query the machine cannot judge must never pass silently."""
    baseline, candidate = tmp_path / "baseline", tmp_path / "candidate"
    baseline.mkdir()
    candidate.mkdir()
    (tmp_path / "home").mkdir()
    payload = _payload()
    monkeypatch.setattr(
        retrieval_compare,
        "_run_search",
        _runner(
            {baseline: [payload] * 7, candidate: [payload] * 7},
            {
                baseline: [0.0, 0.0, 1.0, 1.5, 2.0, 2.5, 3.0],
                candidate: [0.0, 0.0, 1.1, 1.6, 2.16, 2.7, 3.2],
            },
            [],
        ),
    )

    retrieval_compare.run_comparison(
        baseline_repo=baseline,
        candidate_repo=candidate,
        home=tmp_path / "home",
        queries=["needle"],
        repeats=5,
        warmup=0,
    )

    assert "inside this machine's noise floor" in capsys.readouterr().err


def test_query_file_replaces_the_built_in_defaults(tmp_path, monkeypatch):
    """A supplied query set must not silently inherit the default queries.

    The defaults include a term that matches almost any live session, so
    extending rather than replacing made runs fail on an unrelated query.
    """
    queries = tmp_path / "queries.txt"
    queries.write_text("only mine\n")
    seen = {}
    monkeypatch.setattr(
        retrieval_compare,
        "run_comparison",
        lambda **kwargs: seen.update(kwargs) or {"passed": True},
    )

    retrieval_compare.main(
        [
            "--baseline-repo",
            str(tmp_path),
            "--candidate-repo",
            str(tmp_path),
            "--home",
            str(tmp_path),
            "--query-file",
            str(queries),
            "--report",
            str(tmp_path / "r.json"),
        ]
    )

    assert seen["queries"] == ["only mine"]
