"""The performance gate: exact work counts, checked on every test run.

Split into two layers, because they fail for different reasons and want
different responses.

The budget test compares every counter against ``docs/perf_budgets.json`` and
fails on any movement. It is deliberately unforgiving and deliberately dumb: it
knows nothing about what the numbers mean, only that they changed. Re-record
with ``python -m session_browser.perf_budget bless``.

The invariant tests below assert the *properties* the budgets exist to protect,
in terms that survive a re-bless. A blessed budget can quietly encode a
regression if nobody reads the diff; an invariant cannot. If someone routes the
CLI through the TUI's progress path and blesses the result, the budget test goes
quiet and ``test_cli_search_fires_no_progress_callbacks`` does not.
"""

from __future__ import annotations

import inspect
import json
import subprocess

import pytest

from session_browser import discovery, perf_budget, transcript
from session_browser.perf_budget import (
    compare,
    format_report,
    load_budgets,
    measure,
    rg_available,
)


@pytest.fixture(scope="module")
def measured(tmp_path_factory) -> dict[str, dict[str, int]]:
    """Every workload's counters, measured once for the whole module.

    Module-scoped because the numbers are counts: re-measuring per test would
    cost the corpus build again and could not produce a different answer.
    """
    return measure(tmp_path_factory.mktemp("perf"))


class TestBudgets:
    def test_no_workload_exceeds_its_recorded_budget(self, measured):
        budgets = load_budgets()
        assert budgets, (
            "docs/perf_budgets.json is missing or empty. Record it with "
            "`python -m session_browser.perf_budget bless`."
        )
        breaches, unbudgeted = compare(measured, budgets)
        report = format_report(
            breaches,
            unbudgeted,
            {w: b for w, b in _guards().items()},
            skipped=not rg_available(),
        )
        assert not breaches, "\n" + report

    def test_every_measured_workload_has_a_budget(self, measured):
        """An unbudgeted workload is not a passing one.

        Without this, adding a scenario and forgetting to bless leaves it
        measured, reported and enforcing nothing.
        """
        budgets = load_budgets()
        missing = sorted(set(measured) - set(budgets))
        assert not missing, (
            f"No budget recorded for {missing}. Run "
            "`python -m session_browser.perf_budget bless`."
        )

    def test_counts_are_reproducible(self, tmp_path_factory, measured):
        """The gate is worthless if it is noisy.

        A second independent measurement, against a freshly built corpus in a
        different directory, must agree exactly. If this ever fails the counters
        have picked up something environmental and the budgets below cannot be
        trusted until it is found.
        """
        again = measure(tmp_path_factory.mktemp("perf-again"))
        assert again == measured


def _guards() -> dict[str, str]:
    return {w.name: w.guards for w in perf_budget.workloads(perf_budget.WorkLedger())}


@pytest.fixture(scope="module")
def loop_counts(tmp_path_factory) -> dict[str, int]:
    return perf_budget.measure_loops(tmp_path_factory.mktemp("perf-loops"))


class TestLoopOpcodeBudgets:
    """The layer that sees inside a loop.

    Everything in TestBudgets counts work at the grain of a file, a statement,
    a parse, and is blind by construction to CPU added inside a loop that does
    the same I/O. These probes count executed Python opcodes over fixed inputs,
    which stays a count rather than a timing -- identical on every machine --
    while resolving per-iteration changes a stopwatch cannot.
    """

    def test_no_loop_exceeds_its_recorded_opcode_budget(self, loop_counts):
        budgets, recorded_python = perf_budget.load_loop_budgets()
        assert budgets, (
            "docs/perf_budgets.json records no loop opcode budgets. Record "
            "them with `python -m session_browser.perf_budget bless`."
        )
        if not perf_budget.loops_enforceable(recorded_python):
            pytest.skip(
                f"loop budgets recorded on Python {recorded_python}, running "
                f"{perf_budget.python_tag()}; opcode counts are interpreter "
                "specific. Re-bless on this version to enforce them."
            )
        breaches, unbudgeted = perf_budget.compare_loops(loop_counts, budgets)
        report = perf_budget.format_loop_report(
            breaches,
            unbudgeted,
            perf_budget._loop_guards(),
            recorded_python=recorded_python,
        )
        assert not breaches, "\n" + report

    def test_every_probe_has_a_budget(self, loop_counts):
        budgets, _ = perf_budget.load_loop_budgets()
        missing = sorted(set(loop_counts) - set(budgets))
        assert not missing, (
            f"No loop opcode budget recorded for {missing}. Run "
            "`python -m session_browser.perf_budget bless`."
        )

    def test_opcode_counts_are_reproducible(self, tmp_path_factory, loop_counts):
        """Same reasoning as test_counts_are_reproducible, one layer down.

        Opcode counting is the more fragile of the two mechanisms -- it traces
        the interpreter -- so it has to prove it is deterministic before any
        budget derived from it means anything.
        """
        again = perf_budget.measure_loops(tmp_path_factory.mktemp("perf-loops-again"))
        assert again == loop_counts

    def test_probes_actually_execute_their_loops(self, loop_counts):
        """A probe that silently stopped running would pass every check above.

        Each of these drives a loop over hundreds of items, so a count in the
        low hundreds means the body was never entered -- an empty input, a
        generator never drained, a renamed function caught by a try/except
        somewhere. The budget test cannot tell that apart from a legitimate
        optimisation; this can.
        """
        too_small = {name: n for name, n in loop_counts.items() if n < 1000}
        assert not too_small, (
            f"these probes executed almost no opcodes: {too_small}. The loop "
            "is not being exercised, so its budget is guarding nothing."
        )


class TestBlessedDelta:
    """The record a bless leaves behind in the diff.

    This block is what replaced the confirm modal that used to guard blessing.
    Its whole value is that it is *derived*: an agent that blesses a regression
    records the regression whether or not it says so. If it ever silently
    computed an empty movement, the gate would look defended and would not be,
    which is precisely the failure the modal was removed to avoid -- so the
    empty case is asserted here as explicitly as the moved one.
    """

    def test_a_rise_is_recorded_with_its_direction_and_size(self):
        blessed = perf_budget.blessed_delta(
            {"w": {"transcripts_parsed": 400}},
            {"w": {"transcripts_parsed": 500}},
            {},
            None,
        )
        assert blessed["moved"] == {"w.transcripts_parsed": "400 -> 500 (+25.0%)"}
        assert blessed["added"] == []

    def test_a_fall_is_recorded_too(self):
        """A fall is usually a win, but it is never silent.

        The usual cause of a drop is a probe that stopped exercising its loop,
        so it has to appear in the diff for the same reason a rise does.
        """
        blessed = perf_budget.blessed_delta(
            {"w": {"corpus_bytes_read": 1000}},
            {"w": {"corpus_bytes_read": 750}},
            {},
            None,
        )
        assert blessed["moved"] == {"w.corpus_bytes_read": "1000 -> 750 (-25.0%)"}

    def test_an_unchanged_baseline_records_no_movement(self):
        blessed = perf_budget.blessed_delta(
            {"w": {"transcripts_parsed": 400}},
            {"w": {"transcripts_parsed": 400}},
            {"probe": 900},
            {"probe": 900},
        )
        assert blessed["moved"] == {}
        assert blessed["added"] == []

    def test_a_new_workload_is_added_not_reported_as_movement(self):
        """Adding a scenario is normal, and must not read as a regression."""
        blessed = perf_budget.blessed_delta(
            {}, {"w": {"transcripts_parsed": 5}}, {}, {}
        )
        assert blessed["moved"] == {}
        assert blessed["added"] == ["w"]

    def test_loop_opcodes_are_diffed_when_they_were_remeasured(self):
        blessed = perf_budget.blessed_delta(
            {}, {}, {"parse_jsonl": 1000}, {"parse_jsonl": 1118}
        )
        assert blessed["moved"] == {"loop_opcodes.parse_jsonl": "1000 -> 1118 (+11.8%)"}

    def test_loops_not_re_measured_claim_no_movement(self):
        """``loops=None`` means this bless did not look at them.

        Reporting a change it never observed would put a number in the diff
        that no run produced -- the same defect as hand-editing the file.
        """
        blessed = perf_budget.blessed_delta({}, {}, {"parse_jsonl": 1000}, None)
        assert blessed["moved"] == {}
        assert blessed["added"] == []

    def test_a_counter_rising_from_zero_is_reported_without_a_percentage(self):
        """0 -> n has no meaningful percentage; it must not divide by zero."""
        blessed = perf_budget.blessed_delta(
            {"w": {"sqlite_connections": 0}},
            {"w": {"sqlite_connections": 3}},
            {},
            None,
        )
        assert blessed["moved"] == {"w.sqlite_connections": "0 -> 3"}


@pytest.mark.skipif(
    not rg_available(), reason="ripgrep absent: native candidate scan not exercised"
)
class TestPrefilterInvariants:
    """Properties that must hold whatever the budgets happen to say."""

    def test_a_term_in_no_session_parses_no_transcript(self, measured):
        counts = measured["cli.search.absent"]
        assert counts["transcripts_parsed"] == 0
        assert counts["opencode_sessions_parsed"] == 0

    def test_rare_term_parses_far_less_than_the_whole_corpus(self, measured):
        rare = measured["cli.search.rare"]["transcripts_parsed"]
        everything = measured["cli.search.common"]["transcripts_parsed"]
        assert rare * 10 < everything, (
            f"the prefilter parsed {rare} of {everything} sessions for a term "
            "present in a handful; it is no longer ruling the corpus out"
        )

    def test_native_scan_reads_less_of_the_corpus_than_the_fallback(self, measured):
        """Why ripgrep is worth a subprocess at all.

        Stated as a comparison rather than an absolute so it keeps meaning if
        the corpus is ever resized.
        """
        native = measured["cli.search.rare"]["corpus_bytes_read"]
        fallback = measured["cli.search.rare.no_rg"]["corpus_bytes_read"]
        assert native < fallback

    def test_candidate_files_are_not_read_twice_to_partition_them(self, measured):
        """The trap the fallback path documents and avoids.

        Partitioning candidates in the parent and then parsing them in the
        worker reads every file twice. One prefilter scan per file-backed
        session is the ceiling.
        """
        counts = measured["cli.search.rare.no_rg"]
        file_sessions = (
            perf_budget.CLAUDE_SESSIONS
            + perf_budget.CODEX_SESSIONS
            + perf_budget.PI_SESSIONS
        )
        assert counts["prefilter_file_scans"] == file_sessions


class TestProviderScoping:
    def test_searching_claude_opens_no_database(self, measured):
        if "cli.search.claude" not in measured:
            pytest.skip("ripgrep absent")
        assert measured["cli.search.claude"]["sqlite_connections"] == 0

    def test_searching_opencode_reads_no_transcript_files(self, measured):
        counts = measured["cli.search.opencode"]
        assert counts["corpus_bytes_read"] == 0
        assert counts["corpus_file_opens"] == 0


class TestMetadataPathsStayCheap:
    """Listing and statistics must never fall through to the parser."""

    @pytest.mark.parametrize("workload", ["cli.list", "cli.stats"])
    def test_no_transcript_is_parsed(self, measured, workload):
        assert measured[workload]["transcripts_parsed"] == 0

    def test_get_parses_exactly_one_transcript(self, measured):
        assert measured["cli.get"]["transcripts_parsed"] == 1


@pytest.mark.skipif(
    not rg_available(), reason="ripgrep absent: native candidate scan not exercised"
)
class TestTheCliDoesNotPayForTheTui:
    """The property the progress readout was restructured to preserve.

    A ``_ticking_rows``-style wrapper can be replaced by an inline check in a
    hot loop without any test noticing -- the output is identical and the cost
    is far below what this machine can time. These assertions notice.
    """

    def test_cli_search_fires_no_progress_callbacks(self, measured):
        assert measured["cli.search.rare"]["progress_callbacks"] == 0

    def test_the_readout_costs_the_tui_no_extra_corpus_reads(self, measured):
        """Reporting progress must observe the scan, not re-walk it."""
        cli = measured["cli.search.rare"]
        tui = measured["tui.search.rare"]
        assert tui["corpus_bytes_read"] == cli["corpus_bytes_read"]
        assert tui["corpus_file_opens"] == cli["corpus_file_opens"]

    def test_row_progress_stays_outside_the_scan_loop(self, measured):
        """The regression that prompted all of this, pinned structurally.

        The opencode prefilter scan reads the entire part table. Reporting its
        depth from an inline ``if progress is not None`` inside that loop costs
        the CLI hundreds of thousands of branch evaluations for a readout only
        the TUI displays -- and costs it invisibly, since the rewrite changes no
        output, no I/O and no other counter in this file. Wrapping the cursor
        instead keeps the loop byte-identical when nobody is watching. That the
        wrapper is applied for the TUI and never for the CLI is the property;
        assert it, because no timing on this machine can.
        """
        assert measured["cli.search.rare"]["row_tick_wrappers"] == 0
        assert measured["tui.search.rare"]["row_tick_wrappers"] == 1

    def test_progress_is_reported_per_session_not_per_entry(self, measured):
        """Callback volume must scale with sessions, never with content.

        A readout that ticks per entry or per database row is the regression
        this bound exists to catch; it would be invisible in every other test
        and unmeasurable on a stopwatch.
        """
        callbacks = measured["tui.search.rare"]["progress_callbacks"]
        assert 0 < callbacks <= perf_budget.OPENCODE_SESSIONS * 2


@pytest.mark.skipif(
    not rg_available(), reason="ripgrep absent: native candidate scan not exercised"
)
class TestProcessRouting:
    def test_snippet_shape_above_the_threshold_goes_to_processes(self, measured):
        routed = measured["lib.search.process_route"]["process_pool_sessions"]
        assert routed >= transcript._PROC_MIN_CANDIDATES

    def test_entry_retaining_shape_stays_in_process(self, measured):
        """Whole transcripts must not be pickled back from worker processes."""
        assert measured["lib.search.keep_entries"]["process_pool_sessions"] == 0


class TestTheCorpusCoversEveryCodexEra:
    """The workloads can only guard shapes the corpus actually contains.

    A budget cannot tell "discovery stopped recognising paginated rollouts"
    from "the corpus never had one". Every rollout here used to be written in
    Codex's legacy vocabulary, which left two of its three eras unexercised --
    and left ``cli.list``'s guard, that read volume stays bounded by the
    discovery window rather than by session length, unfalsifiable as well:
    every rollout was smaller than that window, so a scan that recognised
    nothing and widened to the whole file read exactly the same bytes as one
    that stopped at the first turn it found.
    """

    @pytest.fixture(scope="class")
    @classmethod
    def rollouts(cls, tmp_path_factory) -> list:
        # classmethod because the fixture is class-scoped: pytest 9.1 warns
        # that an instance-method fixture runs once per class while each test
        # gets a fresh instance, and pytest 10 removes it.
        home = perf_budget.build_corpus(tmp_path_factory.mktemp("perf-corpus"))
        return sorted((home / ".codex" / "sessions").rglob("rollout-*.jsonl"))

    @staticmethod
    def _era(path) -> str:
        """The era a rollout is written in, read back off disk.

        Detected from the file rather than recomputed from the index, so this
        checks the corpus builder instead of restating it.
        """
        record = json.loads(path.read_text().split("\n")[1])
        payload = record["payload"]
        if record["type"] == "response_item":
            return "response_item"
        return "legacy" if payload["type"] == "user_message" else "paginated"

    # Named here rather than read from ``perf_budget.CODEX_ERAS``: comparing
    # the corpus against the constant that shapes it passes whatever the
    # constant says, including a constant narrowed back to one era.
    ERAS = frozenset({"legacy", "paginated", "response_item"})

    def test_all_three_eras_are_present(self, rollouts):
        assert {self._era(p) for p in rollouts} == self.ERAS

    def test_both_scans_recognise_a_turn_in_every_rollout(self, rollouts):
        """The failure that hid in the real corpus, asserted directly.

        An unrecognised vocabulary costs a blank summary and a blank last
        activity, and costs them expensively: the head scan runs to the end of
        the file looking for a user record, and the tail scan quadruples its
        window until the whole file is read. 106 MB and 93 MB respectively on
        the real corpus, both to return "".
        """
        blank = [
            (p.name, self._era(p))
            for p in rollouts
            if not discovery._codex_first_user_message(p)
            or not discovery._last_activity_iso(p, "codex")
        ]
        assert not blank, f"a scan found no turn in {blank}"

    def test_every_era_has_a_rollout_larger_than_the_tail_window(self, rollouts):
        """Otherwise the window bound is pinned by nothing.

        Below the initial window a file is read whole either way, so the
        widening a lost turn causes never reaches a counter.
        """
        window = (
            inspect.signature(discovery._last_activity_iso).parameters["window"].default
        )
        deep = {self._era(p) for p in rollouts if p.stat().st_size > window}
        assert deep == self.ERAS


class TestTheRecordSurvivesReBlessing:
    """The ``_blessed`` block must describe the diff, not the last bless.

    It used to be computed against the working copy, so a second bless
    compared the measurement against numbers the first bless had already
    written and recorded an empty movement. The commit then carried the new
    counts with a block asserting that nothing moved — no hand-edit, no flag,
    no warning, and the one mechanism standing between a quiet regression and
    review said the regression had not happened. Reached by accident, by
    adjusting a guard string (they live in the same generated file) and
    re-recording.
    """

    @staticmethod
    def _repo(tmp_path, budgets: dict):
        root = tmp_path / "repo"
        (root / "docs").mkdir(parents=True)
        path = root / "docs" / "perf_budgets.json"
        path.write_text(json.dumps(budgets, indent=2) + "\n")
        for command in (
            ["git", "init", "--quiet", "-b", "main"],
            ["git", "config", "user.email", "t@example.test"],
            ["git", "config", "user.name", "T"],
            ["git", "add", "-A"],
            ["git", "commit", "--quiet", "-m", "baseline"],
        ):
            subprocess.run(command, cwd=root, check=True, capture_output=True)
        return path

    def test_a_committed_file_is_read_from_head_not_the_working_copy(
        self, tmp_path, monkeypatch
    ):
        path = self._repo(tmp_path, {"workloads": {"w": {"transcripts_parsed": 400}}})
        path.write_text(json.dumps({"workloads": {"w": {"transcripts_parsed": 1}}}))
        monkeypatch.setattr(perf_budget, "BUDGETS_PATH", path)
        workloads, _ = perf_budget.committed_baseline()
        assert workloads == {"w": {"transcripts_parsed": 400}}

    def test_no_committed_version_falls_back_rather_than_failing(
        self, tmp_path, monkeypatch
    ):
        """First bless in a fresh checkout, or outside git entirely."""
        loose = tmp_path / "loose.json"
        loose.write_text("{}")
        monkeypatch.setattr(perf_budget, "BUDGETS_PATH", loose)
        assert perf_budget.committed_baseline() is None

    def test_blessing_twice_records_the_movement_both_times(
        self, tmp_path, monkeypatch
    ):
        """The regression this class exists for.

        Two blesses of the same measurement against one commit. The second
        must say exactly what the first said, because the diff it describes is
        the same diff.
        """
        path = self._repo(tmp_path, {"workloads": {"w": {"transcripts_parsed": 400}}})
        monkeypatch.setattr(perf_budget, "BUDGETS_PATH", path)
        measured = {"w": dict.fromkeys(perf_budget.COUNTERS, 0)}
        measured["w"]["transcripts_parsed"] = 500

        first = perf_budget.save_budgets(measured, {}, loops={}, loop_guards={})
        second = perf_budget.save_budgets(measured, {}, loops={}, loop_guards={})

        assert first["moved"] == {"w.transcripts_parsed": "400 -> 500 (+25.0%)"}
        assert second["moved"] == first["moved"]
        written = json.loads(path.read_text())
        assert written["_blessed"]["moved"] == first["moved"]

    def test_the_merge_still_uses_the_working_copy(self, tmp_path, monkeypatch):
        """Only the *record* moved to HEAD; the merge must not.

        Blessing on a machine without ripgrep measures a subset, and the
        unmeasured workloads are preserved from the file on disk. Taking them
        from HEAD instead would resurrect numbers a previous bless had already
        superseded.
        """
        path = self._repo(tmp_path, {"workloads": {"kept": {"transcripts_parsed": 7}}})
        monkeypatch.setattr(perf_budget, "BUDGETS_PATH", path)
        perf_budget.save_budgets(
            {
                "kept": dict(
                    dict.fromkeys(perf_budget.COUNTERS, 0), transcripts_parsed=9
                )
            },
            {},
            loops={},
            loop_guards={},
        )
        # A second bless that measures a different workload entirely must carry
        # the 9 forward, not fall back to the committed 7.
        perf_budget.save_budgets(
            {"other": dict.fromkeys(perf_budget.COUNTERS, 0)},
            {},
            loops={},
            loop_guards={},
        )
        written = json.loads(path.read_text())
        assert written["workloads"]["kept"]["transcripts_parsed"] == 9
