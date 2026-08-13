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

import pytest

from session_browser import perf_budget, transcript
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
        file_sessions = perf_budget.CLAUDE_SESSIONS + perf_budget.CODEX_SESSIONS
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
