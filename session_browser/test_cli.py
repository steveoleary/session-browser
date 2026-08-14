"""End-to-end CLI tests: argument parsing, exit codes, stdout/stderr shapes."""

from __future__ import annotations

import json
from datetime import UTC
from pathlib import Path

import pytest

from session_browser.cli import run_cli
from session_browser.discovery import Session


def write_claude(path: Path, texts: list[str]) -> None:
    lines = [
        json.dumps(
            {
                "type": "user",
                "message": {"content": t},
                "timestamp": "2026-06-01T10:00:00Z",
            }
        )
        for t in texts
    ]
    path.write_text("\n".join(lines) + "\n")


def write_codex(path: Path, texts: list[str]) -> None:
    lines = [
        json.dumps(
            {"type": "user_message", "message": t, "timestamp": "2026-06-05T10:00:00Z"}
        )
        for t in texts
    ]
    path.write_text("\n".join(lines) + "\n")


@pytest.fixture
def sessions(tmp_path):
    fa = tmp_path / "aaa.jsonl"
    write_claude(fa, ["alpha wombat message", "second wombat"])
    fb = tmp_path / "bbb.jsonl"
    write_codex(fb, ["beta quokka message"])
    fc = tmp_path / "ccc.jsonl"
    write_claude(fc, ["gamma wombat note"])
    return [
        Session(
            id="aaa",
            provider="claude",
            summary="alpha",
            cwd="/home/u/projA",
            repository="org/alpha",
            branch="main",
            updated_at="2026-06-01T10:00:00+00:00",
            content_path=str(fa),
        ),
        Session(
            id="bbb",
            provider="codex",
            summary="beta",
            cwd="/home/u/projB",
            repository="org/beta",
            branch="dev",
            updated_at="2026-06-05T10:00:00+00:00",
            content_path=str(fb),
        ),
        Session(
            id="ccc",
            provider="claude",
            summary="gamma",
            cwd="/home/u/projC",
            repository="org/gamma",
            branch="main",
            created_at="2026-06-09T08:00:00+00:00",
            updated_at="2026-06-09T10:00:00+00:00",
            content_path=str(fc),
        ),
    ]


@pytest.fixture
def cli(monkeypatch, capsys, sessions):
    """Run the CLI against the fixture sessions; returns (code, stdout, stderr)."""

    def run(*argv: str):
        monkeypatch.setattr(
            "session_browser.cli.discover_all", lambda *a, **k: sessions
        )
        code = run_cli(list(argv))
        captured = capsys.readouterr()
        return code, captured.out, captured.err

    return run


class TestList:
    def test_json_default_most_recent_first(self, cli):
        code, out, err = cli("list")
        assert code == 0 and err == ""
        data = json.loads(out)
        assert [s["id"] for s in data["sessions"]] == [
            "claude:ccc",
            "codex:bbb",
            "claude:aaa",
        ]
        first = data["sessions"][0]
        assert first["provider"] == "claude"
        assert first["session_id"] == "ccc"
        assert first["cwd"] == "/home/u/projC"
        assert first["updated_at"] == "2026-06-09T10:00:00+00:00"

    def test_provider_filter(self, cli):
        _, out, _ = cli("list", "--provider", "claude")
        assert [s["id"] for s in json.loads(out)["sessions"]] == [
            "claude:ccc",
            "claude:aaa",
        ]

    def test_provider_filter_restricts_discovery(self, sessions, monkeypatch, capsys):
        hints = []

        def fake(providers=None):
            hints.append(providers)
            return sessions

        monkeypatch.setattr("session_browser.cli.discover_all", fake)
        assert run_cli(["list", "--provider", "claude"]) == 0
        capsys.readouterr()
        assert hints == [["claude"]]

    def test_repo_and_cwd_substring_filters_case_insensitive(self, cli):
        _, out, _ = cli("list", "--repo", "ALPHA")
        assert [s["id"] for s in json.loads(out)["sessions"]] == ["claude:aaa"]
        _, out, _ = cli("list", "--cwd", "projb")
        assert [s["id"] for s in json.loads(out)["sessions"]] == ["codex:bbb"]

    def test_repo_miss_with_unpopulated_sessions_warns(self, cli, sessions):
        """An empty result and an empty *field* must not look alike.

        This flag returned an empty set for every input for as long as it
        existed, because no scanner ever populated the field it reads. A
        caller cannot tell that apart from "no sessions in that repo", so a
        session with nothing to match against says so.
        """
        sessions[0].repository = ""
        _, out, _ = cli("list", "--repo", "nosuchproject")
        data = json.loads(out)
        assert data["sessions"] == []
        warning = next(w for w in data["warnings"] if "--repo matched nothing" in w)
        assert "1 session(s) have no recorded project name" in warning
        assert "--cwd" in warning

    def test_repo_miss_with_every_session_populated_stays_quiet(self, cli):
        """No blanks means the empty result is the real answer."""
        _, out, _ = cli("list", "--repo", "nosuchproject")
        data = json.loads(out)
        assert data["sessions"] == []
        assert not any(
            "no recorded project name" in w for w in data.get("warnings", [])
        )

    def test_date_filters_inclusive_boundaries(self, cli):
        _, out, _ = cli("list", "--since", "2026-06-05")
        assert [s["id"] for s in json.loads(out)["sessions"]] == [
            "claude:ccc",
            "codex:bbb",
        ]
        _, out, _ = cli("list", "--until", "2026-06-05")  # whole day included
        assert [s["id"] for s in json.loads(out)["sessions"]] == [
            "codex:bbb",
            "claude:aaa",
        ]
        _, out, _ = cli("list", "--since", "2026-06-02", "--until", "2026-06-08")
        assert [s["id"] for s in json.loads(out)["sessions"]] == ["codex:bbb"]

    def test_limit(self, cli):
        _, out, _ = cli("list", "--limit", "1")
        assert [s["id"] for s in json.loads(out)["sessions"]] == ["claude:ccc"]

    def test_text_format(self, cli):
        code, out, _ = cli("list", "--format", "text", "--limit", "1")
        assert code == 0
        assert out.splitlines()[0].startswith("claude:ccc\t")

    def test_invalid_date_errors_structured(self, cli):
        code, out, err = cli("list", "--since", "junk")
        assert code == 1 and out == ""
        assert json.loads(err)["error"]["code"] == "invalid_date"

    def test_entry_count_and_duration_triage_signals(self, cli):
        _, out, _ = cli("list")
        by_id = {s["id"]: s for s in json.loads(out)["sessions"]}
        assert by_id["claude:aaa"]["total_entries"] == 2
        assert by_id["codex:bbb"]["total_entries"] == 1
        assert by_id["claude:ccc"]["duration_seconds"] == 7200
        # No created_at recorded: the span cannot be derived.
        assert by_id["claude:aaa"]["duration_seconds"] is None

    def test_unreadable_transcript_counts_null(self, sessions, monkeypatch, capsys):
        sessions.append(
            Session(
                id="zzz",
                provider="claude",
                updated_at="2026-06-10T10:00:00+00:00",
                content_path="/nonexistent/zzz.jsonl",
            )
        )
        monkeypatch.setattr(
            "session_browser.cli.discover_all", lambda *a, **k: sessions
        )
        assert run_cli(["list"]) == 0
        out = capsys.readouterr().out
        by_id = {s["id"]: s for s in json.loads(out)["sessions"]}
        assert by_id["claude:zzz"]["total_entries"] is None
        assert by_id["claude:aaa"]["total_entries"] == 2

    def test_text_format_carries_entries_and_duration_columns(self, cli):
        _, out, _ = cli("list", "--format", "text", "--limit", "1")
        fields = out.splitlines()[0].split("\t")
        assert fields[0] == "claude:ccc"
        assert fields[2] == "1"  # entry count
        assert fields[3] == "2h00m"  # duration
        assert fields[4] == "/home/u/projC"


class TestAround:
    def test_window_filters_excludes_anchor_sorts_nearest_first(self, cli):
        code, out, _err = cli("list", "--around", "codex:bbb", "--window", "1w")
        assert code == 0
        data = json.loads(out)
        # aaa and ccc are both 4d from bbb — the tie breaks by recency; the
        # anchor itself is excluded.
        assert [s["id"] for s in data["sessions"]] == ["claude:ccc", "claude:aaa"]
        assert {s["id"]: s["offset"] for s in data["sessions"]} == {
            "claude:ccc": "+4d00h",
            "claude:aaa": "-4d00h",
        }

    def test_default_window_is_one_day(self, cli):
        _, out, _ = cli("list", "--around", "codex:bbb")
        assert json.loads(out)["sessions"] == []

    def test_window_accepts_leading_dash(self, cli):
        _, out, _ = cli("list", "--around", "codex:bbb", "--window", "-1w")
        assert [s["id"] for s in json.loads(out)["sessions"]] == [
            "claude:ccc",
            "claude:aaa",
        ]

    def test_composes_with_provider_filter(self, cli):
        _, out, _ = cli(
            "list", "--around", "codex:bbb", "--window", "1w", "--provider", "claude"
        )
        assert [s["id"] for s in json.loads(out)["sessions"]] == [
            "claude:ccc",
            "claude:aaa",
        ]

    def test_anchor_provider_joins_discovery_hint(self, sessions, monkeypatch, capsys):
        hints = []

        def fake(providers=None):
            hints.append(providers)
            return sessions

        monkeypatch.setattr("session_browser.cli.discover_all", fake)
        assert (
            run_cli(
                [
                    "list",
                    "--provider",
                    "claude",
                    "--around",
                    "codex:bbb",
                    "--window",
                    "1w",
                ]
            )
            == 0
        )
        capsys.readouterr()
        assert hints == [["claude", "codex"]]

    def test_anchor_prefix_resolves(self, cli):
        _, out, _ = cli("list", "--around", "bb", "--window", "1w")
        assert [s["id"] for s in json.loads(out)["sessions"]] == [
            "claude:ccc",
            "claude:aaa",
        ]

    def test_text_format_carries_offset_column(self, cli):
        _, out, _ = cli(
            "list", "--around", "codex:bbb", "--window", "1w", "--format", "text"
        )
        first = out.splitlines()[0].split("\t")
        assert first[0] == "claude:ccc"
        assert first[2] == "+4d00h"

    def test_search_composes_and_results_carry_offset(self, cli):
        _, out, _ = cli(
            "search",
            "wombat",
            "--around",
            "codex:bbb",
            "--window",
            "1w",
            "--mode",
            "ids",
        )
        data = json.loads(out)
        assert data["filters"]["around"] == "codex:bbb"
        assert data["filters"]["window"] == "1w"
        assert [r["id"] for r in data["results"]] == ["claude:ccc", "claude:aaa"]
        assert [r["offset"] for r in data["results"]] == ["+4d00h", "-4d00h"]

    def test_window_without_around_errors(self, cli):
        code, _, err = cli("list", "--window", "1d")
        assert code == 1
        assert json.loads(err)["error"]["code"] == "invalid_filter"

    def test_around_with_since_errors(self, cli):
        code, _, err = cli("list", "--around", "codex:bbb", "--since", "-1d")
        assert code == 1
        assert json.loads(err)["error"]["code"] == "invalid_filter"

    def test_invalid_window_errors(self, cli):
        code, _, err = cli("list", "--around", "codex:bbb", "--window", "junk")
        assert code == 1
        assert json.loads(err)["error"]["code"] == "invalid_window"

    def test_unknown_anchor_errors(self, cli):
        code, _, err = cli("list", "--around", "zzz")
        assert code == 1
        assert json.loads(err)["error"]["code"] == "unknown_session"


class TestListSortAndTruncation:
    def test_sort_oldest_reverses_recency(self, cli):
        _, out, _ = cli("list", "--sort", "oldest")
        assert [s["id"] for s in json.loads(out)["sessions"]] == [
            "claude:aaa",
            "codex:bbb",
            "claude:ccc",
        ]

    def test_limit_truncation_is_loud(self, cli):
        _, out, _ = cli("list", "--limit", "1")
        data = json.loads(out)
        assert [s["id"] for s in data["sessions"]] == ["claude:ccc"]
        w = [w for w in data["warnings"] if "--limit 1 dropped" in w]
        assert len(w) == 1
        assert "dropped 2 session(s)" in w[0]
        assert "2026-06-01" in w[0]  # oldest dropped session's date
        assert "oldest sessions were dropped first" in w[0]

    def test_hint_names_only_sorts_list_actually_has(self, cli):
        """The bug this warning exists to prevent: pointing the reader at
        `--sort matches`, which only `search` accepts."""
        _, out, _ = cli("list", "--limit", "1")
        w = next(w for w in json.loads(out)["warnings"] if "dropped" in w)
        assert "--sort oldest" in w
        assert "--sort matches" not in w

    def test_no_truncation_no_warning(self, cli):
        _, out, _ = cli("list", "--limit", "5")
        assert not any("dropped" in w for w in json.loads(out).get("warnings", []))

    def test_truncation_warning_reaches_stderr_in_text_mode(self, cli):
        _, _out, err = cli("list", "--limit", "1", "--format", "text")
        assert "--limit 1 dropped 2 session(s)" in err

    def test_around_keeps_nearest_first_over_sort(self, cli):
        """--around answers "what was next to this", so its nearest-first
        order outranks --sort rather than being reversed by it."""
        ids = []
        for order in ("recent", "oldest"):
            _, out, _ = cli("list", "--around", "claude:ccc", "--sort", order)
            ids.append([s["id"] for s in json.loads(out)["sessions"]])
        assert ids[0] == ids[1]

    def test_limit_applied_before_entry_counts(self, cli, monkeypatch):
        """Deferring the limit must not widen the I/O: a listing reads
        exactly as many transcripts as it prints."""
        # Local import: the module-level name `cli` is the fixture.
        from session_browser import cli as cli_module

        counted = []
        real = cli_module._entry_count
        monkeypatch.setattr(
            "session_browser.cli._entry_count",
            lambda s: (counted.append(s.id), real(s))[1],
        )
        cli("list", "--limit", "1")
        assert counted == ["ccc"]


class TestExcludeCwd:
    """--exclude-cwd: the only negative metadata filter. Opt-in, repeatable,
    and applied before --limit so scratch sessions cannot eat the budget."""

    def test_absent_flag_changes_nothing(self, cli):
        """The not-default guarantee: silently dropping results is the failure
        this filter is written to avoid, so exclusion never happens
        uninvited."""
        _, out, _ = cli("list")
        assert [s["id"] for s in json.loads(out)["sessions"]] == [
            "claude:ccc",
            "codex:bbb",
            "claude:aaa",
        ]

    def test_repeatable_case_insensitive_and_ored(self, cli):
        _, out, _ = cli("list", "--exclude-cwd", "projA", "--exclude-cwd", "PROJC")
        assert [s["id"] for s in json.loads(out)["sessions"]] == ["codex:bbb"]

    def test_composes_with_positive_cwd_as_intersection(self, cli):
        """--cwd narrows, --exclude-cwd then subtracts; not a union, not an
        override."""
        _, out, _ = cli("list", "--cwd", "/home/u", "--exclude-cwd", "projA")
        assert [s["id"] for s in json.loads(out)["sessions"]] == [
            "claude:ccc",
            "codex:bbb",
        ]

    def test_applied_before_limit_and_entry_count_io(self, cli, monkeypatch):
        """The whole point of the issue. If exclusion ran after --limit, the
        noise would still consume the budget and the feature would look
        implemented while doing nothing. Also pins that excluded rows never
        have their transcripts opened."""
        from session_browser import cli as cli_module

        counted = []
        real = cli_module._entry_count
        monkeypatch.setattr(
            "session_browser.cli._entry_count",
            lambda s: (counted.append(s.id), real(s))[1],
        )
        _, out, _ = cli("list", "--exclude-cwd", "projC", "--limit", "1")
        assert [s["id"] for s in json.loads(out)["sessions"]] == ["codex:bbb"]
        assert counted == ["bbb"]

    def test_search_excludes_candidates_before_the_scan(self, cli, monkeypatch):
        """Excluded sessions must not reach the transcript scan — that is
        where search's cost lives, so this is the perf property."""
        from session_browser import cli as cli_module

        seen = []
        real = cli_module.search_sessions

        def spy(candidates, queries, **kw):
            seen.extend(s.id for s in candidates)
            return real(candidates, queries, **kw)

        monkeypatch.setattr("session_browser.cli.search_sessions", spy)
        cli("search", "wombat", "--mode", "ids", "--exclude-cwd", "projA")
        assert "aaa" not in seen and "ccc" in seen

    def test_missing_cwd_is_retained(self, sessions, monkeypatch, capsys):
        """A session with no recorded cwd has nothing to match against, so a
        non-empty pattern never removes it. --here is the flag that drops
        those, and it says so."""
        rows = sessions + [
            Session(
                id="nocwd",
                provider="claude",
                updated_at="2026-06-10T10:00:00+00:00",
            )
        ]
        monkeypatch.setattr("session_browser.cli.discover_all", lambda *a, **k: rows)
        run_cli(["list", "--exclude-cwd", "projA"])
        out = capsys.readouterr().out
        assert "claude:nocwd" in [s["id"] for s in json.loads(out)["sessions"]]

    def test_around_anchor_resolves_even_when_excluded(self, cli):
        """Anchor lookup reads the unfiltered discovery set, so excluding the
        anchor's own cwd must not make it unfindable."""
        code, out, _ = cli(
            "list",
            "--around",
            "claude:aaa",
            "--window",
            "30d",
            "--exclude-cwd",
            "projA",
        )
        assert code == 0
        assert [s["id"] for s in json.loads(out)["sessions"]] == [
            "codex:bbb",
            "claude:ccc",
        ]

    @pytest.mark.parametrize("value", ["", "   "])
    @pytest.mark.parametrize("command", ["list", "search", "stats"])
    def test_blank_pattern_is_a_structured_error(self, cli, command, value):
        """An empty substring is contained in every string, so a silent no-op
        here would return an empty corpus and read as 'nothing matched'."""
        argv = [command] + (["wombat"] if command == "search" else [])
        # stats defaults to text; ask all three for json so one assertion holds.
        code, _, err = cli(*argv, "--format", "json", "--exclude-cwd", value)
        assert code == 1
        assert json.loads(err)["error"]["code"] == "invalid_filter"

    def test_warns_with_count_and_patterns(self, cli):
        _, out, _ = cli("list", "--exclude-cwd", "projA")
        warnings = json.loads(out)["warnings"]
        assert any("removed 1 session(s) matching proja" in w for w in warnings)

    def test_count_is_measured_against_the_filtered_view(self, cli):
        """Regression: the block first sat above --since, so it counted
        sessions the caller was never going to see — reporting `removed 2` on
        a listing that lost 1. Membership was right, the number was not, and
        an overstated warning misleads exactly like a silent one."""
        _, out, _ = cli(
            "list", "--since", "2026-06-04", "--exclude-cwd", "/home/u/proj"
        )
        data = json.loads(out)
        # aaa (06-01) is outside the window; only bbb and ccc were on offer.
        assert data["sessions"] == []
        assert any("removed 2 session(s)" in w for w in data["warnings"])

    def test_no_warning_when_nothing_matched(self, cli):
        _, out, _ = cli("list", "--exclude-cwd", "nosuchdir")
        assert "warnings" not in json.loads(out)

    def test_filters_echo_records_the_patterns(self, cli):
        """search/stats JSON echo every filter; an omitted key would make the
        artifact misreport what actually ran."""
        _, out, _ = cli("search", "wombat", "--mode", "ids", "--exclude-cwd", "projA")
        assert json.loads(out)["filters"]["exclude_cwd"] == ["projA"]


class TestListDiagnostics:
    def test_unreadable_and_empty_rows_flagged_in_warnings(
        self, sessions, tmp_path, monkeypatch, capsys
    ):
        hollow_file = tmp_path / "hollow.jsonl"
        hollow_file.write_text("")
        rows = sessions + [
            Session(
                id="bad",
                provider="claude",
                updated_at="2026-06-07T10:00:00+00:00",
                content_path="/nonexistent/bad.jsonl",
            ),
            Session(
                id="hollow",
                provider="claude",
                updated_at="2026-06-08T10:00:00+00:00",
                content_path=str(hollow_file),
            ),
        ]
        monkeypatch.setattr("session_browser.cli.discover_all", lambda *a, **k: rows)
        assert run_cli(["list"]) == 0
        data = json.loads(capsys.readouterr().out)
        warns = " ".join(data["warnings"])
        assert "claude:bad" in warns and "unreadable" in warns
        assert "claude:hollow" in warns and "zero entries" in warns

    def test_healthy_sessions_produce_no_diagnostics(self, cli):
        _, out, err = cli("list")
        assert "warnings" not in json.loads(out) and err == ""


class TestGetBatch:
    def test_json_wraps_sessions_in_order(self, cli):
        code, out, _ = cli("get", "claude:aaa", "codex:bbb", "--format", "json")
        assert code == 0
        data = json.loads(out)
        assert [p["session"]["id"] for p in data["sessions"]] == [
            "claude:aaa",
            "codex:bbb",
        ]
        assert "skipped" not in data

    def test_single_id_keeps_flat_shape(self, cli):
        _, out, _ = cli("get", "claude:aaa", "--format", "json")
        assert json.loads(out)["session"]["id"] == "claude:aaa"

    def test_text_concatenates_with_separator(self, cli):
        _, out, _ = cli("get", "claude:aaa", "codex:bbb")
        assert "# Session claude:aaa" in out
        assert "# Session codex:bbb" in out
        assert "\n---\n" in out

    def test_duplicate_ids_dedupe(self, cli):
        _, out, _ = cli("get", "claude:aaa", "aaa", "--format", "json")
        data = json.loads(out)
        assert [p["session"]["id"] for p in data["sessions"]] == ["claude:aaa"]

    def test_output_with_multiple_ids_errors(self, cli, tmp_path):
        code, _, err = cli(
            "get", "claude:aaa", "codex:bbb", "--output", str(tmp_path / "x.md")
        )
        assert code == 1 and "single transcript" in err

    def test_batch_skips_unreadable_and_continues(self, sessions, monkeypatch, capsys):
        broken = sessions + [
            Session(id="ddd", provider="claude", content_path="/nonexistent/ddd.jsonl")
        ]
        monkeypatch.setattr("session_browser.cli.discover_all", lambda *a, **k: broken)
        code = run_cli(["get", "claude:aaa", "claude:ddd", "--format", "json"])
        data = json.loads(capsys.readouterr().out)
        assert code == 0
        assert [p["session"]["id"] for p in data["sessions"]] == ["claude:aaa"]
        assert data["skipped"][0]["id"] == "claude:ddd"

    def test_all_unreadable_is_error(self, sessions, monkeypatch, capsys):
        broken = sessions + [
            Session(id="ddd", provider="claude", content_path="/nonexistent/1.jsonl"),
            Session(id="eee", provider="claude", content_path="/nonexistent/2.jsonl"),
        ]
        monkeypatch.setattr("session_browser.cli.discover_all", lambda *a, **k: broken)
        code = run_cli(["get", "ddd", "eee"])
        err = capsys.readouterr().err
        assert code == 1 and "no readable sessions" in err


class TestNearMissSuggestions:
    REAL = "d753769a-f162-4b2d-afd3-42d257893e57"
    TYPO = "d753769a-f162-4b2d-af62-42d257893e57"

    @pytest.fixture
    def longcli(self, sessions, tmp_path, monkeypatch, capsys):
        f = tmp_path / "long.jsonl"
        write_claude(f, ["hello"])
        rows = sessions + [
            Session(
                id=self.REAL,
                provider="claude",
                updated_at="2026-06-11T10:00:00+00:00",
                content_path=str(f),
            )
        ]

        def run(*argv: str):
            monkeypatch.setattr(
                "session_browser.cli.discover_all", lambda *a, **k: rows
            )
            code = run_cli(list(argv))
            cap = capsys.readouterr()
            return code, cap.out, cap.err

        return run

    def test_mistyped_full_id_suggests_near_match(self, longcli):
        code, _, err = longcli("get", f"claude:{self.TYPO}", "--format", "json")
        assert code == 1
        error = json.loads(err)["error"]
        assert error["code"] == "unknown_session"
        assert f"claude:{self.REAL}" in error["message"]
        assert error["details"]["suggestions"] == [f"claude:{self.REAL}"]

    def test_short_unknown_id_gets_no_suggestions(self, longcli):
        code, _, err = longcli("get", "zzz", "--format", "json")
        error = json.loads(err)["error"]
        assert code == 1 and "did you mean" not in error["message"]

    def test_unrelated_long_id_gets_no_suggestions(self, longcli):
        code, _, err = longcli(
            "get", "claude:ffffffff-0000-1111-2222", "--format", "json"
        )
        error = json.loads(err)["error"]
        assert code == 1 and "did you mean" not in error["message"]


class TestClip:
    @pytest.fixture
    def bigcli(self, tmp_path, monkeypatch, capsys):
        f = tmp_path / "big.jsonl"
        write_claude(f, ["small entry", "B" * 10000])
        ss = [
            Session(
                id="big",
                provider="claude",
                summary="big",
                cwd="/x",
                updated_at="2026-06-01T10:00:00+00:00",
                content_path=str(f),
            )
        ]

        def run(*argv: str):
            monkeypatch.setattr("session_browser.cli.discover_all", lambda *a, **k: ss)
            code = run_cli(list(argv))
            cap = capsys.readouterr()
            return code, cap.out, cap.err

        return run

    def test_stdout_clips_by_default_with_marker(self, bigcli):
        code, out, _ = bigcli("get", "claude:big")
        assert code == 0
        assert "B" * 4000 in out
        assert "B" * 4001 not in out
        assert "clipped 6000 chars" in out
        assert "small entry" in out  # short entries untouched

    def test_clip_zero_disables(self, bigcli):
        _, out, _ = bigcli("get", "claude:big", "--clip", "0")
        assert "B" * 10000 in out and "clipped" not in out

    def test_json_carries_clip_and_clipped_text(self, bigcli):
        _, out, _ = bigcli("get", "claude:big", "--format", "json")
        data = json.loads(out)
        assert data["clip"] == 4000
        assert len(data["entries"][1]["text"]) < 4100

    def test_output_file_complete_by_default(self, bigcli, tmp_path):
        dest = tmp_path / "full.md"
        code, _, _ = bigcli("get", "claude:big", "--output", str(dest))
        assert code == 0
        assert "B" * 10000 in dest.read_text()

    def test_explicit_clip_applies_even_with_output(self, bigcli, tmp_path):
        dest = tmp_path / "clipped.md"
        bigcli("get", "claude:big", "--output", str(dest), "--clip", "100")
        text = dest.read_text()
        assert "clipped 9900 chars" in text and "B" * 101 not in text

    def test_negative_clip_errors(self, bigcli):
        code, _, err = bigcli("get", "claude:big", "--clip", "-5")
        assert code == 1 and "--clip must be >= 0" in err


class TestRoleWindowComposition:
    @pytest.fixture
    def mixedcli(self, tmp_path, monkeypatch, capsys):
        f = tmp_path / "mixed.jsonl"
        lines = []
        for i in range(3):
            lines.append(
                json.dumps(
                    {
                        "type": "user",
                        "message": {"content": f"user {i}"},
                        "timestamp": "2026-06-01T10:00:00Z",
                    }
                )
            )
            lines.append(
                json.dumps(
                    {
                        "type": "assistant",
                        "message": {"content": f"reply {i}"},
                        "timestamp": "2026-06-01T10:00:01Z",
                    }
                )
            )
        f.write_text("\n".join(lines) + "\n")
        ss = [
            Session(
                id="mix",
                provider="claude",
                summary="mix",
                cwd="/x",
                updated_at="2026-06-01T10:00:00+00:00",
                content_path=str(f),
            )
        ]

        def run(*argv: str):
            monkeypatch.setattr("session_browser.cli.discover_all", lambda *a, **k: ss)
            code = run_cli(list(argv))
            cap = capsys.readouterr()
            return code, cap.out, cap.err

        return run

    def test_tail_bounds_kept_roles(self, mixedcli):
        # The last raw entry is "reply 2"; --role user --tail 1 must return
        # the last *user* turn, not an empty window.
        _, out, _ = mixedcli(
            "get", "claude:mix", "--role", "user", "--tail", "1", "--format", "json"
        )
        data = json.loads(out)
        assert [e["text"] for e in data["entries"]] == ["user 2"]
        assert data["entries"][0]["index"] == 4
        assert "entry_range" not in data

    def test_head_bounds_kept_roles(self, mixedcli):
        _, out, _ = mixedcli(
            "get",
            "claude:mix",
            "--role",
            "assistant",
            "--head",
            "2",
            "--format",
            "json",
        )
        data = json.loads(out)
        assert [e["text"] for e in data["entries"]] == ["reply 0", "reply 1"]
        assert [e["index"] for e in data["entries"]] == [1, 3]

    def test_entries_stays_absolute_with_role(self, mixedcli):
        _, out, _ = mixedcli(
            "get",
            "claude:mix",
            "--role",
            "user",
            "--entries",
            "2:5",
            "--format",
            "json",
        )
        data = json.loads(out)
        assert [e["index"] for e in data["entries"]] == [2, 4]
        assert data["entry_range"] == {"start": 2, "end": 5}


class TestMainDispatch:
    def test_bare_invocation_launches_tui(self, monkeypatch):
        from session_browser import app as app_mod

        launched = []
        monkeypatch.setattr(
            app_mod.SessionBrowser, "run", lambda self, *a, **k: launched.append(True)
        )
        monkeypatch.setattr("sys.argv", ["session-browser"])
        app_mod.main()
        assert launched == [True]

    def test_args_dispatch_to_cli_with_exit_code(self, monkeypatch):
        from session_browser import app as app_mod
        from session_browser import cli as cli_mod

        monkeypatch.setattr(cli_mod, "run_cli", lambda argv: 7)
        monkeypatch.setattr("sys.argv", ["session-browser", "list"])
        with pytest.raises(SystemExit) as exc:
            app_mod.main()
        assert exc.value.code == 7


class TestGet:
    def test_canonical_id_text_output_is_complete(self, cli):
        code, out, _err = cli("get", "claude:aaa")
        assert code == 0
        assert out.startswith("# Session claude:aaa")
        assert "User: alpha wombat message" in out
        assert "User: second wombat" in out

    def test_unique_raw_id_resolves(self, cli):
        code, out, _ = cli("get", "bbb")
        assert code == 0 and "# Session codex:bbb" in out

    def test_provider_qualified_id_restricts_discovery(
        self, sessions, monkeypatch, capsys
    ):
        hints = []

        def fake(providers=None):
            hints.append(providers)
            return [s for s in sessions if s.provider == "claude"]

        monkeypatch.setattr("session_browser.cli.discover_all", fake)
        assert run_cli(["get", "claude:aaa"]) == 0
        capsys.readouterr()
        assert hints == [["claude"]]

    def test_raw_id_scans_all_providers(self, sessions, monkeypatch, capsys):
        hints = []

        def fake(providers=None):
            hints.append(providers)
            return sessions

        monkeypatch.setattr("session_browser.cli.discover_all", fake)
        assert run_cli(["get", "bbb"]) == 0
        capsys.readouterr()
        assert hints == [None]

    def test_unknown_id_exits_nonzero(self, cli):
        code, _, err = cli("get", "zzz")
        assert code == 1 and "unknown session id" in err

    def test_unique_raw_prefix_resolves(self, cli):
        code, out, _ = cli("get", "bb")
        assert code == 0 and "# Session codex:bbb" in out

    def test_unique_provider_qualified_prefix_resolves(self, cli):
        code, out, _ = cli("get", "claude:aa")
        assert code == 0 and "# Session claude:aaa" in out

    def test_ambiguous_prefix_lists_candidates(self, sessions, monkeypatch, capsys):
        monkeypatch.setattr(
            "session_browser.cli.discover_all", lambda *a, **k: sessions
        )
        code = run_cli(["get", "claude:"])
        err = capsys.readouterr().err
        assert code == 1 and "unknown session id" in err
        # "aaa" and "ccc" share no prefix, but two same-provider ids that do:
        dup = sessions + [
            Session(id="aab", provider="claude", content_path=sessions[0].content_path)
        ]
        monkeypatch.setattr("session_browser.cli.discover_all", lambda *a, **k: dup)
        code = run_cli(["get", "claude:aa"])
        err = capsys.readouterr().err
        assert code == 1
        assert "ambiguous session id" in err
        assert "claude:aaa" in err and "claude:aab" in err

    def test_exact_id_wins_over_prefix_of_another(
        self, sessions, tmp_path, monkeypatch, capsys
    ):
        longer = tmp_path / "long.jsonl"
        write_claude(longer, ["longer twin"])
        twins = sessions + [
            Session(id="aaa-longer", provider="claude", content_path=str(longer))
        ]
        monkeypatch.setattr("session_browser.cli.discover_all", lambda *a, **k: twins)
        code = run_cli(["get", "claude:aaa"])
        out = capsys.readouterr().out
        assert code == 0 and "# Session claude:aaa\n" in out

    def test_ambiguous_raw_id_lists_all_candidates(
        self, sessions, tmp_path, monkeypatch, capsys
    ):
        dup = tmp_path / "dup.jsonl"
        write_claude(dup, ["dup content"])
        ambiguous = sessions + [
            Session(id="aaa", provider="codex", content_path=str(dup))
        ]
        monkeypatch.setattr(
            "session_browser.cli.discover_all", lambda *a, **k: ambiguous
        )
        code = run_cli(["get", "aaa"])
        err = capsys.readouterr().err
        assert code == 1
        assert "claude:aaa" in err and "codex:aaa" in err

    def test_canonical_id_still_works_when_raw_is_ambiguous(
        self, sessions, tmp_path, monkeypatch, capsys
    ):
        dup = tmp_path / "dup.jsonl"
        write_claude(dup, ["dup content"])
        ambiguous = sessions + [
            Session(id="aaa", provider="codex", content_path=str(dup))
        ]
        monkeypatch.setattr(
            "session_browser.cli.discover_all", lambda *a, **k: ambiguous
        )
        code = run_cli(["get", "codex:aaa"])
        out = capsys.readouterr().out
        assert code == 0 and "# Session codex:aaa" in out

    def test_json_format_structured_entries(self, cli):
        _code, out, _ = cli("get", "claude:aaa", "--format", "json")
        data = json.loads(out)
        assert data["session"]["id"] == "claude:aaa"
        assert data["entries"][0] == {
            "role": "user",
            "text": "alpha wombat message",
            "timestamp": "2026-06-01T10:00:00Z",
            "metadata": None,
        }
        assert data["warnings"] == []

    def test_output_file_with_overwrite_protection(self, cli, tmp_path):
        target = tmp_path / "exports" / "session.md"
        code, out, _ = cli("get", "claude:aaa", "--output", str(target))
        assert code == 0
        assert str(target) in out  # confirmation contains the path
        assert "alpha wombat" not in out  # ...not the content
        assert "User: alpha wombat message" in target.read_text()
        code2, _, err2 = cli("get", "claude:aaa", "--output", str(target))
        assert code2 == 1 and "refusing to overwrite" in err2
        code3, _, _ = cli("get", "claude:aaa", "--output", str(target), "--overwrite")
        assert code3 == 0

    def test_json_output_file_writes_structured_data(self, cli, tmp_path):
        target = tmp_path / "session.json"
        code, out, _ = cli(
            "get", "claude:aaa", "--format", "json", "--output", str(target)
        )
        assert code == 0
        confirmation = json.loads(out)
        assert confirmation["written"] == str(target)
        data = json.loads(target.read_text())
        assert data["session"]["id"] == "claude:aaa"

    def test_unreadable_session_is_an_error(self, sessions, monkeypatch, capsys):
        broken = sessions + [
            Session(id="ddd", provider="claude", content_path="/nope/missing")
        ]
        monkeypatch.setattr("session_browser.cli.discover_all", lambda *a, **k: broken)
        code = run_cli(["get", "ddd"])
        err = capsys.readouterr().err
        assert code == 1 and "could not read session" in err

    def test_partial_parse_succeeds_with_warnings(
        self, sessions, tmp_path, monkeypatch, capsys
    ):
        f = tmp_path / "part.jsonl"
        f.write_text(
            json.dumps({"type": "user", "message": {"content": "good line"}})
            + "\n{broken\n"
        )
        partial = sessions + [Session(id="eee", provider="claude", content_path=str(f))]
        monkeypatch.setattr("session_browser.cli.discover_all", lambda *a, **k: partial)
        code = run_cli(["get", "eee", "--format", "json"])
        out = capsys.readouterr().out
        data = json.loads(out)
        assert code == 0  # partial parse is not an error
        assert data["entries"][0]["text"] == "good line"
        assert len(data["warnings"]) == 1

    def test_partial_parse_text_output_writes_file_and_warns_stderr(
        self, sessions, tmp_path, monkeypatch, capsys
    ):
        """Text mode --output with partial parse: file is written and warning
        is emitted to stderr (same as stdout path)."""
        f = tmp_path / "part.jsonl"
        f.write_text(
            json.dumps({"type": "user", "message": {"content": "good line"}})
            + "\n{broken\n"
        )
        partial = sessions + [Session(id="eee", provider="claude", content_path=str(f))]
        monkeypatch.setattr("session_browser.cli.discover_all", lambda *a, **k: partial)
        target = tmp_path / "partial.txt"
        code = run_cli(["get", "eee", "--output", str(target)])
        out, err = capsys.readouterr()
        assert code == 0
        assert "good line" not in out  # confirmation message, not content
        assert "wrote" in out
        assert "User: good line" in target.read_text()
        assert "parse warning" in err.lower()

    def test_output_parent_is_file_errors_cleanly(self, tmp_path, monkeypatch, capsys):
        """When the parent of --output is a regular file, get a structured CLI
        error, not a raw traceback."""
        parent = tmp_path / "not-a-dir"
        parent.write_text("i am a file, not a directory")
        target = parent / "out.txt"
        f = tmp_path / "dummy.jsonl"
        f.write_text(
            json.dumps({"type": "user", "message": {"content": "hello"}}) + "\n"
        )
        session = Session(id="fff", provider="claude", content_path=str(f))
        monkeypatch.setattr(
            "session_browser.cli.discover_all", lambda *a, **k: [session]
        )
        code = run_cli(["get", "claude:fff", "--output", str(target)])
        _out, err = capsys.readouterr()
        assert code == 1
        assert "not a directory" in err or "could not write output" in err

    def test_atomic_write_helper(self, tmp_path):
        """_write_text_atomic writes final content, handles overwrite, and
        does not leave a temp file behind."""
        from session_browser.cli import _write_text_atomic

        target = tmp_path / "out.txt"
        _write_text_atomic(target, "hello")
        assert target.read_text() == "hello"
        # Overwrite
        _write_text_atomic(target, "world")
        assert target.read_text() == "world"
        # No temp files remain in the directory
        leftovers = [
            p for p in tmp_path.iterdir() if p.name.startswith(".") and p != tmp_path
        ]
        assert len(leftovers) == 0


class TestSearch:
    def test_ids_mode_counts_occurrences_in_recency_order(self, cli):
        code, out, _ = cli("search", "wombat", "--mode", "ids")
        assert code == 0
        data = json.loads(out)
        assert data["query"] == "wombat" and data["mode"] == "ids"
        assert [r["id"] for r in data["results"]] == ["claude:ccc", "claude:aaa"]
        assert {r["id"]: r["match_count"] for r in data["results"]} == {
            "claude:aaa": 2,
            "claude:ccc": 1,
        }
        assert "snippets" not in data["results"][0]
        assert "entries" not in data["results"][0]

    def test_snippets_default_mode_respects_context(self, cli):
        _, out, _ = cli("search", "wombat", "--context", "6")
        data = json.loads(out)
        assert data["mode"] == "snippets"
        snips = {r["id"]: r["snippets"] for r in data["results"]}
        # "wombat" sits at offset 6 of "gamma wombat note": context 6 reaches
        # offset 0 on the left and the end of the text on the right, so the
        # snippet has no ellipsis on either side.
        assert snips["claude:ccc"] == [
            {"role": "user", "entry_index": 0, "text": "gamma wombat note"}
        ]
        assert len(snips["claude:aaa"]) == 2  # one snippet per occurrence
        # "alpha wombat message" truncates on the right: context 6 past the
        # match ends mid-word, so the snippet carries a trailing ellipsis.
        assert snips["claude:aaa"][0]["text"] == "alpha wombat messa…"

    def test_full_mode_embeds_complete_entries(self, cli):
        _, out, _ = cli("search", "quokka", "--mode", "full")
        data = json.loads(out)
        assert len(data["results"]) == 1
        entries = data["results"][0]["entries"]
        assert entries[0]["role"] == "user"
        assert entries[0]["text"] == "beta quokka message"

    def test_limit_bounds_results_not_candidates(self, cli):
        # 'alpha' only appears in the *oldest* session. A pre-search candidate
        # limit of 1 would scan only the newest session and find nothing.
        code, out, _ = cli("search", "alpha", "--limit", "1")
        assert code == 0
        assert [r["id"] for r in json.loads(out)["results"]] == ["claude:aaa"]

    def test_metadata_filters_compose_with_search(self, cli):
        _, out, _ = cli("search", "wombat", "--provider", "claude", "--cwd", "proja")
        assert [r["id"] for r in json.loads(out)["results"]] == ["claude:aaa"]

    def test_text_format_ids(self, cli):
        code, out, _ = cli("search", "quokka", "--mode", "ids", "--format", "text")
        assert code == 0
        assert out.splitlines()[0].startswith("codex:bbb\t1\t")

    def test_empty_query_is_an_error(self, cli):
        code, _, err = cli("search", "   ")
        assert code == 1
        assert json.loads(err)["error"]["code"] == "invalid_query"

    def test_unreadable_sessions_reported_as_skipped(
        self, sessions, monkeypatch, capsys
    ):
        broken = sessions + [
            Session(id="bad", provider="claude", content_path="/nope/missing")
        ]
        monkeypatch.setattr("session_browser.cli.discover_all", lambda *a, **k: broken)
        code = run_cli(["search", "wombat"])
        data = json.loads(capsys.readouterr().out)
        assert code == 0  # other sessions still searched
        assert [s["id"] for s in data["skipped"]] == ["claude:bad"]
        assert [r["id"] for r in data["results"]] == ["claude:ccc", "claude:aaa"]

    def test_no_matches_is_empty_results_exit_zero(self, cli):
        code, out, _ = cli("search", "notthere")
        assert code == 0 and json.loads(out)["results"] == []

    # ── text-format branches ─────────────────────────────────────────────

    def test_text_format_snippets(self, cli):
        """Text mode with --mode snippets includes an indented snippet line."""
        code, out, _ = cli(
            "search",
            "wombat",
            "--format",
            "text",
            "--mode",
            "snippets",
            "--context",
            "6",
        )
        assert code == 0
        lines = out.splitlines()
        # Header line for claude:ccc (the newest match): canonical_id, count, date, summary
        assert any(l.startswith("claude:ccc\t1\t") for l in lines)
        # At least one indented snippet line for the match
        snippet_lines = [l for l in lines if l.startswith("  [user]")]
        assert len(snippet_lines) > 0
        assert "wombat" in snippet_lines[0]

    def test_text_format_full(self, cli):
        """Text mode with --mode full includes a Markdown session header
        and full entry text on stdout."""
        code, out, _ = cli("search", "quokka", "--format", "text", "--mode", "full")
        assert code == 0
        assert "# Session codex:bbb" in out
        assert "beta quokka message" in out

    def test_text_format_skipped_on_stderr(self, sessions, monkeypatch, capsys):
        """Text format with an unreadable session emits the skipped session
        id and error to stderr while matched sessions still appear on stdout."""
        broken = sessions + [
            Session(id="bad", provider="claude", content_path="/nope/missing")
        ]
        monkeypatch.setattr("session_browser.cli.discover_all", lambda *a, **k: broken)
        code = run_cli(["search", "wombat", "--format", "text"])
        out, err = capsys.readouterr()
        assert code == 0
        # Matched sessions still on stdout
        assert "claude:ccc" in out and "claude:aaa" in out
        # Skipped session id on stderr
        assert "claude:bad" in err

    # ── context boundary behavior ────────────────────────────────────────

    def test_negative_context_error(self, cli):
        """Negative --context exits 1 with invalid_filter error code."""
        code, _, err = cli("search", "wombat", "--context", "-1")
        assert code == 1
        assert json.loads(err)["error"]["code"] == "invalid_filter"

    def test_context_zero_is_valid(self, cli):
        """Context 0 produces snippets with only the matched text
        (ellipsized as appropriate)."""
        code, out, _ = cli("search", "wombat", "--context", "0")
        assert code == 0
        data = json.loads(out)
        # gamma wombat note: "wombat" at offset 6, match_len=6, context=0
        # → make_snippet returns "…wombat…" (ellipsis on both sides)
        ccc_snippets = [
            s for r in data["results"] if r["id"] == "claude:ccc" for s in r["snippets"]
        ]
        assert len(ccc_snippets) == 1
        assert ccc_snippets[0]["text"] == "…wombat…"


class TestSummaryMatch:
    """search scans summaries too: a session whose *title* names the topic
    must surface even when the transcript never repeats the phrase."""

    @pytest.fixture
    def summary_cli(self, monkeypatch, capsys, tmp_path):
        f = tmp_path / "sql.jsonl"
        write_claude(f, ["please adapt the scripts", "done, all read only"])
        g = tmp_path / "other.jsonl"
        write_claude(g, ["unrelated content"])
        sessions = [
            Session(
                id="sql",
                provider="claude",
                summary="Adapt SQL Scripts For Admin App",
                cwd="/home/u/app_v4",
                updated_at="2026-05-14T10:00:00+00:00",
                content_path=str(f),
            ),
            Session(
                id="oth",
                provider="claude",
                summary="Other Work",
                updated_at="2026-07-01T10:00:00+00:00",
                content_path=str(g),
            ),
        ]

        def run(*argv: str):
            monkeypatch.setattr(
                "session_browser.cli.discover_all", lambda *a, **k: sessions
            )
            code = run_cli(list(argv))
            captured = capsys.readouterr()
            return code, captured.out, captured.err

        return run

    def test_summary_only_match_surfaces_with_phrases(self, summary_cli):
        code, out, _ = summary_cli("search", "sql scripts", "--mode", "ids")
        assert code == 0
        results = json.loads(out)["results"]
        assert [r["id"] for r in results] == ["claude:sql"]
        assert results[0]["match_count"] == 0
        assert results[0]["summary_matches"] == ["sql scripts"]
        # Prefilter-skipped sessions were never parsed; the entry count
        # must be backfilled so the result doesn't look corrupt.
        assert results[0]["total_entries"] == 2

    def test_summary_match_is_markdown_and_case_insensitive(self, summary_cli):
        code, out, _ = summary_cli("search", "`admin` app", "--mode", "ids")
        assert code == 0
        assert [r["id"] for r in json.loads(out)["results"]] == ["claude:sql"]

    def test_content_match_reports_summary_hits_too(self, summary_cli):
        _, out, _ = summary_cli("search", "read only", "admin app", "--mode", "ids")
        results = json.loads(out)["results"]
        assert [r["id"] for r in results] == ["claude:sql"]
        assert results[0]["match_count"] == 1  # content: "read only"
        assert results[0]["summary_matches"] == ["admin app"]

    def test_match_all_satisfied_across_summary_and_content(self, summary_cli):
        _code, out, _ = summary_cli(
            "search", "read only", "admin app", "--match-all", "--mode", "ids"
        )
        assert [r["id"] for r in json.loads(out)["results"]] == ["claude:sql"]
        _code, out, _ = summary_cli(
            "search", "read only", "not anywhere", "--match-all", "--mode", "ids"
        )
        assert json.loads(out)["results"] == []

    def test_text_mode_tags_summary_matches(self, summary_cli):
        _, out, _ = summary_cli(
            "search", "sql scripts", "--mode", "ids", "--format", "text"
        )
        assert "[summary match]" in out.splitlines()[0]


class TestSearchSortAndTruncation:
    def test_sort_oldest_reverses_recency(self, cli):
        _, out, _ = cli("search", "wombat", "--mode", "ids", "--sort", "oldest")
        assert [r["id"] for r in json.loads(out)["results"]] == [
            "claude:aaa",
            "claude:ccc",
        ]

    def test_limit_truncation_is_loud(self, cli):
        _, out, _ = cli("search", "wombat", "--mode", "ids", "--limit", "1")
        data = json.loads(out)
        assert [r["id"] for r in data["results"]] == ["claude:ccc"]
        w = [w for w in data["warnings"] if "--limit 1 dropped" in w]
        assert len(w) == 1
        assert "2026-06-01" in w[0]  # dropped session's date
        assert "oldest matches were dropped first" in w[0]

    def test_no_truncation_no_warning(self, cli):
        _, out, _ = cli("search", "wombat", "--mode", "ids", "--limit", "5")
        assert not any("dropped" in w for w in json.loads(out).get("warnings", []))

    def test_truncation_warning_reaches_stderr_in_text_mode(self, cli):
        _, _out, err = cli(
            "search", "wombat", "--mode", "ids", "--limit", "1", "--format", "text"
        )
        assert "--limit 1 dropped 1 matched session(s)" in err


class TestSnippetCap:
    @pytest.fixture
    def dense_cli(self, monkeypatch, capsys, tmp_path):
        f = tmp_path / "dense.jsonl"
        write_claude(f, ["tok " * 25])  # 25 occurrences in one entry
        sessions = [
            Session(
                id="dense",
                provider="claude",
                updated_at="2026-06-01T10:00:00+00:00",
                content_path=str(f),
            )
        ]

        def run(*argv: str):
            monkeypatch.setattr(
                "session_browser.cli.discover_all", lambda *a, **k: sessions
            )
            code = run_cli(list(argv))
            captured = capsys.readouterr()
            return code, captured.out, captured.err

        return run

    def test_default_cap_and_omitted_count(self, dense_cli):
        _, out, _ = dense_cli("search", "tok", "--context", "0")
        r = json.loads(out)["results"][0]
        assert r["match_count"] == 25
        assert len(r["snippets"]) == 20
        assert r["snippets_omitted"] == 5

    def test_explicit_cap(self, dense_cli):
        _, out, _ = dense_cli("search", "tok", "--max-snippets", "2")
        r = json.loads(out)["results"][0]
        assert len(r["snippets"]) == 2 and r["snippets_omitted"] == 23

    def test_zero_disables_cap(self, dense_cli):
        _, out, _ = dense_cli("search", "tok", "--max-snippets", "0")
        r = json.loads(out)["results"][0]
        assert len(r["snippets"]) == 25
        assert "snippets_omitted" not in r

    def test_negative_cap_is_error(self, dense_cli):
        code, _, err = dense_cli("search", "tok", "--max-snippets", "-1")
        assert code == 1
        assert json.loads(err)["error"]["code"] == "invalid_filter"

    def test_text_mode_reports_omission(self, dense_cli):
        _, out, _ = dense_cli(
            "search", "tok", "--format", "text", "--max-snippets", "2", "--context", "0"
        )
        assert "23 more snippet(s) omitted" in out


class TestMarkdownInsensitiveCli:
    def test_plain_query_finds_backticked_text(self, monkeypatch, capsys, tmp_path):
        f = tmp_path / "md.jsonl"
        write_claude(f, ["every SQL statement is `SELECT` only."])
        sessions = [
            Session(
                id="md",
                provider="claude",
                updated_at="2026-06-01T10:00:00+00:00",
                content_path=str(f),
            )
        ]
        monkeypatch.setattr(
            "session_browser.cli.discover_all", lambda *a, **k: sessions
        )
        assert run_cli(["search", "SELECT only", "--mode", "snippets"]) == 0
        data = json.loads(capsys.readouterr().out)
        assert [r["id"] for r in data["results"]] == ["claude:md"]
        assert "`SELECT` only" in data["results"][0]["snippets"][0]["text"]


class TestGetEntryWindow:
    """`get --entries/--head/--tail` return a slice of the transcript so a
    matched region (or the ending) can be read without exporting the file."""

    def test_tail_returns_last_entries_with_slice_header(self, cli):
        code, out, _ = cli("get", "claude:aaa", "--tail", "1")
        assert code == 0
        assert "- Entries: 1–1 of 2" in out
        assert "User: second wombat" in out
        assert "alpha wombat message" not in out

    def test_head_clamps_to_total(self, cli):
        code, out, _ = cli("get", "claude:aaa", "--head", "99")
        assert code == 0
        assert "- Entries: 0–1 of 2" in out
        assert "alpha wombat message" in out and "second wombat" in out

    def test_entries_range_json_reports_slice(self, cli):
        code, out, _ = cli("get", "claude:aaa", "--format", "json", "--entries", "1:1")
        data = json.loads(out)
        assert code == 0
        assert data["total_entries"] == 2
        assert data["entry_range"] == {"start": 1, "end": 1}
        assert [e["text"] for e in data["entries"]] == ["second wombat"]

    def test_entries_open_ended_forms(self, cli):
        _, out, _ = cli("get", "claude:aaa", "--format", "json", "--entries", "1:")
        assert json.loads(out)["entry_range"] == {"start": 1, "end": 1}
        _, out, _ = cli("get", "claude:aaa", "--format", "json", "--entries", ":0")
        assert json.loads(out)["entry_range"] == {"start": 0, "end": 0}
        _, out, _ = cli("get", "claude:aaa", "--format", "json", "--entries", "0")
        assert json.loads(out)["entry_range"] == {"start": 0, "end": 0}

    def test_entries_end_clamped_to_last(self, cli):
        _, out, _ = cli("get", "claude:aaa", "--format", "json", "--entries", "0:99")
        assert json.loads(out)["entry_range"] == {"start": 0, "end": 1}

    def test_unwindowed_json_still_reports_total(self, cli):
        _, out, _ = cli("get", "claude:aaa", "--format", "json")
        data = json.loads(out)
        assert data["total_entries"] == 2 and "entry_range" not in data

    def test_start_beyond_end_is_structured_error(self, cli):
        code, _, err = cli("get", "claude:aaa", "--format", "json", "--entries", "9")
        assert code == 1
        error = json.loads(err)["error"]
        assert error["code"] == "invalid_range"
        assert error["details"]["total_entries"] == 2

    def test_malformed_and_inverted_ranges_error(self, cli):
        for bad in ("junk", "1:0", ":"):
            code, _, err = cli(
                "get", "claude:aaa", "--format", "json", "--entries", bad
            )
            assert code == 1
            assert json.loads(err)["error"]["code"] == "invalid_range"

    def test_nonpositive_head_tail_error(self, cli):
        for flag in ("--head", "--tail"):
            code, _, err = cli("get", "claude:aaa", "--format", "json", flag, "0")
            assert code == 1
            assert json.loads(err)["error"]["code"] == "invalid_range"

    def test_windowed_output_file(self, cli, tmp_path):
        target = tmp_path / "tail.md"
        code, _, _ = cli("get", "claude:aaa", "--tail", "1", "--output", str(target))
        assert code == 0
        body = target.read_text()
        assert "- Entries: 1–1 of 2" in body
        assert "alpha wombat message" not in body


class TestGetRoleFilter:
    """`get --role` keeps only entries of the named roles (plus the "error"
    pseudo-role for failed tool calls); kept entries carry their absolute
    indices so results compose with --entries and search's entry_index."""

    @pytest.fixture
    def mixed_cli(self, monkeypatch, capsys, tmp_path):
        f = tmp_path / "mmm.jsonl"
        lines = [
            {
                "type": "user",
                "message": {"content": "please fix the login bug"},
                "timestamp": "2026-06-01T10:00:00Z",
            },
            {
                "type": "assistant",
                "message": {
                    "content": [
                        {"type": "text", "text": "looking at it now"},
                        {
                            "type": "tool_use",
                            "name": "Bash",
                            "input": {"command": "pytest -q"},
                        },
                    ]
                },
            },
            {
                "type": "user",
                "message": {
                    "content": [
                        {
                            "type": "tool_result",
                            "content": "1 failed",
                            "is_error": True,
                        },
                    ]
                },
            },
            {"type": "assistant", "message": {"content": "fixed it"}},
        ]
        f.write_text("\n".join(json.dumps(x) for x in lines) + "\n")
        sessions = [
            Session(
                id="mmm",
                provider="claude",
                summary="mixed",
                updated_at="2026-06-01T10:00:00+00:00",
                content_path=str(f),
            )
        ]
        # entries: 0 user, 1 assistant, 2 tool call, 3 tool error, 4 assistant

        def run(*argv: str):
            monkeypatch.setattr(
                "session_browser.cli.discover_all", lambda *a, **k: sessions
            )
            code = run_cli(list(argv))
            captured = capsys.readouterr()
            return code, captured.out, captured.err

        return run

    def test_user_role_text_shows_indices_and_header(self, mixed_cli):
        code, out, _ = mixed_cli("get", "claude:mmm", "--role", "user")
        assert code == 0
        assert "- Entries: 1 of 5 (roles: user)" in out
        assert "[0] User: please fix the login bug" in out
        assert "looking at it now" not in out and "1 failed" not in out

    def test_error_pseudo_role_finds_failed_tool_calls(self, mixed_cli):
        code, out, _ = mixed_cli(
            "get", "claude:mmm", "--role", "error", "--format", "json"
        )
        assert code == 0
        data = json.loads(out)
        assert data["roles"] == ["error"]
        assert data["total_entries"] == 5
        assert [(e["index"], e["text"]) for e in data["entries"]] == [(3, "1 failed")]

    def test_error_label_marked_in_text(self, mixed_cli):
        _, out, _ = mixed_cli("get", "claude:mmm", "--role", "error")
        assert "[3] Tool output (error): 1 failed" in out

    def test_comma_and_repeat_forms_canonical_order(self, mixed_cli):
        code, out, _ = mixed_cli(
            "get",
            "claude:mmm",
            "--role",
            "error,user",
            "--role",
            "assistant",
            "--format",
            "json",
        )
        assert code == 0
        data = json.loads(out)
        assert data["roles"] == ["user", "assistant", "error"]
        assert [e["index"] for e in data["entries"]] == [0, 1, 3, 4]

    def test_composes_with_entry_window_absolute_indices(self, mixed_cli):
        code, out, _ = mixed_cli(
            "get",
            "claude:mmm",
            "--entries",
            "2:",
            "--role",
            "assistant",
            "--format",
            "json",
        )
        assert code == 0
        data = json.loads(out)
        assert data["entry_range"] == {"start": 2, "end": 4}
        assert [(e["index"], e["text"]) for e in data["entries"]] == [(4, "fixed it")]

    def test_no_matches_reported_not_empty_session(self, mixed_cli):
        code, out, _ = mixed_cli("get", "claude:mmm", "--role", "system")
        assert code == 0
        assert "- Entries: 0 of 5 (roles: system)" in out
        assert "(no entries with roles: system)" in out
        assert "(empty session)" not in out

    def test_invalid_role_structured_error(self, mixed_cli):
        code, out, err = mixed_cli(
            "get", "claude:mmm", "--role", "robot", "--format", "json"
        )
        assert code == 1 and out == ""
        assert json.loads(err)["error"]["code"] == "invalid_role"

    def test_unfiltered_entries_carry_no_index(self, mixed_cli):
        _, out, _ = mixed_cli("get", "claude:mmm", "--format", "json")
        data = json.loads(out)
        assert "roles" not in data
        assert all("index" not in e for e in data["entries"])


class TestMultiPhraseSearch:
    """Several literal phrases are OR'd in one scan; --match-all narrows to
    sessions containing every phrase."""

    def test_or_semantics_single_scan(self, cli):
        code, out, _ = cli("search", "alpha", "quokka", "--mode", "ids")
        assert code == 0
        data = json.loads(out)
        assert data["query"] == ["alpha", "quokka"]
        assert {r["id"] for r in data["results"]} == {"claude:aaa", "codex:bbb"}

    def test_single_query_keeps_string_shape(self, cli):
        _, out, _ = cli("search", "wombat", "--mode", "ids")
        assert json.loads(out)["query"] == "wombat"

    def test_match_all_requires_every_phrase(self, cli):
        # "wombat" hits aaa and ccc; only aaa also contains "alpha".
        _, out, _ = cli("search", "alpha", "wombat", "--mode", "ids", "--match-all")
        assert [r["id"] for r in json.loads(out)["results"]] == ["claude:aaa"]

    def test_multi_query_snippets_name_their_phrase(self, cli):
        _, out, _ = cli("search", "alpha", "quokka", "--context", "6")
        data = json.loads(out)
        snips = {s["query"] for r in data["results"] for s in r["snippets"]}
        assert snips == {"alpha", "quokka"}

    def test_single_query_snippets_have_no_query_key(self, cli):
        _, out, _ = cli("search", "wombat", "--context", "6")
        data = json.loads(out)
        assert all("query" not in s for r in data["results"] for s in r["snippets"])

    def test_any_empty_phrase_is_an_error(self, cli):
        code, _, err = cli("search", "wombat", "   ")
        assert code == 1
        assert json.loads(err)["error"]["code"] == "invalid_query"


class TestSearchTriageSignals:
    """Search results carry total_entries and first/last match indices, and
    can be ordered by match count."""

    def test_results_report_size_and_match_span(self, cli):
        _, out, _ = cli("search", "wombat", "--mode", "ids")
        by_id = {r["id"]: r for r in json.loads(out)["results"]}
        aaa = by_id["claude:aaa"]
        assert aaa["total_entries"] == 2
        assert aaa["first_match"] == 0 and aaa["last_match"] == 1
        ccc = by_id["claude:ccc"]
        assert ccc["total_entries"] == 1
        assert ccc["first_match"] == 0 and ccc["last_match"] == 0

    def test_sort_matches_orders_by_count(self, cli):
        # Recency order puts ccc (1 hit) first; --sort matches flips it.
        _, out, _ = cli("search", "wombat", "--mode", "ids", "--sort", "matches")
        assert [r["id"] for r in json.loads(out)["results"]] == [
            "claude:aaa",
            "claude:ccc",
        ]

    def test_default_sort_stays_recent(self, cli):
        _, out, _ = cli("search", "wombat", "--mode", "ids")
        assert [r["id"] for r in json.loads(out)["results"]] == [
            "claude:ccc",
            "claude:aaa",
        ]


class TestSearchArtifacts:
    def test_full_mode_writes_manifest_and_transcripts(self, cli, tmp_path):
        out_dir = tmp_path / "results"
        code, out, _ = cli(
            "search", "wombat", "--mode", "full", "--output-dir", str(out_dir)
        )
        assert code == 0
        summary = json.loads(out)
        assert summary["results"] == 2 and summary["files_written"] == 3
        assert "alpha wombat" not in out  # summary only, no content replay
        manifest = json.loads((out_dir / "manifest.json").read_text())
        assert manifest["query"] == "wombat"
        assert manifest["filters"]["provider"] is None
        assert {r["id"]: r["file"] for r in manifest["results"]} == {
            "claude:aaa": "claude-aaa.md",
            "claude:ccc": "claude-ccc.md",
        }
        assert {r["id"]: r["match_count"] for r in manifest["results"]} == {
            "claude:aaa": 2,
            "claude:ccc": 1,
        }
        body = (out_dir / "claude-aaa.md").read_text()
        assert body.startswith("# Session claude:aaa")
        assert "User: alpha wombat message" in body

    def test_snippets_mode_writes_manifest_only(self, cli, tmp_path):
        out_dir = tmp_path / "results"
        code, _out, _ = cli("search", "wombat", "--output-dir", str(out_dir))
        assert code == 0
        assert sorted(p.name for p in out_dir.iterdir()) == ["manifest.json"]
        manifest = json.loads((out_dir / "manifest.json").read_text())
        assert "snippets" in manifest["results"][0]
        assert "file" not in manifest["results"][0]

    def test_overwrite_protection_checks_before_writing_anything(self, cli, tmp_path):
        out_dir = tmp_path / "results"
        out_dir.mkdir()
        (out_dir / "claude-aaa.md").write_text("precious")
        code, _, err = cli(
            "search", "wombat", "--mode", "full", "--output-dir", str(out_dir)
        )
        assert code == 1 and "refusing to overwrite" in err
        assert (out_dir / "claude-aaa.md").read_text() == "precious"
        assert not (out_dir / "manifest.json").exists()  # nothing was written
        code2, _, _ = cli(
            "search",
            "wombat",
            "--mode",
            "full",
            "--output-dir",
            str(out_dir),
            "--overwrite",
        )
        assert code2 == 0
        assert (out_dir / "manifest.json").exists()

    def test_filenames_are_sanitized(self, sessions, tmp_path, monkeypatch, capsys):
        weird = tmp_path / "weird.jsonl"
        write_claude(weird, ["needle here"])
        sess = sessions + [
            Session(
                id="a/b:c",
                provider="claude",
                updated_at="2026-06-10T00:00:00+00:00",
                content_path=str(weird),
            )
        ]
        monkeypatch.setattr("session_browser.cli.discover_all", lambda *a, **k: sess)
        out_dir = tmp_path / "res"
        code = run_cli(
            ["search", "needle", "--mode", "full", "--output-dir", str(out_dir)]
        )
        capsys.readouterr()
        assert code == 0
        manifest = json.loads((out_dir / "manifest.json").read_text())
        assert manifest["results"][0]["file"] == "claude-a-b-c.md"
        assert (out_dir / "claude-a-b-c.md").exists()

    def test_manifest_records_skipped_sessions(
        self, sessions, tmp_path, monkeypatch, capsys
    ):
        broken = sessions + [
            Session(id="bad", provider="claude", content_path="/nope/missing")
        ]
        monkeypatch.setattr("session_browser.cli.discover_all", lambda *a, **k: broken)
        out_dir = tmp_path / "res"
        code = run_cli(["search", "wombat", "--output-dir", str(out_dir)])
        _out, err = capsys.readouterr()
        assert code == 0
        manifest = json.loads((out_dir / "manifest.json").read_text())
        assert [s["id"] for s in manifest["skipped"]] == ["claude:bad"]
        # Artifact path also reports skipped sessions to stderr
        assert "claude:bad" in err
        assert "skipped" in err

    def test_cleanup_on_write_failure(self, sessions, tmp_path, monkeypatch, capsys):
        """When _write_text_atomic fails mid-operation, already-written
        files from this invocation are cleaned up."""
        import session_browser.cli as cli_mod
        from session_browser.cli import _write_text_atomic as real_write

        broken = sessions + [
            Session(id="bad", provider="claude", content_path="/nope/missing")
        ]
        monkeypatch.setattr("session_browser.cli.discover_all", lambda *a, **k: broken)

        call_count = [0]
        first_path = [None]

        def failing_write(path, content):
            call_count[0] += 1
            if call_count[0] == 1:
                first_path[0] = path
                real_write(path, content)
            else:
                raise cli_mod.CliError("write failed", code="write_error")

        monkeypatch.setattr(cli_mod, "_write_text_atomic", failing_write)

        out_dir = tmp_path / "results"
        code = run_cli(
            ["search", "wombat", "--mode", "full", "--output-dir", str(out_dir)]
        )
        capsys.readouterr()  # discard output
        assert code == 1
        # The file written in the successful first call should be cleaned up
        assert first_path[0] is not None
        assert not first_path[0].exists()


def test_atomic_write_mkstemp_oserror_wrapped(tmp_path, monkeypatch):
    """When tempfile.mkstemp raises OSError, _write_text_atomic wraps it as
    CliError with code='write_error'."""
    import session_browser.cli as cli_mod
    from session_browser.cli import CliError, _write_text_atomic

    def failing_mkstemp(*args, **kwargs):
        raise OSError("no space left on device")

    monkeypatch.setattr(cli_mod.tempfile, "mkstemp", failing_mkstemp)
    target = tmp_path / "out.txt"
    with pytest.raises(CliError) as exc_info:
        _write_text_atomic(target, "hello")
    assert exc_info.value.code == "write_error"
    error_msg = str(exc_info.value)
    assert "no space" in error_msg or str(target) in error_msg


class TestHereFilter:
    """`--here` scopes to os.getcwd() by exact path-prefix and warns about
    sessions excluded for a missing cwd."""

    def test_keeps_only_project_tree(self, capsys, monkeypatch, tmp_path):
        base = "/work/proj"
        f = tmp_path / "x.jsonl"
        write_claude(f, ["needle"])
        sess = [
            Session(
                id="root",
                provider="claude",
                summary="r",
                cwd=base,
                updated_at="2026-06-01T10:00:00+00:00",
                content_path=str(f),
            ),
            Session(
                id="sub",
                provider="claude",
                summary="s",
                cwd=base + "/pkg",
                updated_at="2026-06-02T10:00:00+00:00",
                content_path=str(f),
            ),
            Session(
                id="sibling",
                provider="codex",
                summary="x",
                cwd=base + "x",
                updated_at="2026-06-03T10:00:00+00:00",
                content_path=str(f),
            ),
            Session(
                id="nocwd",
                provider="opencode",
                summary="g",
                cwd="",
                updated_at="2026-06-04T10:00:00+00:00",
                content_path=str(f),
            ),
        ]
        monkeypatch.setattr("session_browser.cli.discover_all", lambda *a, **k: sess)
        monkeypatch.setattr("os.getcwd", lambda: base)
        run_cli(["list", "--here"])
        out = capsys.readouterr().out
        # root and its subdir survive; sibling (path-boundary) and empty cwd drop
        assert [s["id"] for s in json.loads(out)["sessions"]] == [
            "claude:sub",
            "claude:root",
        ]

    def test_warns_about_missing_cwd(self, capsys, monkeypatch, tmp_path):
        base = "/work/proj"
        f = tmp_path / "x.jsonl"
        write_claude(f, ["needle"])
        sess = [
            Session(
                id="root",
                provider="claude",
                summary="r",
                cwd=base,
                updated_at="2026-06-01T10:00:00+00:00",
                content_path=str(f),
            ),
            Session(
                id="nocwd",
                provider="opencode",
                summary="g",
                cwd="",
                updated_at="2026-06-04T10:00:00+00:00",
                content_path=str(f),
            ),
        ]
        monkeypatch.setattr("session_browser.cli.discover_all", lambda *a, **k: sess)
        monkeypatch.setattr("os.getcwd", lambda: base)
        run_cli(["list", "--here"])
        data = json.loads(capsys.readouterr().out)
        assert len(data["warnings"]) == 1
        assert "no recorded cwd" in data["warnings"][0]

    def test_no_warning_when_all_have_cwd(self, capsys, monkeypatch, tmp_path):
        base = "/work/proj"
        f = tmp_path / "x.jsonl"
        write_claude(f, ["needle"])
        sess = [
            Session(
                id="root",
                provider="claude",
                summary="r",
                cwd=base,
                updated_at="2026-06-01T10:00:00+00:00",
                content_path=str(f),
            )
        ]
        monkeypatch.setattr("session_browser.cli.discover_all", lambda *a, **k: sess)
        monkeypatch.setattr("os.getcwd", lambda: base)
        run_cli(["list", "--here"])
        assert "warnings" not in json.loads(capsys.readouterr().out)

    def test_text_warning_on_stderr(self, capsys, monkeypatch, tmp_path):
        base = "/work/proj"
        f = tmp_path / "x.jsonl"
        write_claude(f, ["needle"])
        sess = [
            Session(
                id="root",
                provider="claude",
                summary="r",
                cwd=base,
                updated_at="2026-06-01T10:00:00+00:00",
                content_path=str(f),
            ),
            Session(
                id="nocwd",
                provider="opencode",
                summary="g",
                cwd="",
                updated_at="2026-06-04T10:00:00+00:00",
                content_path=str(f),
            ),
        ]
        monkeypatch.setattr("session_browser.cli.discover_all", lambda *a, **k: sess)
        monkeypatch.setattr("os.getcwd", lambda: base)
        run_cli(["list", "--here", "--format", "text"])
        assert "no recorded cwd" in capsys.readouterr().err

    def test_search_scopes_and_warns(self, capsys, monkeypatch, tmp_path):
        base = "/work/proj"
        f = tmp_path / "x.jsonl"
        write_claude(f, ["needle"])
        sess = [
            Session(
                id="root",
                provider="claude",
                summary="r",
                cwd=base,
                updated_at="2026-06-01T10:00:00+00:00",
                content_path=str(f),
            ),
            Session(
                id="sub",
                provider="claude",
                summary="s",
                cwd=base + "/pkg",
                updated_at="2026-06-02T10:00:00+00:00",
                content_path=str(f),
            ),
            Session(
                id="nocwd",
                provider="opencode",
                summary="g",
                cwd="",
                updated_at="2026-06-04T10:00:00+00:00",
                content_path=str(f),
            ),
        ]
        monkeypatch.setattr("session_browser.cli.discover_all", lambda *a, **k: sess)
        monkeypatch.setattr("os.getcwd", lambda: base)
        run_cli(["search", "needle", "--here", "--mode", "ids"])
        data = json.loads(capsys.readouterr().out)
        assert {r["id"] for r in data["results"]} == {"claude:root", "claude:sub"}
        assert len(data["warnings"]) == 1


class TestCurrentSessionExclusion:
    """The caller's own live session(s) are auto-excluded by matching agent
    session-id env vars against canonical ids. `get` is never affected."""

    def test_excludes_caller_session_by_claude_env(self, cli, monkeypatch):
        monkeypatch.setenv("CLAUDE_CODE_SESSION_ID", "ccc")
        monkeypatch.delenv("CODEX_THREAD_ID", raising=False)
        _code, out, _ = cli("list")
        data = json.loads(out)
        assert [s["id"] for s in data["sessions"]] == ["codex:bbb", "claude:aaa"]
        assert any("claude:ccc" in w for w in data["warnings"])

    def test_excludes_caller_session_by_codex_env(self, cli, monkeypatch):
        monkeypatch.delenv("CLAUDE_CODE_SESSION_ID", raising=False)
        monkeypatch.setenv("CODEX_THREAD_ID", "bbb")
        _, out, _ = cli("list")
        assert [s["id"] for s in json.loads(out)["sessions"]] == [
            "claude:ccc",
            "claude:aaa",
        ]

    def test_union_excludes_whole_spawn_chain(self, cli, monkeypatch):
        # A Claude session that launched Codex carries both vars.
        monkeypatch.setenv("CLAUDE_CODE_SESSION_ID", "ccc")
        monkeypatch.setenv("CODEX_THREAD_ID", "bbb")
        _, out, _ = cli("list")
        assert [s["id"] for s in json.loads(out)["sessions"]] == ["claude:aaa"]

    def test_include_current_keeps_them(self, cli, monkeypatch):
        monkeypatch.setenv("CLAUDE_CODE_SESSION_ID", "ccc")
        monkeypatch.delenv("CODEX_THREAD_ID", raising=False)
        _, out, _ = cli("list", "--include-current")
        data = json.loads(out)
        assert [s["id"] for s in data["sessions"]] == [
            "claude:ccc",
            "codex:bbb",
            "claude:aaa",
        ]
        assert "warnings" not in data  # nothing excluded → no note

    def test_no_known_env_no_exclusion(self, cli, monkeypatch):
        monkeypatch.delenv("CLAUDE_CODE_SESSION_ID", raising=False)
        monkeypatch.delenv("CODEX_THREAD_ID", raising=False)
        _, out, _ = cli("list")
        assert [s["id"] for s in json.loads(out)["sessions"]] == [
            "claude:ccc",
            "codex:bbb",
            "claude:aaa",
        ]

    def test_empty_env_value_is_ignored(self, cli, monkeypatch):
        monkeypatch.setenv("CLAUDE_CODE_SESSION_ID", "   ")
        monkeypatch.delenv("CODEX_THREAD_ID", raising=False)
        _, out, _ = cli("list")
        assert [s["id"] for s in json.loads(out)["sessions"]] == [
            "claude:ccc",
            "codex:bbb",
            "claude:aaa",
        ]

    def test_search_excludes_current(self, cli, monkeypatch):
        # ccc holds "gamma wombat note"; it must drop out of a wombat search.
        monkeypatch.setenv("CLAUDE_CODE_SESSION_ID", "ccc")
        monkeypatch.delenv("CODEX_THREAD_ID", raising=False)
        _, out, _ = cli("search", "wombat", "--mode", "ids")
        data = json.loads(out)
        assert [r["id"] for r in data["results"]] == ["claude:aaa"]
        assert any("claude:ccc" in w for w in data["warnings"])

    def test_get_is_unaffected(self, cli, monkeypatch):
        # Explicit retrieval of your own session must still work.
        monkeypatch.setenv("CLAUDE_CODE_SESSION_ID", "ccc")
        code, out, _ = cli("get", "claude:ccc")
        assert code == 0 and out.startswith("# Session claude:ccc")


class TestRelativeDates:
    """`--since` / `--until` accept relative bounds like -30m / -2h / -1d."""

    def test_relative_delta_parsing(self):
        from datetime import timedelta

        from session_browser.cli import _relative_delta

        assert _relative_delta("-45s") == timedelta(seconds=45)
        assert _relative_delta("-30m") == timedelta(minutes=30)
        assert _relative_delta("-2h") == timedelta(hours=2)
        assert _relative_delta("-1d") == timedelta(days=1)
        assert _relative_delta("-1w") == timedelta(weeks=1)
        assert _relative_delta("2026-06-01") is None
        assert _relative_delta("-30x") is None
        assert _relative_delta("junk") is None

    def test_parse_date_relative_resolves_to_now_minus_delta(self):
        from datetime import datetime, timedelta

        from session_browser.cli import _parse_date

        before = datetime.now(UTC) - timedelta(hours=1)
        got = _parse_date("-1h", "--until")
        after = datetime.now(UTC) - timedelta(hours=1)
        assert before <= got <= after

    def test_relative_until_includes_older_sessions(self, cli, monkeypatch):
        # Fixtures are dated early June 2026; "-1d" is well after them.
        monkeypatch.delenv("CLAUDE_CODE_SESSION_ID", raising=False)
        monkeypatch.delenv("CODEX_THREAD_ID", raising=False)
        _, out, _ = cli("list", "--until", "-1d")
        assert [s["id"] for s in json.loads(out)["sessions"]] == [
            "claude:ccc",
            "codex:bbb",
            "claude:aaa",
        ]

    def test_relative_since_excludes_older_sessions(self, cli, monkeypatch):
        monkeypatch.delenv("CLAUDE_CODE_SESSION_ID", raising=False)
        monkeypatch.delenv("CODEX_THREAD_ID", raising=False)
        _, out, _ = cli("list", "--since", "-1d")
        assert json.loads(out)["sessions"] == []

    def test_relative_equals_form_also_works(self, cli, monkeypatch):
        monkeypatch.delenv("CLAUDE_CODE_SESSION_ID", raising=False)
        monkeypatch.delenv("CODEX_THREAD_ID", raising=False)
        _, out, _ = cli("list", "--since=-1d")
        assert json.loads(out)["sessions"] == []

    def test_garbage_relative_is_a_structured_error(self, cli):
        code, _, err = cli("list", "--since", "-30x")
        assert code == 1
        assert json.loads(err)["error"]["code"] == "invalid_date"


class TestStats:
    @pytest.fixture
    def recent_sessions(self):
        """Sessions with activity relative to *now* so the --days window is
        deterministic regardless of when the tests run."""
        from datetime import datetime, timedelta

        now = datetime.now(UTC)

        def iso(days_ago):
            return (now - timedelta(days=days_ago)).isoformat()

        return [
            Session(id="s0", provider="claude", cwd="/home/u/projA", updated_at=iso(0)),
            Session(id="s1", provider="claude", cwd="/home/u/projA", updated_at=iso(1)),
            Session(id="s2", provider="codex", cwd="/home/u/projB", updated_at=iso(1)),
            Session(
                id="s3", provider="opencode", cwd="", updated_at=iso(40)
            ),  # outside any small --days window
        ]

    @pytest.fixture
    def stats_cli(self, monkeypatch, capsys, recent_sessions):
        def run(*argv: str):
            monkeypatch.setattr(
                "session_browser.cli.discover_all", lambda *a, **k: recent_sessions
            )
            monkeypatch.delenv("CLAUDE_CODE_SESSION_ID", raising=False)
            monkeypatch.delenv("CODEX_THREAD_ID", raising=False)
            code = run_cli(list(argv))
            captured = capsys.readouterr()
            return code, captured.out, captured.err

        return run

    def test_json_shape_and_provider_breakdown(self, stats_cli):
        code, out, err = stats_cli("stats", "--format", "json")
        assert code == 0 and err == ""
        data = json.loads(out)
        assert data["total"] == 4
        assert [(p["provider"], p["count"]) for p in data["providers"]] == [
            ("claude", 2),
            ("codex", 1),
            ("opencode", 1),
        ]
        assert data["providers"][0]["percent"] == 50
        assert data["providers"][0]["updated_at"] is not None
        assert data["oldest"] < data["newest"]

    def test_last_activity_is_named_updated_at_everywhere(self, stats_cli):
        """stats provider rows once called this last_activity while list
        session rows called it updated_at. Reading one name off the other's
        rows returned None rather than raising, so a sweep printed '?' for
        every date and looked like it had worked."""
        _, out, _ = stats_cli("stats", "--format", "json")
        assert "last_activity" not in out
        assert all(p["updated_at"] for p in json.loads(out)["providers"])

    def test_activity_survives_a_piped_head(self, stats_cli):
        """`| head -N` on a JSON tool is near-universal agent behaviour. A
        90-day window used to put activity below 12 lines of mostly-null
        filters and then spend 90 lines on counts, so head -60 returned a
        truncated fragment of the array instead of the answer."""
        _, out, _ = stats_cli("stats", "--days", "90", "--format", "json")
        lines = out.splitlines()
        assert len(json.loads(out)["activity"]["counts"]) == 90
        # All 90 buckets on one line, so the whole block clears a head -20.
        # Asserted as "one line, inside the budget" rather than as an exact
        # index: the index moved when `transcript_health` was added, which is
        # a legitimate key addition, while the property the docstring names —
        # the array does not explode across lines and does not sink below a
        # head -20 — is what actually protects the reader.
        counts_lines = [i for i, ln in enumerate(lines) if '"counts"' in ln]
        assert len(counts_lines) == 1
        assert counts_lines[0] < 20
        assert lines.index('  "filters": {') > lines.index('  "activity": {')

    def test_activity_buckets_by_day(self, stats_cli):
        _, out, _ = stats_cli("stats", "--days", "7", "--format", "json")
        act = json.loads(out)["activity"]
        assert act["days"] == 7 and len(act["counts"]) == 7
        # s3 (40d ago) is outside the window; today has s0, yesterday s1+s2.
        assert sum(act["counts"]) == 3
        assert act["counts"][-1] == 1
        assert act["counts"][-2] == 2

    def test_top_cwds_ranked_and_capped(self, stats_cli):
        _, out, _ = stats_cli("stats", "--format", "json", "--top", "1")
        data = json.loads(out)
        assert data["top_cwds"] == [{"cwd": "/home/u/projA", "count": 2}]

    def test_filters_apply(self, stats_cli):
        _, out, _ = stats_cli("stats", "--provider", "claude", "--format", "json")
        data = json.loads(out)
        assert data["total"] == 2
        assert [p["provider"] for p in data["providers"]] == ["claude"]

    def test_exclude_cwd_updates_every_aggregate(self, stats_cli):
        """stats derives all of its numbers from the filtered set, so an
        exclusion has to move total, providers and top_cwds together — a
        partially-filtered dashboard would be worse than none."""
        _, out, _ = stats_cli("stats", "--exclude-cwd", "projA", "--format", "json")
        data = json.loads(out)
        assert data["total"] == 2  # s0/s1 gone; s2 and the blank-cwd s3 stay
        assert [p["provider"] for p in data["providers"]] == ["codex", "opencode"]
        assert [d["cwd"] for d in data["top_cwds"]] == ["/home/u/projB"]
        assert data["filters"]["exclude_cwd"] == ["projA"]

    def test_text_dashboard(self, stats_cli):
        code, out, err = stats_cli("stats")
        assert code == 0 and err == ""
        assert out.startswith("4 sessions · 3 providers · 2 directories")
        assert "claude" in out and "█" in out
        assert "activity · last 30 days" in out
        assert "top working directories" in out

    def test_text_empty_state(self, stats_cli):
        # A filter that matches nothing but is *valid*. It used to be
        # --provider nonexistent, which no longer parses: an unknown provider
        # is now rejected rather than quietly filtering every session out.
        code, out, _ = stats_cli("stats", "--cwd", "no-such-directory")
        assert code == 0
        assert "no sessions match" in out

    def test_unknown_provider_is_rejected_not_filtered_with(self, capsys):
        """`--provider claude-code` used to report that there were no Claude
        Code sessions. Naming the valid values beats an empty set that reads
        as an answer.

        Goes through argparse rather than the fixture, because argparse exits
        the process on a bad value instead of returning a code — which is the
        behaviour being asserted, and is what the CLI's other enum flags do.
        """
        with pytest.raises(SystemExit) as exc:
            run_cli(["stats", "--provider", "claude-code"])
        assert exc.value.code == 2
        err = capsys.readouterr().err
        assert "invalid choice: 'claude-code'" in err
        assert "claude" in err and "codex" in err and "opencode" in err

    def test_provider_stays_case_insensitive(self, stats_cli):
        code, out, _ = stats_cli("stats", "--provider", "CLAUDE")
        assert code == 0
        assert "no sessions match" not in out

    def test_invalid_days_and_top(self, stats_cli):
        code, _, err = stats_cli("stats", "--days", "0", "--format", "json")
        assert code == 1
        assert json.loads(err)["error"]["code"] == "invalid_filter"
        code, _, err = stats_cli("stats", "--top", "-1", "--format", "json")
        assert code == 1
        assert json.loads(err)["error"]["code"] == "invalid_filter"
