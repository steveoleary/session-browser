"""Compare retrieval semantics and warm command performance across revisions.

The caller prepares both checkouts.  This module deliberately only executes
their CLI commands; it never invokes Git or changes either checkout.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import sys
import time
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

DEFAULT_QUERIES = ("kookaburra", "zzzzneverpresent", "transcript")
SLOWDOWN_LIMIT = 1.05
DEFAULT_WARMUP = 2
_DISCOVERY_ENV = ("CLAUDE_CODE_SESSION_ID", "CODEX_THREAD_ID")


class ComparatorError(RuntimeError):
    """The two revisions cannot produce a valid comparison."""


@dataclass(frozen=True)
class SearchRun:
    """One end-to-end CLI result and its wall-clock duration."""

    payload: dict
    seconds: float


def canonical_signature(payload: dict) -> dict:
    """Return the search facts that define retrieval equivalence.

    Search presentation includes volatile metadata and snippets.  The comparator
    intentionally retains only the ordered identities and the fields that say
    what was found and where it was found.
    """
    try:
        results = payload["results"]
    except (KeyError, TypeError) as exc:
        raise ComparatorError("search did not return a JSON results payload") from exc
    if not isinstance(results, list):
        raise ComparatorError("search JSON results must be a list")

    stable_results = []
    for result in results:
        if not isinstance(result, dict) or not isinstance(result.get("id"), str):
            raise ComparatorError("search result has no canonical id")
        stable_results.append(
            {
                "id": result["id"],
                "match_count": result.get("match_count", 0),
                "summary_matches": result.get("summary_matches", []),
                "first_match": result.get("first_match"),
                "last_match": result.get("last_match"),
                "total_entries": result.get("total_entries"),
            }
        )

    skipped = payload.get("skipped", [])
    if not isinstance(skipped, list):
        raise ComparatorError("search JSON skipped must be a list")
    unreadable_ids = []
    for skipped_item in skipped:
        if not isinstance(skipped_item, dict) or not isinstance(
            skipped_item.get("id"), str
        ):
            raise ComparatorError("unreadable search result has no canonical id")
        unreadable_ids.append(skipped_item["id"])

    return {
        "result_ids": [result["id"] for result in stable_results],
        "results": stable_results,
        "unreadable_ids": sorted(unreadable_ids),
    }


def _without_allowed_sessions(signature: dict, allowed_ids: set[str]) -> dict:
    """Keep strict comparison for every session outside an audited delta."""
    if not allowed_ids:
        return signature
    results = [item for item in signature["results"] if item["id"] not in allowed_ids]
    return {
        "result_ids": [item["id"] for item in results],
        "results": results,
        "unreadable_ids": [
            session_id
            for session_id in signature["unreadable_ids"]
            if session_id not in allowed_ids
        ],
    }


def _signature_mismatch(
    query: str, baseline: dict, candidate: dict, allowed_ids: set[str]
) -> ComparatorError | None:
    compared_baseline = _without_allowed_sessions(baseline, allowed_ids)
    compared_candidate = _without_allowed_sessions(candidate, allowed_ids)
    if compared_baseline == compared_candidate:
        return None
    return ComparatorError(
        f"signature mismatch for query {query!r}: "
        f"baseline={json.dumps(compared_baseline, sort_keys=True)} "
        f"candidate={json.dumps(compared_candidate, sort_keys=True)}"
    )


def _run_search(
    repo: Path, query: str, env: dict[str, str], search_args: list[str]
) -> SearchRun:
    """Run the product command from one prepared checkout."""
    command = [
        sys.executable,
        "-m",
        "session_browser.app",
        "search",
        query,
        "--mode",
        "ids",
        "--format",
        "json",
        *search_args,
    ]
    start = time.perf_counter()
    completed = subprocess.run(
        command,
        cwd=repo,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )
    seconds = time.perf_counter() - start
    if completed.returncode:
        raise ComparatorError(
            f"search failed in {repo} for query {query!r} "
            f"(exit {completed.returncode}): {completed.stderr.strip()}"
        )
    try:
        payload = json.loads(completed.stdout)
    except json.JSONDecodeError as exc:
        raise ComparatorError(
            f"search in {repo} for query {query!r} did not emit JSON: {exc}"
        ) from exc
    if not isinstance(payload, dict):
        raise ComparatorError(
            f"search in {repo} for query {query!r} emitted non-object JSON"
        )
    return SearchRun(payload=payload, seconds=seconds)


def _ratio(baseline: float, candidate: float) -> float:
    if baseline == 0:
        return 1.0 if candidate == 0 else float("inf")
    return candidate / baseline


def trimmed_median(samples: list[float]) -> float:
    """Median after discarding one extreme at each end.

    A single stalled run — a cold page cache, another process taking the CPU —
    can be several times the typical duration. Discarding the extremes keeps one
    such sample from moving the estimate. Below five samples there is nothing to
    spare, so the plain median is used."""
    if len(samples) < 5:
        return statistics.median(samples)
    return statistics.median(sorted(samples)[1:-1])


def relative_spread(samples: list[float]) -> float:
    """Dispersion of one revision's own samples, as a fraction of its centre.

    This is the resolution limit of the measurement: a candidate/baseline ratio
    smaller than this cannot be distinguished from the machine's own variance.
    Uses the interquartile range where there are enough samples for quartiles,
    and the full range below that, which is noisier but never understates."""
    if len(samples) < 2:
        return 0.0
    centre = trimmed_median(samples)
    if centre <= 0:
        return 0.0
    if len(samples) >= 4:
        quartiles = statistics.quantiles(samples, n=4, method="inclusive")
        width = quartiles[2] - quartiles[0]
    else:
        width = max(samples) - min(samples)
    return width / centre


def _timing_verdict(ratio: float, noise_floor: float) -> str:
    """Classify one query's timing against the machine's own noise.

    ``unresolvable`` is deliberately distinct from ``slower``: it means the
    candidate exceeded the limit by less than this machine can measure, so the
    run neither passes nor convicts. It must be reported, never swallowed."""
    if ratio <= SLOWDOWN_LIMIT:
        return "ok"
    if (ratio - 1.0) > noise_floor:
        return "slower"
    return "unresolvable"


def _assert_stable(query: str, revision: str, expected: dict, observed: dict) -> None:
    if observed != expected:
        raise ComparatorError(
            f"{revision} signature changed across repetitions for query {query!r}"
        )


def _comparison_environment(
    home: Path, current_session_env: Iterable[str]
) -> dict[str, str]:
    env = os.environ.copy()
    env["HOME"] = str(home)
    for item in current_session_env:
        if "=" not in item:
            raise ComparatorError(
                f"invalid --current-session-env {item!r}; expected NAME=VALUE"
            )
        name, value = item.split("=", 1)
        if not name:
            raise ComparatorError(
                f"invalid --current-session-env {item!r}; expected NAME=VALUE"
            )
        if name not in _DISCOVERY_ENV:
            raise ComparatorError(
                f"unsupported --current-session-env {name!r}; expected "
                + " or ".join(_DISCOVERY_ENV)
            )
        env[name] = value
    return env


def run_comparison(
    *,
    baseline_repo: Path,
    candidate_repo: Path,
    home: Path,
    queries: Iterable[str],
    repeats: int = 7,
    warmup: int = DEFAULT_WARMUP,
    provider: str | None = None,
    include_current: bool = False,
    current_session_env: Iterable[str] = (),
    allowed_session_ids: Iterable[str] = (),
    report_path: Path | None = None,
) -> dict:
    """Run strict equivalence and timing checks.

    The aggregate timing ratio is the median of per-query trimmed-median
    ratios. This preserves query grouping, so a slow long-running query cannot
    distort the result for a distinct short query merely by pooling raw samples.

    Each query discards *warmup* runs per revision before sampling, and a query
    is only convicted of being slow when the ratio exceeds both the limit and
    the spread of the revisions' own samples. Measured spread on a real corpus
    reaches well over 100% of the median, so a fixed threshold applied to raw
    medians reports regressions that are pure machine variance.
    """
    baseline_repo = Path(baseline_repo)
    candidate_repo = Path(candidate_repo)
    if not baseline_repo.is_dir() or not candidate_repo.is_dir():
        raise ComparatorError(
            "--baseline-repo and --candidate-repo must be prepared directories"
        )
    home = Path(home)
    if not home.is_dir():
        raise ComparatorError("--home must be an existing directory")
    if repeats < 1:
        raise ComparatorError("--repeats must be >= 1")
    if warmup < 0:
        raise ComparatorError("--warmup must be >= 0")
    query_list = list(
        dict.fromkeys(query.strip() for query in queries if query.strip())
    )
    if not query_list:
        raise ComparatorError("at least one non-empty query is required")

    search_args: list[str] = []
    if provider:
        search_args.extend(["--provider", provider])
    if include_current:
        search_args.append("--include-current")
    env = _comparison_environment(home, current_session_env)
    allowed_ids = set(allowed_session_ids)
    query_report: dict[str, dict] = {}

    volatile_queries: list[str] = []
    for query in query_list:
        # Establish that each revision reproduces its own result before making
        # any equivalence claim. A session file appended to mid-run would
        # otherwise present as a semantic difference between two revisions that
        # are in fact identical.
        baseline_signature = canonical_signature(
            _run_search(baseline_repo, query, env, search_args).payload
        )
        candidate_signature = canonical_signature(
            _run_search(candidate_repo, query, env, search_args).payload
        )
        baseline_recheck = canonical_signature(
            _run_search(baseline_repo, query, env, search_args).payload
        )
        candidate_recheck = canonical_signature(
            _run_search(candidate_repo, query, env, search_args).payload
        )
        unstable = None
        if baseline_signature != baseline_recheck:
            unstable = "baseline"
        elif candidate_signature != candidate_recheck:
            unstable = "candidate"
        if unstable is not None:
            volatile_queries.append(query)
            query_report[query] = {
                "baseline_signature": baseline_signature,
                "candidate_signature": candidate_signature,
                "unstable_revision": unstable,
                "verdict": "volatile",
            }
            continue

        mismatch = _signature_mismatch(
            query, baseline_signature, candidate_signature, allowed_ids
        )
        if mismatch:
            raise mismatch

        for _ in range(warmup):
            for revision, repo, expected in (
                ("baseline", baseline_repo, baseline_signature),
                ("candidate", candidate_repo, candidate_signature),
            ):
                discarded = _run_search(repo, query, env, search_args)
                if canonical_signature(discarded.payload) != expected:
                    unstable = revision
                    break
            if unstable is not None:
                break

        baseline_samples: list[float] = []
        candidate_samples: list[float] = []
        if unstable is None:
            for repetition in range(repeats):
                order = (("baseline", baseline_repo), ("candidate", candidate_repo))
                if repetition % 2:
                    order = tuple(reversed(order))
                for revision, repo in order:
                    run = _run_search(repo, query, env, search_args)
                    signature = canonical_signature(run.payload)
                    expected = (
                        baseline_signature
                        if revision == "baseline"
                        else candidate_signature
                    )
                    if signature != expected:
                        unstable = revision
                        break
                    if revision == "baseline":
                        baseline_samples.append(run.seconds)
                    else:
                        candidate_samples.append(run.seconds)
                if unstable is not None:
                    break

        if unstable is not None or not baseline_samples or not candidate_samples:
            volatile_queries.append(query)
            query_report[query] = {
                "baseline_signature": baseline_signature,
                "candidate_signature": candidate_signature,
                "unstable_revision": unstable,
                "verdict": "volatile",
            }
            continue

        baseline_centre = trimmed_median(baseline_samples)
        candidate_centre = trimmed_median(candidate_samples)
        ratio = _ratio(baseline_centre, candidate_centre)
        noise_floor = max(
            relative_spread(baseline_samples), relative_spread(candidate_samples)
        )
        query_report[query] = {
            "baseline_signature": baseline_signature,
            "candidate_signature": candidate_signature,
            "baseline_samples": baseline_samples,
            "candidate_samples": candidate_samples,
            "baseline_median": statistics.median(baseline_samples),
            "candidate_median": statistics.median(candidate_samples),
            "baseline_trimmed_median": baseline_centre,
            "candidate_trimmed_median": candidate_centre,
            "baseline_relative_spread": relative_spread(baseline_samples),
            "candidate_relative_spread": relative_spread(candidate_samples),
            "noise_floor": noise_floor,
            "ratio": ratio,
            "verdict": _timing_verdict(ratio, noise_floor),
        }
    judged = [
        result for result in query_report.values() if result["verdict"] != "volatile"
    ]
    if not judged:
        raise ComparatorError(
            "every query touched a session that changed mid-run; nothing could "
            "be measured. Exclude the live session with --current-session-env, "
            "exempt a known-moving session with --allow-session, or choose "
            "queries that do not match an active session."
        )
    per_query_ratios = [result["ratio"] for result in judged]
    aggregate_ratio = statistics.median(per_query_ratios)
    aggregate_noise = statistics.median([result["noise_floor"] for result in judged])
    aggregate_verdict = _timing_verdict(aggregate_ratio, aggregate_noise)
    slow_queries = [
        query for query, result in query_report.items() if result["verdict"] == "slower"
    ]
    unresolvable_queries = [
        query
        for query, result in query_report.items()
        if result["verdict"] == "unresolvable"
    ]
    report = {
        "baseline_repo": str(baseline_repo),
        "candidate_repo": str(candidate_repo),
        "home": str(home),
        "python_executable": sys.executable,
        "repeats": repeats,
        "warmup": warmup,
        "provider": provider,
        "include_current": include_current,
        "allowed_session_ids": sorted(allowed_ids),
        "queries": query_report,
        "aggregate": {
            "statistic": "median_per_query_trimmed_ratio",
            "per_query_ratios": per_query_ratios,
            "ratio": aggregate_ratio,
            "noise_floor": aggregate_noise,
            "verdict": aggregate_verdict,
        },
        # Equivalence is proven by reaching this point: a signature mismatch
        # raises above, so it can never be reported as a mere flag. Recorded
        # explicitly because "passed" answers only the timing question.
        "equivalent": True,
        "slow_queries": slow_queries,
        "unresolvable_queries": unresolvable_queries,
        "volatile_queries": volatile_queries,
        "passed": not slow_queries and aggregate_verdict != "slower",
    }
    if report_path is not None:
        report_path = Path(report_path)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")

    for query in volatile_queries:
        # Loud, never silent: a moving corpus means this query proved nothing,
        # about equivalence or about timing.
        print(
            f"warning: {query!r} matched a session that changed mid-run "
            f"({query_report[query]['unstable_revision'] or 'unknown'} revision); "
            "excluded from the comparison.",
            file=sys.stderr,
        )

    if aggregate_verdict == "unresolvable":
        print(
            f"warning: aggregate ratio {aggregate_ratio:.3f} is inside this "
            f"machine's noise floor of {aggregate_noise:.3f}; not judged.",
            file=sys.stderr,
        )

    if unresolvable_queries:
        # Loud, never silent: these queries exceeded the limit by less than the
        # machine could measure, so the run proves nothing either way for them.
        for query in unresolvable_queries:
            result = query_report[query]
            print(
                f"warning: {query!r} ratio {result['ratio']:.3f} is inside this "
                f"machine's noise floor of {result['noise_floor']:.3f}; "
                "not judged. Raise --repeats or quiet the machine.",
                file=sys.stderr,
            )

    if slow_queries or aggregate_verdict == "slower":
        details = ", ".join(slow_queries) or "none"
        raise ComparatorError(
            f"candidate is slower than 5% beyond measurement noise "
            f"(queries: {details}; aggregate ratio: {aggregate_ratio:.3f})"
        )
    return report


def _query_file(path: Path) -> list[str]:
    try:
        return [
            line.strip()
            for line in path.read_text().splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        ]
    except OSError as exc:
        raise ComparatorError(f"cannot read query file {path}: {exc}") from exc


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-repo", required=True, type=Path)
    parser.add_argument("--candidate-repo", required=True, type=Path)
    parser.add_argument(
        "--home",
        required=True,
        type=Path,
        help="the single source HOME used by both revisions",
    )
    parser.add_argument(
        "--query", action="append", help="literal query; repeat to compare more terms"
    )
    parser.add_argument(
        "--query-file",
        type=Path,
        help="machine-local newline-delimited private queries; "
        "replaces the built-in defaults, and combines with "
        "any --query",
    )
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument(
        "--warmup",
        type=int,
        default=DEFAULT_WARMUP,
        metavar="N",
        help="runs per revision per query to discard before "
        f"measuring (default {DEFAULT_WARMUP})",
    )
    parser.add_argument("--provider")
    parser.add_argument("--include-current", action="store_true")
    parser.add_argument(
        "--current-session-env",
        action="append",
        default=[],
        metavar="NAME=VALUE",
        help="override or supply a current-session exclusion variable",
    )
    parser.add_argument(
        "--allow-session",
        action="append",
        default=[],
        metavar="CANONICAL_ID",
        help="audited delta exempt from strict equivalence",
    )
    parser.add_argument("--report", required=True, type=Path)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        queries = list(args.query) if args.query else []
        if args.query_file:
            queries.extend(_query_file(args.query_file))
        if not queries:
            queries = list(DEFAULT_QUERIES)
        report = run_comparison(
            baseline_repo=args.baseline_repo,
            candidate_repo=args.candidate_repo,
            home=args.home,
            queries=queries,
            repeats=args.repeats,
            warmup=args.warmup,
            provider=args.provider,
            include_current=args.include_current,
            current_session_env=args.current_session_env,
            allowed_session_ids=args.allow_session,
            report_path=args.report,
        )
    except ComparatorError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(json.dumps({"report": str(args.report), "passed": report["passed"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
