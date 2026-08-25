"""Build a synthetic corpus and compare two revisions against it, in one shot.

``retrieval_compare.py`` needs a prepared ``--home``; ``synthetic_corpus.py``
builds one. Doing both by hand is three commands and a temporary directory
someone has to remember to delete, which is how a comparator ends up pointed at
a real ``$HOME`` again. So:

    python benchmarks/compare_synthetic.py \\
      --baseline-repo /tmp/baseline-main --candidate-repo . \\
      --scale 8 --report /tmp/ab.json

What this wrapper adds beyond convenience:

* **The corpus is deleted.** A x8 corpus is tens of megabytes and exists only
  for the length of one comparison. It goes to a temporary directory that is
  removed even when the comparison raises — a failing run is exactly when a
  forgotten corpus would survive. ``--keep-corpus`` overrides that, and a
  ``--corpus`` you name yourself is never removed, because you named it.
* **The report says what it was measured against.** The corpus manifest is
  merged into the report under ``corpus``, so a ratio read back next month
  carries the scale, seed and the exact session count behind every query. The
  corpus itself will be long gone.
* **Queries are chosen by selectivity, not by name.** The default set spans the
  cost curve — a term in no session at all (discovery and the prefilter, with
  nothing parsed), one in ten, and one in every session (the parser reads the
  whole corpus). A change that only touches parsing is invisible to the first
  and loudest in the last.
* **A run that could not have failed says so.** ``ok`` from a run whose noise
  floor sits above the limit means "no regression was resolvable", which is a
  different claim from "no regression exists". Both are recorded and the weaker
  one is warned about, under ``resolved_the_limit``.
* **No ``--current-session-env``.** A synthetic ``$HOME`` contains no live
  session, so the exclusion step a real-corpus run needs does not apply, and
  the ``volatile`` verdict it exists to prevent cannot arise.

It does not soften anything the comparator decides. ``--repeats`` is checked
against the comparator's own ``MIN_REPEATS`` *before* a corpus is built, so a
run that would be refused for sampling too thinly is refused before it spends a
minute writing files — the guard is honoured earlier, never argued with.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import tempfile
from pathlib import Path

try:
    from benchmarks import retrieval_compare, synthetic_corpus
except ModuleNotFoundError:  # run as a script: sys.path[0] is benchmarks/
    # The other benchmarks import nothing of each other, so this is the first
    # one to need the repo root on the path. Under pytest it is already there.
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from benchmarks import retrieval_compare, synthetic_corpus

# Absent, one session in ten, every session. Three points on the cost curve
# rather than three arbitrary words: the aggregate is a median of per-query
# ratios, so the set decides which kind of regression the run can see.
DEFAULT_SELECTIVITIES = (0.0, 0.10, 1.0)


class WrapperError(RuntimeError):
    """The comparison cannot be set up."""


def queries_for(selectivities: list[float]) -> list[str]:
    """Resolve requested selectivities to the terms the generator plants."""
    available = {term.selectivity: term.term for term in synthetic_corpus.TERMS}
    queries = []
    for wanted in selectivities:
        try:
            queries.append(available[wanted])
        except KeyError as exc:
            offered = ", ".join(str(value) for value in sorted(available))
            raise WrapperError(
                f"no term is planted at selectivity {wanted}; the generator "
                f"plants {offered}"
            ) from exc
    return queries


def _merge_manifest(report_path: Path, manifest: dict) -> None:
    """Record the corpus inside the report, whatever the verdict was.

    Called after a failure as well as after a pass: a report that convicts a
    candidate is the one most likely to be read months later, and by then the
    corpus it was measured against has been deleted.
    """
    if not report_path.exists():
        return
    report = json.loads(report_path.read_text())
    report["corpus"] = manifest
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")


def run(
    *,
    baseline_repo: Path,
    candidate_repo: Path,
    report_path: Path,
    scale: int = 1,
    seed: int = synthetic_corpus.DEFAULT_SEED,
    corpus: Path | None = None,
    keep_corpus: bool = False,
    queries: list[str] | None = None,
    selectivities: list[float] | None = None,
    repeats: int = 7,
    warmup: int = retrieval_compare.DEFAULT_WARMUP,
    provider: str | None = None,
) -> dict:
    """Build a corpus, compare both revisions against it, then clean up."""
    if repeats < retrieval_compare.MIN_REPEATS:
        # The comparator would refuse this anyway, and for a good reason: below
        # MIN_REPEATS its noise floor stops being an interquartile range and
        # becomes a full range, which is wider, and a wider floor *accepts*
        # more. Refusing here as well only moves the refusal ahead of the
        # minutes spent writing a corpus that would then go unused.
        raise WrapperError(
            f"--repeats must be >= {retrieval_compare.MIN_REPEATS}; got "
            f"{repeats}. The comparator refuses to rest a verdict on a noise "
            "floor estimated from fewer samples, and no corpus size fixes that."
        )
    if queries is None:
        queries = queries_for(
            list(DEFAULT_SELECTIVITIES if selectivities is None else selectivities)
        )
    if not queries:
        raise WrapperError("at least one query is required")

    named = corpus is not None
    home = Path(corpus) if named else Path(tempfile.mkdtemp(prefix="sb-corpus-"))
    remove = not named and not keep_corpus
    try:
        manifest = synthetic_corpus.generate(
            home, scale=scale, seed=seed, force=not named
        )
        report_path = Path(report_path)
        try:
            retrieval_compare.run_comparison(
                baseline_repo=baseline_repo,
                candidate_repo=candidate_repo,
                home=home,
                queries=queries,
                repeats=repeats,
                warmup=warmup,
                provider=provider,
                report_path=report_path,
            )
        finally:
            _merge_manifest(report_path, manifest)
    finally:
        if remove:
            shutil.rmtree(home, ignore_errors=True)
    report = json.loads(report_path.read_text())
    # A kept corpus is a temporary directory with a generated name, so the
    # caller cannot reconstruct where it went. --keep-corpus promises to say.
    report["corpus"]["home"] = str(home)
    report["resolved_the_limit"] = _warn_if_unresolvable(report)
    # Persist it: a report read back months later should carry whether the run
    # could have convicted anything, not just what it happened to conclude.
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return report


def _warn_if_unresolvable(report: dict) -> bool:
    """Say so when the run could not have seen a regression at the limit.

    A passing run and a run that could not have failed look identical from the
    outside: both print ``ok``. They are not the same claim. The comparator
    already refuses a *verdict* it cannot support, per query -- but the
    aggregate is a median of per-query floors, so a run whose floor sits above
    the limit still reports ``ok`` overall while being unable to convict
    anything smaller than its own jitter.

    Measured 2026-08-25, only one lever moves this, and it is not one the tool
    can pull. A bigger corpus does nothing: across 15 null runs the floor at a
    fixed scale varied more (0.019-0.081 at x16) than it did between scales,
    and the parse-sensitive share of a sample stayed flat near 47% from x1 to
    x32 because discovery scales with the corpus too. More samples do nothing
    either, and may hurt -- 9 repeats measured 0.008-0.013 on an idle machine
    where 25 measured 0.024, a longer sampling window admitting more drift.

    What moved it was an idle machine: the same null comparison scored
    0.020-0.031 while an agent worked and 0.008-0.013 with nothing else
    running. So the warning names that, and only that.
    """
    limit = retrieval_compare.SLOWDOWN_LIMIT - 1.0
    floor = report["aggregate"]["noise_floor"]
    if floor < limit:
        return True
    print(
        f"warning: this run's noise floor is {floor:.1%}, at or above the "
        f"{limit:.0%} the comparator judges against. A verdict of 'ok' here "
        "means no regression was resolvable, not that none exists. Quiet the "
        "machine and run it again -- measurably, that is the only lever: "
        "neither a larger --scale nor more --repeats moves this floor, and "
        "an agent working the machine is usually what raised it.",
        file=sys.stderr,
    )
    return False


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build a scalable synthetic corpus and run "
            "benchmarks/retrieval_compare.py against it."
        ),
        epilog=(
            "The corpus is temporary and is deleted afterwards, including when "
            "the comparison fails; its manifest is merged into the report so "
            "the numbers stay readable once it is gone. Verdicts are the "
            "comparator's own: ok, slower, unresolvable, volatile."
        ),
    )
    parser.add_argument("--baseline-repo", required=True, type=Path)
    parser.add_argument("--candidate-repo", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    parser.add_argument(
        "--scale",
        type=int,
        default=1,
        metavar="N",
        help=(
            "corpus size, in units of "
            f"{synthetic_corpus.BASE_TOTAL} sessions (default 1). Raise it to "
            "shrink the share of each sample spent on interpreter startup, "
            "which is what dilutes a regression before the comparator sees it."
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=synthetic_corpus.DEFAULT_SEED,
        metavar="N",
        help="corpus seed; the same seed and scale rebuild the same bytes",
    )
    parser.add_argument(
        "--corpus",
        type=Path,
        help=(
            "build in this directory instead of a temporary one, and do not "
            "delete it afterwards"
        ),
    )
    parser.add_argument(
        "--keep-corpus",
        action="store_true",
        help="keep the temporary corpus and print where it is",
    )
    parser.add_argument(
        "--query",
        action="append",
        help="literal query; replaces the selectivity-chosen defaults",
    )
    parser.add_argument(
        "--selectivity",
        action="append",
        type=float,
        metavar="F",
        help=(
            "choose a planted query by the fraction of sessions carrying it "
            "(repeatable; default "
            + ", ".join(str(value) for value in DEFAULT_SELECTIVITIES)
            + ")"
        ),
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=7,
        metavar="N",
        help=(
            f"measured runs per revision per query (default 7, minimum "
            f"{retrieval_compare.MIN_REPEATS})"
        ),
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=retrieval_compare.DEFAULT_WARMUP,
        metavar="N",
        help=f"discarded runs before measuring (default "
        f"{retrieval_compare.DEFAULT_WARMUP})",
    )
    parser.add_argument("--provider")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        report = run(
            baseline_repo=args.baseline_repo,
            candidate_repo=args.candidate_repo,
            report_path=args.report,
            scale=args.scale,
            seed=args.seed,
            corpus=args.corpus,
            keep_corpus=args.keep_corpus,
            queries=args.query,
            selectivities=args.selectivity,
            repeats=args.repeats,
            warmup=args.warmup,
            provider=args.provider,
        )
    except (
        WrapperError,
        retrieval_compare.ComparatorError,
        synthetic_corpus.CorpusError,
    ) as exc:
        print(f"error: {exc}", file=sys.stderr)
        if Path(args.report).exists():
            # A conviction is the report most worth reading, and it is already
            # written by the time the comparator raises. Say where it is.
            print(f"report: {args.report}", file=sys.stderr)
        return 1
    summary = {
        "report": str(args.report),
        "passed": report["passed"],
        "scale": report["corpus"]["scale"],
        "sessions": report["corpus"]["sessions"],
        "aggregate": report["aggregate"],
        "resolved_the_limit": report["resolved_the_limit"],
    }
    if args.corpus is not None or args.keep_corpus:
        summary["corpus"] = report["corpus"]["home"]
    print(json.dumps(summary))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
