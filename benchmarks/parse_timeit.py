"""Time the transcript parse at another revision against this checkout.

    python benchmarks/parse_timeit.py                       # HEAD vs the tree
    python benchmarks/parse_timeit.py --rev main~3 --provider codex

When a loop opcode probe over the parse moves -- loop.parse_jsonl,
loop.claude_tool_entries and loop.codex_entries all drive ``_parse_jsonl`` --
AGENTS.md asks for the two forms timed directly on realistic input, since the
retrieval comparator cannot resolve an effect that small. This parses real
transcripts from this machine with each revision's ``_parse_jsonl`` and record
parser: small ones, where a cost paid once per file shows, and mid-sized ones,
where a cost paid per line does. It prints the best of several runs of each,
twice, so drift between the two revisions is visible, and their ratio. It
only reads transcripts; neither cache is touched.

Only ``transcript.py`` is taken from the revision. What it imports comes from
this checkout, which is right for a change to the parse loop and wrong for
one elsewhere. The revision is loaded as a module of its own, registered in
``sys.modules`` before it runs: its dataclasses look their module up there,
and a load that skips that fails inside ``dataclasses`` before a line is
timed.
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import subprocess
import sys
import tempfile
import timeit
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# The record parser each provider's rollouts go through. Codex's own-records
# variant for spawned children and its turn dedupe sit outside the loop the
# probes count, so they are left out here too.
PARSERS = {"claude": "_claude_entries", "codex": "_codex_entries", "pi": "_pi_entries"}


def load_revision(rev: str, directory: Path):
    """``session_browser/transcript.py`` as it stood at *rev*, as a module."""
    source = subprocess.run(
        ["git", "-C", str(ROOT), "show", f"{rev}:session_browser/transcript.py"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    path = directory / "transcript_at_revision.py"
    path.write_text(source)
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def transcripts(provider: str, count: int) -> dict[str, list[str]]:
    """Plain transcripts of *provider* on this machine, in two size bands.

    Compressed rollouts are left out: a revision from before the parser read
    them as streams cannot open one.
    """
    from session_browser.discovery import discover_all

    files = sorted(
        {
            s.content_path
            for s in discover_all(providers=[provider])
            if s.content_path.endswith(".jsonl") and os.path.isfile(s.content_path)
        },
        key=os.path.getsize,
    )
    return {
        "small (2-20 KB)": [f for f in files if 2_000 <= os.path.getsize(f) < 20_000][
            : count * 10
        ],
        "mid (0.2-2 MB)": [
            f for f in files if 200_000 <= os.path.getsize(f) < 2_000_000
        ][:count],
    }


def parse_all(module, parser_name: str, files: list[str]) -> None:
    event_fn = getattr(module, parser_name)
    for f in files:
        for _ in module._parse_jsonl(Path(f), [], event_fn):
            pass


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--rev", default="HEAD", help="revision to compare against")
    parser.add_argument("--provider", choices=sorted(PARSERS), default="claude")
    parser.add_argument(
        "--count",
        type=int,
        default=20,
        help="mid-sized transcripts to parse, and ten times as many small ones",
    )
    parser.add_argument("--repeat", type=int, default=7)
    args = parser.parse_args()

    from session_browser import transcript as current

    with tempfile.TemporaryDirectory(prefix="sb-parse-timeit-") as tmp:
        then = load_revision(args.rev, Path(tmp))
    name = PARSERS[args.provider]
    for band, files in transcripts(args.provider, args.count).items():
        if not files:
            print(f"{band}: no {args.provider} transcripts of that size here")
            continue
        for _ in range(2):
            runs = {}
            for label, module in ((args.rev, then), ("working tree", current)):
                best = min(
                    timeit.repeat(
                        lambda m=module, fs=files: parse_all(m, name, fs),
                        number=3,
                        repeat=args.repeat,
                    )
                )
                runs[label] = best / 3
            old, new = runs[args.rev], runs["working tree"]
            print(
                f"{band}, {len(files)} files: {args.rev} {old * 1000:.2f} ms, "
                f"working tree {new * 1000:.2f} ms, ratio {new / old:.4f}"
            )
    return 0


if __name__ == "__main__":
    sys.exit(main())
