#!/usr/bin/env python3
"""Regenerate this fixture's frozen ``home/`` tree.

The tree is committed, not built on demand: a fixture whose corpus moves
cannot have known answers, and known answers are the whole point of this case.
This script exists so the corpus stays *readable* — the shape of each trap is
easier to check here than across sixty JSONL lines — and so it can be rebuilt
byte-identically if a provider layout ever changes.

Run it from the repository root:

    .venv/bin/python docs/fixtures/fresh-agent-skill-brief/build_home.py

Every session is deliberately small and deliberately dull except where it is
carrying a trap. The four traps are the four the skill warns about, and each
one is arranged so that the obvious command gives a wrong answer that looks
exactly like a right one.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

FIXTURE_DIR = Path(__file__).resolve().parent
HOME = FIXTURE_DIR / "home"

# Two projects whose names are prefixes of one another. --repo and --cwd are
# substring filters, so every command scoped to "coffee_run" silently includes
# the private one as well.
PUBLIC = "/fixture/coffee_run"
PRIVATE = "/fixture/coffee_run_private"
OTHER = "/fixture/tea_break"


def _claude_dir(cwd: str) -> Path:
    return HOME / ".claude" / "projects" / cwd.replace("/", "-")


def _line(role: str, text: str, ts: str, cwd: str) -> str:
    if role == "user":
        message = {"role": "user", "content": text}
    else:
        message = {"role": "assistant", "content": [{"type": "text", "text": text}]}
    return json.dumps(
        {
            "type": role,
            "cwd": cwd,
            "gitBranch": "main",
            "timestamp": ts,
            "message": message,
        }
    )


def _tool_line(text: str, ts: str, cwd: str) -> str:
    """A tool *result*: text the agent read, not text anybody said."""
    return json.dumps(
        {
            "type": "user",
            "cwd": cwd,
            "gitBranch": "main",
            "timestamp": ts,
            "message": {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "toolu_fixture",
                        "content": text,
                    }
                ],
            },
        }
    )


def write(cwd: str, name: str, day: int, lines: list[str]) -> None:
    directory = _claude_dir(cwd)
    directory.mkdir(parents=True, exist_ok=True)
    (directory / f"{name}.jsonl").write_text("\n".join(lines) + "\n")


def build() -> None:
    if HOME.exists():
        shutil.rmtree(HOME)

    def ts(day: int, minute: int) -> str:
        return f"2026-04-{day:02d}T10:{minute:02d}:00.000Z"

    # --- coffee_run: three sessions, one of them unreadable ------------------
    write(
        PUBLIC,
        "fixture-coffee-0001",
        1,
        [
            _line("user", "plan the coffee run rota for April", ts(1, 0), PUBLIC),
            _line("assistant", "Rota drafted for four people.", ts(1, 1), PUBLIC),
        ],
    )
    write(
        PUBLIC,
        "fixture-coffee-0002",
        2,
        [
            _line("user", "who is on the rota this week", ts(2, 0), PUBLIC),
            _line("assistant", "Priya, then Sam.", ts(2, 1), PUBLIC),
        ],
    )
    # Trap 4: discovery finds this file; nothing can read it. `stats` counts
    # it because it never opens a transcript; `list` opens it, reports it in
    # counts.unreadable, and still returns it in .sessions.
    _claude_dir(PUBLIC).mkdir(parents=True, exist_ok=True)
    (_claude_dir(PUBLIC) / "fixture-coffee-0003.jsonl").write_text("")

    # --- coffee_run_private: the substring shadow ----------------------------
    write(
        PRIVATE,
        "fixture-private-0001",
        3,
        [
            _line("user", "budget for the private coffee run", ts(3, 0), PRIVATE),
            _line("assistant", "Twelve pounds a head.", ts(3, 1), PRIVATE),
        ],
    )
    write(
        PRIVATE,
        "fixture-private-0002",
        4,
        [
            _line("user", "confirm the private order", ts(4, 0), PRIVATE),
            _line("assistant", "Confirmed with the roastery.", ts(4, 1), PRIVATE),
        ],
    )
    write(
        PRIVATE,
        "fixture-private-0003",
        5,
        [
            _line("user", "cancel one of the private orders", ts(5, 0), PRIVATE),
            _line("assistant", "Cancelled.", ts(5, 1), PRIVATE),
        ],
    )

    # --- tea_break: provenance, and a decision that gets reversed ------------
    # Trap 2: the phrase is everywhere in this session, and every occurrence
    # but one is inside a file the agent read. A hit count says "heavily
    # discussed"; the roles say it was read four times and proposed once.
    phrase = "sunset the loyalty tier"
    write(
        OTHER,
        "fixture-tea-0001",
        6,
        [
            _line("user", "read the pricing memo and summarise it", ts(6, 0), OTHER),
            _tool_line(
                f"MEMO DRAFT\nOption one is to {phrase} in Q3.\n"
                f"Option two is to {phrase} only for lapsed accounts.\n"
                f"Finance would prefer we {phrase} after the migration.\n"
                f"Legal has no view on whether we {phrase}.",
                ts(6, 1),
                OTHER,
            ),
            _line(
                "assistant",
                "The memo lists four options; none is a recommendation.",
                ts(6, 2),
                OTHER,
            ),
            _line(
                "user",
                f"for what it is worth I think we should {phrase} in Q3",
                ts(6, 3),
                OTHER,
            ),
            _line("assistant", "Noted as your position.", ts(6, 4), OTHER),
        ],
    )
    # Trap 3: the session reverses itself. A mid-session snippet is confident
    # and wrong; the ending is the answer.
    write(
        OTHER,
        "fixture-tea-0002",
        7,
        [
            _line("user", "which espresso machine are we ordering", ts(7, 0), OTHER),
            _line(
                "assistant",
                "We are going with the Rocket R58. It fits the counter and the "
                "budget, so I have put it forward as the order.",
                ts(7, 1),
                OTHER,
            ),
            _line(
                "user",
                "the R58 is plumbed-in only and we have no water line",
                ts(7, 2),
                OTHER,
            ),
            _line(
                "assistant",
                "I was wrong about the Rocket R58 — it cannot run on a tank in "
                "the model we priced.",
                ts(7, 3),
                OTHER,
            ),
            _line(
                "assistant",
                "The order is the Lelit Bianca. It runs from a tank or a line, "
                "and it is the machine we are ordering.",
                ts(7, 4),
                OTHER,
            ),
        ],
    )
    print(f"wrote {sum(1 for _ in HOME.rglob('*.jsonl'))} sessions to {HOME}")


if __name__ == "__main__":
    build()
