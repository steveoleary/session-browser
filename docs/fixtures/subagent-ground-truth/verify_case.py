#!/usr/bin/env python3
"""Check subagent linking against a parent->child graph recorded at spawn time.

``ground_truth.json`` was read from the spawning side of every edge -- the
tool result that answered each spawn call -- which is not the field discovery
reads, so a linking bug cannot agree with itself here. See README.md.
"""

from __future__ import annotations

import json
import os
import shutil
import sqlite3
import subprocess
import sys
import tempfile
from pathlib import Path

FIXTURE_DIR = Path(__file__).resolve().parent
REPO_ROOT = FIXTURE_DIR.parents[2]
TRUTH = json.loads((FIXTURE_DIR / "ground_truth.json").read_text())
OPENCODE_ROWS = json.loads((FIXTURE_DIR / "opencode_sessions.json").read_text())

CLAUDE = TRUTH["claude"]
OPENCODE = TRUTH["opencode"]
CLAUDE_PARENT = CLAUDE["parent_session"]
SUBAGENTS_DIR = (
    FIXTURE_DIR
    / "home"
    / ".claude"
    / "projects"
    / "-fixture-spawn-claude"
    / CLAUDE_PARENT
    / "subagents"
)


def build_home(root: Path) -> Path:
    """A private copy of ``home/`` plus an OpenCode database built from JSON.

    The database is generated rather than committed so the rows stay
    reviewable in a diff, and it goes into a temporary copy so the verifier
    never writes into the repository.
    """
    home = root / "home"
    shutil.copytree(FIXTURE_DIR / "home", home)
    db_dir = home / ".local" / "share" / "opencode"
    db_dir.mkdir(parents=True)
    conn = sqlite3.connect(db_dir / "opencode.db")
    conn.executescript(
        "CREATE TABLE project (id TEXT PRIMARY KEY, worktree TEXT, name TEXT);"
        "CREATE TABLE session (id TEXT PRIMARY KEY, project_id TEXT, parent_id TEXT,"
        " title TEXT, directory TEXT, agent TEXT, time_created INTEGER,"
        " time_updated INTEGER);"
    )
    conn.executemany(
        "INSERT INTO session (id, project_id, parent_id, title, directory, agent,"
        " time_created, time_updated) VALUES (?, NULL, ?, ?, ?, ?, ?, ?)",
        [
            (
                r["id"],
                r["parent_id"],
                r["title"],
                r["directory"],
                r["agent"],
                r["time_created"],
                r["time_updated"],
            )
            for r in OPENCODE_ROWS
        ],
    )
    conn.commit()
    conn.close()
    return home


def list_sessions(home: Path, provider: str) -> list[dict]:
    env = os.environ.copy()
    env["HOME"] = str(home)
    for var in ("CLAUDE_CODE_SESSION_ID", "CODEX_THREAD_ID"):
        env.pop(var, None)
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "session_browser.app",
            "list",
            "--provider",
            provider,
            "--format",
            "json",
        ],
        cwd=REPO_ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )
    if result.returncode:
        sys.stderr.write(result.stderr)
        raise SystemExit(f"session-browser list --provider {provider} failed")
    payload = json.loads(result.stdout)
    return payload["sessions"] if isinstance(payload, dict) else payload


# ----------------------------------------------------------------- raw facts
# These hold in both states. They are claims about what the providers write,
# checked against the recorded graph with no session-browser code involved, so
# a later implementation can rely on them -- and a provider upgrade that breaks
# one shows up here rather than as a mysteriously wrong tree.


def check_claude_raw_facts() -> None:
    edges = CLAUDE["edges"]
    by_child = {e["child_agent_id"]: e for e in edges}
    files = sorted(SUBAGENTS_DIR.glob("agent-*.jsonl"))
    on_disk = {f.stem.removeprefix("agent-") for f in files}
    assert on_disk == set(by_child), f"subagent files {on_disk} != spawned {by_child}"

    # A grandchild is stored FLAT beside its parent agent, not nested under
    # it: nesting lives in the sidecar, never in the directory layout.
    nested = [p for p in SUBAGENTS_DIR.iterdir() if p.is_dir()]
    assert not nested, f"unexpected nested directories: {nested}"

    for f in files:
        agent_id = f.stem.removeprefix("agent-")
        edge = by_child[agent_id]
        meta = json.loads(f.with_name(f.stem + ".meta.json").read_text())
        assert meta["toolUseId"] == edge["tool_use_id"], (agent_id, meta)
        assert meta["description"] == edge["description"], (agent_id, meta)
        if edge["spawner"] == CLAUDE_PARENT:
            assert meta["spawnDepth"] == 1 and "parentAgentId" not in meta, meta
        else:
            assert meta["spawnDepth"] == 2, meta
            assert meta["parentAgentId"] == edge["spawner"], meta

        first = json.loads(f.read_text().splitlines()[0])
        # sessionId on a subagent's records names the PARENT session. Using it
        # as the child's id files every child under its parent's identity --
        # the same trap Codex's payload.session_id sets.
        assert first["sessionId"] == CLAUDE_PARENT, first["sessionId"]
        assert first["agentId"] == agent_id and first["isSidechain"] is True
        assert first["promptId"] == edge["spawn_prompt_id"], (agent_id, first)

    # promptId names the ROOT turn in the parent, not the direct spawner: the
    # grandchild shares it with the child that spawned it. So it groups a
    # fan-out, and it cannot on its own say which agent spawned which.
    turns: dict[str, set[str]] = {}
    for e in edges:
        turns.setdefault(e["spawn_prompt_id"], set()).add(e["child_agent_id"])
    grouped = sorted(
        sorted(by_child[c]["description"] for c in g) for g in turns.values()
    )
    assert grouped == [["alpha", "beta"], ["delta", "gamma"]], grouped


def check_opencode_raw_facts() -> None:
    rows = {r["id"]: r for r in OPENCODE_ROWS}
    fork = OPENCODE["fork"]
    # A fork is not a child: it records no parent at all...
    assert rows[fork["session"]]["parent_id"] is None
    # ...but it carries copies of its source's spawn records, still naming the
    # source. Rebuilding edges from task parts would credit both to the fork.
    copied = OPENCODE["fork_copied_spawn_records"]
    assert {c["in_session"] for c in copied} == {fork["session"]}, copied
    assert {c["names_spawner"] for c in copied} == {fork["forked_from"]}, copied
    for e in OPENCODE["edges"]:
        assert rows[e["child"]]["parent_id"] == e["spawner"], e


# ------------------------------------------------------- session-browser view


def check_opencode_graph(home: Path) -> None:
    listed = {s["id"]: s for s in list_sessions(home, "opencode")}
    expected_ids = {f"opencode:{r['id']}" for r in OPENCODE_ROWS}
    assert set(listed) == expected_ids, sorted(set(listed) ^ expected_ids)
    spawner_of = {e["child"]: e["spawner"] for e in OPENCODE["edges"]}
    for row in OPENCODE_ROWS:
        s = listed[f"opencode:{row['id']}"]
        if row["id"] in spawner_of:
            assert s["parent_id"] == spawner_of[row["id"]], s
            assert s["subagent_kind"] == "general", s
        else:
            assert s["parent_id"] == "" and s["subagent_kind"] == "", s


def claude_children(listed: list[dict]) -> dict[str, dict]:
    """Listed sessions that are recorded children, keyed by agent id.

    Matched by agent-id suffix on purpose: what id a Claude subagent should be
    addressed by is not decided, and this fixture must not decide it.
    """
    out = {}
    for agent_id in (e["child_agent_id"] for e in CLAUDE["edges"]):
        hits = [s for s in listed if s["id"].endswith(agent_id)]
        assert len(hits) <= 1, (agent_id, hits)
        if hits:
            out[agent_id] = hits[0]
    return out


def check_baseline() -> None:
    check_claude_raw_facts()
    check_opencode_raw_facts()
    with tempfile.TemporaryDirectory() as tmp:
        home = build_home(Path(tmp))
        check_opencode_graph(home)
        listed = list_sessions(home, "claude")
        ids = [s["id"] for s in listed]
        assert ids == [f"claude:{CLAUDE_PARENT}"], ids
        assert listed[0]["parent_id"] is None, listed[0]
        found = claude_children(listed)
        assert not found, f"Claude subagents are now discovered: {sorted(found)}"
    print(
        "PASS: OpenCode graph matches spawn-time ground truth; "
        "Claude subagents are still invisible to discovery"
    )


def check_candidate() -> None:
    check_claude_raw_facts()
    check_opencode_raw_facts()
    with tempfile.TemporaryDirectory() as tmp:
        home = build_home(Path(tmp))
        check_opencode_graph(home)
        listed = list_sessions(home, "claude")
        found = claude_children(listed)
        missing = sorted({e["child_agent_id"] for e in CLAUDE["edges"]} - set(found))
        assert not missing, f"Claude subagents not discovered: {missing}"
        parent = next(s for s in listed if s["id"] == f"claude:{CLAUDE_PARENT}")
        assert parent["parent_id"] == "", parent
        for e in CLAUDE["edges"]:
            child = found[e["child_agent_id"]]
            if e["spawner"] == CLAUDE_PARENT:
                want = CLAUDE_PARENT
            else:
                want = found[e["spawner"]]["id"].split(":", 1)[1]
            assert child["parent_id"] == want, (e, child)
            assert child["subagent_kind"], child
    print("PASS: Claude and OpenCode graphs match spawn-time ground truth")


def main() -> None:
    modes = {"baseline": check_baseline, "candidate": check_candidate}
    if len(sys.argv) != 2:
        raise SystemExit(f"usage: {Path(sys.argv[0]).name} {{baseline,candidate}}")
    try:
        check = modes[sys.argv[1]]
    except KeyError:
        raise SystemExit(
            f"unknown mode {sys.argv[1]!r}; choose baseline or candidate"
        ) from None
    check()


if __name__ == "__main__":
    main()
