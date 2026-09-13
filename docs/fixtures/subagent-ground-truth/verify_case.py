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
CODEX_INDEX = json.loads((FIXTURE_DIR / "codex_threads.json").read_text())

CLAUDE = TRUTH["claude"]
OPENCODE = TRUTH["opencode"]
CODEX = TRUTH["codex"]
CODEX_HOME = FIXTURE_DIR / "home" / ".codex"
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


def build_home(root: Path, *, codex_index: bool = True) -> Path:
    """A private copy of ``home/`` plus databases built from JSON.

    The databases are generated rather than committed so the rows stay
    reviewable in a diff, and they go into a temporary copy so the verifier
    never writes into the repository. ``codex_index=False`` leaves Codex's
    ``state_5.sqlite`` out, so discovery takes the rollout file scan instead.
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
    if codex_index:
        build_codex_index(home)
    return home


def build_codex_index(home: Path) -> None:
    """``~/.codex/state_5.sqlite`` from the rows Codex wrote for this run.

    Only the columns discovery reads, plus the ones the raw facts cite.
    ``rollout_path`` is stored relative to ``~/.codex`` and made absolute here,
    because discovery checks every rollout on disk has an index row.
    """
    conn = sqlite3.connect(home / ".codex" / "state_5.sqlite")
    conn.executescript(
        "CREATE TABLE threads (id TEXT PRIMARY KEY, rollout_path TEXT NOT NULL,"
        " created_at_ms INTEGER, updated_at_ms INTEGER, source TEXT NOT NULL,"
        " thread_source TEXT, cwd TEXT NOT NULL, git_branch TEXT,"
        " git_origin_url TEXT, first_user_message TEXT NOT NULL DEFAULT '',"
        " archived INTEGER NOT NULL DEFAULT 0, agent_path TEXT,"
        " agent_nickname TEXT);"
        "CREATE TABLE thread_spawn_edges (parent_thread_id TEXT NOT NULL,"
        " child_thread_id TEXT NOT NULL PRIMARY KEY, status TEXT NOT NULL);"
    )
    columns = list(CODEX_INDEX["threads"][0])
    conn.executemany(
        f"INSERT INTO threads ({', '.join(columns)})"
        f" VALUES ({', '.join('?' * len(columns))})",
        [
            tuple(
                str(home / ".codex" / row[c]) if c == "rollout_path" else row[c]
                for c in columns
            )
            for row in CODEX_INDEX["threads"]
        ],
    )
    conn.executemany(
        "INSERT INTO thread_spawn_edges VALUES (?, ?, ?)",
        [
            (e["parent_thread_id"], e["child_thread_id"], e["status"])
            for e in CODEX_INDEX["thread_spawn_edges"]
        ],
    )
    conn.commit()
    conn.close()


def list_sessions(home: Path, provider: str, *, quiet: bool = False) -> list[dict]:
    """``session-browser list`` against *home*.

    ``quiet`` fails on anything written to stderr. Discovery logs there when
    it abandons the Codex index for the file scan, so a quiet run is proof
    the index path produced the result, not a silent fallback.
    """
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
    assert not (quiet and result.stderr), result.stderr
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


def codex_rollout(thread_id: str) -> list[dict]:
    (path,) = CODEX_HOME.glob(f"sessions/*/*/*/rollout-*-{thread_id}.jsonl")
    return [json.loads(line) for line in path.read_text().splitlines()]


def check_codex_raw_facts() -> None:
    edges = CODEX["edges"]
    spawner_of = {e["child"]: e["spawner"] for e in edges}
    by_child = {e["child"]: e for e in edges}

    def root_of(thread_id: str) -> str:
        while thread_id in spawner_of:
            thread_id = spawner_of[thread_id]
        return thread_id

    def depth_of(thread_id: str) -> int:
        return 0 if thread_id not in spawner_of else 1 + depth_of(spawner_of[thread_id])

    def path_of(thread_id: str) -> str:
        e = by_child.get(thread_id)
        return "/root" if e is None else f"{path_of(e['spawner'])}/{e['task_name']}"

    for e in edges:
        child = e["child"]
        records = codex_rollout(child)
        metas = [r["payload"] for r in records if r["type"] == "session_meta"]
        meta = metas[0]
        spawn = meta["source"]["subagent"]["thread_spawn"]
        assert meta["id"] == child and meta["thread_source"] == "subagent", meta
        # Nesting is recursive: the parent is the DIRECT spawner, depth counts
        # from the root, and agent_path spells out the whole chain rather than
        # a leaf name.
        assert spawn["parent_thread_id"] == e["spawner"], (e, spawn)
        assert spawn["depth"] == depth_of(child), (e, spawn)
        assert spawn["agent_path"] == meta["agent_path"] == e["agent_path"], e
        assert e["agent_path"] == path_of(child), e
        # The spawner is also written at the TOP level, the shape agent-sessions
        # guards against -- here on a subagent, never on a fork.
        assert meta["parent_thread_id"] == e["spawner"], meta
        # EVERY SUBAGENT IS ALSO A FORK of its spawner. forked_from_id alone
        # cannot tell a fork from a spawn; source/thread_source can.
        assert meta["forked_from_id"] == e["spawner"], meta
        # session_id names the ROOT of the tree, not the direct parent: at
        # depth 2 the two differ. Never use it as either id.
        assert meta["session_id"] == root_of(child), (e, meta)
        # The rollout embeds its spawner's session_meta as its second record
        # (inherited history). Only the first session_meta describes the file.
        assert [m["id"] for m in metas] == [child, e["spawner"]], metas

    # Turn grouping mirrors Claude's promptId: a spawn's root_turn_id names the
    # ROOT turn, so a grandchild shares it with the child that spawned it.
    turns: dict[str, set[str]] = {}
    for e in edges:
        if root_of(e["child"]) == CODEX["roots"][0]:
            turns.setdefault(e["spawn_root_turn_id"], set()).add(e["task_name"])
    assert sorted(sorted(g) for g in turns.values()) == [
        ["alpha", "beta"],
        ["delta", "gamma"],
    ], turns

    forked = set()
    for fork in CODEX["forks"]:
        records = codex_rollout(fork["session"])
        metas = [r["payload"] for r in records if r["type"] == "session_meta"]
        assert len(metas) == 1, metas
        meta = metas[0]
        forked.add(fork["session"])
        assert meta["forked_from_id"] == fork["forked_from"], meta
        assert meta["history_base"]["thread_id"] == fork["forked_from"], meta
        # A fork -- even of a subagent -- inherits no subagent identity...
        assert meta["source"] == "exec" and meta["thread_source"] == "user", meta
        assert "parent_thread_id" not in meta and "agent_path" not in meta, meta
        # ...and, unlike an OpenCode fork, copies none of its source's spawn
        # records: it points at the source's history instead of repeating it.
        assert not any(
            (r["payload"].get("item") or {}).get("type") == "SubAgentActivity"
            for r in records
            if isinstance(r.get("payload"), dict)
        ), fork
    assert not forked & set(spawner_of), forked

    rows = {r["id"]: r for r in CODEX_INDEX["threads"]}
    with_forked_from = {
        tid for tid in rows if "forked_from_id" in codex_rollout(tid)[0]["payload"]
    }
    assert with_forked_from == forked | set(spawner_of), with_forked_from
    assert {
        r["child_thread_id"]: r["parent_thread_id"]
        for r in CODEX_INDEX["thread_spawn_edges"]
    } == spawner_of

    # An interrupted child is visible only as a started event with no matching
    # completion. The index edge status does not record it: every edge here is
    # still "open", finished or killed.
    killed = CODEX["interrupted"]
    assert not by_child[killed["child"]]["completed"]
    assert all(e["completed"] for e in edges if e["child"] != killed["child"])
    for tid in (killed["session"], killed["child"]):
        aborted = [
            r["payload"]["reason"]
            for r in codex_rollout(tid)
            if r["type"] == "event_msg" and r["payload"]["type"] == "turn_aborted"
        ]
        assert aborted == ["interrupted"], (tid, aborted)
    statuses = {r["status"] for r in CODEX_INDEX["thread_spawn_edges"]}
    assert statuses == {"open"}, statuses


# ------------------------------------------------------- session-browser view


def check_codex_graph() -> None:
    """The Codex graph through both discovery paths: index and file scan."""
    spawner_of = {e["child"]: e["spawner"] for e in CODEX["edges"]}
    expected_ids = {f"codex:{r['id']}" for r in CODEX_INDEX["threads"]}
    for use_index in (True, False):
        with tempfile.TemporaryDirectory() as tmp:
            home = build_home(Path(tmp), codex_index=use_index)
            listed = {s["id"]: s for s in list_sessions(home, "codex", quiet=use_index)}
        where = "index" if use_index else "file scan"
        assert set(listed) == expected_ids, (where, sorted(set(listed) ^ expected_ids))
        for sid, s in listed.items():
            thread_id = sid.removeprefix("codex:")
            if thread_id in spawner_of:
                assert s["parent_id"] == spawner_of[thread_id], (where, s)
                assert s["subagent_kind"] == "thread_spawn", (where, s)
            else:
                # Roots and forks alike: a fork is not a child.
                assert s["parent_id"] == "" and s["subagent_kind"] == "", (where, s)


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
    check_codex_raw_facts()
    check_codex_graph()
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
        "PASS: OpenCode and Codex graphs match spawn-time ground truth; "
        "Claude subagents are still invisible to discovery"
    )


def check_candidate() -> None:
    check_claude_raw_facts()
    check_opencode_raw_facts()
    check_codex_raw_facts()
    check_codex_graph()
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
    print("PASS: Claude, OpenCode and Codex graphs match spawn-time ground truth")


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
