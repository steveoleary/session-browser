"""Permanently delete a session from its provider's own store.

Everywhere else this package only reads other tools' files. Deleting is the one
place it writes to them, so each provider gets the least invasive route that
was proven against the real tool:

- **Codex** ships ``codex delete --force <uuid>``. It removes the ``threads``
  row, the paginated history rows and the rollout file, and deleting a parent
  takes its spawned subagents with it -- verified on codex-cli 0.162.1 with a
  throwaway session that spawned one child. So this module hands Codex the
  whole job and never opens Codex's SQLite for writing.
- **Claude Code** has no per-session delete (``claude rm`` is for background
  sessions, ``claude purge`` is per project), so the files are removed here.
  A session is its ``<id>.jsonl`` plus the ``<id>/`` directory beside it
  (subagents, workflow agents, persisted tool output), and two per-session
  directories under ``~/.claude``: ``session-env/<id>`` and, if the session
  edited files, ``file-history/<id>``. Two shared files also mention the id
  and are left alone: ``history.jsonl``, the prompt history every running
  ``claude`` appends to, and ``backups/``, which rotates out by itself and
  which ``claude purge`` leaves too. Claude Code ages transcripts out on its
  own retention timer, so a missing transcript is a state it already handles.
- **OpenCode** 2.x ships ``opencode session delete <id>``, which removes the
  session's row, its messages, its event sequence and instruction state, and
  every child session with it -- verified on OpenCode 2.2.4 with a throwaway
  session that spawned one subagent; afterwards the id survived only in
  OpenCode's own shared log. So it gets the whole job, as Codex does. A
  database still in the 1.x schema is refused: no 1.x delete was proven.
- **Pi** has no delete command, so the files are removed here. A session is
  one ``<stamp>_<id>.jsonl`` under ``~/.pi/agent/sessions/<encoded-cwd>/``;
  nothing in Pi itself references the id anywhere else (verified on Pi 1.4.0
  with throwaway sessions). Extensions can keep per-session state named for
  the id -- pi-blackhole writes ``<id>-pending.json`` cursors -- so a file
  directly inside an ``~/.pi/agent/<extension>/`` directory whose name starts
  with the id goes too, and is listed. Session ids are UUIDv7s, so a name
  that starts with one belongs to that session and nothing else. Pi appends
  to its file by path after the first write, so a running session would
  recreate a headerless fragment of itself: a delete is refused while any
  ``pi`` process is running in the session's directory.

A subagent is never deleted on its own -- its parent's transcript still holds
the spawning call -- and a Claude or Pi session that is still running is
refused, since its process would go on appending to a transcript that no
longer exists.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import sqlite3
import subprocess
import uuid
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import NamedTuple

from . import compressed, entry_cache
from .discovery import (
    OPENCODE_V2,
    Session,
    _opencode_db_path,
    _readonly_uri,
    claude_root_session,
    opencode_execute,
)


class DeleteRefused(Exception):
    """This session cannot be deleted; the message says why, for the user."""


class DeleteFailed(Exception):
    """The deletion was attempted and did not complete."""


class Removal(NamedTuple):
    """One thing a delete removes, for the confirmation: what kind of thing
    it is, the path or name that identifies it, and an optional aside."""

    label: str
    value: str
    note: str = ""


@dataclass(frozen=True)
class DeletePlan:
    """What deleting one session will remove, worked out before anything is."""

    session: Session
    # Paths removed by this module, in removal order (transcript last), or
    # empty when the provider's own command does the removing.
    paths: tuple[Path, ...]
    # The provider's own delete command, when it has one.
    command: tuple[str, ...]
    # Sessions in the list that go with it: subagents at any depth.
    subagents: int
    # What the confirmation lists, one per thing removed.
    removes: tuple[Removal, ...]


def plan_delete(session: Session, sessions: Iterable[Session]) -> DeletePlan:
    """Work out how *session* would be deleted, or raise DeleteRefused.

    *sessions* is everything discovered, used only to count the subagents
    that will go with it.
    """
    if session.parent_id or claude_root_session(session):
        raise DeleteRefused(
            "This is a subagent; delete the session that spawned it instead"
        )
    subagents = _count_descendants(session, sessions)
    if session.provider == "claude":
        return _plan_claude(session, subagents)
    if session.provider == "codex":
        return _plan_codex(session, subagents)
    if session.provider == "opencode":
        return _plan_opencode(session, subagents)
    if session.provider == "pi":
        return _plan_pi(session, subagents)
    raise DeleteRefused(f"Deleting {session.provider} sessions is not supported yet")


def delete_session(plan: DeletePlan) -> str:
    """Carry out *plan*. Returns a status line, or raises DeleteFailed."""
    try:
        if plan.command:
            return _run_provider_delete(plan)
        for path in plan.paths:
            try:
                if path.is_dir() and not path.is_symlink():
                    shutil.rmtree(path)
                else:
                    path.unlink(missing_ok=True)
            except OSError as exc:
                raise DeleteFailed(f"Could not remove {path}: {exc}") from exc
        return f"Deleted {plan.session.provider} session {plan.session.id}"
    finally:
        # The parsed text this package cached from the session, and from the
        # subagents that went with it, must not outlive them. Swept even when
        # the delete failed part-way: whatever did go is then gone here too.
        entry_cache.sweep()


# ---------------------------------------------------------------------------
# Claude Code
# ---------------------------------------------------------------------------


def _plan_claude(session: Session, subagents: int) -> DeletePlan:
    claude_home = Path.home() / ".claude"
    transcript = Path(session.content_path)
    # The path comes from discovery, but this is about to remove things, so
    # it is checked against the one shape a top-level transcript has rather
    # than trusted: <home>/.claude/projects/<project>/<id>.jsonl.
    if (
        transcript.name != f"{session.id}.jsonl"
        or transcript.parent.parent != claude_home / "projects"
    ):
        raise DeleteRefused(f"Unexpected transcript location: {transcript}")
    if _claude_session_running(claude_home, session.id):
        raise DeleteRefused("This session is still running; exit it first")
    aux = [
        transcript.parent / session.id,
        claude_home / "session-env" / session.id,
        claude_home / "file-history" / session.id,
    ]
    # Auxiliary state first and the transcript last: a failure part-way
    # leaves the session visible in the list, where it can be deleted again,
    # rather than an invisible remainder nothing will ever clean up.
    paths = tuple(p for p in aux if p.exists()) + (transcript,)
    removes = [Removal("transcript", str(transcript))]
    if aux[0] in paths:
        removes.append(Removal("session dir", str(aux[0]), "subagents, tool output"))
    removes += [Removal("state", str(p)) for p in aux[1:] if p in paths]
    return DeletePlan(session, paths, (), subagents, tuple(removes))


def _claude_session_running(claude_home: Path, session_id: str) -> bool:
    """Whether a live ``claude`` process has this session open.

    Each running Claude Code writes ``sessions/<pid>.json`` naming its
    session; a file whose process is gone is a leftover, not a live session.
    """
    for marker in (claude_home / "sessions").glob("*.json"):
        try:
            info = json.loads(marker.read_text())
        except (OSError, ValueError):
            continue
        if not isinstance(info, dict) or info.get("sessionId") != session_id:
            continue
        pid = info.get("pid")
        if isinstance(pid, int) and _pid_alive(pid):
            return True
    return False


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


# ---------------------------------------------------------------------------
# Codex
# ---------------------------------------------------------------------------


def _plan_codex(session: Session, subagents: int) -> DeletePlan:
    codex = shutil.which("codex")
    if codex is None:
        raise DeleteRefused("Deleting a Codex session needs the codex CLI on PATH")
    # `codex delete` also accepts a session *name*, and --force insists on a
    # UUID; checking here keeps an id that is not one from ever being read as
    # a name.
    try:
        uuid.UUID(session.id)
    except ValueError:
        raise DeleteRefused(f"Not a Codex thread id: {session.id}") from None
    # A rewound thread is several rollouts, and codex delete removes them all
    # (checked on codex-cli 0.169.0), so the plan names each one.
    removes = (
        *(Removal("rollout", p) for p in (*session.segments, session.content_path)),
        Removal("index", "Codex's own thread and history rows"),
    )
    command = (codex, "delete", "--force", session.id)
    return DeletePlan(session, (), command, subagents, removes)


def _run_provider_delete(plan: DeletePlan) -> str:
    tool = f"{Path(plan.command[0]).name} {plan.command[1]}"
    try:
        done = subprocess.run(
            plan.command,
            stdin=subprocess.DEVNULL,
            capture_output=True,
            text=True,
            timeout=60,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise DeleteFailed(f"{tool} did not run: {exc}") from exc
    if done.returncode != 0:
        detail = (done.stderr or done.stdout).strip().splitlines()
        raise DeleteFailed(
            f"{tool} exited {done.returncode}" + (f": {detail[-1]}" if detail else "")
        )
    # Trust the result, not the exit code alone: whatever the provider left
    # behind is a session that will be rediscovered on the next refresh.
    provider = plan.session.provider
    if provider == "codex":
        _check_codex_gone(plan.session, tool)
    elif provider == "opencode":
        _check_opencode_gone(plan.session, tool)
    return f"Deleted {provider} session {plan.session.id}"


def _check_codex_gone(session: Session, tool: str) -> None:
    rollouts = (*session.segments, session.content_path)
    for rollout in rollouts:
        if rollout and Path(rollout).exists():
            raise DeleteFailed(f"{tool} reported success but {rollout} remains")
    for rollout in rollouts:
        if rollout.endswith(compressed.SUFFIX):
            # A compressed rollout leaves its decompressed copy behind in this
            # package's own cache, which nothing else would ever remove.
            compressed.discard(Path(rollout))


# ---------------------------------------------------------------------------
# OpenCode
# ---------------------------------------------------------------------------

# OpenCode's own validation, seen refusing anything else: 'Expected a string
# starting with "ses"'. Checked here too, so nothing that is not a session id
# ever reaches its argv.
_OPENCODE_ID = re.compile(r"ses_[A-Za-z0-9]+")

_OPENCODE_HAS_SESSION_SQL = (
    "SELECT count(*) FROM session WHERE id = ?",
    "SELECT count(*) FROM session_v2 WHERE id = ?",
)


def _opencode_session_rows(session_id: str) -> tuple[int, int]:
    """(rows holding *session_id*, schema) in OpenCode's database."""
    db = _opencode_db_path()
    if not db.is_file():
        return 0, OPENCODE_V2
    try:
        conn = sqlite3.connect(_readonly_uri(db), uri=True, timeout=3)
        try:
            cur, schema = opencode_execute(
                conn, _OPENCODE_HAS_SESSION_SQL, (session_id,)
            )
            return cur.fetchone()[0], schema
        finally:
            conn.close()
    except sqlite3.Error as exc:
        raise DeleteRefused(f"Could not read OpenCode's database: {exc}") from exc


def _plan_opencode(session: Session, subagents: int) -> DeletePlan:
    opencode = shutil.which("opencode")
    if opencode is None:
        raise DeleteRefused(
            "Deleting an OpenCode session needs the opencode CLI on PATH"
        )
    if not _OPENCODE_ID.fullmatch(session.id):
        raise DeleteRefused(f"Not an OpenCode session id: {session.id}")
    rows, schema = _opencode_session_rows(session.id)
    if schema != OPENCODE_V2:
        raise DeleteRefused(
            "This OpenCode database predates 2.x; deleting from it is not "
            "supported (only OpenCode 2.x's delete has been proven)"
        )
    if not rows:
        raise DeleteRefused(f"OpenCode has no session {session.id}")
    removes = (
        Removal(
            "session", "its row in OpenCode's database", "by OpenCode's own delete"
        ),
        Removal("history", "its messages and event history"),
    )
    command = (opencode, "session", "delete", session.id)
    return DeletePlan(session, (), command, subagents, removes)


def _check_opencode_gone(session: Session, tool: str) -> None:
    try:
        rows, _schema = _opencode_session_rows(session.id)
    except DeleteRefused as exc:
        raise DeleteFailed(
            f"{tool} ran, but the result could not be checked: {exc}"
        ) from exc
    if rows:
        raise DeleteFailed(f"{tool} reported success but the session is still there")


# ---------------------------------------------------------------------------
# Pi
# ---------------------------------------------------------------------------


def _plan_pi(session: Session, subagents: int) -> DeletePlan:
    agent = Path.home() / ".pi" / "agent"
    transcript = Path(session.content_path)
    # Checked against the one shape a Pi transcript has rather than trusted:
    # <home>/.pi/agent/sessions/<encoded-cwd>/<stamp>_<id>.jsonl, with an id
    # that really is a UUID so the extension sweep below matches nothing else.
    try:
        uuid.UUID(session.id)
    except ValueError:
        raise DeleteRefused(f"Not a Pi session id: {session.id}") from None
    if (
        not transcript.name.endswith(f"_{session.id}.jsonl")
        or transcript.parent.parent != agent / "sessions"
    ):
        raise DeleteRefused(f"Unexpected transcript location: {transcript}")
    if session.cwd and _pi_running_in(session.cwd):
        raise DeleteRefused("pi is running in this session's directory; exit it first")
    sidecars = sorted(_pi_extension_files(agent, session.id))
    # Extension state first and the transcript last, as for Claude: a
    # failure part-way leaves the session listed, where it can be retried.
    paths = (*sidecars, transcript)
    removes = [Removal("transcript", str(transcript))]
    removes += [Removal("extension", str(p)) for p in sidecars]
    return DeletePlan(session, paths, (), subagents, tuple(removes))


def _pi_extension_files(agent: Path, session_id: str) -> Iterable[Path]:
    """Files directly inside ~/.pi/agent/<dir>/ whose name starts with the
    session id. One level only, and never the sessions tree itself."""
    try:
        dirs = [d for d in agent.iterdir() if d.is_dir() and d.name != "sessions"]
    except OSError:
        return
    for d in dirs:
        try:
            for f in d.iterdir():
                if f.name.startswith(session_id) and f.is_file():
                    yield f
        except OSError:
            continue


def _pi_running_in(cwd: str) -> bool:
    """Whether a ``pi`` process has *cwd* as its working directory.

    Pi leaves no marker naming the session it has open, so this is the
    closest honest answer: any pi working in the session's directory might
    be that session. A machine without ps or lsof answers no, as the Claude
    check does when there is no marker to read.
    """
    try:
        ps = subprocess.run(
            ["ps", "-axo", "pid=,args="],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return False
    pids = []
    for line in ps.stdout.splitlines():
        pid, _, args = line.strip().partition(" ")
        argv = [Path(a).name for a in args.split()[:2]]
        # Pi runs as `node .../bin/pi ...` (its shebang is env node), or as
        # `pi` itself where the kernel reports the script. An argument named
        # pi to some other program (`vim pi`) is not a pi process.
        if argv[:1] == ["pi"] or (
            argv[:1] in (["node"], ["bun"]) and argv[1:] == ["pi"]
        ):
            pids.append(pid)
    if not pids:
        return False
    try:
        lsof = subprocess.run(
            ["lsof", "-a", "-d", "cwd", "-Fn", "-p", ",".join(pids)],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return False
    want = os.path.realpath(cwd)
    return any(
        line.startswith("n") and os.path.realpath(line[1:]) == want
        for line in lsof.stdout.splitlines()
    )


# ---------------------------------------------------------------------------


def _count_descendants(session: Session, sessions: Iterable[Session]) -> int:
    """Subagents of *session* at any depth.

    Claude children are found by path, since ``parent_id`` names a
    grandchild's spawning agent rather than the session; everything else
    by following ``parent_id``.
    """
    if session.provider == "claude":
        return sum(
            1
            for s in sessions
            if s.provider == "claude" and claude_root_session(s) == session.id
        )
    children: dict[str, list[str]] = {}
    for s in sessions:
        if s.provider == session.provider and s.parent_id:
            children.setdefault(s.parent_id, []).append(s.id)
    seen: set[str] = set()
    stack = list(children.get(session.id, ()))
    while stack:
        sid = stack.pop()
        if sid not in seen:
            seen.add(sid)
            stack.extend(children.get(sid, ()))
    return len(seen)
