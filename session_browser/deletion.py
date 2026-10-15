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

OpenCode and Pi are refused: neither was installed where this was built, so
no deletion of theirs has been proven.

A subagent is never deleted on its own -- its parent's transcript still holds
the spawning call -- and a Claude session that is still running is refused,
since its process would go on appending to a transcript that no longer exists.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import uuid
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

from .discovery import Session, claude_root_session


class DeleteRefused(Exception):
    """This session cannot be deleted; the message says why, for the user."""


class DeleteFailed(Exception):
    """The deletion was attempted and did not complete."""


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
    # Human-readable lines for the confirmation, one per thing removed.
    removes: tuple[str, ...]


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
    raise DeleteRefused(f"Deleting {session.provider} sessions is not supported yet")


def delete_session(plan: DeletePlan) -> str:
    """Carry out *plan*. Returns a status line, or raises DeleteFailed."""
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
    removes = [f"transcript  {transcript}"]
    if aux[0] in paths:
        removes.append(f"session dir {aux[0]}  (subagents, tool output)")
    removes += [f"state       {p}" for p in aux[1:] if p in paths]
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
    removes = (
        f"rollout     {session.content_path}",
        "index       Codex's own thread and history rows",
    )
    command = (codex, "delete", "--force", session.id)
    return DeletePlan(session, (), command, subagents, removes)


def _run_provider_delete(plan: DeletePlan) -> str:
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
        raise DeleteFailed(f"codex delete did not run: {exc}") from exc
    if done.returncode != 0:
        detail = (done.stderr or done.stdout).strip().splitlines()
        raise DeleteFailed(
            f"codex delete exited {done.returncode}"
            + (f": {detail[-1]}" if detail else "")
        )
    # Trust the result, not the exit code alone: a rollout still on disk is a
    # session that will be rediscovered on the next refresh.
    rollout = plan.session.content_path
    if rollout and Path(rollout).exists():
        raise DeleteFailed(f"codex delete reported success but {rollout} remains")
    return f"Deleted codex session {plan.session.id}"


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
