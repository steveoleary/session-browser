"""Which sessions an agent process still has open, right now.

Claude Code and Grok say so themselves: each running Claude Code keeps
``~/.claude/sessions/<pid>.json`` naming its session, and Grok lists the
sessions it has open in ``~/.grok/active_sessions.json`` (discovery reads
both). The other agents keep no such record, so the answer comes from the
running processes themselves:

- a session id in a process's arguments -- ``codex resume <id>``,
  ``pi --session <id>``, ``opencode -s <id>`` -- names the session it runs;
- a transcript a process holds open is the one it is writing. Codex keeps
  its rollout open for as long as the session runs; Pi appends by path and
  OpenCode writes one shared database, so for those two only the arguments
  can tell.

Three answers, as everywhere on a Session: True when a running process is
shown to hold the session, False when no process of that agent is running at
all, and None when one is running but nothing ties it to this session -- a
Pi started without ``--session`` might be any of the sessions in its
directory, and saying False would be a guess.

Cheap and quiet by construction. The process table is read through the
kernel's own calls (``proc_listallpids`` and ``sysctl`` on macOS, ``/proc``
on Linux): about 2 ms for 840 processes here, 2026-11-24, with no
subprocess. ``lsof`` runs only when a Codex process exists, for those pids
alone (13 ms). Any failure marks nothing: a session is never called live, or
not live, on the strength of a probe that did not work.
"""

from __future__ import annotations

import logging
import os
import subprocess
import sys
from collections.abc import Iterable
from dataclasses import dataclass

from .discovery import Session, _claude_live_session_ids, _grok_live_session_ids

log = logging.getLogger(__name__)

# The agents found by their processes, by the program name each runs as.
# Pi is a node script (``node /opt/homebrew/bin/pi``); the rest are native
# binaries. Claude Code and Grok are not here: each keeps a registry of the
# sessions open right now, which discovery reads and which answers for them.
_AGENTS = frozenset({"codex", "pi", "opencode"})
_REGISTRIES = {"claude": _claude_live_session_ids, "grok": _grok_live_session_ids}
_INTERPRETERS = frozenset({"node", "bun"})
# Agents whose process keeps its own transcript open while it runs.
_HOLDS_TRANSCRIPT_OPEN = frozenset({"codex"})


@dataclass(frozen=True)
class AgentProcess:
    pid: int
    agent: str
    argv: tuple[str, ...]


def _program(argv: list[str]) -> str:
    """The agent *argv* runs, or "": its first argument, or its second when
    the first is an interpreter."""
    if not argv:
        return ""
    name = os.path.basename(argv[0])
    if name in _INTERPRETERS and len(argv) > 1:
        name = os.path.basename(argv[1])
    return name if name in _AGENTS else ""


def agent_processes() -> list[AgentProcess] | None:
    """Every running agent process, or None when the process table cannot be
    read on this machine."""
    try:
        if sys.platform == "darwin":
            table = _darwin_processes()
        elif sys.platform.startswith("linux"):
            table = _linux_processes()
        else:
            return None
    except Exception as exc:
        log.debug("could not read the process table: %s", exc)
        return None
    found = []
    for pid, argv in table:
        agent = _program(argv)
        if agent:
            found.append(AgentProcess(pid, agent, tuple(argv)))
    return found


# Names worth reading the arguments of: the agents themselves, and the
# interpreters one of them runs under. Everything else is skipped on its
# name alone, which is what keeps a scan of every process at a few ms.
_CANDIDATE_NAMES = _AGENTS | _INTERPRETERS


def _darwin_processes() -> list[tuple[int, list[str]]]:
    # Imported here, not at the top: ctypes is 3 ms of import, and only a
    # command that asks about live sessions on macOS should pay it.
    import ctypes

    libc = ctypes.CDLL(None, use_errno=True)
    count = libc.proc_listallpids(None, 0)
    if count <= 0:
        raise OSError("proc_listallpids reported no processes")
    pids = (ctypes.c_int * (count + 64))()
    count = libc.proc_listallpids(pids, ctypes.sizeof(pids))
    name = ctypes.create_string_buffer(64)
    out = []
    for i in range(max(count, 0)):
        pid = pids[i]
        if pid <= 0 or libc.proc_name(pid, name, len(name)) <= 0:
            continue
        if name.value.decode(errors="replace") not in _CANDIDATE_NAMES:
            continue
        argv = _darwin_argv(ctypes, libc, pid)
        if argv:
            out.append((pid, argv))
    return out


_CTL_KERN, _KERN_PROCARGS2 = 1, 49


def _darwin_argv(ctypes, libc, pid: int) -> list[str]:
    """A process's arguments, from ``sysctl kern.procargs2``: the argument
    count, the executable's path, NUL padding, then the arguments."""
    import struct

    mib = (ctypes.c_int * 3)(_CTL_KERN, _KERN_PROCARGS2, pid)
    size = ctypes.c_size_t(0)
    if libc.sysctl(mib, 3, None, ctypes.byref(size), None, 0) != 0:
        return []
    buf = ctypes.create_string_buffer(size.value)
    if libc.sysctl(mib, 3, buf, ctypes.byref(size), None, 0) != 0:
        return []
    raw = buf.raw[: size.value]
    if len(raw) < 4:
        return []
    (argc,) = struct.unpack_from("i", raw)
    rest = raw[4:]
    rest = rest[rest.find(b"\0") :].lstrip(b"\0")
    return [a.decode(errors="replace") for a in rest.split(b"\0")[:argc]]


def _linux_processes() -> list[tuple[int, list[str]]]:
    out = []
    for entry in os.scandir("/proc"):
        if not entry.name.isdigit():
            continue
        try:
            with open(f"/proc/{entry.name}/comm") as fh:
                if fh.read().strip() not in _CANDIDATE_NAMES:
                    continue
            with open(f"/proc/{entry.name}/cmdline", "rb") as fh:
                raw = fh.read()
        except OSError:
            continue  # gone, or not ours to read
        argv = [a.decode(errors="replace") for a in raw.split(b"\0") if a]
        if argv:
            out.append((int(entry.name), argv))
    return out


def _open_transcripts(pids: Iterable[int]) -> set[str] | None:
    """Real paths of the transcripts *pids* hold open, or None if lsof could
    not say."""
    wanted = ",".join(str(p) for p in pids)
    if not wanted:
        return set()
    try:
        result = subprocess.run(
            ["lsof", "-n", "-P", "-Fn", "-p", wanted],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        log.debug("lsof failed: %s", exc)
        return None
    # lsof exits 1 when one of the pids has gone since the scan, and still
    # lists the others, so the output is what decides.
    if result.returncode not in (0, 1):
        return None
    return {
        os.path.realpath(line[1:])
        for line in result.stdout.splitlines()
        if line.startswith("n") and line.endswith((".jsonl", ".jsonl.zst"))
    }


def fill_live(sessions: Iterable[Session], *, refresh: bool = False) -> None:
    """Say, for each session, whether a running agent process holds it.

    Claude and Grok sessions keep what discovery read from their registries,
    unless *refresh* asks for them again -- what a guard about to act
    on the answer wants. Everything else is asked of the process table, once.
    """
    sessions = list(sessions)
    if refresh:
        for provider, read in _REGISTRIES.items():
            held = [
                s for s in sessions if s.provider == provider and not s.subagent_kind
            ]
            if not held:
                continue
            ids = read()
            if ids is None:
                continue  # a registry that cannot be read leaves the answer
            for s in held:
                s.live = s.id in ids
    others = [s for s in sessions if s.provider in _AGENTS]
    if not others:
        return
    procs = agent_processes()
    if procs is None:
        return
    by_agent: dict[str, list[AgentProcess]] = {}
    for proc in procs:
        by_agent.setdefault(proc.agent, []).append(proc)
    holding = [p.pid for p in procs if p.agent in _HOLDS_TRANSCRIPT_OPEN]
    open_paths = _open_transcripts(holding) if holding else set()
    for s in others:
        running = by_agent.get(s.provider)
        if not running:
            s.live = False
            continue
        # By id, or by the transcript's path, which Pi's --session also takes.
        named = any(
            s.id in proc.argv or (s.content_path and s.content_path in proc.argv)
            for proc in running
        )
        held = (
            open_paths is not None
            and s.provider in _HOLDS_TRANSCRIPT_OPEN
            and _realpath(s.content_path) in open_paths
        )
        s.live = True if named or held else None


def _realpath(path: str) -> str:
    return os.path.realpath(path) if path else ""


def is_live(session: Session) -> bool:
    """Whether a running process holds *session*, asked afresh: for a delete
    or a resume about to act on the answer."""
    fill_live([session], refresh=True)
    return session.live is True


def describe(session: Session) -> str:
    """A few words on what holds *session* open, for a warning."""
    return f"open in a {session.provider} process"
