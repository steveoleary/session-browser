"""Open a session's folder in the Herdr app, whose terminals herdr-core runs.

This is the successor to ``herdr.py``. That module drives the original
``herdr`` — one CLI and one TUI over a server socket. Its replacement is split
in two, and a handoff needs both halves:

- **herdr-core** (CLI ``hc``) owns the terminals and keeps them alive. It is
  what can say which program a terminal is running, and in which folder.
- **the Herdr app** (CLI ``herdr-app``) is the window. It owns spaces, tabs and
  panes, and is the only thing that can create or show one.

``multiplexer.py`` decides which of the two herdrs a handoff goes to; both
modules present the same interface (``available``, ``in_*``,
``prepare_session`` and a plan with ``switch_commands``/``attach_commands``/
``label``/``reused``).

The layout mirrors ``herdr.py``: a folder maps to one space named after its
basename, and each conversation gets its own tab in it, running the provider's
resume command. The app runs a tab's agent inside the user's interactive shell,
which stays open after the agent exits — the same "left at a prompt" outcome
``tmux.py`` gets from send-keys.

Finding a conversation that is already open is where the two differ. The old
herdr reported each pane's agent session id; herdr-core does not. It does
report each terminal's *foreground process* — argv, folder and pid — which
answers the question two ways:

- a terminal opened by a resume command carries the session id in its argv
  (``claude --resume <id>``, ``codex resume <id>``), whoever started it;
- a Claude Code process records its live session id in
  ``~/.claude/sessions/<pid>.json``, so a Claude terminal is recognised even
  when it was started bare or has since forked to a new id.

As in the other two modules, every id in the conversation's fork lineage
(``transcript.lineage_ids``) is compared, because Claude's ``--resume``
continues under a fresh id the user never picked.
"""

from __future__ import annotations

import json
import os
import shlex
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

from .resume import resume_command
from .transcript import lineage_ids

# Availability is probed on the UI thread when the user presses the handoff
# key, so a wedged socket must not hang the browser. The same bound applies to
# every lookup below: each is a local socket round trip measured in
# milliseconds, and one that is not answering will not start to.
_PROBE_TIMEOUT = 2.0

# Labels herdr-core carries on every terminal the app shows, naming its space
# and tab. A terminal without them is a background worker nobody can see, so
# it is never a handoff destination.
_SPACE_LABEL = "herdr.space"
_TAB_LABEL = "herdr.tab"


class HerdrCoreError(RuntimeError):
    """Raised when a session cannot be opened in the Herdr app."""


def space_name_for_path(path: str) -> str:
    """Space name for a folder: its basename, as ``herdr.py`` names workspaces."""
    base = os.path.basename(os.path.normpath(path)) if path else ""
    return base or "session"


@dataclass(frozen=True)
class CoreTerminal:
    """One live terminal as ``hc get`` reports it, reduced to what matching needs.

    ``argv``, ``cwd`` and ``pid`` describe the *foreground* process — the agent
    when one is running, otherwise the shell. ``space`` and ``tab`` are empty
    for a terminal the app does not show.
    """

    address: str
    argv: tuple[str, ...]
    cwd: str
    pid: int | None
    space: str
    tab: str


@dataclass(frozen=True)
class AppSpace:
    """One space as ``herdr-app ls`` reports it.

    The app reports a ``folder`` for each space too, but it is only the folder
    the space's first terminal started in, inherited from whatever was focused
    when the space was made, so it says nothing reliable about what the space
    is for and is not read.
    """

    space_id: str
    name: str


@dataclass
class HerdrCorePlan:
    """Where a session ended up in the Herdr app, and how to hand off to it."""

    label: str
    cwd: str
    resume: str
    session_id: str
    space: str
    tab: str
    terminal: str
    reused: bool

    def switch_commands(self) -> list[list[str]]:
        """Show the conversation's terminal in the running app window.

        Space, then tab, then the pane showing the terminal, each by id: a
        reused terminal may share its tab with others, and naming every level
        leaves nothing to whichever space or tab happened to be selected.
        """
        do = ["herdr-app", "do"]
        commands = []
        if self.space:
            commands.append([*do, "space.select", "--id", self.space])
        if self.tab:
            commands.append([*do, "tab.select", "--id", self.tab])
        commands.append([*do, "pane.focus", "--terminal", self.terminal])
        return commands

    def attach_commands(self) -> list[list[str]]:
        """Show it in the window, then attach this terminal to it as well.

        From a plain terminal the window may be behind something else, so the
        conversation is also put on the screen the user is actually looking at:
        ``hc attach`` takes this terminal over until Ctrl+\\ detaches, at which
        point the browser resumes — the same round trip as attaching the old
        herdr's TUI.
        """
        return [*self.switch_commands(), ["hc", "attach", self.terminal]]


def available() -> bool:
    """True when ``hc`` and ``herdr-app`` are installed and the window is up.

    The window is the one to ask: a handoff creates and shows tabs in it, and
    a running window implies a running core (it is that core's client). Asking
    the core first would not do — on some platforms an ``hc`` command starts a
    core when none is running, and pressing a key in a browser must not.
    """
    if shutil.which("hc") is None or shutil.which("herdr-app") is None:
        return False
    try:
        result = subprocess.run(
            ["herdr-app", "status"],
            capture_output=True,
            text=True,
            timeout=_PROBE_TIMEOUT,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return False
    if result.returncode != 0:
        return False
    try:
        status = json.loads(result.stdout)
    except ValueError:
        return False
    return isinstance(status, dict) and "error" not in status


def in_herdr_core() -> bool:
    """True when running in a herdr-core terminal, so the window can be
    switched rather than attached. herdr-core gives every terminal it runs its
    own address in ``HC_TERMINAL_ID``."""
    return bool(os.environ.get("HC_TERMINAL_ID"))


def _json(command: list[str]) -> object:
    """Run a CLI that answers with JSON on stdout, raising on any failure.

    Both CLIs report a failure as ``{"error": ...}`` — ``hc`` on stderr with a
    non-zero exit, ``herdr-app do`` on stdout with exit 1 — so either stream
    is searched for the message.
    """
    try:
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=_PROBE_TIMEOUT,
            check=False,
        )
    except subprocess.TimeoutExpired:
        raise HerdrCoreError(f"{command[0]} did not answer") from None
    except OSError as exc:
        raise HerdrCoreError(f"{command[0]} failed: {exc}") from None
    if result.returncode != 0:
        raise HerdrCoreError(
            f"{command[0]} failed: {_error_detail(result) or ' '.join(command[1:])}"
        )
    try:
        payload = json.loads(result.stdout)
    except ValueError:
        raise HerdrCoreError(
            f"{command[0]} gave no JSON for '{' '.join(command[1:])}'"
        ) from None
    if isinstance(payload, dict) and payload.get("error"):
        raise HerdrCoreError(f"{command[0]} failed: {_message(payload['error'])}")
    return payload


def _error_detail(result: subprocess.CompletedProcess) -> str:
    """The message from a failed command, unwrapped if it was JSON."""
    for raw in (result.stderr, result.stdout):
        raw = (raw or "").strip()
        if not raw:
            continue
        try:
            payload = json.loads(raw)
        except ValueError:
            return raw
        if isinstance(payload, dict) and payload.get("error"):
            return _message(payload["error"])
        return raw
    return ""


def _message(error: object) -> str:
    """``hc`` errors are ``{kind, message, hint}``; the app's are a string."""
    if isinstance(error, dict):
        text = error.get("message") or error.get("kind")
        return str(text) if text else json.dumps(error)
    return str(error)


def list_terminals() -> list[CoreTerminal]:
    """Every live local terminal herdr-core runs, with its foreground process.

    ``hc ls`` names the terminals but leaves out the foreground process and
    the labels, so each is then read with ``hc get``. A handful of terminals is
    the normal case and each read is a local round trip; a terminal that ends
    between the two calls is simply skipped.
    """
    listing = _json(["hc", "ls", "--json"])
    rows = listing.get("terminals") if isinstance(listing, dict) else None
    if not isinstance(rows, list):
        raise HerdrCoreError("hc did not list any terminals")
    terminals = []
    for row in rows:
        address = row.get("address") if isinstance(row, dict) else None
        # Another machine's terminal (``box/t7``) runs in that machine's
        # folders, which can never be this session's.
        if not isinstance(address, str) or not address or "/" in address:
            continue
        try:
            detail = _json(["hc", "get", address, "--json"])
        except HerdrCoreError:
            continue
        if isinstance(detail, dict):
            terminals.append(_as_terminal(address, detail))
    return terminals


def _as_terminal(address: str, raw: dict) -> CoreTerminal:
    foreground = raw.get("foreground")
    foreground = foreground if isinstance(foreground, dict) else {}
    argv = foreground.get("argv")
    labels = raw.get("labels")
    labels = labels if isinstance(labels, dict) else {}
    pid = foreground.get("pid")
    return CoreTerminal(
        address=address,
        argv=tuple(str(a) for a in argv) if isinstance(argv, list) else (),
        cwd=str(foreground.get("cwd") or raw.get("cwd") or ""),
        pid=pid if isinstance(pid, int) and not isinstance(pid, bool) else None,
        space=str(labels.get(_SPACE_LABEL) or ""),
        tab=str(labels.get(_TAB_LABEL) or ""),
    )


def list_spaces() -> list[AppSpace]:
    """Every local space the app window holds."""
    listing = _json(["herdr-app", "ls"])
    rows = listing.get("spaces") if isinstance(listing, dict) else None
    if not isinstance(rows, list):
        raise HerdrCoreError("herdr-app did not list any spaces")
    spaces = [
        AppSpace(space_id=str(row.get("id") or ""), name=str(row.get("name") or ""))
        for row in rows
        if isinstance(row, dict) and not row.get("machine")
    ]
    return [space for space in spaces if space.space_id]


def claude_live_session(pid: int | None, home: Path | None = None) -> str:
    """The session a running Claude Code process is on right now, if it says.

    Claude Code keeps ``~/.claude/sessions/<pid>.json`` for each live process,
    naming its current ``sessionId`` — the id *after* any resume fork, which is
    exactly the one the browser lists as the newest row.
    """
    if pid is None:
        return ""
    path = (home or Path.home()) / ".claude" / "sessions" / f"{pid}.json"
    try:
        record = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return ""
    if not isinstance(record, dict) or record.get("pid") not in (pid, None):
        return ""
    value = record.get("sessionId")
    return value if isinstance(value, str) else ""


def _runs_provider(argv: tuple[str, ...], provider: str) -> bool:
    """True when the foreground process is ``provider``'s CLI.

    An id appearing in argv is only evidence when the program is the agent
    itself; ``session-browser get <id>`` or ``less <id>.jsonl`` carries the
    same id and is not the conversation. The program is the first argument,
    or the second when the first is an interpreter (``node .../codex``).
    """
    return any(os.path.basename(arg) == provider for arg in argv[:2])


def find_terminal(
    terminals: list[CoreTerminal],
    provider: str,
    session_ids: set[str],
    *,
    home: Path | None = None,
) -> CoreTerminal | None:
    """The shown terminal already running this conversation, if any."""
    wanted = {i for i in session_ids if i}
    if not wanted:
        return None
    for terminal in terminals:
        if not terminal.space:
            continue
        if _runs_provider(terminal.argv, provider) and wanted.intersection(
            terminal.argv
        ):
            return terminal
        if provider == "claude" and claude_live_session(terminal.pid, home) in wanted:
            return terminal
    return None


def space_for_cwd(
    terminals: list[CoreTerminal], spaces: list[AppSpace], cwd: str
) -> AppSpace | None:
    """The space already holding this folder: one with a shown terminal in it.

    A terminal's folder is where the user is actually working, which is why it
    decides rather than the space's own reported ``folder`` (see ``AppSpace``).
    Paths are normalised because a provider-recorded ``cwd`` may carry a
    trailing slash.
    """
    target = os.path.normpath(cwd)
    by_id = {space.space_id: space for space in spaces}
    for terminal in terminals:
        if (
            terminal.space in by_id
            and terminal.cwd
            and os.path.normpath(terminal.cwd) == target
        ):
            return by_id[terminal.space]
    return None


def prepare_session(
    provider: str, session_id: str, cwd: str, content_path: str = ""
) -> HerdrCorePlan:
    """Find or create the session's tab in the Herdr app and return the plan.

    A terminal already running this conversation is reused as is. Otherwise a
    tab running the resume command is added to the folder's space, creating
    the space if the folder has none. The app shows a new tab as soon as it is
    made, so unlike ``herdr.py`` nothing here stays hidden until the caller
    switches; the plan's commands still name it, so the handoff lands the same
    way whether the tab was made or found.
    """
    if not cwd:
        raise HerdrCoreError("session has no folder (cwd) to open")
    if shutil.which("hc") is None or shutil.which("herdr-app") is None:
        raise HerdrCoreError("hc and herdr-app are not both on PATH")

    resume = resume_command(provider, session_id, None, content_path=content_path)
    if resume.startswith("#"):
        raise HerdrCoreError(resume.lstrip("# ").strip() or "no resume command")

    terminals = list_terminals()
    spaces = list_spaces()
    existing = find_terminal(
        terminals, provider, lineage_ids(provider, session_id, content_path)
    )
    if existing:
        space = next((s for s in spaces if s.space_id == existing.space), None)
        return HerdrCorePlan(
            label=space.name if space and space.name else space_name_for_path(cwd),
            cwd=cwd,
            resume=resume,
            session_id=session_id,
            space=existing.space,
            tab=existing.tab,
            terminal=existing.address,
            reused=True,
        )

    argv = json.dumps(shlex.split(resume))
    space = space_for_cwd(terminals, spaces, cwd)
    if space:
        label = space.name or space_name_for_path(cwd)
        created = _json(
            [
                *("herdr-app", "do", "tab.new"),
                *("--space", space.space_id, "--folder", cwd, "--argv", argv),
            ]
        )
    else:
        label = space_name_for_path(cwd)
        created = _json(
            [
                *("herdr-app", "do", "space.new"),
                *("--name", label, "--folder", cwd, "--argv", argv),
            ]
        )
    made = created if isinstance(created, dict) else {}
    terminal = str(made.get("terminal") or "")
    if not terminal:
        raise HerdrCoreError("herdr-app did not report the new tab's terminal")
    return HerdrCorePlan(
        label=label,
        cwd=cwd,
        resume=resume,
        session_id=session_id,
        space=str(made.get("space") or (space.space_id if space else "")),
        tab=str(made.get("tab") or ""),
        terminal=terminal,
        reused=False,
    )
