"""Open a session's folder in herdr, the way ``tmux.py`` does for tmux.

Herdr organises terminals as workspaces → tabs → panes and drives them through
a server socket, reachable from the ``herdr`` CLI as JSON-returning commands.
A folder (a session's ``cwd``) maps to one workspace labelled after the
folder's basename — the same rule ``tmux.py`` applies to tmux session names —
and each conversation gets its own tab in it, whose root pane is handed the
provider's resume command.

Unlike tmux, herdr can tell us which agent session a pane is running: it
detects the agent and reports the id as ``agent_session.value``. That removes
the window-tagging ``tmux.py`` needs, and one ``herdr pane list`` answers both
questions a handoff has — is this conversation already open in a pane, and does
a workspace for this folder already exist. Matching is therefore not limited to
panes this browser opened; a pane the user started by hand is found and reused
too.

How far that reaches depends on herdr, not on us: it populates
``agent_session`` for the agents it has a detector for (observed for ``claude``,
with ``source: "herdr:claude"``) and leaves it null otherwise — opencode panes
report ``agent`` but no session. For those, ``find_pane`` cannot match and every
handoff opens a fresh tab, duplicating a conversation that is already on screen.
That is a gap in detection upstream, so it is not worked around here; when herdr
starts reporting the id, reuse begins working with no change to this module.

The fork problem from ``tmux.py`` still applies: Claude Code's ``--resume``
continues the conversation under a fresh id, so the pane herdr reports may be
sitting on an id the user never picked. ``lineage_ids`` recovers the earlier
ids from the transcript, and every one of them is compared against what herdr
reports.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from dataclasses import dataclass

from .resume import resume_command
from .transcript import lineage_ids

# Availability is probed on the UI thread when the user presses the handoff
# key, so a wedged socket must not hang the browser.
_PROBE_TIMEOUT = 2.0


class HerdrError(RuntimeError):
    """Raised when a session cannot be opened in herdr."""


def workspace_label_for_path(path: str) -> str:
    """Workspace label for a folder: its basename, as herdr labels its own.

    Herdr labels are free-form display text (it labels a git-worktree
    workspace with the checkout's basename), so unlike tmux session names
    there is nothing to sanitize away.
    """
    base = os.path.basename(os.path.normpath(path)) if path else ""
    return base or "session"


@dataclass(frozen=True)
class HerdrPane:
    """One pane as ``herdr pane list`` reports it.

    ``agent`` is the agent kind herdr detected in the pane and ``session_id``
    the session that agent is on; both are empty for a pane running an
    ordinary shell.
    """

    pane_id: str
    tab_id: str
    workspace_id: str
    cwd: str
    agent: str
    session_id: str


@dataclass
class HerdrPlan:
    """Where a session ended up in herdr, and how to hand off to it.

    ``reused`` records that ``pane`` was already running this conversation
    rather than being created for it — the same distinction ``TmuxPlan``
    draws with ``existing_window``.
    """

    label: str
    cwd: str
    resume: str
    session_id: str
    tab: str
    pane: str
    reused: bool

    def switch_commands(self) -> list[list[str]]:
        """Bring the conversation's pane to the front of a running herdr UI.

        The analogue of ``sesh connect --switch``: one herdr focus command
        switches workspace, tab and pane together. A reused pane is targeted
        through ``agent focus`` so that a pane sharing its tab with others is
        the one focused; a tab created here has no agent to target yet, and its
        root pane is the only pane in it, so the tab is focused instead.
        """
        if self.reused:
            return [["herdr", "agent", "focus", self.pane]]
        return [["herdr", "tab", "focus", self.tab]]

    def attach_commands(self) -> list[list[str]]:
        """Focus, then attach the herdr TUI — for a browser outside herdr.

        Focusing only rearranges a herdr UI that is already on someone's
        screen, which from a plain terminal is invisible. Bare ``herdr``
        attaches to (or launches a client for) the running session, so the
        focused tab is what the user lands on — mirroring the plain-terminal
        ``sesh connect`` path in ``tmux.py``.
        """
        return [*self.switch_commands(), ["herdr"]]


def available() -> bool:
    """True when herdr is installed and its server is up and compatible.

    This gates whether the handoff key is offered at all, so it has to mean "a
    handoff would work", not merely "the binary exists": every command below
    goes through the server's socket, and a protocol the server doesn't speak
    fails them just as a stopped server does. ``status server`` reports both
    and exits 0 either way, so the answer is read from its JSON rather than
    from the exit status.
    """
    if shutil.which("herdr") is None:
        return False
    try:
        result = subprocess.run(
            ["herdr", "status", "server", "--json"],
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
    if not isinstance(status, dict):
        return False
    return bool(status.get("running")) and bool(status.get("compatible"))


def in_herdr() -> bool:
    """True when running inside a herdr-managed pane (so we can focus, not
    attach). Herdr sets ``HERDR_ENV=1`` in every pane it owns."""
    return os.environ.get("HERDR_ENV") == "1"


def _run(args: list[str]) -> str:
    """Run a ``herdr`` command and return its stdout, raising on failure.

    Herdr reports failures as JSON on stderr with a non-zero exit, so the
    error's own message is preferred over the raw stderr when one can be read.
    Success output varies: the commands that only send input to a pane print
    nothing at all and answer through the exit status alone, which is why this
    does not insist on a body — see ``_api`` for the ones that do return data.
    """
    result = subprocess.run(
        ["herdr", *args], capture_output=True, text=True, check=False
    )
    if result.returncode != 0:
        raise HerdrError(f"herdr failed: {_error_detail(result) or ' '.join(args)}")
    return result.stdout


def _api(args: list[str]) -> dict:
    """Run a ``herdr`` command that answers with data, and return its result.

    Those commands print ``{"id": ..., "result": {...}}`` on stdout.
    """
    stdout = _run(args)
    try:
        payload = json.loads(stdout)
    except ValueError:
        raise HerdrError(f"herdr gave no JSON for '{' '.join(args)}'") from None
    body = payload.get("result") if isinstance(payload, dict) else None
    if not isinstance(body, dict):
        raise HerdrError(f"herdr gave no result for '{' '.join(args)}'")
    return body


def _error_detail(result: subprocess.CompletedProcess) -> str:
    """The message from a failed herdr command, unwrapped if it was JSON."""
    raw = (result.stderr or result.stdout or "").strip()
    try:
        payload = json.loads(raw)
    except ValueError:
        return raw
    error = payload.get("error") if isinstance(payload, dict) else None
    if isinstance(error, dict):
        message = error.get("message") or error.get("code")
        if isinstance(message, str) and message:
            return message
    return raw


def list_panes() -> list[HerdrPane]:
    """Every pane herdr currently holds, across all workspaces."""
    panes = _api(["pane", "list"]).get("panes")
    if not isinstance(panes, list):
        raise HerdrError("herdr did not list any panes")
    return [_as_pane(raw) for raw in panes if isinstance(raw, dict)]


def _as_pane(raw: dict) -> HerdrPane:
    session = raw.get("agent_session")
    session_id = session.get("value") if isinstance(session, dict) else None
    return HerdrPane(
        pane_id=str(raw.get("pane_id") or ""),
        tab_id=str(raw.get("tab_id") or ""),
        workspace_id=str(raw.get("workspace_id") or ""),
        cwd=str(raw.get("cwd") or ""),
        agent=str(raw.get("agent") or ""),
        session_id=str(session_id or ""),
    )


def find_pane(
    panes: list[HerdrPane], provider: str, session_ids: set[str]
) -> HerdrPane | None:
    """The pane already running this conversation, if herdr reports one.

    The agent kind is compared as well as the id, because an id alone could
    match a pane running a different tool on a coincidentally equal id. The
    three providers this browser supports are named exactly as herdr names its
    agent kinds (``claude``, ``codex``, ``opencode``), so ``provider`` needs no
    translation.
    """
    for pane in panes:
        if (
            pane.session_id
            and pane.session_id in session_ids
            and pane.agent == provider
        ):
            return pane
    return None


def workspace_for_cwd(panes: list[HerdrPane], cwd: str) -> str | None:
    """The workspace already rooted at ``cwd``, found through its panes.

    Herdr reports a workspace's own path only for the git-worktree-backed ones,
    but every pane reports its ``cwd``, so a folder's workspace is whichever
    one holds a pane sitting in that folder. Paths are normalised because the
    provider-recorded ``cwd`` may carry a trailing slash herdr's does not.
    """
    target = os.path.normpath(cwd)
    for pane in panes:
        if pane.cwd and os.path.normpath(pane.cwd) == target:
            return pane.workspace_id
    return None


def prepare_session(
    provider: str, session_id: str, cwd: str, content_path: str = ""
) -> HerdrPlan:
    """Create-or-find the session's herdr pane and return the plan.

    Runs the commands that leave the resume window existing but unfocused, then
    hands the caller a plan whose ``switch_commands()``/``attach_commands()``
    do the visible handoff — the split ``tmux.py`` makes for the same reason:
    attaching from outside the multiplexer has to take over the terminal, which
    only the caller can arrange (see ``SessionBrowser._open_in_multiplexer``).

    A pane already running this conversation — matched on any id in its fork
    lineage, recovered from the transcript at ``content_path`` — is reused as
    is. Otherwise a tab is added to the folder's workspace (creating that
    workspace if the folder has none) and the resume command is run in its root
    pane.
    """
    if not cwd:
        raise HerdrError("session has no folder (cwd) to open")
    if shutil.which("herdr") is None:
        raise HerdrError("herdr not found on PATH")

    resume = resume_command(provider, session_id, None)
    if resume.startswith("#"):
        raise HerdrError(resume.lstrip("# ").strip() or "no resume command")

    label = workspace_label_for_path(cwd)
    panes = list_panes()
    existing = find_pane(
        panes, provider, lineage_ids(provider, session_id, content_path)
    )
    if existing:
        return HerdrPlan(
            label=label,
            cwd=cwd,
            resume=resume,
            session_id=session_id,
            tab=existing.tab_id,
            pane=existing.pane_id,
            reused=True,
        )

    workspace = workspace_for_cwd(panes, cwd)
    if workspace:
        created = _api(
            ["tab", "create", "--workspace", workspace, "--cwd", cwd, "--no-focus"]
        )
    else:
        created = _api(
            ["workspace", "create", "--cwd", cwd, "--label", label, "--no-focus"]
        )
    tab = _created_id(created, "tab", "tab_id")
    pane = _created_id(created, "root_pane", "pane_id")
    if not tab or not pane:
        raise HerdrError("herdr did not report the new tab's root pane")
    # The pane is a plain login shell, so the agent is typed into it rather
    # than being the pane's command — same reasoning as tmux.py's send-keys:
    # when the agent exits the user is left in their configured shell. This
    # one answers with no body, only an exit status.
    _run(["pane", "run", pane, resume])
    return HerdrPlan(
        label=label,
        cwd=cwd,
        resume=resume,
        session_id=session_id,
        tab=tab,
        pane=pane,
        reused=False,
    )


def _created_id(created: dict, key: str, field: str) -> str:
    """Pull an id out of a ``workspace create``/``tab create`` response."""
    value = created.get(key)
    if not isinstance(value, dict):
        return ""
    return str(value.get(field) or "")
