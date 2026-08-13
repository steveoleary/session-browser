"""Open a session's folder in tmux, mirroring worktrees.sh.

The session browser is otherwise read-only, but this module lets you jump from
a highlighted session straight into a live terminal. Each distinct folder
(a session's ``cwd``) maps to one tmux session named after the folder's
basename — exactly like ``session_name_for_root`` in ``worktrees.sh``.

If that tmux session does not exist yet it is created detached, with its first
window running the provider's resume command. If it already exists, we first
check whether one of its windows is already running this same conversation
(each window created here is tagged with the session id(s) it has hosted, via
a ``@sb_session_id`` window option); if so we reuse that window instead of
opening a duplicate. Otherwise a new window running resume is added. Finally
we hand off to ``sesh connect --switch``, the same call ``worktrees.sh``'s
``ensure_repo_session`` uses, so the behaviour is "create, reuse, or connect
to a tmux session for this folder".

Matching a window cannot be done on the selected session id alone: Claude
Code's ``--resume`` *forks* — the window opened for id X immediately continues
the conversation under a fresh id Y, so the row the user picks on the next
visit (the live, most recent one) never equals the id the window was tagged
with. ``transcript.lineage_ids`` recovers the conversation's earlier ids from
its transcript so the lookup can match the tag stamped on the original window.

``herdr.py`` is the sibling of this module for the other multiplexer, and
``multiplexer.py`` picks between them; the two share the plan interface
``app.py`` drives (``available``, ``in_*``, ``prepare_session``, and a plan
with ``switch_commands``/``attach_commands``/``label``/``reused``).
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
from dataclasses import dataclass

from .resume import resume_command
from .transcript import lineage_ids

# Availability is probed on the UI thread when the user presses the handoff
# key, so a wedged tmux server must not hang the browser.
_PROBE_TIMEOUT = 2.0

# tmux forbids '.' and ':' in session names; restrict to the same safe set
# worktrees.sh uses (`tr -c 'A-Za-z0-9_-' '_'`).
_SANITIZE_RE = re.compile(r"[^A-Za-z0-9_-]")

# Window option used to remember which session ids a tmux window has hosted
# (space-separated; resume forks grow the conversation new ids over time), so
# a later visit to the same conversation can reuse the window instead of
# opening a duplicate.
_SESSION_TAG_OPTION = "@sb_session_id"


class TmuxError(RuntimeError):
    """Raised when a session cannot be opened in tmux."""


def session_name_for_path(path: str) -> str:
    """tmux session name for a folder, mirroring ``session_name_for_root``.

    Uses the folder's basename with every character outside ``[A-Za-z0-9_-]``
    replaced by ``_``.
    """
    base = os.path.basename(os.path.normpath(path)) if path else ""
    name = _SANITIZE_RE.sub("_", base)
    return name or "session"


@dataclass
class TmuxPlan:
    """The tmux/sesh commands needed to open one folder's session."""

    session: str
    cwd: str
    resume: str
    session_exists: bool
    session_id: str
    existing_window: str | None = None
    existing_tag: str = ""

    def create_commands(self) -> list[list[str]]:
        """tmux command(s) that open the window and start the agent in it.

        If ``existing_window`` is set, a window already running this
        conversation was found (see ``_find_window``); we select it rather
        than typing the resume command into a second, duplicate window, and
        append the id the user picked to the window's tag so the next visit
        can exact-match it without walking the fork lineage again.

        Otherwise the window is started as a *plain* shell (no command), so
        tmux launches your normal login shell with its usual prompt and
        config — then the resume command is typed into it via ``send-keys``.
        When the agent exits you land back in that same configured shell,
        instead of the bare ``exec $SHELL`` prompt that never sourced your
        dotfiles. The window is then tagged with the session id so a future
        visit can find and reuse it.

        Mirrors ``worktrees.sh``: a fresh folder gets a detached ``new-session``,
        an existing one gets another ``new-window``, both rooted at ``cwd`` via
        ``-c`` and left unnamed so automatic-rename tracks the running command.
        ``send-keys`` targets the session, whose newly created window is active.
        """
        if self.existing_window:
            commands = [["tmux", "select-window", "-t", self.existing_window]]
            tags = self.existing_tag.split()
            if self.session_id not in tags:
                commands.append(
                    [
                        "tmux",
                        "set-option",
                        "-t",
                        self.existing_window,
                        "-w",
                        _SESSION_TAG_OPTION,
                        " ".join(tags + [self.session_id]),
                    ]
                )
            return commands
        if self.session_exists:
            open_window = ["tmux", "new-window", "-t", self.session, "-c", self.cwd]
        else:
            open_window = [
                "tmux",
                "new-session",
                "-d",
                "-s",
                self.session,
                "-c",
                self.cwd,
            ]
        return [
            open_window,
            # -l: send the command as literal text (don't parse it as key names).
            ["tmux", "send-keys", "-t", self.session, "-l", self.resume],
            ["tmux", "send-keys", "-t", self.session, "Enter"],
            [
                "tmux",
                "set-option",
                "-t",
                self.session,
                "-w",
                _SESSION_TAG_OPTION,
                self.session_id,
            ],
        ]

    def connect_command(self, *, switch: bool) -> list[str]:
        """``sesh connect`` for the folder's session.

        ``switch=True`` adds ``--switch`` (``tmux switch-client``), matching
        ``worktrees.sh``'s ``ensure_repo_session`` — correct only when already
        inside tmux. ``switch=False`` attaches, which is what's needed from a
        plain terminal; ``--switch`` there is a no-op (there is no client to
        switch) and would just flicker back to the caller.
        """
        cmd = ["sesh", "connect"]
        if switch:
            cmd.append("--switch")
        cmd.append(self.session)
        return cmd

    def switch_commands(self) -> list[list[str]]:
        """Handoff from inside tmux — the shared plan interface ``app.py``
        drives, implemented for herdr by ``HerdrPlan``."""
        return [self.connect_command(switch=True)]

    def attach_commands(self) -> list[list[str]]:
        """Handoff from a plain terminal; the caller must give up the terminal
        for these, since attaching takes it over."""
        return [self.connect_command(switch=False)]

    @property
    def label(self) -> str:
        """What to call this destination in the status bar."""
        return self.session

    @property
    def reused(self) -> bool:
        """True when the conversation was already open and got reused."""
        return bool(self.existing_window)


def build_plan(
    provider: str,
    session_id: str,
    cwd: str,
    *,
    session_exists: bool,
    existing_window: str | None = None,
    existing_tag: str = "",
) -> TmuxPlan:
    """Assemble a :class:`TmuxPlan` (cwd handled by ``-c``, so no ``cd``)."""
    resume = resume_command(provider, session_id, None)
    return TmuxPlan(
        session=session_name_for_path(cwd),
        cwd=cwd,
        resume=resume,
        session_exists=session_exists,
        session_id=session_id,
        existing_window=existing_window,
        existing_tag=existing_tag,
    )


def _session_exists(name: str) -> bool:
    """True when a tmux session called ``name`` already exists."""
    result = subprocess.run(
        ["tmux", "has-session", "-t", f"={name}"],
        capture_output=True,
        check=False,
    )
    return result.returncode == 0


def _find_window(session: str, session_ids: set[str]) -> tuple[str, str] | None:
    """``(window_id, tag)`` of a window already hosting this conversation.

    Looks up the ``@sb_session_id`` option ``create_commands`` stamps onto
    each window it creates (a space-separated id list — see the retag in
    ``create_commands``) and matches it against every id the conversation is
    known by (``lineage_ids``), so a session already open in some window can
    be reused instead of duplicated.
    """
    result = subprocess.run(
        [
            "tmux",
            "list-windows",
            "-t",
            session,
            "-F",
            "#{window_id} #{" + _SESSION_TAG_OPTION + "}",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        return None
    for line in result.stdout.splitlines():
        window_id, _, tag = line.partition(" ")
        if session_ids.intersection(tag.split()):
            return window_id, tag
    return None


def prepare_session(
    provider: str, session_id: str, cwd: str, content_path: str = ""
) -> TmuxPlan:
    """Create-or-reuse the folder's tmux session and return the plan.

    Runs the detached create command(s) so the resume window exists, then hands
    the caller a plan whose ``connect_command()`` switches/attaches to it. The
    connect step is left to the caller because attaching (when not already in
    tmux) must take over the terminal — see ``SessionBrowser.action_open_tmux``.

    If a window already exists for this conversation — tagged by a previous
    visit with ``session_id`` or any earlier id in its fork lineage, recovered
    from the transcript at ``content_path`` — that window is reused instead of
    opening a duplicate.
    """
    if not cwd:
        raise TmuxError("session has no folder (cwd) to open")
    if shutil.which("tmux") is None:
        raise TmuxError("tmux not found on PATH")
    if shutil.which("sesh") is None:
        raise TmuxError("sesh not found on PATH")

    resume = resume_command(provider, session_id, None)
    if resume.startswith("#"):
        raise TmuxError(resume.lstrip("# ").strip() or "no resume command")

    session = session_name_for_path(cwd)
    exists = _session_exists(session)
    found = (
        _find_window(session, lineage_ids(provider, session_id, content_path))
        if exists
        else None
    )
    existing_window, existing_tag = found if found else (None, "")
    plan = build_plan(
        provider,
        session_id,
        cwd,
        session_exists=exists,
        existing_window=existing_window,
        existing_tag=existing_tag,
    )
    for cmd in plan.create_commands():
        result = subprocess.run(cmd, capture_output=True, text=True, check=False)
        if result.returncode != 0:
            detail = (result.stderr or result.stdout or "").strip()
            raise TmuxError(f"tmux failed: {detail or ' '.join(cmd)}")
    return plan


def in_tmux() -> bool:
    """True when running inside a tmux client (so we can switch, not attach)."""
    return bool(os.environ.get("TMUX"))


def available() -> bool:
    """True when tmux is installed and a server is actually running.

    This gates whether the handoff key is offered, so "installed" is not
    enough — on a machine where the user works in another multiplexer, an
    installed-but-idle tmux would advertise a destination they never asked
    for. ``list-sessions`` exits non-zero when no server is running, and a
    server with no sessions does not stay alive, so its exit status is the
    test; inside tmux there is a server by definition, so ``$TMUX``
    short-circuits the probe.

    ``sesh`` is deliberately not required here: it is needed only for the
    final connect, and ``prepare_session`` names it in an error the user can
    act on, which is more use than the key silently disappearing.
    """
    if shutil.which("tmux") is None:
        return False
    if in_tmux():
        return True
    try:
        result = subprocess.run(
            ["tmux", "list-sessions"],
            capture_output=True,
            timeout=_PROBE_TIMEOUT,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return False
    return result.returncode == 0
