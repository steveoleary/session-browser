"""Resume command generation and clipboard support."""

from __future__ import annotations

import json
import logging
import platform
import re
import shlex
import subprocess
from pathlib import Path

from . import compressed
from .discovery import claude_root_session

log = logging.getLogger(__name__)

# {flags} is the session's own mode, as resume_flags reads it: empty when the
# transcript records none, so the tool starts in whatever its config says.
RESUME_PATTERNS: dict[str, str] = {
    "claude": "claude{flags} --resume {id}",
    "codex": "codex resume{flags} {id}",
    "opencode": "opencode -s {id}",
    # --session takes a session file or a (partial) uuid and resolves it
    # within the project's own session directory, which is why the cwd
    # prefix matters. Not --session-id, which CREATES the session when the
    # id does not resolve — a typo would open an empty conversation rather
    # than fail.
    "pi": "pi --session {id}",
}

# Claude Code's --permission-mode choices (2.1.331), plus "default": no
# longer listed, still accepted, and what it records for a session started
# with no mode. A value outside these is a mode this list has not met, and is
# left out rather than handed to a CLI that would refuse to start.
_CLAUDE_MODES = frozenset(
    {"acceptEdits", "auto", "bypassPermissions", "default", "dontAsk", "manual", "plan"}
)
# Codex's --ask-for-approval and --sandbox choices (0.172.0). Older sessions
# can record approval policies this CLI no longer takes ("untrusted",
# "on-failure"); those are left out for the same reason.
_CODEX_APPROVALS = frozenset({"on-request", "never"})
_CODEX_SANDBOXES = frozenset({"read-only", "workspace-write", "danger-full-access"})


def resume_target(session) -> tuple[str, str]:
    """``(session id, transcript path)`` that resuming *session* opens.

    Claude Code resumes a conversation, not an agent inside one, so a
    subagent resolves to the root session it ran in, and to that session's
    transcript, which sits beside the directory holding the subagent's.
    """
    root = claude_root_session(session)
    if not root:
        return session.id, session.content_path
    path = Path(session.content_path)
    for parent in path.parents:
        if parent.name == root:
            return root, str(parent.parent / f"{root}.jsonl")
    return root, ""


def resume_flags(provider: str, content_path: str) -> tuple[str, ...]:
    """The flags that resume a session in the mode it was started in.

    Read from the session's own transcript, so the command depends on
    nothing but the file. Empty when the provider records no mode, when the
    transcript names none this CLI accepts, or when it cannot be read: a
    resume with no mode flag starts in whatever the tool's config says,
    which is never more permissive than the session was by accident.

    Claude Code records the mode on ``permission-mode`` records and on some
    user records; the first one is the mode it started in (on this machine,
    297 of 343 transcripts record one, the rest being sessions a ``/clear``
    opened, which record none until the mode changes). Codex records its
    approval and sandbox policies on every ``turn_context``, the first by
    the tenth line of every rollout here. OpenCode's sessions carry no mode
    (``permission`` is empty on every row), and Pi has no permission modes.
    """
    if not content_path:
        return ()
    try:
        if provider == "claude":
            return _claude_mode_flags(Path(content_path))
        if provider == "codex":
            return _codex_mode_flags(Path(content_path))
    except (OSError, ValueError) as exc:
        log.debug("no resume mode from %s: %s", content_path, exc)
    return ()


def _claude_mode_flags(path: Path) -> tuple[str, ...]:
    mode = _first_record_field(path, b'"permissionMode"', "permissionMode", None)
    if mode not in _CLAUDE_MODES:
        return ()
    if mode == "bypassPermissions":
        # The spelling this browser always used for it, and the one that
        # skips Claude Code's are-you-sure prompt like the session did.
        return ("--dangerously-skip-permissions",)
    return ("--permission-mode", mode)


def _codex_mode_flags(path: Path) -> tuple[str, ...]:
    context = _first_record_field(path, b'"turn_context"', "payload", "turn_context")
    if not isinstance(context, dict):
        return ()
    approval = context.get("approval_policy")
    sandbox = context.get("sandbox_policy")
    if isinstance(sandbox, dict):
        sandbox = sandbox.get("type")
    if approval == "never" and sandbox == "danger-full-access":
        return ("--dangerously-bypass-approvals-and-sandbox",)
    flags: list[str] = []
    if sandbox in _CODEX_SANDBOXES:
        flags += ["--sandbox", sandbox]
    if approval in _CODEX_APPROVALS:
        flags += ["--ask-for-approval", approval]
    return tuple(flags)


def _first_record_field(path: Path, marker: bytes, key: str, record_type: str | None):
    """*key* of the first top-level record carrying it, or None.

    Only a line holding *marker* is decoded. Inside a JSON string a quote is
    escaped, so a transcript that merely quotes ``"permissionMode"`` -- a tool
    result showing another transcript -- holds the marker with backslashes
    and never matches; a structured tool result nesting the key is decoded
    and passed over, since only a top-level key is the record's own.
    """
    with compressed.open_binary(path) as fh:
        for line in fh:
            if marker not in line:
                continue
            try:
                record = json.loads(line)
            except ValueError:
                continue
            if not isinstance(record, dict) or key not in record:
                continue
            if record_type is not None and record.get("type") != record_type:
                continue
            return record[key]
    return None


def resume_command(
    provider: str,
    session_id: str,
    cwd: str | None = None,
    *,
    content_path: str = "",
) -> str:
    """Build the provider-specific resume command string.

    *content_path* is the session's transcript, which says which mode to
    resume it in (see :func:`resume_flags`); without one the command carries
    no mode flag.
    """
    pattern = RESUME_PATTERNS.get(provider)
    if not pattern:
        return f"# unknown provider: {provider}"
    flags = resume_flags(provider, content_path)
    cmd = pattern.format(
        id=shlex.quote(session_id),
        flags="".join(f" {shlex.quote(flag)}" for flag in flags),
    )
    if cwd:
        return f"cd {shlex.quote(cwd)} && {cmd}"
    return cmd


def copy_to_clipboard(text: str) -> bool:
    """Copy text to system clipboard. Returns True on success."""
    system = platform.system()
    try:
        if system == "Darwin":
            subprocess.run(["pbcopy"], input=text.encode(), check=True, timeout=5)
        elif system == "Linux":
            for tool in (
                "clip.exe",
                "xclip -selection clipboard",
                "xsel --clipboard --input",
            ):
                parts = tool.split()
                try:
                    subprocess.run(parts, input=text.encode(), check=True, timeout=5)
                    break
                except FileNotFoundError:
                    continue
            else:
                return False
        else:
            return False
        return True
    except (subprocess.SubprocessError, OSError):
        return False


def build_chat_export(session, content: str) -> str:
    """Build the plain text handoff export for a session."""
    title = (getattr(session, "summary", "") or getattr(session, "id", "")).strip()
    body = (content or "(no content)").strip()
    return f"{title}\n\n{body}\n"


def export_chat_to_file(
    session, content: str, output_dir: Path | str | None = None
) -> Path:
    """Write the plain text chat export to a deterministic .txt file."""
    base_dir = Path(output_dir) if output_dir is not None else Path.cwd()
    provider = _filename_part(getattr(session, "provider", "session") or "session")
    session_id = _filename_part(getattr(session, "id", "unknown") or "unknown")
    path = base_dir / f"session-export-{provider}-{session_id}.txt"
    path.write_text(build_chat_export(session, content))
    return path


def _filename_part(value: str) -> str:
    """Return a filesystem-friendly filename segment."""
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "-", value.strip())
    return cleaned.strip("-") or "unknown"
