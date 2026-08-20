"""Resume command generation and clipboard support."""

from __future__ import annotations

import platform
import re
import shlex
import subprocess
from pathlib import Path

RESUME_PATTERNS: dict[str, str] = {
    "claude": "claude --dangerously-skip-permissions --resume {id}",
    "codex": "codex resume {id}",
    "opencode": "opencode -s {id}",
    # --session takes a session file or a (partial) uuid and resolves it
    # within the project's own session directory, which is why the cwd
    # prefix matters. Not --session-id, which CREATES the session when the
    # id does not resolve — a typo would open an empty conversation rather
    # than fail.
    "pi": "pi --session {id}",
}


def resume_command(provider: str, session_id: str, cwd: str | None = None) -> str:
    """Build the provider-specific resume command string."""
    pattern = RESUME_PATTERNS.get(provider)
    if not pattern:
        return f"# unknown provider: {provider}"
    cmd = pattern.format(id=shlex.quote(session_id))
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
