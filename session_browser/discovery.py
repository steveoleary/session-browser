"""Session discovery for multiple agent providers."""

from __future__ import annotations

import json
import logging
import sqlite3
from collections.abc import Iterable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Unified session model
# ---------------------------------------------------------------------------


@dataclass
class Session:
    id: str
    provider: str
    summary: str = ""
    cwd: str = ""
    branch: str = ""
    repository: str = ""
    created_at: str = ""
    updated_at: str = ""
    content_path: str = ""  # file/dir path for rg search & content loading

    @property
    def sort_key(self) -> str:
        """Newest first — use updated_at, fallback to created_at, then id."""
        return self.updated_at or self.created_at or self.id

    def matches(self, query: str) -> bool:
        """Case-insensitive metadata match for global search."""
        q = query.lower()
        return any(
            q in (getattr(self, f) or "").lower()
            for f in ("summary", "provider", "branch", "repository", "cwd", "id")
        )


# ---------------------------------------------------------------------------
# Provider scanners
# ---------------------------------------------------------------------------


def scan_claude() -> list[Session]:
    """Discover Claude Code sessions from JSONL files."""
    sessions: list[Session] = []
    root = Path.home() / ".claude" / "projects"
    if not root.is_dir():
        return sessions
    for project_dir in root.iterdir():
        if not project_dir.is_dir():
            continue
        for f in project_dir.glob("*.jsonl"):
            try:
                session_id = f.stem
                summary, cwd, branch, created_at = "", "", "", ""
                with open(f) as fh:
                    for line in fh:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            obj = json.loads(line)
                        except json.JSONDecodeError:
                            continue
                        # Extract metadata from first user message
                        if obj.get("type") == "user" and not obj.get("isMeta"):
                            msg = obj.get("message", {})
                            content = msg.get("content", "")
                            if isinstance(content, str):
                                summary = content[:120].replace("\n", " ")
                            cwd = obj.get("cwd", "")
                            branch = obj.get("gitBranch", "")
                            created_at = obj.get("timestamp", "")
                            break
                # Decode project path from dir name
                decoded_project = project_dir.name.replace("-", "/")
                sessions.append(
                    Session(
                        id=session_id,
                        provider="claude",
                        summary=summary,
                        cwd=cwd or decoded_project,
                        branch=branch,
                        created_at=created_at,
                        updated_at=(
                            _last_activity_iso(f, "claude")
                            or created_at
                            or _file_mtime_iso(f)
                        ),
                        content_path=str(f),
                    )
                )
            except Exception as exc:
                log.warning("Skipping claude session %s: %s", f.name, exc)
    return sessions


def scan_codex() -> list[Session]:
    """Discover Codex CLI sessions from dated JSONL tree."""
    sessions: list[Session] = []
    root = Path.home() / ".codex" / "sessions"
    if not root.is_dir():
        return sessions
    for f in root.rglob("rollout-*.jsonl"):
        try:
            with open(f) as fh:
                first_line = fh.readline().strip()
            if not first_line:
                continue
            obj = json.loads(first_line)
            if obj.get("type") != "session_meta":
                continue
            payload = obj.get("payload", {})
            sid = payload.get("id", f.stem)
            cwd = payload.get("cwd", "")
            git = payload.get("git", {})
            branch = git.get("branch", "")
            ts = payload.get("timestamp", obj.get("timestamp", ""))
            # Try to get first user message for summary
            summary = _codex_first_user_message(f)
            sessions.append(
                Session(
                    id=sid,
                    provider="codex",
                    summary=summary,
                    cwd=cwd,
                    branch=branch,
                    created_at=ts,
                    updated_at=(
                        _last_activity_iso(f, "codex") or ts or _file_mtime_iso(f)
                    ),
                    content_path=str(f),
                )
            )
        except Exception as exc:
            log.warning("Skipping codex session %s: %s", f.name, exc)
    return sessions


def _codex_first_user_message(path: Path) -> str:
    """Extract first user message from a codex JSONL for the summary."""
    try:
        with open(path) as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue
                # Codex wraps events in event_msg envelope
                payload = (
                    obj.get("payload", {}) if obj.get("type") == "event_msg" else obj
                )
                ptype = payload.get("type", "")
                if ptype == "user_message":
                    msg = payload.get("message", "")
                    if isinstance(msg, str) and msg:
                        return msg[:120].replace("\n", " ")
                    content = payload.get("content", "")
                    if isinstance(content, str) and content:
                        return content[:120].replace("\n", " ")
                    if isinstance(content, list):
                        for part in content:
                            if (
                                isinstance(part, dict)
                                and part.get("type") == "input_text"
                            ):
                                return part.get("text", "")[:120].replace("\n", " ")
    except Exception as exc:
        # Best-effort summary: an unreadable file costs a blank summary, not
        # the session. Logged at debug so it is recoverable when one is blank.
        log.debug("No codex summary from %s: %s", path.name, exc)
    return ""


def scan_opencode() -> list[Session]:
    """Discover opencode sessions from its SQLite database."""
    sessions: list[Session] = []
    db_path = _opencode_db_path()
    if not db_path.exists():
        return sessions
    try:
        conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True, timeout=3)
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            "SELECT s.id, s.title, s.directory, s.time_created, s.time_updated, "
            "       p.worktree, p.name AS project_name "
            "FROM session s "
            "LEFT JOIN project p ON s.project_id = p.id "
            "ORDER BY s.time_updated DESC"
        ).fetchall()
        conn.close()
        for r in rows:
            sessions.append(
                Session(
                    id=r["id"],
                    provider="opencode",
                    summary=r["title"] or "",
                    cwd=r["directory"] or r["worktree"] or "",
                    repository=r["project_name"] or "",
                    created_at=_epoch_ms_to_iso(r["time_created"]),
                    updated_at=_epoch_ms_to_iso(r["time_updated"]),
                    content_path=str(db_path),
                )
            )
    except (sqlite3.Error, OSError) as exc:
        log.warning("opencode discovery failed: %s", exc)
    return sessions


def _opencode_db_path() -> Path:
    return Path.home() / ".local" / "share" / "opencode" / "opencode.db"


def _epoch_ms_to_iso(ts) -> str:
    """Convert epoch-millisecond integer to UTC ISO string."""
    if not ts:
        return ""
    from datetime import timezone

    try:
        return datetime.fromtimestamp(int(ts) / 1000, tz=timezone.utc).isoformat()
    except (ValueError, OSError, OverflowError):
        return ""


def _file_mtime_iso(path: Path) -> str:
    """Return file modification time as UTC ISO string."""
    from datetime import timezone

    try:
        ts = path.stat().st_mtime
        return datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()
    except OSError:
        return ""


# Which JSONL lines count as a genuine conversation turn (vs session-open /
# resume / mode / lifecycle events that get written — or re-touch the file —
# without any new activity). The timestamp of the *last* such turn is the true
# "last activity" time, immune to file-mtime drift.
_CLAUDE_TURN_TYPES = {"user", "assistant"}
_CODEX_TURN_PAYLOADS = {"user_message", "agent_message"}


def _is_turn(obj: dict, provider: str) -> bool:
    """True if a parsed JSONL object represents a real conversation turn."""
    if provider == "claude":
        return obj.get("type") in _CLAUDE_TURN_TYPES
    if provider == "codex":
        if obj.get("type") != "event_msg":
            return False
        payload = obj.get("payload")
        return isinstance(payload, dict) and payload.get("type") in _CODEX_TURN_PAYLOADS
    return False


def _last_activity_iso(path: Path, provider: str, *, window: int = 16384) -> str:
    """Timestamp of the last genuine conversation turn in a JSONL transcript.

    Reads the file tail backwards so a bumped mtime (a read/sync/resume that
    touches the file, or a trailing session-open event carrying a fresh
    timestamp) can't masquerade as recent activity. Widens the read window if
    the last turn sits deeper than the initial tail. Returns "" when no
    timestamped turn is found; callers fall back to created_at, then mtime.
    """
    try:
        size = path.stat().st_size
    except OSError:
        return ""
    if size == 0:
        return ""
    while True:
        read = min(window, size)
        try:
            with open(path, "rb") as fh:
                fh.seek(size - read)
                data = fh.read(read)
        except OSError:
            return ""
        lines = data.decode("utf-8", errors="replace").split("\n")
        if read < size:
            lines = lines[1:]  # drop the possibly-partial leading fragment
        for line in reversed(lines):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(obj, dict):
                continue
            ts = obj.get("timestamp")
            if ts and _is_turn(obj, provider):
                return ts
        if read >= size:
            return ""  # whole file scanned, no timestamped turn found
        window *= 4  # last turn is deeper than the tail — widen and retry


# ---------------------------------------------------------------------------
# Aggregate discovery
# ---------------------------------------------------------------------------

ALL_SCANNERS = {
    "claude": scan_claude,
    "codex": scan_codex,
    "opencode": scan_opencode,
}


def discover_all(providers: Iterable[str] | None = None) -> list[Session]:
    """Run provider scanners concurrently, return unified sorted list.

    *providers* restricts the scan to those provider names (case-insensitive);
    unknown names simply match no scanner. None means scan everything.
    Scanners are I/O-bound (per-file JSONL reads, SQLite opens), so one
    thread per scanner overlaps their waits.
    """
    if providers is None:
        scanners = ALL_SCANNERS
    else:
        wanted = {p.lower() for p in providers}
        scanners = {n: fn for n, fn in ALL_SCANNERS.items() if n in wanted}
    all_sessions: list[Session] = []
    if scanners:
        with ThreadPoolExecutor(max_workers=len(scanners)) as ex:
            futures = {name: ex.submit(fn) for name, fn in scanners.items()}
        for name, fut in futures.items():
            try:
                all_sessions.extend(fut.result())
            except Exception as exc:
                log.warning("Provider %s discovery failed: %s", name, exc)
    all_sessions.sort(key=lambda s: s.sort_key, reverse=True)
    return all_sessions
