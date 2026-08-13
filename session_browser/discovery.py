"""Session discovery for multiple agent providers."""

from __future__ import annotations

import json
import logging
import os
import re
import sqlite3
from collections.abc import Iterable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import UTC, datetime
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
    """Discover Codex CLI sessions, preferring its SQLite index.

    Codex keeps its own session index in ``~/.codex/state_*.sqlite``, one
    ``threads`` row per rollout. Reading it replaces hundreds of file opens
    with one query. The file scan remains the fallback for when the index is
    absent, renamed by an upgrade, locked, or lagging a just-written rollout
    -- the DB must never make a session that exists on disk invisible, so the
    DB path is wrapped broadly enough that any failure of it degrades to the
    file scan rather than to zero sessions.
    """
    db_path = _codex_db_path()
    if db_path is not None:
        try:
            sessions = _scan_codex_db(db_path)
        except Exception as exc:
            # Deliberately broad: the promise above is absolute, and an
            # unexpected exception (a schema surprise, a sqlite3.Warning,
            # anything) must cost the fast path, not the sessions.
            log.warning("codex discovery via %s failed: %s", db_path.name, exc)
        else:
            if sessions is not None:
                return sessions
    return _scan_codex_files()


def _codex_db_path() -> Path | None:
    """The newest ``state_*.sqlite`` under ``~/.codex``, or None.

    The filename carries a version (``state_5.sqlite``); a Codex upgrade
    writes the next number and may leave the old file behind. Picking the
    highest keeps discovery alive across upgrades instead of silently
    returning no Codex sessions when a hard-coded path goes stale.
    """
    candidates = list((Path.home() / ".codex").glob("state_*.sqlite"))
    if not candidates:
        return None
    return max(candidates, key=lambda p: (_codex_state_version(p), p.name))


def _codex_state_version(path: Path) -> int:
    match = re.match(r"state_(\d+)", path.stem)
    return int(match.group(1)) if match else 0


def _readonly_uri(path: Path) -> str:
    """A read-only SQLite URI for *path*, with the path itself escaped.

    ``Path.as_uri()`` percent-encodes characters that would otherwise be
    interpreted as URI syntax (``?``, ``#``, spaces), so a home directory
    with any of them in its name still opens correctly.
    """
    return f"{path.as_uri()}?mode=ro"


_CODEX_DB_TIMEOUT = 0.3  # seconds; a locked index is ~1.8s of file scan away
_warned_codex_fallbacks: set[str] = set()


def _warn_codex_fallback_once(message: str) -> None:
    """Warn once per message per process.

    A disagreement can persist for a whole session (or forever, on a machine
    whose tree Codex never fully indexed), and a warning on every invocation
    is one nobody reads."""
    if message not in _warned_codex_fallbacks:
        _warned_codex_fallbacks.add(message)
        log.warning(message)


def _scan_codex_db(db_path: Path) -> list[Session] | None:
    """Build sessions from Codex's own ``threads`` index.

    Returns None when a rollout on disk is missing from the index, so the
    caller falls back to the file scan: Codex writes the rollout file before
    it records the thread, and a file the index has never heard of must not
    be dropped by discovery. The check is one-directional coverage, not
    identity -- a row whose file is gone is simply dropped, since the file
    scan could not have found it either, and an archived thread whose file
    remains in the tree costs the filter, not the fast path.

    The columns this query depends on (``first_user_message``,
    ``created_at_ms``, ``updated_at_ms``) are migration-added to ``threads``.
    A dropped column degrades safely (OperationalError -> file scan); a
    column that survives an upgrade but changes meaning is invisible to this
    code and to every test, which is why the fallback must stay reachable.

    ``updated_at_ms`` is used over ``recency_at_ms`` on purpose: it tracks
    the last written rollout line, while recency lags it. Either can disagree
    with the file scan's last-turn heuristic for sessions written in Codex's
    current event schema; the DB is the truthful one of the two.
    """
    conn = sqlite3.connect(_readonly_uri(db_path), uri=True, timeout=_CODEX_DB_TIMEOUT)
    conn.row_factory = sqlite3.Row
    try:
        rows = conn.execute(
            "SELECT id, rollout_path, cwd, git_branch, first_user_message, "
            "created_at_ms, updated_at_ms, archived FROM threads"
        ).fetchall()
    finally:
        conn.close()
    sessions_root = Path.home() / ".codex" / "sessions"
    # realpath both sides: the index may store resolved paths while the
    # walk starts from a symlinked home (or vice versa), and the two must
    # not disagree for a reason that has nothing to do with the data.
    file_paths = {os.path.realpath(p) for p in sessions_root.rglob("rollout-*.jsonl")}
    rows_by_path = {
        os.path.realpath(r["rollout_path"]): r for r in rows if r["rollout_path"]
    }
    missing = file_paths - rows_by_path.keys()
    if missing:
        # The fast path turning itself off, and the repo treats retrieval
        # speed as a gate. A silent fallback reads as "the tool got slow for
        # no reason"; deduped so a persistent gap is said once, not shouted.
        _warn_codex_fallback_once(
            f"codex index is missing {len(missing)} rollout file(s); using file scan"
        )
        return None
    return [
        _codex_session_from_row(r)
        for r in rows
        if r["rollout_path"]
        and os.path.realpath(r["rollout_path"]) in file_paths
        and not r["archived"]
    ]


def _codex_session_from_row(r) -> Session:
    """One Session from a ``threads`` row, fields as the file scan builds them."""
    return Session(
        id=r["id"],
        provider="codex",
        # The index stores the message with leading whitespace trimmed; the
        # file scan reads the raw event and applies this same
        # truncate-and-collapse rule. The two agree except for one leading
        # space on messages that began with whitespace.
        summary=(r["first_user_message"] or "")[:120].replace("\n", " "),
        cwd=r["cwd"] or "",
        branch=r["git_branch"] or "",
        created_at=_epoch_ms_to_iso(r["created_at_ms"], zulu=True),
        updated_at=_epoch_ms_to_iso(r["updated_at_ms"], zulu=True)
        or _epoch_ms_to_iso(r["created_at_ms"], zulu=True),
        content_path=r["rollout_path"],
    )


def _scan_codex_files() -> list[Session]:
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
        conn = sqlite3.connect(_readonly_uri(db_path), uri=True, timeout=3)
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


def _epoch_ms_to_iso(ts, *, zulu: bool = False) -> str:
    """Convert epoch-millisecond integer to UTC ISO string.

    Default shape is ``.isoformat()`` (microseconds, ``+00:00`` offset).
    ``zulu=True`` returns the ``2026-01-02T14:50:26.684Z`` shape Codex writes
    into its rollout files, so a timestamp sourced from the index is
    indistinguishable from one read off a file."""
    if not ts:
        return ""

    try:
        dt = datetime.fromtimestamp(int(ts) / 1000, tz=UTC)
    except (ValueError, OSError, OverflowError):
        return ""
    if zulu:
        return dt.strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"
    return dt.isoformat()


def _file_mtime_iso(path: Path) -> str:
    """Return file modification time as UTC ISO string."""

    try:
        ts = path.stat().st_mtime
        return datetime.fromtimestamp(ts, tz=UTC).isoformat()
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
