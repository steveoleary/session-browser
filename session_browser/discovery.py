"""Session discovery for multiple agent providers."""

from __future__ import annotations

import json
import logging
import os
import re
import sqlite3
from collections import deque
from collections.abc import Callable, Iterable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field, fields
from datetime import UTC, datetime
from pathlib import Path

from . import compressed, scan_memo

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Unified session model
# ---------------------------------------------------------------------------

# How a conversation ended, read off its last records. Four states, and
# None for "cannot say":
#
#   completed     the agent answered: its last turn is a reply
#   unanswered    the person's last message got no reply
#   mid-action    it stopped during a tool call: a call with no result, or a
#                 result the agent never answered
#   interrupted   the person cancelled the turn
COMPLETED, UNANSWERED, MID_ACTION, INTERRUPTED = (
    "completed",
    "unanswered",
    "mid-action",
    "interrupted",
)
ENDINGS = (COMPLETED, UNANSWERED, MID_ACTION, INTERRUPTED)


@dataclass
class Session:
    id: str
    provider: str
    summary: str = ""
    cwd: str = ""
    # None means the provider has no branch field at all; "" means it has
    # one, but this session recorded no branch. Keeping those states distinct
    # prevents a consumer from reading "unavailable" as "known empty".
    branch: str | None = ""
    repository: str = ""
    created_at: str = ""
    updated_at: str = ""
    content_path: str = ""  # file/dir path for rg search & content loading
    # The session that spawned this one. None means the provider exposes no
    # parent-child link at all; "" means it does and this session has none.
    # Same three-state discipline as *branch*, and for the same reason: a
    # consumer must not read "unavailable" as "known to have no parent".
    parent_id: str | None = None
    # What kind of subagent this is -- "thread_spawn", "review", "guardian",
    # or a provider's own agent name. "" when the provider says it is not a
    # subagent, None when it cannot say.
    #
    # SEPARATE FROM parent_id ON PURPOSE. 11 of the 74 Codex subagents on
    # this machine (every "review" and "guardian" one) carry no parent
    # anywhere -- not in source, not in thread_spawn_edges. "Is a subagent"
    # and "has a known parent" are different facts, and collapsing them
    # would file those 11 as ordinary sessions.
    subagent_kind: str | None = None
    # The model the session ran on, spelled as that tool's own --model flag
    # takes it: "gpt-6.1-sol" for Codex, "opencode-go/deepseek-v4.1-flash"
    # for Pi, "opencode/deepseek-v4.1-flash-free" for OpenCode 2.x. None when
    # it is not recorded anywhere discovery reads -- Claude writes it only on
    # assistant records past the head, OpenCode 1.x only per message -- so
    # None is "not known", never "no model".
    model: str | None = None
    # One conversation in two files. When Claude Code resumes a session into
    # a new one, it copies the history into the new file and writes a
    # "continued-in" record at the end of the old one -- the only place the
    # link exists. continued_in is set on the old session, continued_from on
    # the new, as raw ids of the same provider, like parent_id. None means no
    # record of a link, not proof there is none.
    #
    # These three are init=False on purpose. A defaulted init=False field is
    # never assigned in __init__ -- reads fall through to the class attribute
    # -- so a session that never has one costs nothing to construct. As
    # ordinary fields they put 6 opcodes on every Session built, which the
    # loop.codex_db_rows probe caught on a provider that never sets them.
    continued_in: str | None = field(default=None, init=False)
    continued_from: str | None = field(default=None, init=False)
    # True when a running process holds this session, so its transcript may
    # still grow and any count or ending read from it is as of this read.
    # False when no running process claims it. None where the provider keeps
    # no record of its running processes (Codex, OpenCode, Pi) or for a
    # Claude subagent, whose process is its root's.
    live: bool | None = field(default=None, init=False)
    # True when *summary* is the agent's own recap of the session rather than
    # its opening prompt: Claude Code writes one (a system record, subtype
    # away_summary) for a person coming back to the terminal, and when it is
    # the last thing in the conversation it says where things stand better
    # than the first prompt does. Never produced by this browser; it only
    # reads what the agent wrote.
    summary_is_recap: bool = field(default=False, init=False)
    # How the session's last turn ended: one of ENDINGS, read off the end of
    # its transcript, or None when that cannot be said -- a provider that
    # records too little (OpenCode 1.x), a transcript with no conversation in
    # it, or a Codex session whose rollout has not been read yet (see
    # fill_endings). As of the read, like every count: a live session that
    # is mid-action now may finish.
    ending: str | None = field(default=None, init=False)
    # A rewound Codex thread's EARLIER rollout files, oldest first; the newest
    # stays content_path, because that is the file Codex goes on appending to
    # and the one its index names. Codex's Esc-Esc rewind starts a new file
    # for the same thread, rollout-<ts>-<thread>_<segment>.jsonl, holding only
    # the records written after the rewind; its history_base says where in
    # the previous file the history it continues from ends. Every reader of a
    # transcript -- parser, prefilter, deletion -- has to see the whole chain,
    # or text said before the rewind is unreachable. init=False for the same
    # reason as the three above: almost no session has one.
    segments: tuple[str, ...] = field(default=(), init=False)

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
    """Discover Claude Code sessions, and their subagents, from JSONL files.

    A session is ``<project>/<session-id>.jsonl``. Its subagents are separate
    transcripts under ``<project>/<session-id>/subagents/``: flat for the Agent
    tool at any depth, one level further down in ``workflows/<run-id>/`` for
    the Workflow tool. Each carries an ``agent-<id>.meta.json`` sidecar, and
    that sidecar -- about 140 bytes -- is where the link comes from, so linking
    reads no transcript it would not have read anyway.
    """
    sessions: list[Session] = []
    root = Path.home() / ".claude" / "projects"
    if not root.is_dir():
        return sessions
    memo = scan_memo.open_memo("claude")
    for project_dir in root.iterdir():
        if not project_dir.is_dir():
            continue
        # One pass over the directory yields the transcripts and the
        # per-session directories together; only the latter can hold
        # subagents, and most hold none (tool-results only, or nothing).
        transcripts: list[Path] = []
        session_dirs: list[Path] = []
        try:
            with os.scandir(project_dir) as it:
                for entry in it:
                    # Hidden names skipped, as the ``*.jsonl`` glob this
                    # replaced did.
                    if entry.name.startswith("."):
                        continue
                    if entry.name.endswith(".jsonl"):
                        transcripts.append(Path(entry.path))
                    elif entry.is_dir():
                        session_dirs.append(Path(entry.path))
        except OSError as exc:
            log.warning("Skipping claude project %s: %s", project_dir.name, exc)
            continue
        for f in transcripts:
            try:
                session = _claude_session_remembered(memo, f, project_dir)
            except Exception as exc:
                log.warning("Skipping claude session %s: %s", f.name, exc)
                continue
            if session is not None:
                sessions.append(session)
        for session_dir in session_dirs:
            subagents = session_dir / "subagents"
            if not subagents.is_dir():
                continue
            for f in subagents.rglob("agent-*.jsonl"):
                try:
                    session = _claude_session_remembered(
                        memo, f, project_dir, subagent_of=session_dir.name
                    )
                except Exception as exc:
                    log.warning("Skipping claude subagent %s: %s", f.name, exc)
                    continue
                if session is not None:
                    sessions.append(session)
    if memo is not None:
        memo.save()
    _link_claude_continuations(sessions)
    live = _claude_live_session_ids()
    for s in sessions:
        if not s.subagent_kind:
            s.live = s.id in live
    return sessions


# What a per-file scan decides about a session, which is every field its
# constructor takes plus the one link the tail read can find. The rest --
# continued_from, live, segments -- is worked out across sessions afterwards,
# on every run, and is never remembered.
_SCANNED_FIELDS = tuple(f.name for f in fields(Session) if f.init)


def _scanned_fields(session: Session | None) -> dict | None:
    """*session* as the scan memo keeps it; None for a file that is no session."""
    if session is None:
        return None
    kept = {name: getattr(session, name) for name in _SCANNED_FIELDS}
    if session.continued_in is not None:
        kept["continued_in"] = session.continued_in
    if session.summary_is_recap:
        kept["summary_is_recap"] = True
    if session.ending is not None:
        kept["ending"] = session.ending
    return kept


def _session_from_scanned(kept: dict | None) -> Session | None:
    if kept is None:
        return None
    kept = dict(kept)  # the memo's own copy is written back unchanged
    continued_in = kept.pop("continued_in", None)
    recap = kept.pop("summary_is_recap", False)
    ending = kept.pop("ending", None)
    session = Session(**kept)
    if continued_in is not None:
        session.continued_in = continued_in
    if recap:
        session.summary_is_recap = True
    if ending is not None:
        session.ending = ending
    return session


def _claude_session_remembered(
    memo: scan_memo.Memo | None,
    f: Path,
    project_dir: Path,
    *,
    subagent_of: str | None = None,
) -> Session | None:
    """:func:`_claude_session`, from the scan memo while *f* is unchanged.

    Everything that function reads is in the signature: the transcript, and
    for a subagent the sidecar beside it that names its kind and its spawner.
    The project directory and the root session are both part of *f*'s path.
    """
    if memo is None:
        return _claude_session(f, project_dir, subagent_of=subagent_of)
    path = str(f)
    if subagent_of is None:
        sig = scan_memo.signature(path)
    else:
        sig = scan_memo.signature(path, path.removesuffix(".jsonl") + ".meta.json")
    if sig is None:
        return _claude_session(f, project_dir, subagent_of=subagent_of)
    known = memo.get(path, sig)
    if known is not scan_memo.MISS:
        return _session_from_scanned(known)
    session = _claude_session(f, project_dir, subagent_of=subagent_of)
    memo.put(path, sig, _scanned_fields(session))
    return session


def _link_claude_continuations(sessions: list[Session]) -> None:
    """Set continued_from on each session another one names as continued_in.

    The record exists only in the old file, so the new session learns its
    predecessor here, by inverting the link over what discovery found. A
    successor that has aged out of the tree is still named on the old
    session; it just has nothing to point back."""
    by_id = {s.id: s for s in sessions if not s.subagent_kind}
    for s in sessions:
        if s.continued_in and (successor := by_id.get(s.continued_in)):
            successor.continued_from = s.id


def _claude_live_session_ids() -> set[str]:
    """Session ids a running Claude Code process holds right now.

    Claude Code keeps one ``~/.claude/sessions/<pid>.json`` per running
    process, naming the session it is writing, and removes it on exit. A
    crash can leave one behind, so the pid must also still be running. A few
    files of a few hundred bytes, read once per discovery."""
    live: set[str] = set()
    registry = Path.home() / ".claude" / "sessions"
    try:
        entries = list(registry.glob("*.json"))
    except OSError:
        return live
    for f in entries:
        try:
            with open(f) as fh:
                record = json.load(fh)
            pid, sid = record.get("pid"), record.get("sessionId")
        except (OSError, ValueError, AttributeError):
            continue
        if not isinstance(pid, int) or not isinstance(sid, str) or pid <= 0:
            continue
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            continue
        except PermissionError:
            pass  # running, under another user
        except OSError:
            continue
        live.add(sid)
    return live


def _claude_subagent_link(f: Path, root_session: str) -> tuple[str, str, str]:
    """``(parent_id, subagent_kind, label)`` for the subagent transcript *f*.

    The parent is the session directory the transcript sits under, unless the
    sidecar names a spawning agent: a grandchild is stored flat beside the
    agent that spawned it, and only ``parentAgentId`` says so. Never taken from
    the records' ``sessionId``, which names the root session at every depth.

    A missing or unreadable sidecar still leaves a subagent -- the directory
    says so -- linked to its root session, with a generic kind.
    """
    meta: dict = {}
    try:
        with open(f.with_name(f.name.removesuffix(".jsonl") + ".meta.json")) as fh:
            loaded = json.load(fh)
        if isinstance(loaded, dict):
            meta = loaded
    except (OSError, ValueError) as exc:
        log.warning("claude subagent %s has no readable sidecar: %s", f.name, exc)
    spawner = meta.get("parentAgentId")
    parent = f"agent-{spawner}" if isinstance(spawner, str) and spawner else ""
    kind = meta.get("agentType")
    label = meta.get("description")
    return (
        parent or root_session,
        kind if isinstance(kind, str) and kind else "subagent",
        label if isinstance(label, str) else "",
    )


def claude_root_session(session: Session) -> str | None:
    """The top-level session a Claude subagent ran under, or None.

    Read from the path, ``<project>/<root>/subagents/.../agent-<id>.jsonl``,
    because that is the one thing a child, a grandchild and a workflow agent
    have in common; ``parent_id`` names only the direct spawner. Claude Code
    cannot resume a subagent, so this is also what "resume" means for one.
    """
    if session.provider != "claude" or not session.subagent_kind:
        return None
    parts = Path(session.content_path).parts
    for i in range(len(parts) - 2, 0, -1):
        if parts[i] == "subagents":
            return parts[i - 1]
    return None


def _claude_session(
    f: Path, project_dir: Path, *, subagent_of: str | None = None
) -> Session | None:
    """One Claude transcript as a Session, or None for a resume stub.

    *subagent_of* is the root session id when *f* is a subagent transcript.
    Its id is then the file stem, ``agent-<agentId>``, which is how Claude Code
    names the file and cannot collide with a session's UUID.
    """
    session_id = f.stem
    summary, cwd, branch, created_at = "", "", "", ""
    # Any earlier line's cwd, kept because the first user message
    # sometimes carries none and the decoded directory name is a
    # lossy last resort (see below).
    seen_cwd = ""
    # Whether the file holds anything to show, or anything that
    # might have been a turn before it was damaged. See the skip
    # below the loop.
    has_turn = undecodable = False
    obj = None  # still None after the loop: nothing decoded
    # Records left to read past a bare slash command looking for the first
    # real prompt; None until one is met. See _claude_summary.
    lookahead: int | None = None
    with open(f) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                undecodable = True
                continue
            if not seen_cwd:
                seen_cwd = obj.get("cwd", "")
            kind = obj.get("type")
            if lookahead is not None:
                if kind == "user" and not obj.get("isMeta"):
                    content = obj.get("message", {}).get("content", "")
                    if isinstance(content, str):
                        text, bare = _claude_summary(content)
                        if text and not bare and not _claude_command_output(content):
                            summary = text
                            break
                lookahead -= 1
                if not lookahead:
                    break
                continue
            # Extract metadata from first user message
            if kind == "user" and not obj.get("isMeta"):
                msg = obj.get("message", {})
                content = msg.get("content", "")
                cwd = obj.get("cwd", "")
                branch = obj.get("gitBranch", "")
                created_at = obj.get("timestamp", "")
                has_turn = True
                if isinstance(content, str):
                    summary, bare = _claude_summary(content)
                    if bare:
                        lookahead = _CLAUDE_PROMPT_LOOKAHEAD
                        continue
                break
            if kind in ("user", "assistant"):
                has_turn = True
    # A file Claude Code wrote with no conversation in it is not a
    # session. The common case is a session opened only to run
    # /resume: it gets a one-line "bridge-session" record, the
    # resumed conversation goes on writing to its own file, and
    # the stub is left behind. Listed, it looks exactly like the
    # session it was opened to reach (same project, same age), and
    # `claude --resume` on it fails with "No conversation found".
    # Kept, though just as empty: a file with an undecodable line,
    # or with no records at all. Either may be a transcript that
    # was damaged or never finished being written, not one that
    # was complete and empty, and the zero-entries warning exists
    # to flag exactly that.
    if obj is not None and not has_turn and not undecodable:
        return None
    # Decode project path from dir name. Claude Code encodes "/"
    # as "-", which is lossy: a directory whose real name contains
    # a hyphen (session-browser, feed-finder-chrome) decodes into
    # extra separators and yields a path that matches nothing, so
    # --here and --cwd silently skip the session. Hence the order:
    # the first user message's cwd, then any cwd seen earlier in
    # the file, then the decode.
    decoded_project = project_dir.name.replace("-", "/")
    resolved_cwd = cwd or seen_cwd or decoded_project
    parent_id, subagent_kind = "", ""
    if subagent_of is not None:
        parent_id, subagent_kind, label = _claude_subagent_link(f, subagent_of)
        # The spawner's description is a better label than the brief's first
        # line, which for a subagent is usually boilerplate.
        summary = label[:120] or summary
    # The tail read finds the last turn walking backwards, and a
    # continued-in record is written after it, so the walk passes it first.
    # So does a recap that nothing has been said since.
    tail: dict[str, str] = {}
    last_activity = _last_activity_iso(f, "claude", seen=tail)
    recap = "" if subagent_of is not None else claude_recap_text(tail.get("recap"))
    if recap:
        summary = _summary_text(recap)
    session = Session(
        id=session_id,
        provider="claude",
        summary=summary,
        cwd=resolved_cwd,
        branch=branch,
        repository=_cwd_project(resolved_cwd),
        created_at=created_at,
        updated_at=last_activity or created_at or _file_mtime_iso(f),
        content_path=str(f),
        parent_id=parent_id,
        subagent_kind=subagent_kind,
    )
    if "continued_in" in tail:
        session.continued_in = tail["continued_in"]
    if recap:
        session.summary_is_recap = True
    session.ending = tail.get("ending")
    return session


# Claude Code closes each recap with a pointer to where it can be switched
# off -- "(disable recaps in /config)" in 77 of the 97 transcripts here that
# hold one, "(turn off recaps in /config)" in 12 -- which says nothing about
# the session.
_RECAP_SETTING_NOTE = re.compile(r"\s*\([^()]*/config\)\s*$")


def claude_recap_text(content) -> str:
    """The text of a Claude Code recap, without its settings pointer."""
    if not isinstance(content, str):
        return ""
    return _RECAP_SETTING_NOTE.sub("", content).strip()


def scan_codex() -> list[Session]:
    """Discover Codex CLI sessions, preferring its SQLite index.

    Codex keeps its own session index in ``~/.codex/state_*.sqlite``, one
    ``threads`` row per rollout. Reading it replaces hundreds of file opens
    with one query. The whole-tree file scan remains the fallback for when the
    index is absent, renamed by an upgrade, locked or unreadable -- the DB must
    never make a session that exists on disk invisible, so the DB path is
    wrapped broadly enough that any failure of it degrades to the file scan
    rather than to zero sessions. An index that is merely incomplete costs
    only the files it lacks; see ``_codex_index_gap``.
    """
    db_path = _codex_db_path()
    if db_path is not None:
        try:
            return _scan_codex_db(db_path)
        except Exception as exc:
            # Deliberately broad: the promise above is absolute, and an
            # unexpected exception (a schema surprise, a sqlite3.Warning,
            # anything) must cost the fast path, not the sessions.
            log.warning("codex discovery via %s failed: %s", db_path.name, exc)
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


def _scan_codex_db(db_path: Path) -> list[Session]:
    """Build sessions from Codex's own ``threads`` index.

    A rollout on disk that the index has no row for is read from the file
    instead (see ``_codex_index_gap``): Codex writes the rollout before it
    records the thread, and some it never records at all, and a file the index
    has never heard of must not be dropped by discovery. The check is
    one-directional coverage, not identity -- a row whose file is gone is
    simply dropped, since the file scan could not have found it either, and an
    archived thread whose file remains in the tree costs the filter, not the
    fast path.

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
        # Normalize the optional origin while SQLite already materializes the
        # row, keeping Python's per-session construction loop at its existing
        # opcode budget. SQLite has no reverse(); rtrim(value,
        # replace(value, separator, '')) removes every trailing non-separator
        # character and therefore leaves the string through its last slash or
        # colon. The final CASE removes the transport-only .git suffix and
        # falls back to cwd before Python applies the usual final-segment rule.
        query = (
            "WITH base AS ("
            "  SELECT threads.*, "
            "         rtrim(COALESCE(git_origin_url, ''), '/') AS origin "
            "  FROM threads"
            "), tails AS ("
            "  SELECT base.*, "
            "         substr(origin, length(rtrim(origin, replace(origin, '/', ''))) + 1) "
            "           AS path_tail "
            "  FROM base"
            "), names AS ("
            "  SELECT tails.*, "
            "         substr(path_tail, "
            "                length(rtrim(path_tail, replace(path_tail, ':', ''))) + 1) "
            "           AS origin_name "
            "  FROM tails"
            ") "
            # threads.id is the thread's own id. See _scan_codex_files on why
            # the rollout's payload["session_id"] must never be used for this.
            # Subagent classification stays in SQLite for the same reason
            # the origin normalization above does: it keeps the Python
            # per-row construction loop, which loop.codex_db_rows guards, off
            # the JSON parsing. source is a bare word ("cli") on ordinary
            # threads and JSON on subagents, so every read is guarded by
            # json_valid -- json_extract raises on malformed input, and 889
            # of 963 rows on a real machine are that bare word.
            #
            # The parent is read ONLY from inside the subagent object. A
            # FORKED session also carries a parent id but has source "cli",
            # so an unguarded read would misfile forks as subagents;
            # agent-sessions guards the same way and their tests say why.
            # A spawned child with copied history has no first user message
            # of its own (the index stores ''), so it is named by the task
            # path its spawner gave it, read from source like the parent is.
            # Kept in SQL so the per-row Python loop does not change; the
            # file scan builds the same label in _codex_child_label.
            "SELECT id, rollout_path, cwd, git_branch, "
            "       CASE WHEN first_user_message = '' AND json_valid(source) "
            "         THEN COALESCE(json_extract("
            "           source, '$.subagent.thread_spawn.agent_path') || "
            "           COALESCE(' · ' || json_extract("
            "             source, '$.subagent.thread_spawn.agent_nickname'), ''), "
            "           '') "
            "         ELSE first_user_message END AS first_user_message, "
            "       CASE "
            "         WHEN origin_name = '' THEN cwd "
            "         WHEN origin_name LIKE '%.git' "
            "           THEN substr(origin_name, 1, length(origin_name) - 4) "
            "         ELSE origin_name "
            "       END AS repository_source, "
            "       created_at_ms, updated_at_ms, archived, "
            "       CASE WHEN json_valid(source) "
            "              AND json_type(source, '$.subagent') IS NOT NULL "
            "         THEN COALESCE("
            "           CASE WHEN json_type(source, '$.subagent') = 'text' "
            "                THEN json_extract(source, '$.subagent') END, "
            "           CASE WHEN json_type(source, '$.subagent.thread_spawn') "
            "                       IS NOT NULL THEN 'thread_spawn' END, "
            "           json_extract(source, '$.subagent.other'), "
            "           'subagent') "
            "         ELSE '' END AS subagent_kind, "
            "       CASE WHEN json_valid(source) THEN json_extract("
            "         source, '$.subagent.thread_spawn.parent_thread_id') "
            "       END AS parent_thread_id, "
            # The latest model the thread ran on (a thread that switched
            # mid-session records the last). '' on rows older Codex wrote,
            # which is "not recorded", so NULL in SQL keeps the Python row
            # loop to a plain read.
            "       {MODEL} AS model "
            "FROM names"
        )
        try:
            rows = conn.execute(
                query.replace("{MODEL}", "NULLIF(model, '')")
            ).fetchall()
        except sqlite3.OperationalError as exc:
            # model is migration-added. An index from before it still serves
            # every other column, so its absence must not cost the fast path;
            # SQLite rejects the first attempt while preparing, before any
            # row is read.
            if "no such column: model" not in str(exc):
                raise
            rows = conn.execute(query.replace("{MODEL}", "NULL")).fetchall()
    finally:
        conn.close()
    # realpath both sides: the index may store resolved paths while the
    # walk starts from a symlinked home (or vice versa), and the two must
    # not disagree for a reason that has nothing to do with the data.
    realpath = _realpath_by_directory()
    on_disk = {
        realpath(os.fspath(logical)): actual
        for logical, actual in _codex_rollout_files().items()
    }
    file_paths = on_disk.keys()
    row_paths = [realpath(r["rollout_path"]) if r["rollout_path"] else "" for r in rows]
    rows_by_path = {
        path: r for path, r in zip(row_paths, rows, strict=True) if r["rollout_path"]
    }
    missing = file_paths - rows_by_path.keys()
    # The index goes on naming a rollout by its .jsonl path after Codex has
    # compressed it, so the row is matched on that spelling and then pointed
    # at whichever file is actually there.
    sessions = []
    for r, row_path in zip(rows, row_paths, strict=True):
        if not r["rollout_path"] or r["archived"]:
            continue
        actual = on_disk.get(row_path)
        if actual is None:
            continue
        s = _codex_session_from_row(r)
        if actual.name.endswith(compressed.SUFFIX):
            s.content_path = r["rollout_path"] + compressed.SUFFIX
        sessions.append(s)
    if missing:
        gap = _codex_attach_segments(sessions, [on_disk[p] for p in missing], rows)
        if gap:
            sessions.extend(_codex_index_gap(sessions, gap))
    return sessions


def _realpath_by_directory() -> Callable[[str], str]:
    """``os.path.realpath``, resolving each directory once.

    realpath lstats every component of every path it is given. Rollouts sit
    a few dozen to a directory under ``sessions/<year>/<month>/<day>/``, so
    resolving 1,448 of them twice over -- once from the walk, once from the
    index, and the index's a second time in the loop below -- was 34,802
    lstats, 75 ms of a 156 ms scan (2026-11-14). Here a directory is resolved
    the first time it is met and each file costs the one lstat that says
    whether it is itself a link. Anything but an ordinary last component is
    handed to realpath unchanged.
    """
    directories: dict[str, str] = {}

    def resolve(path: str) -> str:
        head, tail = os.path.split(path)
        if tail in ("", os.curdir, os.pardir):
            return os.path.realpath(path)
        real_head = directories.get(head)
        if real_head is None:
            real_head = directories[head] = os.path.realpath(head)
        joined = os.path.join(real_head, tail)
        return os.path.realpath(joined) if os.path.islink(joined) else joined

    return resolve


_UUID = r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}"
# rollout-<local time>-<thread id>[_<segment id>].jsonl[.zst]
_CODEX_ROLLOUT_NAME = re.compile(
    rf"rollout-\d{{4}}-\d\d-\d\dT\d\d-\d\d-\d\d-({_UUID})(?:_({_UUID}))?\.jsonl",
    re.IGNORECASE,
)


def codex_rollout_ids(name: str) -> tuple[str, str] | None:
    """(thread id, segment id) from a rollout's file name; the segment id is
    '' for a thread's first file. None for a name of another shape.

    The name is the one place a rewind segment says what it is without being
    opened: its session_meta carries the thread's own id, no forked_from_id,
    and a history_base that otherwise reads exactly like a fork's."""
    match = _CODEX_ROLLOUT_NAME.match(name)
    if match is None:
        return None
    return match.group(1), match.group(2) or ""


def codex_segment_key(path: str | Path) -> str:
    """Sort key putting one thread's rollout files in the order they began.

    The segment id, or the thread id for the first file: both are UUIDv7, so
    their text sorts by creation time in UTC. The name's leading timestamp is
    local time and goes backwards for an hour each autumn."""
    name = os.path.basename(path)
    ids = codex_rollout_ids(name)
    return (ids[1] or ids[0]).lower() if ids is not None else name


def _codex_attach_segments(
    indexed: list[Session], unindexed: list[Path], rows
) -> list[Path]:
    """Hand a rewound thread's other files to its session; return the rest.

    Codex's index names only one rollout per thread, so every other file of
    a rewound thread looks unindexed, and the gap reader used to open each
    one -- a 39 MB first file decompressed, on every discovery -- only to drop
    it for having an id already known. A file is attached by its name alone,
    to the session of the thread it names, when that thread has a segment
    among its files. The newest becomes content_path whichever file the index
    names: a rewind lands on disk before the index catches up with it.

    A file of an archived thread is dropped, as the thread's own row is.
    """
    by_id: dict[str, Session] | None = None
    archived: set[str] | None = None
    files: dict[str, list[Path]] = {}
    rest: list[Path] = []
    for path in unindexed:
        ids = codex_rollout_ids(path.name)
        if ids is None:
            rest.append(path)
            continue
        if by_id is None:
            by_id = {s.id: s for s in indexed}
        if ids[0] in by_id:
            files.setdefault(ids[0], []).append(path)
            continue
        if archived is None:
            archived = {r["id"] for r in rows if r["archived"]}
        if ids[0] not in archived:
            rest.append(path)
    for thread_id, extra in files.items():
        owner = by_id[thread_id]
        everything = [Path(owner.content_path), *extra]
        named = [codex_rollout_ids(p.name) for p in everything]
        if not any(ids is not None and ids[1] for ids in named):
            # Not a rewind: a second plain file for a known thread. Left to
            # the gap reader, which drops it for its known id, as before.
            rest.extend(extra)
            continue
        everything.sort(key=codex_segment_key)
        newest = everything.pop()
        if str(newest) != owner.content_path:
            owner.content_path = str(newest)
            try:
                written = _epoch_ms_to_iso(
                    newest.stat().st_mtime_ns // 1_000_000, zulu=True
                )
            except OSError:
                written = ""
            # Same spelling as the index's, so the two compare as times.
            owner.updated_at = max(owner.updated_at, written)
        _codex_set_segments(owner, everything)
    return rest


def _codex_set_segments(session: Session, earlier: list[Path]) -> None:
    """Record *earlier* as *session*'s first files, and date it from the first.

    The index's created_at_ms is the newest segment's, so a rewound thread
    would otherwise claim to have started at its last rewind. Its id is a
    UUIDv7, whose top 48 bits are the creation time in milliseconds -- within
    20 ms of the first file's session_meta on every rewound thread measured --
    so the start is known without opening the first file.
    """
    session.segments = tuple(str(p) for p in sorted(earlier, key=codex_segment_key))
    started = _uuid7_iso(session.id)
    if started:
        session.created_at = started


def _uuid7_iso(value: str) -> str:
    """The creation time a version-7 UUID carries, as Codex spells times."""
    if len(value) != 36 or value[14] != "7":
        return ""
    try:
        ms = int(value[:8] + value[9:13], 16)
    except ValueError:
        return ""
    return _epoch_ms_to_iso(ms, zulu=True)


# Above this share of the tree unindexed, reading the gap from disk stops
# being cheap: at the whole tree it IS the full file scan, ~1.5 s on a real
# 1,447-rollout corpus against 86 ms for the index path.
_CODEX_GAP_WARN_SHARE = 0.1


def _codex_index_gap(indexed: list[Session], unindexed: list[Path]) -> list[Session]:
    """Sessions for the rollouts Codex's index has no row for, read from disk.

    Two kinds of gap reach here. A just-written rollout lags its row by a
    moment, and Codex also leaves some rollouts out for good: on one real
    machine 5 of 1,447 had never been indexed, the oldest for six weeks.
    Answering either by file-scanning the WHOLE tree, as this once did, put
    ~1.5 s on every call for as long as the gap lasted, which for the
    permanent kind is forever -- and no warning reads as that.

    Reading only the gap keeps the guarantee that an unindexed file is never
    dropped, at the cost of those few files. It is said once per process,
    quietly unless the gap is large enough to be slow in its own right, since
    then the index is probably broken rather than lagging.
    """
    known = {s.id for s in indexed}
    found = [
        s for s in _codex_sessions_from_files(sorted(unindexed)) if s.id not in known
    ]
    share = len(unindexed) / (len(indexed) + len(unindexed))
    names = ", ".join(p.name for p in sorted(unindexed)[:3])
    more = f" and {len(unindexed) - 3} more" if len(unindexed) > 3 else ""
    message = (
        f"codex index has no row for {len(unindexed)} rollout file(s), read "
        f"from disk instead: {names}{more}"
    )
    if share > _CODEX_GAP_WARN_SHARE:
        _warn_codex_fallback_once(f"{message} ({share:.0%} of the tree)")
    else:
        log.debug(message)
    return found


def _codex_rollout_files() -> dict[Path, Path]:
    """Every rollout under ``~/.codex/sessions``, keyed by its .jsonl name.

    The value is the file to read: the same path, or the ``.jsonl.zst`` Codex
    compressed it into (see ``compressed``). Should both exist at once, the
    plain file wins -- it is the one Codex writes to.
    """
    found: dict[Path, Path] = {}
    for p in (Path.home() / ".codex" / "sessions").rglob("rollout-*.jsonl*"):
        name = p.name
        if name.endswith(".jsonl"):
            found[p] = p
        elif name.endswith(".jsonl" + compressed.SUFFIX):
            found.setdefault(p.with_name(name.removesuffix(compressed.SUFFIX)), p)
    return found


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
        repository=_repo_name(r["repository_source"] or ""),
        created_at=_epoch_ms_to_iso(r["created_at_ms"], zulu=True),
        updated_at=_epoch_ms_to_iso(r["updated_at_ms"], zulu=True)
        or _epoch_ms_to_iso(r["created_at_ms"], zulu=True),
        content_path=r["rollout_path"],
        parent_id=r["parent_thread_id"] or "",
        subagent_kind=r["subagent_kind"],
        model=r["model"],
    )


def _codex_child_label(payload: dict) -> str:
    """'<agent_path> · <nickname>' for a spawned child, as the index query
    builds it, or '' when the rollout records neither."""
    spawn = payload.get("source")
    spawn = spawn.get("subagent") if isinstance(spawn, dict) else None
    spawn = spawn.get("thread_spawn") if isinstance(spawn, dict) else None
    if not isinstance(spawn, dict) or not isinstance(spawn.get("agent_path"), str):
        return ""
    nickname = spawn.get("agent_nickname")
    if isinstance(nickname, str):
        return _summary_text(f"{spawn['agent_path']} · {nickname}")
    return _summary_text(spawn["agent_path"])


def _codex_subagent(payload: dict) -> tuple[str, str]:
    """(subagent_kind, parent_id) from a rollout's session_meta payload.

    The mirror of the CASE expression in _scan_codex_db, for the fallback
    scan. ``source`` is a bare string on an ordinary session and an object on
    a subagent, in three observed shapes::

        {"subagent": "review"}                                  a bare kind
        {"subagent": {"thread_spawn": {"parent_thread_id": …}}}  the linked one
        {"subagent": {"other": "guardian"}}                      the catch-all

    ``other`` is Codex's catch-all variant, so an unrecognised object falls
    back to its first key rather than to "not a subagent" -- a new shape
    should surface as an oddly-named subagent, never as an ordinary session.

    The parent is read only from inside ``subagent``. A forked session carries
    a top-level parent id with source "cli", and reading that unconditionally
    would file forks as subagents.
    """
    source = payload.get("source")
    if not isinstance(source, dict):
        return "", ""
    sub = source.get("subagent")
    if sub is None:
        return "", ""
    if isinstance(sub, str):
        return sub, ""
    if not isinstance(sub, dict):
        return "subagent", ""
    spawn = sub.get("thread_spawn")
    if isinstance(spawn, dict):
        parent = spawn.get("parent_thread_id")
        return "thread_spawn", parent if isinstance(parent, str) else ""
    other = sub.get("other")
    if isinstance(other, str):
        return other, ""
    return (next(iter(sub), "subagent"), "")


def _scan_codex_files() -> list[Session]:
    """Discover Codex CLI sessions from dated JSONL tree."""
    root = Path.home() / ".codex" / "sessions"
    if not root.is_dir():
        return []
    return _codex_sessions_from_files(_codex_rollout_files().values())


def _codex_sessions_from_files(files: Iterable[Path]) -> list[Session]:
    """One Session per readable rollout in *files*, each read from disk.

    The whole of the file scan, and also how the index path covers the few
    rollouts Codex never indexed. A file that cannot be read is logged and
    skipped rather than allowed to cost the others.

    A rewound thread's files make one session, read from its newest file as
    the index would describe it (see ``_codex_rewound_threads``)."""
    files, chains = _codex_rewound_threads(files)
    sessions: list[Session] = []
    for f in files:
        try:
            session = _codex_session_from_file(f)
        except Exception as exc:
            log.warning("Skipping codex session %s: %s", f.name, exc)
            continue
        if session is not None:
            sessions.append(session)
    for chain in chains:
        try:
            session = _codex_session_from_chain(chain)
        except Exception as exc:
            log.warning("Skipping codex session %s: %s", chain[-1].name, exc)
            continue
        if session is not None:
            sessions.append(session)
    return sessions


def _codex_rewound_threads(
    files: Iterable[Path],
) -> tuple[list[Path], list[list[Path]]]:
    """(plain files, rewound threads) -- each thread its files, oldest first.

    A thread counts as rewound when one of its files is a segment
    (``rollout-<ts>-<thread>_<segment>``); every file named for that thread
    then belongs to it, in the order ``codex_segment_key`` gives."""
    named = [(f, codex_rollout_ids(f.name)) for f in files]
    rewound = {ids[0] for _f, ids in named if ids is not None and ids[1]}
    if not rewound:
        return [f for f, _ids in named], []
    plain: list[Path] = []
    threads: dict[str, list[Path]] = {}
    for f, ids in named:
        if ids is not None and ids[0] in rewound:
            threads.setdefault(ids[0], []).append(f)
        else:
            plain.append(f)
    chains = [sorted(group, key=codex_segment_key) for group in threads.values()]
    return plain, chains


def _codex_session_from_chain(chain: list[Path]) -> Session | None:
    """One session for a rewound thread's files, as the index describes it:
    the newest file's metadata, the thread's first prompt as the summary, and
    the thread's own start as created_at."""
    session = _codex_session_from_file(chain[-1])
    if session is None or len(chain) == 1:
        return session
    first = chain[0]
    ids = codex_rollout_ids(first.name)
    if ids is not None and not ids[1]:
        # The thread's own first file survives, so its first prompt is the
        # thread's -- the newest file's is whatever was sent after a rewind.
        session.summary = _codex_first_user_message(compressed.readable(first))
    _codex_set_segments(session, chain[:-1])
    return session


def _codex_session_from_file(f: Path) -> Session | None:
    """A Session from one rollout's own lines, or None if it opens with no
    ``session_meta`` record."""
    readable = compressed.readable(f)
    with open(readable) as fh:
        first_line = fh.readline().strip()
    if not first_line:
        return None
    obj = json.loads(first_line)
    if obj.get("type") != "session_meta":
        return None
    payload = obj.get("payload", {})
    # payload["id"], never payload["session_id"] -- they are different
    # fields and on a subagent rollout the latter is the PARENT's uuid.
    # Measured on this machine 2026-09-08: of 74 Codex subagent
    # rollouts, 71 had session_id != id, and in every one session_id
    # named the parent. Reading it would file a child under its
    # parent's identity, so `codex resume` on the child would resume
    # the parent and search would report one session twice. The same
    # rule holds for the DB fast path, which selects threads.id.
    # agent-sessions shipped that bug and documents the fix; we are
    # correct here by one identifier, which is why this says so.
    sid = payload.get("id", f.stem)
    kind, parent_id = _codex_subagent(payload)
    cwd = payload.get("cwd", "")
    git = payload.get("git", {})
    branch = git.get("branch", "")
    origin_url = git.get("repository_url", "")
    ts = payload.get("timestamp", obj.get("timestamp", ""))
    if "subagent_history_start_ordinal" in payload:
        # A child whose rollout opens with its spawner's history: the
        # first user message in the file is the ROOT's prompt, not
        # this session's. Named as the index names it instead.
        summary, model = _codex_child_label(payload), None
    else:
        summary, model = _codex_head(readable)
    tail: dict[str, str] = {}
    session = Session(
        id=sid,
        provider="codex",
        summary=summary,
        cwd=cwd,
        branch=branch,
        repository=(
            _codex_repo_name(cwd, origin_url) if origin_url else _repo_name(cwd)
        ),
        created_at=ts,
        updated_at=(
            _last_activity_iso(readable, "codex", seen=tail) or ts or _file_mtime_iso(f)
        ),
        content_path=str(f),
        parent_id=parent_id,
        subagent_kind=kind,
        model=model,
    )
    session.ending = tail.get("ending")
    return session


# The Codex record vocabulary is shared by this module's summary/activity
# scans and by the parser in transcript.py, and it lives here because
# transcript.py already imports from discovery — the reverse would be a cycle.
# The three-era story it encodes is written out above the Codex parser.
#
# Enumerated from every tag that opens a role=user response item across a
# 784-rollout corpus, not guessed. Two tags found there are deliberately
# absent: ``<image>`` prefixes a genuine turn ("<image name=[Image #1]>…Why am
# I seeing dupes…"), and dropping it would lose the question with it; an
# ``<INSTRUCTIONS>`` block only ever appears inside the AGENTS.md preamble.
_CODEX_INJECTED_TAGS = frozenset(
    {
        "codex_internal_context",
        "environment_context",
        "heartbeat",
        "recommended_plugins",
        "skill",
        "turn_aborted",
        "user_action",
        "user_instructions",
        "user_shell_command",
    }
)
_CODEX_AGENTS_PREAMBLE = "# AGENTS.md instructions"


def _codex_injected(text: str) -> bool:
    """True when a role=user response item is injected context, not speech."""
    stripped = text.lstrip()
    if not stripped.startswith("<"):
        return stripped.startswith(_CODEX_AGENTS_PREAMBLE)
    # The tag can carry attributes — ``<codex_internal_context source="goal">``
    # — so the name ends at the first space when one precedes the bracket.
    close = stripped.find(">", 1)
    if close < 0:
        return False
    space = stripped.find(" ", 1)
    if 0 < space < close:
        close = space
    return stripped[1:close] in _CODEX_INJECTED_TAGS


def _codex_parts_text(content, part_type: str) -> str:
    """Join the *part_type* text parts of a Codex content list.

    Each era spells the part differently — ``input_text`` on a response item,
    ``text`` on a TurnItem, ``output_text`` on the assistant side — so the
    spelling is the caller's to name.
    """
    if not isinstance(content, list):
        return ""
    return "".join(
        c.get("text", "")
        for c in content
        if isinstance(c, dict) and c.get("type") == part_type
    )


def _summary_text(text: str) -> str:
    """One line of at most 120 characters, for a session's summary field."""
    return text[:120].replace("\n", " ")


# Claude Code records a slash command as an ordinary user message made of
# markup -- <command-name>/goal</command-name> <command-message>goal
# </command-message> <command-args>...</command-args> -- and a paste as the
# pasted text wrapped in <pasted_content id="..."> tags. Used verbatim as the
# summary, either one fills the list, the TUI and the delete confirmation with
# XML: 130 of 295 Claude sessions on one machine (2026-11-01), most of them
# sessions that a /clear or /goal opened.
_CLAUDE_COMMAND_HEADS = ("<command-name>", "<command-message>")
_CLAUDE_OUTPUT_HEADS = ("<local-command-stdout>", "<local-command-stderr>")
_PASTE_TAG = re.compile(r"</?pasted_content\b[^>]*>")
# How far past a bare command (/clear, /model with no arguments) to look for
# the prompt that follows it. Measured shape: the command, its stdout record
# and a meta caveat, then the prompt; a session that never gets one stops at
# end of file, so this only bounds a pathological run of non-prompt records.
_CLAUDE_PROMPT_LOOKAHEAD = 40


def _claude_tag(text: str, tag: str) -> str:
    start = text.find(f"<{tag}>")
    if start < 0:
        return ""
    start += len(tag) + 2
    end = text.find(f"</{tag}>", start)
    return " ".join(text[start : end if end >= 0 else len(text)].split())


def _claude_command_output(text: str) -> bool:
    return text.lstrip().startswith(_CLAUDE_OUTPUT_HEADS)


def _claude_summary(content: str) -> tuple[str, bool]:
    """A readable summary of one Claude user message, and whether it is a
    bare slash command -- one with no arguments, which says too little to name
    a session by when a real prompt follows it.

    A command reads as typed (``/goal work the open beads``), a paste as its
    text without the wrapper. Everything else is the message as it always was.
    The cleaning happens before the 120-character cut: cutting first is what
    left summaries ending ``<command-args></com``.
    """
    head = content.lstrip()
    if head.startswith(_CLAUDE_COMMAND_HEADS):
        name = _claude_tag(head, "command-name")
        if name:
            args = _claude_tag(head, "command-args")
            return _summary_text(f"{name} {args}" if args else name), not args
    elif head.startswith("<pasted_content"):
        return _summary_text(" ".join(_PASTE_TAG.sub(" ", content).split())), False
    return _summary_text(content), False


def _repo_name(cwd: str) -> str:
    """Final path segment used when a provider exposes no better project name.

    Callers choose the path: OpenCode passes its recorded project worktree,
    while Claude, Pi and origin-missing Codex sessions pass their cwd. Codex
    origin metadata is normalized separately. This fallback is string-only on
    purpose: deleted worktrees still retain a usable name, and discovery never
    walks ``.git`` files or invokes Git per session.

    A trailing slash is trimmed first, so ``/a/b/`` and ``/a/b`` agree.
    """
    trimmed = cwd.rstrip("/")
    return trimmed[trimmed.rfind("/") + 1 :]


def worktree_project(cwd: str) -> str:
    """The project a worktree checkout belongs to, by where it lives, or ''.

    Worktrees are kept in a directory named ``worktrees``, and the layouts in
    use put the project's name at a known place under it:

        ~/Projects/worktrees/<project>/<checkout>    a worktrees.sh layout
        ~/.herdr/worktrees/<project>/<checkout>      Herdr's
        <repo>/.claude/worktrees/<checkout>          Claude Code's own
        ~/.codex/worktrees/<id>/<project>            the Codex app's

    A session run in one is otherwise named after the checkout's directory --
    a branch name or a generated one like worktree-quiet-cloud-12cd -- and
    filed apart from its project. Every layout needs its checkout directory
    present, so a session run in a worktrees directory itself, or in one
    project's folder of them, is not taken for one. String-only, like
    ``_repo_name``: a worktree is often long deleted, and discovery never
    walks ``.git``.
    """
    if "/worktrees/" not in cwd:
        return ""
    parts = cwd.rstrip("/").split("/")
    for i in range(len(parts) - 2, 0, -1):
        if parts[i] != "worktrees":
            continue
        parent = parts[i - 1]
        if parent == ".claude":
            return parts[i - 2] if i >= 2 else ""
        if len(parts) - i < 3:
            continue  # <project> with no checkout under it
        if parent == ".codex":
            return parts[i + 2]
        return parts[i + 1]
    return ""


def _cwd_project(cwd: str) -> str:
    """A project name from a session directory alone: its worktree's project
    when it is one, else the directory's own name."""
    return worktree_project(cwd) or _repo_name(cwd)


def _codex_repo_name(cwd: str, origin_url: str) -> str:
    """Best available Codex project name, without filesystem discovery.

    Codex records the Git origin in both its SQLite index and every rollout's
    first-line metadata. Its final path segment names the parent project even
    when *cwd* is a disposable worktree with an agent/tool name. The field is
    optional, so old/non-Git sessions retain the cwd-derived value rather than
    disappearing from ``--repo``. This deliberately differs from Claude and
    Pi, whose formats expose no project root or origin; their honest fallback
    remains the cwd name.

    The common ``.git`` transport suffix is not part of the project name.
    ``_repo_name`` handles HTTPS, SSH and scp-like remote spellings because all
    place the repository after the last slash; the colon-only form is covered
    for completeness.
    """
    remote = (
        origin_url.rstrip("/").rsplit("/", 1)[-1].rsplit(":", 1)[-1]
    ).removesuffix(".git")
    return remote or _repo_name(cwd)


def _codex_first_user_message(path: Path) -> str:
    """Extract first user message from a codex JSONL for the summary."""
    return _codex_head(path)[0]


def _codex_head(path: Path) -> tuple[str, str | None]:
    """(summary, model) from a rollout's head: the first user message, and the
    model of a ``turn_context`` the scan passes on the way to it.

    All three eras, for the same reason ``_is_turn`` covers all three: a
    paginated rollout carries no ``user_message`` event, so this returned ""
    for 171 of 784 real sessions -- after reading every one of them to the
    end, 106 MB to produce a blank.

    The model is only what the scan reads anyway. Measured 2026-11-14 over
    1,453 rollouts: 757 of the 1,040 with a user turn record a turn_context
    above it, and the scan reads the 413 without one to the end regardless,
    304 of which carry one. The other 283 keep model None rather than pay for
    reading past their first turn. It is the thread's FIRST model, where the
    index records its latest.
    """
    model: str | None = None
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
                # Codex wraps both events and raw model records in an
                # envelope; only the oldest rollouts put the record itself at
                # the top level.
                otype = obj.get("type")
                if otype == "turn_context":
                    if model is None:
                        context = obj.get("payload")
                        if isinstance(context, dict):
                            found = context.get("model")
                            if isinstance(found, str) and found:
                                model = found
                    continue
                if otype in ("event_msg", "response_item"):
                    payload = obj.get("payload", {})
                else:
                    payload = obj
                if not isinstance(payload, dict):
                    continue
                ptype = payload.get("type", "")
                if ptype == "user_message":
                    msg = payload.get("message", "")
                    if isinstance(msg, str) and msg:
                        return _summary_text(msg), model
                    content = payload.get("content", "")
                    if isinstance(content, str) and content:
                        return _summary_text(content), model
                    if isinstance(content, list):
                        for part in content:
                            if (
                                isinstance(part, dict)
                                and part.get("type") == "input_text"
                            ):
                                return _summary_text(part.get("text", "")), model
                elif ptype == "item_completed":
                    item = payload.get("item")
                    if isinstance(item, dict) and item.get("type") == "UserMessage":
                        text = _codex_parts_text(item.get("content"), "text")
                        if text:
                            return _summary_text(text), model
                elif otype == "response_item" and ptype == "message":
                    if payload.get("role") != "user":
                        continue
                    # The fallback record, and the only one an era-less
                    # rollout has. Injected context is skipped here for the
                    # same reason the parser skips it: a summary reading
                    # "# AGENTS.md instructions for ..." names every session
                    # in the repository identically.
                    text = _codex_parts_text(payload.get("content"), "input_text")
                    if text and not _codex_injected(text):
                        return _summary_text(text), model
    except Exception as exc:
        # Best-effort summary: an unreadable file costs a blank summary, not
        # the session. Logged at debug so it is recoverable when one is blank.
        log.debug("No codex summary from %s: %s", path.name, exc)
    return "", model


def scan_opencode() -> list[Session]:
    """Discover opencode sessions from its SQLite database."""
    sessions: list[Session] = []
    db_path = _opencode_db_path()
    if not db_path.exists():
        return sessions
    try:
        conn = sqlite3.connect(_readonly_uri(db_path), uri=True, timeout=3)
        conn.row_factory = sqlite3.Row
        try:
            sql, schema, missing = _OPENCODE_SESSIONS_SQL, 0, frozenset()
            while True:
                try:
                    cur, schema = opencode_execute(conn, sql, schema=schema)
                    rows = cur.fetchall()
                    break
                except sqlite3.OperationalError as exc:
                    # Rejected while preparing, before any row is read.
                    column = _opencode_missing_column(str(exc), missing)
                    if column is None:
                        raise
                    missing |= {column}
                    sql = (sql[0], _opencode_v2_sessions_sql(missing))
                    schema = OPENCODE_V2
        finally:
            conn.close()
        for r in rows:
            session = Session(
                id=r["id"],
                provider="opencode",
                summary=r["title"] or "",
                cwd=r["directory"] or r["worktree"] or "",
                # OpenCode's session/project schema has no branch column.
                # Do not serialize that structural absence as a known-
                # empty string, which is what Claude/Codex legitimately
                # record for a session with no active branch.
                branch=None,
                # The project table has a name column and it is NULL for
                # every row in a real install, so the path is what there
                # is. Preferred over it anyway when set: it is the name
                # the user gave the project.
                #
                # Worktree first, since it is the project root and so
                # still names the repository for a session started in a
                # subdirectory of it. It is "/" for opencode's catch-all
                # "global" project, which names nothing, and those
                # sessions fall through to their own directory.
                repository=r["project_name"]
                or _repo_name(r["worktree"] or "")
                or _repo_name(r["directory"] or ""),
                created_at=_epoch_ms_to_iso(r["time_created"]),
                updated_at=_epoch_ms_to_iso(r["time_updated"]),
                content_path=str(db_path),
                parent_id=r["parent_id"] or "",
                subagent_kind=((r["agent"] or "subagent") if r["parent_id"] else ""),
                model=r["model"],
            )
            if r["idle_outcome"] is not None:
                session.ending = _OPENCODE_ENDINGS.get(r["idle_outcome"])
            sessions.append(session)
    except (sqlite3.Error, OSError) as exc:
        log.warning("opencode discovery failed: %s", exc)
    return sessions


def _opencode_missing_column(message: str, missing: frozenset[str]) -> str | None:
    """The optional session_v2 column *message* says is missing, if it is one
    not already read as NULL."""
    if "no such column" not in message:
        return None
    for column in _OPENCODE_V2_OPTIONAL:
        if column not in missing and column in message:
            return column
    return None


def _opencode_db_path() -> Path:
    return Path.home() / ".local" / "share" / "opencode" / "opencode.db"


# OpenCode keeps its sessions in one SQLite file whose schema changed at v2.
# v1 has session / message / part, a part row per piece of a message. v2 (2.x,
# installed from opencode.ai/v2) has session_v2 / session_message, one row per
# message with every piece inside its JSON, and none of the v1 tables. The two
# carry the same facts for discovery under different table names, so each
# statement here comes as a (v1, v2) pair and opencode_execute picks one.
OPENCODE_V1, OPENCODE_V2 = 1, 2

# Same columns from v2. fork_session_id (a fork, not a spawn) is left out on
# purpose: reading it as parent_id would file a fork as a subagent.
_OPENCODE_V2_SESSIONS = (
    "SELECT s.id, s.title, s.directory, s.time_created, s.time_updated, "
    "       s.parent_id, s.agent, {model} AS model, {idle_outcome} AS idle_outcome, "
    "       p.worktree, p.name AS project_name "
    "FROM session_v2 s "
    "LEFT JOIN project p ON s.project_id = p.id "
    "ORDER BY s.time_updated DESC"
)
# Columns migration-added to session_v2, each with what reads it. A 2.x
# install from before one still serves every other column, and losing all its
# sessions over an optional field would be the wrong trade, so a statement the
# database rejects for one is asked again with that column read as NULL.
#
# v2 records the session's model as JSON, {"id": ..., "providerID": ...,
# "variant": ...}, spelled here providerID/id as opencode's own --model takes
# it; v1 records it only per message, hence NULL there. Built in SQL so the
# row loop stays a plain read. json_valid guards it, since json_extract raises
# on malformed input.
#
# idle_outcome is how the session's last turn closed, as OpenCode itself
# records it on the row: "succeeded" or "interrupted" on all 7 v2 sessions
# here (2026-11-24), NULL while it is busy. v1 has no such column, and its
# messages are not read for one, hence NULL there too.
_OPENCODE_V2_OPTIONAL = {
    "model": (
        "CASE WHEN json_valid(s.model) "
        "      AND json_extract(s.model, '$.id') IS NOT NULL "
        "  THEN COALESCE(json_extract(s.model, '$.providerID') || '/', '') "
        "       || json_extract(s.model, '$.id') "
        "END"
    ),
    "idle_outcome": "s.idle_outcome",
}


def _opencode_v2_sessions_sql(missing: frozenset[str] = frozenset()) -> str:
    return _OPENCODE_V2_SESSIONS.format(
        **{
            column: "NULL" if column in missing else expr
            for column, expr in _OPENCODE_V2_OPTIONAL.items()
        }
    )


_OPENCODE_SESSIONS_SQL = (
    # parent_id and agent are two more columns on a query that already reads
    # this table, so the sqlite_statements counter is unmoved. agent names the
    # AGENT TYPE and is set on parents too (build 239, orchestrator 101, ... on
    # a real install), so it is a label for a child rather than evidence of
    # being one -- parent_id is the only thing that says "this was spawned".
    (
        "SELECT s.id, s.title, s.directory, s.time_created, s.time_updated, "
        "       s.parent_id, s.agent, NULL AS model, NULL AS idle_outcome, "
        "       p.worktree, p.name AS project_name "
        "FROM session s "
        "LEFT JOIN project p ON s.project_id = p.id "
        "ORDER BY s.time_updated DESC"
    ),
    _opencode_v2_sessions_sql(),
)
_OPENCODE_ENDINGS = {"succeeded": COMPLETED, "interrupted": INTERRUPTED}


def opencode_execute(
    conn: sqlite3.Connection,
    sql: tuple[str, str],
    params: tuple = (),
    schema: int = 0,
) -> tuple[sqlite3.Cursor, int]:
    """Run the statement of *sql* matching the database's schema.

    *schema* is what an earlier call on this connection returned, or 0 when
    nothing is known yet. Unknown tries v1 first and falls back to v2 only on
    a missing table, so a v1 database runs exactly the statements it always
    did. The failed attempt costs nothing measurable on a v2 one: SQLite
    rejects it while preparing, before any row is read, which is also why the
    perf gate's statement counter (a trace callback) never sees it.

    A database carrying both sets of tables mid-migration reads as v1. OpenCode
    replaces the v1 tables rather than keeping both, so that is a transient
    state, and v1 is what it held until the migration finished.
    """
    if schema == OPENCODE_V2:
        return conn.execute(sql[1], params), OPENCODE_V2
    try:
        return conn.execute(sql[0], params), OPENCODE_V1
    except sqlite3.OperationalError as exc:
        if schema or "no such table" not in str(exc):
            raise
    return conn.execute(sql[1], params), OPENCODE_V2


def _pi_model(record: dict) -> str | None:
    """'<provider>/<model>' from a Pi ``model_change`` record, as pi's own
    --model flag takes it, or the bare model id when no provider is named."""
    model_id = record.get("modelId")
    if not isinstance(model_id, str) or not model_id:
        return None
    provider = record.get("provider")
    return (
        f"{provider}/{model_id}" if isinstance(provider, str) and provider else model_id
    )


def scan_pi() -> list[Session]:
    """Discover Pi sessions from JSONL files.

    Layout is ``~/.pi/agent/sessions/<encoded-cwd>/<stamp>_<uuid>.jsonl``.
    The directory name encodes "/" as "-" exactly as Claude's does, and is
    lossy in the same way — a project called ``agent-loadout`` decodes into a
    path that exists nowhere. Nothing here decodes it: every Pi transcript
    opens with a header line carrying the real cwd, so the lossy name is never
    the best source available.
    """
    sessions: list[Session] = []
    root = Path.home() / ".pi" / "agent" / "sessions"
    if not root.is_dir():
        return sessions
    memo = scan_memo.open_memo("pi")
    for project_dir in root.iterdir():
        if not project_dir.is_dir():
            continue
        for f in project_dir.glob("*.jsonl"):
            try:
                sessions.append(_pi_session_remembered(memo, f))
            except Exception as exc:
                log.warning("Skipping pi session %s: %s", f.name, exc)
    if memo is not None:
        memo.save()
    return sessions


def _pi_session_remembered(memo: scan_memo.Memo | None, f: Path) -> Session:
    """:func:`_pi_session`, from the scan memo while *f* is unchanged."""
    if memo is None:
        return _pi_session(f)
    path = str(f)
    sig = scan_memo.signature(path)
    if sig is None:
        return _pi_session(f)
    known = memo.get(path, sig)
    if known is not scan_memo.MISS:
        return _session_from_scanned(known)
    session = _pi_session(f)
    memo.put(path, sig, _scanned_fields(session))
    return session


def _pi_session(f: Path) -> Session:
    """One Pi transcript as a Session, read off its head and its tail."""
    session_id, cwd, created_at, summary = "", "", "", ""
    model = None
    with open(f) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(obj, dict):
                continue
            otype = obj.get("type")
            if otype == "session":
                session_id = obj.get("id", "") or session_id
                cwd = obj.get("cwd", "") or cwd
                created_at = obj.get("timestamp", "") or created_at
            elif otype == "model_change" and model is None:
                # Line 2 of a Pi session, above the first user turn
                # this scan stops at, so it costs no extra read.
                model = _pi_model(obj)
            elif otype == "message":
                msg = obj.get("message")
                if not isinstance(msg, dict):
                    continue
                if msg.get("role") != "user":
                    continue
                summary = _summary_text(_pi_message_text(msg))
                # First user turn: everything this scan wants is
                # at or above it.
                break
    # The filename's uuid suffix is the same id the header line
    # carries, so a file whose header is missing or unparseable
    # still gets the id Pi itself would resume by.
    session_id = session_id or f.stem.split("_", 1)[-1]
    tail: dict[str, str] = {}
    session = Session(
        id=session_id,
        provider="pi",
        summary=summary,
        cwd=cwd,
        # Pi records no branch anywhere in its format. None,
        # not "" — see the note on the field.
        branch=None,
        repository=_cwd_project(cwd),
        created_at=created_at,
        updated_at=(
            _last_activity_iso(f, "pi", seen=tail) or created_at or _file_mtime_iso(f)
        ),
        content_path=str(f),
        model=model,
    )
    session.ending = tail.get("ending")
    return session


def _pi_message_text(msg: dict) -> str:
    """Readable text of one Pi message, ignoring non-text content parts."""
    content = msg.get("content", "")
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return ""
    parts = [
        p.get("text", "")
        for p in content
        if isinstance(p, dict) and p.get("type") == "text"
    ]
    return " ".join(t for t in parts if t)


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
# Codex's legacy vocabulary. Its paginated rollouts carry neither of these --
# see the era note above the Codex parser in transcript.py -- so a file
# written in that mode had no recognisable turn at all, and its last activity
# collapsed to created_at. Worse, it collapsed expensively: finding no turn
# means widening the tail window until the whole file has been read, which on
# the real corpus was 93 MB read across 147 rollouts to return "".
_CODEX_TURN_PAYLOADS = {"user_message", "agent_message"}
_CODEX_TURN_ITEMS = {"UserMessage", "AgentMessage"}
# Both roles: recognising only the user side would leave a session whose last
# act was the assistant's reply timestamped at its previous user turn.
_CODEX_TURN_ROLES = {"user", "assistant"}
# Pi writes one "message" line per turn and one per tool result, tagging the
# role inside. Only the two conversational roles count: "toolResult" is the
# harness answering itself, which is the same line the claude and codex rules
# draw (neither counts tool traffic as activity).
_PI_TURN_ROLES = {"user", "assistant"}


def _is_turn(obj: dict, provider: str) -> bool:
    """True if a parsed JSONL object represents a real conversation turn."""
    if provider == "claude":
        return obj.get("type") in _CLAUDE_TURN_TYPES
    if provider == "codex":
        otype = obj.get("type")
        payload = obj.get("payload")
        if not isinstance(payload, dict):
            return False
        if otype == "response_item":
            # Era-independent, and the record nearest the tail in practice.
            return (
                payload.get("type") == "message"
                and payload.get("role") in _CODEX_TURN_ROLES
            )
        if otype != "event_msg":
            return False
        ptype = payload.get("type")
        if ptype in _CODEX_TURN_PAYLOADS:
            return True
        if ptype == "item_completed":
            item = payload.get("item")
            return isinstance(item, dict) and item.get("type") in _CODEX_TURN_ITEMS
        return False
    if provider == "pi":
        if obj.get("type") != "message":
            return False
        msg = obj.get("message")
        return isinstance(msg, dict) and msg.get("role") in _PI_TURN_ROLES
    return False


def _last_activity_iso(
    path: Path, provider: str, *, window: int = 16384, seen: dict | None = None
) -> str:
    """Timestamp of the last genuine conversation turn in a JSONL transcript.

    Reads the file tail backwards so a bumped mtime (a read/sync/resume that
    touches the file, or a trailing session-open event carrying a fresh
    timestamp) can't masquerade as recent activity. Widens the read window if
    the last turn sits deeper than the initial tail. Returns "" when no
    timestamped turn is found; callers fall back to created_at, then mtime.

    *seen*, when given, collects facts recorded after the last turn, which
    this walk passes on the way to it: a Claude "continued-in" record lands
    there as ``seen["continued_in"]``, and the newest Claude Code recap (a
    system record, subtype away_summary) as ``seen["recap"]``, so a recap is
    found exactly when no turn of the conversation followed it. It also gets
    ``seen["ending"]``, how the conversation ended (see ``ENDINGS``), when the
    records this walk reads decide it: almost always the last turn itself,
    sometimes a record or two past it -- a bare slash command, a task
    notification -- which the walk then goes on to read in the window it
    already holds. It never widens the window for the ending alone, so all
    of this costs no read of its own.
    """
    try:
        size = path.stat().st_size
    except OSError:
        return ""
    if size == 0:
        return ""
    make_ending = _ENDING_CLASSIFIERS.get(provider) if seen is not None else None
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
        # Afresh for each window: a widened read starts again from the end.
        ending = make_ending() if make_ending is not None else None
        found = ""
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
            if ending is not None and ending.feed(obj):
                seen["ending"] = ending.ending
                ending = None
            if found:
                if ending is None:
                    return found
                continue
            ts = obj.get("timestamp")
            if ts and _is_turn(obj, provider):
                found = ts
                if ending is None:
                    return found
                continue
            if seen is not None:
                otype = obj.get("type")
                if otype == "continued-in":
                    target = obj.get("continuedInSessionId")
                    if isinstance(target, str) and target:
                        seen.setdefault("continued_in", target)
                elif otype == "system" and obj.get("subtype") == "away_summary":
                    # Walking backwards, so the first one met is the newest.
                    seen.setdefault("recap", obj.get("content"))
        if ending is not None and (found or read >= size) and ending.done():
            # Nothing older in reach decided it, which leaves the
            # classifier's own verdict on what it met, if it has one.
            seen["ending"] = ending.ending
        if found or read >= size:
            return found  # "" when the whole file holds no timestamped turn
        window *= 4  # last turn is deeper than the tail — widen and retry


class _Ending:
    """Decides how a transcript ended, fed its records newest first.

    ``feed`` returns True once the ending is decided, and the caller stops.
    ``done`` says whether what was fed decides it even with nothing older to
    come -- the start of the file is then the answer.
    """

    __slots__ = ("ending",)

    def __init__(self) -> None:
        self.ending: str | None = None

    def feed(self, obj: dict) -> bool:  # pragma: no cover - overridden
        raise NotImplementedError

    def done(self) -> bool:
        return self.ending is not None


# What Claude Code writes as a user message that nobody typed. Measured on the
# 609 transcripts here (2026-11-24): every ending on a user record is one of
# these, an interruption, a tool result, a slash command, or a real prompt.
_CLAUDE_HARNESS_HEADS = (
    "<task-notification>",
    "<system-reminder>",
    "<local-command-caveat>",
    "Stop hook feedback:",
    "This session is being continued",
)
_CLAUDE_REJECTED = "The user doesn't want to proceed with this tool use"


class _ClaudeEnding(_Ending):
    """Claude Code: the last assistant record that says something, or the
    last user record somebody typed, whichever is newer.

    An assistant reply arrives as one record per content block, so a record
    holding only thinking is passed over. Harness traffic is passed over too
    -- meta records, task notifications, a bare slash command and what it
    printed -- since none of it asks or answers anything; a command with
    arguments is a request typed through a command, as everywhere else.
    """

    __slots__ = ()

    def feed(self, obj: dict) -> bool:
        otype = obj.get("type")
        if otype == "assistant":
            msg = obj.get("message")
            content = msg.get("content") if isinstance(msg, dict) else None
            if isinstance(content, str):
                return self._decide(COMPLETED) if content else False
            if not isinstance(content, list):
                return False
            said = False
            for block in content:
                if not isinstance(block, dict):
                    continue
                kind = block.get("type")
                if kind == "tool_use":
                    return self._decide(MID_ACTION)
                if kind == "text" and block.get("text"):
                    said = True
            return self._decide(COMPLETED) if said else False
        if otype == "user":
            if obj.get("isMeta"):
                return False
            msg = obj.get("message")
            content = msg.get("content") if isinstance(msg, dict) else None
            if isinstance(content, str):
                return self._user_text(content)
            if not isinstance(content, list):
                return False
            for block in content:
                if not isinstance(block, dict):
                    continue
                kind = block.get("type")
                if kind == "tool_result":
                    rejected = block.get("is_error") and _tool_result_text(
                        block.get("content")
                    ).startswith(_CLAUDE_REJECTED)
                    return self._decide(INTERRUPTED if rejected else MID_ACTION)
                if kind == "text" and self._user_text(block.get("text") or ""):
                    return True
            return False
        if otype == "attachment":
            # A prompt typed while the agent was busy, queued for its next
            # turn; the parser counts a human one as a user turn.
            att = obj.get("attachment")
            if isinstance(att, dict) and att.get("type") == "queued_command":
                origin = att.get("origin")
                if isinstance(origin, dict) and origin.get("kind") == "human":
                    return self._decide(UNANSWERED)
        return False

    def _user_text(self, text: str) -> bool:
        head = text.lstrip()
        if not head:
            return False
        if head.startswith("[Request interrupted by user"):
            return self._decide(INTERRUPTED)
        if head.startswith(_CLAUDE_HARNESS_HEADS) or _claude_command_output(head):
            return False
        if head.startswith(_CLAUDE_COMMAND_HEADS) and _claude_summary(head)[1]:
            return False  # a bare /clear, /model, /effort
        return self._decide(UNANSWERED)

    def _decide(self, ending: str) -> bool:
        self.ending = ending
        return True


def _tool_result_text(content) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        for part in content:
            if isinstance(part, dict) and isinstance(part.get("text"), str):
                return part["text"]
    return ""


class _PiEnding(_Ending):
    """Pi: its last message record, which says how it stopped.

    An assistant message carries a stopReason: "stop" is an answer, "toolUse"
    a call left without its result, "aborted" a cancelled turn, and "error"
    a turn that failed -- interrupted when the error is the abort Pi reports
    as "This operation was aborted", otherwise unanswered, since the person's
    message got no reply. Measured on the 620 Pi sessions here, 2026-11-24.
    """

    __slots__ = ()

    def feed(self, obj: dict) -> bool:
        if obj.get("type") != "message":
            return False
        msg = obj.get("message")
        if not isinstance(msg, dict):
            return False
        role = msg.get("role")
        if role == "user":
            self.ending = UNANSWERED
        elif role == "toolResult":
            self.ending = MID_ACTION
        elif role == "assistant":
            stop = msg.get("stopReason")
            if stop == "aborted":
                self.ending = INTERRUPTED
            elif stop == "error":
                error = msg.get("errorMessage")
                aborted = isinstance(error, str) and "aborted" in error.lower()
                self.ending = INTERRUPTED if aborted else UNANSWERED
            elif stop == "toolUse":
                self.ending = MID_ACTION
            else:
                self.ending = self._by_content(msg.get("content"))
        return self.ending is not None

    @staticmethod
    def _by_content(content) -> str | None:
        if isinstance(content, str):
            return COMPLETED if content else None
        if not isinstance(content, list):
            return None
        kinds = {p.get("type") for p in content if isinstance(p, dict)}
        if "toolCall" in kinds:
            return MID_ACTION
        return COMPLETED if "text" in kinds else None


# Codex response items that are a tool call or its output, in every era seen.
_CODEX_CALLS = frozenset(
    {
        "function_call",
        "function_call_output",
        "custom_tool_call",
        "custom_tool_call_output",
        "local_shell_call",
        "web_search_call",
        "tool_search_call",
        "tool_search_output",
    }
)


class _CodexEnding(_Ending):
    """Codex: how its last turn closed.

    A turn that ends normally writes task_complete, and one the person
    cancels turn_aborted, so those decide it -- after task_complete the walk
    goes on to the turn's last message, because a turn can complete without a
    reply (the person's message, then task_complete: unanswered). Without
    either, the turn was cut off, and the newest call, reply or message says
    where. Measured over 1,453 rollouts here, 2026-11-24: 2,784 task_complete
    and 478 turn_aborted records, every one of the latter with reason
    "interrupted". A user shell command run after a turn closes is passed
    over, as are reasoning, token counts and the context Codex injects.
    """

    __slots__ = ("closed",)

    def __init__(self) -> None:
        super().__init__()
        self.closed = False

    def feed(self, obj: dict) -> bool:
        otype = obj.get("type")
        payload = obj.get("payload")
        if otype not in ("event_msg", "response_item"):
            payload = obj  # the oldest rollouts put the record at the top
        if not isinstance(payload, dict):
            return False
        ptype = payload.get("type")
        if otype == "event_msg":
            if ptype == "turn_aborted":
                return self._decide(INTERRUPTED)
            if ptype == "task_complete":
                self.closed = True
                return False
            if ptype == "agent_message":
                return self._decide(COMPLETED)
            if ptype == "user_message":
                return self._decide(UNANSWERED)
            if ptype == "item_completed":
                item = payload.get("item")
                kind = item.get("type") if isinstance(item, dict) else None
                if kind == "AgentMessage":
                    return self._decide(COMPLETED)
                if kind == "UserMessage":
                    return self._decide(UNANSWERED)
            return False
        if ptype == "message":
            role = payload.get("role")
            if role == "assistant":
                return self._decide(COMPLETED)
            if role == "user":
                text = _codex_parts_text(payload.get("content"), "input_text")
                if text and not _codex_injected(text):
                    return self._decide(UNANSWERED)
            return False
        if ptype in _CODEX_CALLS:
            return self._decide(COMPLETED if self.closed else MID_ACTION)
        return False

    def done(self) -> bool:
        if self.ending is None and self.closed:
            self.ending = COMPLETED
        return self.ending is not None

    def _decide(self, ending: str) -> bool:
        self.ending = ending
        return True


_ENDING_CLASSIFIERS: dict[str, Callable[[], _Ending]] = {
    "claude": _ClaudeEnding,
    "codex": _CodexEnding,
    "pi": _PiEnding,
}


# A compressed rollout is read from the start to reach its end, keeping this
# many of its last lines: the record that decides the ending is the last or
# one of the last few in every rollout here.
_CODEX_ENDING_LINES = 400


def fill_endings(sessions: Iterable[Session]) -> list[Session]:
    """Set the ending of each session discovery could not, where it can be read.

    Returns the sessions it gave an ending to.

    Codex's index path reads no rollout at all, which is what makes it fast,
    so a Codex session found through it has no ending until this asks its
    rollout. Called for the sessions a command is about to show or filter
    on, never for the whole history by default: 1,400 of the 1,453 rollouts
    here are compressed, and reading every one to its end took 1.1 s
    (2026-11-24). Each answer is remembered by the rollout's size and mtime
    beside the scan memo, so a session is read once while it stays the
    same; with the cache switched off, each is read when asked.
    """
    wanted = [
        s
        for s in sessions
        if s.ending is None and s.provider == "codex" and s.content_path
    ]
    if not wanted:
        return []
    memo = scan_memo.open_memo("codex-endings", keep_unmet=True)
    for s in wanted:
        path = s.content_path
        sig = scan_memo.signature(path) if memo is not None else None
        if sig is not None:
            known = memo.get(path, sig)
            if known is not scan_memo.MISS:
                s.ending = known
                continue
        try:
            ending = _codex_rollout_ending(Path(path))
        except Exception as exc:
            # Same rule as discovery's: one unreadable rollout costs its own
            # answer, never the others'.
            log.debug("no ending for codex session %s: %s", s.id, exc)
            continue
        if sig is not None:
            memo.put(path, sig, ending)
        s.ending = ending
    if memo is not None:
        memo.save()
    return [s for s in wanted if s.ending is not None]


def _codex_rollout_ending(path: Path) -> str | None:
    if path.name.endswith(compressed.SUFFIX):
        # A copy a scan made can be read from its end, like a plain rollout.
        path = compressed.existing_copy(path) or path
    if not path.name.endswith(compressed.SUFFIX):
        tail: dict[str, str] = {}
        _last_activity_iso(path, "codex", seen=tail)
        return tail.get("ending")
    with compressed.open_binary(path) as fh:
        last = deque(fh, maxlen=_CODEX_ENDING_LINES)
    ending = _CodexEnding()
    for raw in reversed(last):
        try:
            obj = json.loads(raw)
        except ValueError:
            continue
        if isinstance(obj, dict) and ending.feed(obj):
            return ending.ending
    return ending.ending if ending.done() else None


# ---------------------------------------------------------------------------
# Aggregate discovery
# ---------------------------------------------------------------------------

ALL_SCANNERS = {
    "claude": scan_claude,
    "codex": scan_codex,
    "opencode": scan_opencode,
    "pi": scan_pi,
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
