"""Shared transcript service for parsing/rendering/searching agent sessions.

Parsers must not truncate — display-level limits are the TUI's concern.
"""

from __future__ import annotations

# Per-session scans are I/O-bound; a small pool overlaps reads. Gains
# plateau around 4 workers (JSON parsing is GIL-bound), 8 leaves headroom.
_SEARCH_THREADS = 8

import io
import json
import logging
import multiprocessing
import os
import re
import shutil
import sqlite3
import subprocess
import sys
import threading
from bisect import bisect_right
from collections import OrderedDict, deque
from collections.abc import Callable, Iterable, Iterator
from concurrent.futures import (
    FIRST_COMPLETED,
    Future,
    ProcessPoolExecutor,
    ThreadPoolExecutor,
    wait,
)
from dataclasses import dataclass, field
from datetime import UTC, datetime
from functools import lru_cache
from itertools import accumulate
from itertools import count as _count
from pathlib import Path

from session_browser import compressed, entry_cache
from session_browser.discovery import (
    OPENCODE_V2,
    Session,
    _claude_command_output,
    _claude_summary,
    _codex_injected,
    _opencode_db_path,
    claude_recap_text,
    codex_rollout_ids,
    codex_segment_key,
    opencode_execute,
)

log = logging.getLogger(__name__)


class TranscriptUnreadable(Exception):
    """Raised when a session's transcript cannot be read."""


@dataclass(slots=True)
class TranscriptEntry:
    """One normalized turn. Slotted: a broad search constructs one of these
    per transcript entry across the whole corpus, and the four fixed fields
    cost measurably less to store in slots than in a per-instance dict. The
    generated __init__/__eq__/__repr__ and dataclasses.replace() behave
    identically; nothing sets attributes outside these four."""

    role: str
    text: str
    timestamp: str = ""
    metadata: dict | None = None


@dataclass
class Transcript:
    session: Session
    entries: list[TranscriptEntry]
    warnings: list[str] = field(default_factory=list)


def canonical_id(session: Session) -> str:
    """Return a globally-unique id for a session across all providers."""
    return f"{session.provider}:{session.id}"


# Providers whose ``content_path`` is this conversation's own JSONL
# transcript, and so is safe and cheap to scan for fork ancestors. Opencode is
# absent on purpose: its ``content_path`` is the shared ``opencode.db``, one
# multi-gigabyte SQLite file holding *every* session, and streaming that to
# recover one conversation's lineage froze the handoff key for seconds (6.6s
# against a 4.3 GB database, versus 0.08s for a claude row). The list is an
# allowlist rather than "not opencode" so that a future provider backed by a
# shared store is excluded by default instead of reintroducing that stall.
_LINEAGE_PROVIDERS = frozenset({"claude", "codex"})


def lineage_ids(provider: str, session_id: str, content_path: str) -> set[str]:
    """Every session id this conversation has lived under.

    Claude Code's ``--resume`` forks: the continued conversation is written
    under a new session id, but entries copied from the parent keep the id
    they were generated under in a top-level snake_case ``session_id`` field
    (the camelCase ``sessionId`` is rewritten to the fork's own id). Scanning
    the transcript for those values recovers the ancestor ids, so a terminal
    already running the conversation under an earlier id can still be matched
    to the row the user picked. Only top-level fields are read — ids merely
    *mentioned* in message content never make a foreign conversation look
    related.

    ``provider`` decides whether there is a transcript worth scanning at all;
    see ``_LINEAGE_PROVIDERS``. A provider that is not scanned still gets its
    selected id back, which is what matching then falls back to.

    Used by both multiplexer integrations (``tmux.py``, ``herdr.py``), which
    each have to recognise a live terminal by whatever id it was started on.
    """
    ids = {session_id}
    if provider not in _LINEAGE_PROVIDERS:
        return ids
    if not content_path or not os.path.isfile(content_path):
        return ids
    try:
        raw = compressed.open_binary(Path(content_path))
        with io.TextIOWrapper(raw, encoding="utf-8", errors="replace") as fh:
            for line in fh:
                if '"session_id"' not in line:
                    continue
                try:
                    entry = json.loads(line)
                except ValueError:
                    continue
                if isinstance(entry, dict):
                    past = entry.get("session_id")
                    if isinstance(past, str) and past:
                        ids.add(past)
    except OSError:
        pass
    return ids


_ROLE_LABELS = {
    "user": "User",
    "assistant": "Assistant",
    "tool": "Tool",
    "system": "System",
}

# Filterable roles: the four entry roles plus the "error" pseudo-role
# (failed tool calls, where the provider records a failure flag).
FILTER_ROLES = ("user", "assistant", "tool", "system", "error")


def entry_label(entry: TranscriptEntry) -> str:
    label = _ROLE_LABELS.get(entry.role, entry.role.capitalize() or "Unknown")
    if entry.role == "tool" and entry.metadata:
        kind = entry.metadata.get("kind")
        if kind == "call":
            label = "Tool call"
        elif kind == "output":
            label = "Tool output"
        if entry.metadata.get("is_error"):
            label += " (error)"
    if entry.metadata and entry.metadata.get("rewound"):
        # Undone by a Codex rewind: what happened, but not what the model
        # went on from. See the note above _codex_link_ordinal.
        label += " (rewound)"
    return label


def entry_matches_roles(entry: TranscriptEntry, roles: set[str]) -> bool:
    """True if the entry's role is in *roles*; the "error" pseudo-role
    matches tool entries whose provider recorded a failure."""
    if entry.role in roles:
        return True
    return (
        "error" in roles
        and entry.metadata is not None
        and bool(entry.metadata.get("is_error"))
    )


def render_text(
    transcript: Transcript,
    *,
    entry_indices: list[int] | None = None,
    timestamps: bool = False,
) -> str:
    """Readable, Markdown-compatible transcript body.

    *entry_indices* are the absolute positions of a role-filtered subset;
    each block is then prefixed with its position so it can be fed back to
    get --entries or matched against an entry_index from search or from
    get's JSON.

    *timestamps* puts each entry's time before its role, in UTC as the JSON
    has it: ``10:12:35Z``, with the date as well on the first entry and
    whenever the day changes. Whether a reply came before or after something
    in another session is a question about times, and without them the text
    form could not answer it. Off for the TUI, which places highlights by
    the label's length."""
    stamps = (
        _entry_stamps(transcript.entries)
        if timestamps
        else [""] * len(transcript.entries)
    )
    if entry_indices is None:
        blocks = [
            f"{stamp}{entry_label(e)}: {e.text}"
            for stamp, e in zip(stamps, transcript.entries, strict=True)
        ]
    else:
        blocks = [
            f"[{i}] {stamp}{entry_label(e)}: {e.text}"
            for i, stamp, e in zip(
                entry_indices, stamps, transcript.entries, strict=True
            )
        ]
    return "\n\n".join(blocks) if blocks else "(empty session)"


def _entry_stamps(entries: list[TranscriptEntry]) -> list[str]:
    """'HH:MM:SSZ ' per entry, dated when the day changes; '' when untimed."""
    stamps: list[str] = []
    day = None
    for e in entries:
        ts = _parse_ts(e.timestamp)
        if ts is None:
            stamps.append("")
            continue
        ts = ts.astimezone(UTC)
        clock = ts.strftime("%H:%M:%SZ ")
        if ts.date() != day:
            day = ts.date()
            clock = f"{day.isoformat()} {clock}"
        stamps.append(clock)
    return stamps


def render_markdown(
    transcript: Transcript,
    *,
    total_entries: int | None = None,
    entry_range: tuple[int, int] | None = None,
    entry_indices: list[int] | None = None,
    roles: list[str] | None = None,
    brief: tuple[int, int] | None = None,
    timestamps: bool = False,
) -> str:
    """Markdown document: metadata header + transcript body.

    *total_entries*/*entry_range* describe a windowed transcript (get
    --entries/--head/--tail): the header then says which slice this is.
    *roles*/*entry_indices* describe a role-filtered transcript (get
    --role): the header names the roles and each block carries its
    absolute entry index.
    *brief* is the two windows get --brief asked for. It is stated even when
    a session had fewer turns than that to give: "the first 6 user turns"
    and "every user turn there was" look identical in the body, and only the
    header can tell them apart.
    """
    s = transcript.session
    total = total_entries if total_entries is not None else len(transcript.entries)
    if brief is not None:
        entries_line = (
            f"- Entries: {len(transcript.entries)} of {total} "
            f"(brief: first {brief[0]} user, last {brief[1]} assistant)"
        )
    elif roles is not None:
        entries_line = (
            f"- Entries: {len(transcript.entries)} of {total} "
            f"(roles: {', '.join(roles)})"
        )
        if entry_range is not None:
            entries_line += f", within {entry_range[0]}–{entry_range[1]}"
    elif entry_range is not None:
        entries_line = f"- Entries: {entry_range[0]}–{entry_range[1]} of {total}"
    else:
        entries_line = f"- Entries: {total}"
    lines = [
        f"# Session {canonical_id(s)}",
        "",
        f"- Provider: {s.provider}",
        f"- {'Recap' if s.summary_is_recap else 'Summary'}: {s.summary or '—'}",
        f"- CWD: {s.cwd or '—'}",
        f"- Branch: {s.branch or '—'}",
        f"- Repository: {s.repository or '—'}",
        f"- Created: {s.created_at or '—'}",
        f"- Updated: {s.updated_at or '—'}",
        f"- Ended: {s.ending or '—'}",
        entries_line,
    ]
    if transcript.warnings:
        lines.append(f"- Parse warnings: {len(transcript.warnings)}")
    if brief is not None and not transcript.entries:
        body = "(no user or assistant turns in this session)"
    elif roles is not None and not transcript.entries:
        body = f"(no entries with roles: {', '.join(roles)})"
    else:
        body = render_text(
            transcript, entry_indices=entry_indices, timestamps=timestamps
        )
    lines += ["", "---", "", body, ""]
    return "\n".join(lines)


def _parse_ts(raw: str) -> datetime | None:
    if not raw:
        return None
    try:
        dt = datetime.fromisoformat(raw)
    except ValueError:
        return None
    return dt if dt.tzinfo else dt.replace(tzinfo=UTC)


def session_duration_seconds(s: Session) -> int | None:
    """Wall-clock span from created_at to updated_at (last activity), in
    whole seconds; None when either timestamp is missing or unparseable.

    Summaries are often just the first user message, so a bare-looking
    session can hide substantial work — duration is the cheap triage
    signal that catches it without opening the transcript."""
    start, end = _parse_ts(s.created_at), _parse_ts(s.updated_at)
    if start is None or end is None:
        return None
    return max(0, int((end - start).total_seconds()))


def session_to_dict(s: Session) -> dict:
    return {
        "id": canonical_id(s),
        "provider": s.provider,
        "model": s.model,
        "session_id": s.id,
        "summary": s.summary,
        "summary_is_recap": s.summary_is_recap,
        "cwd": s.cwd,
        "branch": s.branch,
        "repository": s.repository,
        "created_at": s.created_at,
        "updated_at": s.updated_at,
        "duration_seconds": session_duration_seconds(s),
        # Both three-state, and the distinction is load-bearing: null means
        # the provider exposes no parent-child link at all (Pi today), while
        # "" means the provider does expose one and this session has none.
        # Reading null as "not a subagent" is the mistake this shape exists
        # to prevent.
        "parent_id": s.parent_id,
        "subagent_kind": s.subagent_kind,
        "continued_in": s.continued_in,
        "continued_from": s.continued_from,
        "live": s.live,
        "ending": s.ending,
    }


def transcript_to_dict(
    t: Transcript, *, entry_indices: list[int] | None = None
) -> dict:
    """Every entry carries its absolute position as "entry_index" -- one
    name for the concept everywhere, and present on every entry, so a
    reader arriving from a search snippet finds the same key here without
    having to know which flags produced the payload.

    It used to be "index" while search snippets said "entry_index", and a
    reader who looked for the search name in this payload, did not find it,
    and concluded get carried no positions at all. Two names for one field
    fail that way silently: the miss reads as "not recorded", not as "wrong
    key". Same argument as last_activity/updated_at, same resolution. A key
    that is present only sometimes fails the same way, so it is now always
    present.

    *entry_indices* supplies the positions when *t* is a windowed or
    role-filtered subset, whose entries' offsets in the list are not their
    positions in the session. Omit it for a whole transcript, whose
    positions are 0..n-1."""
    positions = range(len(t.entries)) if entry_indices is None else entry_indices
    entries = []
    for pos, e in zip(positions, t.entries, strict=True):
        meta = e.metadata
        if meta is not None and ("id" in meta or "call_id" in meta):
            # The provider's call ids pair a call with its output for
            # `calls`, which reads the entries themselves. In this view they
            # are a token per tool entry that no reader asked for.
            meta = {k: v for k, v in meta.items() if k != "id" and k != "call_id"}
        entries.append(
            {
                "entry_index": pos,
                "role": e.role,
                "text": e.text,
                "timestamp": e.timestamp or None,
                "metadata": meta,
            }
        )
    return {
        "session": session_to_dict(t.session),
        "entries": entries,
        "warnings": t.warnings,
    }


@dataclass
class EntryMatch:
    entry_index: int
    entry: TranscriptEntry
    offsets: list[int]
    query: str = ""  # the phrase that matched (multi-query searches)


@dataclass
class SessionSearchResult:
    session: Session
    match_count: int = 0
    matches: list[EntryMatch] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    unreadable: bool = False
    entries: list[TranscriptEntry] | None = None
    total_entries: int = 0
    # Known to match, and not yet loaded: only ever set when the caller asked
    # search_sessions to defer, and then everything above is still empty.
    # search_each loads it.
    deferred: bool = False


@dataclass(frozen=True)
class ContentHit:
    """Why one session matched a global content search.

    ``before``/``match``/``after`` are a snippet around the strongest match
    in original casing and formatting, pre-split so a UI can highlight the
    matched text itself. ``role`` is the role of the entry the snippet was
    taken from (the snippet prefers user over assistant over tool entries);
    the TUI demotes sessions whose only evidence is tool traffic — a session
    merely *quoting* a phrase in tool output should not eclipse the session
    where someone actually said it.
    """

    count: int
    role: str
    before: str
    match: str
    after: str


_SNIPPET_ROLE_RANK = {"user": 0, "assistant": 1}
_SNIPPET_CONTEXT = 80


def _snippet_role_rank(role: str) -> int:
    return _SNIPPET_ROLE_RANK.get(role, 2)


@dataclass(frozen=True)
class _CachedTranscript:
    fingerprint: tuple[str, int, int] | None
    texts: tuple[str, ...]
    roles: tuple[str, ...]
    folded_texts: tuple[str, ...]
    size_bytes: int
    unreadable: str | None = None


class ContentSearchCache:
    """Bounded canonical text cache for successive TUI content searches.

    A global transcript search walks the entire history. Ordinary LRU
    admission would therefore churn: uncached sessions near the end of one
    scan would evict entries needed at the start of the next. This cache keeps
    a stable, recently touched subset and declines new entries once full.
    Modified source files/databases invalidate their entries automatically.
    """

    # The cache stores each entry's original text (for snippets/roles) plus
    # its folded form (for C-speed membership), roughly doubling the bytes
    # per session versus folded-only. The budget is sized so a broad query's
    # hit set still fits — evicted sessions get fully re-searched (ripgrep +
    # parse) on every subsequent keystroke, which costs far more than RAM.
    DEFAULT_MAX_BYTES = 192 * 1024 * 1024

    def __init__(self, max_bytes: int = DEFAULT_MAX_BYTES) -> None:
        self.max_bytes = max(0, max_bytes)
        self._items: OrderedDict[tuple[str, str, str], _CachedTranscript] = (
            OrderedDict()
        )
        self._bytes = 0
        self._lock = threading.Lock()

    @property
    def size_bytes(self) -> int:
        with self._lock:
            return self._bytes

    @property
    def entry_count(self) -> int:
        with self._lock:
            return len(self._items)

    @staticmethod
    def _key(session: Session) -> tuple[str, str, str]:
        return session.provider, session.id, session.content_path

    @staticmethod
    def _source(session: Session) -> Path | None:
        if session.provider == "opencode":
            return _opencode_db_path()
        return Path(session.content_path) if session.content_path else None

    @classmethod
    def _fingerprint(cls, session: Session) -> tuple[str, int, int] | None:
        source = cls._source(session)
        if source is None:
            return None
        try:
            stat = source.stat()
        except OSError:
            return str(source), -1, -1
        return str(source), stat.st_mtime_ns, stat.st_size

    @staticmethod
    def _measure(*text_tuples: tuple[str, ...]) -> int:
        return sum(sys.getsizeof(t) + sum(map(sys.getsizeof, t)) for t in text_tuples)

    def _lookup(self, session: Session) -> _CachedTranscript | None:
        key = self._key(session)
        fingerprint = self._fingerprint(session)
        with self._lock:
            cached = self._items.get(key)
            if cached is not None and cached.fingerprint == fingerprint:
                self._items.move_to_end(key)
                return cached
            if cached is not None:
                self._bytes -= cached.size_bytes
                del self._items[key]
        return None

    def _admit(
        self, session: Session, texts: tuple[str, ...], roles: tuple[str, ...]
    ) -> _CachedTranscript | None:
        key = self._key(session)
        fingerprint = self._fingerprint(session)
        base_size = self._measure(texts, roles)
        # Folding is one of the larger costs of a broad first search. Once
        # the scan-resistant cache is nearly full, don't fold a transcript
        # whose original text and roles alone already exceed the remaining
        # budget: adding the folded tuple can only make it larger. The caller
        # still has the canonical search result; this only declines caching.
        with self._lock:
            existing = self._items.get(key)
            if existing is not None and existing.fingerprint == fingerprint:
                self._items.move_to_end(key)
                return existing
            if existing is not None:
                self._bytes -= existing.size_bytes
                del self._items[key]
            if base_size >= self.max_bytes - self._bytes:
                return None
        folded_texts = tuple(normalize_match_text(t) for t in texts)
        fresh = _CachedTranscript(
            fingerprint,
            texts,
            roles,
            folded_texts,
            base_size + self._measure(folded_texts),
        )
        with self._lock:
            existing = self._items.get(key)
            if existing is not None and existing.fingerprint == fingerprint:
                self._items.move_to_end(key)
                return existing
            if existing is not None:
                self._bytes -= existing.size_bytes
                del self._items[key]
            # Scan-resistant admission: never evict valid entries merely to
            # cache the uncached tail of a complete-history traversal.
            if fresh.size_bytes <= self.max_bytes - self._bytes:
                self._items[key] = fresh
                self._bytes += fresh.size_bytes
        return fresh

    def search(self, sessions: list[Session], query: str) -> set[str]:
        """Ids-only view of :meth:`search_hits` (kept for existing callers)."""
        return set(self.search_hits(sessions, query))

    def search_hits(
        self,
        sessions: list[Session],
        query: str,
        *,
        cancelled: Callable[[], bool] | None = None,
        progress: ProgressFn | None = None,
    ) -> dict[str, ContentHit]:
        """Search cached transcripts and native-prefilter the uncached tail.

        Only confirmed matches are admitted. This avoids parsing the entire
        history merely to warm the cache and makes the first query retain the
        ripgrep speedup; later related queries reuse canonical text directly.
        """
        if progress is not None:
            progress("scanning", 0, len(sessions))
        needle = normalize_match_text(query)
        hits: dict[str, ContentHit] = {}
        uncached: list[Session] = []
        for session in sessions:
            if cancelled is not None and cancelled():
                return {}
            cached = self._lookup(session)
            if cached is None:
                uncached.append(session)
                continue
            hit = _hit_from_cached(cached, query, needle, cancelled=cancelled)
            if hit is not None:
                hits[session.id] = hit

        results = search_sessions(
            uncached, query, keep_entries=True, cancelled=cancelled, progress=progress
        )
        if cancelled is not None and cancelled():
            return {}
        # Admission folds the full text of every matching transcript, so on a
        # broad query this tail is a visible fraction of the search. It gets
        # its own phase for the same reason the DB scan does.
        matched = [r for r in results if not r.unreadable and r.match_count > 0]
        for done, result in enumerate(matched, 1):
            hits[result.session.id] = _hit_from_result(result, query)
            if result.entries is not None:
                self._admit(
                    result.session,
                    tuple(entry.text for entry in result.entries),
                    tuple(entry.role for entry in result.entries),
                )
            if progress is not None:
                progress("indexing", done, len(matched))
        return hits


# Markdown emphasis/code markers are deleted from both haystack and needle
# before matching, so the natural phrase "SELECT only" finds the transcript
# text "`SELECT` only" (and "**SELECT** only"). Deletion keeps literal
# semantics — no fuzziness — while making matching insensitive to inline
# formatting an assistant sprinkled through its prose.
MATCH_STRIP_CHARS = "`*"


def normalize_match_text(text: str) -> str:
    """Casefold + markdown-strip, at C speed (no offset map)."""
    # str.translate pays for a general Unicode mapping table. There are only
    # two literal deletion targets here, and str.replace's specialized C loop
    # is materially faster over the multi-megabyte tool outputs search sees.
    return text.casefold().replace("`", "").replace("*", "")


def _normalize_map(text: str) -> tuple[str, list[int]]:
    """Return normalized text and a list mapping each normalized character
    position back to its original character index.

    Character-by-character equivalent of :func:`normalize_match_text`
    (guaranteed by test), built only after a membership hit because the
    per-character Python pass is far slower than the C fast path.

    Example: ``"Straße"`` casefolds to ``"strasse"`` (7 chars), and the
    returned index map is ``[0, 1, 2, 3, 4, 4, 5]`` — the two ``s``
    characters produced by ``ß`` both point to original index 4. Stripped
    markdown characters contribute no positions at all.
    """
    folded_chars: list[str] = []
    index_map: list[int] = []
    for i, ch in enumerate(text):
        for folded in ch.casefold():
            if folded in MATCH_STRIP_CHARS:
                continue
            folded_chars.append(folded)
            index_map.append(i)
    return "".join(folded_chars), index_map


@dataclass(frozen=True)
class _StrippedOffsetMap:
    """Normalized positions of deleted markdown markers.

    This retains one integer per marker rather than one per transcript
    character, while making every offset lookup a binary search instead of a
    rescan from the start of the entry.
    """

    normalized_positions: tuple[int, ...]


@dataclass(frozen=True)
class _SparseOffsetMap:
    """Position-changing normalization events for a mostly-stable string.

    Unicode casefold expansions are rare, but one ``ß`` anywhere in a large
    entry used to force an integer mapping for every normalized character.
    These parallel tuples retain only deletions/expansions. ``deltas_after``
    is ``normalized_position - original_position`` immediately after each
    event, which makes an arbitrary offset recoverable with one binary search.
    """

    normalized_starts: tuple[int, ...]
    original_indices: tuple[int, ...]
    output_lengths: tuple[int, ...]
    deltas_after: tuple[int, ...]


_OFFSET_EVENT_RE = re.compile(r"[`*]|[^\x00-\x7f]")
_OffsetMap = list[int] | _StrippedOffsetMap | _SparseOffsetMap | None


def _stripped_offset_map(text: str) -> _StrippedOffsetMap:
    """Normalized positions of every deleted markdown marker.

    A regex scan pays the engine's per-character dispatch over the whole
    entry, plus a match object per marker, to learn ~50 positions in a
    multi-kilobyte entry. Two bulk C string operations answer the same
    question: mapping ``*`` onto ``` ` ``` lets ``str.split`` find the
    markers with the single-character (memchr) search path, and each
    resulting segment holds exactly the characters that survive between two
    deletions. So the running total of segment lengths *is* the next
    marker's position in the normalized text — the same ``start - deleted``
    the old loop computed, with the deletion counter folded into the
    accumulation. Measured ~5x faster on the real corpus, and faster on
    every input shape tried (sparse 10 MB entries and 25%-marker text
    alike), because neither operation visits a character in Python.
    """
    # split() yields one more segment than there are markers; the trailing
    # segment sits after the last marker and contributes no position.
    segments = text.replace("*", "`").split("`")
    del segments[-1]
    return _StrippedOffsetMap(tuple(accumulate(map(len, segments))))


def _sparse_offset_map(text: str, folded: str) -> _SparseOffsetMap:
    """Map only characters which delete or expand during normalization.

    ASCII characters other than the two markdown markers always remain one
    character after casefolding. Letting the regex engine jump between the
    exceptional characters avoids a Python-level pass over the ASCII bulk.
    """
    normalized_starts: list[int] = []
    original_indices: list[int] = []
    output_lengths: list[int] = []
    deltas_after: list[int] = []
    delta = 0
    for match in _OFFSET_EVENT_RE.finditer(text):
        output_length = len(normalize_match_text(match.group()))
        if output_length == 1:
            continue
        original_index = match.start()
        normalized_starts.append(original_index + delta)
        original_indices.append(original_index)
        output_lengths.append(output_length)
        delta += output_length - 1
        deltas_after.append(delta)
    # Every length-changing character is either an ASCII marker or non-ASCII,
    # so the event scan must account for the complete normalization delta.
    assert len(text) + delta == len(folded)
    return _SparseOffsetMap(
        tuple(normalized_starts),
        tuple(original_indices),
        tuple(output_lengths),
        tuple(deltas_after),
    )


def _offset_map(text: str, folded: str) -> _OffsetMap:
    """Return the cheapest map from normalized offsets to *text* offsets.

    ``None`` is the identity map. Markdown-only deletions use one compact
    position tuple; Unicode expansions such as ``ß`` -> ``ss`` use a sparse
    event map.

    Python casefolding never deletes a character, so after subtracting the
    explicit markdown deletions, equal lengths prove that every remaining
    character maps one-to-one.  Keeping the deletion count separate avoids
    the old length-cancellation trap (``ß``, plus a deleted backtick).
    """
    if text.isascii():
        # Full casefolding of ASCII is strictly one character in, one ASCII
        # character out (verified over the whole ASCII range: A-Z lowercase,
        # nothing else moves, and nothing folds *into* a marker). Marker
        # deletion is therefore the only way an ASCII entry's length can
        # change, so the length difference *is* the deletion count and the
        # cancellation trap cannot arise here: equal lengths prove zero
        # deletions and hence the identity map. This skips two full C
        # count() scans of every matched ASCII entry -- and str.isascii()
        # itself is a flag read on CPython's string header, not a scan.
        return None if len(folded) == len(text) else _stripped_offset_map(text)
    # Non-ASCII text keeps the explicit count, because there an expansion
    # ("ß" -> "ss") can hide a deletion in the total length; only counting
    # the markers separately can tell the two maps apart.
    stripped = text.count("`") + text.count("*")
    if len(folded) == len(text) - stripped:
        return _stripped_offset_map(text) if stripped else None
    return _sparse_offset_map(text, folded)


def _original_index(text: str, normalized_index: int, offset_map: _OffsetMap) -> int:
    """Map one normalized index back to its original character index.

    The branches test mutually exclusive types, so their order is pure cost:
    every matched offset of a broad query pays for the tests it falls
    through. The marker-only map is what :func:`_offset_map` returns for
    virtually every non-identity entry, so it is checked first; the full
    index list is last because ``_offset_map`` never builds one (only
    :func:`_normalize_map` does, for callers that pass ``norm`` in).
    """
    if offset_map is None:
        return normalized_index
    if isinstance(offset_map, _StrippedOffsetMap):
        return normalized_index + bisect_right(
            offset_map.normalized_positions, normalized_index
        )
    if isinstance(offset_map, list):
        return offset_map[normalized_index]
    if isinstance(offset_map, _SparseOffsetMap):
        event = bisect_right(offset_map.normalized_starts, normalized_index) - 1
        if event < 0:
            return normalized_index
        start = offset_map.normalized_starts[event]
        output_length = offset_map.output_lengths[event]
        if output_length and normalized_index < start + output_length:
            return offset_map.original_indices[event]
        return normalized_index - offset_map.deltas_after[event]

    raise TypeError(f"unsupported offset map: {type(offset_map)!r}")


def find_text_spans(
    text: str,
    query: str,
    norm: tuple[str, _OffsetMap] | None = None,
) -> list[tuple[int, int]]:
    """(start, end) spans of markdown-insensitive matches, in original text.

    Flat-string counterpart of :func:`find_entry_matches`: identical
    normalization, but each match is reported as a span because stripped
    markdown characters inside a match make it longer in the original text
    than in the query. *norm* lets callers reuse a precomputed
    ``(folded_text, offset_map)`` pair across repeated searches of the same
    text; the map may be identity, compact marker-only, or a full index list.
    """
    needle = normalize_match_text(query.strip())
    if not needle or not text:
        return []
    if norm is not None:
        folded, idx_map = norm
    else:
        folded = normalize_match_text(text)
        if needle not in folded:
            # Membership before mapping: the per-character offset map is a
            # slow Python pass, and most texts don't contain the needle.
            return []
        idx_map = _offset_map(text, folded)
    spans: list[tuple[int, int]] = []
    qlen = len(needle)
    start = 0
    while True:
        idx = folded.find(needle, start)
        if idx == -1:
            break
        if idx_map is None:
            spans.append((idx, idx + qlen))
        else:
            spans.append(
                (
                    _original_index(text, idx, idx_map),
                    _original_index(text, idx + qlen - 1, idx_map) + 1,
                )
            )
        start = idx + 1
    # Fold expansion can map distinct folded offsets to one original span.
    return list(dict.fromkeys(spans))


def _find_first_text_span(text: str, query: str) -> tuple[int, int] | None:
    """Return only the first mapped span without enumerating every hit.

    Global-search snippets display one match while their count is calculated
    separately. A broad prefix may occur hundreds of thousands of times in a
    large tool result, so building and mapping every span merely to take
    ``spans[0]`` can dominate the entire search.
    """
    needle = normalize_match_text(query.strip())
    if not needle or not text:
        return None
    folded = normalize_match_text(text)
    index = folded.find(needle)
    if index == -1:
        return None
    offset_map = _offset_map(text, folded)
    if offset_map is None:
        return index, index + len(needle)
    return (
        _original_index(text, index, offset_map),
        _original_index(text, index + len(needle) - 1, offset_map) + 1,
    )


def find_entry_matches(
    entries: Iterable[TranscriptEntry],
    query: str | list[str],
    *,
    cancelled: Callable[[], bool] | None = None,
) -> Iterator[EntryMatch]:
    """Case-insensitive literal match over complete entry text.

    *query* may be one phrase or a list of alternates (OR semantics): all
    phrases are checked in a single pass over the entries, and each match
    records which phrase hit via ``EntryMatch.query``. Matches are yielded
    in entry order, one per (entry, phrase) pair.
    """
    queries = [query] if isinstance(query, str) else list(query)
    pairs = [(q.strip(), normalize_match_text(q.strip())) for q in queries]
    pairs = [(orig, folded) for orig, folded in pairs if folded]
    if not pairs:
        return
    for i, entry in enumerate(entries):
        if cancelled is not None and cancelled():
            return
        # normalize_match_text applies the same character transform as
        # _normalize_map, at C speed. Offset recovery stays lazy until a hit;
        # ordinary case changes and marker-only deletions use compact maps,
        # while fold expansions such as "ß" retain the full map.
        text = normalize_match_text(entry.text)
        idx_map: _OffsetMap = None
        map_ready = False
        for orig, q in pairs:
            if q not in text:
                continue
            if not map_ready:
                idx_map = _offset_map(entry.text, text)
                map_ready = True
            offsets: list[int] = []
            start = 0
            while True:
                idx = text.find(q, start)
                if idx == -1:
                    break
                offsets.append(_original_index(entry.text, idx, idx_map))
                start = idx + 1
                if (len(offsets) & 1023) == 0 and cancelled is not None and cancelled():
                    return
            # Deduplicate offsets — a single original character may expand to
            # multiple folded characters, each producing the same original index.
            if offsets:
                offsets = list(dict.fromkeys(offsets))
                yield EntryMatch(i, entry, offsets, orig)


def make_snippet(text: str, offset: int, match_len: int, context: int) -> str:
    """Excerpt around one match with ellipses where text is elided."""
    start = max(0, offset - context)
    end = min(len(text), offset + match_len + context)
    prefix = "…" if start > 0 else ""
    suffix = "…" if end < len(text) else ""
    return f"{prefix}{text[start:end]}{suffix}"


def _make_hit(text: str, role: str, span: tuple[int, int], count: int) -> ContentHit:
    start, end = span
    return ContentHit(
        count,
        role,
        before=text[max(0, start - _SNIPPET_CONTEXT) : start],
        match=text[start:end],
        after=text[end : end + _SNIPPET_CONTEXT],
    )


def _count_overlapping(
    text: str, needle: str, *, cancelled: Callable[[], bool] | None = None
) -> int:
    count = 0
    start = 0
    while True:
        idx = text.find(needle, start)
        if idx == -1:
            return count
        count += 1
        start = idx + 1
        if (count & 1023) == 0 and cancelled is not None and cancelled():
            return count


def _hit_from_cached(
    cached: _CachedTranscript,
    query: str,
    needle: str,
    *,
    cancelled: Callable[[], bool] | None = None,
) -> ContentHit | None:
    """Hit details from a cached transcript, or None if it doesn't match.

    This runs per cached session per keystroke, so everything except the
    one snippet stays on C-speed string primitives over the folded text;
    the original-text span mapping is built for the best entry only.
    """
    count = 0
    best: tuple[int, int] | None = None
    for i, folded in enumerate(cached.folded_texts):
        if cancelled is not None and cancelled():
            return None
        if needle not in folded:
            continue
        count += _count_overlapping(folded, needle, cancelled=cancelled)
        if cancelled is not None and cancelled():
            return None
        rank = _snippet_role_rank(cached.roles[i])
        if best is None or rank < best[0]:
            best = (rank, i)
    if best is None:
        return None
    i = best[1]
    span = _find_first_text_span(cached.texts[i], query)
    if span is None:
        span = (0, min(len(cached.texts[i]), len(needle)))
    return _make_hit(cached.texts[i], cached.roles[i], span, count)


def _hit_from_result(result: SessionSearchResult, query: str) -> ContentHit:
    """Hit details from a search result known to have at least one match."""
    best = min(
        result.matches, key=lambda m: (_snippet_role_rank(m.entry.role), m.entry_index)
    )
    span = _find_first_text_span(best.entry.text, query)
    if span is None:
        span = (best.offsets[0], best.offsets[0] + len(query.strip()))
    return _make_hit(best.entry.text, best.entry.role, span, result.match_count)


def _search_entries(
    session: Session,
    entries_iter: Iterable[TranscriptEntry],
    query: str | list[str],
    *,
    keep_entries: bool,
    warnings: list[str] | None = None,
    cancelled: Callable[[], bool] | None = None,
) -> SessionSearchResult:
    """Build a SessionSearchResult from an already-obtained entry stream.

    *warnings* is kept by reference (not copied): lazily-parsing streams
    append to it while this function consumes them.
    """
    if warnings is None:
        warnings = []
    total = 0

    def counted(it: Iterable[TranscriptEntry]) -> Iterator[TranscriptEntry]:
        nonlocal total
        for e in it:
            total += 1
            yield e

    if keep_entries:
        entries = list(entries_iter)
        total = len(entries)
        matches = list(find_entry_matches(entries, query, cancelled=cancelled))
    else:
        entries = None
        matches = list(
            find_entry_matches(counted(entries_iter), query, cancelled=cancelled)
        )
    count = sum(len(m.offsets) for m in matches)
    if not matches:
        entries = None
    return SessionSearchResult(
        session, count, matches, warnings, entries=entries, total_entries=total
    )


def search_session(
    session: Session,
    query: str | list[str],
    *,
    keep_entries: bool = False,
    cancelled: Callable[[], bool] | None = None,
) -> SessionSearchResult:
    """Search one session's complete transcript; never raises.

    The result records ``total_entries`` (the session's full length) so
    callers can gauge session size and place matches without re-reading.
    """
    warnings: list[str] = []
    try:
        return _search_entries(
            session,
            iter_entries(session, warnings),
            query,
            keep_entries=keep_entries,
            warnings=warnings,
            cancelled=cancelled,
        )
    except (TranscriptUnreadable, OSError) as exc:
        return SessionSearchResult(session, unreadable=True, warnings=[str(exc)])


# ---------------------------------------------------------------------------
# Bulk search: provider-aware, over the entry cache where a session has one
# ---------------------------------------------------------------------------
#
# A session whose parsed entries are on disk (see entry_cache) is searched
# there: its entry text is scanned and its matches loaded, and none of what
# follows applies to it. Everything below is the stateless path -- no index,
# straight off the provider's own files -- which answers for every session the
# cache does not hold: all of them when it is switched off, and otherwise the
# few it cannot keep. It is also the reference the cached path is tested
# against, so it stays exact.
#
# Raw prefiltering. File-backed sessions can often be ruled out by scanning
# the raw transcript bytes for the query phrases, skipping canonical parsing
# entirely. That is only sound when a raw miss *proves* a canonical miss.
# Normalized entry text differs from raw bytes in three ways:
#
#   1. JSON string escapes: "A" decodes to "A", "\/" to "/", and
#      "\n"/"\"" etc. to control chars/quotes/backslashes.
#   2. json.dumps re-serialization (tool call/result formatting re-dumps
#      parsed values): token spacing (", ", ": "), number reformatting
#      (1e2 -> 100.0, 0.00000001 -> 1e-08, 1e999 -> Infinity), and {} for
#      absent arguments can synthesize characters not present in the raw.
#   3. Formatter-synthesized adjacency: "name(args) [status]" style joins
#      insert (, ), [, ], ? around verbatim fragments, and Codex joins a
#      message's output_text parts with "" (no separator at all).
#
# A phrase is prefilter-safe only when none of those can produce or break a
# match: printable ASCII, no digits or number/float-repr chars, none of the
# structural chars escaping or formatting can synthesize. Escape-produced
# characters (1) are handled by also scanning a \uXXXX-decoded haystack and
# folding "\/" to "/"; the Codex zero-separator join (3) is handled by
# treating any line with multiple output_text parts as a candidate. Anything
# else falls back to a full canonical scan. False positives only cost a
# canonical parse; candidates are always canonically parsed, so reported
# matches, counts, and indexes are exact.

_RAW_UNSAFE = set('"\\(){}[],:.+-?0123456789')

_U_ESCAPE = re.compile(r"\\u[0-9a-fA-F]{4}")
_U_ESCAPE_B = re.compile(rb"\\u[0-9a-fA-F]{4}")


def _u_unescape_bytes(m: re.Match) -> bytes:
    try:
        # The slice drops a "\u" prefix, not "0x", so the base=0 form ruff's
        # FURB166 suggests would not parse these at all.
        return chr(int(m.group()[2:], 16)).encode("utf-8")  # noqa: FURB166
    except UnicodeEncodeError:
        return m.group()  # lone surrogate half: never part of an ASCII match


def _prefilter_needles(query: str | list[str]) -> list[str] | None:
    """Normalized (casefolded, markdown-stripped) phrases safe for raw
    prefiltering, or None to force a full canonical scan (any phrase
    unsafe, or no non-empty phrase)."""
    queries = [query] if isinstance(query, str) else list(query)
    needles: list[str] = []
    for q in queries:
        folded = normalize_match_text(q.strip())
        if not folded:
            continue
        if any(c in _RAW_UNSAFE or not 32 <= ord(c) < 127 for c in folded):
            return None
        # json.dumps can synthesize "Infinity" from a raw "1e999" and a lone
        # exponent "e" from float repr; no other letters appear from nowhere.
        if folded in "infinity" or folded == "e":
            return None
        needles.append(folded)
    return needles or None


def _entry_text_needles(query: str | list[str]) -> list[str] | None:
    """Normalized phrases a scan of cached entry text can decide, or None.

    The entry cache holds the text the matcher itself reads, so none of the
    raw scan's hazards apply: no JSON escapes, no re-serialized numbers, no
    parts the parser has yet to join. That leaves only what the scanner
    cannot express -- a phrase outside printable ASCII, whose casefold
    ripgrep's simple case folding does not reproduce -- and so digits, dots
    and brackets, which send a raw scan to a full parse, are ordinary here.
    """
    queries = [query] if isinstance(query, str) else list(query)
    needles: list[str] = []
    for q in queries:
        folded = normalize_match_text(q.strip())
        if not folded:
            continue
        if not (folded.isascii() and folded.isprintable()):
            return None
        needles.append(folded)
    return needles or None


# Canonical matching deletes markdown markers, so a raw scan must not rule
# out "sel`ect" when the needle is "select": between every two needle chars,
# tolerate a run of the stripped markers. The plain substring check stays
# the fast path; the regex only runs against haystacks that contain a marker
# at all (a marker-free haystack can't hide an interrupted match).


@lru_cache(maxsize=64)
def _tolerant_re(needle: str) -> re.Pattern[str]:
    joiner = f"[{re.escape(MATCH_STRIP_CHARS)}]*"
    return re.compile(joiner.join(re.escape(c) for c in needle))


@lru_cache(maxsize=64)
def _tolerant_re_bytes(needle: str) -> re.Pattern[bytes]:
    return re.compile(_tolerant_re(needle).pattern.encode("ascii"))


# A \uXXXX escape can only create, extend, or complete a match of a
# printable-ASCII needle if it decodes to a character that can take part in
# one: a needle character (either case — the haystack is lowered/casefolded
# after decoding), a stripped markdown marker (the tolerant joiner), or a
# fold-risk character (whose casefold supplies ASCII). Everything else —
# ANSI escapes, emoji surrogates, accented letters — decodes to a
# character that cannot appear anywhere in a match, so text between escapes
# is byte-identical to the raw scan's haystack and needs no second pass. This
# keeps the escape-decoded rescan (and ripgrep's escape marker, below) off
# the overwhelmingly common files whose only escapes are irrelevant.


@lru_cache(maxsize=64)
def _relevant_escape_pattern(needles: tuple[str, ...]) -> str:
    chars = set(MATCH_STRIP_CHARS) | set(_FOLD_RISK_CHARS)
    for n in needles:
        for c in n:
            chars.update((c, c.upper(), c.lower()))
    hexes = sorted(f"{ord(c):04x}" for c in chars)
    return r"\\u(?i:" + "|".join(hexes) + ")"


@lru_cache(maxsize=64)
def _relevant_escape_re(needles: tuple[str, ...]) -> re.Pattern[str]:
    return re.compile(_relevant_escape_pattern(needles))


@lru_cache(maxsize=64)
def _relevant_escape_re_bytes(needles: tuple[str, ...]) -> re.Pattern[bytes]:
    return re.compile(_relevant_escape_pattern(needles).encode("ascii"))


def _raw_text_may_match(raw: str, needles: list[str]) -> bool:
    """True unless *raw* (a JSON-encoded fragment or file) proves that no
    needle can appear in the text canonically decoded from it."""
    # casefold unconditionally: it is the correct fold for non-ASCII, and it
    # agrees with lower on every ASCII codepoint, so the ASCII special case
    # this line used to carry was only ever a speed claim. That claim is false
    # on this interpreter -- CPython dispatches an ASCII casefold to the same
    # C routine lower() uses, measured at 0.982x lower() on 2 MB of ASCII, so
    # the branch cost an isascii() test to buy nothing. Do not reintroduce it.
    hays = [raw.casefold()]
    if "\\u" in raw and _relevant_escape_re(tuple(needles)).search(raw):
        hays.append(
            _U_ESCAPE.sub(
                lambda m: chr(int(m.group()[2:], 16)),  # noqa: FURB166  (see above)
                raw,
            ).casefold()
        )
    if any("/" in n for n in needles):
        hays = [h.replace("\\/", "/") for h in hays]
    if any(n in h for h in hays for n in needles):
        return True
    return any(
        _tolerant_re(n).search(h) is not None
        for h in hays
        if any(c in h for c in MATCH_STRIP_CHARS)
        for n in needles
    )


def _raw_bytes_may_match(raw: bytes, needles: list[str]) -> bool:
    """Bytes counterpart of :func:`_raw_text_may_match`.

    Database-backed providers can use this without first decoding every JSON
    part to a Python string.  UTF-8 decoding and full Unicode casefolding are
    deferred to the rare rows containing a character that can fold into the
    printable-ASCII query alphabet.
    """
    hays = [raw.lower()]
    if b"\\u" in raw and _relevant_escape_re_bytes(tuple(needles)).search(raw):
        hays.append(_U_ESCAPE_B.sub(_u_unescape_bytes, raw).lower())
    if any("/" in n for n in needles):
        hays = [h.replace(b"\\/", b"/") for h in hays]
    encoded = [n.encode("ascii") for n in needles]
    if any(n in h for h in hays for n in encoded):
        return True
    strip_bytes = MATCH_STRIP_CHARS.encode("ascii")
    if any(
        _tolerant_re_bytes(n).search(h) is not None
        for h in hays
        if any(h.find(c) != -1 for c in strip_bytes)
        for n in needles
    ):
        return True
    if any(not h.isascii() and _FOLD_RISK_RE.search(h) for h in hays):
        return _raw_text_may_match(raw.decode("utf-8", errors="replace"), needles)
    return False


# The only codepoints whose casefold contains an ASCII character (ß -> ss,
# ﬁ -> fi, Kelvin K -> k, ...) — i.e. the only non-ASCII characters that can
# take part in a casefolded match of a pure-ASCII needle. Verified
# exhaustively against unicodedata by test_fold_risk_chars_exhaustive, so a
# Unicode upgrade fails a test instead of silently missing one.
_FOLD_RISK_CHARS = "ßİŉſǰẖẗẘẙẚẞKﬀﬁﬂﬃﬄﬅﬆ"
_FOLD_RISK_RE = re.compile("|".join(map(re.escape, _FOLD_RISK_CHARS)).encode("utf-8"))


def _file_may_match(path: Path, needles: list[str], provider: str) -> bool:
    """Raw candidate check for one file-backed session.

    Scans raw bytes with ASCII lowering (needles are printable ASCII by
    construction), decoding to str only for files where \\uXXXX escapes or
    fold-risk characters could hide a match — UTF-8 decode plus full
    casefold costs several times the bytes scan.
    """
    raw = path.read_bytes()
    if _raw_bytes_may_match(raw, needles):
        return True
    if provider == "codex":
        # An assistant message's text is "".join of its output_text parts,
        # so a phrase can span two parts of one message (= one raw line)
        # without appearing contiguously anywhere in the file. Candidate iff
        # two output_text markers share a line (no newline between them).
        pos = raw.find(b"output_text")
        while pos != -1:
            nxt = raw.find(b"output_text", pos + 1)
            if nxt == -1:
                break
            if raw.find(b"\n", pos, nxt) == -1:
                return True
            pos = nxt
    return False


def _prefilter_file(session: Session) -> Path | None:
    """The raw file a prefilter may scan for *session*, if any.

    For a compressed rollout that is its decompressed copy, made here on the
    first search that reaches it. A copy that cannot be made leaves no file
    to scan, so the session goes on to a canonical parse, which reports why.
    """
    if session.provider in ("claude", "codex", "pi"):
        content_path = session.content_path
        if not content_path:
            return None
        # Tested on the string: this runs for every file-backed session, several
        # times per search, and nearly all of them are not compressed.
        if content_path.endswith(compressed.SUFFIX):
            try:
                return compressed.readable(Path(content_path))
            except OSError:
                return None
        return Path(content_path)
    return None


def _segment_prefilter_files(sessions: Iterable[Session]) -> Iterator[Path]:
    """The raw files a prefilter may scan for rewound threads' earlier files.

    Kept apart from ``_prefilter_file``, which runs for every file-backed
    session several times per search; this loop does real work only for the
    handful of sessions a rewind left in more than one file."""
    for s in sessions:
        if not s.segments:
            continue
        for p in s.segments:
            if p.endswith(compressed.SUFFIX):
                try:
                    yield compressed.readable(Path(p))
                except OSError:
                    continue  # the parse reports it
            else:
                yield Path(p)


def _chain_may_match(
    session: Session, needles: list[str] | None, rg_candidates: set[Path] | None
) -> bool:
    """``must_parse`` for a rewound thread: a candidate when ANY of its files
    may match, since a phrase said before the rewind is only in an earlier
    one. A file that cannot be scanned leaves it to the parse to report."""
    if needles is None:
        return True
    newest = _prefilter_file(session)
    if newest is None:
        return True
    for path in (newest, *_segment_prefilter_files((session,))):
        try:
            if not path.is_file():
                return True
            if rg_candidates is not None:
                if path in rg_candidates:
                    return True
            elif _file_may_match(path, needles, session.provider):
                return True
        except OSError:
            return True
    return False


def _rg_candidate_paths(
    sessions: list[Session], needles: list[str]
) -> set[Path] | None:
    """Return file-backed sessions which may canonically match via ripgrep.

    ``None`` means the native scan was unavailable or failed and tells the
    caller to use the in-process prefilter.  Raw literal hits are candidates;
    so are files containing JSON unicode escapes or characters whose casefold
    expands into ASCII, since either can hide a canonical hit.  A second scan
    retains Codex files whose adjacent ``output_text`` parts can synthesize a
    match when the parser joins them.

    ripgrep is intentionally only a candidate finder.  Canonical parsing still
    decides every result, count, offset, and entry index.
    """
    rg = shutil.which("rg")
    if rg is None:
        return None
    wanted = {
        p for s in sessions if (p := _prefilter_file(s)) is not None and p.is_file()
    }
    wanted.update(p for p in _segment_prefilter_files(sessions) if p.is_file())
    paths = sorted(wanted, key=os.fspath)
    if not paths:
        return set()
    # Keep argv comfortably below platform limits on installations with many
    # thousands of sessions.  A failed batch falls back for the whole scan.
    batches: list[list[Path]] = []
    batch: list[Path] = []
    argv_bytes = 0
    for path in paths:
        size = len(os.fsencode(path)) + 1
        if batch and (len(batch) >= 512 or argv_bytes + size > 256_000):
            batches.append(batch)
            batch, argv_bytes = [], 0
        batch.append(path)
        argv_bytes += size
    if batch:
        batches.append(batch)

    common = [
        rg,
        "--no-config",
        "--no-ignore",
        "--hidden",
        "--text",
        "--color=never",
        "--files-with-matches",
        "--null",
    ]

    def scan(args: list[str], selected: list[Path]) -> set[Path] | None:
        """Paths matched by one rg invocation; ``None`` means "fall back".

        Each invocation builds its own set instead of updating a shared one so
        that concurrent scans cannot interleave inside a mutation: ``Path``
        hashes through a Python-level ``__hash__``, which can drop the GIL
        with a set add half done.  Unioning afterwards loses nothing — a set
        union has no order and no duplicates to preserve.
        """
        try:
            proc = subprocess.run(
                [*common, *args, "--", *map(os.fspath, selected)],
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                timeout=30,
                check=False,
            )
        except (OSError, subprocess.SubprocessError):
            return None
        # 0 = matches, 1 = no matches.  2 includes races/read failures, for
        # which a global fallback preserves search_session's warning behavior.
        if proc.returncode not in (0, 1):
            return None
        return {Path(os.fsdecode(p)) for p in proc.stdout.split(b"\0") if p}

    # Backslash-u and fold-risk markers make the scan conservative where raw
    # bytes do not directly contain the casefolded ASCII query.  Query and
    # markers are alternatives of a single union — a file is a candidate if
    # *any* of them matches it — so one invocation carrying all of them
    # answers exactly what two passes answered, while reading the corpus once
    # instead of twice (0.25s of rg becomes 0.15s on the frozen corpus).  The
    # arguments do not vary with the batch, so they are built once.
    #
    # Case sensitivity is the only reason the two were ever separate
    # invocations, and an inline group scopes it per alternative.  The needle
    # patterns get (?i:...) rather than the --ignore-case flag, which would
    # apply to every pattern in the invocation: the safety markers must stay
    # case-sensitive, since under Unicode ignore-case e.g. long-s would match
    # every ordinary ASCII "s" and destroy selectivity.  (?i:...) selects the
    # same Unicode case folding --ignore-case does, merely scoped — which is
    # what _relevant_escape_pattern already does for its hex digits.
    scan_args: list[str] = []
    joiner = f"[{re.escape(MATCH_STRIP_CHARS)}]*"
    for needle in needles:
        # The same marker-tolerant pattern the in-process scan uses, in rg's
        # regex syntax: needle chars joined by an optional run of stripped
        # markdown markers. re.escape only escapes ASCII punctuation (plus
        # space, which Rust's regex crate rejects escaped — keep it raw),
        # and Rust regex accepts any escaped ASCII punctuation.
        pattern = joiner.join(ch if ch == " " else re.escape(ch) for ch in needle)
        scan_args.extend(("--regexp", f"(?i:{pattern})"))
    # The escape marker matches only \uXXXX escapes that decode to a
    # character able to take part in a match (see _relevant_escape_pattern)
    # — nearly every transcript contains irrelevant escapes like \u001b,
    # and flagging those would forward most of the corpus to a canonical
    # parse.
    scan_args.extend(
        (
            "--regexp",
            _relevant_escape_pattern(tuple(needles)),
            "--regexp",
            "|".join(map(re.escape, _FOLD_RISK_CHARS)),
        )
    )

    codex_wanted = {
        p
        for s in sessions
        if s.provider == "codex"
        and (p := _prefilter_file(s)) is not None
        and p.is_file()
    }
    codex_wanted.update(p for p in _segment_prefilter_files(sessions) if p.is_file())
    codex_paths = sorted(codex_wanted, key=os.fspath)

    # The codex join marker stays a separate invocation on purpose: folding it
    # into scan_args would apply it to claude files too, and a claude file that
    # merely mentions output_text twice on a line would become a candidate and
    # be parsed canonically for nothing.
    jobs: list[tuple[list[str], list[Path]]] = [
        (scan_args, selected) for selected in batches
    ]
    for start in range(0, len(codex_paths), 512):
        jobs.append(
            (
                ["--regexp", r"output_text[^\r\n]*output_text"],
                codex_paths[start : start + 512],
            )
        )

    # Every job's paths and pattern are known before any of them runs and no
    # job reads another's output: the answer is the union of them all, and one
    # failure discards the lot however they are ordered.  So they run
    # concurrently.  One at a time, each blocked in subprocess.run with this
    # process idle and a single rg child alive; measured on the frozen corpus
    # (three jobs over 867 files), 0.245s serial against 0.192s overlapped.
    # subprocess.run releases the GIL while it waits, so these threads overlap
    # children rather than contend for the interpreter.
    with ThreadPoolExecutor(max_workers=min(len(jobs), _SEARCH_THREADS)) as ex:
        scanned = list(ex.map(lambda job: scan(*job), jobs))
    # One failed job still falls back for the whole scan, exactly as when
    # the first failure in sequence returned early.
    if any(out is None for out in scanned):
        return None
    candidates: set[Path] = set()
    for out in scanned:
        candidates |= out
    return candidates


def _entry_text_may_match(paths: tuple[str, ...], needles: list[str]) -> bool:
    """In-process candidate check of one session's cached entry text.

    The fallback for a machine without ripgrep, and deliberately the same
    rule ripgrep is given below. *paths* is the text file, then the folded
    file if the session has one. The first holds original text: a phrase is
    there after ASCII lowering, or once markdown markers are skipped. The
    second is already normalized, so a phrase is there or it is not. An
    unreadable file is a candidate, so that the load -- and behind it the
    parse -- decides.
    """
    encoded = [n.encode("ascii") for n in needles]
    try:
        with open(paths[0], "rb") as fh:
            hay = fh.read().lower()
        if any(n in hay for n in encoded):
            return True
        if (b"`" in hay or b"*" in hay) and any(
            _tolerant_re_bytes(n).search(hay) is not None for n in needles
        ):
            return True
        for path in paths[1:]:
            with open(path, "rb") as fh:
                folded = fh.read()
            if any(n in folded for n in encoded):
                return True
    except OSError:
        return True
    return False


def _entry_text_pattern(needle: str) -> str:
    """The ripgrep pattern a scan of cached entry text looks for *needle* by.

    Case-insensitive, with a run of the stripped markdown markers allowed
    between any two characters, which is the matcher's normalization spelled
    as a pattern. That makes the scan exact rather than merely safe, and
    search leans on it: ``search_sessions(defer=True)`` reports a session as
    a match on this pattern's word alone. It holds because ripgrep folds no
    character onto a printable-ASCII one that ``str.casefold`` does not, and
    ``test_ripgrep_folds_onto_ascii_exactly_as_casefold_does`` holds ripgrep
    to that for every codepoint.
    """
    joiner = f"[{re.escape(MATCH_STRIP_CHARS)}]*"
    pattern = joiner.join(ch if ch == " " else re.escape(ch) for ch in needle)
    return f"(?i:{pattern})"


def _rg_entry_text_candidates(texts: list[str], needles: list[str]) -> set[str] | None:
    """Cached entry-text files that hold a needle, found by ripgrep.

    ``None`` means the native scan was unavailable or failed, and the caller
    falls back to :func:`_entry_text_may_match` per session. The pattern is
    the marker-tolerant one the raw scan uses and nothing else. That scan's
    three blanket rules are all gone: parsed text has no ``\\uXXXX`` escapes
    and no Codex parts left to join, and an entry whose casefold a
    case-insensitive scan cannot follow is in the session's folded file,
    already normalized, which is scanned alongside. So a file this returns
    holds a real match, and a session is loaded only to report one.
    """
    rg = shutil.which("rg")
    if rg is None:
        return None
    if not texts:
        return set()
    args = [
        rg,
        "--no-config",
        "--no-ignore",
        "--hidden",
        "--text",
        "--color=never",
        "--files-with-matches",
        "--null",
    ]
    for needle in needles:
        args.extend(("--regexp", _entry_text_pattern(needle)))
    args.append("--")

    # Named relative to the cache directory and run from inside it. A cache
    # path is about 100 bytes of which the last 48 say which file it is, so
    # this is what lets one invocation carry a whole history: 2,574 sessions
    # fit one argv where absolute paths needed two, and the count-limited
    # batches the raw scan uses made six. Each invocation is a fork and an
    # exec of this process -- about 10 ms apiece, measured -- against a scan
    # that takes 50-100 ms in all, and ripgrep spreads the files over every
    # core by itself.
    root = os.fspath(entry_cache.cache_dir())
    prefix = root + os.sep
    inside = [t for t in texts if t.startswith(prefix)]
    if len(inside) != len(texts):
        return None  # not this cache's files; let the caller read them
    batches: list[list[str]] = []
    batch: list[str] = []
    argv_bytes = 0
    for path in inside:
        relative = path[len(prefix) :]
        size = len(os.fsencode(relative)) + 1
        if batch and argv_bytes + size > 256_000:
            batches.append(batch)
            batch, argv_bytes = [], 0
        batch.append(relative)
        argv_bytes += size
    batches.append(batch)

    def scan(selected: list[str]) -> set[str] | None:
        try:
            proc = subprocess.run(
                [*args, *selected],
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                timeout=30,
                check=False,
                cwd=root,
            )
        except (OSError, subprocess.SubprocessError):
            return None
        if proc.returncode not in (0, 1):
            return None  # e.g. a file replaced by a newer parse mid-scan
        return {prefix + os.fsdecode(p) for p in proc.stdout.split(b"\0") if p}

    with ThreadPoolExecutor(max_workers=min(len(batches), _SEARCH_THREADS)) as ex:
        scanned = list(ex.map(scan, batches))
    if any(out is None for out in scanned):
        return None
    found: set[str] = set()
    for out in scanned:
        found |= out
    return found


# Canonical parsing is GIL-bound Python, so the thread pool above overlaps I/O
# waits but not the parse itself: measured on an 896-session corpus, one worker
# takes 2.93s and twenty-four take 2.32s. Processes are the only way to put that
# work on more than one core, and they pay for themselves once enough sessions
# survive the prefilter to amortise interpreter start-up.
#
# The crossover was measured, not guessed. Parse phase only, query "session",
# threads against the pool below, each point a fresh interpreter so the pool
# pays its full first-use cost (the honest case for a one-shot CLI run):
#
#     candidates      50      100      150      200      300
#     threads     0.107s   0.211s   0.311s   0.396s   0.599s
#     processes   0.176s   0.170s   0.197s   0.196s   0.247s
#
# Break-even is near 70. Within one long-lived process (the second search
# onwards) the forkserver is already up and the pool costs 0.009s instead of
# 0.126s, which moves break-even below 50 — so the threshold is set from the
# cold column, the worse of the two, and keeps roughly 2x margin over it.
# Below it the thread pool genuinely wins and is left alone.
_PROC_MIN_CANDIDATES = 150
_PROC_WORKERS = min(10, max(1, (os.cpu_count() or 4) - 2))
# Bounded in-flight window. Cancellation cannot interrupt a worker mid-session,
# so the wait on shutdown is one window's worth of parses; keeping the window
# small is what makes a cancelled search return promptly instead of draining a
# queue of 800 already-submitted files.
_PROC_WINDOW = _PROC_WORKERS * 2


def _parse_worker(args: tuple[Session, str | list[str]]):
    """Canonically parse one file-backed session in a worker process.

    Returns the result's fields rather than a SessionSearchResult, because the
    parent must re-attach its *own* Session object: callers read summary, cwd
    and branch off ``result.session``, and the worker's is a pickled copy.
    Never raises for the same reason :func:`search_session` never does.

    The worker gets the whole Session, not just the fields that locate its
    file. How a file is read depends on more than where it is: a thread_spawn
    child skips its spawner's copied history, a guardian's prompt is relabelled,
    a rewound thread spans several files. Rebuilt from id, provider and path,
    a worker read all three as plain rollouts, so a search that crossed the
    process threshold answered differently from one that did not.
    """
    session, query = args
    r = search_session(session, query)
    return r.match_count, r.matches, r.warnings, r.unreadable, r.total_entries


_PROC_FALLBACK = object()  # pool unusable; caller should use the thread path
_PROC_FALLBACK_WARNED = False
_PROC_CTX_LOCK = threading.Lock()
_PROC_CTX: object | None = None
_PROC_CTX_TRIED = False


def _process_ctx():
    """A start-method context whose workers can import this module, or None.

    ``forkserver`` is strongly preferred over ``spawn``: its workers fork from
    one clean, single-threaded server, so the first pool costs 0.126s against
    spawn's 0.285s and every later pool in the same process costs 0.009s
    because the server is already up.  Forking from the *search* process
    instead would be unsafe -- it holds a thread pool, and fork with threads is
    undefined on macOS -- which is exactly the hazard the forkserver removes.
    """
    global _PROC_CTX, _PROC_CTX_TRIED
    with _PROC_CTX_LOCK:
        if not _PROC_CTX_TRIED:
            _PROC_CTX_TRIED = True
            for method in ("forkserver", "spawn"):
                try:
                    ctx = multiprocessing.get_context(method)
                    if method == "forkserver":
                        # Preload this module so a forked worker already has
                        # the parser resident. Naming it also drops "__main__"
                        # from the preload list, which is the default.
                        ctx.set_forkserver_preload(["session_browser.transcript"])
                    _PROC_CTX = ctx
                    break
                except (ValueError, OSError, AttributeError):
                    continue
        return _PROC_CTX


def prepare_worker_processes() -> None:
    """Start multiprocessing's resource tracker while stderr is still real.

    A process pool needs the tracker, and the tracker hands the child the
    parent's ``sys.stderr.fileno()``. Inside a running Textual app stderr is
    its print capture, whose fileno() is -1, so starting the tracker there
    raised ValueError ("bad value(s) in fds_to_keep"), the pool fell back to
    threads, and every TUI search with enough candidates parsed under the GIL:
    the words wave for "codex pane review" took 17.8 s in the app and 4.6 s
    standalone over the same 1,883 sessions (2026-11-14). An app calls this
    before it runs; once started, the tracker serves the whole process.
    About a millisecond, and a no-op where no pool could be used anyway.
    """
    if not _process_pool_usable():
        return
    try:
        from multiprocessing import resource_tracker

        resource_tracker.ensure_running()
    except Exception as exc:
        # The pool falls back to threads, as it would have without this.
        log.debug("could not start the multiprocessing resource tracker: %s", exc)


def _process_pool_usable() -> bool:
    """Whether worker processes can be started in this environment.

    Every worker re-runs the parent's ``__main__`` before it can unpickle
    anything (``spawn._main`` -> ``prepare`` -> ``_fixup_main_from_path``).
    Choosing forkserver does not avoid this -- its forks go through the same
    path, which was measured, not assumed. That leaves four cases:

    * ``__main__`` is not a readable file (a ``-c`` snippet, a piped script, a
      REPL). Every worker dies with FileNotFoundError and the pool breaks. The
      answer is still correct, because the caller falls back to threads, but a
      library has no business spraying a traceback per worker to get there --
      so this returns False and no pool is ever started.
    * ``python -m session_browser``. Safe, and not by luck: CPython's
      ``_fixup_main_from_name`` returns early for any module name ending in
      ``.__main__``, so the child never re-executes it.
    * A console script. Safe: the generated wrapper carries the standard
      ``if __name__ == "__main__"`` guard.
    * An embedder whose ``__main__`` is a real file with no such guard. Its
      module body runs once per worker. That is the documented contract for
      every library that uses multiprocessing and cannot be detected from
      here; it is why the two cases above were checked rather than assumed.
    """
    if multiprocessing.current_process().name != "MainProcess":
        return False  # never nest pools inside a worker
    if _process_ctx() is None:
        return False
    main_file = getattr(sys.modules.get("__main__"), "__file__", None)
    try:
        return bool(main_file) and os.path.isfile(main_file)
    except (OSError, ValueError):
        return False


def _map_in_processes(
    worker: Callable,
    jobs: list[tuple[int, object]],
    cancelled: Callable[[], bool] | None,
    tick: Callable[[], None] | None = None,
) -> dict[int, object] | object:
    """Run ``worker(payload)`` for each ``(key, payload)`` in worker processes.

    Returns ``{key: result}`` for the jobs that finished -- all of them unless
    *cancelled* fired -- or ``_PROC_FALLBACK`` when no pool can be used here.

    Work is submitted through a bounded window rather than all at once, and
    ``cancelled`` is polled between completions; on cancellation the queued
    futures are dropped. Granularity is one job: a worker already inside one
    runs to the end of it. *tick*, when given, is called once per finished
    job, on this thread.
    """
    if not _process_pool_usable():
        return _PROC_FALLBACK
    out: dict[int, object] = {}
    remaining = iter(jobs)
    ex = None
    try:
        ex = ProcessPoolExecutor(max_workers=_PROC_WORKERS, mp_context=_process_ctx())
        inflight: dict[Future, int] = {}

        def feed() -> None:
            while len(inflight) < _PROC_WINDOW:
                nxt = next(remaining, None)
                if nxt is None:
                    return
                key, payload = nxt
                inflight[ex.submit(worker, payload)] = key

        feed()
        while inflight:
            done, _ = wait(list(inflight), timeout=0.05, return_when=FIRST_COMPLETED)
            if cancelled is not None and cancelled():
                for fut in inflight:
                    fut.cancel()
                break
            for fut in done:
                out[inflight.pop(fut)] = fut.result()
                if tick is not None:
                    tick()
            feed()
    except Exception as exc:
        # A broken pool, an unpicklable payload, or an environment that cannot
        # spawn at all: discard partial work and let the caller re-run these
        # sessions in threads, where the answer is identical. Said once,
        # because the answer is identical and only the time is not: this
        # fallback was silent while every search in the TUI took it, 4x
        # slower, until a wave was timed inside the app.
        global _PROC_FALLBACK_WARNED
        if not _PROC_FALLBACK_WARNED:
            _PROC_FALLBACK_WARNED = True
            log.warning("worker processes unavailable, parsing on threads: %s", exc)
        return _PROC_FALLBACK
    finally:
        if ex is not None:
            ex.shutdown(wait=True, cancel_futures=True)
    return out


def _probe_in_processes(
    pairs: list[tuple[int, Session]],
    query: str | list[str],
    cancelled: Callable[[], bool] | None,
) -> dict[int, SessionSearchResult] | object:
    """Parse *pairs* across worker processes, honouring *cancelled*.

    On cancellation every session that had not finished reports an empty
    result — exactly what the thread probe returns when it observes
    cancellation, so a cancelled search still yields one result per input
    session rather than a short list.

    Cancellation granularity is one session: a worker already inside a parse
    runs to the end of it, where the in-process path can also break out of a
    long offset loop.  Returns ``_PROC_FALLBACK`` if the pool cannot be used,
    so the caller can fall back to threads exactly as a failed native
    candidate scan does.
    """
    if not _process_pool_usable():
        return _PROC_FALLBACK
    if cancelled is not None and cancelled():
        # Already cancelled: starting a pool only to tear it down would put
        # interpreter start-up between the cancel and the return.
        return {i: SessionSearchResult(s) for i, s in pairs}
    done = _map_in_processes(
        _parse_worker, [(i, (s, query)) for i, s in pairs], cancelled
    )
    if done is _PROC_FALLBACK:
        return _PROC_FALLBACK
    out: dict[int, SessionSearchResult] = {}
    for i, s in pairs:
        fields = done.get(i)
        if fields is None:
            # Dropped by cancellation: exactly what a cancelled thread probe
            # reports.
            out[i] = SessionSearchResult(s)
            continue
        count, matches, warns, unreadable, total = fields
        out[i] = SessionSearchResult(
            s, count, matches, warns, unreadable, entries=None, total_entries=total
        )
    return out


def _store_parse(session: Session, ref: entry_cache.Ref) -> bool:
    """Parse *session* and keep the result at *ref*; whether it is now kept.

    False for anything that stops the parse, whatever it is. The session then
    simply has no cached entries, so search reads it the way it always has and
    that read reports the failure in its own words -- an unreadable file as
    ``unreadable``, an undecodable one by raising, exactly as before.
    """
    warnings: list[str] = []
    try:
        entries = list(_parse_entries(session, warnings))
    except Exception as exc:
        log.debug("not caching %s:%s: %s", session.provider, session.id, exc)
        return False
    return _keep_parse(session, ref, entries, warnings)


def _store_worker(args: tuple[Session, entry_cache.Ref]) -> bool:
    """:func:`_store_parse` in a worker process. Only a bool crosses back:
    the entries go to disk from the worker, which is the point of doing the
    parse there."""
    return _store_parse(*args)


def _store_in_processes(
    jobs: list[tuple[int, Session, entry_cache.Ref]],
    cancelled: Callable[[], bool] | None,
    tick: Callable[[], None] | None = None,
) -> set[int] | object:
    """Parse and cache *jobs* across worker processes.

    Returns the keys now cached, or ``_PROC_FALLBACK`` when no pool can be
    used. Its own function, like :func:`_probe_in_processes`, so the routing
    decision can be observed apart from the parsing it routes.
    """
    done = _map_in_processes(
        _store_worker, [(i, (s, ref)) for i, s, ref in jobs], cancelled, tick
    )
    if done is _PROC_FALLBACK:
        return _PROC_FALLBACK
    return {i for i, ok in done.items() if ok}


def _ensure_entry_texts(
    sessions: list[Session],
    file_ix: list[int],
    cancelled: Callable[[], bool] | None = None,
    progress: ProgressFn | None = None,
) -> dict[int, tuple[str, ...]]:
    """Bring the entry cache up to date for *file_ix*; map index -> its files
    to scan: the text file, then the folded file if the session has one.

    A session in the result has its parsed entries on disk, current as of this
    call: search scans the returned files in place of the raw transcript
    and reads its entries with a load in place of a parse. A session missing
    from it could not be cached -- no file, an unreadable one, a cache that
    cannot be written -- and is searched statelessly, as every session was
    before the cache existed.

    Sessions whose parse is missing or out of date are parsed here, once, and
    that is the whole cost of the cache: the first search of a history parses
    all of it (worker processes, a few seconds), and each later one parses
    only the transcripts that changed in between. Parsing a stale session
    outright, instead of first asking the raw prefilter whether it could match
    this query, is what makes the next query free whatever it is.
    """
    present = entry_cache.listing()
    texts: dict[int, tuple[str, ...]] = {}
    missing: list[tuple[int, Session, entry_cache.Ref]] = []
    root = os.fspath(entry_cache.cache_dir())
    for i in file_ix:
        s = sessions[i]
        ref = entry_cache.locate(
            s.provider, s.id, s.content_path, s.subagent_kind, s.segments, root=root
        )
        if ref is None:
            continue
        if ref.text in present and ref.index in present:
            fold = ref.fold
            texts[i] = (ref.text, fold) if fold in present else (ref.text,)
        elif _held_unstored(ref) is None:
            missing.append((i, s, ref))
        # else: already parsed in this process and refused by the cache; search
        # reads it from memory, statelessly.
    # Asked on every search, answered by one stat of a marker file: what was
    # cached from a transcript its provider has since removed is cleared out
    # daily whether or not anything new is being parsed.
    entry_cache.sweep_if_due()
    if not missing or (cancelled is not None and cancelled()):
        return texts
    tick = None
    if progress is not None:
        built = _count(1)
        total = len(missing)

        def tick() -> None:
            progress("parsing", next(built), total)

        progress("parsing", 0, total)
    stored = None
    if len(missing) >= _PROC_MIN_CANDIDATES:
        stored = _store_in_processes(missing, cancelled, tick)
        if stored is _PROC_FALLBACK:
            stored = None  # no pool here; threads build the same files
    if stored is None:

        def build(job: tuple[int, Session, entry_cache.Ref]) -> bool:
            if cancelled is not None and cancelled():
                return False
            try:
                return _store_parse(job[1], job[2])
            finally:
                if tick is not None:
                    tick()

        with ThreadPoolExecutor(max_workers=_SEARCH_THREADS) as ex:
            stored = {
                job[0]
                for job, ok in zip(missing, ex.map(build, missing), strict=True)
                if ok
            }
    for i, _s, ref in missing:
        if i in stored:
            fold = ref.fold
            texts[i] = (ref.text, fold) if os.path.exists(fold) else (ref.text,)
    return texts


# (phase, done, total), where phase is "scanning" while the prefilter is
# ruling sessions out and "reading" while survivors are parsed. "parsing"
# comes before both, and only when it has work: transcripts that are new or
# changed since they were last parsed are parsed into the entry cache. Called from
# the search worker threads, so an implementation must be cheap and must not
# block — store the numbers and let the UI's own timer pick them up.
ProgressFn = Callable[[str, int, int], None]


def search_sessions(
    sessions: list[Session],
    query: str | list[str],
    *,
    keep_entries: bool = False,
    cancelled: Callable[[], bool] | None = None,
    progress: ProgressFn | None = None,
    defer: bool = False,
) -> list[SessionSearchResult]:
    """Search many sessions; returns one result per session, in input order.

    Semantically identical to calling :func:`search_session` per session,
    but retrieves in bulk: the database-backed provider (opencode) is
    scanned with one read-only connection and query for all its selected
    sessions, and file-backed sessions that provably cannot match are
    skipped via the raw prefilter above. Never raises; unreadable
    sessions are reported exactly as search_session reports them. Sessions
    the prefilter rules out report total_entries == 0 without being parsed;
    they have no matches, so search output never surfaces them.

    *defer* is for a caller that will show only some of the matches. A
    session whose parsed entries are cached is then decided by the scan of
    that text alone, which is exact -- a file it returns holds a match and
    one it passes over holds none -- and a match comes back ``deferred``,
    with nothing loaded, for :func:`search_each` to load if it is wanted.
    Every other session is searched as it would be without the flag: one the
    cache does not hold, one in OpenCode's database, and every session when
    no scan of cached text ran (no ripgrep, or a phrase it cannot take).
    """
    results: dict[int, SessionSearchResult] = {}
    opencode_ix: list[int] = []
    file_ix: list[int] = []
    for i, s in enumerate(sessions):
        if cancelled is not None and cancelled():
            return []
        if s.provider == "opencode":
            opencode_ix.append(i)
        else:
            file_ix.append(i)
    needles = _prefilter_needles(query)
    rg_candidates = None
    # File sessions whose parsed entries are on disk, by index -> the cached
    # files a scan reads for them. For these the cache answers both questions
    # the raw files otherwise answer: the candidate scan reads that text, and
    # a parse is a load. Empty when the cache is off, and for any session it
    # cannot hold, which is then searched statelessly below exactly as before.
    texts: dict[int, tuple[str, ...]] = {}
    text_needles: list[str] | None = None
    text_candidates: set[str] | None = None
    # Whether every file session's fate is known without reading it again.
    scanned = False

    def must_parse(i: int) -> bool:
        """Whether the prefilter leaves ``sessions[i]`` needing its entries.

        Sole owner of that decision: both the thread probe below and the
        process path's parent-side partition ask this, so the two cannot
        drift apart on a subtlety like a file that vanished between the
        candidate scan and the probe.
        """
        cached = texts.get(i)
        if cached is not None:
            if text_needles is None:
                return True
            if text_candidates is not None:
                return any(path in text_candidates for path in cached)
            return _entry_text_may_match(cached, text_needles)
        s = sessions[i]
        if s.segments:
            return _chain_may_match(s, needles, rg_candidates)
        path = _prefilter_file(s)
        if needles is not None and path is not None:
            try:
                if rg_candidates is not None and path.is_file():
                    return path in rg_candidates
                if not _file_may_match(path, needles, s.provider):
                    return False
            except OSError:
                pass  # let the canonical path report the failure
        return True

    # Only sessions the prefilter could not rule out are counted: they are the
    # ones that cost real time, and counting the ruled-out majority would race
    # to 95% in a quarter of a second and then appear to stall.
    read_count = _count(1)
    read_total = 0

    def probe(i: int) -> SessionSearchResult:
        s = sessions[i]
        if cancelled is not None and cancelled():
            return SessionSearchResult(s)
        if not must_parse(i):
            return SessionSearchResult(s)
        try:
            return search_session(
                s, query, keep_entries=keep_entries, cancelled=cancelled
            )
        finally:
            if progress is not None:
                # next() on an itertools.count is atomic in CPython, so the
                # thread pool cannot lose increments to a read-modify-write
                # race the way `n += 1` would.
                progress("reading", next(read_count), read_total)

    # The DB bulk scan overlaps the file probes: sqlite releases the GIL
    # around its C calls, and the probes are read-heavy.
    with ThreadPoolExecutor(max_workers=_SEARCH_THREADS + 2) as ex:
        oc_fut = None
        if opencode_ix:
            oc_fut = ex.submit(
                _search_opencode_bulk,
                [sessions[i] for i in opencode_ix],
                query,
                keep_entries=keep_entries,
                progress=progress,
            )
        if file_ix:
            # Decompressed copies of Codex rollouts nothing has wanted for a
            # day, rationed like the entry cache's sweep to one stat a search.
            # With the cache on that is nearly all of them -- a parse streams
            # a rollout and keeps no copy -- and with it off, what a provider
            # has pruned.
            compressed.sweep_if_due()
        if file_ix and entry_cache.enabled():
            texts = _ensure_entry_texts(sessions, file_ix, cancelled, progress)
            text_needles = _entry_text_needles(query)
        raw_ix = [i for i in file_ix if i not in texts] if texts else file_ix
        # The candidate scans run on this thread while the DB scan runs on its
        # worker.  They read only the file corpus (or its cached text) and the
        # DB scan reads only the database, so neither can observe the other
        # and the results are unchanged whichever finishes first; previously
        # the DB scan could not even be submitted until rg had returned, so
        # its cost was added to rg's instead of hidden behind it.  Escaped
        # slashes need an in-process transformed haystack; all other safe
        # needles share one native scan across the raw files.  The probes
        # below do consume the candidates, so they start only after both are
        # assigned.
        if raw_ix and needles is not None and not any("/" in n for n in needles):
            rg_candidates = _rg_candidate_paths([sessions[i] for i in raw_ix], needles)
        if texts and text_needles is not None:
            text_candidates = _rg_entry_text_candidates(
                sorted(path for cached in texts.values() for path in cached),
                text_needles,
            )
        # The file sessions still to probe. Deferring takes the cached ones
        # out here, answered, so nothing below sees them: not the probes, not
        # the process route, not the progress count. A broad phrase then
        # costs its caller the scan and no more -- "the" over 2,574 sessions
        # was 1.6 s, 1.4 s of it loading 2,384 matches to show five
        # (2026-11-14).
        probe_ix = file_ix
        if defer and text_candidates is not None:
            probe_ix = []
            for i in file_ix:
                cached = texts.get(i)
                if cached is None:
                    probe_ix.append(i)
                else:
                    results[i] = SessionSearchResult(
                        sessions[i],
                        deferred=any(path in text_candidates for path in cached),
                    )
        # With every scan answered, deciding a session is a set lookup, and
        # for a phrase no scan can take it is a constant yes. Only a phrase
        # that could have been scanned and was not -- no ripgrep, or a slash
        # for the raw files -- makes the decision read the file, and then the
        # parent must not ask ahead of the probe: it would read each file
        # here and again where it is parsed.
        scanned = (not raw_ix or needles is None or rg_candidates is not None) and (
            not texts or text_needles is None or text_candidates is not None
        )
        if progress is not None:
            # The prefilter has answered, so the denominator is knowable and
            # the display can move off "scanning" onto a count that means
            # something. It must be the same predicate `probe` applies, or
            # sessions whose file has since vanished tick a counter they were
            # never counted in. With candidates in hand that costs one stat
            # each -- the same partition the process path already does. With
            # no candidate scan every file session is read anyway, so the
            # count is exact without asking, and asking would be ruinous:
            # must_parse would read each file here and again in the worker.
            read_total = (
                sum(1 for i in probe_ix if must_parse(i)) if scanned else len(probe_ix)
            )
            progress("reading", 0, read_total)
        if probe_ix:
            # Only the ids/snippets shape goes to processes. keep_entries=True
            # asks for every entry of every matching session, which is the
            # whole corpus text for a broad query -- and it is what the TUI's
            # transcript cache consumes, so it stays where the objects already
            # live. A missing candidate scan also stays on threads: the
            # partition below would then have to run the expensive
            # _file_may_match serially in the parent, reading each file once
            # here and again in the worker.
            parsed = None
            if not keep_entries and scanned and len(probe_ix) >= _PROC_MIN_CANDIDATES:
                proc_ix = [i for i in probe_ix if must_parse(i)]
                if len(proc_ix) >= _PROC_MIN_CANDIDATES:
                    parsed = _probe_in_processes(
                        [(i, sessions[i]) for i in proc_ix], query, cancelled
                    )
                    if parsed is _PROC_FALLBACK:
                        parsed = None  # no pool here; threads answer
                    else:
                        # Prefiltered-out sessions never left the parent.
                        for i in set(probe_ix) - set(proc_ix):
                            parsed[i] = SessionSearchResult(sessions[i])
            if parsed is None:
                results.update(zip(probe_ix, ex.map(probe, probe_ix), strict=True))
            else:
                results.update(parsed)
        if oc_fut is not None:
            # The DB scan overlaps the file probes but routinely outlives
            # them -- its prefilter reads the whole part table -- so it
            # reports its own progress from inside, and the display switches
            # to it once the file phase is done.
            results.update(zip(opencode_ix, oc_fut.result(), strict=True))
    return [results[i] for i in range(len(sessions))]


def search_each(
    sessions: list[Session],
    query: str | list[str],
    *,
    keep_entries: bool = False,
) -> list[SessionSearchResult]:
    """:func:`search_session` for each of *sessions*, in order, and no scan.

    For sessions already known to match -- the ones ``search_sessions(...,
    defer=True)`` left unloaded -- so there is nothing to rule out first and
    the whole cost is the load. Routed as search_sessions routes its own:
    enough of them in the ids/snippets shape go to worker processes, and the
    rest are read on threads.
    """
    if not keep_entries and len(sessions) >= _PROC_MIN_CANDIDATES:
        done = _probe_in_processes(list(enumerate(sessions)), query, None)
        if done is not _PROC_FALLBACK:
            return [done[i] for i in range(len(sessions))]
    if len(sessions) < 2:
        return [search_session(s, query, keep_entries=keep_entries) for s in sessions]
    with ThreadPoolExecutor(max_workers=min(len(sessions), _SEARCH_THREADS)) as ex:
        return list(
            ex.map(
                lambda s: search_session(s, query, keep_entries=keep_entries), sessions
            )
        )


def _ticking_rows(rows: Iterable[tuple], progress: ProgressFn):
    """Yield database rows, reporting how far the scan has got.

    Wrapping the cursor rather than testing inside the caller's loop keeps
    the check off the path entirely when no progress is wanted: a search with
    no callback iterates the cursor exactly as it did before. The rows arrive
    ungrouped, so depth is all there is to report — but a number that keeps
    moving is the whole requirement here, and this scan reads the entire part
    table, so it is where an unreported search looks hung.
    """
    for scanned, row in enumerate(rows, 1):
        if not scanned & 0xFFF:  # every 4096 rows
            progress("conversations", scanned, 0)
        yield row


def _search_opencode_bulk(
    sessions: list[Session],
    query: str | list[str],
    *,
    keep_entries: bool,
    progress: ProgressFn | None = None,
) -> list[SessionSearchResult]:
    """Read-only bulk scan for all selected opencode sessions, streamed and
    grouped per session; rows and parsing match _parse_opencode exactly.

    For prefilter-safe queries a first thin pass streams only the raw part
    JSON (no join, no ordering — the ordered message×part join materializes
    the whole corpus and dominates the scan): the same raw-vs-canonical
    safety argument as for files says a session none of whose raw part JSON
    can contain a needle cannot match, so only surviving sessions pay for
    the ordered join and canonical parse."""
    by_sid: dict[str, list[int]] = {}
    for i, s in enumerate(sessions):
        by_sid.setdefault(s.id, []).append(i)
    results: list[SessionSearchResult | None] = [None] * len(sessions)
    needles = _prefilter_needles(query)

    schema = 0  # learnt from the first statement; see opencode_execute

    def emit(sid: str, rows: list[tuple]) -> None:
        to_entries = (
            _opencode_v2_entries_from_rows
            if schema == OPENCODE_V2
            else _opencode_entries_from_rows
        )
        for i in by_sid.get(sid, ()):
            results[i] = _search_entries(
                sessions[i],
                to_entries(rows),
                query,
                keep_entries=keep_entries,
            )

    db_path = _opencode_db_path()
    if not db_path.is_file():
        return [
            SessionSearchResult(
                s, unreadable=True, warnings=["opencode database not found"]
            )
            for s in sessions
        ]
    try:
        conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True, timeout=3)
        try:
            scan_sids = sorted(by_sid)
            if needles is not None:
                candidates: set[str] = set()
                # Return raw JSON as a BLOB during the broad prefilter. On the
                # live corpus this avoids decoding/casefolding 126 MB of part
                # data; the canonical candidate query below still receives
                # normal strings for parsing exact hits and offsets.
                cur, schema = opencode_execute(
                    conn, _OPENCODE_PREFILTER_SQL, (json.dumps(scan_sids),)
                )
                if progress is not None:
                    cur = _ticking_rows(cur, progress)
                for sid, pdata in cur:
                    if (
                        pdata
                        and sid not in candidates
                        and _raw_bytes_may_match(pdata, needles)
                    ):
                        candidates.add(sid)
                for sid in by_sid.keys() - candidates:
                    for i in by_sid[sid]:
                        results[i] = SessionSearchResult(sessions[i])
                scan_sids = sorted(candidates)
            if scan_sids:
                cur, schema = opencode_execute(
                    conn, _OPENCODE_BULK_ROWS_SQL, (json.dumps(scan_sids),), schema
                )
                # Grouped by session now, so the remaining work is countable.
                emitted = _count(1)
                if progress is not None:
                    progress("conversations", 0, len(scan_sids))

                def emit_counted(sid: str, rows: list[tuple]) -> None:
                    emit(sid, rows)
                    if progress is not None:
                        progress("conversations", next(emitted), len(scan_sids))

                cur_sid: str | None = None
                rows: list[tuple] = []
                for sid, mid, mdata, pdata in cur:
                    if sid != cur_sid:
                        if cur_sid is not None:
                            emit_counted(cur_sid, rows)
                        cur_sid, rows = sid, []
                    rows.append((mid, mdata, pdata))
                if cur_sid is not None:
                    emit_counted(cur_sid, rows)
        finally:
            conn.close()
    except sqlite3.Error as exc:
        err = f"opencode database error: {exc}"
        return [
            r
            if r is not None
            else SessionSearchResult(s, unreadable=True, warnings=[err])
            for s, r in zip(sessions, results, strict=True)
        ]
    # Sessions with no rows in the database: empty transcript, no matches.
    return [
        r if r is not None else _search_entries(s, (), query, keep_entries=keep_entries)
        for s, r in zip(sessions, results, strict=True)
    ]


def session_contains(session: Session, query: str) -> bool:
    """Boolean probe that short-circuits on the first matching entry."""
    try:
        for _ in find_entry_matches(iter_entries(session, []), query):
            return True
    except (TranscriptUnreadable, OSError):
        return False
    return False


# ---------------------------------------------------------------------------
# Word matching (TUI only)
# ---------------------------------------------------------------------------
#
# Literal phrase search misses a session whose words were typed in another
# order, with other words between them, or in different turns altogether.
# "codex pane review" did not find a user turn saying "there is a sibling
# codex pane you can ask for a second opinion or review"; a three-word query
# did not find the session where two of its words were discussed and the
# third was typed 75 entries later. The second is what a person holds of a
# session they want back: a few cues, not a sentence. The TUI therefore runs
# a second, lower-ranked wave for a multi-word query. The rules, stated once
# here because the global search, its ranking and the detail pane all depend
# on them agreeing:
#
#   * Terms are the query's whitespace-separated words after the same
#     normalization literal search uses, so matching is substring and
#     markdown-insensitive as it is there ("pane" also matches "panes").
#     Repeated terms count once. One-character terms are dropped: on their
#     own they match nearly every entry. Stop words are not special.
#   * A query containing a double quote is a literal phrase and gets no
#     words wave: quoting is how to ask for exactly that.
#   * A session matches when EVERY term is found in its conversation -- its
#     user and assistant entries, in any of them, any distance apart -- or
#     in its metadata: where it ran, its summary, its branch, its provider.
#     So each cue added narrows the result and never loses the session, and
#     a project name is a cue like any other. Not the session id: "dead",
#     "face" and "cafe" are hexadecimal.
#   * Tool and system text is where the noise is, so there the older, tighter
#     rule still holds: every term in ONE entry, within UNORDERED_WINDOW
#     characters of normalized text, measured as the smallest window holding
#     them all. Words scattered across hours of tool output are not a match.
#   * Matches are graded, strongest first (WORDS_YOURS, WORDS_CLOSE,
#     WORDS_SAID, WORDS_QUOTED below), and recency orders each grade.
#   * Literal phrase hits always outrank word hits; the CLI's search contract
#     does not change.
#
# 200 was the proximity measured to keep the reported target while cutting
# 111 candidate sessions to 12; 160 also kept it. The wider is the safer
# miss rate for a fallback whose job is recall.
UNORDERED_WINDOW = 200
# Occurrences of one term considered per entry. A term repeated thousands of
# times in a tool dump would otherwise make the window scan quadratic-ish;
# the first few hundred decide the minimal window in any realistic entry.
_UNORDERED_MAX_HITS = 256

# How strongly a session matched the words, strongest first.
#
# YOURS: a person typed every term, or where the session ran holds it. What
# someone remembers of a session is mostly what they said in it.
#
# CLOSE: every term sits together in one user or assistant entry, within the
# window -- said in one breath, by the agent or by a subagent's prompt.
#
# SAID: every term is in the conversation, but spread across turns, with at
# least one of them only in the agent's words, a subagent's prompt, the
# branch or the provider.
#
# QUOTED: the terms sit together in one tool or system entry and are not all
# in the conversation -- something the agent read, not something said.
#
# YOURS over CLOSE is a split by speaker, which SessionBrowser._match_tier
# records as a mistake for literal hits. It is made here on a measurement
# (2026-11-12, 2,193 sessions, against a session known to be the one
# wanted) and because word evidence, unlike a phrase, is not equally strong
# in either voice:
#
#   query                rows  YOURS  CLOSE  SAID  QUOTED   wanted session's row
#                                                         recency  merged  split
#   A B                   288     19     42   109     118       6       4      1
#   C A B                 136      5     14    77      40       4       4      1
#   A B D                  70      6      0    64       0       3       1      1
#   codex pane review     365     20     74   115     156    (not a match)
#
# A to D stand for the words of the real search, which name private work.
# "Merged" is YOURS and CLOSE as one grade; "recency" is no grades at all.
# Merged, the top grade is a fifth of the rows and decides little: the
# wanted session held row 4 only because it was from that morning, and an
# older one falls as far as the grade is deep. The cost of the split is
# bounded by the YOURS column: a recent reply that says the words together
# sits under at most that many sessions, and above every scattered one.
#
# Replayed 2026-11-14 (2,576 sessions) on what that one target could not
# test: wanted sessions five to six weeks old, so recency no longer lifts
# them, and remembered words that were the agent's. Rows are word hits only;
# the wanted session's row is among them:
#
#   query   words by   rows  YOURS  CLOSE  SAID  QUOTED   recency  merged  split
#   E F G   person      122     19      0    17      86        27       8      8
#   H I J   person       29      1      0    23       5        11       1      1
#   K L M   agent         8      0      1     5       2         4       1      1
#   N O P   agent         4      0      2     2       0         2       1      1
#   Q R S   agent        60     13      1    42       4        39      11     14
#   T U V   agent        42      3      5    30       4        10       3      5
#
# Grades beat recency on every one, by most where it matters most: an old
# target with common words went from 27th to 8th. An agent-worded target does
# sink under the YOURS rows, as predicted and no further than that column --
# 14th against 11th merged with 13 of them, 5th against 3rd with 3 -- and
# still far above recency. Merging would buy those few rows back by giving up
# the person-worded case the split exists for, so the split stays.
WORDS_YOURS = 0
WORDS_CLOSE = 1
WORDS_SAID = 2
WORDS_QUOTED = 3

_CONVERSATION_ROLES = ("user", "assistant")


@dataclass(frozen=True)
class WordsHit:
    """Why one session matched a words search, and how strongly.

    ``hit`` is the snippet to show for it, or None when metadata alone holds
    the terms and no user or assistant entry has any of them.
    """

    strength: int
    hit: ContentHit | None


def unordered_terms(query: str) -> list[str]:
    """The terms an unordered match needs, or [] when the query gets none."""
    if '"' in query:
        return []
    seen: list[str] = []
    for word in normalize_match_text(query).split():
        if len(word) > 1 and word not in seen:
            seen.append(word)
    return seen if len(seen) > 1 else []


def unordered_window(
    folded: str, terms: list[str], limit: int = UNORDERED_WINDOW
) -> tuple[int, int, list[tuple[int, int]]] | None:
    """Smallest (start, end) in *folded* holding every term, within *limit*.

    Also returns the term occurrences inside it as (start, end) pairs in
    *folded*, for highlighting. None when some term is absent or the
    tightest window is wider than *limit*.
    """
    hits: list[tuple[int, int]] = []  # (position, term index)
    for t, term in enumerate(terms):
        start, found = 0, 0
        while found < _UNORDERED_MAX_HITS:
            i = folded.find(term, start)
            if i < 0:
                break
            hits.append((i, t))
            found += 1
            start = i + 1
        if not found:
            return None
    hits.sort()
    ends = [p + len(terms[t]) for p, t in hits]
    need = len(terms)
    have = [0] * need
    covered = 0
    best: tuple[int, int] | None = None
    left = 0
    # Indices into hits whose ends decrease: the front is the furthest end in
    # the current window, so its right edge is known without rescanning it.
    tail: deque[int] = deque()
    for right, (_pos, t) in enumerate(hits):
        if not have[t]:
            covered += 1
        have[t] += 1
        while tail and ends[tail[-1]] <= ends[right]:
            tail.pop()
        tail.append(right)
        while covered == need:
            lpos, lt = hits[left]
            end = ends[tail[0]]
            if best is None or end - lpos < best[1] - best[0]:
                best = (lpos, end)
            have[lt] -= 1
            if not have[lt]:
                covered -= 1
            if tail[0] == left:
                tail.popleft()
            left += 1
    if best is None or best[1] - best[0] > limit:
        return None
    inside = [
        (p, p + len(terms[k]))
        for p, k in hits
        if best[0] <= p and p + len(terms[k]) <= best[1]
    ]
    return best[0], best[1], inside


def unordered_spans(
    text: str, terms: list[str], norm: tuple[str, _OffsetMap] | None = None
) -> list[tuple[int, int]]:
    """Original-text spans of each term inside *text*'s tightest window."""
    if norm is None:
        folded = normalize_match_text(text)
        idx_map = _offset_map(text, folded)
    else:
        folded, idx_map = norm
    found = unordered_window(folded, terms)
    if found is None:
        return []
    spans = []
    for start, end in found[2]:
        if idx_map is None:
            spans.append((start, end))
        else:
            spans.append(
                (
                    _original_index(text, start, idx_map),
                    _original_index(text, end - 1, idx_map) + 1,
                )
            )
    return spans


def _metadata_grades(session: Session, terms: list[str], home: str) -> list[int]:
    """Per term: 2 when a person typed it or the session's location holds it,
    1 when weaker metadata does, 0 when no metadata holds it.

    The location is the working directory and the repository. A summary is
    the opening prompt, so it is a person's words -- except in a subagent,
    whose prompt its parent wrote, and when it is the agent's own recap. Provider and branch admit a session
    without counting as typed: "codex" and "main" would otherwise put every
    Codex session, or every session on main, in the strongest grade.

    Two parts of the path are not the location. The home prefix is in every
    path, so it is cut: with it the user's own account name would be a term
    every session holds. And the first directory below home, when the session
    ran deeper than that, is the shelf projects are kept on rather than a
    project: measured 2026-11-12, "project" was in 2,161 of 2,195 working
    directories here through ~/Projects alone, which made it a free word at
    the strongest grade. It admits, like a provider, and does not count.
    """
    cwd = session.cwd or ""
    shelf = ""
    if home and (cwd == home or cwd.startswith(home + os.sep)):
        cwd = cwd[len(home) :]
        head, _, deeper = cwd.lstrip(os.sep).partition(os.sep)
        if deeper:
            shelf, cwd = head, deeper
    place = normalize_match_text(f"{cwd} {session.repository or ''}")
    summary = normalize_match_text(session.summary or "")
    other = normalize_match_text(f"{session.provider} {session.branch or ''} {shelf}")
    # A recap is the agent's words about the session, not the person's.
    person = not (
        session.parent_id or session.subagent_kind or session.summary_is_recap
    )
    grades = []
    for term in terms:
        if term in place or (person and term in summary):
            grades.append(2)
        elif term in summary or term in other:
            grades.append(1)
        else:
            grades.append(0)
    return grades


def metadata_word_strengths(sessions: list[Session], query: str) -> dict[str, int]:
    """Sessions whose metadata alone holds every term of *query*, by id, with
    the strength of that match.

    Reads nothing from disk, so the TUI shows these rows on the keystroke,
    before any transcript has been opened.
    """
    terms = unordered_terms(query)
    if not terms:
        return {}
    home = os.path.expanduser("~")
    found: dict[str, int] = {}
    for s in sessions:
        grades = _metadata_grades(s, terms, home)
        if all(grades):
            found[s.id] = WORDS_YOURS if min(grades) == 2 else WORDS_SAID
    return found


def _words_hit(
    result: SessionSearchResult, terms: list[str], held: list[int]
) -> WordsHit | None:
    """Grade one scanned session against *terms*, or None when it does not
    match. *held* is its _metadata_grades."""
    session = result.session
    person = not (session.parent_id or session.subagent_kind)
    index = {term: t for t, term in enumerate(terms)}
    grade = list(held)
    # entry index -> (entry, indices of the terms it holds)
    per_entry: dict[int, tuple[TranscriptEntry, list[int]]] = {}
    for m in result.matches:
        t = index[m.query]
        slot = per_entry.get(m.entry_index)
        if slot is None:
            per_entry[m.entry_index] = (m.entry, [t])
        else:
            slot[1].append(t)
        role = m.entry.role
        if role == "user":
            said = 2 if person else 1
        elif role == "assistant":
            said = 1
        else:
            continue
        grade[t] = max(grade[t], said)
    # Entries holding every term, checked for the window: the tight rule,
    # which is all that tool and system text can match by.
    need = len(terms)
    windows = 0
    close: tuple[tuple[int, int], TranscriptEntry] | None = None
    for i, (entry, held_terms) in per_entry.items():
        if len(held_terms) < need:
            continue
        if unordered_window(normalize_match_text(entry.text), terms) is None:
            continue
        windows += 1
        key = (_snippet_role_rank(entry.role), i)
        if close is None or key < close[0]:
            close = (key, entry)
    close_said = close is not None and close[1].role in _CONVERSATION_ROLES
    if all(grade):
        if min(grade) == 2:
            strength = WORDS_YOURS
        else:
            strength = WORDS_CLOSE if close_said else WORDS_SAID
    elif close is not None:
        strength = WORDS_QUOTED
    else:
        return None
    if close is not None and (close_said or strength == WORDS_QUOTED):
        entry = close[1]
        spans = unordered_spans(entry.text, terms)
        return WordsHit(
            strength,
            _make_hit(entry.text, entry.role, (spans[0][0], spans[-1][1]), windows),
        )
    # Scattered through the conversation: show the turn holding the most of
    # the terms, a person's before the agent's, and highlight one of them.
    best: tuple[tuple[int, int, int], TranscriptEntry, list[int]] | None = None
    for i, (entry, held_terms) in per_entry.items():
        if entry.role not in _CONVERSATION_ROLES:
            continue
        key = (-len(held_terms), _snippet_role_rank(entry.role), i)
        if best is None or key < best[0]:
            best = (key, entry, held_terms)
    if best is None:
        return WordsHit(strength, None)
    entry = best[1]
    term = terms[min(best[2])]
    span = _find_first_text_span(entry.text, term)
    if span is None:
        return WordsHit(strength, None)
    return WordsHit(strength, _make_hit(entry.text, entry.role, span, 1))


def search_session_word_hits(
    sessions: list[Session],
    query: str,
    *,
    cancelled: Callable[[], bool] | None = None,
    progress: ProgressFn | None = None,
) -> dict[str, WordsHit]:
    """Sessions matching every term of *query* by the word rules above.

    Parsing is what costs, so the work is arranged to parse as little as
    possible, and each candidate exactly once however many words were typed.
    First the native candidate scan runs once per term -- ripgrep over raw
    bytes, about 0.25 s a term on a 2,367-session corpus -- and a file
    session survives a term only if its file can contain it or its metadata
    already holds it; each scan reads only what the terms before it left,
    longest term first because the longest is usually the rarest. Then ONE
    scan for all the terms together parses the survivors, and its matches
    are graded per session. Callers pass only sessions the literal phrase did
    not already find. Without ripgrep the narrowing is skipped and that
    scan's own in-process prefilter decides, as it would for any search.

    The narrowing is only as selective as the words are rare: three words
    nearly every session contains ("codex pane review") still leave almost
    the whole corpus to the scan. Measured 2026-11-12 on 2,193 sessions, best
    of three, against the wave this replaced -- one scan for the longest term
    and a window check on its matches -- over the same six queries: 1.4-2.8 s
    here, 2.0-4.1 s there. Scanning for every term at once is not the dearer
    shape, because the same files are parsed either way. An earlier figure of
    about 20 s for that query (2026-11-01, 2,367 sessions) is what the gate
    in the TUI was set on: it runs this wave only when the literal phrase
    came up short. Two native proximity scans were tried then and dropped: a
    same-line PCRE2 lookahead (0.5 s, 1,478 files, barely better) and a
    bounded-gap regex over every term order (20 s for the scan alone).
    """
    terms = unordered_terms(query)
    if not terms:
        return {}
    home = os.path.expanduser("~")
    held = [_metadata_grades(s, terms, home) for s in sessions]
    alive = list(range(len(sessions)))
    # Sessions with cached entries are narrowed over that text, where every
    # printable term can be scanned; the rest over their raw files, as before.
    texts: dict[int, tuple[str, ...]] = {}
    if entry_cache.enabled():
        texts = _ensure_entry_texts(
            sessions,
            [i for i in alive if sessions[i].provider != "opencode"],
            cancelled,
            progress,
        )
    for t in sorted(range(len(terms)), key=lambda t: -len(terms[t])):
        if cancelled is not None and cancelled():
            return {}
        asked = [
            i for i in alive if sessions[i].provider != "opencode" and not held[i][t]
        ]
        if not asked:
            continue
        # None for a group means "not scanned": every session in it survives
        # this term, and the terms that can be scanned still narrow.
        in_text: set[str] | None = None
        text_needles = _entry_text_needles(terms[t])
        text_asked = [path for i in asked for path in texts.get(i, ())]
        if text_asked and text_needles is not None:
            in_text = _rg_entry_text_candidates(sorted(text_asked), text_needles)
            if in_text is None:
                break
        in_raw: set[Path] | None = None
        needles = _prefilter_needles(terms[t])
        raw_asked = [sessions[i] for i in asked if i not in texts]
        if raw_asked and needles is not None and not any("/" in n for n in needles):
            in_raw = _rg_candidate_paths(raw_asked, needles)
            if in_raw is None:
                break

        def survives(i: int, t=t, in_text=in_text, in_raw=in_raw, needles=needles):
            s = sessions[i]
            if s.provider == "opencode" or held[i][t]:
                return True
            cached = texts.get(i)
            if cached is not None:
                return in_text is None or any(path in in_text for path in cached)
            if in_raw is None:
                return True
            return _prefilter_file(s) in in_raw or bool(
                s.segments and _chain_may_match(s, needles, in_raw)
            )

        alive = [i for i in alive if survives(i)]
    hits: dict[str, WordsHit] = {}
    results = search_sessions(
        [sessions[i] for i in alive], terms, cancelled=cancelled, progress=progress
    )
    for i, r in zip(alive, results, strict=False):
        if r.unreadable:
            continue
        hit = _words_hit(r, terms, held[i])
        if hit is not None:
            hits[r.session.id] = hit
    return hits


def search_session_contents(
    sessions: list[Session], query: str, *, cache: ContentSearchCache | None = None
) -> set[str]:
    """Ids of sessions whose complete transcript contains `query`.

    Same contract the TUI global search has always used, now over the
    complete normalized transcript instead of the truncated rendering.
    """
    q = query.strip()
    if not q:
        return set()
    if cache is not None:
        return cache.search(sessions, q)
    return {
        r.session.id
        for r in search_sessions(sessions, q)
        if not r.unreadable and r.match_count > 0
    }


def search_session_hits(
    sessions: list[Session],
    query: str,
    *,
    cache: ContentSearchCache | None = None,
    cancelled: Callable[[], bool] | None = None,
    progress: ProgressFn | None = None,
) -> dict[str, ContentHit]:
    """Per-session hit evidence for sessions whose transcript contains `query`.

    Same matching contract as :func:`search_session_contents`, but each hit
    carries what the TUI needs to *show* the match: total count, the role of
    the strongest matching entry, and a highlightable snippet around it.

    `progress` is optional and off by default, so callers that only want the
    answer (the CLI) pay one `is not None` test per search, not per session.
    """
    q = query.strip()
    if not q:
        return {}
    if cache is not None:
        return cache.search_hits(sessions, q, cancelled=cancelled, progress=progress)
    return {
        r.session.id: _hit_from_result(r, q)
        for r in search_sessions(sessions, q, cancelled=cancelled, progress=progress)
        if not r.unreadable and r.match_count > 0
    }


def load_session_content(session: Session) -> str:
    """TUI-facing convenience: complete rendered text, never raises."""
    try:
        return render_text(load_transcript(session))
    except (TranscriptUnreadable, OSError) as exc:
        return f"(could not read session: {exc})"


def load_transcript(session: Session) -> Transcript:
    """Parse a session's transcript into structured entries.

    Warnings for recoverable issues (e.g. malformed lines) are collected
    rather than raised.
    """
    warnings: list[str] = []
    entries = list(iter_entries(session, warnings))
    return Transcript(session=session, entries=entries, warnings=warnings)


def _entry_ref(session: Session) -> entry_cache.Ref | None:
    """Where *session*'s parsed entries are kept, or None if they never are.

    OpenCode is left out: its transcripts live in one database that is
    rewritten in place, so there is no per-session file whose size and mtime
    could say a stored parse still holds.
    """
    if session.provider == "opencode" or not entry_cache.enabled():
        return None
    return entry_cache.locate(
        session.provider,
        session.id,
        session.content_path,
        session.subagent_kind,
        session.segments,
    )


def iter_entries(session: Session, warnings: list[str]) -> Iterator[TranscriptEntry]:
    """Yield transcript entries for *session*, collecting warnings.

    From the entry cache when the session's files are as they were when it was
    last parsed, and from a parse otherwise -- which is then kept, so each
    version of a transcript is parsed once. A parse abandoned part-way keeps
    nothing: only a consumer that reads to the end has seen every entry there
    is to store. Either way the entries, their order and the warnings are the
    parser's own; see ``entry_cache`` for what makes a stored parse current.

    A plain function handing back one of two generators, rather than a
    generator delegating to them: a session that is not cached then costs the
    parser's own iterator and nothing wrapped around it.
    """
    ref = _entry_ref(session)
    if ref is None:
        return _parse_entries(session, warnings)
    return _cached_entries(session, ref, warnings)


# Parses the entry cache would not store, held in this process so the
# same transcript is not parsed again: once by _ensure_entry_texts trying to
# cache it, then again by the search that reads it statelessly, and again on
# every later search the TUI runs. Keyed by the cache path, which carries the
# transcript's fingerprint, so a changed transcript is a new key and never served
# stale. Refusal is rare by construction (metadata JSON cannot carry), but so is
# a cache that cannot be written at all, which refuses everything, so only the
# last few are held.
_UNSTORED_MAX = 8
_unstored: OrderedDict[str, tuple[list[TranscriptEntry], list[str]]] = OrderedDict()
_unstored_lock = threading.Lock()


def _hold_unstored(
    ref: entry_cache.Ref, entries: list[TranscriptEntry], warnings: list[str]
) -> None:
    with _unstored_lock:
        _unstored[ref.text] = (entries, list(warnings))
        _unstored.move_to_end(ref.text)
        while len(_unstored) > _UNSTORED_MAX:
            _unstored.popitem(last=False)


def _held_unstored(
    ref: entry_cache.Ref,
) -> tuple[list[TranscriptEntry], list[str]] | None:
    with _unstored_lock:
        return _unstored.get(ref.text)


def _cached_entries(
    session: Session, ref: entry_cache.Ref, warnings: list[str]
) -> Iterator[TranscriptEntry]:
    """*session*'s entries from *ref* if stored there, else parsed and stored."""
    stored = entry_cache.load(ref, TranscriptEntry)
    if stored is None:
        stored = _held_unstored(ref)
    if stored is not None:
        entries, stored_warnings = stored
        warnings.extend(stored_warnings)
        yield from entries
        return
    first_warning = len(warnings)
    parsed: list[TranscriptEntry] = []
    keep = parsed.append
    for entry in _parse_entries(session, warnings):
        keep(entry)
        yield entry
    _keep_parse(session, ref, parsed, warnings[first_warning:])


def _keep_parse(
    session: Session,
    ref: entry_cache.Ref,
    entries: list[TranscriptEntry],
    warnings: list[str],
) -> bool:
    """Store a finished parse at *ref*, unless it may not be the last word.

    A fingerprint says the files have not changed; it cannot say the read went
    well. For one file that is enough -- it opens or the parse raises -- but a
    rewound thread reads several, and an earlier one that cannot be opened or
    decompressed becomes a warning and a shorter transcript rather than an
    error. Stored, that shorter transcript would be served for as long as the
    files stayed the same, long after whatever stopped the read had passed.
    So a rewound thread whose parse warned about anything is parsed afresh
    each time, as it always was; there are a handful of them at most.
    """
    if warnings and session.segments:
        return False
    if entry_cache.store(ref, entries, warnings, _fold_text(entries)):
        return True
    _hold_unstored(ref, entries, warnings)
    return False


def _fold_text(entries: list[TranscriptEntry]) -> bytes:
    """Normalized text of the entries a scan of their original text cannot
    decide, each followed by a newline; empty for nearly every session.

    A case-insensitive scan of original text finds every match of a
    printable-ASCII phrase except one made through a character whose casefold
    is not itself -- "ss" inside a sharp s, "fi" inside a ligature. Those are
    exactly ``_FOLD_RISK_CHARS``, so an entry holding one is stored a second
    time as the matcher will see it, and the scan reads both. ``isascii()``
    is a flag test on a str, so the common entry pays nothing for this.
    """
    parts: list[bytes] = []
    for entry in entries:
        text = entry.text
        if not text.isascii() and any(c in text for c in _FOLD_RISK_CHARS):
            parts.append(normalize_match_text(text).encode("utf-8", "surrogatepass"))
            parts.append(b"\n")
    return b"".join(parts)


def count_entries(session: Session) -> int:
    """How many entries *session*'s transcript has.

    From the entry cache's header when the session is cached -- one short
    line, where counting used to mean parsing the whole transcript for every
    row a listing returned -- and by reading the entries otherwise, which
    caches them for next time. Raises as :func:`iter_entries` does.
    """
    ref = _entry_ref(session)
    if ref is not None:
        known = entry_cache.count(ref)
        if known is not None:
            return known
    return sum(1 for _ in iter_entries(session, []))


def _parse_entries(session: Session, warnings: list[str]) -> Iterator[TranscriptEntry]:
    """Parse *session*'s transcript from its provider's own files."""
    if session.provider == "claude":
        path = _existing_file(session)
        yield from _parse_jsonl(path, warnings, _claude_entries)
    elif session.provider == "codex":
        path = _existing_file(session)
        # Decided once per session, not per record: only a spawned child can
        # open with a copy of its spawner's history, so every other rollout
        # keeps the plain record parser and pays nothing for this.
        event_fn = (
            _codex_own_records()
            if session.subagent_kind == "thread_spawn"
            else _codex_entries
        )
        if session.segments:
            entries = _codex_rewound_entries(session, path, warnings, event_fn)
        else:
            # Likewise how the file is opened: Codex is the provider that
            # compresses its rollouts, and the rest open as they always have.
            opener = (
                compressed.open_binary
                if session.content_path.endswith(compressed.SUFFIX)
                else None
            )
            entries = _dedupe_codex_turns(
                _parse_jsonl(path, warnings, event_fn, opener)
            )
        if session.subagent_kind == "guardian":
            entries = _codex_guardian_entries(entries)
        yield from entries
    elif session.provider == "pi":
        path = _existing_file(session)
        yield from _parse_jsonl(path, warnings, _pi_entries)
    elif session.provider == "opencode":
        db_path = _opencode_db_path()
        if not db_path.is_file():
            raise TranscriptUnreadable("opencode database not found")
        yield from _parse_opencode(db_path, session.id)
    else:
        raise TranscriptUnreadable(f"unsupported provider: {session.provider}")


# ---------------------------------------------------------------------------
# Opencode SQLite parser
# ---------------------------------------------------------------------------


_OPENCODE_SESSION_ROWS_SQL = (
    (
        "SELECT m.id, m.data, p.data "
        "FROM message m "
        "LEFT JOIN part p ON p.message_id = m.id "
        "WHERE m.session_id = ? "
        "ORDER BY m.time_created, m.id, p.time_created, p.id"
    ),
    # v2: one row per message, every piece inside its JSON. seq is the order
    # OpenCode itself replays them in (unique per session).
    "SELECT type, data FROM session_message WHERE session_id = ? ORDER BY seq",
)

# The bulk search's two statements. The prefilter streams raw JSON only, with
# no join and no ordering; the canonical one is grouped by session.
_OPENCODE_PREFILTER_SQL = (
    (
        "SELECT session_id, CAST(data AS BLOB) FROM part "
        "WHERE session_id IN (SELECT value FROM json_each(?))"
    ),
    (
        "SELECT session_id, CAST(data AS BLOB) FROM session_message "
        "WHERE session_id IN (SELECT value FROM json_each(?))"
    ),
)
_OPENCODE_BULK_ROWS_SQL = (
    (
        "SELECT m.session_id, m.id, m.data, p.data "
        "FROM message m "
        "LEFT JOIN part p ON p.message_id = m.id "
        "WHERE m.session_id IN (SELECT value FROM json_each(?)) "
        "ORDER BY m.session_id, m.time_created, m.id, "
        "p.time_created, p.id"
    ),
    # The NULL keeps the row the same width as v1's, so one grouping loop
    # serves both; the v2 converter ignores it.
    (
        "SELECT session_id, type, data, NULL FROM session_message "
        "WHERE session_id IN (SELECT value FROM json_each(?)) "
        "ORDER BY session_id, seq"
    ),
)


def _parse_opencode(db_path: Path, session_id: str) -> Iterator[TranscriptEntry]:
    """Read an opencode SQLite DB and yield transcript entries."""
    try:
        conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True, timeout=3)
        try:
            cur, schema = opencode_execute(
                conn, _OPENCODE_SESSION_ROWS_SQL, (session_id,)
            )
            rows = cur.fetchall()
        finally:
            conn.close()
    except sqlite3.Error as exc:
        raise TranscriptUnreadable(f"opencode database error: {exc}") from exc
    if schema == OPENCODE_V2:
        yield from _opencode_v2_entries_from_rows(rows)
    else:
        yield from _opencode_entries_from_rows(rows)


def _opencode_entries_from_rows(
    rows: Iterable[tuple],
) -> Iterator[TranscriptEntry]:
    """Yield entries from (msg_id, msg_data, part_data) rows of one session,
    already ordered by message then part time/id."""
    seen_msg, role = None, ""
    for msg_id, msg_data, part_data in rows:
        if msg_id != seen_msg:
            seen_msg = msg_id
            try:
                parsed = json.loads(msg_data)
                role = parsed.get("role", "") if isinstance(parsed, dict) else ""
                role = role or "system"
            except (json.JSONDecodeError, TypeError):
                role = "system"
        if not part_data:
            continue
        entry = _opencode_part_entry(part_data, role)
        if entry is not None:
            yield entry


def _opencode_part_entry(raw: str, msg_role: str) -> TranscriptEntry | None:
    """Convert a single opencode part row into a TranscriptEntry, or None."""
    try:
        data = json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return None
    if not isinstance(data, dict):
        return None
    ptype = data.get("type", "")
    if ptype == "text":
        text = data.get("text", "")
        return TranscriptEntry(msg_role, text) if text else None
    if ptype == "tool":
        name = data.get("tool", "") or "?"
        state = data.get("state", {}) or {}
        if not isinstance(state, dict):
            state = {}
        inp = state.get("input", {})
        args = inp if isinstance(inp, str) else json.dumps(inp, ensure_ascii=False)
        text = f"{name}({args})"
        status = state.get("status", "")
        if status:
            text += f" [{status}]"
        out = state.get("output", "")
        if isinstance(out, str) and out:
            text += f"\n{out}"
        metadata: dict = {"tool": name}
        if status == "error":
            metadata["is_error"] = True
        return TranscriptEntry("tool", text, metadata=metadata)
    # step-start / step-finish / reasoning / snapshots: internal, skipped.
    return None


def _opencode_v2_entries_from_rows(
    rows: Iterable[tuple],
) -> Iterator[TranscriptEntry]:
    """Yield entries from (type, data, ...) session_message rows of one v2
    session, already in seq order.

    A user row carries its prompt as ``text``. An assistant row carries an
    ordered ``content`` list of the same kinds of piece v1 stored as parts:
    ``text``, ``tool`` and ``reasoning``. Rows of any other type (``idle``,
    which closes a turn with its outcome, and whatever a later release adds)
    are bookkeeping and yield nothing, as v1's step-start/step-finish do --
    except a background subagent's result (see _opencode_v2_synthetic_entry).
    """
    for mtype, data, *_ in rows:
        if mtype != "user" and mtype != "assistant":
            # Tested only once a row is neither of the two that carry the
            # conversation, so they pay nothing for it.
            if mtype == "synthetic":
                entry = _opencode_v2_synthetic_entry(data)
                if entry is not None:
                    yield entry
            continue
        try:
            msg = json.loads(data)
        except (json.JSONDecodeError, TypeError):
            continue
        if not isinstance(msg, dict):
            continue
        if mtype == "user":
            text = msg.get("text")
            if isinstance(text, str) and text:
                yield TranscriptEntry("user", text)
            continue
        content = msg.get("content")
        if not isinstance(content, list):
            continue
        for piece in content:
            if isinstance(piece, dict):
                entry = _opencode_v2_piece_entry(piece)
                if entry is not None:
                    yield entry


_OPENCODE_SUBAGENT_RESULT = {"kind": "injected", "subtype": "task_notification"}


def _opencode_v2_synthetic_entry(data) -> TranscriptEntry | None:
    """A background subagent's result, as the parent received it.

    OpenCode 2.3's subagent tool returns at once ("working in the background")
    and the child's answer arrives later in the PARENT as a ``synthetic`` row:
    ``metadata`` {source: "subagent", childID, agent, state} and ``text``
    ``<subagent sessionID=... state=...>result</subagent>``, which the
    parent's next turn answers. Under 2.0 that result came back inside the
    tool output. Dropped, the parent read as a spawn and then a reply to
    nothing. Kept as a system entry, the shape a Claude task notification
    takes, since nobody typed it.

    Keyed on source == "subagent" rather than on the row type: one such row
    was all there was to see (2026-11-14), and another source could mean
    anything.
    """
    try:
        msg = json.loads(data)
    except (json.JSONDecodeError, TypeError):
        return None
    if not isinstance(msg, dict):
        return None
    meta = msg.get("metadata")
    text = msg.get("text")
    if not isinstance(meta, dict) or meta.get("source") != "subagent":
        return None
    if not isinstance(text, str) or not text:
        return None
    entry_meta = dict(_OPENCODE_SUBAGENT_RESULT)
    child = meta.get("childID")
    if isinstance(child, str) and child:
        entry_meta["child_id"] = child
    return TranscriptEntry("system", text, "", entry_meta)


def _opencode_v2_piece_entry(piece: dict) -> TranscriptEntry | None:
    """One piece of a v2 assistant message, rendered as v1 renders its part.

    The tool shape moved: ``name`` for v1's ``tool``, output as a list of
    content blocks in ``state.content`` rather than a ``state.output`` string,
    and a failure as ``state.error.message``. The rendered text and metadata
    are kept identical to v1's, so a session reads the same whichever schema
    it was stored under.
    """
    ptype = piece.get("type")
    if ptype == "text":
        text = piece.get("text")
        return (
            TranscriptEntry("assistant", text)
            if isinstance(text, str) and text
            else None
        )
    if ptype != "tool":
        return None  # reasoning, and anything newer: internal, skipped
    name = piece.get("name", "") or "?"
    state = piece.get("state")
    if not isinstance(state, dict):
        state = {}
    inp = state.get("input", {})
    args = inp if isinstance(inp, str) else json.dumps(inp, ensure_ascii=False)
    text = f"{name}({args})"
    status = state.get("status", "")
    if status:
        text += f" [{status}]"
    blocks = state.get("content")
    if isinstance(blocks, list):
        out = "\n".join(
            b["text"]
            for b in blocks
            if isinstance(b, dict) and isinstance(b.get("text"), str) and b["text"]
        )
        if out:
            text += f"\n{out}"
    error = state.get("error")
    if isinstance(error, dict):
        message = error.get("message")
        if isinstance(message, str) and message:
            text += f"\n{message}"
    metadata: dict = {"tool": name}
    if status == "error":
        metadata["is_error"] = True
    return TranscriptEntry("tool", text, metadata=metadata)


# ---------------------------------------------------------------------------
# File-level helpers
# ---------------------------------------------------------------------------


# One shared decoder, default-configured exactly like the module-level instance
# json.loads() uses for a no-kwargs call, so its raw_decode/decode parse a str
# precisely the way json.loads(str) would. Instantiating it once also skips the
# per-call scanner setup json.JSONDecoder() would otherwise repeat. Sharing it
# across the search pool's threads is safe for the same reason json.loads can
# share its own module-level instance: a decoder holds configuration, not
# per-parse state.
_JSON_DECODER = json.JSONDecoder()

# Reading a transcript is one long sequence of readline() calls, so the only
# thing the buffer size changes is how many read(2) syscalls back them. The
# default sizes the buffer to the filesystem block, which for a multi-megabyte
# transcript means thousands of syscalls; a megabyte buffer turns that into a
# handful. It costs one 1 MiB allocation per open file, which is bounded by the
# search pool's worker count.
_READ_BUFFER = 1 << 20


def _existing_file(session: Session) -> Path:
    """Validate and return the content path for *session*.

    A compressed rollout is returned as it is, for :func:`_parse_jsonl` to
    open through ``compressed.open_binary``, which makes no copy of it. One
    that will not decompress then raises OSError from the read, which every
    caller reports as unreadable, as it reports a missing file.
    """
    if not session.content_path:
        raise TranscriptUnreadable("no content path")
    path = Path(session.content_path)
    if not path.is_file():
        raise TranscriptUnreadable(f"content path not found: {path}")
    return path


def _parse_jsonl(
    path: Path,
    warnings: list[str],
    event_fn,
    opener: Callable | None = None,
) -> Iterator[TranscriptEntry]:
    """Stream *path* line-by-line, yielding entries from *event_fn*.

    *opener*, when given, is called as ``open`` is; ``compressed.open_binary``
    reads a compressed rollout that way. The caller picks it once per
    session, and a plain transcript is opened by ``open`` itself, looked up
    when it is called -- not bound as a default, which would freeze whatever
    ``open`` was at import.
    """
    # Keep the stream as bytes and decode each line ourselves. TextIOWrapper
    # otherwise decodes and scans every byte in Python's text-I/O path before
    # the JSON decoder sees it; that is especially costly for the
    # multi-megabyte tool-output records common in agent transcripts.
    #
    # Bind everything the loop touches up front: a transcript is tens of
    # thousands of iterations, and a global+attribute lookup per line for
    # json.loads / len is pure overhead once the body is this small.
    decode_json = _JSON_DECODER.decode
    raw_decode = _JSON_DECODER.raw_decode
    loads = json.loads
    _len = len
    with (
        open(path, "rb", buffering=_READ_BUFFER)
        if opener is None
        else opener(path, "rb", buffering=_READ_BUFFER)
    ) as fh:
        for lineno, raw in enumerate(fh, start=1):
            raw = raw.strip()
            if not raw:
                continue
            # json.loads(bytes) does three things before the scanner ever
            # runs: the loads() wrapper's kwarg/isinstance checks, a Python
            # detect_encoding() call that BOM-sniffs with four startswith
            # probes, and a decode() with a non-default error handler. Doing
            # the decode here and calling the decoder directly skips all
            # three, which on a transcript this long is a measurable share of
            # the parse.
            #
            # The two forms agree only for lines that are strict UTF-8 and
            # that detect_encoding() would call 'utf-8'. Every input where
            # they diverge is routed back to json.loads unchanged, so its
            # behaviour on those is preserved by construction:
            #   * a leading UTF-8 BOM (EF BB BF) makes detect_encoding pick
            #     'utf-8-sig', so json.loads silently drops it and parses the
            #     rest, where a plain decode keeps it as U+FEFF and the parse
            #     fails. Any line starting with EF is deferred, which
            #     over-covers the BOM but costs nothing real: a JSON record
            #     starts with '{'.
            #   * a NUL at offset 0 or 1 makes detect_encoding guess one of
            #     the UTF-16/UTF-32 variants, which decodes to something
            #     entirely different. find() bounded to those two bytes spots
            #     it without scanning the line.
            #   * bytes that are not strict UTF-8. json.loads decodes with
            #     'surrogatepass', which accepts encoded lone surrogates that
            #     strict UTF-8 rejects; and for genuinely invalid bytes it
            #     raises UnicodeDecodeError, not JSONDecodeError, which this
            #     function has always let escape to the caller rather than
            #     turning into a warning. Re-running json.loads on the
            #     original bytes reproduces both exactly. It also catches the
            #     whole-line UTF-16/UTF-32 records, whose BOM (FF FE / FE FF)
            #     is itself invalid UTF-8.
            # Strict decode never differs from a successful 'surrogatepass'
            # decode: an error handler only changes the result of input that
            # strict rejects outright.
            text = None
            if raw[0] != 0xEF and raw.find(b"\x00", 0, 2) < 0:
                # SIM105 wants contextlib.suppress. Measured at 4.85x the
                # cost of try/except on the non-raising path, and this runs
                # once per line of every transcript read. try/except is free
                # until it fires; suppress() builds a context manager every
                # time. No counter in the perf gate would see the difference.
                try:  # noqa: SIM105
                    text = raw.decode()
                except UnicodeDecodeError:
                    pass  # handler exits cleanly, so no chaining on the raise
            try:
                if text is None:
                    obj = loads(raw)
                else:
                    # decode() = skip leading whitespace, raw_decode, skip
                    # trailing whitespace, then "Extra data" if anything is
                    # left. The leading skip is dead here because strip()
                    # already removed b" \t\n\r\x0b\x0c", a superset of the
                    # " \t\n\r" that skip consumes — so idx 0 is where decode()
                    # would have started. The tail is only reachable when the
                    # line has trailing junk, so hand those few lines to
                    # decode() itself instead of restating its rule.
                    obj, end = raw_decode(text, 0)
                    if end != _len(text):
                        obj = decode_json(text)
            except json.JSONDecodeError:
                warnings.append(f"line {lineno}: invalid JSON, skipped")
                continue
            if not isinstance(obj, dict):
                warnings.append(f"line {lineno}: JSON value is not an object, skipped")
                continue
            yield from event_fn(obj)


# ---------------------------------------------------------------------------
# Claude JSONL parser
# ---------------------------------------------------------------------------
#
# Everything below runs once per JSONL record — tens of thousands of times per
# broad-query search — so the shapes are decided with the fewest possible dict
# lookups and type tests. Two facts license the cheap forms used here:
#
#   1. Every value reached from *obj* came out of ``json.loads`` with no
#      object/parse hooks installed, so containers and strings are always the
#      exact builtin types, never subclasses. ``type(x) is dict`` therefore
#      accepts and rejects exactly the same values as ``isinstance(x, dict)``
#      while skipping the subclass walk (which is what makes isinstance
#      noticeably slower on a *negative* test).
#   2. The record-type branches are mutually exclusive on one string, so their
#      order is free. They are ordered by real-corpus frequency: "assistant"
#      and "user" together are ~60% of records, and every unhandled type
#      ("mode", "ai-title", "permission-mode", "file-history-*", "summary", …)
#      yields nothing, so falling through the chain costs it nothing.
#
# TranscriptEntry is also constructed positionally here rather than by keyword.
# The field order is (role, text, timestamp, metadata) and the dataclass
# __init__ binds either form identically, but CPython's keyword call path skips
# the specialised frame push, which measures at ~80ns per entry — the single
# largest per-entry cost in this parser once the shape tests are cheap. The
# Codex parser below has always built them positionally.

# json.dumps() has no cached encoder for ensure_ascii=False: each call builds a
# kwargs dict and a fresh JSONEncoder before encoding. One module-level encoder
# removes that per-call setup. Its output is character-identical because every
# other option keeps its json.dumps default (indent=None, separators=None ->
# ", "/": ", sort_keys=False, allow_nan=True, skipkeys=False, default=None).
# check_circular is turned off rather than defaulted on: the values encoded here
# are always freshly parsed JSON, which cannot contain a cycle, so the marker
# bookkeeping can only ever cost time. Dropping it also leaves the encoder with
# no mutable state, which matters because search parses sessions on a thread
# pool and this encoder is shared.
_json_compact = json.JSONEncoder(ensure_ascii=False, check_circular=False).encode


def _claude_entries(obj: dict) -> Iterator[TranscriptEntry]:
    """Parse a single Claude JSONL object into zero or more entries."""
    # Skip meta and protocol-noise lines
    if obj.get("isMeta"):
        return

    etype = obj.get("type", "")

    if etype == "assistant":
        # ``.get("message")`` replaces the old ``.get("message", {})``: the
        # default only ever fed an empty dict into the content lookup below,
        # which then yielded nothing, exactly as a non-dict message does.
        msg = obj.get("message")
        if type(msg) is not dict:
            return
        content = msg.get("content", "")
        ctype = type(content)
        ts = obj.get("timestamp", "")
        if ctype is str:
            if content:
                yield TranscriptEntry("assistant", content, ts)
        elif ctype is list:
            for block in content:
                if type(block) is not dict:
                    continue
                btype = block.get("type", "")
                if btype == "text":
                    text = block.get("text", "")
                    if text:
                        yield TranscriptEntry("assistant", text, ts)
                elif btype == "tool_use":
                    name = block.get("name", "") or "?"
                    inp = block.get("input", {})
                    text = f"{name}({_json_compact(inp)})"
                    # The id pairs the call with its result. Claude records no
                    # entry for an empty result, so order alone cannot: one
                    # silent command and every later call took the wrong
                    # output. Measured at ~40 ns a block, 0.02% of a parse.
                    meta: dict = {"kind": "call", "tool": name}
                    call_id = block.get("id")
                    if call_id:
                        meta["id"] = call_id
                    yield TranscriptEntry("tool", text, ts, meta)

    elif etype == "user":
        msg = obj.get("message")
        if type(msg) is not dict:
            return
        content = msg.get("content", "")
        ctype = type(content)
        ts = obj.get("timestamp", "")
        if ctype is str:
            if content:
                yield _claude_user_entry(content, ts)
        elif ctype is list:
            for block in content:
                if type(block) is not dict:
                    continue
                btype = block.get("type", "")
                if btype == "text":
                    text = block.get("text", "")
                    if text:
                        yield _claude_user_entry(text, ts)
                elif btype == "tool_result":
                    text = _claude_tool_result_text(block.get("content", ""))
                    if text:
                        meta = {"kind": "output"}
                        call_id = block.get("tool_use_id")
                        if call_id:
                            meta["call_id"] = call_id
                        if block.get("is_error"):
                            meta["is_error"] = True
                        yield TranscriptEntry("tool", text, ts, meta)

    elif etype == "attachment":
        att = obj.get("attachment")
        if type(att) is not dict or att.get("type") != "queued_command":
            return
        # Over half of real queued_command records are machine-generated
        # task notifications, which carry no origin. Only a human-authored
        # prompt is a user turn; skip every other shape rather than guess.
        origin = att.get("origin")
        if type(origin) is not dict or origin.get("kind") != "human":
            return
        text = _claude_queued_prompt_text(att.get("prompt"))
        if text:
            yield TranscriptEntry("user", text, obj.get("timestamp", ""))

    elif etype == "system":
        # Claude Code's own recap of the session, written for a person coming
        # back to it. Kept, so search finds a session by what it says; every
        # other system subtype (turn_duration, local_command, ...) is
        # bookkeeping and still yields nothing. Last in the chain: the
        # unhandled types below pay one more comparison, the two common
        # ones none.
        if obj.get("subtype") == "away_summary":
            text = claude_recap_text(obj.get("content"))
            if text:
                yield TranscriptEntry(
                    "system", text, obj.get("timestamp", ""), _CLAUDE_RECAP
                )

    # No else: "summary" and the bookkeeping types fall out here with nothing
    # yielded, which is what the explicit ``etype == "summary"`` early-out used
    # to do — it never guarded anything, since no branch below claimed it.


# One shared dict, as _CODEX_GUARDIAN_PROMPT is: nothing mutates entry
# metadata after a parse, and a recap needs no per-entry fields.
_CLAUDE_RECAP = {"kind": "recap"}

_CLAUDE_INJECTED_PREFIXES = (
    ("<task-notification>", "task_notification"),
    ("[Request interrupted", "interrupted"),
    ("This session is being continued", "continuation"),
    ("<system-reminder>", "system_reminder"),
    ("Stop hook feedback:", "stop_hook"),
)

# The prefixes alone, for the single-call "is this injected at all?" screen.
_CLAUDE_INJECTED_HEADS = tuple(p for p, _ in _CLAUDE_INJECTED_PREFIXES)


def _claude_injected_subtype(text: str) -> str | None:
    """Name the harness that authored this text, or None if a human did.

    Claude records injected content — task notifications, interrupt markers,
    compaction summaries — as ordinary ``user`` entries. Around a sixth of
    non-meta user records in a real corpus are one of these, so leaving them
    as user turns makes them answer role-filtered reads and literal search as
    though they were typed. Match on a leading marker only: a user quoting one
    of these strings is still speaking."""
    stripped = text.lstrip()
    # Five sixths of user texts match none of the markers, and that verdict is
    # reachable in one C-level call: str.startswith accepts a tuple and returns
    # True iff some member matches, which is exactly the loop's exit condition.
    # Only the rare hit pays the Python loop, and it re-tests in declaration
    # order, so an overlapping pair would still resolve to the same subtype the
    # loop alone picked.
    if not stripped.startswith(_CLAUDE_INJECTED_HEADS):
        return None
    for prefix, subtype in _CLAUDE_INJECTED_PREFIXES:
        if stripped.startswith(prefix):
            return subtype
    return None


# Slash commands are recorded by Claude Code as ordinary ``user`` entries:
# the invocation as a <command-name>/<command-message>/<command-args> block,
# and anything the command printed as <local-command-stdout>. Unlike the
# injected shapes above these are *not* reclassified during parsing -- doing
# so would change what literal search indexes, which is a corpus decision
# rather than a display one. They are named here so the reader that wants
# them gone can ask, and so both halves of the knowledge live together.
#
# Measured over the whole Claude corpus (209 sessions, 2026-09-08): 77
# command-name, 55 command-message, 43 local-command-stdout, 1
# local-command-stderr. <local-command-caveat> leads 69 *raw records* and
# zero parsed entries -- the parser drops them before an entry exists -- so
# it is deliberately absent rather than listed on a guess; add it here if a
# harness change ever lands one in a user entry.
_CLAUDE_SLASH_COMMAND_HEADS = (
    "<command-name>",
    "<command-message>",
    "<local-command-stdout>",
    "<local-command-stderr>",
)


# Pi expands a skill invocation in place: the user entry is the whole SKILL.md
# wrapped in <skill name="..." location="...">...</skill>, followed by whatever
# was typed after ``/skill:<name>``. Measured over all 614 Pi sessions on
# 2026-11-12: 126 expansions in 101 sessions, 37 with nothing typed after them,
# and together 81% of every character of Pi user text (1,886,285 of 2,333,010)
# -- of which the person had typed 19,474. Zero matches among 300 sessions each
# of Claude, Codex and OpenCode.
# Claude Code records the same act as a <command-name> block with the skill
# text in a meta record the parser already drops, so this is the Pi half of
# one idea rather than a second one.
_PI_SKILL_HEAD = '<skill name="'
_PI_SKILL_CLOSE = "</skill>"


def pi_skill_invocation(text: str) -> tuple[str, int] | None:
    """The skill name and the offset where the typed text begins, if *text*
    is a Pi skill expansion; None otherwise.

    The offset points just past the closing tag, so ``text[offset:]`` is what
    the person typed, surrounded by whatever whitespace Pi put there. Leading
    marker only, as everywhere else here, and the closing tag must be present:
    a message that merely opens with the marker is someone talking about it.
    """
    head = len(text) - len(text.lstrip())
    if not text.startswith(_PI_SKILL_HEAD, head):
        return None
    close = text.rfind(_PI_SKILL_CLOSE)
    if close < 0:
        return None
    name_start = head + len(_PI_SKILL_HEAD)
    name_end = text.find('"', name_start)
    if name_end < 0 or name_end > close:
        return None
    return text[name_start:name_end], close + len(_PI_SKILL_CLOSE)


def is_slash_command(entry: TranscriptEntry) -> bool:
    """True if this user entry is a slash-command invocation or its output.

    Leading marker only, exactly as _claude_injected_subtype matches: someone
    who writes "<command-name> shows up as a user turn" is still speaking,
    and a filter that swallowed that message would be hiding conversation.

    Provider-specific by measurement, not by assumption. Surveyed 60 sessions
    each of Codex, OpenCode and Pi on 2026-09-08: none records a slash command
    as a leading-marker user entry. Pi's skill commands are the exception
    found since -- see ``pi_skill_invocation``.
    """
    if entry.role != "user":
        return False
    text = entry.text
    return (
        text.lstrip().startswith(_CLAUDE_SLASH_COMMAND_HEADS)
        or pi_skill_invocation(text) is not None
    )


def is_command_noise(entry: TranscriptEntry) -> bool:
    """True if this user entry drives the harness rather than asks for
    anything: a slash command with no arguments (``/clear``, ``/effort``, a
    bare ``/yolo``) or the output a command printed.

    A command WITH arguments is not noise -- ``/goal fix the login bug`` is
    the request itself, typed through a command. The same split names a
    session in discovery (see ``_claude_summary``), so a session's summary and
    what ``get --brief`` calls its opening request agree on what a request is.
    A Pi skill expansion splits the same way, on what follows its body.
    """
    if not is_slash_command(entry):
        return False
    skill = pi_skill_invocation(entry.text)
    if skill is not None:
        return not entry.text[skill[1] :].strip()
    if _claude_command_output(entry.text):
        return True
    return _claude_summary(entry.text)[1]


def _claude_user_entry(text: str, ts: str) -> TranscriptEntry:
    """A user text entry, demoted to ``system`` when the harness authored it."""
    subtype = _claude_injected_subtype(text)
    if subtype is None:
        return TranscriptEntry("user", text, ts)
    return TranscriptEntry("system", text, ts, {"kind": "injected", "subtype": subtype})


def _claude_queued_prompt_text(prompt) -> str:
    """Extract the text of a delivered queued command.

    A pasted-image prompt arrives as a content-block list carrying inline
    base64 data; keep only the text blocks so image bytes never reach
    rendered output or the search index."""
    if isinstance(prompt, str):
        return prompt
    if isinstance(prompt, list):
        parts = []
        for block in prompt:
            if isinstance(block, dict) and block.get("type") == "text":
                parts.append(block.get("text", ""))
        return "\n".join(p for p in parts if p)
    return ""


def _claude_tool_result_text(content) -> str:
    """Extract text from a tool_result block's content field."""
    ctype = type(content)
    if ctype is str:
        return content
    if ctype is list:
        parts = []
        for item in content:
            if type(item) is dict and item.get("type") == "text":
                parts.append(item.get("text", ""))
        return "\n".join(parts)
    return ""


# ---------------------------------------------------------------------------
# Codex JSONL parser
# ---------------------------------------------------------------------------


# Codex writes a user turn under two different vocabularies, and which one
# depends on the thread's history mode. Upstream's rollout policy persists
# ``user_message`` and ``agent_message`` events only in Legacy mode; Paginated
# mode persists ``item_completed`` events carrying TurnItems instead. Neither
# is present in every file — measured over a 784-rollout corpus, 614 files
# carry the first, 42 the second, and 128 carry neither. What survives both
# eras is the raw ``response_item`` the model was sent, because that side of
# the policy is history-mode independent.
#
# So the response item is the fallback, not the canonical record: Codex also
# presents *injected* input as role=user (hook prompts, the AGENTS.md
# preamble, environment context), and upstream keeps those distinct from a
# TurnItem::UserMessage precisely because they are not something a human
# typed. Taking role=user as the primary signal would index every session's
# boilerplate. Precedence is therefore canonical-event first, response item
# only when no canonical record claims the same turn.
_CODEX_RESPONSE_USER = {"source": "response_item"}


def _dedupe_codex_turns(
    entries: Iterator[TranscriptEntry],
) -> Iterator[TranscriptEntry]:
    """Collapse Codex's two-record turns to one entry each.

    Two separate duplications, both from the same cause — Codex writes a turn
    as an event *and* as the raw response item, milliseconds apart.

    Assistant: keep the first of two identical adjacent entries. Any
    intervening entry resets the comparison, so a genuinely repeated answer
    across turns survives, and a file carrying only one of the two event kinds
    still yields its assistant turns.

    User: the pairing is resolved by *source*, not by text. Measured over the
    real corpus, all 2015 response/canonical pairs are adjacent with the
    response item first — but 19 of them are not text-identical, because the
    response item carries extra model-context wrapping, so an equality test
    leaks those. A response-item user entry is therefore held back one entry
    and dropped when a canonical record follows it, whatever that record says.
    Only response→canonical collapses: canonical→canonical is left alone, and
    60 of those are real retries after an aborted turn.
    """
    pending: TranscriptEntry | None = None
    prev_text: str | None = None
    for e in entries:
        role = e.role
        if pending is not None:
            # The canonical record claims the turn; otherwise the held-back
            # response item was the only record of it and must be emitted --
            # unless it was injected context rather than something a human
            # typed. That test runs here rather than in the parser on purpose:
            # on a legacy rollout every response item is claimed by the event
            # that follows it, so checking at the parse site would pay for
            # 1790 records to reject 230.
            if (
                role != "user" or e.metadata is _CODEX_RESPONSE_USER
            ) and not _codex_injected(pending.text):
                yield pending
            pending = None
            prev_text = None
        if role == "user" and e.metadata is _CODEX_RESPONSE_USER:
            pending = e
            prev_text = None
            continue
        if role == "assistant":
            if e.text == prev_text:
                continue
            prev_text = e.text
        else:
            prev_text = None
        yield e
    if pending is not None and not _codex_injected(pending.text):
        yield pending


# ---------------------------------------------------------------------------
# Codex forks
# ---------------------------------------------------------------------------
#
# A fork (`codex fork`, or a resumed thread forked in the app) is a
# continuation, not a spawned child: its rollout stores only the new records,
# and its first line's `history_base` names the source thread and the byte
# boundary in the source's rollout where the fork's history ends. Every
# thread_spawn child also carries forked_from_id == its spawner, so
# forked_from_id alone does not mean "fork"; history_base on a non-subagent
# does. The Codex index has no column for any of it, so lineage is read here,
# per session asked about, never during discovery -- reading every rollout's
# first line on the index path would cost a file open per Codex session.


def codex_fork_base(session: Session) -> tuple[str, int] | None:
    """(source thread id, byte boundary) when *session* is a Codex fork.

    Read from the thread's own first file, the one named for the thread
    alone. A rewind segment's history_base names the same thread's previous
    file in exactly a fork's shape, so the newest file of a rewound thread
    would otherwise be reported as a fork of itself; a segment can only
    continue its own thread, so it is never asked. A thread whose first file
    is gone cannot say what it was forked from."""
    if session.provider != "codex" or session.subagent_kind:
        return None
    first_file = None
    for p in (*session.segments, session.content_path):
        ids = codex_rollout_ids(os.path.basename(p))
        if ids is None or not ids[1]:
            first_file = p
            break
    if first_file is None:
        return None
    base = _codex_history_base(Path(first_file))
    if base is None:
        return None
    source, end = base.get("thread_id"), base.get("end_byte_offset")
    if not isinstance(source, str) or not isinstance(end, int) or end < 0:
        return None
    if source == session.id:
        return None
    return source, end


def codex_thread_of_segment(segment_id: str, sessions: Iterable[Session]):
    """The rewound Codex session one of whose files is segment *segment_id*.

    A fork names its source by the file its history ends in, and once a
    source has been rewound that can be a segment rather than the thread."""
    for s in sessions:
        if s.provider == "codex" and s.segments:
            for p in (*s.segments, s.content_path):
                ids = codex_rollout_ids(os.path.basename(p))
                if ids is not None and ids[1] == segment_id:
                    return s
    return None


def _codex_history_base(path: Path) -> dict | None:
    """The ``history_base`` on a rollout's first line, if it has one."""
    try:
        with compressed.open_binary(path) as fh:
            first = json.loads(fh.readline())
    except (OSError, ValueError):
        return None
    meta = first.get("payload") if isinstance(first, dict) else None
    base = meta.get("history_base") if isinstance(meta, dict) else None
    return base if isinstance(base, dict) else None


# ---------------------------------------------------------------------------
# Codex rewinds
# ---------------------------------------------------------------------------
#
# An Esc-Esc rewind in Codex's TUI (interrupt, pick an earlier prompt, edit,
# resend) does not truncate the rollout. It starts a new file for the same
# thread, rollout-<ts>-<thread>_<segment>.jsonl, holding only what is written
# after the rewind, and its first line's history_base names the previous file
# (by segment id, or the thread id for the thread's first file) and the
# ordinal and byte offset where the history it continues from ends. The
# previous file keeps everything past that point: the turn or turns the rewind
# undid.
#
# The split is made on the record's ordinal, inside the record callback, so
# the per-line loop in _parse_jsonl stays the one every other transcript goes
# through. The byte offset is the authority, and it is turned into an ordinal
# by reading the one record that starts there; measured on every link on one
# machine (five, codex-cli 0.149-0.169), that record's ordinal is the link's
# end_ordinal_exclusive and ordinals rise by one per record. Once a record is
# past the boundary everything after it is too, ordinal or not: a record that
# carried none would otherwise put an undone turn back into the live history.
#
# The undone turns are KEPT, where they happened, rather than dropped as
# Codex's own view of the thread drops them. They are not drafts: one thread
# here kept 18 records of its first file and undid 189, commands and edits
# that really ran. Each is marked ``rewound`` in its metadata, which every
# renderer shows beside the role (see entry_label), so an abandoned prompt can
# never be read as the instruction the model went on to follow. Nothing is
# added to the text: a phrase the raw file does not contain must not match,
# or the prefilter would rule out a file a canonical parse then matches.


def _codex_link_ordinal(path: Path, base: dict, warnings: list[str]) -> int | None:
    """The first undone ordinal in *path*, from a history_base naming it.

    Read from the record at end_byte_offset; end_ordinal_exclusive answers
    when that record cannot be. None, with a warning, when neither can, and
    then nothing in the file is marked undone."""
    offset = base.get("end_byte_offset")
    stated = base.get("end_ordinal_exclusive")
    found = None
    if type(offset) is int and offset > 0:
        try:
            with open(path, "rb") as fh:
                fh.seek(offset - 1)
                if fh.read(1) == b"\n":
                    record = json.loads(fh.readline() or b"null")
                    if isinstance(record, dict):
                        found = record.get("ordinal")
        except (OSError, ValueError):
            pass
    if type(found) is int:
        if type(stated) is int and stated != found:
            warnings.append(
                f"rewind boundary disagrees with itself in {path.name}: "
                f"ordinal {found} at the byte offset, {stated} stated; "
                "split at the byte offset"
            )
        return found
    if type(stated) is int:
        return stated
    warnings.append(f"rewind boundary unreadable in {path.name}; nothing marked undone")
    return None


def _codex_rewind_plan(
    session: Session, newest: Path, warnings: list[str]
) -> list[tuple[Path, int | None]]:
    """(readable file, first undone ordinal), in the order the files began.

    The newest file is read whole (None). The chain is walked back from it
    through each history_base; a file of the thread that no link reaches
    -- a rewind past the start of the file it was made in -- was undone
    entire, so it is read with every record undone (0) rather than dropped.
    A link to a file that is not here ends the walk with a warning."""
    by_id: dict[str, str] = {}
    for p in session.segments:
        ids = codex_rollout_ids(os.path.basename(p))
        if ids is not None:
            by_id[ids[1] or ids[0]] = p
    splits: dict[str, int | None] = {session.content_path: None}
    readable: dict[str, Path] = {session.content_path: newest}
    current = session.content_path
    while True:
        ids = codex_rollout_ids(os.path.basename(current))
        if ids is None or not ids[1]:
            break  # a thread's first file: its history_base, if any, is a fork
        base = _codex_history_base(readable[current])
        previous = base.get("thread_id") if base is not None else None
        prior = by_id.get(previous) if isinstance(previous, str) else None
        if prior is None or prior in splits:
            warnings.append(f"history before a rewind is unavailable: {previous}")
            break
        try:
            readable[prior] = compressed.readable(Path(prior))
        except OSError as exc:
            warnings.append(f"history before a rewind is unavailable: {exc}")
            break
        splits[prior] = _codex_link_ordinal(readable[prior], base, warnings)
        current = prior
    plan: list[tuple[Path, int | None]] = []
    for p in sorted((*session.segments, session.content_path), key=codex_segment_key):
        if p in splits:
            plan.append((readable[p], splits[p]))
            continue
        warnings.append(
            f"{os.path.basename(p)} is on no rewind link; read as undone entire"
        )
        try:
            plan.append((compressed.readable(Path(p)), 0))
        except OSError as exc:
            warnings.append(f"rewound history unavailable: {exc}")
    return plan


def _codex_rewound_entries(
    session: Session,
    newest: Path,
    warnings: list[str],
    event_fn: Callable[[dict], Iterable[TranscriptEntry]],
) -> Iterator[TranscriptEntry]:
    """A rewound thread's entries across all its files, in the order they
    happened: each file's kept history, then what a rewind undid there,
    marked, then the next file."""
    for n, (path, end) in enumerate(_codex_rewind_plan(session, newest, warnings)):
        # Only the thread's first file can open with a spawner's copied
        # history, and the stateful parser that skips it is meant for one file.
        fn = event_fn if n == 0 else _codex_entries
        # The newest file comes as it is, compressed or not; the earlier ones
        # as copies, which the rewind plan needs on disk to seek in.
        opener = compressed.open_binary
        if end is None:
            yield from _dedupe_codex_turns(_parse_jsonl(path, warnings, fn, opener))
            continue
        undone: list[dict] = []

        def kept(obj: dict, fn=fn, end=end, undone=undone):
            if undone:
                undone.append(obj)
                return ()
            ordinal = obj.get("ordinal")
            if type(ordinal) is int and ordinal >= end:
                undone.append(obj)
                return ()
            return fn(obj)

        yield from _dedupe_codex_turns(_parse_jsonl(path, warnings, kept, opener))
        for entry in _dedupe_codex_turns(
            e for obj in undone for e in _codex_entries(obj)
        ):
            meta = dict(entry.metadata) if entry.metadata else {}
            meta["rewound"] = True
            yield TranscriptEntry(entry.role, entry.text, entry.timestamp, meta)


def codex_prefix_entries(
    source: Session, byte_end: int, warnings: list[str]
) -> list[TranscriptEntry] | None:
    """The entries of *source*'s rollout up to *byte_end*: a fork's history.

    None, with a warning, when the source's rollout is shorter than the
    boundary or the boundary does not fall at the end of a record."""
    try:
        with compressed.open_binary(Path(source.content_path)) as fh:
            prefix = fh.read(byte_end)
    except OSError as exc:
        warnings.append(f"fork history unavailable: {exc}")
        return None
    if len(prefix) != byte_end or (prefix and not prefix.endswith(b"\n")):
        warnings.append(f"fork history incomplete: {source.id}")
        return None

    def raw_entries():
        for line in prefix.splitlines():
            try:
                obj = json.loads(line)
            except (UnicodeDecodeError, json.JSONDecodeError):
                warnings.append(f"invalid fork history record: {source.id}")
                continue
            if isinstance(obj, dict):
                yield from _codex_entries(obj)

    return list(_dedupe_codex_turns(raw_entries()))


def _codex_own_records() -> Callable[[dict], Iterable[TranscriptEntry]]:
    """A Codex record parser that skips a spawned child's copied history.

    Newer Codex builds (0.154.0 in the ground-truth fixture) start a
    thread_spawn child's rollout with a copy of its spawner's history: the
    child's own session_meta, then the spawner's session_meta, then the
    spawner's turns up to the one that spawned it. Parsed as the child's own,
    that put the root's prompt in every descendant -- a search for it
    returned the root and all four of its descendants.

    The copy is marked by ``subagent_history_start_ordinal`` on the child's
    own (first) session_meta; a rollout without it is parsed exactly as any
    other. The field is an ordinal in the spawner's in-memory history, not a
    line number, so it cannot be used to cut the file. The boundary is the
    first ``task_started`` that began no earlier than the child itself
    (``started_at`` against the child's creation time, at the one-second
    grain started_at has): the copied block is written in one burst at spawn
    time, and the turns in it started 8-58 s before the spawn in every
    fixture child. The records' own timestamps are the copy time, so they
    cannot separate the halves.

    Each uncertainty errs towards keeping text: a creation time that will not
    parse, or a task_started without started_at, ends the skip at once, so a
    shape this has not seen reads as it did before rather than losing the
    child's own turns. A spawn in the same second as its spawning turn began
    keeps that one spawner turn.
    """
    state = [0, 0]  # phase (0 first record, 1 in the copy, 2 own), spawn second

    def event_fn(obj: dict) -> Iterable[TranscriptEntry]:
        phase = state[0]
        if phase == 2:
            return _codex_entries(obj)
        payload = obj.get("payload")
        if type(payload) is not dict:
            payload = {}
        if phase == 0:
            if (
                obj.get("type") == "session_meta"
                and "subagent_history_start_ordinal" in payload
            ):
                state[0], state[1] = 1, _epoch_second(payload.get("timestamp"))
                return ()
            state[0] = 2
            return _codex_entries(obj)
        if obj.get("type") == "event_msg" and payload.get("type") == "task_started":
            started = payload.get("started_at")
            if type(started) is not int or started >= state[1]:
                state[0] = 2
                return _codex_entries(obj)
        return ()

    return event_fn


_CODEX_GUARDIAN_PROMPT = {"kind": "injected", "subtype": "guardian_review"}


def _codex_guardian_entries(
    entries: Iterable[TranscriptEntry],
) -> Iterator[TranscriptEntry]:
    """A Guardian review thread's user turns, relabelled as system.

    Codex opens a Guardian thread to judge an escalation, and its prompt is
    the parent's history quoted wholesale -- Codex's own source calls it
    "synthetic review context, not user-authored messages". Read as `user`,
    a search for something the human typed in the parent credited the
    guardian with saying it too. Labelled, not hidden: the verdict and the
    quote stay readable, with the role saying what they are.
    """
    for e in entries:
        if e.role == "user":
            yield TranscriptEntry("system", e.text, e.timestamp, _CODEX_GUARDIAN_PROMPT)
        else:
            yield e


def _epoch_second(ts) -> int:
    """Whole seconds since the epoch for an ISO timestamp, or 0."""
    if not isinstance(ts, str):
        return 0
    try:
        dt = datetime.fromisoformat(ts)
    except ValueError:
        return 0
    # Codex writes UTC with a Z. A naive one is read as UTC too: as local
    # time, west of Greenwich it would land later and skip the child's turns.
    return int((dt if dt.tzinfo else dt.replace(tzinfo=UTC)).timestamp())


def _codex_entries(obj: dict) -> Iterator[TranscriptEntry]:
    """Parse a single Codex JSONL object into zero or more entries."""
    # ``type(x) is T`` throughout, for the reason given above the Claude
    # parser: these values all come straight from json.loads, so they are
    # never subclasses and the exact-type test decides them identically.
    etype = obj.get("type", "")
    ts = obj.get("timestamp", "")

    if etype == "event_msg":
        # ``.get("payload")`` replaces ``.get("payload", {})``: the empty
        # default produced etype "" and then fell through every branch below,
        # yielding nothing — the same as bailing out on a non-dict payload.
        payload = obj.get("payload")
        if type(payload) is not dict:
            return
        etype = payload.get("type", "")
        obj = payload

    if etype == "response_item":
        payload = obj.get("payload")
        if type(payload) is dict:
            yield from _codex_response_item(payload, ts)
    elif etype == "user_message":
        text = _codex_user_text(obj)
        if text:
            yield TranscriptEntry("user", text, ts)
    elif etype == "agent_message":
        msg = obj.get("message", "")
        if type(msg) is str and msg:
            yield TranscriptEntry("assistant", msg, ts)
    elif etype == "item_completed":
        yield from _codex_item_completed(obj, ts)
    elif etype == "response.output_item.done":
        item = obj.get("item")
        if type(item) is not dict:
            return
        if item.get("type") == "message":
            for c in item.get("content", []):
                if type(c) is dict and c.get("type") == "output_text" and c.get("text"):
                    yield TranscriptEntry("assistant", c["text"], ts)
    elif etype == "patch_apply_end":
        stdout = obj.get("stdout", "")
        if type(stdout) is str:
            stripped = stdout.strip()
            if stripped:
                yield TranscriptEntry("tool", stripped, ts, {"kind": "output"})


def _codex_response_item(payload: dict, ts: str) -> Iterator[TranscriptEntry]:
    """Parse a response_item payload into zero or more entries.

    The caller has already established that *payload* is a dict; this is the
    hottest Codex record type, so it is not re-checked here."""
    ptype = payload.get("type", "")
    if ptype == "message":
        role = payload.get("role")
        if role == "assistant":
            content = payload.get("content")
            # Bailing out replaces substituting an empty list: an empty list
            # produced no texts, hence an empty join, hence no entry.
            if type(content) is not list:
                return
            texts = [
                c.get("text", "")
                for c in content
                if type(c) is dict and c.get("type") == "output_text"
            ]
            text = "".join(texts)
            if text:
                yield TranscriptEntry("assistant", text, ts)
        elif role == "user":
            # The era-independent record of a user turn, tagged so the dedupe
            # pass can drop it when a canonical event claims the same turn.
            # ``input_text`` is the shape here, not the ``output_text`` the
            # assistant side uses; an ``input_image`` part carries no text and
            # so yields nothing, though the record still counts as activity
            # for discovery.
            content = payload.get("content")
            if type(content) is not list:
                return
            texts = [
                c.get("text", "")
                for c in content
                if type(c) is dict and c.get("type") == "input_text"
            ]
            text = "".join(texts)
            if text:
                yield TranscriptEntry("user", text, ts, _CODEX_RESPONSE_USER)
        # ``developer`` messages carry the system prompt: no readable turn.
    elif ptype in ("function_call", "custom_tool_call"):
        name = payload.get("name", "?")
        args = (
            payload.get("arguments")
            if ptype == "function_call"
            else payload.get("input")
        )
        if type(args) is not str:
            args = _json_compact(args or {})
        yield TranscriptEntry(
            "tool", f"{name}({args})", ts, {"kind": "call", "tool": name}
        )
    elif ptype in ("function_call_output", "custom_tool_call_output"):
        output = payload.get("output")
        if output is None:
            return
        if type(output) is not str:
            output = _codex_structured_output(output)
        if output:
            # One C-level prefix test is all a non-shell output pays.
            if output.startswith(_CODEX_EXIT_HEADS):
                yield TranscriptEntry("tool", output, ts, _codex_exit_metadata(output))
            else:
                yield TranscriptEntry("tool", output, ts, {"kind": "output"})
    # reasoning / system payloads: no readable content, skipped.


def _codex_structured_output(output) -> str:
    """The readable text of a tool output Codex recorded as something other
    than a string.

    Code-mode tools write a list of content items --
    ``[{"type": "input_text", "text": "Script completed\\n..."}, ...]`` --
    and serializing that list whole put JSON-escaped text into every snippet
    and every read: literal ``\\n`` where the tool printed line breaks, which
    two agents in a dogfood run reported as double-escaped output. The texts
    are joined as a user message's are. Any other item, an image above all,
    is left out rather than letting its base64 into the search index. A shape
    with no text items at all keeps the old compact JSON, so nothing that
    used to be readable goes blank.
    """
    if type(output) is list:
        texts = [
            c["text"] for c in output if type(c) is dict and type(c.get("text")) is str
        ]
        if texts:
            return "".join(texts)
    return _json_compact(output)


# How a Codex tool output that ran exactly one command reports its exit code,
# by era, measured over sampled rollouts on one machine (2026-11-01):
#   legacy shell tool   '{"output": ..., "metadata": {"exit_code": N, ...}}'
#   exec_command        'Exit code: N\nWall time: ...\nOutput:\n...'
#   unified exec        'Chunk ID: ...\nWall time: ...\nProcess exited with
#                       code N\n...', sometimes behind a 'Command: ...' line
# One command per output, so the output's exit code is the command's and a
# boolean on it is accurate. Code-mode scripts are different -- many commands
# per output, which only ever says "Script completed" -- and are covered by
# their CommandExecution items instead. No rollout sampled carries both.
_CODEX_EXIT_HEADS = ('{"output":', "Exit code: ", "Chunk ID: ", "Command: ")
_CODEX_EXIT_INT = re.compile(r"-?\d+")
_CODEX_EXIT_JSON = re.compile(r'"exit_code"\s*:\s*')
_CODEX_EXITED = "\nProcess exited with code "
# The header block is short; the output after it can be megabytes and may
# itself print "Process exited with code", so the search stops at its start.
_CODEX_EXIT_HEADER_MAX = 4096


def _codex_exit_metadata(output: str) -> dict:
    """Metadata for a single-command tool output: its exit code, and
    ``is_error`` when that is non-zero, as Claude marks a failed tool.

    str.find and an anchored match rather than a regex search over the
    header: timed at under half the cost, on a path every older shell output
    takes."""
    if output[0] == "{":
        # The metadata object follows the escaped output string, inside which
        # a bare '"exit_code"' cannot occur, so the last one is the real one.
        # Codex writes it compact; a space after the colon is tolerated.
        at = output.rfind('"exit_code"')
        m = _CODEX_EXIT_JSON.match(output, at) if at >= 0 else None
        at = m.end() if m is not None else -1
    elif output[0] == "E":  # "Exit code: N", the one shape that leads with it
        at = 11
    else:
        end = output.find("\nOutput:", 0, _CODEX_EXIT_HEADER_MAX)
        at = output.find(_CODEX_EXITED, 0, end if end >= 0 else _CODEX_EXIT_HEADER_MAX)
        at = at + 26 if at >= 0 else -1
    m = _CODEX_EXIT_INT.match(output, at) if at >= 0 else None
    if m is None:
        return {"kind": "output"}
    code = int(m.group())
    if code:
        return {"kind": "output", "exit_code": code, "is_error": True}
    return {"kind": "output", "exit_code": code}


def _codex_item_completed(payload: dict, ts: str) -> Iterator[TranscriptEntry]:
    """Parse a paginated-era ``item_completed`` event into zero or one entry.

    ``UserMessage`` is taken, and a ``CommandExecution`` that failed (see
    _codex_failed_command). An ``AgentMessage`` item repeats the response
    item the model returned, which is already parsed above, and every other
    TurnItem (reasoning, a command that succeeded, file change) either
    duplicates a response item or carries no readable turn.
    """
    item = payload.get("item")
    if type(item) is not dict:
        return
    itype = item.get("type")
    if itype == "CommandExecution":
        code = item.get("exit_code")
        if item.get("status") == "failed" or (type(code) is int and code):
            yield _codex_failed_command(item, code, ts)
        return
    if itype != "UserMessage":
        return
    content = item.get("content")
    if type(content) is not list:
        return
    # ``text`` here, a third spelling: the item vocabulary is neither the
    # response item's ``input_text`` nor the assistant's ``output_text``. A
    # ``skill`` part names an invoked skill and carries no user prose.
    texts = [
        c.get("text", "")
        for c in content
        if type(c) is dict and c.get("type") == "text"
    ]
    text = "".join(texts)
    if text:
        yield TranscriptEntry("user", text, ts)


def _codex_failed_command(item: dict, code, ts: str) -> TranscriptEntry:
    """A command a code-mode script ran that failed, as an entry of its own.

    A code-mode tool call runs any number of commands and its output says
    only "Script completed", so a failure inside it is invisible to anything
    reading outputs. The CommandExecution item carries it, per command. Only
    failures become entries: a successful command adds nothing the output
    does not, and an entry per command would multiply the transcript.
    """
    cmd = item.get("command")
    if type(cmd) is list:
        # ["/bin/zsh", "-lc", "herdr pane"]: the shell wrapper is noise.
        if len(cmd) == 3 and cmd[1] in ("-lc", "-c"):
            cmd = cmd[2]
        else:
            cmd = " ".join(str(c) for c in cmd)
    elif type(cmd) is not str:
        cmd = "?"
    status = f"exit {code}" if type(code) is int else "failed"
    text = f"{cmd} [{status}]"
    out = item.get("aggregated_output")
    if type(out) is str and out.strip():
        text += f"\n{out.rstrip()}"
    meta: dict = {"kind": "command", "is_error": True}
    if type(code) is int:
        meta["exit_code"] = code
    return TranscriptEntry("tool", text, ts, meta)


def _codex_user_text(payload: dict) -> str:
    """Extract user text from a user_message payload."""
    msg = payload.get("message", "")
    if type(msg) is str and msg:
        return msg
    content = payload.get("content", "")
    ctype = type(content)
    if ctype is str:
        if content:
            return content
    elif ctype is list:
        parts = [
            p.get("text", "")
            for p in content
            if type(p) is dict and p.get("type") == "input_text"
        ]
        return "\n".join(x for x in parts if x)
    return ""


# ---------------------------------------------------------------------------
# Pi JSONL parser
# ---------------------------------------------------------------------------

# The two roles Pi records as conversation. "toolResult" is handled separately
# below, and any other role is a shape this parser has not seen.
_PI_MESSAGE_ROLES = ("user", "assistant")


def _pi_entries(obj: dict) -> Iterator[TranscriptEntry]:
    """Parse a single Pi JSONL object into zero or more entries.

    Everything readable in a Pi transcript is a ``message`` line; ``session``,
    ``model_change``, ``thinking_level_change``, ``compaction`` and ``custom``
    lines are bookkeeping and fall out yielding nothing, exactly as the other
    providers' non-turn records do.
    """
    if obj.get("type") != "message":
        return
    msg = obj.get("message")
    if type(msg) is not dict:
        return
    role = msg.get("role", "")
    # The line's own timestamp, never the one nested in ``message``: that one
    # is epoch milliseconds as an int, and this field is a string everywhere
    # else. A fallback to it would put an int where _relative_time slices.
    ts = obj.get("timestamp", "")

    if role == "toolResult":
        # Pi splits a tool call and its output across two lines, so the output
        # arrives as its own message rather than inside the next user turn.
        # It still becomes a "tool" entry with kind "output", which is the
        # shape the role filter and the error pseudo-role already read.
        #
        # One entry per text part, never a join. Joining is what forces
        # codex's extra ripgrep pass: an entry whose text spans two parts of
        # one line matches a query that appears nowhere contiguously in the
        # raw bytes, so the cheap candidate scan skips a file that canonically
        # matches. Keeping parts separate means Pi needs no such pass.
        name = msg.get("toolName", "") or "?"
        meta: dict = {"kind": "output", "tool": name}
        if msg.get("isError"):
            meta["is_error"] = True
        for text in _pi_text_parts(msg.get("content", "")):
            yield TranscriptEntry("tool", text, ts, meta)
        return

    if role not in _PI_MESSAGE_ROLES:
        return

    content = msg.get("content", "")
    ctype = type(content)
    if ctype is str:
        if content:
            yield TranscriptEntry(role, content, ts)
        return
    if ctype is not list:
        return
    for block in content:
        if type(block) is not dict:
            continue
        btype = block.get("type", "")
        if btype == "text":
            text = block.get("text", "")
            if text:
                yield TranscriptEntry(role, text, ts)
        elif btype == "toolCall":
            name = block.get("name", "") or "?"
            args = block.get("arguments", {})
            text = f"{name}({_json_compact(args)})"
            yield TranscriptEntry("tool", text, ts, {"kind": "call", "tool": name})
        # "thinking" and "image" parts are skipped: reasoning is dropped for
        # every provider here, and an image part carries base64 bytes, not
        # text anyone can read or search.


def _pi_text_parts(content) -> Iterator[str]:
    """The non-empty text parts of a Pi content list, in order.

    A bare string is one part; image parts carry base64 bytes and are skipped.
    """
    ctype = type(content)
    if ctype is str:
        if content:
            yield content
        return
    if ctype is not list:
        return
    for c in content:
        if type(c) is dict and c.get("type") == "text":
            text = c.get("text", "")
            if text:
                yield text
