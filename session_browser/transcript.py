"""Shared transcript service for parsing/rendering/searching agent sessions.

Parsers must not truncate — display-level limits are the TUI's concern.
"""

from __future__ import annotations

# Per-session scans are I/O-bound; a small pool overlaps reads. Gains
# plateau around 4 workers (JSON parsing is GIL-bound), 8 leaves headroom.
_SEARCH_THREADS = 8

import json
import multiprocessing
import os
import re
import shutil
import sqlite3
import subprocess
import sys
import threading
from bisect import bisect_right
from collections import OrderedDict
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

from session_browser.discovery import Session, _opencode_db_path


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
        with open(content_path, encoding="utf-8", errors="replace") as fh:
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
    transcript: Transcript, *, entry_indices: list[int] | None = None
) -> str:
    """Readable, Markdown-compatible transcript body.

    *entry_indices* are the absolute positions of a role-filtered subset;
    each block is then prefixed with its index so it can be fed back to
    get --entries or matched against search's entry_index."""
    if entry_indices is None:
        blocks = [f"{entry_label(e)}: {e.text}" for e in transcript.entries]
    else:
        blocks = [
            f"[{i}] {entry_label(e)}: {e.text}"
            for i, e in zip(entry_indices, transcript.entries, strict=True)
        ]
    return "\n\n".join(blocks) if blocks else "(empty session)"


def render_markdown(
    transcript: Transcript,
    *,
    total_entries: int | None = None,
    entry_range: tuple[int, int] | None = None,
    entry_indices: list[int] | None = None,
    roles: list[str] | None = None,
) -> str:
    """Markdown document: metadata header + transcript body.

    *total_entries*/*entry_range* describe a windowed transcript (get
    --entries/--head/--tail): the header then says which slice this is.
    *roles*/*entry_indices* describe a role-filtered transcript (get
    --role): the header names the roles and each block carries its
    absolute entry index.
    """
    s = transcript.session
    total = total_entries if total_entries is not None else len(transcript.entries)
    if roles is not None:
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
        f"- Summary: {s.summary or '—'}",
        f"- CWD: {s.cwd or '—'}",
        f"- Branch: {s.branch or '—'}",
        f"- Repository: {s.repository or '—'}",
        f"- Created: {s.created_at or '—'}",
        f"- Updated: {s.updated_at or '—'}",
        entries_line,
    ]
    if transcript.warnings:
        lines.append(f"- Parse warnings: {len(transcript.warnings)}")
    if roles is not None and not transcript.entries:
        body = f"(no entries with roles: {', '.join(roles)})"
    else:
        body = render_text(transcript, entry_indices=entry_indices)
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
        "session_id": s.id,
        "summary": s.summary,
        "cwd": s.cwd,
        "branch": s.branch,
        "repository": s.repository,
        "created_at": s.created_at,
        "updated_at": s.updated_at,
        "duration_seconds": session_duration_seconds(s),
    }


def transcript_to_dict(
    t: Transcript, *, entry_indices: list[int] | None = None
) -> dict:
    """*entry_indices* (role-filtered subset) adds each entry's absolute
    position as "index", matching get --entries / search entry_index."""
    entries = []
    for pos, e in enumerate(t.entries):
        item = {
            "role": e.role,
            "text": e.text,
            "timestamp": e.timestamp or None,
            "metadata": e.metadata,
        }
        if entry_indices is not None:
            item = {"index": entry_indices[pos], **item}
        entries.append(item)
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
# Bulk search: provider-aware, stateless, no persistent index
# ---------------------------------------------------------------------------
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
    """The raw file a prefilter may scan for *session*, if any."""
    if session.provider in ("claude", "codex"):
        return Path(session.content_path) if session.content_path else None
    return None


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
    paths = sorted(
        {p for s in sessions if (p := _prefilter_file(s)) is not None and p.is_file()},
        key=os.fspath,
    )
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

    codex_paths = sorted(
        {
            p
            for s in sessions
            if s.provider == "codex"
            and (p := _prefilter_file(s)) is not None
            and p.is_file()
        },
        key=os.fspath,
    )

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


def _parse_worker(args: tuple[str, str, str, str | list[str]]):
    """Canonically parse one file-backed session in a worker process.

    Returns the result's fields rather than a SessionSearchResult, because the
    parent must re-attach its *own* Session object: callers read summary, cwd
    and branch off ``result.session``, and a worker only receives the three
    fields it needs to read the transcript.  Never raises for the same reason
    :func:`search_session` never does.
    """
    provider, sid, path, query = args
    r = search_session(Session(id=sid, provider=provider, content_path=path), query)
    return r.match_count, r.matches, r.warnings, r.unreadable, r.total_entries


_PROC_FALLBACK = object()  # pool unusable; caller should use the thread path
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


def _probe_in_processes(
    pairs: list[tuple[int, Session]],
    query: str | list[str],
    cancelled: Callable[[], bool] | None,
) -> dict[int, SessionSearchResult] | object:
    """Parse *pairs* across worker processes, honouring *cancelled*.

    Work is submitted through a bounded window rather than all at once, and
    ``cancelled`` is polled between completions.  On cancellation the queued
    futures are dropped and every session that had not finished reports an
    empty result — exactly what the thread probe returns when it observes
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
    out: dict[int, SessionSearchResult] = {}
    remaining = iter(pairs)
    ex = None
    try:
        ex = ProcessPoolExecutor(max_workers=_PROC_WORKERS, mp_context=_process_ctx())
        inflight: dict[Future, tuple[int, Session]] = {}

        def feed() -> None:
            while len(inflight) < _PROC_WINDOW:
                nxt = next(remaining, None)
                if nxt is None:
                    return
                i, s = nxt
                inflight[
                    ex.submit(_parse_worker, (s.provider, s.id, s.content_path, query))
                ] = (i, s)

        feed()
        while inflight:
            done, _ = wait(list(inflight), timeout=0.05, return_when=FIRST_COMPLETED)
            if cancelled is not None and cancelled():
                for fut in inflight:
                    fut.cancel()
                break
            for fut in done:
                i, s = inflight.pop(fut)
                count, matches, warns, unreadable, total = fut.result()
                out[i] = SessionSearchResult(
                    s,
                    count,
                    matches,
                    warns,
                    unreadable,
                    entries=None,
                    total_entries=total,
                )
            feed()
    except Exception:
        # A broken pool, an unpicklable payload, or an environment that cannot
        # spawn at all: discard partial work and let the caller re-run these
        # sessions in threads, where the answer is identical.
        return _PROC_FALLBACK
    finally:
        if ex is not None:
            ex.shutdown(wait=True, cancel_futures=True)
    # Sessions dropped by cancellation report exactly what a cancelled thread
    # probe reports.
    for i, s in pairs:
        out.setdefault(i, SessionSearchResult(s))
    return out


# (phase, done, total), where phase is "scanning" while the prefilter is
# ruling sessions out and "reading" while survivors are parsed. Called from
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

    def must_parse(s: Session) -> bool:
        """Whether the prefilter leaves *s* needing a canonical parse.

        Sole owner of that decision: both the thread probe below and the
        process path's parent-side partition ask this, so the two cannot
        drift apart on a subtlety like a file that vanished between the
        candidate scan and the probe.
        """
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

    def probe(s: Session) -> SessionSearchResult:
        if cancelled is not None and cancelled():
            return SessionSearchResult(s)
        if not must_parse(s):
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
        # The candidate scan runs on this thread while the DB scan runs on its
        # worker.  It reads only the file corpus and the DB scan reads only the
        # database, so neither can observe the other and the results are
        # unchanged whichever finishes first; previously the DB scan could not
        # even be submitted until rg had returned, so its cost was added to
        # rg's instead of hidden behind it.  Escaped slashes need an in-process
        # transformed haystack; all other safe needles share one native scan
        # across the file corpus.  The probes below do consume rg_candidates,
        # so they still start only after it is assigned.
        if needles is not None and not any("/" in n for n in needles):
            rg_candidates = _rg_candidate_paths([sessions[i] for i in file_ix], needles)
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
                len(file_ix)
                if rg_candidates is None
                else sum(1 for i in file_ix if must_parse(sessions[i]))
            )
            progress("reading", 0, read_total)
        if file_ix:
            # Only the ids/snippets shape goes to processes. keep_entries=True
            # asks for every entry of every matching session, which is the
            # whole corpus text for a broad query -- and it is what the TUI's
            # transcript cache consumes, so it stays where the objects already
            # live. A missing candidate scan also stays on threads: the
            # partition below would then have to run the expensive
            # _file_may_match serially in the parent, reading each file once
            # here and again in the worker.
            parsed = None
            if (
                not keep_entries
                and rg_candidates is not None
                and len(file_ix) >= _PROC_MIN_CANDIDATES
            ):
                proc_ix = [i for i in file_ix if must_parse(sessions[i])]
                if len(proc_ix) >= _PROC_MIN_CANDIDATES:
                    parsed = _probe_in_processes(
                        [(i, sessions[i]) for i in proc_ix], query, cancelled
                    )
                    if parsed is _PROC_FALLBACK:
                        parsed = None  # no pool here; threads answer
                    else:
                        # Prefiltered-out sessions never left the parent.
                        for i in set(file_ix) - set(proc_ix):
                            parsed[i] = SessionSearchResult(sessions[i])
            if parsed is None:
                results.update(
                    zip(
                        file_ix,
                        ex.map(probe, [sessions[i] for i in file_ix]),
                        strict=True,
                    )
                )
            else:
                results.update(parsed)
        if oc_fut is not None:
            # The DB scan overlaps the file probes but routinely outlives
            # them -- its prefilter reads the whole part table -- so it
            # reports its own progress from inside, and the display switches
            # to it once the file phase is done.
            results.update(zip(opencode_ix, oc_fut.result(), strict=True))
    return [results[i] for i in range(len(sessions))]


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

    def emit(sid: str, rows: list[tuple]) -> None:
        for i in by_sid.get(sid, ()):
            results[i] = _search_entries(
                sessions[i],
                _opencode_entries_from_rows(rows),
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
                cur = conn.execute(
                    "SELECT session_id, CAST(data AS BLOB) FROM part "
                    "WHERE session_id IN (SELECT value FROM json_each(?))",
                    (json.dumps(scan_sids),),
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
                cur = conn.execute(
                    "SELECT m.session_id, m.id, m.data, p.data "
                    "FROM message m "
                    "LEFT JOIN part p ON p.message_id = m.id "
                    "WHERE m.session_id IN (SELECT value FROM json_each(?)) "
                    "ORDER BY m.session_id, m.time_created, m.id, "
                    "p.time_created, p.id",
                    (json.dumps(scan_sids),),
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


def iter_entries(session: Session, warnings: list[str]) -> Iterator[TranscriptEntry]:
    """Yield transcript entries for *session*, collecting warnings."""
    if session.provider == "claude":
        path = _existing_file(session)
        yield from _parse_jsonl(path, warnings, _claude_entries)
    elif session.provider == "codex":
        path = _existing_file(session)
        yield from _dedupe_adjacent_assistant(
            _parse_jsonl(path, warnings, _codex_entries)
        )
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


def _parse_opencode(db_path: Path, session_id: str) -> Iterator[TranscriptEntry]:
    """Read an opencode SQLite DB and yield transcript entries."""
    try:
        conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True, timeout=3)
        rows = conn.execute(
            "SELECT m.id, m.data, p.data "
            "FROM message m "
            "LEFT JOIN part p ON p.message_id = m.id "
            "WHERE m.session_id = ? "
            "ORDER BY m.time_created, m.id, p.time_created, p.id",
            (session_id,),
        ).fetchall()
        conn.close()
    except sqlite3.Error as exc:
        raise TranscriptUnreadable(f"opencode database error: {exc}") from exc
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
    """Validate and return the content path for *session*."""
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
) -> Iterator[TranscriptEntry]:
    """Stream *path* line-by-line, yielding entries from *event_fn*."""
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
    with open(path, "rb", buffering=_READ_BUFFER) as fh:
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
                    yield TranscriptEntry(
                        "tool", text, ts, {"kind": "call", "tool": name}
                    )

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
                        meta: dict = {"kind": "output"}
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

    # No else: "summary" and the bookkeeping types fall out here with nothing
    # yielded, which is what the explicit ``etype == "summary"`` early-out used
    # to do — it never guarded anything, since no branch below claimed it.


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


def _dedupe_adjacent_assistant(
    entries: Iterator[TranscriptEntry],
) -> Iterator[TranscriptEntry]:
    """Drop an assistant entry that repeats the previous one verbatim.

    Codex records each assistant message twice — an event_msg
    ``agent_message`` and a ``response_item`` message, milliseconds apart —
    so every assistant turn parsed naively appears as two identical
    adjacent entries. Keep the first of each pair. Only *adjacent*
    assistant repeats are dropped: any intervening entry (a user turn, a
    tool call) resets the comparison, so a genuinely repeated answer
    across turns survives. Deduping both sources beats parsing just one:
    a file that carries only one of the two event kinds still yields its
    assistant turns."""
    prev_text: str | None = None
    for e in entries:
        if e.role == "assistant":
            if e.text == prev_text:
                continue
            prev_text = e.text
        else:
            prev_text = None
        yield e


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
    # Assistant-only on purpose: user turns also arrive as user_message
    # events, so taking both would duplicate them.
    if ptype == "message" and payload.get("role") == "assistant":
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
            output = _json_compact(output)
        if output:
            yield TranscriptEntry("tool", output, ts, {"kind": "output"})
    # reasoning / developer / system payloads: no readable content, skipped.


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
