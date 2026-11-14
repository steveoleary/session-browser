"""Parsed transcripts, kept on disk so the next read does not parse them again.

A provider's transcript is mostly not transcript. Measured 2026-11-14 on the
author's Mac, 2,574 file-backed sessions:

    raw JSONL on disk                 6,679 MB
    the entry text search matches       499 MB   (7.5%)

The rest is base64 images, tool results stored twice, file snapshots and
encrypted reasoning -- bytes the parser reads and throws away. A stateless
search still has to read every one of them, and at that size the corpus no
longer stays in the page cache of a 16 GB machine: reading it back cost
2.3-3.5 s at about 2 GB/s on every search, whatever the query, before a single
candidate was parsed. Queries the raw prefilter cannot take at all (anything
with a digit or a dot, so every file name and commit hash) parsed the whole
corpus instead: 14.7 s.

So each session's parsed entries are written once, beside the directory where
``compressed.py`` keeps the decompressed rollouts a raw scan needs, and
everything downstream reads those instead:

* **search** scans the entry text with ripgrep. It is the canonical text, so
  there are no JSON escapes to see through and no parts the parser would have
  joined, which makes every printable-ASCII query scannable and every
  candidate a real match: the scan is exact, not merely safe, so a search
  that shows only some of its matches loads only those. 0.05-0.12 s over the
  same corpus.
* **a parse** becomes a load: 54 sessions in 0.025 s against about a second.

Two files per session, and for a few a third, in
``$XDG_CACHE_HOME/session-browser/entries`` (default ``~/.cache``), spread over
256 directories by the first byte of the name:

``<stem>.<fingerprint>.txt``
    Every entry's text, each followed by a newline, and nothing else. This is
    the file ripgrep scans, which is why it holds no framing: a role name or a
    byte count here would match queries nobody typed.
``<stem>.<fingerprint>.fold``
    Only for a session with an entry the scan above cannot decide: one holding
    a character whose casefold is longer than itself, like the sharp s in
    "Strasse" spelled the German way, which a case-insensitive scan of the
    original text will never match against "ss". Those entries are written
    here already normalized, so the same scan decides them too. Without it
    every such session was a candidate for every query: 34 sessions loaded on
    each search of the author's corpus, 0.14 s, to find nothing.
``<stem>.<fingerprint>.idx``
    A JSON header line naming the source and counting the entries, then one
    JSON line with the roles, timestamps, metadata and byte lengths that cut
    the text back into entries, and the warnings the parse raised.

The stem names the session (and everything about it that changes how its file
is read); the fingerprint is the size and mtime of every file the parse read,
plus a hash of the parser's own source. So an appended transcript, a rewritten
one and an upgraded parser all miss, and a miss is only ever a parse: nothing
here can make an answer wrong, only stale files can make it slow, and stale
files are never read because their names no longer match.

Deleting the directory loses nothing but the next search's head start.
``SESSION_BROWSER_CACHE=0`` turns the whole thing off, and every caller then
does exactly what it did before this module existed.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import logging
import os
import tempfile
import time
from collections.abc import Iterable
from functools import lru_cache
from pathlib import Path
from typing import NamedTuple

log = logging.getLogger(__name__)

# Bumped when the files' own layout changes. A change to what the parser
# produces needs no bump: its source is hashed into every fingerprint.
FORMAT = 2

TEXT_SUFFIX = ".txt"
INDEX_SUFFIX = ".idx"
FOLD_SUFFIX = ".fold"

# Parsed text can hold a lone surrogate (a "\ud83d" escape with no partner),
# which strict UTF-8 refuses to encode. surrogatepass round-trips it exactly.
_ENCODING_ERRORS = "surrogatepass"

# A sweep opens every index to read one line, so it is rationed: at most once
# in this many seconds. A day bounds how long a transcript its provider has
# pruned can outlive it here, on a machine that is searched at all.
SWEEP_EVERY = 24 * 3600
_SWEEP_MARKER = ".swept"

_OFF = frozenset({"0", "off", "false", "no"})


def enabled() -> bool:
    """Whether the cache may be read and written.

    Read from the environment on every call rather than once at import: the
    switch has to reach worker processes, and a test or a benchmark has to be
    able to flip it without reloading the package.
    """
    return os.environ.get("SESSION_BROWSER_CACHE", "").strip().lower() not in _OFF


def cache_dir() -> Path:
    base = os.environ.get("XDG_CACHE_HOME")
    root = Path(base) if base else Path.home() / ".cache"
    return root / "session-browser" / "entries"


# The modules whose code decides what a parse yields. transcript.py holds the
# parsers; it borrows record classifiers from discovery.py and reads through
# compressed.py. This module is in the list for its own layout, and
# scan_memo.py, which shares this version, for its own.
_PARSER_SOURCES = (
    "transcript.py",
    "discovery.py",
    "compressed.py",
    "entry_cache.py",
    "scan_memo.py",
)


@lru_cache(maxsize=1)
def parser_version() -> str:
    """A hash of the parser's source, so a changed parser never reads old work.

    Derived rather than declared. A version constant has to be remembered by
    whoever changes what a parse yields, and the failure of forgetting it is
    silent: yesterday's entries served as today's. Hashing five files costs
    about half a millisecond once per process and cannot be forgotten. The
    price is that an edit which changes no output still discards the cache --
    a few seconds of parsing on the next search, which is the cheap side to
    err on.
    """
    digest = hashlib.blake2b(f"format {FORMAT}\n".encode(), digest_size=8)
    here = Path(__file__).resolve().parent
    for name in _PARSER_SOURCES:
        try:
            digest.update((here / name).read_bytes())
        except OSError:
            # No readable source (a zipped install): the layout version alone
            # then stands in, and FORMAT must be bumped by hand for a parser
            # change. Said here because it is the one case where it must be.
            digest.update(name.encode())
    return digest.hexdigest()


class Ref(NamedTuple):
    """Where one version of one session's parse lives, present or not."""

    text: str
    index: str
    source: str

    @property
    def fold(self) -> str:
        """The optional third file; see the module docstring."""
        return self.text[: -len(TEXT_SUFFIX)] + FOLD_SUFFIX

    @property
    def names(self) -> tuple[str, str, str]:
        return (
            os.path.basename(self.text),
            os.path.basename(self.index),
            os.path.basename(self.fold),
        )


def locate(
    provider: str,
    session_id: str,
    content_path: str,
    subagent_kind: str | None = None,
    segments: Iterable[str] = (),
    *,
    root: str | None = None,
) -> Ref | None:
    """The cache files for a session as its sources stand now, or None.

    None means "do not cache this": no file to fingerprint, or one that cannot
    be statted, in which case the parse is left to report why. The stat is
    taken here, before any parse, on purpose. A transcript appended to while
    it is being parsed is then stored under the fingerprint of its *older*
    state, which nothing will ever look up again, rather than the newer state
    being credited with entries it may not all hold.
    """
    if not content_path:
        return None
    segments = tuple(segments)
    try:
        stats = [os.stat(p) for p in (content_path, *segments)]
    except OSError:
        return None
    # Everything that changes how the same bytes are read belongs in the stem:
    # a thread_spawn child skips its spawner's copied history, a guardian's
    # prompt is relabelled, a rewound thread reads several files in order.
    stem = hashlib.blake2b(
        "\0".join(
            (provider, session_id, content_path, subagent_kind or "", *segments)
        ).encode("utf-8", _ENCODING_ERRORS),
        digest_size=12,
    ).hexdigest()
    fingerprint = hashlib.blake2b(
        "|".join(
            (parser_version(), *(f"{st.st_size}-{st.st_mtime_ns}" for st in stats))
        ).encode(),
        digest_size=8,
    ).hexdigest()
    if root is None:
        root = os.fspath(cache_dir())
    base = os.path.join(root, stem[:2], f"{stem}.{fingerprint}")
    return Ref(base + TEXT_SUFFIX, base + INDEX_SUFFIX, content_path)


def listing() -> set[str]:
    """Every file path in the cache, from one walk of its 256 directories.

    A search asks about thousands of sessions at once; two ``stat`` calls each
    is thousands of syscalls where this is a few hundred ``scandir`` batches.
    """
    found: set[str] = set()
    root = os.fspath(cache_dir())
    try:
        shards = [e.path for e in os.scandir(root) if e.is_dir(follow_symlinks=False)]
    except OSError:
        return found
    for shard in shards:
        try:
            found.update(e.path for e in os.scandir(shard))
        except OSError:
            continue
    return found


def load(ref: Ref, make) -> tuple[list, list[str]] | None:
    """The entries and warnings stored at *ref*, or None if there are none.

    *make* builds one entry from ``(role, text, timestamp, metadata)``; it is
    passed in so this module needs nothing from the parser it serves. Anything
    short of a whole, self-consistent pair is a miss -- a missing file, a
    truncated one, lengths that do not add up -- and a miss is only a parse.
    """
    try:
        with open(ref.index, "rb") as fh:
            fh.readline()  # the header is for sweeps; a load never needs it
            body = json.loads(fh.read())
        with open(ref.text, "rb") as fh:
            data = fh.read()
        roles, sizes, stamps, metas = (
            body["roles"],
            body["sizes"],
            body["stamps"],
            body["metas"],
        )
        if sum(sizes) + len(sizes) != len(data) or not (
            len(roles) == len(sizes) == len(stamps) == len(metas)
        ):
            return None
        entries = []
        add = entries.append
        pos = 0
        for role, size, stamp, meta in zip(roles, sizes, stamps, metas, strict=True):
            end = pos + size
            add(
                make(role, data[pos:end].decode("utf-8", _ENCODING_ERRORS), stamp, meta)
            )
            pos = end + 1
        return entries, list(body["warnings"])
    except (OSError, ValueError, KeyError, TypeError):
        return None


def count(ref: Ref) -> int | None:
    """How many entries are stored at *ref*, read from the header line alone.

    A listing wants each session's length and nothing else; this answers from
    the first hundred-odd bytes of the index where a load would decode every
    entry's text to count them. None when there is no index to ask.
    """
    try:
        with open(ref.index, "rb") as fh:
            entries = json.loads(fh.readline())["entries"]
    except (OSError, ValueError, KeyError, TypeError):
        return None
    return entries if type(entries) is int else None


def store(
    ref: Ref, entries: Iterable, warnings: Iterable[str], fold: bytes = b""
) -> bool:
    """Write *entries* (anything with role/text/timestamp/metadata) at *ref*.

    *fold* is the normalized text of the entries a scan of the original text
    cannot decide, if there are any; it is stored as given.

    Returns whether the files are now on disk. Never raises for a cache that
    cannot be written -- a read-only home, a full disk, metadata JSON cannot
    carry -- because the caller already holds the answer and only the next
    search's speed is at stake.

    Each file is written whole and renamed into place, the text file last: it
    is the one whose presence says a session is cached, so by the time it
    exists everything beside it does too, while a stray index is merely
    ignored.
    """
    try:
        parts: list[bytes] = []
        roles: list[str] = []
        sizes: list[int] = []
        stamps: list[str] = []
        metas: list[dict | None] = []
        for entry in entries:
            raw = entry.text.encode("utf-8", _ENCODING_ERRORS)
            parts.append(raw)
            parts.append(b"\n")
            roles.append(entry.role)
            sizes.append(len(raw))
            stamps.append(entry.timestamp)
            metas.append(entry.metadata)
        header = json.dumps(
            {"format": FORMAT, "source": ref.source, "entries": len(sizes)}
        )
        body = json.dumps(
            {
                "roles": roles,
                "sizes": sizes,
                "stamps": stamps,
                "metas": metas,
                "warnings": list(warnings),
            },
            ensure_ascii=False,
            separators=(",", ":"),
        )
        index = f"{header}\n{body}".encode("utf-8", _ENCODING_ERRORS)
        # A round trip that would not give back what was stored is refused
        # here, where it costs one more decode of an index a few kilobytes
        # long, rather than discovered later as a search that disagrees with
        # a parse: JSON turns a tuple into a list and a non-string key into a
        # string, silently, and NaN comes back unequal to itself.
        decoded = json.loads(body)
        if (
            decoded["metas"] != metas
            or decoded["stamps"] != stamps
            or decoded["roles"] != roles
        ):
            return False
        shard = os.path.dirname(ref.text)
        os.makedirs(shard, mode=0o700, exist_ok=True)
        _write_atomic(ref.index, index, shard)
        if fold:
            _write_atomic(ref.fold, fold, shard)
        _write_atomic(ref.text, b"".join(parts), shard)
    except (OSError, ValueError, TypeError, AttributeError) as exc:
        log.debug("could not cache parsed entries for %s: %s", ref.source, exc)
        return False
    _drop_other_versions(ref)
    return True


def _write_atomic(target: str, data: bytes, directory: str) -> None:
    fd, tmp = tempfile.mkstemp(dir=directory, prefix=".partial-")
    try:
        with os.fdopen(fd, "wb") as out:
            out.write(data)
        os.replace(tmp, target)
    except BaseException:
        with contextlib.suppress(OSError):
            os.unlink(tmp)
        raise


def _drop_other_versions(ref: Ref) -> None:
    """Remove what an earlier fingerprint of the same session left behind.

    One scandir of a directory holding 1/256th of the cache, so writing the
    whole corpus does not rescan the whole cache once per session.
    """
    shard = os.path.dirname(ref.text)
    keep = ref.names
    prefix = keep[0].split(".", 1)[0] + "."
    try:
        stale = [
            e.path
            for e in os.scandir(shard)
            if e.name.startswith(prefix) and e.name not in keep
        ]
    except OSError:
        return
    for path in stale:
        with contextlib.suppress(OSError):
            os.unlink(path)


def sweep() -> int:
    """Remove entries whose source file no longer exists; returns how many.

    Deleting a session must delete what was cached from it: the text file
    here is that session's words, and "permanently delete" would be a false
    promise if a copy survived in a cache directory. Checking sources rather
    than being told which session went covers a parent's subagents and
    whatever a provider removed along with it. One line read per cached
    session -- about 50 ms for 2,574 of them.
    """
    removed = 0
    for path in sorted(listing()):
        if not path.endswith(INDEX_SUFFIX):
            continue
        try:
            with open(path, "rb") as fh:
                source = json.loads(fh.readline())["source"]
        except (OSError, ValueError, KeyError, TypeError):
            source = None  # unreadable is as good as orphaned
        if source is not None and os.path.exists(source):
            continue
        base = path[: -len(INDEX_SUFFIX)]
        for doomed in (path, base + TEXT_SUFFIX, base + FOLD_SUFFIX):
            with contextlib.suppress(OSError):
                os.unlink(doomed)
        removed += 1
    return removed


def sweep_if_due(now: float | None = None) -> int:
    """:func:`sweep`, at most once every ``SWEEP_EVERY`` seconds.

    Providers prune their own history -- Claude Code's is a rolling window --
    and nothing else would ever remove what was cached for a transcript that
    has since aged out. Rationed by a marker file's mtime, so asking costs a
    search one stat and the sweep itself runs once a day.
    """
    root = cache_dir()
    marker = root / _SWEEP_MARKER
    now = time.time() if now is None else now
    try:
        if now - marker.stat().st_mtime < SWEEP_EVERY:
            return 0
    except OSError:
        pass  # never swept
    try:
        root.mkdir(mode=0o700, parents=True, exist_ok=True)
        marker.touch()
        os.utime(marker, (now, now))
    except OSError:
        return 0  # cannot record a sweep, so do not repeat one every search
    return sweep()
