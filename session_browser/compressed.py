"""Read zstd-compressed transcripts as the plain JSONL every reader expects.

Codex compresses rollouts in the background. codex-cli 0.164.0 ships a
"rollout compression worker" that rewrites ``rollout-*.jsonl`` as
``rollout-*.jsonl.zst`` in place, keeping the original mtime. Its first run on
the author's work Mac, on 2026-10-20, compressed 684 of 709 rollouts
(8 Jun - 9 Oct): 532 MB down to 104 MB. Codex's own ``threads`` index still
names each one by its ``.jsonl`` path, so discovery has to look for the
compressed spelling as well.

Two kinds of reader, and they want different things.

A parse reads a rollout once, start to end, and with the entry cache on that
is the last time the rollout is read while it stays the same: what search,
``get`` and the TUI read afterwards is the cached parse. So :func:`open_binary`
decompresses as it reads, and keeps nothing.

A scan is different. With the entry cache off -- and for the few sessions it
cannot hold -- every search scans the raw files, and over those 684 files
decompress-and-find cost 0.9-1.2s per pass in Python and 1.2s through
``rg -z``, against a whole-corpus search measured in tenths of a second. So
:func:`readable` decompresses a file once, into a copy keyed by its size and
mtime, and ripgrep and the byte prefilter read that copy as they would any
other rollout; a parse that finds a copy already there reads it too. Should
Codex ever rewrite a compressed file, its size or mtime moves and the next
read makes a new copy.

The cost of a copy is disk: about five times the compressed size. Copies made
for a parse, before the entry cache existed, came to 1,400 files and 1.8 GB on
the author's Mac (2026-11-14), and nothing read them again. They live under
``$XDG_CACHE_HOME`` (default ``~/.cache``) in ``session-browser/zst``; a copy
nothing has wanted for a day is swept away (:func:`sweep_if_due`), and
deleting them all loses nothing but the next scan's head start.
"""

from __future__ import annotations

import contextlib
import hashlib
import io
import os
import shutil
import sys
import tempfile
import time
from pathlib import Path

if sys.version_info >= (3, 14):
    from compression import zstd
else:  # pragma: no cover - exercised on the older interpreters only
    from backports import zstd

SUFFIX = ".zst"


def cache_dir() -> Path:
    base = os.environ.get("XDG_CACHE_HOME")
    return (Path(base) if base else Path.home() / ".cache") / "session-browser" / "zst"


def _stem(path: Path) -> str:
    """The cache-file prefix shared by every version of *path*."""
    return hashlib.sha256(os.fsencode(os.path.abspath(path))).hexdigest()[:24]


_known: dict[str, tuple[int, int, Path]] = {}


def _copy(path: Path) -> tuple[Path, float | None, os.stat_result]:
    """Where the copy of *path* as it stands now goes, when that copy was
    last used (None if there is none), and the stat that named it. Raises
    OSError when *path* cannot be statted.
    """
    # Search asks for the same file three or four times per query (candidate
    # scan, partition, parse), so the answer is remembered for as long as the
    # source's size and mtime say it still holds and the copy is still there.
    # That halves a lookup's cost; across 684 rollouts, ~7ms per pass to ~3ms.
    # One stat of the copy answers both whether it is there and how recently
    # it was used, which is the same syscall the isfile() it replaced made.
    key = os.fspath(path)
    st = os.stat(key)
    known = _known.get(key)
    if known is not None and known[0] == st.st_size and known[1] == st.st_mtime_ns:
        target = known[2]
    else:
        target = cache_dir() / f"{_stem(path)}-{st.st_size}-{st.st_mtime_ns}.jsonl"
    try:
        used = os.stat(target).st_mtime
    except OSError:
        return target, None, st
    _known[key] = (st.st_size, st.st_mtime_ns, target)
    return target, used, st


def readable(path: Path) -> Path:
    """*path* itself, or its decompressed copy when it is a ``.zst`` file.

    For a reader that needs a file on disk -- ripgrep, and the byte scans
    that stand in for it; a reader that only wants the bytes once should use
    :func:`open_binary`, which makes no copy. The copy is made on first use
    and written atomically, so a reader in another thread or process sees
    either no copy or a whole one. Copies of earlier versions of the same
    file are removed when a new one is made. Raises OSError when the file
    cannot be read or does not decompress.
    """
    if not path.name.endswith(SUFFIX):
        return path
    target, used, st = _copy(path)
    if used is None:
        _decompress(path, target)
        _known[os.fspath(path)] = (st.st_size, st.st_mtime_ns, target)
    elif time.time() - used > SWEEP_EVERY / 2:
        # A copy's mtime says when it was last wanted, which is what the
        # sweep reads. Refreshed twice a day at most, so a scan that keeps
        # using a copy keeps it, at the cost of one utime per copy per half
        # day; a copy only a parse ever read is left to age out.
        with contextlib.suppress(OSError):
            os.utime(target)
    return target


def open_binary(path: Path, mode: str = "rb", buffering: int = -1):
    """*path* opened for reading as plain JSONL bytes, with no copy made.

    A plain file is opened as it is. A compressed one is read from its copy
    if :func:`readable` has made one already -- a scan of it has, or the
    entry cache is off -- and otherwise decompressed as it is read, which
    for 400 real rollouts cost 0.11 s against 0.04 s from copies
    (2026-11-14): a parse is the one read most of them ever get, so the copy
    it used to make was five times the file's size kept for nothing.

    Raises OSError when the file cannot be opened, and OSError again, not
    zstd's own error, when it stops decompressing part-way, so a corrupt
    rollout reads as unreadable exactly as a missing one does. Takes the
    arguments ``open`` takes for that, so a reader can be handed either.
    """
    if mode != "rb":
        raise ValueError(f"open_binary reads; mode {mode!r} is not 'rb'")
    if not os.fspath(path).endswith(SUFFIX):
        return open(path, "rb", buffering=buffering)
    # Reading a copy does not count as wanting it: only readable() refreshes
    # one, so a copy left over from a parse expires however often the
    # rollout is parsed again.
    target, used, _st = _copy(Path(path))
    if used is not None:
        try:
            return open(target, "rb", buffering=buffering)
        except FileNotFoundError:
            pass  # swept between the look and the open: read the source
    return _stream(Path(path), buffering)


class _Decompressing(io.RawIOBase):
    """A compressed rollout as the plain bytes it holds, failing as OSError.

    Handed a source already open, and never asked to open one: an open that
    failed inside the constructor would leave a half-made reader whose
    finalizer then failed as well, printing a traceback for what is only an
    unreadable file.
    """

    def __init__(self, source, path: Path) -> None:
        self._path = path
        self._source = source

    def readable(self) -> bool:
        return True

    def readinto(self, buffer) -> int:
        try:
            return self._source.readinto(buffer)
        except (zstd.ZstdError, EOFError) as exc:
            # EOFError is a truncated or empty file, which is neither zstd's
            # error nor an OSError, and so was caught by nothing at all.
            raise OSError(f"cannot decompress {self._path}: {exc}") from exc

    def close(self) -> None:
        if not self.closed:
            self._source.close()
        super().close()


def _stream(path: Path, buffering: int = -1) -> io.BufferedReader:
    # Buffered over the raw stream rather than read through ZstdFile itself:
    # ZstdFile splits lines in Python, one call per line, and the reader
    # around it does that in C -- 0.23 s against 0.11 s over 400 rollouts.
    size = buffering if buffering > 1 else io.DEFAULT_BUFFER_SIZE
    source = zstd.ZstdFile(path, "rb")  # raises OSError before anything is built
    return io.BufferedReader(_Decompressing(source, path), buffer_size=size)


def _decompress(path: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=target.parent, prefix=".partial-")
    try:
        with os.fdopen(fd, "wb") as out, zstd.open(path, "rb") as src:
            shutil.copyfileobj(src, out, 1 << 20)
        os.replace(tmp, target)
    except BaseException as exc:
        with contextlib.suppress(OSError):
            os.unlink(tmp)
        # A truncated or empty file raises EOFError rather than zstd's own
        # error, and before it was converted here too it went past every
        # reader's OSError handler and took the whole search down with it.
        if isinstance(exc, (zstd.ZstdError, EOFError)):
            raise OSError(f"cannot decompress {path}: {exc}") from exc
        raise
    for stale in target.parent.glob(f"{_stem(path)}-*.jsonl"):
        if stale != target:
            with contextlib.suppress(OSError):
                stale.unlink()


def discard(path: Path) -> None:
    """Remove every cached copy of *path*, for when the original is deleted."""
    for copy in cache_dir().glob(f"{_stem(path)}-*.jsonl"):
        with contextlib.suppress(OSError):
            copy.unlink()


# A day, as for the entry cache's own sweep, and for the same reason: asking
# costs a search one stat, and the sweep itself runs once a day.
SWEEP_EVERY = 24 * 3600
_SWEEP_MARKER = ".swept"


def sweep(now: float | None = None) -> int:
    """Remove the copies nothing has wanted for a day; returns how many.

    What still wants a copy is a reader that needs one on disk -- the raw
    scans, for the sessions the entry cache cannot hold or for every session
    with it off; discovery of a rollout the Codex index does not name, which
    seeks its tail on every run; a rewound thread's earlier files -- and
    :func:`readable` keeps the copies those use fresh. A parse wants none,
    so every copy made only for a parse, which before the entry cache was all
    of them, ages out here, as does one of a rollout its provider has pruned.
    A copy swept from under a reader is not a wrong answer: a scan that loses
    its file sends the session on to a parse, and :func:`open_binary` reads
    the rollout itself. A partial file is left until it is a day old too,
    which no decompression still running is.
    """
    now = time.time() if now is None else now
    removed = 0
    try:
        found = list(os.scandir(cache_dir()))
    except OSError:
        return 0
    for entry in found:
        name = entry.name
        if not (name.endswith(".jsonl") or name.startswith(".partial-")):
            continue
        try:
            if now - entry.stat().st_mtime < SWEEP_EVERY:
                continue
            os.unlink(entry.path)
        except OSError:
            continue
        if name.endswith(".jsonl"):
            removed += 1
    _known.clear()
    return removed


def sweep_if_due(now: float | None = None) -> int:
    """:func:`sweep`, at most once every ``SWEEP_EVERY`` seconds.

    Rationed by a marker file's mtime, as the entry cache's sweep is. Where
    nothing was ever decompressed there is no directory, and asking creates
    none.
    """
    root = cache_dir()
    marker = root / _SWEEP_MARKER
    now = time.time() if now is None else now
    try:
        if now - marker.stat().st_mtime < SWEEP_EVERY:
            return 0
    except OSError:
        if not root.is_dir():
            return 0
    try:
        marker.touch()
        os.utime(marker, (now, now))
    except OSError:
        return 0  # cannot record a sweep, so do not repeat one every search
    return sweep(now)
