"""Read zstd-compressed transcripts as the plain JSONL every reader expects.

Codex compresses rollouts in the background. codex-cli 0.164.0 ships a
"rollout compression worker" that rewrites ``rollout-*.jsonl`` as
``rollout-*.jsonl.zst`` in place, keeping the original mtime. Its first run on
the author's work Mac, on 2026-10-20, compressed 684 of 709 rollouts
(8 Jun - 9 Oct): 532 MB down to 104 MB. Codex's own ``threads`` index still
names each one by its ``.jsonl`` path, so discovery has to look for the
compressed spelling as well.

Search cannot afford to decompress on every query. Over those 684 files,
decompress-and-find cost 0.9-1.2s per pass in Python and 1.2s through
``rg -z``, against a whole-corpus search measured in tenths of a second. So
each file is decompressed once, into a cache keyed by its size and mtime, and
everything downstream -- ripgrep, the byte prefilter, the parser -- reads the
decoded copy as it would any other rollout. Should Codex ever rewrite a
compressed file, its size or mtime moves and the next read makes a new copy.

The cost is disk: the cache holds what Codex's compression saved, about five
times the compressed size. It lives under ``$XDG_CACHE_HOME`` (default
``~/.cache``) in ``session-browser/zst``, and deleting it loses nothing but the
next search's head start.
"""

from __future__ import annotations

import contextlib
import hashlib
import os
import shutil
import sys
import tempfile
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


def readable(path: Path) -> Path:
    """*path* itself, or its decompressed copy when it is a ``.zst`` file.

    The copy is made on first use and written atomically, so a reader in
    another thread or process sees either no copy or a whole one. Copies of
    earlier versions of the same file are removed when a new one is made.
    Raises OSError when the file cannot be read or does not decompress.
    """
    if not path.name.endswith(SUFFIX):
        return path
    # Search asks for the same file three or four times per query (candidate
    # scan, partition, parse), so the answer is remembered for as long as the
    # source's size and mtime say it still holds and the copy is still there.
    # That halves a lookup's cost; across 684 rollouts, ~7ms per pass to ~3ms.
    key = os.fspath(path)
    st = os.stat(key)
    known = _known.get(key)
    if (
        known is not None
        and known[0] == st.st_size
        and known[1] == st.st_mtime_ns
        and os.path.isfile(known[2])
    ):
        return known[2]
    target = cache_dir() / f"{_stem(path)}-{st.st_size}-{st.st_mtime_ns}.jsonl"
    if not target.is_file():
        _decompress(path, target)
    _known[key] = (st.st_size, st.st_mtime_ns, target)
    return target


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
        if isinstance(exc, zstd.ZstdError):
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
