"""What discovery read out of each transcript, remembered between runs.

Claude Code and Pi keep no index, so discovery opens every one of their
transcripts: a few records off the top for the summary, directory and start
time, and a window off the end for the last turn. Every command pays for it,
whatever it goes on to do. Measured 2026-11-14 on the author's Mac, 1,126
such files, warm page cache:

    scan_claude   0.17 s   6,039 JSON records decoded, 1,229 opens
    scan_pi       0.20 s   3,841 JSON records decoded, 1,243 opens

against 0.16 s for Codex's 1,448 sessions, which come out of its own index.
Once a search itself took a tenth of a second, this was most of every command.

What those reads yield is a function of the file and nothing else, so it is
kept here, one JSON file per provider in
``$XDG_CACHE_HOME/session-browser/discovery``, and a transcript is opened
again only when its size or mtime has moved. Each file costs one ``stat`` in
place of two opens and a dozen decodes.

The rules are the entry cache's, for the same reasons: the file is named for a
hash of the parser's source, so a changed scanner starts from nothing rather
than serve what the old one read; a miss is only ever a scan; deleting the
directory loses nothing; and ``SESSION_BROWSER_CACHE=0`` switches it off.
"""

from __future__ import annotations

import contextlib
import json
import logging
import os
import tempfile
from pathlib import Path

from . import entry_cache

log = logging.getLogger(__name__)

# Returned by Memo.get for a file with nothing remembered. Not None, which is
# itself a remembered answer: "this file is not a session".
MISS = object()


def directory() -> Path:
    return entry_cache.cache_dir().parent / "discovery"


def signature(path: str, *more: str) -> list[int] | None:
    """Size and mtime of *path*, then of each of *more*, or None.

    None when *path* cannot be statted: there is then nothing to check a
    remembered answer against, and the scan is left to say why. A file in
    *more* that is missing is part of the answer rather than a failure -- a
    subagent without its sidecar scans differently from one with it -- and
    is recorded as such.

    Taken before the file is read, like an entry cache fingerprint: a
    transcript appended to during the scan is then remembered under its older
    state, which is never asked for again.
    """
    try:
        st = os.stat(path)
    except OSError:
        return None
    sig = [st.st_size, st.st_mtime_ns]
    for extra in more:
        try:
            st = os.stat(extra)
            sig.extend((st.st_size, st.st_mtime_ns))
        except OSError:
            sig.extend((-1, -1))
    return sig


class Memo:
    """One provider's remembered scans, loaded once and saved if they changed.

    Not shared between threads: discovery runs one scanner per provider, each
    with a memo of its own.
    """

    def __init__(self, provider: str, *, keep_unmet: bool = False) -> None:
        self._provider = provider
        self._keep_unmet = keep_unmet
        self._path = directory() / f"{provider}.{entry_cache.parser_version()}.json"
        self._before: dict[str, list] = {}
        self._after: dict[str, list] = {}
        try:
            with open(self._path, "rb") as fh:
                loaded = json.loads(fh.read())
            if isinstance(loaded, dict):
                self._before = loaded
        except (OSError, ValueError):
            pass  # nothing remembered yet, or nothing usable

    def get(self, path: str, sig: list[int]):
        """What was remembered for *path* in the state *sig*, or ``MISS``."""
        known = self._before.get(path)
        if type(known) is list and len(known) == 2 and known[0] == sig:
            self._after[path] = known
            return known[1]
        return MISS

    def put(self, path: str, sig: list[int], value) -> None:
        self._after[path] = [sig, value]

    def save(self) -> None:
        """Write what this run saw, if it differs from what was loaded.

        Only files met this run are kept, so a transcript that has gone takes
        its entry with it. A run that found nothing new and nothing gone
        writes nothing. Failure is silent for the reason a failed entry-cache
        store is: the caller already has its answer.

        A memo opened with *keep_unmet* is read for a few files at a time
        rather than for every one, so it keeps what this run did not meet as
        well, for as long as the file is still there.
        """
        if self._keep_unmet:
            unmet = {
                path: known
                for path, known in self._before.items()
                if path not in self._after and os.path.exists(path)
            }
            if not any(
                self._before.get(path) != known for path, known in self._after.items()
            ) and len(unmet) + len(self._after) == len(self._before):
                return
            self._after = {**unmet, **self._after}
        if self._after == self._before:
            return
        try:
            parent = self._path.parent
            parent.mkdir(mode=0o700, parents=True, exist_ok=True)
            data = json.dumps(self._after, ensure_ascii=False, separators=(",", ":"))
            fd, tmp = tempfile.mkstemp(dir=parent, prefix=".partial-")
            try:
                with os.fdopen(fd, "wb") as out:
                    out.write(data.encode("utf-8", "surrogatepass"))
                os.replace(tmp, self._path)
            except BaseException:
                with contextlib.suppress(OSError):
                    os.unlink(tmp)
                raise
            # An older parser's file for this provider is of no further use.
            prefix = f"{self._provider}."
            for entry in os.scandir(parent):
                if entry.name.startswith(prefix) and entry.name != self._path.name:
                    with contextlib.suppress(OSError):
                        os.unlink(entry.path)
        except (OSError, ValueError, TypeError) as exc:
            log.debug("could not save the %s discovery memo: %s", self._provider, exc)


def open_memo(provider: str, *, keep_unmet: bool = False) -> Memo | None:
    """A memo for *provider*, or None when the cache is switched off."""
    return Memo(provider, keep_unmet=keep_unmet) if entry_cache.enabled() else None
