"""User configuration: the ignore file, and the TUI's one setting.

Both live in one directory, ``$XDG_CONFIG_HOME/session-browser`` (by default
``~/.config/session-browser``):

``ignore``
    Sessions to leave out of every listing, search and stats count, in
    gitignore syntax, matched against each session's recorded working
    directory. The model is git's ``core.excludesFile`` and ripgrep's ignore
    handling, copied rather than invented: silent by default, ``--no-ignore``
    to reach past it, and anything named explicitly (``get <id>``, an
    ``--around`` anchor) is never ignored::

        # loop and dogfood runs: real work, little value as history
        ~/Projects/bloodhound-dogfood/

    ``~/`` at the start of a pattern means the home directory. A session with
    no recorded cwd has nothing to match and is never ignored.

    There is deliberately no per-directory ignore file. A session's cwd is a
    historical path, and the worktrees that produce the most noise are exactly
    the ones that get deleted -- an ignore file inside one would be deleted with
    it, and the noise would come back. Reading one would also cost a file probe
    per ancestor of every distinct cwd on every run.

``config.toml``
    ``[tui] ignored_notice = false`` stops the TUI header reporting how many
    sessions the ignore file hid. The notice is for a human, who already knows
    what they told it to ignore, so turning it off is theirs to choose.
"""

from __future__ import annotations

import os
import tomllib
from pathlib import Path

from pathspec import GitIgnoreSpec


class ConfigError(Exception):
    """A config file exists but cannot be used. Carries the path in its text."""


def config_dir() -> Path:
    base = os.environ.get("XDG_CONFIG_HOME")
    return (Path(base) if base else Path.home() / ".config") / "session-browser"


def ignore_path() -> Path:
    return config_dir() / "ignore"


def settings_path() -> Path:
    return config_dir() / "config.toml"


def _read(path: Path) -> str | None:
    """File text, or None when there is no file.

    NotADirectoryError counts as absent: a config home pointed at a file
    (the comparator points it at os.devnull to read no config at all) has no
    config in it, which is not an error the user made."""
    try:
        return path.read_text(encoding="utf-8")
    except (FileNotFoundError, NotADirectoryError):
        return None
    except (OSError, UnicodeDecodeError) as exc:
        raise ConfigError(f"cannot read {path}: {exc}") from exc


class IgnoreRules:
    """Compiled ignore patterns, memoised per distinct cwd.

    Sessions share working directories heavily -- a loop that spawns a hundred
    subagents spawns them all in one cwd -- so each cwd is matched once per
    process however many sessions carry it."""

    def __init__(self, path: Path, spec: GitIgnoreSpec) -> None:
        self.path = path
        self._spec = spec
        self._memo: dict[str, bool] = {}

    def ignores(self, cwd: str | None) -> bool:
        if not cwd:
            return False
        hit = self._memo.get(cwd)
        if hit is None:
            # The filesystem root is the gitignore root, so a pattern with a
            # leading slash is an absolute path and one without matches at any
            # depth. The trailing slash marks the cwd as a directory, which is
            # what makes a `dir/` pattern apply to it.
            hit = self._memo[cwd] = self._spec.match_file(cwd.lstrip("/") + "/")
        return hit


def _expand_home(line: str, home: str) -> str:
    negated = line.startswith("!")
    body = line[1:] if negated else line
    if body.startswith("~/"):
        body = home + body[1:]
    return ("!" if negated else "") + body


def load_ignore() -> IgnoreRules | None:
    """The user's ignore rules, or None when there is no file or no pattern.

    An invalid pattern is an error naming its line, not a pattern skipped:
    a rule that silently matches nothing leaves the user believing noise is
    hidden while it is not."""
    path = ignore_path()
    text = _read(path)
    if text is None:
        return None
    home = str(Path.home())
    lines = [_expand_home(line, home) for line in text.splitlines()]
    # pathspec documents its pattern error only as a ValueError subclass.
    try:
        spec = GitIgnoreSpec.from_lines(lines)
    except ValueError:
        for number, line in enumerate(lines, 1):
            try:
                GitIgnoreSpec.from_lines([line])
            except ValueError as exc:
                raise ConfigError(f"{path}:{number}: invalid pattern: {exc}") from exc
        raise
    if not spec.patterns or all(p.include is None for p in spec.patterns):
        return None
    return IgnoreRules(path, spec)


def load_tui_settings() -> dict:
    """``[tui]`` from config.toml, defaults filled in."""
    settings = {"ignored_notice": True}
    path = settings_path()
    text = _read(path)
    if text is None:
        return settings
    try:
        data = tomllib.loads(text)
    except tomllib.TOMLDecodeError as exc:
        raise ConfigError(f"{path}: {exc}") from exc
    tui = data.get("tui", {})
    notice = tui.get("ignored_notice", True) if isinstance(tui, dict) else True
    if not isinstance(notice, bool):
        raise ConfigError(f"{path}: [tui] ignored_notice must be true or false")
    settings["ignored_notice"] = notice
    return settings
