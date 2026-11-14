"""The ``session-browser`` command: the CLI with arguments, the TUI without.

Its own module so that the command line never imports the TUI. ``app.py``
defines Textual widgets at import time, so importing it to reach ``main()``
loaded Textual and Rich on every ``list``, ``get`` and ``search`` -- 130 ms of
a command whose own work can now take less than that (measured 2026-11-14,
``python -X importtime``: 132 ms for ``session_browser.app`` against 59 ms
for ``session_browser.cli``).
"""

from __future__ import annotations

import sys


def main() -> None:
    if len(sys.argv) > 1:
        from .cli import run_cli

        raise SystemExit(run_cli(sys.argv[1:]))
    from .app import SessionBrowser

    SessionBrowser().run()


if __name__ == "__main__":
    main()
