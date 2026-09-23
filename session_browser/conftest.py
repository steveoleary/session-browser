"""Suite-wide isolation from the developer's own configuration.

``config.py`` reads ``$XDG_CONFIG_HOME/session-browser`` (else
``~/.config/session-browser``). Many tests leave the real HOME alone, so without
this a developer's ignore file would silently drop fixture sessions -- a suite
that passes or fails depending on whose machine it runs on. Subprocesses inherit
the variable too, so CLI and capture tests are covered by the same line.
"""

from __future__ import annotations

import pytest


@pytest.fixture(autouse=True, scope="session")
def _isolated_config_home(tmp_path_factory):
    patch = pytest.MonkeyPatch()
    patch.setenv("XDG_CONFIG_HOME", str(tmp_path_factory.mktemp("config-home")))
    yield
    patch.undo()
