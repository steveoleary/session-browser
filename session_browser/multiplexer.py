"""Which terminal multiplexer the ``t`` handoff can reach right now.

``t`` opens the highlighted session in a multiplexer, but only in one that is
actually running: tmux with a live server, or herdr with a live server. On a
machine where neither is up the key says so instead of failing deep inside a
create command, and when both are up the caller asks which.

``tmux`` and ``herdr`` are interchangeable here because both modules implement
the same small interface — ``available()``, ``in_*()`` for "are we already
inside it", and ``prepare_session()`` returning a plan with
``switch_commands()``, ``attach_commands()``, ``label`` and ``reused``. The
differences between the two (window tags versus herdr's own agent-session
reporting, ``sesh connect`` versus focus-then-attach) stay inside those
modules; this one only decides which to use.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from types import ModuleType

from . import herdr, tmux


@dataclass(frozen=True)
class Target:
    """One multiplexer the browser can hand a session over to.

    ``key`` is what picks it when more than one is running, so it must stay
    distinct across ``TARGETS``. ``error`` is the module's own exception type
    and ``inside`` its "are we already in it" test, both carried here so the
    caller can drive a handoff without knowing which multiplexer it is.
    """

    name: str
    key: str
    module: ModuleType
    error: type[Exception]
    inside: Callable[[], bool]


TARGETS: tuple[Target, ...] = (
    Target("tmux", "t", tmux, tmux.TmuxError, tmux.in_tmux),
    Target("herdr", "h", herdr, herdr.HerdrError, herdr.in_herdr),
)


def available_targets() -> list[Target]:
    """The multiplexers running right now, in ``TARGETS`` order.

    Probed on each use rather than cached at startup, because a browser left
    open across a day outlives the servers it is reporting on — and each probe
    is a bounded, sub-10ms local call.
    """
    return [target for target in TARGETS if target.module.available()]


def available_names() -> list[str]:
    """Names of the running multiplexers, for the key map."""
    return [target.name for target in available_targets()]
