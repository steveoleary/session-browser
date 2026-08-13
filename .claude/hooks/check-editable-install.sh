#!/usr/bin/env bash
# Assert that the `session-browser` command resolves to this checkout.
#
# AGENTS.md requires an editable install, because a non-editable
# `uv tool install .` copies a snapshot into uv's tool environment and the
# command then runs stale code. That failure is silent: the CLI works, it just
# is not the code you are editing, so behaviour "verified" through it is
# meaningless. It has cost real sessions, and it is written into three
# separate memory files, which is the tell that prose is not enough.
#
# Emits SessionStart context only when something is wrong. Silence means the
# install is editable and nothing needs saying.

set -uo pipefail

tool_root="${HOME}/.local/share/uv/tools/session-browser"

emit() {
  jq -nc --arg ctx "$1" \
    '{hookSpecificOutput:{hookEventName:"SessionStart",additionalContext:$ctx}}'
  exit 0
}

if [[ ! -d "$tool_root" ]]; then
  emit "session-browser is not installed as a uv tool. AGENTS.md requires the command to run this checkout: run 'uv tool install --force --editable .' from the repo root before verifying any behaviour through the CLI."
fi

# A non-editable install has no __editable__ finder; the package is a copy.
if ! compgen -G "${tool_root}/lib/python*/site-packages/__editable__.session_browser*.pth" >/dev/null; then
  emit "session-browser is installed NON-EDITABLE, so the command is running a stale snapshot, not this checkout. Any behaviour verified through the CLI right now is about the wrong code. Fix before trusting it: 'uv tool install --force --editable .' from the repo root, then restart any running TUI."
fi

exit 0
