#!/usr/bin/env bash
# Lint a Python file the moment it is edited.
#
# The repo went from zero lint to clean in one pass. Staying clean is the hard
# part: the findings arrive one edit at a time, and a linter nobody runs until
# CI is a linter that reports a hundred things at the worst moment.
#
# ruff on a single file is a few milliseconds, so this costs nothing per edit
# and keeps the baseline at zero. Deliberately silent on success.
#
# It reports; it does not fix. Several rules are suppressed in this repo for
# reasons a fixer cannot know (hot paths that no counter guards, a TUI's
# deliberate glyphs) -- see pyproject.toml and the noqa comments in
# transcript.py. Read the finding before changing the code.

set -uo pipefail

payload=$(cat)
file=$(printf '%s' "$payload" | jq -r '.tool_response.filePath // .tool_input.file_path // empty')

[[ "$file" == *.py ]] || exit 0

cd "${CLAUDE_PROJECT_DIR:-.}" || exit 0

# Only files inside the repo. Scratch scripts and temp analysis live outside it
# and are not held to the repo's baseline; linting those is friction, not care.
[[ "$file" == "$PWD"/* ]] || exit 0

# No venv, or ruff not installed into it: not this hook's problem to report.
[[ -x .venv/bin/ruff ]] || exit 0

status=0

if ! out=$(.venv/bin/ruff check --force-exclude \
             --output-format=concise "$file" 2>&1); then
  {
    echo "ruff findings in ${file##*/} after the edit just made:"
    echo
    printf '%s\n' "$out" | head -40
    echo
    echo "Fix them now -- the repo's baseline is zero findings. If a rule is"
    echo "wrong here, suppress it with '# noqa: RULE' and a comment saying why,"
    echo "rather than leaving the finding to accumulate."
  } >&2
  status=2
fi

# Formatting is checked, not applied. Rewriting a file underneath an agent
# mid-edit invites it to work from a stale copy; reporting lets it fix its own
# file and keeps the next diff free of drive-by reformatting.
if ! .venv/bin/ruff format --check --force-exclude "$file" >/dev/null 2>&1; then
  {
    echo
    echo "${file##*/} is not formatted. The repo is ruff-format clean, so an"
    echo "unformatted file leaks whitespace noise into an unrelated diff."
    echo
    echo "    .venv/bin/ruff format ${file}"
  } >&2
  status=2
fi

exit $status
