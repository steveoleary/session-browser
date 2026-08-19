#!/usr/bin/env bash
#
# The one place that knows where this clone's blocked patterns live. Sourced by
# every guard that needs them — never executed.
#
# THE PATTERNS ARE NOT IN THIS REPOSITORY, AND THAT IS THE WHOLE DESIGN. A guard
# that shipped its own blocklist would publish the very strings it exists to keep
# out — the list is the leak. So they live in .git/config, which is never pushed,
# one per clone:
#
#     git config --add hooks.leakpattern 'some-internal-name'
#
# There are now three readers — the pre-commit guard, the commit-msg guard, and
# the preflight — and they must not be able to disagree about what is blocked. A
# second copy of this loop is a copy that can drift, and a guard reading a stale
# list is a guard that passes what its neighbour refuses.
#
# Usage:
#
#     . "$(git rev-parse --show-toplevel)/scripts/hooks/leak-patterns.sh"
#     leak_read_patterns          # fills the global array `patterns`
#
# It fills a global rather than printing, because a pattern may contain any
# character a shell would otherwise mangle, and command substitution eats the
# last trailing newline — which silently drops a pattern.

# Read into a plain loop rather than mapfile: bash 3.2 is what /usr/bin/env bash
# resolves to on macOS, and mapfile does not exist there.
leak_read_patterns() {
  patterns=()
  while IFS= read -r _leak_line; do
    [ -n "$_leak_line" ] && patterns+=("$_leak_line")
  done < <(git config --get-all hooks.leakpattern 2>/dev/null || true)
  unset _leak_line
}
