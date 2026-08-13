#!/usr/bin/env bash
#
# Install this repository's git hooks into .git/hooks.
#
# Git does not clone hooks, so they need an install step. This one appends a
# MARKED SECTION to .git/hooks/pre-commit rather than owning the file, because
# the tracker manages its own section in the same hook with the same technique
# (`--- BEGIN TRACKER INTEGRATION ---`). Owning the file, or pointing
# core.hooksPath at a tracked directory, would silently disable the tracker's hooks the
# next time it installed them — silently being the problem.
#
# Re-running is safe: the existing LEAK-GUARD section is replaced, everything
# else in the file is left exactly as it was.
#
# Usage:
#   scripts/install-hooks.sh            install / refresh the hooks
#   scripts/install-hooks.sh --check    report status, change nothing (exit 1 if not installed)
#   scripts/install-hooks.sh --add-pattern TEXT   add a blocked pattern to THIS clone
#
# The blocked patterns are deliberately NOT tracked — see scripts/hooks/leak-guard.

set -uo pipefail

begin='# --- BEGIN LEAK-GUARD (managed by scripts/install-hooks.sh) ---'
end='# --- END LEAK-GUARD ---'

die() { printf 'install-hooks: %s\n' "$*" >&2; exit 1; }

git rev-parse --git-dir >/dev/null 2>&1 || die "not inside a git repository."
git_dir="$(git rev-parse --git-dir)"
top="$(git rev-parse --show-toplevel)"
hook="$git_dir/hooks/pre-commit"
guard="$top/scripts/hooks/leak-guard"

[ -f "$guard" ] || die "missing $guard"

mode="install"
add_pattern=""
while [ $# -gt 0 ]; do
  case "$1" in
    --check) mode="check"; shift ;;
    --add-pattern) add_pattern="${2:?--add-pattern needs a value}"; shift 2 ;;
    --add-pattern=*) add_pattern="${1#*=}"; shift ;;
    -h|--help) sed -n '3,20p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *) die "unknown option: $1" ;;
  esac
done

pattern_count() { git config --get-all hooks.leakpattern 2>/dev/null | grep -c . || true; }

if [ -n "$add_pattern" ]; then
  git config --add hooks.leakpattern "$add_pattern" \
    || die "could not add the pattern."
  printf 'Added pattern to this clone only (.git/config). Now %s configured.\n' "$(pattern_count)"
  exit 0
fi

if [ "$mode" = "check" ]; then
  status=0
  if [ -f "$hook" ] && grep -qF "$begin" "$hook"; then
    printf 'pre-commit  leak-guard section installed\n'
  else
    printf 'pre-commit  NOT installed — run scripts/install-hooks.sh\n'
    status=1
  fi
  if [ -f "$hook" ] && grep -q 'BEGIN TRACKER INTEGRATION' "$hook"; then
    printf 'pre-commit  tracker section present and preserved\n'
  fi
  n="$(pattern_count)"
  printf 'patterns    %s configured in this clone\n' "${n:-0}"
  [ "${n:-0}" -gt 0 ] || printf 'patterns    none — the guard will pass everything until you add some\n'
  exit "$status"
fi

mkdir -p "$git_dir/hooks"

# Start from whatever is already there, minus any previous section of ours.
tmp="$(mktemp)"
trap 'rm -f "$tmp"' EXIT

if [ -f "$hook" ]; then
  awk -v b="$begin" -v e="$end" '
    $0 == b { skip = 1; next }
    $0 == e { skip = 0; next }
    !skip   { print }
  ' "$hook" > "$tmp"
else
  printf '#!/usr/bin/env sh\n' > "$tmp"
fi

# A file that existed but had no shebang would silently not run.
head -1 "$tmp" | grep -q '^#!' || {
  printf '#!/usr/bin/env sh\n%s' "$(cat "$tmp")" > "$tmp.s" && mv "$tmp.s" "$tmp"
}

{
  printf '%s\n' "$begin"
  printf 'if [ -x "$(git rev-parse --show-toplevel)/scripts/hooks/leak-guard" ]; then\n'
  printf '  "$(git rev-parse --show-toplevel)/scripts/hooks/leak-guard" || exit 1\n'
  printf 'fi\n'
  printf '%s\n' "$end"
} >> "$tmp"

cp "$tmp" "$hook"
chmod +x "$hook" "$guard"

printf 'Installed leak-guard into %s\n' "$hook"
if grep -q 'BEGIN TRACKER INTEGRATION' "$hook"; then
  printf 'Preserved the existing tracker section in the same hook.\n'
fi

n="$(pattern_count)"
if [ "${n:-0}" -eq 0 ]; then
  printf '\nNo patterns configured yet — the guard will pass everything.\n'
  printf 'Add them to this clone (they are never tracked or pushed):\n'
  printf '  scripts/install-hooks.sh --add-pattern <text>\n'
else
  printf '%s pattern(s) configured in this clone.\n' "$n"
fi
