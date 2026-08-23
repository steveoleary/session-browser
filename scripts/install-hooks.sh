#!/usr/bin/env bash
#
# Install this repository's git hooks alongside the active hook owner.
#
# Git does not clone hooks, so they need an install step. In a Beads Dolt
# workspace, `bd hooks install --beads` owns core.hooksPath and the Beads-managed
# sections under .beads/hooks; this installer adds the project's sections
# outside those markers. Each installer can then regenerate its own section
# without erasing the other. A clone with no .beads workspace keeps the normal
# .git/hooks location and has no Beads dependency.
#
# Two hooks, because one cannot do both jobs:
#   pre-commit   scripts/hooks/leak-guard      staged file contents and paths
#   commit-msg   scripts/hooks/leak-guard-msg  the prepared commit message
# At pre-commit time git has not written a message yet, so the message check
# has to be a hook of its own rather than more of the first one.
#
# Re-running is safe: the existing marked sections are replaced, everything
# else in those files is left exactly as it was.
#
# Usage:
#   scripts/install-hooks.sh            install / refresh the hooks
#   scripts/install-hooks.sh --check    report status, change nothing (exit 1 if not installed)
#   scripts/install-hooks.sh --add-pattern TEXT   add a blocked pattern to THIS clone
#
# The blocked patterns are deliberately NOT tracked — see scripts/hooks/leak-guard.

set -uo pipefail

pre_begin='# --- BEGIN LEAK-GUARD (managed by scripts/install-hooks.sh) ---'
pre_end='# --- END LEAK-GUARD ---'
msg_begin='# --- BEGIN LEAK-GUARD-MSG (managed by scripts/install-hooks.sh) ---'
msg_end='# --- END LEAK-GUARD-MSG ---'

die() { printf 'install-hooks: %s\n' "$*" >&2; exit 1; }

git rev-parse --git-dir >/dev/null 2>&1 || die "not inside a git repository."
git_dir="$(git rev-parse --absolute-git-dir)"
top="$(git rev-parse --show-toplevel)"
guard="$top/scripts/hooks/leak-guard"
msg_guard="$top/scripts/hooks/leak-guard-msg"
lib="$top/scripts/hooks/leak-patterns.sh"

[ -f "$guard" ] || die "missing $guard"
[ -f "$msg_guard" ] || die "missing $msg_guard"
[ -f "$lib" ] || die "missing $lib"

mode="install"
add_pattern=""
while [ $# -gt 0 ]; do
  case "$1" in
    --check) mode="check"; shift ;;
    --add-pattern) add_pattern="${2:?--add-pattern needs a value}"; shift 2 ;;
    --add-pattern=*) add_pattern="${1#*=}"; shift ;;
    -h|--help) sed -n '3,26p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *) die "unknown option: $1" ;;
  esac
done

pattern_count() { git config --get-all hooks.leakpattern 2>/dev/null | grep -c . || true; }

# Lines in a hook that belong to something other than us: not one of OUR
# sections, not the shebang, not blank. Reported so a re-run visibly leaves them
# alone. Both markers are stripped whichever hook is being inspected, so the two
# sections never count each other as foreign.
foreign_lines() {
  [ -f "$1" ] || { printf '0\n'; return; }
  awk -v b1="$pre_begin" -v e1="$pre_end" -v b2="$msg_begin" -v e2="$msg_end" '
    $0 == b1 || $0 == b2 { skip = 1; next }
    $0 == e1 || $0 == e2 { skip = 0; next }
    skip                 { next }
    /^#!/                { next }
    /^[[:space:]]*$/     { next }
                         { n++ }
    END                  { print n + 0 }
  ' "$1"
}

# install_section <hook> <begin> <end> <body>
install_section() {
  hook="$1"; b="$2"; e="$3"; body="$4"

  # Start from whatever is already there, minus any previous section of ours.
  tmp="$(mktemp)"
  if [ -f "$hook" ]; then
    awk -v b="$b" -v e="$e" '
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
    printf '%s\n' "$b"
    printf '%s\n' "$body"
    printf '%s\n' "$e"
  } >> "$tmp"

  cp "$tmp" "$hook"
  chmod +x "$hook"
  rm -f "$tmp"
}

# Single-quoted: "$@" must survive into the hook, not be expanded here. git
# passes commit-msg the path of the prepared message as $1, and a guard that
# never receives it refuses rather than passes.
pre_body='if [ -x "$(git rev-parse --show-toplevel)/scripts/hooks/leak-guard" ]; then
  "$(git rev-parse --show-toplevel)/scripts/hooks/leak-guard" || exit 1
fi'
msg_body='if [ -x "$(git rev-parse --show-toplevel)/scripts/hooks/leak-guard-msg" ]; then
  "$(git rev-parse --show-toplevel)/scripts/hooks/leak-guard-msg" "$@" || exit 1
fi'

if [ -n "$add_pattern" ]; then
  git config --add hooks.leakpattern "$add_pattern" \
    || die "could not add the pattern."
  printf 'Added pattern to this clone only (.git/config). Now %s configured.\n' "$(pattern_count)"
  exit 0
fi

beads_workspace=0
hooks_dir="$git_dir/hooks"
if [ -d "$top/.beads" ]; then
  beads_workspace=1
  command -v bd >/dev/null 2>&1 \
    || die "this Beads workspace needs bd to install its hook owner."
  hooks_dir="$top/.beads/hooks"
fi
pre_hook="$hooks_dir/pre-commit"
msg_hook="$hooks_dir/commit-msg"

beads_hooks_active() {
  active_hooks="$(git config --get core.hooksPath 2>/dev/null || true)"
  case "$active_hooks" in
    .beads/hooks|./.beads/hooks|"$hooks_dir") return 0 ;;
    *) return 1 ;;
  esac
}

if [ "$mode" = "check" ]; then
  status=0
  if [ "$beads_workspace" -eq 1 ]; then
    if beads_hooks_active; then
      printf '%-11s %s\n' "hook path" ".beads/hooks active"
    else
      printf '%-11s %s\n' \
        "hook path" ".beads/hooks NOT active — run scripts/install-hooks.sh"
      status=1
    fi

    beads_list="$(bd hooks list 2>&1)"
    beads_list_status=$?
    beads_ok=1
    [ "$beads_list_status" -eq 0 ] || beads_ok=0
    for hook_name in \
      pre-commit post-merge pre-push post-checkout prepare-commit-msg; do
      printf '%s\n' "$beads_list" | grep -q "${hook_name}: installed" \
        || beads_ok=0
    done
    if [ "$beads_ok" -eq 1 ]; then
      printf '%-11s %s\n' "beads" "all managed hooks installed"
    else
      printf '%-11s %s\n' \
        "beads" "managed hooks incomplete — run scripts/install-hooks.sh"
      status=1
    fi
  fi

  report_hook() {
    if [ -f "$1" ] && grep -qF "$2" "$1"; then
      printf '%-11s %s section installed\n' "$3" "$4"
    else
      printf '%-11s %s NOT installed — run scripts/install-hooks.sh\n' "$3" "$4"
      status=1
    fi
    if [ "$(foreign_lines "$1")" -gt 0 ]; then
      printf '%-11s other sections present and preserved\n' "$3"
    fi
  }
  report_hook "$pre_hook" "$pre_begin" "pre-commit" "leak-guard"
  report_hook "$msg_hook" "$msg_begin" "commit-msg" "leak-guard-msg"
  n="$(pattern_count)"
  printf '%-11s %s configured in this clone\n' "patterns" "${n:-0}"
  [ "${n:-0}" -gt 0 ] || printf '%-11s none — the guards will pass everything until you add some\n' "patterns"
  exit "$status"
fi

if [ "$beads_workspace" -eq 1 ]; then
  bd hooks install --beads \
    || die "bd could not install the Beads-managed hooks."
  beads_hooks_active \
    || die "bd did not activate .beads/hooks through core.hooksPath."
fi

mkdir -p "$hooks_dir"

install_section "$pre_hook" "$pre_begin" "$pre_end" "$pre_body"
install_section "$msg_hook" "$msg_begin" "$msg_end" "$msg_body"
chmod +x "$guard" "$msg_guard"

printf 'Installed leak-guard into %s\n' "$pre_hook"
printf 'Installed leak-guard-msg into %s\n' "$msg_hook"
if [ "$(foreign_lines "$pre_hook")" -gt 0 ] || [ "$(foreign_lines "$msg_hook")" -gt 0 ]; then
  printf 'Preserved the other sections already in those hooks.\n'
fi

n="$(pattern_count)"
if [ "${n:-0}" -eq 0 ]; then
  printf '\nNo patterns configured yet — the guards will pass everything.\n'
  printf 'Add them to this clone (they are never tracked or pushed):\n'
  printf '  scripts/install-hooks.sh --add-pattern <text>\n'
else
  printf '%s pattern(s) configured in this clone.\n' "$n"
fi
