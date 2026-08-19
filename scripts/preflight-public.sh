#!/usr/bin/env bash
#
# Prove a checkout is fit to publish, rather than assume it.
#
# This is the whole-repo counterpart to scripts/hooks/leak-guard. The hook is
# scoped to the files a single commit touches, deliberately, so that one
# offending tracked file cannot block every unrelated commit. Nothing in that
# design looks at the tree as a whole or at history, and this is what does.
#
# Run it before the first push to a public remote, and keep running it after —
# the value is on every push, not only on migration day.
#
# Checks, in order:
#   1  the remote carries only ordinary refs (heads, tags, PRs)
#   2  no configured identifier appears anywhere in the working tree
#   3  no configured identifier appears anywhere in history — blobs AND messages
#   4  every commit in history is authored by the expected identity
#   5  gitleaks finds no secrets across full history
#   6  no locally-excluded path is tracked
#   7  no *.log file is tracked, in the tree or anywhere in history
#   8  .git-blame-ignore-revs names only commits that still exist
#
# Identifiers come from the same per-clone config the hook uses, and are never
# tracked — see AGENTS.md:
#
#     git config --get-all hooks.leakpattern
#
# A run with NO patterns configured FAILS. A preflight that passes because it
# was asked to look for nothing is worse than no preflight: it produces the
# reassurance without the check.
#
# Usage:
#   scripts/preflight-public.sh [--remote NAME|URL] [--quiet]
#
# --remote takes a name or a URL, so the new public repo can be checked before
# the working checkout is repointed at it. Omitted, check 1 is skipped.

set -uo pipefail

remote=""
quiet=false
while [ $# -gt 0 ]; do
  case "$1" in
    --remote) remote="${2:?--remote needs a value}"; shift 2 ;;
    --remote=*) remote="${1#*=}"; shift ;;
    --quiet) quiet=true; shift ;;
    -h|--help) sed -n '3,40p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *) printf 'preflight: unknown option: %s\n' "$1" >&2; exit 2 ;;
  esac
done

git rev-parse --git-dir >/dev/null 2>&1 || { printf 'preflight: not inside a git repository.\n' >&2; exit 2; }

n_pass=0; n_fail=0; n_skip=0
# Bash 3.2 has no lowercase expansion, so pad the label by hand.
pass() { n_pass=$((n_pass+1)); $quiet || printf '  PASS  %s\n' "$1"; }
skip() { n_skip=$((n_skip+1)); printf '  SKIP  %s\n' "$1"; }
fail() { n_fail=$((n_fail+1)); printf '  FAIL  %s\n' "$1"; }
detail() { printf '          %s\n' "$1"; }

lib="$(git rev-parse --show-toplevel)/scripts/hooks/leak-patterns.sh"
[ -r "$lib" ] || { printf 'preflight: cannot read %s\n' "$lib" >&2; exit 2; }
# shellcheck source=scripts/hooks/leak-patterns.sh
. "$lib"
leak_read_patterns

printf 'preflight: %s' "$(basename "$(git rev-parse --show-toplevel)")"
[ -n "$remote" ] && printf ' -> %s' "$remote"
printf '\n\n'

# --- 1. remote carries only ordinary refs --------------------------------------
if [ -n "$remote" ]; then
  refs="$(git ls-remote "$remote" 2>/dev/null)"
  if [ -z "$refs" ]; then
    skip "remote '$remote' returned no refs (empty repo, or unreachable)"
  else
    # Two rules. The structural one: anything outside heads/tags/PRs is some
    # tool using the repository as a data store, and checking the shape catches
    # a tool nobody thought of. It cannot see a tool that hides data in a
    # BRANCH, though, so the second rule is a per-clone list of ref patterns,
    # read from the same untracked config the identifiers use. Unset by
    # default, so this is inert for anyone else.
    bad="$(printf '%s' "$refs" | grep -vE '\srefs/(heads|tags|pull)/|\sHEAD$' || true)"
    while IFS= read -r pat; do
      [ -n "$pat" ] || continue
      hit="$(printf '%s' "$refs" | grep -E "$pat" || true)"
      [ -n "$hit" ] && bad="$(printf '%s\n%s' "$bad" "$hit")"
    done <<EOF
$(git config --get-all hooks.forbiddenref 2>/dev/null)
EOF
    bad="$(printf '%s' "$bad" | grep -v '^$' | sort -u || true)"
    if [ -n "$bad" ]; then
      fail "remote carries refs it should not"
      # printf '%s\n' — without the newline the last line is an incomplete
      # read and `while read` silently drops it, under-reporting the guard.
      printf '%s\n' "$bad" | while IFS= read -r r; do detail "$r"; done
      detail "remove with: git push $remote --delete <ref>"
    else
      pass "remote carries only ordinary refs"
    fi
  fi
else
  skip "no --remote given, so the remote-ref check did not run"
fi

# --- 2/3. identifiers in tree and history -------------------------------------
if [ ${#patterns[@]} -eq 0 ]; then
  fail "no identifiers configured — nothing was checked for"
  detail "add them with: scripts/install-hooks.sh --add-pattern <text>"
  fail "history identifier check could not run either"
else
  # Format each (pattern, file) pair as it is found. Accumulating "pat|files"
  # and splitting later mispairs as soon as one pattern matches two files: the
  # second filename has no delimiter and reads as its own pattern.
  tree_hits=""
  for pat in "${patterns[@]}"; do
    while IFS= read -r f; do
      [ -n "$f" ] || continue
      tree_hits="${tree_hits}${f}  contains '${pat}'
"
    done <<< "$(git grep -I -i -l -e "$pat" -- . 2>/dev/null || true)"
  done
  if [ -n "$tree_hits" ]; then
    fail "identifiers present in the working tree"
    printf '%s\n' "$tree_hits" | while IFS= read -r l; do
      [ -n "$l" ] && detail "$l"
    done
  else
    pass "no identifiers in the working tree"
  fi

  # History is every blob of every commit, which is the only check that proves a
  # deleted file is actually gone rather than merely absent from HEAD.
  revs="$(git rev-list --all 2>/dev/null)"
  hist_hits=""
  if [ -n "$revs" ]; then
    for pat in "${patterns[@]}"; do
      # shellcheck disable=SC2086
      while IFS= read -r f; do
        [ -n "$f" ] || continue
        hist_hits="${hist_hits}${f}  contains '${pat}'
"
      done <<< "$(git grep -I -i -l -e "$pat" $revs -- . 2>/dev/null | sed 's/^[0-9a-f]*://' | sort -u || true)"
    done
  fi
  # Blobs are only half of history. A commit MESSAGE is not a file, so nothing
  # above can see one, and an identifier quoted in a body survives a run that
  # reports all-clear — which is exactly how one got through. There is no line
  # to cite for a message, so hits are labelled by commit instead.
  #
  # Two passes on purpose. The cheap one greps every message at once and costs
  # one grep per pattern; only when it finds something does the expensive one
  # run, walking commits to say which. A clean history — the normal case — never
  # pays for the attribution.
  msg_hits=""
  if [ -n "$revs" ]; then
    all_messages="$(git log --all --format=%B 2>/dev/null || true)"
    for pat in "${patterns[@]}"; do
      printf '%s\n' "$all_messages" | grep -i -q -e "$pat" 2>/dev/null || continue
      while IFS= read -r rev; do
        [ -n "$rev" ] || continue
        if git log -1 --format=%B "$rev" 2>/dev/null | grep -i -q -e "$pat" 2>/dev/null; then
          # The abbreviated hash only. Printing the subject would reproduce the
          # identifier in the output of the tool that exists to keep it in.
          msg_hits="${msg_hits}$(git rev-parse --short "$rev")  (message) contains '${pat}'
"
        fi
      done <<< "$revs"
    done
  fi

  if [ -n "$hist_hits" ] || [ -n "$msg_hits" ]; then
    fail "identifiers present in history"
    printf '%s%s\n' "$hist_hits" "$msg_hits" | while IFS= read -r l; do
      [ -n "$l" ] && detail "$l"
    done
  else
    pass "no identifiers anywhere in history, in blobs or in messages"
  fi
fi

# --- 4. author identity across history ----------------------------------------
expected="$(git config hooks.expectedemail || true)"
if [ -z "$expected" ]; then
  skip "hooks.expectedemail is not set, so authorship was not checked"
else
  others="$(git log --all --format='%ae%n%ce' 2>/dev/null | sort -u | grep -vFx "$expected" || true)"
  if [ -n "$others" ]; then
    fail "history contains commits by another identity"
    printf '%s\n' "$others" | while IFS= read -r e; do
      [ -n "$e" ] && detail "$e  ($(git log --all --format='%ae %ce' | grep -cF "$e") refs)"
    done
  else
    pass "every commit authored and committed by $expected"
  fi
fi

# --- 5. secrets ----------------------------------------------------------------
if ! command -v gitleaks >/dev/null 2>&1; then
  skip "gitleaks is not installed, so secrets were not scanned"
else
  if gitleaks git --log-opts="--all" --no-banner >/dev/null 2>&1; then
    pass "gitleaks found no secrets in history"
  else
    fail "gitleaks reported findings — run it directly for detail"
    detail "gitleaks git --log-opts=--all --redact"
  fi
fi

# --- 6. nothing ignored is also tracked ----------------------------------------
# Asks git directly for files that are both tracked and ignored. A path in
# .gitignore or in this clone's .git/info/exclude that is nevertheless tracked
# means the ignore rule is decorative and the file ships regardless. Checking
# the contradiction rather than one known directory catches whatever was
# excluded for local reasons this script has never heard of.
ignored_tracked="$(git ls-files -c -i --exclude-standard 2>/dev/null | grep -c . || true)"
if [ "${ignored_tracked:-0}" -gt 0 ]; then
  fail "${ignored_tracked} tracked file(s) are also ignored"
  git ls-files -c -i --exclude-standard 2>/dev/null | head -20 | while IFS= read -r f; do
    detail "$f"
  done
  detail "untrack with: git rm -r --cached <path>"
else
  pass "nothing tracked is also ignored"
fi

# --- 7. no tracked *.log, in tree or history -----------------------------------
# Named by shape rather than by filename: the file that made this check
# necessary should not have to be named in a public repository to be caught.
log_now="$(git ls-files '*.log' 2>/dev/null | grep -c . || true)"
log_ever="$(git log --all --diff-filter=A --name-only --pretty=format: 2>/dev/null \
            | grep -E '\.log$' | sort -u || true)"
if [ "${log_now:-0}" -gt 0 ] || [ -n "$log_ever" ]; then
  fail "log files are tracked or were committed historically"
  printf '%s\n' "$log_ever" | while IFS= read -r l; do
    [ -n "$l" ] && detail "$l"
  done
else
  pass "no .log files tracked, now or historically"
fi

# --- 8. .git-blame-ignore-revs points at real commits --------------------------
if [ ! -f .git-blame-ignore-revs ]; then
  pass ".git-blame-ignore-revs absent (nothing to dangle)"
else
  stale=""
  while IFS= read -r rev; do
    case "$rev" in ''|\#*) continue ;; esac
    git cat-file -e "${rev}^{commit}" 2>/dev/null || stale="${stale}${rev}
"
  done < .git-blame-ignore-revs
  if [ -n "$stale" ]; then
    fail ".git-blame-ignore-revs names commits that do not exist"
    printf '%s\n' "$stale" | while IFS= read -r r; do
      [ -n "$r" ] && detail "$r"
    done
    detail "a wrong entry here makes a line's real author unfindable"
  else
    pass ".git-blame-ignore-revs names only existing commits"
  fi
fi

printf '\n%d passed, %d failed, %d skipped\n' "$n_pass" "$n_fail" "$n_skip"
[ "$n_skip" -gt 0 ] && printf 'skipped checks proved nothing — read them before trusting a green run.\n'
[ "$n_fail" -eq 0 ] || exit 1
exit 0
