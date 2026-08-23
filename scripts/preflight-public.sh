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
#   1  local branch/tag names contain no configured identifier, and the remote
#      carries only ordinary refs (heads, tags, PRs)
#   2  no configured identifier appears anywhere in the working tree
#   3  no configured identifier appears anywhere in history — blobs, commit
#      messages, annotated tag objects AND git notes
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
# the working checkout is repointed at it. Local ref names are always checked;
# when this is omitted, only check 1's remote half is skipped.

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

# --- 1. local names are safe; remote refs have an ordinary shape ---------------
# A branch or tag name is publishable text even though it is not an object body.
# This is the one owner of ref names: check 3 deliberately stays about object
# contents. Never print a matching name — report its object ID and commands that
# let the operator find and rename it without reproducing it here.
local_ref_hits=""
local_ref_count=0
if [ ${#patterns[@]} -gt 0 ]; then
  while IFS=' ' read -r object_id ref_name; do
    [ -n "$ref_name" ] || continue
    for pat in "${patterns[@]}"; do
      if printf '%s\n' "$ref_name" | grep -i -q -e "$pat" 2>/dev/null; then
        short_id="$(git rev-parse --short "$object_id")"
        case "$ref_name" in
          refs/heads/*)
            local_ref_hits="${local_ref_hits}${short_id}  (local branch name); inspect: git branch --points-at ${short_id}
"
            ;;
          refs/tags/*)
            local_ref_hits="${local_ref_hits}${short_id}  (local tag name); inspect: git tag --points-at ${short_id}
"
            ;;
        esac
        local_ref_count=$((local_ref_count + 1))
        break
      fi
    done
  done <<EOF
$(git for-each-ref --format='%(objectname) %(refname)' refs/heads refs/tags 2>/dev/null)
EOF
fi
local_ref_hits="$(printf '%s' "$local_ref_hits" | grep -v '^$' | sort -u || true)"

if [ -n "$local_ref_hits" ]; then
  fail "local branch or tag names contain configured identifiers"
  detail "$local_ref_count offending local ref name(s)"
  printf '%s\n' "$local_ref_hits" | while IFS= read -r ref_hit; do
    [ -n "$ref_hit" ] && detail "$ref_hit"
  done
  detail "rename a branch: git branch -m <old> <new>"
  detail "rename a tag without changing its object:"
  detail "git update-ref refs/tags/<new> <object-id>"
  detail "git update-ref -d refs/tags/<old>"
fi

if [ -n "$remote" ]; then
  refs="$(git ls-remote "$remote" 2>/dev/null)"
  if [ -z "$refs" ]; then
    if [ -n "$local_ref_hits" ]; then
      skip "remote '$remote' returned no refs; local names were checked separately"
    else
      skip "local branch and tag names are clean; remote '$remote' returned no refs"
    fi
  else
    # Two remote rules. The structural one catches any namespace outside
    # heads/tags/PRs. The separately configured forbidden-ref patterns catch
    # tools that hide data inside an ordinary-looking remote branch.
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
      printf '%s\n' "$bad" | while IFS= read -r remote_ref; do
        detail "$remote_ref"
      done
      detail "remove with: git push $remote --delete <ref>"
    elif [ -z "$local_ref_hits" ]; then
      pass "local ref names are safe and remote refs have an ordinary shape"
    fi
  fi
elif [ -n "$local_ref_hits" ]; then
  skip "no --remote was given; local names were checked separately"
else
  skip "local branch and tag names are clean; no --remote was given"
fi

# --- 2/3. identifiers in tree and history -------------------------------------
if [ ${#patterns[@]} -eq 0 ]; then
  fail "no identifiers configured — nothing was checked for"
  detail "add them with: scripts/install-hooks.sh --add-pattern <text>"
  fail "history identifier check could not run either (blobs, messages, tags, notes)"
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
  #
  # refs/notes is excluded here and scanned on its own below. It is reachable
  # from --all, so this grep did find note text — but only by accident, and it
  # reported the hit as a *path*, which for a note is the annotated object's
  # hash. A 40-hex "filename" that exists in no tree is a report that sends the
  # reader looking for the wrong thing, and a coverage that nothing states is a
  # coverage the next edit to this line can drop in silence.
  revs="$(git rev-list --exclude=refs/notes/\* --all 2>/dev/null)"
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
    all_messages="$(git log --exclude=refs/notes/\* --all --format=%B 2>/dev/null || true)"
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

  # An ANNOTATED TAG carries free text exactly as a commit does, and nothing
  # above can see it: rev-list peels a tag ref to the commit it points at, so
  # the tag object itself is never visited and `git tag -a -m` sailed through a
  # green run. There is no hook counterpart to add — git has no tag-msg hook —
  # so this is the only place a tag can ever be caught.
  #
  # Only the message is read: the object's headers are stripped at the first
  # blank line, which drops the tagger's address (check 4's business, and a
  # pattern shaped like an address would otherwise fire here as noise) and the
  # `tag <name>` header. A tag whose NAME carries an identifier is therefore not
  # caught here — a ref name is check 1's shape rule and hooks.forbiddenref,
  # tracked separately. Hits are reported by tag name, the way message hits are
  # reported by short hash, and the message itself is never printed.
  tag_hits=""
  while IFS=' ' read -r otype oname rname; do
    [ "$otype" = "tag" ] || continue
    body="$(git cat-file tag "$oname" 2>/dev/null | sed '1,/^$/d' || true)"
    [ -n "$body" ] || continue
    for pat in "${patterns[@]}"; do
      if printf '%s\n' "$body" | grep -i -q -e "$pat" 2>/dev/null; then
        tag_hits="${tag_hits}${rname#refs/tags/}  (tag message) contains '${pat}'
"
      fi
    done
  done <<EOF
$(git for-each-ref --format='%(objecttype) %(objectname) %(refname)' refs/tags 2>/dev/null)
EOF

  # Notes are prose a person writes that travels with the repository, and they
  # are stored as blobs whose PATH is the hash of the object annotated. Report
  # by that object — what `git notes show <commit>` takes — rather than by the
  # path, and walk the notes refs' own history so an edited or removed note is
  # read too, the same reason the blob scan walks every rev.
  note_revs="$(git rev-list --glob=refs/notes/\* 2>/dev/null)"
  note_hits=""
  if [ -n "$note_revs" ]; then
    for pat in "${patterns[@]}"; do
      # shellcheck disable=SC2086
      while IFS= read -r obj; do
        [ -n "$obj" ] || continue
        note_hits="${note_hits}$(git rev-parse --short "$obj" 2>/dev/null || printf '%s' "$obj")  (note) contains '${pat}'
"
      done <<< "$(git grep -I -i -l -e "$pat" $note_revs -- . 2>/dev/null \
                  | sed 's/^[0-9a-f]*://' | tr -d '/' | sort -u || true)"
      # The notes refs' own commit messages are git's boilerplate rather than
      # anyone's prose, but they are messages in history and the scan above
      # excluded them, so nothing else would read them.
      while IFS= read -r rev; do
        [ -n "$rev" ] || continue
        if git log -1 --format=%B "$rev" 2>/dev/null | grep -i -q -e "$pat" 2>/dev/null; then
          note_hits="${note_hits}$(git rev-parse --short "$rev")  (notes ref message) contains '${pat}'
"
        fi
      done <<< "$note_revs"
    done
  fi

  if [ -n "$hist_hits" ] || [ -n "$msg_hits" ] || [ -n "$tag_hits" ] || [ -n "$note_hits" ]; then
    fail "identifiers present in history"
    printf '%s%s%s%s\n' "$hist_hits" "$msg_hits" "$tag_hits" "$note_hits" \
      | while IFS= read -r l; do
      [ -n "$l" ] && detail "$l"
    done
  else
    pass "no identifiers in history — blobs, messages, tag messages or notes"
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
