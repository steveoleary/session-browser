#!/usr/bin/env bash
# Run the qualitative half of this fixture headlessly, once.
#
# A CONVENIENCE, NOT A GATE. `python -m session_browser.case_runner run` does not
# call this and never will: the consumer here is a model, and this repository
# does not gate on stochastic checks. It exists so the half is repeatable without
# rebuilding the invocation from scratch, which is what happened every time
# before. Read the friction report it produces; do not score the answers.
#
#   ./run_headless.sh                          # live SKILL.md, one run
#   ./run_headless.sh --name old --skill /tmp/before.md   # A/B against an edit
#   ./run_headless.sh --out ~/briefs           # keep the output somewhere known
#
# Writes out.json (the agent's reply), err.txt and calls.tsv (every tool call it
# made, with output bytes) into the run directory, and prints that path.
#
# Flags used, and why each is load-bearing:
#   --allowedTools          without it, -p asks for approval and exits
#   --no-session-persistence keeps these runs out of the real corpus
#   --setting-sources project ignores the user's own global agent config
#   --disable-slash-commands nothing here should reach a skill by name
#   env -u CLAUDECODE ...   so a run launched from inside Claude Code is not
#                           detected as nested and silently altered
set -uo pipefail

here=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
repo=$(cd -- "$here/../../.." && pwd)

name=run
model=${SB_BRIEF_MODEL:-claude-sonnet-5}
skill=$repo/skills/using-session-browser/SKILL.md
outroot=

die() { printf 'run_headless.sh: %s\n' "$1" >&2; exit 1; }

while [ $# -gt 0 ]; do
    case $1 in
        --name)  name=${2:?--name needs a value}; shift 2 ;;
        --model) model=${2:?--model needs a value}; shift 2 ;;
        --skill) skill=${2:?--skill needs a value}; shift 2 ;;
        --out)   outroot=${2:?--out needs a value}; shift 2 ;;
        -h|--help) sed -n '2,/^set -uo/p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//;$d'; exit 0 ;;
        *) die "unknown argument: $1" ;;
    esac
done

[ -f "$skill" ] || die "no skill file at $skill"
command -v claude >/dev/null || die "claude is not on PATH"
sb=$(command -v session-browser) || die "session-browser is not on PATH"

# The brief's questions and its friction-report sections, unquoted. Extracted
# rather than duplicated so a reworded brief cannot silently keep testing the old
# one -- if the markers move, this yields nothing and the run refuses to start.
prompt=$(sed -n '/^> You have a tool/,/^> - \*\*Missing/p' "$here/brief.md" \
         | sed 's/^> \{0,1\}//')
[ -n "$prompt" ] || die "found no brief in brief.md -- did its '> ' block change?"
case $prompt in
    *Missing*) ;;
    *) die "brief extraction stopped early -- check the end marker in brief.md" ;;
esac

out=${outroot:-$(mktemp -d "${TMPDIR:-/tmp}/skill-brief-runs.XXXXXX")}/$name
mkdir -p "$out" || die "cannot create $out"

# A scratch cwd, distinct from any real project, so the transcript this run
# writes is identifiable and removable with --exclude-cwd afterwards.
cwd=$(mktemp -d "${TMPDIR:-/tmp}/skill-brief.XXXXXX")

export SB_BRIEF_BIN=$sb
export SB_BRIEF_HOME=$here/home
export SB_BRIEF_LOG=$out/calls.tsv
: > "$SB_BRIEF_LOG"

printf 'run      %s\nmodel    %s\nskill    %s\ncorpus   %s\ncwd      %s\n\n' \
    "$name" "$model" "$skill" "$SB_BRIEF_HOME" "$cwd"

( cd "$cwd" && env -u CLAUDECODE -u CLAUDE_CODE_ENTRYPOINT \
    PATH="$here/headless_bin:$PATH" \
    claude -p --model "$model" --tools Bash \
      --allowedTools 'Bash(session-browser:*)' 'Bash(jq:*)' 'Bash(python3:*)' \
      --disable-slash-commands --setting-sources project \
      --no-session-persistence \
      --append-system-prompt-file "$skill" \
      --output-format json "$prompt" ) > "$out/out.json" 2> "$out/err.txt"
rc=$?

printf 'exit %d, %d tool calls\n%s\n' \
    "$rc" "$(wc -l < "$SB_BRIEF_LOG" | tr -d ' ')" "$out"
[ $rc -eq 0 ] || printf '\n%s\n' "$(tail -5 "$out/err.txt")" >&2
exit $rc
