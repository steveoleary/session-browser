#!/usr/bin/env bash
# Run the transcript test module immediately after transcript.py is edited.
#
# This buys latency, not coverage: the normal suite already runs these tests,
# but it takes ~48s and may not run for another twenty minutes. This module is
# 140 tests in under a second, and it guards the file with the highest churn in
# the repo and a recurring bug class -- entries from a provider's raw format
# getting mapped to the wrong turn role (3bc9bc7, 529f179, 1c20667).
#
# Deliberately silent on success. It only speaks when something broke.

set -uo pipefail

payload=$(cat)
file=$(printf '%s' "$payload" | jq -r '.tool_response.filePath // .tool_input.file_path // empty')

case "$file" in
  */session_browser/transcript.py|*/session_browser/test_transcript.py) ;;
  *) exit 0 ;;
esac

cd "${CLAUDE_PROJECT_DIR:-.}" || exit 0

# No venv is not this hook's problem to report; stay out of the way.
[[ -x .venv/bin/python ]] || exit 0

# -x stops at the first failure: the point is a fast signal, not a full report.
if ! out=$(.venv/bin/python -m pytest session_browser/test_transcript.py \
             -q -x --no-header -p no:cacheprovider 2>&1); then
  {
    echo "test_transcript.py FAILED after editing ${file##*/}."
    echo "This module is fast and normally green; treat it as caused by the edit"
    echo "just made rather than pre-existing. Fix before continuing."
    echo
    printf '%s\n' "$out" | tail -30
  } >&2
  exit 2
fi

exit 0
