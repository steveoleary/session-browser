# Conversation-first overlooked-session snapshot

This is a portable, sanitized reproduction of the real-history incident
documented in
`docs/superpowers/specs/2026-07-23-conversation-first-retrieval-design.md`.
It exists so the failure can be replayed on a machine that does not have the
original Claude Code transcripts.

The snapshot contains two provider-native Claude JSONL sessions:

- `claude:47337291-dbc4-4699-a771-8c521d361953` — a newer, unrelated session
  that contains four `codex` occurrences only in echoed tool output;
- `claude:ed1ba1f0-72a3-497b-80cb-62fd1c1f22be` — the older desired
  conversation, hidden behind a `/clear` summary, with the review request in a
  user turn and subsequent assistant/tool evidence.

The IDs, ordering, timestamps, summary shape, decisive request, and evidence
roles mirror the observed incident. The surrounding content is minimized and
sanitized; the full real-history measurements and timeline remain in the
design document.

## Run the frozen baseline

From the repository root:

```bash
.venv/bin/python \
  docs/fixtures/conversation-first-overlooked-session/verify_case.py \
  baseline
```

The check passes only when it reproduces the current failure:

1. literal search returns the newer tool-output echo first;
2. the desired conversation is second;
3. the false lead's snippets are tool-only;
4. the desired source has user and assistant matches.

The verifier runs Session Browser in a subprocess with `HOME` pointed at the
fixture's private `home/` tree. It does not read or modify the machine's real
session history.

## Check the conversation-first candidate

Once `--sort evidence` is implemented:

```bash
.venv/bin/python \
  docs/fixtures/conversation-first-overlooked-session/verify_case.py \
  candidate
```

The candidate check requires:

- the desired conversation first;
- the tool-output echo second;
- `conversation` / `user` evidence on the desired source; and
- `tool/system-only` / `tool` evidence on the false lead.

This is deliberately a RED check on the current revision because
`--sort evidence` does not exist yet.

## Inspect without assertions

To experiment manually while keeping the fixture isolated:

```bash
fixture_home="$PWD/docs/fixtures/conversation-first-overlooked-session/home"

HOME="$fixture_home" .venv/bin/python -m session_browser.app search \
  "codex" \
  "code review from codex" \
  "request a code review" \
  "codex-plugin-cc" \
  --provider claude \
  --mode snippets \
  --context 100 \
  --limit 10 \
  --format json
```

The expected observations and source-incident metadata are also recorded in
`expected.json` so a future retrieval-lab runner can ingest the snapshot
without scraping this README.
