# Claude queued-command fidelity snapshot

This portable, sanitized Claude JSONL fixture reproduces the delivered
queued-command shape confirmed by a raw-history audit on 2026-07-28. Its record
shapes were checked field-by-field against real provider history; it does not
test retrieval ordering, which remains a separate case.

Claude records a prompt typed while the agent is working as an `attachment` whose
nested type is `queued_command`, not as a `user` message. Human records carry
`origin.kind == "human"` and `commandMode == "prompt"`; machine notifications
share the record type but carry no `origin`, and roughly half of all such records
in a real corpus are machine notifications. That is why the parser gates on
`origin.kind`.

The synthetic transcript includes enqueue and remove bookkeeping, the delivered
human attachment, a machine `task-notification` attachment, and the following
assistant turn. Together they pin both halves of the contract:

- the delivered human prompt appears exactly once, so the enqueue/remove
  bookkeeping cannot become duplicate user turns; and
- the machine notification never appears as a user turn in any state, because it
  shares the `queued_command` record type but carries no `origin` and is not
  human speech.

## Run the accepted candidate

From the repository root:

```bash
.venv/bin/python \
  docs/fixtures/claude-queued-command-fidelity/verify_case.py \
  candidate
```

This is the accepted state since the parser began normalising delivered human
queued commands into `user` entries.

## Run the historical baseline

```bash
.venv/bin/python \
  docs/fixtures/claude-queued-command-fidelity/verify_case.py \
  baseline
```

The baseline records the pre-fix behavior, in which the prompt was omitted
entirely. It is deliberately RED on the current revision and is retained only as
a description of the defect. The independent retrieval-ordering case remains
accepted at its baseline until Phase 3.
