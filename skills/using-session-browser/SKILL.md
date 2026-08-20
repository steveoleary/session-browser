---
name: using-session-browser
description: Use when finding, searching, or retrieving prior agent sessions, transcripts, handoffs, decisions, lessons, or context with the session-browser CLI.
---

# Using Session Browser

`session-browser` searches every Claude Code, Codex, OpenCode and pi session on
this machine. Narrow before you read: metadata, then a rare phrase, then a
window — never a whole transcript to stdout.

## The four traps that produce a wrong answer

Everything after this section makes you faster. These four make you **wrong**,
silently, with exit 0. Read them before your first command.

**1. Text filters are substrings, not names.** `--repo coffee_run` also matches
`coffee_run_private`; `--cwd api` also matches `api-gateway`. So any count off
a filtered `list` or `stats` is an upper bound. The receipt is already in the
output: `stats --format json` returns `top_cwds`, which decomposes your own
filter — read it back before you report a number, and group `list` results by
their `cwd` field. A filter that matches nothing returns an empty set rather
than an error, so a typo (`--repo coffe_run`) looks exactly like an answer.

**2. A hit is as often a quotation as a statement.** A transcript records every
file the agent read, every diff, every pasted doc — not only what someone said.
`--mode snippets` puts a `role` on each snippet: `tool` means the phrase was in
something the agent *read*, `user` and `assistant` mean somebody wrote it, and
`system` is injected or copied text — compaction summaries, task notifications,
pasted reports — which can quote earlier speech and is never fresh evidence
that anyone said it *there*. Check `role` before quoting a hit as evidence that
a thing was discussed. `--mode ids` carries no roles, which is why ids alone
nominate candidates and never confirm a finding. For a provenance census
without pulling whole snippets into context, count roles over a compact
projection — `--max-snippets 0` lifts the default cap of 20 per result, which
otherwise silently truncates the tally you are counting.

**3. Sessions self-correct.** A confident mid-session claim is often reversed a
few entries later: claim → user pushback → correction. Treat any snippet or
mid-session window as a *lead*. Before quoting a decision, read the ending —
`get ID --role assistant --tail 10`. To *find* a reversal rather than guard
against one, search the phrases people actually use — "I was wrong", "I was
mistaken", "actually", "correction" — then check that claim and correction
share a session and that both are `assistant`- or `user`-authored, not `tool`.

**4. `stats` counts what it discovered, not what is readable.** It never opens
a transcript, by design — it says so in the payload, `"transcript_health":
"not_checked"` — so its total cannot see sessions that are corrupt or empty.
`list` does open them, but it does **not** remove them: all of them stay in
`.sessions`, and the warning names a handful of ids and then says "(N more)".
Use its top-level `counts.readable` rather than treating the array length as a
readable count. Also note the total is caller-relative: your own live session
is excluded unless you pass `--include-current`.

## Flags are not documented here

Flag semantics, defaults and enum values live in `session-browser <cmd> --help`,
next to the code that implements them. This skill deliberately does not restate
them: a restatement is a copy, and it goes stale on the commit that changes the
flag. This skill tells you *which* flag and *why* — run `--help` when a flag's
exact behaviour decides your answer.

Two facts are repeated here anyway, because a wrong parse costs a whole task:
`search` returns `{"results": [...]}`, while `list` and multi-id `get` return
`{"sessions": [...]}` (a single-id `get` returns `{"session": ..., "entries":
[...]}`). A `KeyError` on one of those is the envelope, not an empty corpus.

If a command or flag this skill teaches comes back `invalid choice` or
`unrecognized arguments`, suspect the binary on your PATH before you suspect
the skill — refresh the install and retry the narrowing command rather than
working around it by dumping full transcripts. But this skill can be wrong too;
where they disagree, `--help` describes the binary you are actually running.

## Retrieval Workflow

1. **Orient.** `session-browser stats --format json` (shared filters apply,
   e.g. `--here`) gives provider counts, per-day activity and `top_cwds` values
   you can feed straight to `--cwd`. Mind trap 4 on its total.
2. **Discover with metadata.** Two things distort discovery before any trap
   does, and both are properties of the corpus rather than of your query:

   - **Worktrees.** A session run in a git worktree is named after the worktree
     directory unless the provider records a project root, and only opencode
     does — so `--repo <project>` misses claude, codex and pi worktree
     sessions, and a reused worktree path can hold work from several projects
     over time.
   - **Other agents are writing to the corpus while you read it.** Hit counts
     and roles move under you, and a sibling answering your question can put
     its answer into your results. `--until -30m` gives a stable snapshot;
     without one, a measurement is not reproducible even by you.

   Then: never dismiss a session from its `summary` alone
   — the summary is often just the first user message, so a bare `/model` or
   `/clear` label can hide hours of work. Every `list` row carries
   `total_entries` and `duration_seconds`; `total_entries` is the *primary*
   signal, and the trap runs both ways — a huge `duration_seconds` on a
   2-entry session is a stale record whose file was touched days later, not
   deep work. When one session is known and the question is what else was
   happening near it, `list --around ID` returns its temporal neighbours;
   do not hand-compute a window from its timestamp.
3. **Search rare phrases in `--mode ids` first.** Pass alternate phrasings
   together — `search "phrase a" "phrase b"` ORs them in one scan. "Rare" means
   rare *in this corpus*: ordinary English ("delete", "batch", "table") and
   vocabulary shared across projects ("App Store", "tethered") match most of it.
   Three common words matched 71% of the corpus in one measured run, and it
   cost three seconds to find that out — so when a query feels broad, run it in
   `--mode ids` and look at the count before refining. Prefer multiword phrases
   the answer itself would contain. Summaries are scanned too, so words from
   how the work would be *titled* are good phrases; a summary-only hit returns
   `match_count 0` with the phrases in `summary_matches`. Matching is literal
   and markdown-insensitive, never semantic.
4. **Triage with `--mode snippets`.** A high `match_count` on a scoped search is
   the strongest triage signal there is: summaries can be blank or misleading,
   but a session that says the phrase 200 times is where the work happened.
   Check each snippet's `role` (trap 2) and use `first_match`/`last_match` as
   entry indices for step 5.
5. **Retrieve only what you need.** Read a window with `get ID --entries A:B`,
   `--tail N` or `--head N`. To read one voice, filter by role: `--role user`
   is the intent trail, `--role assistant` gives conclusions without tool noise,
   `--role error` the failed calls. `--role user` is *human speech only*;
   interrupt markers, task notifications and the auto-written summary at the top
   of a resumed session are `--role system`, so **when a session was compacted
   its earliest intent lives there, not in `--role user`**. Batch several
   candidates into one call: `get ID1 ID2 ID3 --role user --head 3`.
6. **Synthesize.** Mention `skipped` sessions and parse warnings when the JSON
   reports them.

## Quick Reference

| Need | Command |
|------|---------|
| Orient in an unfamiliar history | `session-browser stats --here --format json` |
| Recent sessions in a project | `session-browser list --cwd project --limit 10` |
| Sessions from this project | `session-browser list --here --limit 10` |
| What else happened around a known session | `session-browser list --around ID --limit 15` |
| Find candidates | `session-browser search "rare phrase" --cwd my-project --mode ids --limit 10` |
| Alternate phrasings, one scan | `session-browser search "phrase a" "phrase b" --mode ids` |
| Inspect matches | `session-browser search "rare phrase" --mode snippets --context 120 --limit 5` |
| Rank by hit count, not recency | add `--sort matches` |
| Read the matched region | `session-browser get ID --entries 38:60` |
| How a session ended | `session-browser get ID --role assistant --tail 20` |
| What the user asked for | `session-browser get ID --role user` |
| Original intent in a compacted session | `session-browser get ID --role system --head 3` |
| Triage several candidates at once | `session-browser get ID1 ID2 ID3 --role user --head 3` |
| Whole transcript | `session-browser get ID --output build/session.md` |
| Drop harness/scratch noise | `--exclude-cwd /private/tmp/` (repeatable, applied before `--limit`) |
| A stable snapshot while other agents are working | add `--until -30m` |
| Every match, not the first N | omit `--limit` entirely — it is unbounded by default |
| Count roles without pulling snippets into context | `--mode snippets --max-snippets 0`, project to `role` only |

A canonical id (`claude:76458688-…`, copyable from the TUI with `i`) goes
straight to `get ID` — no `list`/`search` first. A unique id *prefix* resolves
like a git short hash; an ambiguous one errors and lists candidates, and a full
id matching nothing suggests near misses, which recovers a typo mid-id.

Judgment `--help` does not carry:

- Whenever `--limit` drops results, `list` and `search` say so in `warnings`
  with a count and date range. **A listing with no such warning is complete.**
- `--around ID --limit 15` is a k-nearest query — the 15 closest sessions
  regardless of window. Adjust `--window` only for a hard time cutoff.
- Judge neighbours by `offset` *and* `duration_seconds` together: a small
  offset on a very long session is a concurrent sibling, not a predecessor.
- `--sort` is not a shared filter, and `stats` has none at all.
- `--here` cannot see sessions with no recorded cwd. If its `warnings` says it
  excluded some and they matter, reach them with `--cwd`, a content search, or
  `--provider`.

## Baseline Failures To Avoid

| Failure | Better Move |
|---------|-------------|
| Broad query like `session-browser`, `search`, `output`, `token` | Rare multiword phrases and filters first |
| Triaging a session as noise from its `summary` | Summaries lie; check `total_entries` and `duration_seconds` from the `list` row you already have |
| Reading a huge `duration_seconds` as deep work | `total_entries` is primary — 66h on a 2-entry session is a stale record touched later |
| Reporting a count off `--repo`/`--cwd` without checking it | Both are substrings; read `top_cwds` or group hits by `cwd` before quoting a number |
| Quoting a search hit as proof a topic was discussed | Check the snippet's `role` — `tool` means the agent read it, not said it |
| A hit with `match_count: 1` inside the repo that owns the phrase | That is the shape of a file read, not a discussion; confirm with `role` and surrounding entries |
| Quoting a conclusion from a mid-session snippet | Sessions self-correct; confirm against the ending with `--role assistant --tail N` |
| Reading implementation docs before trying the CLI | Use `--help`, `list` and `search` unless you are changing the tool |
| Blind `get ID` to stdout | `--entries`/`--head`/`--tail` for a window, `--role` for one voice, `--output` for the whole thing |
| One `get` per candidate while triaging | Batch: `get ID1 ID2 ID3 --role user --head 3` |
| Re-running search per alternate phrase | One `search "a" "b" "c"` scans all transcripts once |
| Hand-computing a window from a timestamp to find neighbours | `list --around ID` — one call, anchor excluded, signed `offset` |
| Raising `--limit` to reach an old session | Heed the truncation warning: `--sort oldest` or a `--since`/`--until` window gets there directly |
| Raising `--limit` because scratch sessions fill the results | Subtract them: `--exclude-cwd <path>`, which runs before `--limit` |
| `--repo` or `--cwd` on the parent project when the work ran in a git worktree | Only opencode records a project root; claude, codex and pi name a worktree session after the *worktree directory*, so `--repo` can return a tool name rather than a project. Scope to the worktree's distinctive segment, or search unscoped and read `cwd` off the hits |
| Answer found, but its artifact doesn't match the question's nouns | Wrong workstream — vocabulary recurs across projects; treat it as unconfirmed and keep searching |
| Your own or a sibling session pollutes results | Yours is auto-excluded; for a concurrent sibling add `--until -30m` |
| Pasting a full transcript into the final answer | Summarize it |
