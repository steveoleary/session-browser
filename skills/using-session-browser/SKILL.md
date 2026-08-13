---
name: using-session-browser
description: Use when finding, searching, or retrieving prior agent sessions, transcripts, handoffs, decisions, lessons, or context with the session-browser CLI.
---

# Using Session Browser

`session-browser` retrieves Claude Code, Codex, and OpenCode
sessions. Core rule: narrow before reading; large transcripts go to files.

## Retrieval Workflow

1. If uncertain, run `session-browser --help`. Current commands are `list`,
   `search`, `get`, and `stats` (`show` is an old design name). If a command
   or flag this skill teaches errors with `invalid choice` or `unrecognized
   arguments` (e.g. `stats`, `--here`, `--around`, `--exclude-cwd`, `search`'s
   `--sort oldest`/`--max-snippets`, or `get`'s
   `--entries`/`--head`/`--tail`/`--role`/`--clip`/multi-id batching come
   back unknown), the binary on your PATH is **stale**, not the
   skill wrong — do not work around it by dumping full transcripts. Refresh the
   install (`uv tool install --force --from <repo> session-browser`, or your
   package manager's upgrade) and retry the narrowing command.
2. Discover with metadata first. Prefer `--cwd` over `--repo`; repository
   metadata can be empty. To orient in an unfamiliar history — which
   providers and directories hold the sessions, how recent they are — run
   `session-browser stats --format json` (shared filters apply, e.g.
   `--here`); it returns provider counts with `last_activity`, per-day
   activity buckets, and `top_cwds` values usable directly with `--cwd`.
   Never dismiss a session as noise from its `summary` text alone — the
   summary is often just the first user message, so a bare `/model` or
   `/clear` label can hide hours of real work. Every `list` result carries
   the two signals that catch this: `total_entries` (null if unreadable)
   and `duration_seconds` (null if a timestamp is missing) — check them
   before skipping a session. `total_entries` is the *primary* signal; the
   trap runs both ways, and a huge `duration_seconds` on a 2-entry session
   is a stale record whose file was touched days later, not deep work. `list` also names unreadable and zero-entry
   records in `warnings` — treat those as likely-corrupt records, not as
   sessions where nothing happened. When one session is already known and the
   question is what else was happening near it, `list --around ID` returns
   its temporal neighbors directly (see Shared filters) — do not hand-compute
   a `--since`/`--until` window from its timestamp.
3. Search with rare phrases in `--mode ids` first. Pass alternate phrasings
   together — `search "phrase a" "phrase b"` ORs them in a single scan
   (`--match-all` to require every phrase). Add `--provider`, `--cwd`,
   `--since`, or `--until` when possible. "Rare" means rare *in this
   corpus*: ordinary English words ("digit", "sensitive", "table") and
   domain vocabulary shared across projects ("App Store", "tethered")
   drown the target in cross-project noise — prefer multiword phrases the
   answer itself would contain. Search also scans each session's
   *summary* (title), so words from how the work would be titled ("admin
   app", "revenue report") are good phrases; a summary-only hit returns
   `match_count 0` with the phrases in `summary_matches`. Matching is
   markdown-insensitive (backticks/asterisks stripped: "SELECT only"
   finds "`SELECT` only") but still literal, not semantic. If the target
   may be *old*, don't trust the default recency sort + `--limit`: a
   truncation warning reports how many matches were dropped and their
   date range — when it fires, re-run with `--sort oldest`, a
   `--since`/`--until` window, or `--sort matches`.
4. Triage with `--mode snippets --context 120 --limit N`; `80..200` means
   choose one context number. Each result carries `total_entries`,
   `first_match`, and `last_match` (entry indices); `--sort matches` ranks
   by hit count instead of recency. A high `match_count` on a scoped
   search is the strongest triage signal there is — summaries can be
   blank or misleading, but a session that says the word 200 times is
   where the work happened. Snippets are capped at 20 per result
   (`snippets_omitted` counts the overflow; `--max-snippets N` adjusts,
   0 = unlimited), so a phrase that hits hundreds of times can't flood
   the output.
5. Retrieve only what you need. For a matched region or a session's ending,
   read a window directly: `get ID --entries A:B` (indices from
   `entry_index`/`first_match`/`last_match`), `--tail N`, or `--head N`.
   To read one voice, filter by role: `get ID --role user` (the intent
   trail), `--role assistant` (conclusions without tool noise), or
   `--role error` (failed tool calls only). `--role user` is *human speech
   only*: prompts typed while the agent was still working are included, and
   harness-injected text is not. Interrupt markers, background task
   notifications and the auto-written summary at the top of a resumed
   session are `--role system`, so when a session was compacted its earliest
   intent lives there, not in `--role user`. Kept entries are prefixed
   `[index]` with their absolute position, so feed an interesting index
   back to `--entries` to read its surroundings. `--head`/`--tail` compose
   with `--role`: `get ID --role user --tail 5` is the last five *user
   turns* (`--entries` always stays absolute). Sessions self-correct: a
   confident mid-session claim is often reversed a few entries later
   (wrong conclusion → user pushback → correction), so treat a snippet or
   mid-session window as a *lead*, not a verdict — before quoting a
   decision or finding, check the ending (`--role assistant --tail N`)
   for the final, post-correction state. Several sessions batch into
   one call — `get ID1 ID2 ID3 --role user --head 3` — JSON wraps them as
   `{"sessions": [...], "skipped": [...]}`. stdout clips each entry to
   4000 chars with a `[clipped N chars]` marker so one embedded tool dump
   can't flood your context — no need to pipe through `head -c`; pass
   `--clip 0` for complete text. For complete transcripts
   use `get ID --output FILE` (never clipped by default) or
   `search ... --mode full --output-dir DIR`;
   check the scratch path is ignored.
6. Synthesize. Mention `skipped` sessions or parse warnings when JSON reports
   them.

## Quick Reference

| Need | Command |
|------|---------|
| Recent sessions | `session-browser list --cwd project --limit 10` |
| Sessions from this project | `session-browser list --here --limit 10` |
| What else was happening around a known session | `session-browser list --around ID --limit 15` (nearest-first, so `--limit` = the N nearest neighbors — no window guessing; `--window 2h` to tighten, `--here` for same-project only) |
| Orient: providers/dirs/recency overview | `session-browser stats --here --format json` (text = human dashboard) |
| Find candidates | `session-browser search "rare phrase" --cwd my-project --mode ids --limit 10` |
| Alternate phrasings, one scan | `session-browser search "phrase a" "phrase b" --mode ids` (OR; add `--match-all` for AND) |
| Inspect matches | `session-browser search "rare phrase" --cwd my-project --mode snippets --context 120 --limit 5` |
| Rank by hit count | add `--sort matches` (default is recency) |
| Sweep dominated by harness/scratch sessions | `--exclude-cwd /private/tmp/` (repeatable; scope to the noisy path you actually saw in `stats`' `top_cwds`, not a bare word like `scratchpad` that may be a real project) |
| Target may be old / buried under newer results | `--sort oldest` (on `list` *and* `search`), or a `--since`/`--until` window — heed the truncation warning that reports dropped results and their date range |
| Search by how the work would be *titled* | just `search` — summaries are scanned too; summary-only hits have `match_count 0` + `summary_matches` |
| Phrase contains \`backticks\`/**bold** formatting? | nothing — matching strips backticks/asterisks on both sides ("SELECT only" finds "\`SELECT\` only") |
| One session matches a phrase hundreds of times | nothing — snippets cap at 20/result with `snippets_omitted`; `--max-snippets N` to adjust (0 = unlimited) |
| Is this bare-looking session really noise? | Check `total_entries` and `duration_seconds` in its `list` result — never judge from `summary` alone |
| How a session ended (handoff/decision) | `session-browser get ID --tail 20` |
| Read only the matched region | `session-browser get ID --entries 38:60` (use `first_match`/`last_match`/`entry_index` from search) |
| What the user asked for (intent trail) | `session-browser get ID --role user` |
| Last N user turns (not raw entries) | `session-browser get ID --role user --tail 5` |
| Triage several candidates at once | `session-browser get ID1 ID2 ID3 --role user --head 3` (one call; JSON wraps as `{"sessions": [...]}`) |
| One entry embeds a giant tool dump | nothing — stdout clips entries at 4000 chars with a marker; `--clip 0` or `--output FILE` for complete text |
| What went wrong (failed tool calls) | `session-browser get ID --role error` (also `assistant`, `tool`, `system`; comma-combine like `--role user,assistant`) |
| Original intent in a compacted/resumed session | `session-browser get ID --role system --head 3` (the auto-written summary is not a user turn) |
| Where a run was interrupted | `session-browser get ID --role system` (interrupt markers and task notifications live here) |
| Known session | `session-browser get provider:id --output build/session.md` |
| Multi-session review | `session-browser search "rare phrase" --mode full --output-dir build/session-hunt --limit 3` |
| Structured parsing | add `--format json` |
| Your own session pollutes results | nothing — it is auto-excluded (use `--include-current` to keep it) |
| A *sibling*/concurrent session still pollutes | add `--until -30m` (relative bound) |

If the user hands you a canonical id like `claude:76458688-…` (copyable from
the TUI with `i`), pass it straight to `get ID`; no `list`/`search` first. A
unique id prefix also resolves, like a git short hash — `get claude:7645` or
bare `get 7645` — useful when you only have a truncated id from a snippet or
a partial paste; an ambiguous prefix errors and lists the candidates, and a
full id that matches nothing suggests near-miss ids ("did you mean"), which
recovers a typo mid-id that a prefix can't.

Shared filters: `--provider`, `--repo`, `--cwd`, `--exclude-cwd`, `--here`,
`--since`, `--until`, `--around`/`--window`, `--include-current`, `--limit`.
`--exclude-cwd SUBSTR` is the only *negative* filter: case-insensitive literal
substring, repeatable (a session is dropped if any pattern matches), and ANDed
with `--cwd`/`--here`. It is opt-in and applied **before** `--limit`, so
excluded sessions never consume the result budget. Sessions with no recorded
cwd are kept (nothing to match); `--here` is what drops those.
Whenever `--limit` drops results, `list` and `search` both say so — count and
date range — on `warnings` (stderr in `--format text`). A listing with no such
warning is complete. `--sort` is *not* a shared filter: `list` and `search`
take `recent`/`oldest`, `--sort matches` is `search`-only, and `stats` has no
`--sort` at all (it aggregates every session that passed the filters).
`--around SESSION` (any id form `get` accepts) anchors the date window on
another session: it keeps sessions whose last activity falls within
`--window` (default `1d`) of the anchor's, excludes the anchor itself, sorts
nearest-first, and adds a signed `offset` field (`-24m`, `+2h48m`) to every
result. Adjacent sessions often continue or parallel the anchor's work, so
reach for it whenever one session is known and the question is "what else was
going on". Don't guess window sizes: results sort nearest-first, so
`--around ID --limit 15` is a k-nearest query — the 15 closest sessions
regardless of window; adjust `--window` only when you need a hard time cutoff.
Judge neighbors by `offset` *and* `duration_seconds`
together: a small offset on a very long session is a concurrent sibling
running through the window, not a predecessor. `--around` is not combinable
with `--since`/`--until` (adjust `--window` instead). Dates are ISO 8601 *or* relative
(`-30m`, `-2h`, `-1d`, `-1w`), and bound on **last activity** — the newest
message's timestamp inside the transcript, not the file's mtime — so a session
that was merely read, synced, or reopened won't drift into a date window it
doesn't belong to (e.g. a session last worked on yesterday stays out of a
`--since` filter for today). `--here` scopes to the current working directory
by exact path-prefix (the project tree you are in); use it instead of guessing
a `--cwd` substring. Your own live session is auto-excluded by default
(detected from the agent's session-id env var, reported in `warnings`); pass
`--include-current` to keep it. That covers the common echo case for Claude and
Codex; for a *sibling* or concurrent session that still pollutes, add `--until
-30m`. Search is case-insensitive literal matching; `search --limit` limits
matches after scanning, not candidates.

## Example

```bash
session-browser list --cwd session-browser --limit 10 --format text
session-browser search "agent-queryable CLI" "queryable surface" --cwd session-browser --until 2026-06-10T09:00:00Z --mode ids --limit 5
session-browser search "agent-queryable CLI" --cwd session-browser --until 2026-06-10T09:00:00Z --mode snippets --context 120 --limit 3
session-browser get claude:76458688-b620-42b0-957f-4b64e9cc784b --entries 38:60   # window from first_match/last_match — no file needed
git check-ignore -q build || echo "choose a different ignored scratch path"
session-browser get claude:76458688-b620-42b0-957f-4b64e9cc784b --output build/agent-queryable-session.md   # only if the whole transcript matters
rg -n -C 3 "Command surface|Non-Goals|Output" build/agent-queryable-session.md
```

## Baseline Failures To Avoid

| Failure | Better Move |
|---------|-------------|
| Broad query like `session-browser`, `search`, `full`, `output`, `token` | Use rare phrases and filters first |
| Triaging a session as noise from its `summary` text | Summaries lie (often just the first user message); check the `total_entries` and `duration_seconds` that `list` already returned |
| Reading a huge `duration_seconds` as deep work | `total_entries` is primary — 66h duration on a 2-entry session is a stale record whose file was touched later |
| Reading implementation docs before trying the CLI | Use `--help`, `list`, and `search` unless changing the tool |
| Blind `get ID` to stdout | Use `--entries`/`--head`/`--tail` for a window, `--role` for one voice, `--output` for full transcripts |
| Full read just to recover what the user wanted or where it failed | `get ID --role user` / `get ID --role error`, then window with `--entries` around the interesting index |
| Full export + grep just to see one region | `get ID --entries A:B` straight from snippet `entry_index` |
| One `get` per candidate while triaging several | Batch them: `get ID1 ID2 ID3 --role user --head 3` |
| Piping `get` through `head -c` to protect context | Built-in: stdout clips per-entry at 4000 chars; tune with `--clip N` |
| Re-running search per alternate phrase | One `search "a" "b" "c"` scans all transcripts once |
| Hand-computing `--since`/`--until` from a `get`'d timestamp to find neighbors | `list --around ID` — one call, anchor excluded, signed `offset` per result |
| `search --mode full` to stdout | Use `--output-dir` or snippets first |
| Relying on `--repo` only | Prefer `--cwd`; repo may be empty |
| Common English words as OR terms ("digit", "sensitive", "table") | They match everywhere; use multiword phrases the answer itself would contain |
| Raising `--limit` to reach an old session under the recency sort | Heed the truncation warning: `--sort oldest` or a `--since`/`--until` window gets there directly. Raising the limit is not wrong, just slower — results nest, so a bigger limit is always a superset, never a shifted window |
| Raising `--limit` because throwaway harness/scratch sessions fill the results | Subtract them instead: `--exclude-cwd <path>`. It runs before `--limit`, so the budget is spent on real work. Read the noisy prefix off `stats --format json` → `top_cwds` first |
| `--cwd` on the parent repo when the work ran in a worktree | Worktree sessions record the worktree path — scope to its distinctive segment (e.g. `--cwd worktrees/MyApp/feature`), or search unscoped and read `cwd` off the hits |
| Answer found, but its artifact doesn't match the question's nouns | Wrong workstream — similar vocabulary recurs across projects; treat it as unconfirmed and keep searching (scope by `--cwd`, try the question's nouns as summary phrases) |
| Quoting a conclusion from a mid-session snippet or window | Sessions self-correct (claim → pushback → reversal); confirm against the ending with `--role assistant --tail N` before reporting it |
| Your own session appears | It shouldn't — it is auto-excluded; if it does, the env var was absent, so add `--until -30m` |
| A sibling/concurrent/RED-GREEN session appears | Add `--until -30m`; prefer older source sessions |

## Common Mistakes

- Do not paste full transcripts into the final answer; summarize them.
- Do not assume search is semantic. Pass alternate literal phrases together
  in one `search` call.
- Do not treat `snippets` as complete context. Use `get` when exact decisions
  matter.
- Existing output paths require `--overwrite`; use it only intentionally.
- The 4000-char per-entry clip applies to stdout only; `--output` files are
  complete unless you pass `--clip` explicitly. A `[clipped N chars]` marker
  inside an entry means rerun with `--clip 0` (or `--entries i --clip 0`)
  if the omitted part matters.
- `--here` can't see sessions with no recorded cwd. If its
  `warnings` reports excluded sessions and they matter, drop `--here` and source
  them another way: `--cwd <name>`, a content `search`, or `--provider`.
