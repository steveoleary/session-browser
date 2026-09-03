# session-browser — Agent Instructions

A Python CLI + TUI (`session_browser/`) that discovers and retrieves prior
agent sessions across providers (Claude Code, Codex, OpenCode, Pi). Tests
live alongside the code in `session_browser/`: `tests.py`, `test_cli.py`,
`test_transcript.py`, `test_case_runner.py`, `test_retrieval_compare.py` and
`test_performance.py`.

Committed regression fixtures live in `docs/fixtures/`. Each has a
`verify_case.py` and a declared accepted state; run them with
`python -m session_browser.case_runner run`. One of them,
`fresh-agent-skill-brief`, checks the documented retrieval workflow rather than
the code — run it after changing `skills/using-session-browser/SKILL.md` or the
argparse help, and see its README for the hand-run half.

`CLAUDE.md` is a symlink to this file. There is one set of instructions on
purpose — nothing here is Claude-specific, and a second copy is a copy that
drifts. Edit `AGENTS.md`.

## Running tests

```bash
source .venv/bin/activate
python -m pytest session_browser/ -q
```

## Linting

`ruff check .` — configured in `pyproject.toml`. **The baseline is zero
findings**; a `PostToolUse` hook lints every Python file as it is edited, so
new ones surface immediately rather than piling up. `ruff check --fix` handles
the safe ones.

`ruff format .` — the repo is format-clean and the edit hook checks it. Width is
88, ruff's default; 79 was measured first and produced *more* churn, because
more lines breached it and got exploded one-argument-per-line.

Record any pure whole-repo reformat in `.git-blame-ignore-revs` so `git blame`
skips it (GitHub honours it automatically; locally `git config
blame.ignoreRevsFile .git-blame-ignore-revs`, once per clone). Only list a
revision that really is pure — a wrong one makes a line's real author
unfindable.

The hook reports formatting rather than fixing it: rewriting a file underneath
an agent mid-edit invites it to work from a stale copy.

Three rules are switched off in `pyproject.toml` and each says why: `RUF001`/
`RUF002` (a TUI renders `×`, `›` and en dashes on purpose), `RUF005` (style),
`BLE001` (discovery reads three other tools' files; isolating a bad file is the
design, and every catch logs).

**A `# noqa` in this repo carries a reason, and one of them is perf.**
`transcript.py` suppresses `SIM105` on the per-line decode: `contextlib.suppress`
measured at **4.85x** the cost of `try`/`except` on the non-raising path, and
that path runs once per line of every transcript read. No counter in the perf
gate sees it, so the comment is the only thing defending it. `FURB166` is
suppressed twice as a genuine false positive — the slice drops a `\u` prefix,
not `0x`.

**Suppress on a measurement, not an intuition.** Two further suppressions were
added here on plausible-sounding perf reasoning and both were wrong when
measured: `enumerate()` is *faster* than a manual counter (CPython reuses the
result tuple when its refcount is 1), and collapsing a nested `if` into `and`
is identical to three decimal places, since `and` short-circuits the same way.
Both were reverted to the idiomatic form. A hot loop is exactly where intuition
about CPython is least reliable — `timeit` it before you reach for a noqa.

## Performance is a gate, not a trade-off

Retrieval speed is the product here. A change that slows it down is not weighed
against its benefit by default — the default answer is no, and anything else
needs the user to say so explicitly.

The gate is `session_browser/perf_budget.py`, enforced by `test_performance.py`
in the normal suite. It counts **work** — transcripts parsed, corpus bytes read,
ripgrep invocations, SQL statements, sessions routed to worker processes,
progress callbacks — against a fixed synthetic corpus, and compares every
counter to an exact number in `docs/perf_budgets.json`. It does not time
anything, deliberately: this machine's noise floor is an order of magnitude
larger than the regressions worth catching — nearer 0.2% — and the floor is
set mostly by what else is running, which an agent cannot control. *What moves
a timing measurement, and what does not*, below, has the numbers. Counts are
identical on every machine and every run.

**When a budget fails, do not bless it to make the suite green.** Read the
`Guards:` line in the failure — it says which optimisation the workload exists
to defend. Then either fix the change, or, if the extra work is genuinely
wanted:

1. Measure the real-world cost with `benchmarks/retrieval_compare.py` (below).
2. `python -m session_browser.perf_budget bless`, and say why in the commit
   message — the new numbers land in the diff so the decision stays visible.
3. Surface the trade-off in your handoff, before the commit lands. Consent is
   still required; it is given at review time, on the evidence, rather than
   through a modal mid-run.

Adding a workload is normal; run `bless` and the new numbers get recorded.
`python -m session_browser.perf_budget show` prints current counts without
touching the budgets.

**`bless` records what it changed, and the record is derived.** It writes a
`_blessed` block into `docs/perf_budgets.json` naming every counter that moved
and by how much (`400 -> 486 (+21.5%)`), plus anything newly budgeted, and
prints the same thing. You do not write that block and you cannot omit it, so a
regression blessed quietly still arrives in the diff with its size attached.
That is the whole mechanism: read the block, not the assurance.

The movement is measured against the file's **committed** version, not against
whatever a previous `bless` left on disk, so the block describes the diff a
reviewer is about to read. Blessing twice — which is what happens whenever
anything else in that generated file is adjusted and re-recorded — therefore
says the same thing twice instead of replacing a movement record with an empty
one.

One hook backs this up, and it asks nothing: a confirm modal interrupts a human
who has not yet seen the evidence, and fires on benign blesses too, which
teaches them to approve it unread.

- Writing `docs/perf_budgets.json` with Edit or Write is **denied**. The file is
  generated; a hand-edit leaves numbers no run produced and skips the `_blessed`
  block entirely.
- There is deliberately **no hook on the bless command itself**. One existed and
  was removed: it grepped the shell command as a flat string, so it fired on
  prose that merely mentioned the procedure. The guard lives in `bless`
  instead, which also binds a human, Codex or OpenCode — not just Claude Code.

### The second layer: loop opcode probes

The counters above work at the grain of a file, a statement, a parse, so CPU
added *inside* a loop that does the same I/O moves none of them. A second layer
closes most of that gap: one probe per hot loop drives it over a fixed input
and counts **executed Python opcodes**. Still a count, not a timing — identical
on every machine — but fine enough to see a loop body change.

They are enforced by `TestLoopOpcodeBudgets` alongside the workload budgets,
recorded in the same file under `loop_opcodes`, and re-recorded by the same
`bless`. Measured effects for scale: `contextlib.suppress` over `try`/`except`
in the per-line decode loop is **+11.8%**; a manual counter over `enumerate()`
in the opencode row loop is **+8.7%**.

Reading a loop breach differs from reading a workload breach. It is per
iteration, and `retrieval_compare.py` cannot resolve an effect this small — that
is why the probes exist, so do not reach for the comparator to justify one.
Time the two forms directly with `timeit` on a realistic input. A *fall* is
usually a win, but check first that the probe did not simply stop covering its
loop.

Opcode counts are a property of the interpreter. The budgets record the Python
they were taken on (`loop_python`), and on a different version the probes skip
with a message rather than failing — the one place this module goes quiet, and
it says so when it does. Re-bless after a Python upgrade.

What still is not covered: work that stays inside C. Swapping `str.find` for a
regex is about one opcode either way. The blind spot is now "in-loop CPU that
never returns to the interpreter" rather than all of it — smaller, and far more
visible in review. A green suite is still not proof a hot loop got no heavier.

## Comparing a retrieval change against another revision

Before shipping a change to search behaviour or the transcript parser, check it
against a prepared checkout of the revision you are comparing to. The comparator
never touches Git, so make the worktree yourself:

```bash
git worktree add /tmp/baseline-main main
python benchmarks/retrieval_compare.py \
  --baseline-repo /tmp/baseline-main \
  --candidate-repo . \
  --home "$HOME" \
  --current-session-env CLAUDE_CODE_SESSION_ID=<your live session id> \
  --query "some term" --query "another term" \
  --provider claude --repeats 9 --report /tmp/ab.json
git worktree remove /tmp/baseline-main
```

Reading the result:

- **Equivalence is not a flag.** A signature mismatch raises immediately, so a
  run that completes has already proven both revisions return the same search
  results. `report["equivalent"]` records this; `report["passed"]` answers only
  the timing question. Do not read `passed` as "output unchanged".
- **Exclude your own live session** with `--current-session-env`, or every query
  matching it becomes unjudgeable.
- **`verdict` per query is `ok`, `slower`, `unresolvable`, or `volatile`.**
  - `unresolvable` — the ratio exceeded the 5% limit by less than this machine
    could measure, so no verdict was earned. A fixed threshold on raw medians
    would otherwise report variance as regression. Quiet the machine and run it
    again; before reaching for any flag, read *What moves a timing measurement*
    below, because none of them is the answer.
  - `volatile` — the query matched a session that was appended to mid-run, so
    the query is excluded from the aggregate. A moving session file also makes
    two identical revisions disagree, which is why each revision must reproduce
    its own result before equivalence is compared. The run still fails if
    *every* query was volatile, since then nothing was measured.
  - Both are warned about on stderr and listed in the report. Neither is
    silently dropped.
- A change that does not alter what search reads cannot show a speed
  difference. Pick queries and a workload that actually exercise the change.

### What moves a timing measurement, and what does not

One place, because each of these was reached for first and each was wrong. The
short version: **quiet the machine; nothing else helps.**

**An idle machine is the only lever, and no flag can pull it.** Measured
2026-08-25 against the real `$HOME`. Null runs are a revision against a
worktree of itself, so every ratio should be 1.0 and only the noise floor is
interesting:

| machine state | noise floor, null runs | old-vs-new run |
| --- | --- | --- |
| an agent session working | 0.020, 0.031 | 0.071, 0.072 |
| nothing else running | 0.008, 0.011, 0.013 | 0.042 |

The limit a slowdown is judged against is 5%, so a busy machine cannot deliver
a verdict and an idle one resolves five times finer than it needs to. A
revision's own samples scatter by 1-21% of their median at the default
sampling — measured 2026-08-23 on an M1 Pro against a 1,180-session, 873 MB
`$HOME` — and which end of that band you land on is mostly machine state.

**`--repeats` is a floor on credibility, not a dial for resolution.** On the
idle machine 9 repeats scored 0.008-0.013 where 25 scored 0.024 — *worse*, a
longer sampling window admitting more drift; under load the two were
indistinguishable. The four-sample minimum is the part that matters, and it is
a refusal rather than a warning: fewer samples widen the floor, a wider floor
*accepts* more, and so the thinnest evidence would otherwise buy the loosest
test. Below four, `relative_spread` has no quartiles and returns the full range
instead — a different and much wider quantity — which is why that is a cliff
rather than a slope.

**`--scale` is not a sensitivity knob, however much it looks like one.** It was
built expecting a bigger corpus to quiet the measurement and dilute a
regression less. Fifteen null runs, 2026-08-25:

| corpus | runs | noise floor, lowest to highest |
| --- | --- | --- |
| synthetic x1 | 4 | 0.027, 0.035, 0.062, 0.095 |
| synthetic x8 | 3 | 0.035, 0.051, 0.063 |
| synthetic x16 | 4 | 0.019, 0.061, 0.065, 0.081 |
| synthetic x32 | 2 | 0.033, 0.047 |
| real `$HOME` | 2 | 0.020, 0.031 |

Neither expectation held. There is no ordering by size — run-to-run variation
at *one* size exceeds the difference between sizes, and the real `$HOME` is the
quietest corpus measured. The parse-sensitive share of a sample is flat near
47% from x1 to x32 as well, because discovery grows with the corpus rather than
being a fixed startup cost (bare interpreter and imports are only 76ms of it).
What `--scale` does control is how much work a query does, deterministically,
on any machine, and that is worth having on its own.

The trap that produced the opposite conclusion first is worth naming: one run
per size looked cleanly monotonic and did not survive replication. A noise
floor is itself a noisy quantity, so a single sample of one cannot rank two.

**Practically: an agent cannot take this measurement while it is the thing
making the noise.** A timing verdict wanted from this repo is a job for a human
on an otherwise-idle machine, or for an agent that has been handed a script to
be run after it stops. What an agent relies on instead is the counting gate and
the loop opcode probes, which are exact and identical on any machine, however
busy.

### One command instead of three: `compare_synthetic.py`

Preparing a `--home` by hand is how a comparator ends up pointed at a real
`$HOME` again, so `benchmarks/compare_synthetic.py` builds a throwaway
synthetic corpus, runs both revisions against it, and deletes it — even when
the comparison raises, which is exactly when a forgotten corpus would survive:

```bash
git worktree add /tmp/baseline-main main
python benchmarks/compare_synthetic.py \
  --baseline-repo /tmp/baseline-main --candidate-repo . \
  --repeats 9 --report /tmp/ab.json
git worktree remove /tmp/baseline-main
```

Queries are chosen by *selectivity* rather than by name — a term in no session
at all, one in ten, one in every session — so the default set spans the cost
curve and a parser change is loud in the last and invisible in the first. The
corpus manifest is merged into the report, so a ratio read back next month
still carries the scale, seed and session count behind every query, long after
the corpus is gone. A synthetic `$HOME` holds no live session, so
`--current-session-env` does not apply and `volatile` cannot arise.

Because a floor this wide is normal, the wrapper **warns when its own noise
floor sits at or above the 5% the comparator judges against**, and records
`resolved_the_limit` in the report. `ok` from such a run means nothing was
*resolvable*, not that nothing is wrong — a distinction the word `ok` cannot
carry by itself.

`benchmarks/` holds two smaller scripts as well, and neither judges anything —
they print timings for you to read, against your own corpus, with no baseline
and no verdict. `search_end_to_end.py` times whole-corpus search per query;
`search_ripgrep.py` times the raw candidate scan with ripgrep against the pure
Python fallback, by patching the ripgrep path off. Reach for them while
exploring where time goes; reach for `retrieval_compare.py` when a change needs
a decision.

## Local `session-browser` command

The command must run the current checkout, never a snapshot copied into `uv`'s
tool environment — so install it editable from the repository root, and never
as a plain `uv tool install .`, which does not pick up later source changes:

```bash
uv tool install --force --editable .
```

Verify application changes through `session-browser` itself. If it does not
reflect the checkout, rerun the above and restart any running TUI.

## Git hooks: the leak guard

Install once per clone — git does not clone hooks:

```bash
scripts/install-hooks.sh          # install / refresh
scripts/install-hooks.sh --check  # status, changes nothing
```

Two hooks, because at `pre-commit` time git has not written a message yet and
no amount of code there could ever see one:

| hook | script | what it reads |
| --- | --- | --- |
| `pre-commit` | `scripts/hooks/leak-guard` | staged file contents and staged paths |
| `commit-msg` | `scripts/hooks/leak-guard-msg` | the prepared commit message |

Each refuses a commit carrying text this clone has been told not to publish,
reporting `file:line` or the line within the message. The message guard reads
everything above the scissors line **including `#` comment lines**, since git
strips those only under some cleanup modes and `git commit -m`/`-F` record them
verbatim — so a pattern in a comment is a refusal, not a free pass.

Both load their patterns through `scripts/hooks/leak-patterns.sh` — one place
that knows where the list lives — and both **refuse rather than pass** if they
cannot read it. The `pre-commit` half is scoped to the files the commit
touches, so one offending tracked file cannot block every unrelated commit;
whole-tree and whole-history assurance is the preflight's job. Each script's
header carries the rest of the design, next to the code it defends; the rules
that bind you are here.

**The blocked patterns are not in this repository, and that is the design.** A
guard that shipped its own blocklist would publish the exact strings it exists
to keep out — the list *is* the leak. They live in `.git/config`, never pushed:

```bash
scripts/install-hooks.sh --add-pattern 'some-internal-name'
git config --get-all hooks.leakpattern      # read them back
```

A clone with no patterns configured is a clone with nothing to hide, which is
the right default for a public repo and for anyone else who clones it.

**Ignore rules for whatever you run alongside this repo belong in
`.git/info/exclude`, not in `.gitignore`.** Local tooling — an editor's state, a
tracker, a scratch database — leaves directories in the working tree that are
nobody else's business, and a `.gitignore` entry for one publishes both the tool
and the fact that it is used here. `.git/info/exclude` has the same syntax, is
per-clone, and is never pushed:

```bash
cat >> .git/info/exclude <<'EOF'
.some-tool-state/
EOF
git check-ignore -v .some-tool-state/    # prove it took
```

The cost is that a **fresh clone ignores nothing**, so write that file before
the tool creates its directory — otherwise `git status` offers it as an
ordinary untracked path, and preflight check 6 will not catch it either (it
finds paths that are ignored *and* tracked; here the path is simply not
ignored). Any untracked directory belonging to local tooling belongs here, not
in `.gitignore`.

Optionally the hook also checks the commit author, when a clone sets
`git config hooks.expectedemail 'you@example.com'`. Unset by default, so it is
inert for everyone else. It catches committing from a machine whose global git
identity is somebody else's — the failure that no directory convention covers.

In a checkout with a local Beads Dolt workspace the installer first runs
`bd hooks install --beads`, which makes `.beads/hooks` the active hook
directory; the project hooks are added outside Beads' managed markers, so
either installer can regenerate its own section safely. `--check` reports both
project hooks separately, how many patterns this clone has, and — in a Beads
workspace — that `.beads/hooks` is active with all five Beads hooks installed.
**Do not repoint `core.hooksPath` by hand**: only one directory is
active, so doing that can silently bypass either the project guards or the
tracker hooks. A public clone with no `.beads` workspace uses `.git/hooks` and
needs no Beads.

Escape hatch is git's own — `git commit --no-verify`.

### The whole-repo counterpart: `scripts/preflight-public.sh`

```bash
scripts/preflight-public.sh --remote origin
```

Eight checks, exit non-zero on any failure: local branch and tag names contain
no configured identifier and the remote carries only ordinary refs; no
configured identifier appears in the working tree or **anywhere in history — in
a blob, a commit message, an annotated tag's message or a git note**; every
commit is authored by the expected identity; gitleaks is clean over full
history; nothing tracked is also ignored; no `*.log` is tracked now or
historically; and `.git-blame-ignore-revs` names only commits that still exist.

Two are written by shape rather than by name, so they keep working against
tooling the script has never heard of: check 1 fails on any remote ref outside
`refs/heads`, `refs/tags` and `refs/pull`, and check 6 asks git for files that
are both tracked and ignored. The shape rule has one hole it cannot close — a
tool hiding data in a **branch** is using a perfectly ordinary ref — so check 1
also takes a per-clone list, unset by default like the identifier patterns:

```bash
git config --add hooks.forbiddenref 'some-generated-ref-name'
```

That list is a remote-topology rule; identifier-bearing *local* ref names are
owned by `hooks.leakpattern` and check 1, and object bodies by check 3, so the
two configurations never have to duplicate each other.

**No hit is ever reported by quoting the text**, which would reproduce the
identifier in the output of the tool that exists to contain it. Each is named
by the handle you would use to fix it: a local ref by abbreviated object ID
plus the `git branch --points-at` / `git tag --points-at` command that finds
it, a working-tree hit by `file:line`, a commit message by abbreviated hash, a
tag by tag name, a note by the abbreviated hash of the commit it annotates.

`--remote` takes a name *or* a URL, so a new public repo can be checked before
the working checkout is repointed at it; omitting it skips only check 1's
remote half.

**A run with no patterns configured fails.** A preflight that passes because it
was asked to look for nothing is worse than no preflight — it manufactures the
reassurance without doing the check. Skipped checks are counted separately and
called out in the summary for the same reason: a green run with three skips has
not proved what it appears to.

The script's header documents what each check reads and why, including the
history scan's four object kinds and the traps in the tag and note walks. Two
are worth knowing before you edit it: `git rev-list` peels a tag ref to the
commit it points at, so an annotated tag's own message is visited by nothing
else and this is the only place `git tag -a -m` can ever be caught; and
removing a note does not unreach the blob that held it, so the notes scan walks
those refs' history rather than their tips.

## Bundled skill: `using-session-browser`

`skills/using-session-browser/` is the **source of truth**. Runtime copies live
in `~/.agents/skills/` (Codex + OpenCode) and `~/.claude/skills/` (Claude Code),
and are overwritten wholesale by the installer. **Never edit either copy.**

### After changing the skill, sync it

This skill rides `npx skills`, and did not always: it was copy-managed out of
`~/Projects/skills` until `steveoleary/session-browser` went public, at which
point the skills repo migrated it to the npx lane — that repo's lane rule is
*whether npx can clone the source*, so a public home repo means npx. Its row is
gone from `scripts/external-skills.txt`, and `install.sh` now refuses it.

```bash
git commit -am "skill: <message>" && git push   # npx reads GitHub, so push FIRST
npx skills update --global                      # or: npx skills add \
                                                #   steveoleary/session-browser \
                                                #   -s using-session-browser -g -y
```

**The push is not optional any more.** The old copy path read this working
tree, so an uncommitted edit installed fine; npx reads GitHub, so an unpushed
change simply does not reach either store. That trade was accepted knowingly
when the repo went public — the other half of it is that
`install.sh --check` no longer byte-verifies this skill, so verify by hand:

```bash
diff -r skills/using-session-browser ~/.claude/skills/using-session-browser
diff -r skills/using-session-browser ~/.agents/skills/using-session-browser
```

Start a fresh agent session afterward so stale injected skill text isn't reused.
