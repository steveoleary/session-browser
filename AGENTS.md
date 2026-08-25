# session-browser — Agent Instructions

A Python CLI + TUI (`session_browser/`) that discovers and retrieves prior
agent sessions across providers (Claude Code, Codex, OpenCode, Pi). Tests
live alongside the code in `session_browser/`: `tests.py`, `test_cli.py`,
`test_transcript.py`, `test_case_runner.py`, `test_retrieval_compare.py` and
`test_performance.py`.

Committed regression fixtures live in `docs/fixtures/`. Each has a
`verify_case.py` and a declared accepted state; run them with
`python -m session_browser.case_runner run`.

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

If you ever do a pure whole-repo reformat, record it in a
`.git-blame-ignore-revs` so `git blame` skips it — GitHub honours that file
automatically, and locally it needs `git config blame.ignoreRevsFile
.git-blame-ignore-revs` once per clone. List a revision there only if it really
is pure, since a wrongly listed commit makes a line's real author unfindable.

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
anything, deliberately: a revision's own samples scatter by 1-21% of their
median at the comparator's default sampling — measured 2026-08-23 on an M1 Pro
against a 1,180-session, 873 MB `$HOME`, and far worse on fewer samples — while
the regressions worth catching are nearer 0.2%. Counts are identical on every
machine and every run.

Which end of that 1-21% band you land on is mostly whether anything else was
running: the bottom of it needs an idle machine, and an agent cannot give
itself one. That is the whole reason the primary gate counts instead of
timing — see the comparator section below for the measurements.

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

One hook backs this up, and it asks nothing — a hook that raises a confirm
modal interrupts a human who has not yet seen the evidence, and fires on benign
blesses too, which teaches them to approve it unread. What guards the file now,
and what deliberately does not:

- Writing `docs/perf_budgets.json` with Edit or Write is **denied**. The file is
  generated; a hand-edit leaves numbers no run produced and skips the `_blessed`
  block entirely.
- There is deliberately **no hook on the bless command itself**. One existed and
  was removed: it grepped the shell command as a flat string, so it could not
  tell a command from a quotation of one and fired on prose that merely
  mentioned the procedure. The guard lives in `bless` instead, which also means
  it binds anyone — a human, Codex, OpenCode — not just Claude Code.

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
    can measure. A revision's own samples scatter by 1-21% of their median at
    the default sampling, so a fixed threshold on raw medians reports variance
    as regression. Quiet the machine and run it again — measured 2026-08-25,
    that is the only lever, and where in the 1-21% band you land is mostly
    which machine state you were in. Raising `--repeats` above the minimum
    buys nothing here. Never lower `--repeats` to make a run cheaper: fewer
    samples widen the noise floor, and a wider floor *accepts* more, so the
    thinnest evidence buys the loosest test. Below four samples that stops
    being a matter of degree — `relative_spread` has no quartiles there and
    returns the full range instead, a different and much wider quantity — so
    the comparator refuses the run rather than report a verdict resting on it.
  - `volatile` — the query matched a session that was appended to mid-run, so
    the query is excluded from the aggregate. A moving session file also makes
    two identical revisions disagree, which is why each revision must reproduce
    its own result before equivalence is compared. The run still fails if
    *every* query was volatile, since then nothing was measured.
  - Both are warned about on stderr and listed in the report. Neither is
    silently dropped.
- A change that does not alter what search reads cannot show a speed
  difference. Pick queries and a workload that actually exercise the change.

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

**`--scale` is not a sensitivity knob, however much it looks like one.** It was
built on the expectation that a bigger corpus would quiet the measurement and
dilute a regression less. Measured 2026-08-25 over fifteen null runs — a
revision against a worktree of itself, so every ratio should be 1.0 and only
the noise floor is interesting — neither holds:

| corpus | runs | noise floor, lowest to highest |
| --- | --- | --- |
| synthetic x1 | 4 | 0.027, 0.035, 0.062, 0.095 |
| synthetic x8 | 3 | 0.035, 0.051, 0.063 |
| synthetic x16 | 4 | 0.019, 0.061, 0.065, 0.081 |
| synthetic x32 | 2 | 0.033, 0.047 |
| real `$HOME` | 2 | 0.020, 0.031 |

There is no ordering by size: run-to-run variation at *one* size exceeds the
difference between sizes, and the real `$HOME` is the quietest corpus measured.
The parse-sensitive share of a sample is flat near 47% from x1 to x32 as well,
because discovery grows with the corpus rather than being a fixed startup cost
— bare interpreter and imports are only 76ms of it. So do not raise `--scale`
to resolve a smaller effect. What `--scale` does control is how much work a
query does, deterministically, on any machine, and that is worth having on its
own.

The trap that produced the opposite conclusion first is worth naming: one run
per size looked cleanly monotonic and did not survive replication. A noise
floor is itself a noisy quantity, so a single sample of one cannot rank two.

**The one lever that does work is an idle machine, and no flag can pull it.**
Measured 2026-08-25, the same null comparison against the real `$HOME`:

| machine state | noise floor, null runs | old-vs-new run |
| --- | --- | --- |
| an agent session working | 0.020, 0.031 | 0.071, 0.072 |
| nothing else running | 0.008, 0.011, 0.013 | 0.042 |

More samples do not substitute for it and can cost you: on the idle machine
9 repeats scored 0.008-0.013 where 25 scored 0.024, a longer sampling window
admitting more drift. So `--repeats` is a floor on credibility, not a dial for
resolution — the comparator's four-sample minimum is the part that matters.

**Practically: an agent cannot take this measurement while it is the thing
making the noise.** A timing verdict wanted from this repo is a job for a human
on an otherwise-idle machine, or for an agent that has been handed a script to
be run after it stops. What an agent relies on instead is the counting gate and
the loop opcode probes, which are exact and identical on any machine, however
busy.

Because of all that, the wrapper **warns when its own noise floor sits at or
above the 5% the comparator judges against**, and records `resolved_the_limit`
in the report. `ok` from such a run means nothing was *resolvable*, not that
nothing is wrong — a distinction the word `ok` cannot carry by itself.

`benchmarks/` holds two smaller scripts as well, and neither judges anything —
they print timings for you to read, against your own corpus, with no baseline
and no verdict. `search_end_to_end.py` times whole-corpus search per query;
`search_ripgrep.py` times the raw candidate scan with ripgrep against the pure
Python fallback, by patching the ripgrep path off. Reach for them while
exploring where time goes; reach for `retrieval_compare.py` when a change needs
a decision.

## Local `session-browser` command

The `session-browser` command must always run the code from the current
checkout, not a stale snapshot copied into `uv`'s tool environment. Install it
from the repository root as an editable tool:

```bash
uv tool install --force --editable .
```

After changing application code, verify behavior through `session-browser`
itself. If the command does not reflect the current checkout, rerun the command
above and restart any already-running TUI process. Do not use a non-editable
local `uv tool install .`, because subsequent source changes will not be picked
up.

## Git hooks: the leak guard

Install once per clone — git does not clone hooks:

```bash
scripts/install-hooks.sh          # install / refresh
scripts/install-hooks.sh --check  # status, changes nothing
```

In a checkout with a local Beads Dolt workspace, the installer first runs
`bd hooks install --beads`. That makes `.beads/hooks` the active hook directory
through `core.hooksPath`; the project hooks are added outside Beads' managed
markers, so either installer can regenerate its own section safely. A public
clone with no `.beads` workspace keeps Git's normal `.git/hooks` directory and
does not need Beads.

Two hooks are installed, because one cannot do both jobs:

| hook | script | what it reads |
| --- | --- | --- |
| `pre-commit` | `scripts/hooks/leak-guard` | staged file contents and staged paths |
| `commit-msg` | `scripts/hooks/leak-guard-msg` | the prepared commit message |

`scripts/hooks/leak-guard` refuses a commit that stages text this clone has been
told not to publish, checking both file contents and file paths, and reporting
`file:line`. It only reads the files the commit actually touches. That scoping is
deliberate: an unscoped grep over the index means one offending tracked file
blocks *every* unrelated commit in the repo, which trains you to type
`--no-verify` by reflex and buys nothing. Whole-tree and whole-history assurance
is the preflight script's job, not the hook's.

`scripts/hooks/leak-guard-msg` refuses a commit whose **message** carries one of
the same patterns, reporting the line within the message. It is a second hook
rather than more of the first for a structural reason: at `pre-commit` time git
has not written a message yet, so no amount of added code there could ever see
one. That was the hole an identifier went through — a commit body naming a
tracker ID passed the pre-commit guard cleanly and was caught afterwards by a
human grepping `git log --format=%B`, which is the manual step these guards exist
to remove.

It checks everything above the scissors line, **comment lines included**, which
is deliberate: git only discards `#` lines under some cleanup modes. An editor
commit strips them; `git commit -m` and `-F` record them verbatim. The hook
cannot tell which mode it was invoked under, and guessing permissively means
passing exactly the message that gets published — so a pattern anywhere git might
keep it is a refusal. The scissors tail is the one part always dropped (that is
where `git commit --verbose` writes the staged diff), and it belongs to the other
guard anyway.

Both hooks read their patterns through `scripts/hooks/leak-patterns.sh`, which
exists so there is one place that knows where the list lives. A second copy of
that loop is a copy that can drift, and a guard reading a stale list is a guard
that passes what its neighbour refuses. All three readers — both hooks and the
preflight — go through it, and each **refuses rather than passes** if it cannot
load the list: a guard that cannot read its patterns has checked nothing, and
reporting success would be a lie told on exactly the day it mattered.

**The blocked patterns are not in this repository, and that is the design.** A
guard that shipped its own blocklist would publish the exact strings it exists to
keep out — the list *is* the leak. They live in `.git/config`, which is never
pushed:

```bash
scripts/install-hooks.sh --add-pattern 'some-internal-name'
git config --get-all hooks.leakpattern      # read them back
```

A clone with no patterns configured is a clone with nothing to hide, which is the
right default for a public repo and for anyone else who clones it.

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

The cost is that a **fresh clone ignores nothing**, so the first thing to do
after cloning is write that file — before the tool in question creates its
directory, or `git status` will offer it to you as an ordinary untracked path.
Preflight check 6 does not cover this case: it catches paths that are ignored
*and* tracked, and on a clone with no excludes such a path is simply not ignored.
If you find an untracked directory belonging to local tooling, add it here.
Adding it to `.gitignore` is the wrong fix and undoes the point.

Optionally the hook also checks the commit author, when a clone sets
`git config hooks.expectedemail 'you@example.com'`. Unset by default, so it is
inert for everyone else. It catches committing from a machine whose global git
identity is somebody else's — the failure that no directory convention covers.

The installer appends a **marked section** to the active `pre-commit` and
`commit-msg` rather than owning either file, because another hook owner may
manage its own section in the same file. Re-running replaces only the project
sections and leaves the rest intact, and says how many foreign lines it
preserved. `--check` reports both hooks separately; in a Beads workspace it also
verifies that `.beads/hooks` is active and all five Beads hooks are installed.
Do not repoint `core.hooksPath` by hand: only one directory is active, so doing
that can silently bypass either the project guards or the tracker hooks.

Escape hatch is git's own — `git commit --no-verify`.

### The whole-repo counterpart: `scripts/preflight-public.sh`

```bash
scripts/preflight-public.sh --remote origin
```

Eight checks, exit non-zero on any failure: local branch and tag names contain
no configured identifier and the remote carries only ordinary refs; no
configured identifier appears in the working tree or **anywhere in history —
in a blob, a commit message, an annotated tag's message or a git note**; every
commit is authored by the
expected identity; gitleaks is clean
over full history; nothing tracked is also ignored; no `*.log` is tracked now or
historically; and `.git-blame-ignore-revs` names only commits that still exist.

Two of those are written by shape rather than by name, so they keep working
against tooling this script has never heard of. Check 1 applies the configured
`hooks.leakpattern` list to local `refs/heads` and `refs/tags`, then fails on any
remote ref outside `refs/heads`, `refs/tags` and `refs/pull`, because anything
else is something using the repository as a data store. Check 6 asks git for
files that are both tracked and ignored: an ignore rule that does not actually
keep a path out is decorative, and the file ships regardless of what it says.

A local ref-name hit is reported by abbreviated object ID, never by name: use
the supplied `git branch --points-at` or `git tag --points-at` command to find
it, then the rename commands to repair it. Printing the ref would reproduce the
identifier in the output of the tool that exists to contain it.

The shape rule has one hole it cannot close by itself — a tool that hides data
in a **branch** is using a perfectly ordinary ref. So check 1 also takes a
per-clone list of ref patterns:

```bash
git config --add hooks.forbiddenref 'some-generated-ref-name'
```

Unset by default, exactly like the identifier patterns, and for the same reason:
a clone with none configured is a clone with nothing to hide. This second list
remains a remote-topology rule; local identifier-bearing names are owned by
`hooks.leakpattern`, so the two configurations do not have to duplicate each
other.

**"Anywhere in history" means every object a person can write prose into.**
Check 3 walks every version of every file, then every commit message reachable
from every ref, then every annotated tag's message, then every git note. It did
not always: for a while it read blobs only, and an identifier quoted in a commit
body survived a run reporting eight passes — found by grepping `git log` by hand,
not by this script. Closing that hole is what put the rest on the list, because
each is the same shape — free text in an object no blob scan visits.

Each kind of hit is reported by the handle you would use to go and fix it, and
never by quoting the text: a commit message by abbreviated hash, a tag by tag
name, a note by the abbreviated hash of the commit it annotates (what `git notes
show` takes). Reproducing the identifier in the output of the tool that exists to
contain it defeats the point, and a message has no line worth citing anyway.

The message scan is two-pass. One grep per pattern over all messages at once
answers *whether* there is anything; only a hit triggers the walk that says
*which commit*. A clean history — the normal case — never pays for the
attribution.

Two details of the tag and note scans are deliberate and easy to get wrong:

- A tag object is read from its message down, headers stripped. `git rev-list`
  peels a tag ref to the commit it points at, so the tag object is never visited
  by anything else, and there is **no hook counterpart** — git has no tag-msg
  hook — which makes the preflight the only place `git tag -a -m` can be caught.
  Stripping the headers keeps the tagger's address out of the scan (check 4's
  business) and means a tag whose **name** carries an identifier is not caught
  here; a local ref name belongs to check 1 and `hooks.leakpattern`.
- Notes are scanned from `refs/notes/*` explicitly, and those refs are excluded
  from the blob walk. They are reachable from `--all`, so the blob grep did in
  fact read note text — but it reported the hit as a *path*, and a note's path is
  the annotated object's hash, so the report named a 40-hex "file" present in no
  tree. Coverage nobody wrote down is coverage the next edit can drop in silence.
  The scan walks the notes refs' own history, so an edited or removed note is
  still read: removing a note does not unreach the blob that held it.

`--remote` takes a name *or* a URL, so a new public repo can be checked before
the working checkout is repointed at it. Local branch and tag names are checked
either way; omitting it skips only check 1's remote half.

It reads the same untracked `hooks.leakpattern` config as the hook. **A run with
no patterns configured fails.** A preflight that passes because it was asked to
look for nothing is worse than no preflight — it manufactures the reassurance
without doing the check. Skipped checks are counted separately and called out in
the summary for the same reason: a green run with three skips has not proved what
it appears to.

The `*.log` check is written by shape rather than by filename deliberately, so
the file that made it necessary does not have to be named in a public repository
in order to be caught.

## Bundled skill: `using-session-browser`

This repo bundles one agent skill under `skills/using-session-browser/`. That
directory is the **source of truth** — edit it there, never in a runtime
install directory. The runtime copies are:

- `~/.agents/skills/using-session-browser/` (Codex + OpenCode — universal store)
- `~/.claude/skills/using-session-browser/` (Claude Code — its own real copy)

This skill is **copy-managed** by the skills repo's `scripts/install.sh` (it's
listed in that repo's `scripts/external-skills.txt`). Never edit a runtime copy.

### After changing the skill, sync it

Assumes the standard layout: this repo at `~/Projects/session-browser`, the
skills repo at `~/Projects/skills`. `install.sh` copies the skill's git-tracked
files straight from this checkout, so:

```bash
git commit -am "skill: <message>" && git push        # persist + reach other machines (topgrade)
~/Projects/skills/scripts/install.sh using-session-browser   # copy into both stores + verify
```

That places the skill in `~/.agents/skills/using-session-browser` (Codex +
OpenCode) and `~/.claude/skills/using-session-browser` (Claude Code — a real
copy, not a symlink) and byte-verifies each. Re-check any time with
`~/Projects/skills/scripts/install.sh --check using-session-browser`.

This `install.sh` copy is the **only** way to sync — never `npx skills add`.
Start a fresh agent session afterward so stale injected skill text isn't reused.
