# Subagent ground truth

Everything session-browser knew about subagent links had been read off sessions
that happened to exist. That proves the fields are there, but not that the
graph is right, because no parent-child set was known to be correct. A bug that
attached a child to the wrong parent, dropped one, or double-counted a fan-out
would still look plausible.

This fixture is that known-correct set. Its subagents were spawned on purpose
on 2026-09-13, and every edge in `ground_truth.json` comes from the **spawning
side**: the tool result that answered the spawn call. Discovery reads the other
end of the edge (the sidecar and directory for Claude Code, `session.parent_id`
for OpenCode, `source.subagent.thread_spawn` for Codex), so a linking bug has no
way to agree with itself here.

## What was spawned

**Claude Code 2.1.270** (`claude -p`, Haiku 4.5), one session, two turns:

- turn 1: `alpha` and `beta`, launched in parallel in the background;
- turn 2 (`--resume` of the same session): `gamma` in the foreground, which
  itself spawned `delta`.

Edges come from the `agentId: …` line in each Agent tool result, taken from
whichever transcript made the call. The parent's call is in the parent
transcript and gamma's call is in gamma's transcript.

**Claude Code 2.1.272**, a second session recorded 2026-09-15 under
`-fixture-spawn-claude-kill`: one foreground agent, `epsilon`, whose whole job
was a 300-second Python sleep in the foreground. SIGINT went to the process
group four seconds after the child's Bash `tool_use` reached its transcript;
the run's own result reported `subagent_stats.killed.user: 1`. Its edge is
recorded under `claude.interrupted`, and it comes from the parent's Agent
`tool_use` block rather than a tool result, because a killed child never gets
one (below).

**Claude Code 2.1.272**, a third session recorded 2026-09-15 under
`-fixture-spawn-claude-wf`: one Workflow-tool run, `probe`, whose script made a
single `agent()` call labelled `zeta` in phase `Probe`. The Workflow tool needs
an explicit opt-in, so the headless prompt opened with `ultracode: use a
workflow` and `Workflow` was in both `--tools` and `--allowedTools`; Haiku 4.5
answered without calling anything, twice, and Sonnet 5 complied. Recorded under
`claude.workflow`. The edge here runs through the **run id**, not a tool
result: the parent's Workflow result names `runId`, the transcript directory is
named for it, and the journal in that directory maps the agent id to its label.

**OpenCode 1.18.30** (`opencode run --pure`, `opencode-go/deepseek-v4-flash`,
with the built-in `general` agent enabled by a one-off `OPENCODE_CONFIG`
overlay):

- a parent that spawned `alpha` and `beta` in one message;
- a `--fork` of that parent;
- a parent whose child `beta` tried to spawn and hit OpenCode's default
  `subagent_depth` of 1 (a failed spawn, recorded under `failed_spawns`);
- with `"subagent_depth": 2`, a parent → `beta` → `gamma` chain.

Edges come from each `task` tool part's `state.metadata.sessionId`.

**Codex CLI 0.154.0** (`codex exec`, `agents.max_depth=2`, low reasoning
effort), in a scratch directory:

- a root session whose turn 1 spawned `alpha` and `beta` together;
- turn 2 (`codex exec resume`, same thread id): `gamma`, which itself spawned
  `delta`;
- `codex exec fork` of the root, and a second fork of `gamma`, the subagent;
- a second root that spawned `epsilon` into `sleep 300` and was then
  interrupted with SIGINT while epsilon was still running.
- recorded 2026-09-15 with the same build: `codex review --uncommitted` in a
  scratch repository, which runs as an `exec` parent plus a second thread for
  the review itself. Recorded under `codex.review`; its spawning-side evidence
  is the parent rollout carrying the child's `item_completed` events between
  `EnteredReviewMode` and `ExitedReviewMode`.

Edges come from each `SubAgentActivity` event of kind `started`, in the rollout
that made the `spawn_agent` call. Its `id` is that call's `call_id` and
`agent_thread_id` is the child.

## Facts this fixture pins

It checks these in both states and never runs session-browser for them. They
describe what the providers write, so a later implementation can build on
them. A provider upgrade that breaks one then fails here, where the cause is
visible.

Claude Code:

- A grandchild is stored **flat** in the same `<parent>/subagents/` directory
  as its parent agent. The nesting is recorded only in the sidecar
  (`parentAgentId`, `spawnDepth: 2`), never in the directory layout.
- Each sidecar's `toolUseId` names the exact spawning tool call, whether that
  call is in the parent session or inside another agent.
- **`sessionId` on a subagent's records is the parent's id**, not the child's.
  This is the same trap as Codex's `payload.session_id`: an id taken from it
  files every child under its parent's identity.
- **`promptId` names the root turn in the parent, not the direct spawner.**
  alpha and beta share turn 1's id. gamma and its own child delta share turn
  2's id. So `promptId` groups a fan-out but cannot say which agent spawned
  which; that answer is in `parentAgentId` and `toolUseId`.
- **A killed child is never announced to its parent.** The parent's
  `tool_result` for the Agent call is the plain denial — `toolUseResult` is the
  string `"User rejected tool use"`, `toolDenialKind` is `"user-rejected"` —
  with no `agentId` anywhere, so a linker that waits for the spawn-side id
  never sees this child. The sidecar still carries the parent's `toolUseId`,
  and that is the only edge there is.
- **The kill is legible only in the child.** Its single `tool_use` is answered
  by the same denial record, and the transcript ends on a user record reading
  `[Request interrupted by user for tool use]`, never on a result. The parent
  ends on the same marker one millisecond later, so the two endings are
  simultaneous rather than causal in either direction.
- **A workflow agent is one directory deeper**, at
  `<parent>/subagents/workflows/<runId>/agent-<id>.jsonl`, with nothing stored
  flat in `subagents/` — so a one-level scan of `subagents/` finds no trace of
  the run. Its sidecar says `agentType: "workflow-subagent"`, carries
  `workflowPhase`, and has **no `toolUseId`**. Beside it, `journal.jsonl`
  records `started` and `result` events with `agentId` and `label`, and the
  parent's `workflows/<runId>.json` run record lists every agent with its id,
  model, tokens and state. The launch answer in the parent
  (`toolUseResult.status: "async_launched"`) names the run and its transcript
  directory but no agent, because none has started yet.
- **Completion is a new turn.** The run's result reaches the parent as a
  `task-notification` user record (`origin.kind`) with its own `promptId`, so
  the spawn turn and the completion turn are different prompt ids — the same
  shape as a background Agent finishing.


OpenCode:

- A fork records **no** `parent_id`, so it is not a subagent.
- A fork **copies its source's `task` parts**, and those copies still name the
  original spawner in `parentSessionId`. Rebuilding edges from task parts
  instead of `session.parent_id` would credit the fork with its source's
  children.
- A depth-2 child's `parent_id` is its direct spawner, not the root.

Codex:

- **Nesting is recursive.** `source.subagent.thread_spawn.parent_thread_id` is
  the direct spawner, `depth` counts from the root, and `agent_path` spells out
  the whole chain (`/root/gamma/delta`), not only a leaf name.
- **Every subagent is also a fork of its spawner**: its `session_meta` carries
  `forked_from_id` equal to `parent_thread_id`. So `forked_from_id` does not
  tell a fork from a spawn; `source` and `thread_source` do. A real fork has
  `source: "exec"`, `thread_source: "user"`, and no `parent_thread_id` at any
  level, even when it is a fork of a subagent.
- A subagent also carries `parent_thread_id` at the **top level** of
  `session_meta`, the shape agent-sessions guards against. On this build it
  appears only on subagents, never on forks.
- **`session_id` names the root of the tree**, not the direct parent. At depth 1
  those are the same; at depth 2 they differ.
- **A subagent's rollout starts with its spawner's history.** The spawner's
  `session_meta` is the second record, followed by its prompts, up to
  `subagent_history_start_ordinal`. Only the first `session_meta` describes the
  file.
- **A `review` child has a parent, written where the fork guard cannot see
  it.** Its `source` is the bare string `"review"`, so there is no
  `thread_spawn.parent_thread_id`; the parent is the **top-level**
  `parent_thread_id` (and `session_id`) of its `session_meta`, and every
  review rollout on the recording machine since June 2026 carries it. It has
  no `thread_spawn_edges` row, and the `threads` index has no parent column
  at all, so the index path cannot recover this parent from the row it reads.
  A review child also copies **none** of its spawner's history, unlike a
  `thread_spawn` child.
- A fork copies **no** spawn records. It points at its source through
  `history_base` instead of repeating it, unlike an OpenCode fork.
- A spawn's `root_turn_id` groups like Claude's `promptId`: alpha and beta
  share turn 1, and gamma and its own child delta share turn 2.
- An interrupted child is visible only as a `started` event with no matching
  `completed` one, plus `turn_aborted` with reason `interrupted` in both
  rollouts. `thread_spawn_edges.status` does not record it: every edge here is
  still `open`, finished or killed.

## States

`baseline` (accepted): the OpenCode graph as `session-browser list` reports it
matches the ground truth exactly, fork included. So does the Codex graph,
through both discovery paths: the `state_5.sqlite` index and the rollout file
scan. The index run must write nothing to stderr, because discovery logs there
when it falls back to the file scan, and a silent fallback would otherwise test
the same path twice. Claude subagents are invisible
to discovery, so only the three parent sessions are listed, each with
`parent_id: null`.

`candidate`: the Claude children are listed too, the killed `epsilon` and the
workflow's `zeta` included, each with `parent_id` set to its direct spawner. This state is deliberately red until the Claude half of
subagent linking exists. The check matches children by agent-id suffix
because the id a Claude subagent should be addressed by is not decided yet,
and this fixture should not decide it.

```bash
.venv/bin/python docs/fixtures/subagent-ground-truth/verify_case.py baseline
.venv/bin/python docs/fixtures/subagent-ground-truth/verify_case.py candidate
```

The verifier copies `home/` to a temporary directory and builds the OpenCode
database there from `opencode_sessions.json`, and Codex's `state_5.sqlite` from
`codex_threads.json`. The rows stay reviewable in a
diff, and nothing is written into the repository or read from the machine's
real history.

## How the files were reduced

The Claude transcripts keep only `user` and `assistant` records, the
linking-relevant keys, and text, `tool_use` and `tool_result` blocks.
Attachments, hook output, queue operations and thinking blocks are dropped, so
`parentUuid` chains skip over the removed records. Working directories are
rewritten to `/fixture/...`. Sidecars are verbatim. OpenCode rows keep only the
columns discovery reads.

The Codex rollouts keep `session_meta` (without instructions, tools and context
window), `turn_context` reduced to its turn ids, task start, completion and
abort events, `SubAgentActivity` and `UserMessage` events, user and assistant
messages, inter-agent messages, and `spawn_agent`/`wait_agent` calls with their
outputs. Injected developer and user context is dropped, and so is the
`message` argument of `spawn_agent`, which Codex stores encrypted. The index rows
are the ones Codex wrote, with `rollout_path` made relative to `~/.codex`.

## Not covered yet

- **An interrupted OpenCode subagent.** Codex and Claude Code each have one.
- **Codex `guardian` subagents.** Only the incidental corpus has them, and no
  headless trigger is known.
