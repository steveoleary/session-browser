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
for OpenCode), so a linking bug has no way to agree with itself here.

## What was spawned

**Claude Code 2.1.270** (`claude -p`, Haiku 4.5), one session, two turns:

- turn 1: `alpha` and `beta`, launched in parallel in the background;
- turn 2 (`--resume` of the same session): `gamma` in the foreground, which
  itself spawned `delta`.

Edges come from the `agentId: …` line in each Agent tool result, taken from
whichever transcript made the call. The parent's call is in the parent
transcript and gamma's call is in gamma's transcript.

**OpenCode 1.18.30** (`opencode run --pure`, `opencode-go/deepseek-v4-flash`,
with the built-in `general` agent enabled by a one-off `OPENCODE_CONFIG`
overlay):

- a parent that spawned `alpha` and `beta` in one message;
- a `--fork` of that parent;
- a parent whose child `beta` tried to spawn and hit OpenCode's default
  `subagent_depth` of 1 (a failed spawn, recorded under `failed_spawns`);
- with `"subagent_depth": 2`, a parent → `beta` → `gamma` chain.

Edges come from each `task` tool part's `state.metadata.sessionId`.

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

OpenCode:

- A fork records **no** `parent_id`, so it is not a subagent.
- A fork **copies its source's `task` parts**, and those copies still name the
  original spawner in `parentSessionId`. Rebuilding edges from task parts
  instead of `session.parent_id` would credit the fork with its source's
  children.
- A depth-2 child's `parent_id` is its direct spawner, not the root.

## States

`baseline` (accepted): the OpenCode graph as `session-browser list` reports it
matches the ground truth exactly, fork included. Claude subagents are invisible
to discovery, so only the parent session is listed, with `parent_id: null`.

`candidate`: the Claude children are listed too, each with `parent_id` set to
its direct spawner. This state is deliberately red until the Claude half of
subagent linking exists. The check matches children by agent-id suffix
because the id a Claude subagent should be addressed by is not decided yet,
and this fixture should not decide it.

```bash
.venv/bin/python docs/fixtures/subagent-ground-truth/verify_case.py baseline
.venv/bin/python docs/fixtures/subagent-ground-truth/verify_case.py candidate
```

The verifier copies `home/` to a temporary directory and builds the OpenCode
database there from `opencode_sessions.json`. The rows stay reviewable in a
diff, and nothing is written into the repository or read from the machine's
real history.

## How the files were reduced

The Claude transcripts keep only `user` and `assistant` records, the
linking-relevant keys, and text, `tool_use` and `tool_result` blocks.
Attachments, hook output, queue operations and thinking blocks are dropped, so
`parentUuid` chains skip over the removed records. Working directories are
rewritten to `/fixture/...`. Sidecars are verbatim. OpenCode rows keep only the
columns discovery reads.

## Not covered yet

- **Codex.** The spawn run hit an account usage limit before it produced a
  subagent, so its nesting depth and fork-versus-spawn distinction are still
  unverified against a deliberate run.
- **Claude workflow subagents** (`subagents/workflows/wf_*/`) were not
  produced.
- **An interrupted subagent**, one killed mid-run, as opposed to a failed
  spawn.
