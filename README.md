# session-browser

**Turn your agent transcripts into working memory.**

Claude Code, Codex, OpenCode and Pi all keep session histories. Session
Browser brings them together so you — or your agent — can find lost context,
build a handoff, spot recurring knowledge or repeating pain points, and easily
resume the work instead of starting again.

![Session Browser showing sessions beside a selected transcript](docs/screenshots/01-list-160x45.svg)

Every session adds decisions, investigations, fixes, and working patterns to an
evolving knowledge base. That knowledge is scattered across provider-specific
folders and databases. Session Browser makes it one browsable, searchable
memory: a terminal-native TUI for you and a rich retrieval surface for your
agents.

## Put old sessions back to work

- **Recover lost context.** Search complete Claude Code, Codex, OpenCode and
  Pi transcripts together from the TUI, or let your agent find the useful parts
  through the CLI.
- **Great for handoffs**. Pulls the relevant context from
  previous sessions so the next agent can pick up where you left off.
- **Let agents research your history.** The JSON-friendly CLI and bundled skill
  let an agent find and retrieve earlier work itself, without you pasting old
  conversations into a new chat.
- **Find the patterns worth keeping.** Mine repeated fixes, decisions, and
  workflows across sessions, then turn the useful ones into reusable skills,
  prompts, or playbooks.
- **Resume instead of restarting.** Continue the original session directly in
  tmux or Herdr, or copy the agent specific resume command to the clipboard, all from the TUI.
- **Stay in the terminal.** Browse, search, export, and resume sessions where
  your coding agents already run — no browser tab or separate desktop app
  required.
- **Keep your history local.** Session Browser reads the providers' native
  histories directly.
- **Light by design.** There is no second copy of your history, indexing pass,
  daemon, embedding model, or separate database to set up and maintain. Install
  it, run it, and search the histories you already have.

## Ask your agent

With the bundled skill installed, requests like these can use your existing
session history as evidence:

> Find what we were working on in this repo at the end of last week. Give me a
> quick summary so we can pick it up.

> Look back over the last two weeks. Have we hit the same problems more than
> once? Suggest tooling or skills that would stop us reinventing the wheel.

> I lost the thread this afternoon. Review today's sessions and make me an
> explainer of what changed and why.

## Install and run

### With uv (recommended)

[uv](https://docs.astral.sh/uv/) is available for macOS, Linux, and Windows.
It installs Session Browser in an isolated environment, puts the command on
your path, and supplies a compatible Python version if you need one.

```bash
uv tool install git+https://github.com/steveoleary/session-browser.git
session-browser
```

Upgrade later with:

```bash
uv tool upgrade session-browser
```

### With pipx

If you already use [pipx](https://pipx.pypa.io/), install and upgrade directly
from the same Git repository:

```bash
pipx install git+https://github.com/steveoleary/session-browser.git
pipx upgrade session-browser
```

### From a clone

```bash
git clone https://github.com/steveoleary/session-browser.git
cd session-browser
uv tool install --editable .
session-browser
```

The editable install always runs the code in your checkout.

## See it in action

### Search across every session

Press `/` and search once across all three providers. The session list narrows
to the conversations that contain your phrase, while the transcript shows the
match in context. When you find the right one, press `t` to continue it in tmux
or Herdr, or `c` to copy its resume command — search, read, and resume without
leaving the UI.

![Global transcript search for sqlite](docs/screenshots/02-search-sqlite-120x40.svg)

### Read the part that matters

Press `z` to give the transcript the full terminal, then use `n` and `N` to move
between matches inside the conversation.

![Focused transcript with an in-session search for WAL](docs/screenshots/03-detail-focus-wal-120x40.svg)

The layout adapts to the available space: a wide two-pane browser on a large
terminal, a balanced view at medium sizes, and a full-width sessions/transcript
flow when space is tight.

## Command line

Running `session-browser` with no arguments opens the TUI. The same history is
available through four commands for agents, scripts, and targeted retrieval:

```bash
# Browse recent sessions from this project
session-browser list --here --limit 20

# Search complete transcripts and return useful context
session-browser search --here --mode snippets "pytest fixture"

# Retrieve a session by canonical id, raw id, or unique prefix
session-browser get claude:7645 --output handoff.md

# Summarise providers, activity, and working directories
session-browser stats --here
```

Use `session-browser COMMAND --help` for filters, output formats, date ranges,
and retrieval windows. Commands return JSON by default where it is useful for
automation.

## Skill for agents

The bundled `using-session-browser` skill lets Claude Code, Codex, OpenCode and
Pi recover earlier work themselves, assemble better handoffs, and mine recurring
knowledge from your session history.

Install it with the [skills CLI](https://skills.sh):

```bash
npx skills add steveoleary/session-browser@using-session-browser
```

Its source lives in
[`skills/using-session-browser/`](skills/using-session-browser/SKILL.md).

## Essential keys

Press `?` in the app for the complete key map.

| Key | Action |
|-----|--------|
| `/` | Search all sessions |
| `s` | Find within the selected session |
| `Tab`, `←` / `→` | Switch between sessions and transcript |
| `Enter`, `Esc` | Open a transcript / step back |
| `z` | Focus the active pane |
| `n` / `N` | Next / previous match |
| `c` | Copy a resume command |
| `i` | Copy the canonical session id |
| `e` / `E` | Copy / export the conversation |
| `p` | Toggle this-project scope |
| `?` | Show all shortcuts |
| `q` | Quit |

## Development

```bash
uv sync
uv run pytest session_browser/ -q
uv run ruff check .
uv run ruff format --check .
```

The committed regression fixtures in `docs/fixtures/` can be checked with:

```bash
uv run python -m session_browser.case_runner run
```

## License

MIT — see [LICENSE](LICENSE).
