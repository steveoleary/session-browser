# Fresh-agent research brief

Hand this, unedited, to an agent that has not seen this repository. It is the
half of the fresh-agent test that a machine cannot score: whether the skill and
the `--help` text are enough for someone arriving cold. Read the friction
report; do not grade the answers against a threshold.

`verify_case.py` already checks that the documented commands reach the right
answers against this corpus, so a wrong answer here is a *documentation*
result, not a tooling one.

## Setup, and the observer effect

Run the agent with `HOME` pointed at this fixture's frozen corpus, and from a
scratch working directory it will not confuse with a real project:

```bash
cd "$(mktemp -d /tmp/skill-brief.XXXXXX)"
HOME=<repo>/docs/fixtures/fresh-agent-skill-brief/home <your agent>
```

Two reasons, both learned the hard way. A live corpus makes the answers move —
a council agent's own session was the top hit for a phrase twelve minutes after
it started, and the same search returned 5 results for one reviewer and 7 for
another twenty minutes later. And the agent writes its own transcript while it
works, into whatever corpus it can see; a distinctive scratch cwd keeps that
noise removable with `--exclude-cwd`.

Do not reuse a task set the same agent has already answered. If you want a
second run, either re-derive the tasks or ask about runs 1..N-1.

## The brief

> You have a tool called `session-browser` and a skill describing how to use
> it. Answer these four questions about the session history it can see. For
> each one, say what you did to reach the answer and how confident you are.
>
> 1. How many sessions is work recorded in under the project `coffee_run`
>    itself — not under any other project? How many of those can actually be
>    read?
> 2. `session-browser stats` reports a total. How many of those sessions can
>    be read, and how do you know?
> 3. Did anyone actually propose sunsetting the loyalty tier, or does that
>    phrase only appear in material somebody read? Name where it was proposed,
>    if it was.
> 4. Which espresso machine is being ordered?
>
> Then write a friction report with a section for each of these, saying
> nothing where there is nothing to say:
>
> - **Wrong turns.** Anything you tried that gave a confident wrong answer.
> - **Undocumented.** Anything you had to discover by experiment because no
>   `--help`, skill section or error message said it.
> - **Contradictions.** Anywhere two sources disagreed.
> - **Missing.** A question you could not answer at all, and what would have
>   let you.

## Ground truth

In `expected.json`. Four questions, four traps, one per trap the skill warns
about: `--repo` is a substring filter, `stats` counts what it discovered rather
than what is readable, a hit is as often a quotation as a statement, and
sessions self-correct. Questions 3 and 4 are answerable only through material
under test — nothing in the corpus states either answer plainly.

## What to do with the result

A wrong answer is a bug in the text, not in the agent. Fix the paragraph that
allowed it, in `skills/using-session-browser/SKILL.md`, the argparse help, or
the module docstring, and note which. That is the point of the run: these three
are what drift, and the two ad-hoc runs that prompted this fixture found
defects in text written the same day.
