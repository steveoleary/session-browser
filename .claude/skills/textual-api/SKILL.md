---
name: textual-api
description: >-
  Resolve a Textual or Rich framework question for session-browser — widget,
  reactive, worker, or CSS behaviour that this repo does not already
  demonstrate. SKIP ENTIRELY, do not load, when any of these hold: app.py or
  tests.py already uses the widget, property, or method in question (read those
  instead — they are working, version-correct, and free); the question is about
  the standard library, pytest, or session-browser's own code; you are guessing
  proactively rather than actually blocked. Fires only when this repo has no
  working example and being wrong would cost a rebuild.
---

# Textual API questions in session-browser

The default answer to "how does this Textual thing work" in this repo is **not**
a docs lookup. It is `session_browser/app.py` — around 2,300 lines of Textual
that is already running, already correct for the installed version, and already
tested. Reading it costs nothing and cannot be stale.

Reach outside the repo only when the repo genuinely has no answer.

## 1. Ask the repo first

```bash
grep -n "<WidgetOrProperty>" session_browser/app.py session_browser/tests.py
```

Both files matter and they answer different questions. `app.py` shows the
working usage; `tests.py` shows what behaviour is actually pinned — including
geometry assertions, which are the ones most likely to surprise you.

The CSS lives inline near the top of `app.py` as a single stylesheet string, so
grep it for selectors and properties too. It is the fastest way to see which
layout primitives this app already relies on.

## 2. Know the version you are on

The package pins `textual>=0.80.0`; the installed version is what matters:

```bash
.venv/bin/python -c "import textual; print(textual.__version__)"
```

That floor is very low relative to what is installed, so a wide band of API
drift sits inside the allowed range. Anything you recall about Textual may
predate the running version. The repo does not — it runs against exactly what
is installed. Another reason step 1 beats memory.

## 3. Prototype rather than guess

For anything about *how it looks or feels* rather than what an API is called,
this repo's habit is to build a throwaway with measured geometry, not to reason
about it. `prototypes/throwaway_search_feel.py` is the worked example: it
records the real measurements taken from the app —

```
terminal   left pane   text area inside the search box
  100 x30      41              33
  144 x40      51              43
  180 x45      65              57
```

— and lets you cycle placements interactively. Copy that shape. A prototype
answers layout questions that no documentation can, because the constraint is
this app's actual column budget.

## 4. Only now, look outside

If steps 1–3 leave a real gap — an API this repo has never used, where being
wrong means building the wrong thing — then consult external documentation.

**Context7 is not currently configured for Claude Code in this environment.**
Its global instruction files were deliberately deleted (`~/.codex/AGENTS.md`,
`~/.claude/rules/context7.md`) because a broad always-on rule fired constantly
for trivial lookups and burned API quota. Do not recreate a global rule. If
Context7 is available to you as an MCP tool or CLI, use it here and only here;
if it is not, say so and fall back to WebFetch against the Textual docs for the
specific API, or ask.

Either way this is the *last* step, reached only because the repo had no
answer — not the reflex.

## 5. Close the loop

When an external lookup produces a real answer, leave it in the code as a brief
comment at the point of use, in the style of the surrounding comments — which
explain *why*, not *what*. That converts a one-off lookup into a repo fact, so
the next agent resolves it at step 1 and never reaches step 4 at all.

This step is what keeps the skill from being needed twice for the same
question.
