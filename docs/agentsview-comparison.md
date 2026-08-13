# Session Browser and AgentsView: a local retrieval comparison

> **Disclosure:** This comparison was conducted for, and is published in, the
> Session Browser repository. To reduce hindsight and product-identity effects,
> an independent Codex session assessed an evidence packet in which the products
> were identified only as Tool X and Tool Y. The assessment was completed before
> the mapping—Tool X as Session Browser and Tool Y as AgentsView—was revealed.
> This procedure does not eliminate all potential bias: the cases, protocol,
> evidence packet, and publication context still originated with the Session
> Browser project.

## Scope and conclusion

This was a small, local, descriptive experiment conducted on 2026-07-29. It
compared how two tools helped an AI coding agent reconstruct evidence from prior
agent-session transcripts. The tasks emphasized long, self-correcting technical
histories in which an adequate report needed to recover diagnoses, reversals,
fixes, verification, and remaining uncertainty.

The results support a narrow conclusion. In the strict interactive/vector
condition, Session Browser produced the stronger report in two of three selected
trials, while AgentsView produced the stronger report in one. AgentsView used
fewer retrieval commands and less elapsed time in aggregate, but its largest
time advantage occurred in the trial where its report stopped before material
later evidence. The experiment therefore did not establish that AgentsView was
faster at producing reports of comparable completeness.

The tools also had different setup profiles in this experiment. Session Browser
read provider-native histories at query time without a daemon, synchronization
step, embedding model, or persistent search index. AgentsView imported the
corpus and, for the strict rerun, used a vector path backed by a local embedding
service.

These observations should not be combined into a composite score or a universal
winner. Three manually selected cases, one run per condition, one person's
archive, one machine, and model-based report judging are adequate for a
descriptive comparison, not a general product ranking.

## Experiment metadata and controls

- Experiment date: 2026-07-29
- Session Browser commit:
  [`42d88d0`](https://github.com/steveoleary/session-browser/commit/42d88d0cc9b87315cabb9a87d8ba4d839fd593f2)
- AgentsView commit:
  [`dc0a980`](https://github.com/kenn-io/agentsview/commit/dc0a980106bd7fd190ec46e9762888c124ff1d0a)
- AgentsView archive snapshot: 1,795 sessions and 42,944 messages
- Agent model: `gpt-5.6-sol` at medium reasoning effort
- Reports: no more than 350 words, with exact entry/message citations,
  reversals, final-state verification, and residual uncertainty
- Evidence boundary: original recorded sessions before 2026-07-23

Each trial used a fresh interactive Codex TUI session in a separate Herdr pane.
Agents could use only their assigned retrieval CLI. They were prohibited from
using memory, repository files, Git history, web search, the competing tool, or
prior knowledge. AgentsView's archive daemon was stopped after synchronization
and embedding so that its archive snapshot could not change during the trials.
Session Browser read provider-native histories directly; the date boundary kept
the new trial sessions outside the eligible evidence.

Elapsed time ran from the candidate's first task-message timestamp through its
final `COMPLETE` response, including retrieval, reasoning, and report writing.
A retrieval command was a tool call invoking the candidate's assigned
retrieval CLI; unrelated shell operations were not counted.

A separate fresh Codex session judged anonymized candidate reports, with
candidate order varied between trials. The judge saw elapsed time and command
counts for friction scoring. The findings scores below are therefore judge
assessments, not direct measurements of source recall, precision, factual
accuracy, or human usefulness.

## Strict interactive/vector results

AgentsView used its current vector path with a local 768-dimensional
`nomic-embed-text` model.

| Trial | Session Browser findings | AgentsView findings | Session Browser retrieval | AgentsView retrieval |
|---|---:|---:|---:|---:|
| Sticky F7 diff-view investigation | 9.4/10 | 9.6/10 | 59.826 s, 6 commands | 79.746 s, 8 commands |
| AgentAlert reset/countdown investigation | 9.3/10 | 7.6/10 | 123.249 s, 12 commands | 73.914 s, 8 commands |
| Visual Review blank-launch investigation | 9.7/10 | 9.4/10 | 101.190 s, 10 commands | 99.313 s, 7 commands |
| **Total** | — | — | **284.265 s, 28 commands** | **252.973 s, 23 commands** |

The findings judge narrowly preferred AgentsView for Sticky F7, materially
preferred Session Browser for AgentAlert, and narrowly preferred Session
Browser for Visual Review. The friction judge preferred Session Browser for
Sticky F7 and AgentsView for the other two trials.

### Findings quality

In Sticky F7, both reports recovered the false scroll-state hypothesis and its
falsification, the raw Git-hunk/source-line cause, the durable rendering change,
automated and live verification, and a residual editor-line uncertainty.
AgentsView's 0.2-point preference was attributed to clarity.

AgentAlert contained the only material findings gap between the strict-run
reports. AgentsView recovered the earlier architectural redesign and an
observer-effect reversal, but stopped before later evidence. Its report omitted
the installed application's incorrect countdown, the GUI
executable/Node/`PATH` failure, an incomplete first fix, reconstruction
precedence changes, corrected installed-state verification, a remaining
provider-specific timing uncertainty, and the later display redesign.

Session Browser recovered that later failure-and-repair history. It omitted
some earlier architectural rationale and two implementation-test defects, which
the judge considered less consequential. This supports the conclusion that
Session Browser produced the more complete final-state reconstruction in this
trial. It does not establish uniformly higher recall: raw retrieval results were
not exhaustively scored, and the condition was run only once.

In Visual Review, both reports recovered the CSP false lead and reversal, stale
Vite and Electron cache layers, the durable dependency-optimization exclusion,
two normal persistent-profile launches, tests, and bounded uncertainty. The
0.3-point preference for Session Browser was attributed to its causal chain and
exact final verification.

The two sub-point preferences should be described as narrow report-level
judgments, not meaningful product-level gaps. The content difference in
AgentAlert is more consequential than the 2–1 trial count alone suggests.

### Query-time friction

Across the three strict trials, AgentsView used 31.292 fewer seconds, or 11.0%
less elapsed time, and five fewer retrieval commands, or 17.9% fewer. These are
observed measurements for these runs. Command count remains only a proxy for
effort because commands can differ in scope, output, and cognitive cost.

Elapsed time was not independent of completeness. AgentsView's 49.335-second
advantage in AgentAlert was larger than its 31.292-second aggregate advantage,
and it coincided with the material omissions described above. On the other two
trials, where report content was much closer, Session Browser took 161.016
seconds and AgentsView took 179.059 seconds. Session Browser was therefore
18.043 seconds faster across those two trials.

That derived two-trial comparison does not establish that Session Browser is
intrinsically faster. It does show why the three-trial aggregate cannot be
presented as speed at equivalent quality. The appropriate conclusion is that
AgentsView had lower measured aggregate time and command count in this strict
condition, while the experiment did not isolate a stable, like-for-like
query-time advantage.

## Setup and operational overhead

Session Browser required no separate preparation for these trials. AgentsView
synchronized the frozen corpus in 18.7 seconds and built embeddings in 818.79
seconds. The completed generation contained 7,377 documents and 8,959 chunks,
with zero documents reported missing or stale.

The combined 837.49-second initial preparation cost is a clear operational
difference for this configuration. Dividing it by the observed mean
10.431-second per-task saving gives roughly 80 similar retrieval tasks. This is
illustrative arithmetic, not a break-even estimate. The denominator is
sample-specific and confounded by the incomplete AgentAlert report. The
calculation also does not model refresh costs, corpus growth, caching, service
administration, failure recovery, or the possible value of an already-running
shared archive.

The isolated AgentsView experiment directory occupied 554 MB, but that
directory included the cloned source, compiled artifacts, archive, and vector
index. It was not an index-size measurement and should not be interpreted as
AgentsView's storage overhead in general.

Conversely, Session Browser's lower preparation burden does not establish a
scaling advantage. Its query-time behavior on substantially larger archives was
not measured. AgentsView's semantic retrieval may be useful for fuzzy discovery
or an always-running shared service, but those possible benefits were not
measured here.

## Supporting headless/FTS rerun

A separate current-version rerun used headless agents and AgentsView's FTS path
rather than the interactive/vector setup:

| Trial | Session Browser | AgentsView | Findings preference | Friction preference |
|---|---:|---:|---|---|
| Sticky F7 | 91 s, 8 commands | 100 s, 12 commands | Session Browser | Session Browser |
| AgentAlert | 127 s, 14 commands | 92 s, 10 commands | Session Browser | AgentsView |
| Visual Review | 69 s, 7 commands | 85 s, 10 commands | AgentsView | Session Browser |

This run also produced a 2–1 findings preference for Session Browser, but its
friction result was 2–1 for Session Browser rather than AgentsView. Because both
the execution surface and AgentsView's retrieval mode changed, the experiment
cannot attribute the friction difference to either factor independently. These
results should therefore remain separate from the strict table rather than
being pooled.

An older three-case local comparison preferred Session Browser's findings in
all three trials and AgentsView's friction in one. AgentsView's semantic search
was unavailable in that run because embeddings were not configured, so it used
an FTS fallback. That earlier comparison is contextual evidence only, not a
replication of the strict condition.

## Limitations

- The three cases were manually selected from one person's local archive.
- There was one principal model family, one machine, and one run per condition.
- The tasks focused on long, self-correcting technical histories; other query
  types and usage patterns may produce different tradeoffs.
- A single model session judged candidate reports. No human panel,
  inter-rater assessment, or repeated-trial variance estimate was available.
- Candidate reports, rather than every retrieved result and raw source entry,
  were scored. The experiment therefore did not directly measure recall or
  precision.
- The findings rubric, timing boundary, and command-counting rule would need to
  be published in full for stronger external reproducibility.
- Product architecture, retrieval mode, interaction surface, archive size, and
  agent instructions can all affect outcomes.
- The “zero missing or stale” status for AgentsView's generation is reported in
  the experiment record; the packet does not specify an independent
  source-by-source audit.
- Stopping AgentsView's daemon established its frozen archive state. Session
  Browser used a historical date cutoff rather than a snapshot; the experiment
  did not independently hash the eligible native-history files before and
  after each trial.

## Replication improvements

A stronger follow-up would select tasks prospectively or randomly, run each
condition multiple times, include additional archive sizes and query types, and
keep findings scoring blind to timing. It would publish exact prompts, scoring
rubrics, timing and command definitions, and product configuration. Multiple
human or independently calibrated judges should assess the reports, while a
separate evidence ledger audits recovered and missed source facts directly.

Retrieval mode and interaction surface should be varied one at a time. Setup,
refresh, storage, service operation, query time, and report quality should
remain separate measures. That design would make it possible to test specific
claims about completeness, steady-state friction, semantic discovery, and
scaling without turning unlike tradeoffs into a single ranking.
