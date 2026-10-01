# Chat TTFT repair — 2026-09-17

The native web-chat follow-up showing 14.7 seconds reprocessed all 18,776
tokens: 14.44 seconds were prefill, zero tokens reused. This was a prompt
prefix failure, not a required cost of the earlier decode improvement.

Three defects combined:

1. Swift JSON object order changed between requests. Native templates
   rendered the same tool schema with different leading token sequences.
2. `tool_choice: none` removed the schemas and changed cache identity when
   closing a search round; the next user turn put them back.
3. Postcommit treated the server's temporary closing instruction as client
   history. The next request omitted it and lost reuse of the whole answer.

The shared server now canonicalizes tool-schema object keys, distinguishes
prompt declarations from permission to return calls, and captures the
client-visible history before appending request-only instructions. Lists,
schema values, tool permissions, reasoning, output limits, sampler math,
model weights and the adaptive decode fix are preserved.

Code commits: `9ebeb267` (schema/permission separation) and `329d8211`
(correct postcommit history and prefill boundaries).

## Evidence

The schema repair was tested with an ABBA against `df937f29`, separate
source paths and isolated banks, one resident model, identical launch
settings/native sampling, verified max fans and <=60 C starts. The saved
~19k chat replay used a diagnostic-only 512-token budget and seed 1731.

| Mean of two runs | Baseline | Schema repair |
|---|---:|---:|
| Reordered-schema follow-up TTFT | 15.260 s | 0.569 s |
| Tool-closure TTFT | 13.934 s | 0.661 s |
| Cold prefill | 1,249 tok/s | 1,259 tok/s |
| Follow-up decode | 58.54 tok/s | 69.94 tok/s |
| Tool-closure decode | 59.17 tok/s | 59.26 tok/s |
| Sequence peak memory, decimal GB | 93.78 | 89.73 |

Actual uncapped app QA then exposed the third defect: a 2.815-second
follow-up reprocessed the searched answer. After the postcommit correction,
the repeated native workflow used 11,421 cached tokens out of 11,461 and
returned its first token in **0.291 seconds**, decoding at **69.17 tok/s**.
Its Python example executed correctly. The original conversation's next
follow-up measured **0.277 seconds / 62.27 tok/s** at 12,536 context.

The original conversation had crossed the app's existing 64,000-character
history-compaction threshold before that continuation. Its first upgraded
turn therefore reprocessed 10,133 tokens in 8.740 seconds. That existing
Swift compaction was unchanged; the smaller native continuation is not
presented as an identical-context 19k comparison. Fresh web results and
different client system prompts still require real prefill. Exact background
postcommit also has a cost: 2.226 seconds after the final searched answer.

All seven new regressions fail on their respective baselines. Final suite:
**480 passed**, no skips/failures. Real OpenCode CLI, Pi and Hermes each
fixed mean, added median and passed independent artifact tests. Their cold
and partially changed-input requests remain in the evidence; no universal
subsecond TTFT or 50 tok/s floor is claimed.

## Installed state

Signed build **2011048**, version 2.11.3, contains both rebuilt runtime
wheels. Actual app and global CLI imports match source. The existing Swift
binary and native kernel code are retained; launchers stay at their existing
paths. Dependency and signature checks pass. Runtime backups are preserved.

Detailed local evidence, complete raw receipts, negative attempts and QA:
`/Users/youssof/Projects/MTPLX/outputs/chat-ttft-20260916/REPORT.md`.
Nothing pushed, tagged or published.
