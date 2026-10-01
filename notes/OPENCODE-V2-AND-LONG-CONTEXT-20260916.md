# OpenCode V2 compatibility and long-context investigation

This records the initial investigation. The subsequent measured performance
fix is documented in [the launch/adaptive fix](DECODE-LAUNCH-AND-ADAPTIVE-FIX-20260917.md).
That later change adds the engine scheduling and launcher repairs absent here.

## Release status

The OpenCode V2 session-header plugin is fixed. A separate intermittent
long-context decode slowdown remains unresolved; no decoder optimization is
claimed by this change. The existing foreground-aware SSD eviction repair is
retained without modification.

Issue [#498](https://github.com/youssofal/MTPLX/issues/498) concerns V2's plugin
API, rather than its chat-completions schema. The managed plugin now has a V1
function entrypoint and a server entrypoint supporting both modern V1 and V2.
V2 registers a provider-filtered `model.request` hook to attach session identity.
It does not replace prompts, impose output limits, or rewrite sampling options.
Configuration upgrades the managed registration while retaining other plugins.

Validation:

- 22 OpenCode tests pass, including executing both entrypoints in Node.
- Actual V1 1.18.29 and V2 2.0.5 request captures have correct session headers,
  tools, and reasoning effort; no implicit `max_tokens` was introduced.
- V2 created a CommonJS statistics module, then extended it on a warm follow-up.
  All 7 generated tests pass independently. All 9 follow-on model requests
  reused cache; the 10 requests ranged from 67.3 to 91.5 tok/s.
- Updated Desktop 1.18.31 extended a Python module over two turns and passed
  all 10 tests. All 7 follow-on requests reused cache, with 0.27–0.68 s TTFT.
  The 8 requests ranged from 54.8 to 108.9 tok/s.

The reported four-line instruction was verified in OpenCode's original bundled
system prompt and the captured request. MTPLX did not inject that instruction.
The original session had zero client-system replacements and zero active
transcript compaction. Native sampled settings and compact tool translation
remain in use.

## Performance findings

An original 108,925-token request generated 1,394 tokens at 26.25 tok/s.
Its 92,521-token prefix restored from RAM in 6.38 ms. MTP remained mostly D3
with compiled verification active. Verification consumed 39.97 of 53.11 decode
seconds (78.84 ms/call), and drafting another 10.60 seconds. Acceptance did not
collapse. This was not a full prefix miss or a switch to autoregressive decode.

The original recorder lacked synchronized GPU clocks, power and OS memory
pressure measurements, so the cause of the higher operation costs cannot be
conclusively assigned. Replaying the original 84k→117k message prefixes produced
55.6–63.3 tok/s, including 63.3 at 109k. Another back-to-back sequence reached
40.4 tok/s at 149k: the intermittent problem must not be declared solved by a
faster replay or restart.

A captured OpenCode transcript extended with real source excerpts encoded to
200,073 tokens. Explicit 1,024-token diagnostic runs measured 49.0–52.85 tok/s.
These are bounded decode diagnostics, not complete agent-task quality tests or
a guaranteed 50 tok/s floor. Some warm 200k runs used eager verification because
the additional 5.7 GB compiled-cache promotion did not fit alongside a protected
snapshot; the original slow 109k request stayed compiled. These are distinct
conditions.

Wired-budget, allocator-cache-size and fixed-D3 experiments did not establish a
consistent improvement. Padded short verification did not establish correctness.
None of those experiments is included. Engine, sampler, cache and fan defaults
are unchanged. Measurements used an M5 Max with 128 GiB RAM, MLX 0.32.2, native
sampled settings, verified max fans and this machine's pre-existing elevated
system wired limit; they are not evidence for all Apple Silicon memory tiers.

## Short reproduction without rerunning the agent task

`scripts/replay_chat_request.py` submits a captured chat-completions request to
an already-running local daemon. It records tool calls but does not execute
them. An explicit diagnostic output budget is required; prompts and sampler
settings otherwise remain those of the capture. It verifies actual max-fan
RPM before inference and refuses to overlap another active request.

```sh
python3 scripts/replay_chat_request.py captured-request.json \
  --messages 35 --max-tokens 1024 --seed 1731 \
  --session-id decode-replay --out evidence/new-run
```

Omit `--messages` to replay the full body. Omit `--seed` to preserve its captured
seed semantics. Repeat with the same session to measure warm reuse, or replay
successive message prefixes to retain prefill/snapshot transitions.

Evidence includes the exact request, timestamped stream, one-second flight and
health data, actual fan RPM, VM compression/swap, and the final request receipt.
If `macmon` is available, GPU clock/power/temperature samples are also recorded.
Monitoring failures are surfaced. The tool does not alter daemon settings or
manage the fan policy; restore the prior policy after an experiment.

The tool was exercised against the 200,073-token payload: 52.85 tok/s,
37.45 ms/verify, 96.97 GB active MLX memory, complete monitoring. A detailed
private incident report and raw captures are retained in the research output
directory `long-context-opencode-20260916`.
