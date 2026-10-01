# Decode fix: model-owned launch policy and adaptive startup calibration

Two reproducible performance defects were fixed after the initial long-context
investigation. The earlier V2 compatibility build did not contain these fixes.

## What changed

Coding-agent launchers pinned generic lazy target-distribution and lazy bonus
verification settings. They overrode Flash-Next's batched fixed-M4 policy.
Removing only the first setting activated lazy bonus, reducing a D3 window to
three rows and bypassing compiled M4 entirely: that incomplete experiment fell
to 39.7–41.1 tok/s and was rejected. Both pins are now removed from the shared
native/CLI presets, including Hermes CLI's bonus pin. Model/profile defaults
own evaluation and verify width; explicit operator settings still work.

The expected-value depth controller also treated a one-off restored-call cost
as recurring cost. At 109k, the first D3 verify took 125 ms, the next three about
31 ms, but the EWMA still predicted 95 ms. It therefore selected the slower
38 ms eager D2 route for 453 of 464 cycles. The existing trace-exclusion logic
missed this because a reused compiled function reported no new trace.

The controller now calibrates with the minimum of its existing four initial
cost samples, then resumes normal EWMA updates. It continues to lower depth
when sustained costs rise. This changes scheduling, not target sampling,
acceptance correction, prompts, reasoning budgets or context retention.

## Before and after

Same captured OpenCode request, seed 1731, native sampled settings, expected-value
adaptive policy, verified maximum fans and matched cool starts. Each measured
cell explicitly requests 1,024 output tokens. Baseline/candidate/candidate/baseline:

| Prompt tokens | Baseline A | Candidate A | Candidate B | Baseline B |
|---:|---:|---:|---:|---:|
| 108,919 | 48.02 | 60.83 | 62.72 | 49.64 |
| 200,073 | 47.59 | 49.97 | 50.55 | 50.23 |

At 109k, mean throughput rises **48.83 → 61.77 tok/s (+26.5%)**. Compiled calls
rise from 11 to 376; active memory stays about 94.69 GB and process peak stays
95.42 GB. Warm TTFT remains about 82–86 ms.

At 200k, the candidate holds around 50 tok/s. The small mean difference overlaps
warming variation, so it is not presented as a proven 2.8% gain. The protected
snapshot plus another 5.7 GB of compiled-cache promotion exceeds the admission
budget on warm turns; eager fallback remains enabled. Active memory is 98.01 GB
and peak 103.30 GB in all paired cells. No memory budget was raised.

Hardware: M5 Max, 128 GiB, MLX 0.32.2, Flash-Next Optimized Speed. The machine's
pre-existing system wired-limit override was not changed. These figures are not
a guarantee for every machine, prompt or concurrent workload.

## Validation and evidence

Regression tests exercise actual launcher-to-profile composition for chat,
OpenCode, Pi and Hermes, with Flash-Next and another model family. A new adaptive
test reproduces the measured startup spike, fails on the baseline, then verifies
both faster-depth selection and adaptation to a sustained slowdown. The native
command-builder suite passes 70 tests.

The broad Python run covered launchers, server, generation and adaptive policies.
One unrelated cancellation-worker timing assertion failed once and passed on
rerun without a code change; all adaptive tests passed. Raw logs retain this
distinction. Performance measurements used a private diagnostic observer; no
diagnostic endpoint or monkeypatch is packaged in the app.

Detailed private evidence is in research output `long-context-opencode-20260916`,
with per-cycle cost records and paired receipts under `sampler-causation`.
The original 26.25 tok/s incident remains in its Godview report. That request
was mostly D3, so the adaptive D2 defect alone does not explain all of its cost;
the missing original GPU/resource trace cannot be reconstructed retroactively.
The fixes above are supported by controlled, reversible performance comparisons.

## Installed product check

Signed build 2011046 contains the rebuilt native launcher and both updated
runtime wheels. Its actual OpenCode launch preset resolves to the model's
batched policy. The existing app and global CLI runtimes have matching sources;
the CLI reports the local 2.11.3 candidate without a launcher-path change.

With the native dashboard visible, the installed daemon measured 42.51 tok/s
after a cold 200k prefill, then 50.60 tok/s on a warm 200k follow-up with
200,068 cached tokens and 0.886 s TTFT. The cold run also recorded OS compression
and lower GPU clocks; it is retained as an actual product result, not omitted
in favor of the controlled figures.

OpenCode CLI, Pi and Hermes each completed a real code repair plus a follow-up
feature and passed five independent generated tests. OpenCode Desktop added a
function through the GUI and passed 13 tests. Native chat also produced the
correct visible response at 64.5 tok/s. Detailed receipts and the initial
Hermes fixture-directory mistake are retained in the private report.

The candidate is installed locally for testing. Nothing was pushed, tagged or
published by this work.
