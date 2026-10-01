# An A/B runner flipped a live setting under a fixed 20 s bound while the daemon was in a 42 s idle postcommit. Wait for idle before flipping, and treat a flip as a re-prefill

**Symptom:** The 27B depth-policy pair at 19k context (run_C15.sh, 2026-09-06
08:09) produced arms A1 and B1 and then printed `settings POST failed` for
B2 and A2; the ABBA design lost its drift cancellation and the whole pair
had to be rerun (12 minutes of GPU).

**Cause:** The runner switched the policy through `POST /v1/mtplx/settings`
with `curl -m 20`. The session-bank fingerprint carries the adaptive-depth
config on purpose, so the first flip invalidated the banked 18,930-token
prefix and the daemon's idle postcommit re-prefilled it for 42.5 s
(`cache_miss_reason: policy_mismatch`). The POST waited behind that work,
the 20 s bound fired, and the runner treated the timeout as a hard failure
with no retry. The Flash-Next pairs earlier in the night never hit this
because their postcommits took a few seconds.

**Fix / rule:** A live-settings flip in a runner retries with a generous
bound until the daemon answers (the runner now tries 12 times at 30 s), or
polls `/health` for an idle daemon first. Any setting that sits in the bank
fingerprint (depth, adaptive policy, generation mode, thinking) costs one
cold re-prefill per banked session when flipped; budget that time in the
runner and say it in the release notes for the user-facing switch.
