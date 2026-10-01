# A multi-hour phase chain started under the Bash tool's ten-minute bound, then killed to relaunch, lost a step and orphaned the next — launch long chains detached first, and never kill a chain shell to restart it

**Symptom (2026-09-16 08:53):** The post-app chain (Swift suite → mlx-serve
exactness → session-bank restarts → KL → lane A/B, ~2.5 h) was launched as a
background Bash call whose hard limit is 600 s. Realising that, I killed the
chain to relaunch it detached. The kill hit `swift test` mid-link, the chain
read the dead compile as a finished step and had already spawned phase 4
(mlx-serve booted, harness running, reparented to launchd) when the shell
died. The relaunch then refused on its own engine guard ("an engine is still
running"), and the Swift suite never ran in that pass.

**Cause:** Two habits collided: a long chain under a bounded runner, and a
`pkill` of the parent shell on the assumption that its children stop with
it. In a `for step` chain each step is its own child; killing the parent
lets the current child finish (or fail) and the next already-spawned child
run on.

**Fix / rule:** Anything that runs longer than a few minutes starts as
`(nohup zsh script > log 2>&1 &)` from the first launch and is followed by
polling its log. To restart a chain, wait for its current step to finish
(or stop that step by its own stop path) and re-run only the remaining
steps (`STEPS=...`), never `pkill` the chain shell. Each step should also
refuse to start unless its predecessor's receipt exists.
