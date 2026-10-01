# A script's `pgrep -f` guard matched the launching shell's own argv because the command line quoted the same pattern — keep guard patterns out of the launcher, and match on paths, not fragments

**Symptom (2026-09-16 10:15, twice tonight):** `build_candidate.sh` refused
with "an MTPLX app is running" while no app was running; earlier
`pgrep -f speed_ab_decomp.sh` reported a waiter shell as the runner.

**Cause:** the tool call that launched the script had, in the same
`zsh -c "..."` command line, a `pgrep -f 'MTPLX.app/Contents/MacOS|...'`
of my own. `pgrep -f` matches full argv, and the wrapper shell's argv
contained the literal pattern text, so the child's guard saw its own
ancestor and refused. Same shape when a waiter's `zsh -c 'while ...
speed_ab_decomp ...'` line matched a `pgrep -f speed_ab_decomp.sh`.

**Fix / rule:** launch guarded scripts from a command line that contains no
copy of their guard pattern (separate call, or read the guard's result
from a file); in guards, match on a path that only the real process has
(`/Contents/MacOS/MTPLXApp` as an argv[0] check, `-x` for exact names), and
exclude the caller's process group. When a guard refuses, list the matching
pids before believing it.
