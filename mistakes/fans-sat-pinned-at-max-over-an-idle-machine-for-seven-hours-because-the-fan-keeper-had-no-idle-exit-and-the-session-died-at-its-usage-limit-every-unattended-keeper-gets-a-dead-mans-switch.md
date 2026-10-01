# Fans sat pinned at max over an idle machine for seven hours because the fan keeper had no idle exit and the session died at its usage limit — every unattended keeper gets a dead-man's switch

**Symptom (2026-09-18 05:50 to 12:56):** The GPU queue drained at 05:49.
The session had stopped at its usage limit (three sub-agents plus the main
session on one model; the agents died at 04:45, reset 07:00) and did not
resume by itself. Nothing fed the queue again, and the keeper script kept
both fans at 7,800 rpm until the founder came back at 12:55.

**Cause:** `fan_keeper.sh` re-pinned max fans every 20 s "until killed",
and only the session could kill it. The design assumed the session outlives
its own helpers. The three-agent cap from 2026-09-09 was kept and the limit
still tripped, because each of the three agents ran long, tool-heavy work
at the same time as a main session that was itself reading large files.

**Fix / rule:** A helper that holds the machine in a non-default state
(fans, wired limit, caffeinate) carries its own idle exit: the keeper now
watches the run marker `GPU_WINDOW_ACTIVE` and, after 30 minutes with no
run, restores automatic fans, logs it and exits. Queue enough measured work
before starting agents that the machine stays useful if the session dies,
and end the queue with a line that restores fans. On an overnight window,
start one sub-agent, read the usage figure, and only then add a second.
