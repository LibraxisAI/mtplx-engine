# Test files run, and a signal sent, while a measured cell was live cost a 128K cell 16 percent — no pytest and no signals to the runner while `GPU_WINDOW_ACTIVE` exists

**Symptom (2026-09-18 14:49 to 14:55):** A 128K cold prefill cell read 847
tok/s (forward 148.5 s) where its neighbours on the same code read 974 to
984 (127.8 s). No n-gram stall, no refusal line, nothing in the chunk trace
but every chunk slower across the board.

**Cause:** While that run was live I ran several pytest files (one of them
the 383-test server file) to validate the next edit, and sent the harness a
SIGINT to cut the run short. The interrupt did not stop it, the run carried
on, and its long cell shared the CPU with the test processes. The plan's
own rule 13 already says never to run pytest while `GPU_WINDOW_ACTIVE`
exists; I applied it to sub-agents and not to myself.

**Fix / rule:** While the marker file exists, the only work allowed on the
CPU is reading results and writing text. Tests for the next edit wait for
the gap between queue lines, or the queue gets a `SLEEP` line to make one.
A run that is no longer wanted is left to finish, or its queue line is
edited before it starts; it is never signalled. A cell measured beside any
of this is marked contaminated in the log at once and not quoted.
