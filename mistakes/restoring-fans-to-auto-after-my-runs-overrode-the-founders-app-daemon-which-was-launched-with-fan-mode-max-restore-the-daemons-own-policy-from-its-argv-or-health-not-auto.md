# Restoring fans to auto after my runs overrode the founder's app daemon, which was launched with --fan-mode max. Restore the daemon's own policy from its argv or /health, not auto

**Symptom:** 2026-09-06 13:15, the shutdown receipt showed the founder's
app daemon on :8000 reporting `fan_mode: max` (its argv carries
`--fan-mode max`, the app's performance setting) while `thermalforge
status` had both fans on auto. They had been on auto since 10:41, when I
restored "auto after the run" after restarting his daemon from his app.

**Cause:** The fan rule I follow (max fans verified before any
real-engine run, auto after) is written for my hand daemons. The founder's
app daemon is not my run: it is launched with its own fan policy and
expects it to hold while resident. Applying my rule's "after" step to his
daemon replaced his configuration with mine, and nothing re-asserted it
because the daemon sets the mode at startup, not on a timer.

**Fix / rule:** When the resident daemon belongs to the founder's app,
the state to restore after my work is the daemon's own policy: read
`--fan-mode` from its argv or `fan_mode` from `/health` and set the fans
to that (max here, 7,804 and 7,813 rpm at 13:16). "Auto after" applies
only to daemons I booted myself. The shutdown receipt now prints both the
daemon's fan mode and the physical fan mode so a mismatch is visible.
