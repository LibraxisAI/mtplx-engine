# A trace hook keyed on the first streamed event fired at prefill start on a 128K cell, because the server streams before the first token — key decode hooks on the first content or reasoning delta

**Symptom (2026-09-18 05:09):** The Metal trace meant to show a 128K decode
round held 9 long segments at 98% GPU busy and no 62 ms rounds. It was a
prefill trace. The 64K trace from the same run held 60 GPU intervals and
nothing else, and both analyses first failed with "no GPU intervals for
pid", because the recorded pid was the launcher and the GPU work belongs to
its child process.

**Cause:** `ov_run.py` fires a `tN` hook N seconds after the first event on
the response stream. On a long prefill the server sends events (role chunk,
progress and keep-alive) long before the first generated token, so the hook
ran 150 s early. The run folder's `process.json` stores the launcher pid.

**Fix / rule:** Trigger decode hooks on the first delta that carries
`content` or `reasoning_content`, never on the first event. Store the pid
that owns the GPU work (the listener on the port) in `process.json`, and
have the analyzer fall back to the pid with the most GPU intervals when the
named one has none.
