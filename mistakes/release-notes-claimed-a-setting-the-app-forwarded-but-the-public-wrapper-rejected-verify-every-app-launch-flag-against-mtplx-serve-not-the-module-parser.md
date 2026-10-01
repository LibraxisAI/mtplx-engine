# Release notes claimed a setting the app forwarded but the public wrapper rejected. Verify every app launch flag against `mtplx serve`, not the module parser

**Symptom:** Release candidate 2011019 (2026-09-06 06:30) bricked the daemon
launch as soon as the new Settings stall watchdog was turned on: the app
built `mtplx serve ... --stream-stall-deadline-s 300`, the public wrapper
did not know the flag and exited before the model loaded. The release notes
had already described the setting as working.

**Cause:** The flag existed on the module parser (`python -m
mtplx.server.openai`), which is what the notes and the unit test were
checked against. The app never calls the module parser; it calls the public
`mtplx serve` wrapper in `mtplx/commands/public.py`, which forwards only
the flags it explicitly knows. The same shape hid the inert Adaptive depth
switch one build earlier: the live settings patch whitelisted keys and
dropped the new one.

**Fix / rule:** A setting is real only when the whole chain is exercised:
Settings UI to `MTPLXCommandBuilder` to `mtplx serve` wrapper to module
parser to `/v1/mtplx/settings` echo, on a launched daemon. For every new
app-forwarded flag, add the wrapper forwarding and a `_serve_dry_run_payload`
test in `tests/test_public_cli.py` in the same commit, and toggle the setting
both ways in the built app before the notes mention it.
