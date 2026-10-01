# Mocking the child runner does not block serve direct exec, so inspection must not enter a live launch path

Symptom: an attempted command inspection started the dense model without the required load guard and verified max fans, ending the 2.11.4 release task.

Cause: the inspection mocked `_run_server_child_with_app_parent_watchdog`, but `cmd_serve_public` with default fan mode and no app parent calls `os.execvpe` instead. The inspector was replaced by the real server.

Rule: inspect command construction as source text. Never import or invoke command entry points for inspection, even under mocks or with dry-run flags. Start release servers only through the guarded launcher. Run in-process model diagnostics only after the load guard and verified maximum fans. On accidental startup, stop only the owned process, verify it exited, restore fans, preserve receipts and disclose the violation.

Receipt: `outputs/release-2114/research/A9-stop-cleanup.json` in the main release checkout. Owned PID 45622 on port 18877 exited after SIGINT; QA ports were empty and fans verified automatic. No release work continued after the violation.

The lead accepted that cleanup and resumed the task as A9b. A handled violation is recorded and work continues under the corrected procedure.
