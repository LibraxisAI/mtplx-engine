# Piping a boot script that forks a background watchdog into `tail` hangs the caller until the daemon dies — redirect boot output to a file and read it, never through a pipe

**Symptom (2026-09-16 09:27):** `zsh ocdesktop_daemon.sh start 2>&1 | tail -4`
never returned; the Bash tool moved it to the background after 180 s while
the daemon had been healthy since second 13.

**Cause:** boot.sh starts a free-pages watchdog as a detached subshell
(`( ( while true; ... ) & )`) that inherits stdout. A pipe closes only when
every writer closes it, so `tail` waited on the watchdog, which lives as
long as the daemon. The script had finished; the pipe had not.

**Fix / rule:** Boot and launch scripts that fork helpers are run with
their output redirected to a file (`> log 2>&1`), and the log is read
afterwards. Never put such a script on the left of a pipe, and never judge
its completion by the pipe returning. Same family as the detached-chain
rule: children of a launcher outlive the launcher's stdout.
