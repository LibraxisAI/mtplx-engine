# A pipeline's last command set the exit code I printed, so a failing attribution scan read as rc=0 and reached the push — capture the script's own status before any pipe or tail

**Symptom (2026-09-16 21:2x, release 2.11.3):** `python3 scripts/check_ai_attribution.py 2>&1 | tail -3; echo "rc=$?"` printed the scan's own "Strip the attribution and push again" text followed by `rc=0`, so the pre-push check was read as clean. The first push of main then failed the no-ai-attribution workflow on a contributor commit (PR #502) carrying a "Co-Authored-By: Claude Sonnet 5" trailer, and main had to be rewritten and force-pushed before publication.

**Cause:** `$?` after a pipeline is the exit status of the last command in it (`tail`), not the script's. The scan had exited 1; the text on screen even said so, and I trusted the number over the words.

**Also tonight, a repeat of a ledgered mistake:** the fan keeper loop `while pgrep -f release_macos_v1.sh ...` matched its own `zsh -c` argv and never exited on its own (harmless here, killed by hand). The existing entry on `pgrep -f` self-matches already says to match on paths or `-x` and to keep the pattern out of the launcher's command line.

**Fix / rule:** capture a checker's status directly (`cmd > log 2>&1; rc=$?`) or use `pipefail`, never `$?` after `| tail`. Read the checker's words as well as its number. When a CI gate exists for a rule, run the exact gate command locally with its own exit code before pushing, and treat any advisory text as a failure until the code is confirmed.
