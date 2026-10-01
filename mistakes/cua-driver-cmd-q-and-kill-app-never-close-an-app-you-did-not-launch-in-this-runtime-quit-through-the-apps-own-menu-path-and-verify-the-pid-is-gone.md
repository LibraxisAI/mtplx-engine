# cua-driver cmd+Q and kill_app never close an app instance this runtime did not launch. Quit through the app's own menu path and verify the pid is gone

**Symptom:** The 2011030 QA instance of the candidate app (launched by an
earlier, expired cua-driver session) survived two foreground
`press_key cmd+q` calls ("effect: unverifiable") and `kill_app` refused it
("standard mode may terminate only a process proven to have been launched
by this Cua runtime"). Every rebuild of the candidate bundle was refused
with "target bundle is running" for twelve minutes (2026-09-06, 10:07 to
10:19).

**Cause:** A synthetic cmd+Q needs the key equivalent to reach the app's
main menu; the background and foreground key paths reported delivery but
the app never quit, and there is no read-back that proves a key press
landed. `kill_app` is scoped to pids the current runtime launched, and the
session that launched the instance had expired minutes earlier, so
ownership was gone.

**Fix / rule:** Close a driven app with
`invoke_menu ["<app menu title>", "Quit <app>"]` (the app menu title is
the bundle's display name, "MTPLX post 2112 20260906" for the isolated
build) and then check `pgrep -f <bundle>/Contents/MacOS` before the next
build; the menu path quit the instance on the first call. When a rebuild
is needed while an instance must stay up, build into a separate
`MTPLX_APP_BUNDLE_DIR` and swap it into `dist/` afterwards (with
`lsregister -f`), because `--no-launch` refuses only the exact running
path and kills nothing.
