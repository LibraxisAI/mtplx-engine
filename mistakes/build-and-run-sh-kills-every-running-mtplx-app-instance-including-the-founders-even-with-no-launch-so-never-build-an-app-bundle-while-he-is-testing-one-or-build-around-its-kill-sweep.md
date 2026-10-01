# build_and_run.sh kills every running MTPLX app instance (including the founder's) even with --no-launch — never build an app bundle while he is testing one, or build around its kill sweep

**Symptom:** The founder's release-candidate app window vanished mid-test,
twice on the same night (RC2 at 19:29, RC3 at 00:26), each time exactly
when an agent started another app bundle build with
`apps/MTPLXApp/script/build_and_run.sh --no-launch`.

**Cause:** Before compiling, the script runs `app_pids()` AND
`misdirected_app_pids()` and `kill_tree`s every process whose command
contains `.app/Contents/MacOS/MTPLXApp` — any bundle path, any instance,
plus its children (the daemon). `--no-launch` only skips the relaunch at
the end; it does not skip the sweep. The agent that briefed two subagents
not to run the script then ran it itself for the merged RC build.

**Fix / rule:** Before any `build_and_run.sh` invocation, `ps` for a
running `MTPLXApp` and, if the founder has one open, either wait or tell
him first. For a candidate build while an app is live, compile with
`swift build -c release` and assemble/sign the bundle without the script,
or invoke the script only after he has closed the app himself. Never
treat `--no-launch` as "leaves running apps alone".
