# Editing a running zsh runner shifts its read offset; an unquoted zsh expansion passes a flag and its value as one word

Date: 2026-09-08, overnight quality window, 27B multi-turn probe runner.

What happened
- I edited `loop_probe_ab.sh` while a detached instance of it was still executing (the script's `for` loop was parsed, but zsh reads the trailing commands from the file by byte offset, so lines after the loop can be read from the wrong place after an edit that changes earlier line lengths). Nothing broke that time because I killed the run for another reason, but the exposure was real.
- The same edit added `${EFFORT:+--effort $EFFORT}` to a python invocation. zsh does not word-split a parameter expansion, so python received the single argument `--effort low` and argparse refused it ("unrecognized arguments: --effort low"). Two arms of the chain booted a 27B daemon, failed in twenty seconds each, and had to be rerun.

Rules
- Never edit a script that a running shell is executing. Copy it to a new name, edit the copy, and launch the copy; or wait for the run to end.
- Build optional argv in zsh as an array: `EXTRA=(); [ -n "$EFFORT" ] && EXTRA=(--effort $EFFORT); python3 driver.py "${EXTRA[@]}"`. The `${=var}` form also splits, but the array is clearer and survives values with spaces.
- The same zsh rule already bit git commit arguments and ruff file lists earlier in the window (`gc()` function, `${=FILES}`): treat every unquoted expansion that must become several words as a bug until it is an array.
