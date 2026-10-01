# A test path list built with sed ended in a bare tests directory and ran the full suite inside a GPU window: build pytest argv as an array of existing files and refuse any directory

**Symptom.** A "targeted" run of 26 cold-tier and session-bank test files took 247 s and reported 7,004 passed. It was the whole suite, under `nice -n 15`, from 04:23:38 to 04:27:45 PDT on 2026-09-18, while `GPU_WINDOW_ACTIVE` existed for part of that time. The rule for that window was targeted files only, and the night's budget was two full runs.

**Cause.** The file list was written with `tr '\n' ' '`, which leaves a trailing space, and the `tests/` prefix was added with `sed 's#\([^ ]*\)#tests/\1#g'`. `[^ ]*` also matches the empty string after the last space, so the expansion ended with a bare `tests/`, and pytest collected the directory.

**Fix / rule.**
- Build the argv from real paths, never by text substitution: `files=(tests/test_cold_*.py tests/test_session_bank*.py)` in zsh, or `pytest $(ls tests/test_cold_*.py)`.
- Before any pytest call made under a CPU restriction, print the argv and check that every item is a file (`[[ -f $f ]]`); a directory in the list is a full run.
- A targeted run that takes minutes is not targeted. Stop it and read the collected count (`--collect-only -q | tail -1`) first.
- Tell the session that owns the machine the exact start and end times, so it can discard any measurement cell that overlapped.
