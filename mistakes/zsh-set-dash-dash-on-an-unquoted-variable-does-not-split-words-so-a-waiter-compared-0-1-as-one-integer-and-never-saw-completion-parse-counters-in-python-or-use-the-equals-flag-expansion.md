# zsh `set -- $C` on an unquoted variable does not split words, so a waiter compared "0 1" as one integer and never saw completion — parse counters in Python or use the `${=C}` expansion

**Symptom (2026-09-16 08:25):** A background waiter polling `/health` for
`requests_completed active_requests` printed
`integer expression expected: 0 1` every 20 s and would have run its full
30 minutes without noticing the app turn had finished.

**Cause:** zsh does not word-split unquoted parameter expansions by default,
so `set -- $C` with `C="0 1"` gives one positional parameter `"0 1"`, and
`[ "$1" -ge 1 ]` fails. The same family as the unquoted `$H` command
variable and the flag-plus-value-as-one-word trap already in this ledger.

**Fix / rule:** In zsh runners, never split with `set --`; either parse the
values in a Python one-liner and exit 0/1 on the condition, or force the
split with `${=C}` / `read -A`. Test a waiter's condition once by hand with
a fake value before starting a long poll.
