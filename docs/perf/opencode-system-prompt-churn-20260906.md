# OpenCode system-prompt churn on the first file write: measurement and design note (2026-09-06)

## What was measured
Two OpenCode 1.18.29 sessions ran the same task (build a Flappy Bird game in one HTML file) against the shipped 2.11.2 chat lane on the app's daemon (Flash-Next Optimized-Speed, turbo, depth 3, session bank on), both in fresh non-git directories, both writing index.html during the first user turn. `mtplx trace session` on each:

| session | user turns | requests | warm reuse | re-prefill turns |
| --- | --- | --- | --- | --- |
| A1 04:38 (`opencode run`, then `opencode run -c` with a second prompt) | 2 | 15 | 88.6 percent | turn 11: prompt 25,537, cached 2,048, re-prefilled 23,489 tokens, TTFT 24.2 s, wall 55.9 s |
| gauge 06:18 (`opencode run`, one prompt) | 1 | 13 | 98.5 percent | none; turn 4 after the write reused 34,423 of 34,442 |

The re-prefill in A1 started one second after the second user message was stored (OpenCode DB: user message at 04:42:47, the request at 04:42:48). The common prefix was 2,048 tokens, the block boundary right after OpenCode's base instructions. Every other turn of both sessions reused 99 percent or more.

## Mechanism
OpenCode assembles the system prompt per user turn: base instructions, then an environment block (working directory, git status, platform, date, and the directory tree), then the agent instructions. Tool-call turns inside one user turn reuse the assembled prompt, so files created mid-turn do not touch it. A new user turn rebuilds it; by then the tree contains the files the previous turn created, the tokens after the base instructions differ, and the session bank's exact-prefix match ends there. The gauge session never rebuilt because it had one user turn. The date line has the same effect across midnight.

## Cost
One cold re-prefill of the whole conversation per user turn that follows a file write: 23.5k tokens in 24.2 s here (about 970 tok/s prefill on Flash-Next at that context), 10 percent of a five-minute two-turn task, and it grows with the conversation (a 100k-token session pays a minute and more each time). Interactive users feel it as the pause before the second reply.

## Options
1. Upstream (the real fix): OpenCode moves the volatile environment content (tree, date) out of the shared system prefix, for example into the first user message of the turn or after the stable agent instructions, or keeps the tree snapshot stable for the life of a session unless the user asks for a refresh. Any client that puts volatile text before the conversation defeats every prefix cache, not only MTPLX's; this is worth an issue on OpenCode with the numbers above.
2. MTPLX side, exact: none. The KV of every token after the changed block depends on that block, so the bank cannot reuse it without changing the model's output; a "semantic anchor" that realigns on the first user message would be a different model.
3. MTPLX side, cheaper misses: faster prefill (the Steel path on M5, issue #423) shortens the pause proportionally; the trace flag REDUCED PREFIX REUSE already names the turn so users can see why.
4. Connector side: `mtplx connect opencode` cannot change OpenCode's prompt layout; a note in the OpenCode section of the docs telling users the pause after a file write is the client rebuilding its prompt is the honest interim.

## Receipts
`mtplx trace session ses_f897c58deffeF8eiziD91Mqx85` and `ses_f892111b4ffeNuwxKLJY4CEHTk` (port 8000 request log), OpenCode DB `~/.local/share/opencode/opencode.db` message times, run_phaseA.sh steps A1 and A1b.
