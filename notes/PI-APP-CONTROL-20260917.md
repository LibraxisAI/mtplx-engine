# Connected-app control validation — 2026-09-17

Implementation: a1152ee3 and 25b63b7d, based on public main 7c2205ae.

The native inference switch selects app-owned or client-owned reasoning and
sampling for managed connected clients. Anonymous API traffic and native chat
retain their existing contract. Pi uses a separate settings-mirror extension,
preserving user-owned request bridges.

Validated live native toggle transitions and real Pi requests: client/medium,
then app/xhigh without a daemon restart. Pi RPC state and footer agree. A two-turn
coding task repaired mean and added median; all six artifact tests passed. Final
installed-app Pi discovery and an uncapped test-running task also passed at xhigh.
713 Python tests, 12 scoped release Swift tests and 17 debug localization/ownership
checks passed. Local build 2011052 is installed and signed; no public release made.

The ABBA diagnostic preserved identical sampled output for corresponding request
positions and essentially unchanged peak memory. Concurrent compilation limits
fine-grained timing conclusions; no performance improvement is claimed.
