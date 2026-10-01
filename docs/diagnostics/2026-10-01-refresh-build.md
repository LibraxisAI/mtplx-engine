# Refresh build and native Responses repair

Baseline: `eval/responses-on-2102` at
`ecc780b437b8bef7dea18a340b8bd8e81e01bb37`, a clean committed merge when this
repair began. Work was done in the shared Living Tree.

## Failure and repair

The Python package and Swift application already built after the merge.
The regression was the CI contract: the merge retained the native
`protocols/responses` route, but imported an unused upstream adapter and
selected that adapter's incompatible tests. Those tests expected stateless,
text-only behavior, unlike the mounted GenerationTurn adapter with bounded
storage, continuation and cancellation.

Removed the unmounted `mtplx/server/responses.py`, its unused import, tests
and exclusive fixtures. CI now runs the mounted adapter's existing tests,
store/event suites and hosted-web regressions. Updated the empty-input probe
to assert the native route's existing error shape and corrected the README
and API contract. The generation owner and route implementation are retained.

Added strict SDK validation of actual route JSON, stored replies, text SSE,
reasoning/tool SSE and partial failure events using `openai==2.52.0`, matching
the CI pin. This exposed a missing required usage field:
`input_tokens_details.cache_write_tokens`. The encoder now preserves that
counter from chat usage when provided, otherwise reports zero.

## Verification

- Current CI selection: **1152 passed, 1 skipped**, Python 3.12, SDK 2.52.0,
  97.24 seconds. The workflow itself selects Python 3.11 on macOS 14.
- Native Responses/store/events/SDK subset: **115 passed**.
- `python -m compileall -q mtplx` and import of `mtplx.server.openai`: pass.
- `uv lock --check --offline`: pass; lockfile unchanged.
- `uv build --out-dir /tmp/mtplx-build-20261001-final`: sdist and wheel pass.
  Inspected the wheel: native encoder includes the SDK field and the removed
  adapter is absent. The wheel is built from the sdist.
- `swift build --package-path apps/MTPLXApp -c release --disable-automatic-resolution`:
  pass, 139.20 seconds; pre-existing Swift warnings remain.
- Scoped Ruff F821/F811: pass. Scoped Semgrep on encoder and changed tests:
  1066 rules, 3 files, zero findings. This is not a whole-repository lint or
  security certification.

No application installation, service start/restart or production request is
claimed by this build repair. The companion runner port selects bounded
context-copy and GDN prefill improvements from the frozen merge baseline;
it does not import the upstream Responses adapter or another server owner.

Local execution logs:

- `/tmp/mtplx-ci-sdk-final-20261001.log`
- `/tmp/mtplx-native-sdk-20261001.log`
- `/tmp/mtplx-package-sdk-final-20261001.log`
- `/tmp/mtplx-build-20261001.log`
- `/tmp/mtplx-native-semgrep-20261001.log`
