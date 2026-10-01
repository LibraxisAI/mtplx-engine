# Runtime Contract

Verified models include `mtplx_runtime.json`.

```json
{
  "mtplx_version": "2.4.2",
  "arch_id": "qwen3-next-mtp",
  "mtp_depth_max": 3,
  "recommended_profile": "stable",
  "exactness_baseline": {
    "context": 2048,
    "max_abs_diff": 0.0
  },
  "verified_on": {
    "timestamp": "2026-05-02T00:00:00Z",
    "hardware": "Apple Silicon",
    "macos": "macOS"
  }
}
```

`mtplx_version` is stamped with the runtime's real version at build time, and
`recommended_profile` above is illustrative — `turbo` is the shipped default
profile for the quantized flagships.

Architecture-compatible models without this contract are not supported by default.

Packs can optionally recommend a generation mode independently of their runtime
profile and MTP depth. The typed contract preserves these fields:

```json
{
  "recommended_generation_mode": "ar",
  "recommended_generation_mode_reason": "Depth 1 measured 1.06x on code and 0.94x on reasoning; depths 2 and 3 were slower at the native sampler.",
  "recommended_generation_mode_evidence": {
    "measured_at": "2026-09-21T17:09-07:00",
    "rows": [
      {
        "hardware": "MacBook Pro M5 Max 128 GB, reasoning prompt (thinking on), MTP depth 1",
        "context_tokens": 107,
        "ar_tokens_per_second": 32.2,
        "mtp_tokens_per_second": 30.3,
        "accepted_tokens_per_step": 0.4
      }
    ]
  },
  "mtp_depth_default": 1
}
```

The example shows one row of the Bonsai 2 measurement; the builder records all
supplied rows without recalculating or inventing measurements. Mode accepts only
`mtp` or `ar`; reason is free text, and evidence is a JSON object containing the
speed evidence's `measured_at` and `rows`. These fields are optional for older
packs. Missing mode retains the MTP default, and missing evidence makes no
measurement claim.

`mtplx serve` applies `ar` when the user omits mode controls or passes
`--generation-mode auto`. An explicit `--generation-mode mtp` or `ar` wins.
`--no-mtp` serves AR with the head loaded; `--stock-ar` or `--no-load-mtp`
serves AR without loading it. Explicit `--mtp` or `--load-mtp` retains MTP
unless `--generation-mode auto` asks for the pack default. Missing or unsupported
heads still use the existing compatibility fallback.

A pack recommendation keeps `load_mtp` enabled and preserves `mtp_depth_default`
for switching back to MTP. The daemon supports live mode changes through
`POST /v1/mtplx/settings` with `{"generation_mode": "mtp"}`. `/health` reports the
served default in both `generation_mode` and `default_generation_mode`.
The recommendation notice is emitted once at the public serve handoff; JSON
dry-run callers receive it on stderr so stdout remains valid JSON.

Like the depth default, the recommendation falls back to top-level
`mtplx_runtime.json` metadata when an older inspection result omits the fields.
Typed contract fields take precedence, and a different `--profile` does not
discard the pack recommendation.

The Bonsai builder accepts `--recommended-generation-mode {mtp,ar}` and
`--recommended-generation-mode-reason TEXT` for build, stamp, and restamp.
For the measured Bonsai pack, stamp `--recommended-generation-mode ar` with
`--mtp-depth-default 1` and `--speed-evidence-json` pointing to the measurement.
Restamp preserves the existing depth. New builds retain the existing placeholder
depth until explicitly measured and stamped. Omitted recommendation options
preserve existing metadata; new builds default to MTP. Changing the mode without
a new reason removes the previous mode's rationale. Card refreshes preserve
previously recorded speed and memory evidence when no replacement is supplied.
