# Ternary Bonsai 2 27B: memory and pack metadata

The pack is
[Youssofal/Ternary-Bonsai-2-27B-MTPLX-Optimized-Speed](https://huggingface.co/Youssofal/Ternary-Bonsai-2-27B-MTPLX-Optimized-Speed),
served as `mtplx-bonsai-2-27b-optimized-speed`. It needs MTPLX 2.12.0 or later.
This page explains how its memory was measured, what the memory planner does
with it on small Macs, and how the pack's card and runtime contract are
stamped.

## What was measured

Measured on an M5 Max by limiting the engine to each class's memory budget,
with text prompts, plain decoding, and the draft head and vision tower loaded:

| Mac memory | Engine budget | Context window | Peak, prompts up to 8K | Peak, 16K prompt |
|---|---:|---|---:|---:|
| 16 GB | 12.0 GiB | 8,192 tokens, no warm cache in RAM | 11.55 to 11.80 GiB | 12.11 GiB, over the budget |
| 18 GB | 13.5 GiB | 20,480 tokens (36,864 with 8-bit KV) | 11.55 to 11.78 GiB | 12.28 GiB |
| 24 GB | 18.0 GiB | 94,208 tokens (167,936 with 8-bit KV) | 11.55 to 11.78 GiB | 12.28 GiB |

With the draft head active under the 16 GB budget, a 7,006-token prompt and a
1,024-token answer peaked at 11.54 GiB of GPU memory and 12.79 GiB for the
whole process, with no swap growth. The peak sits about 3.1 GiB above the
weights and the KV cache in every run, which is what the planner reserves for
runtime work. Image activations and a populated session cache were not part of
these runs, and no physical 16 GB Mac was used.

## How the planner treats a 16 GB Mac

A 16 GB Mac gets a 12 GiB engine budget. The planner used to refuse any model
whose budget could not fund, all at once, the weights, the 3 GiB runtime
reserve, the 1 GiB session cache floor and one 4,096-token block of KV cache.
Bonsai on 16 GB misses by the cache floor alone. That floor limits the warm
cache; it is not a physical need. When it is the only thing left unfunded, the
planner now admits the model with the cache floor at zero, a 256 MiB margin
over the runtime reserve, and the KV cache counted at its full width, and
restores come from the SSD cache. The rule applies to a pack whose weights are
no larger than this pack's 8,834,412,216 bytes (`TIGHT_MACHINE_MAX_WEIGHTS_BYTES`),
and to a larger pack only when its runtime contract carries its own measured
memory table (`memory_evidence`) for those exact weights, with a completed run
inside the budget that the rule admitted. The published Bonsai pack carries its
table as well. A pack that can fund the floor keeps it unless dropping it
gives a larger window, capped at what the rule could grant, so a lighter pack
never plans less context than a heavier one. No catalog model's window changes.

## Running the memory table

```sh
PY=/path/to/venv/bin/python
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$PWD" "$PY" scripts/bonsai_memory_table.py \
  --pack "$HOME/.mtplx/models/Ternary-Bonsai-2-27B-MTPLX-Optimized-Speed" \
  --out outputs/bonsai-memory \
  --classes 16 18 24 --contexts 4K 8K 16K
```

Run it outside any sandbox that blocks Metal, with one model loaded at a time.
The output directory must be new; it receives `memory.json` and `memory.md`.
`--dry-run` prints the whole matrix and the planner verdicts without importing
MLX, loading weights or writing files. Classes are GiB of RAM, and a `K`
context suffix means 1,024 tokens.

The table has 36 cases: three RAM classes, three prompt sizes, KV off and
8-bit, and 0 or 1,024 decoded tokens. One child process per RAM class loads
the model once, sets the MLX memory limit to the planner's engine budget and
the wired limit with the formula `mtplx serve` uses, and runs every case of
that class from a fresh request cache. Inherited `MTPLX_*` settings are
excluded from the child, and the settings it actually used are recorded.

The reported peak is the larger of the load peak and the request peak. MLX's
memory limit is a guideline, so a completed row above the engine budget is
marked `within_engine_budget: false`. A failed load or request keeps its error
type and message, marks the rows that could not run, and the next RAM class
starts in a new process. Exit code 0 means every case completed, not that
every case fit its budget.

## Stamping the pack

`scripts/build_bonsai_mtplx_pack.py` builds the pack from Prism ML's release,
restamps an existing pack into a new directory, or records measured results in
a built pack:

```sh
# Copy an existing pack into a new directory and stamp the measured memory table.
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$PWD" "$PY" scripts/build_bonsai_mtplx_pack.py \
  --restamp "$HOME/.mtplx/models/Ternary-Bonsai-2-27B-MTPLX-Optimized-Speed" \
  --out outputs/Ternary-Bonsai-2-27B-MTPLX-Optimized-Speed \
  --memory-json outputs/bonsai-memory/memory.json

# Record measured speed and the reason for the default in a built pack.
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$PWD" "$PY" scripts/build_bonsai_mtplx_pack.py \
  --stamp outputs/Ternary-Bonsai-2-27B-MTPLX-Optimized-Speed \
  --speed-evidence-json speed.json \
  --recommended-generation-mode mtp \
  --recommended-generation-mode-reason "..."
```

A restamp hard-links the weights where it can and copies them otherwise
(`--link-mode copy` forces copies). The license and notice are copied
verbatim, earlier parity results and unknown metadata are kept, the card is
generated again, and the manifest checksums are refreshed. A restamp refuses
an existing destination, a destination that overlaps the source, and a memory
report whose weight sizes do not match the pack. A stamp moves the previous
`mtplx_runtime.json` into `../_aside/` before it writes the new one.

The speed JSON must be an object with `status: "measured"` and a nonempty
`rows` list. Each row gives `hardware`, `context_tokens`,
`ar_tokens_per_second`, `mtp_tokens_per_second` and
`accepted_tokens_per_step`, all finite and positive (zero accepted tokens is
allowed). A whole measurement log whose last line is the JSON object is also
accepted.

The runtime contract names the trunk family `qwen3_8` and the quantization
container `prism_hadamard_qwen35`, and it sets `min_engine_version: 2.12.0`.
The `mtplx_version` field records the MTPLX version that first built the pack.
Its exactness record was measured on 22 September 2026 and stamped as passed
with `--exactness-json` and `--exactness-status`. Against Prism ML's own
runtime, on identical tokens and image pixels over 1,920 text positions, the
mean KL divergence is 4.0e-6 in MTPLX's float16 default, where the top token
matches at 1,919 positions and the other is an exact tie in float16. With the
auxiliary tensors in float32 it is 1.1e-7, and the top token matches at every
position. The card's memory section shows one row per RAM class: the planner's
context window, with the 8-bit KV cache window when it differs, and the highest
peak among the runs that completed inside the budget.
