# Flash-Next Optimized Quality (MTPLX 2.12.0)

The published pack is
[Youssofal/Qwen3.8-Flash-Next-MTPLX-Optimized-Quality](https://huggingface.co/Youssofal/Qwen3.8-Flash-Next-MTPLX-Optimized-Quality)
(169,958,537,520 bytes in 57 files) and needs MTPLX 2.12.0 or later. When it
serves, MTPLX streams the n-gram table from SSD on every Mac, so the weights
need about 128.5 GiB. This page describes how the pack is built.

From this checkout on an Apple Silicon Mac with **256 GB or more**, with
MTPLX's dependencies, **MLX 0.32.2**, and the `hf` CLI already installed:

```bash
bash scripts/build_flash_next_quality_pack.sh Qwen/Qwen3.8-Flash-Next /Volumes/Build/Qwen3.8-Flash-Next-MTPLX-Optimized-Quality
```

Set `PY=/path/to/venv/bin/python` if needed. The first argument can instead be
an existing BF16 checkpoint directory. `Qwen/Qwen3.8-Flash-Next` is the
official BF16 repo, also identified by the existing Speed pack's base-model
credit. Do not use the FP8 repo or a converted Speed pack: quantization cannot
recover the original BF16 weights.

Add `--dry-run` to print the plan and disk arithmetic without creating files,
connecting to the Hub, importing MLX, or running conversion. Add `--upload`
only to upload the completed artifact to
`Youssofal/Qwen3.8-Flash-Next-MTPLX-Optimized-Quality`. Upload uses `hf upload`
and the user's existing `hf` login. No token argument or token handling exists.

The output must be new. Download resumes in the sibling
`Qwen3.8-Flash-Next-BF16` directory (override with `--download-dir`), using
`hf download --local-dir`; current HF downloads resume automatically. The Hub
revision is resolved once to a commit before download (`--revision` defaults
to `main`). Each run writes logs under `<output>.runs/<UTC timestamp>` and
stops on the first failure. Source and failed output are retained. After a
failed conversion choose a new output directory; the BF16 download is reused.

## Storage and memory

Estimates come from the tensor inventory of the official BF16 checkpoint, with
the **Q4 table**, in decimal GB unless labelled GiB. They are planning numbers,
not measured conversion peaks.

| Stage | Disk | RAM |
|---|---|---|
| Download BF16 | About 360 GB (335.28 GiB) source; no duplicate full Hub cache | Download buffers; pipeline preflight requires 256 GB |
| Conversion | Source + about 170 GB output + 40 GB headroom = **570 GB free initially** (530.85 GiB) | Whole-body Q8 peak unmeasured. N-gram input shards are about 0.8 GB; body tensors are evaluated individually into nominal 4 GiB shards |
| Existing source | **210 GB additional free** on the output filesystem | Same conversion uncertainty |
| Header/checksum/sample audit | No second pack copy; metadata/logs only | Host row samples and streaming SHA-256; never materializes an expert bank or whole table |
| Full-load verification and serve smoke | About 170 GB artifact, source retained | About 128.5 GiB of weights, with the n-gram table streamed from SSD. Requires a 256 GB machine for this pipeline |

The 40 GB reserve is an explicit disk safety allowance, not a claimed scratch
peak. Resumed downloads conservatively reserve the full 360 GB again at
preflight. Free space is checked per device; shared APFS space is not summed.
The script records the actual listener PID, sampled listener/process-tree RSS,
available MLX peak counters, request JSON, responses, and runtime snapshots.
The smoke opens a 128K context but sends **short** chat/tool/image requests.
It does not test a 128K workload or a real client.

The explicit smoke response budget is 2,048 tokens per request, with a
1,800-second startup timeout and 600-second request timeout. Override using
`--smoke-max-tokens`, `--startup-timeout`, and `--request-timeout`. A truncated
response fails; budgets are not silently lowered.

## Forge preset and verification

The family-qualified preset follows the standalone converter's `speed` recipe
naming. `flash-next-optimized-speed` preserves that converter's Q8 attention
choice; it is not claimed to reproduce the older local Speed artifact.

```bash
mtplx forge build --repo /data/Flash-Next-BF16 --model-root /data/models --out /data/forge-runs --run-id quality --branded-name Qwen3.8-Flash-Next-MTPLX-Optimized-Quality --recipe flash-next-optimized-quality
```

Quality fixes body/MTP matrices at affine Q8/g64, n-gram at Q4/g32, and
structural/vision tensors at BF16 (integer PLE buffers remain I64). It refuses
precision overrides, quantized sources, and Q8 n-gram tables before conversion.
The default `--verification full-load` runs the existing serving verification
and gates before writing `verified_on` and `verification.status=verified`.
Existing MTP correctness/performance gates remain in force; no Speed-pack
comparison is implied by passing them.

On a smaller Mac, the same build may be attempted with
`--verification streaming`. This runs all-header geometry/precision checks,
two-pass output SHA-256 checks, and source dequantization comparisons for the
first/last tensor of every class at first/middle/last rows. It also samples both
sides of **every** source n-gram shard boundary. Structural samples must match
exactly after documented layout and norm transforms. Quantized samples use a
recorded one-step plus BF16-rounding error bound. Samples are not exhaustive
tensor parity. Source SHA-256s identify local sources without Hub provenance.

Streaming produces **`streaming-audited`**, no `verified_on`, no inherited
speed evidence, and `full_load_verified=false`. It does not silently fall back
from full-load verification. A 128 GB Mac cannot load this model even with its
table streamed, and building it on a 128 GB Mac has not been measured for
memory.
Move the exact artifact to a larger Mac, then explicitly promote it with:

```bash
mtplx forge verify /data/models/Qwen3.8-Flash-Next-MTPLX-Optimized-Quality --stamp
```

The generated card reads the artifact's metadata: every observed tensor class
with its stored precision, the license and credits, and the source revision.
It gives the memory guidance by Mac size, including why a 128 GB Mac cannot
load the pack, and says that its speed is not measured yet. `size-checksums.json`
lists all final files except itself, with actual sizes and SHA-256s.

## Local validation and component parity

```bash
PYTHONPATH=$PWD "$PY" -m pytest tests/test_forge_* -q
PYTHONPATH=$PWD "$PY" scripts/qwen4_quality_component_parity.py --output outputs/quality-component-parity.json
```

The round-trip tests select CPU and convert/load a tiny model with body,
MTP, vision, PLE, GDN and QSA tensors, both raw expert layouts, and deliberately
small output shards. The separate header/corruption tests use NumPy and run
without Metal or importing `mlx.core`.

The parity script uses random BF16 tensors at the official shapes for one
512-expert bank, one QSA layer, one GDN layer and MTP projections. It checks
MLX Q8/g64 matmul against a float32 dequantized reference at M=1, M=4 and
M=2048, reporting BF16-source error separately. It evaluates projections
individually, not whole-layer recurrence/attention, full-head generation, or
checkpoint quality. Output is JSON; missing Metal/import failure returns
status `unavailable` and exit 2. `--dry-run` prints shapes without touching MLX.

These checks do not replace a full load of the real pack on a 256 GB Mac.
