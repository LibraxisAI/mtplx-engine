# Model Compatibility

MTPLX separates detection from support.

| Tier | Meaning | Default behavior |
|---|---|---|
| Verified | `mtplx_runtime.json` exists and matches the expected contract | Run |
| Architecture-compatible, unverified | Qwen3-Next MTP markers exist, but no MTPLX contract | Loads and runs, labeled unverified (regenerate provenance to clear the label) |
| AR-only | An exact architecture-specific AR loader is installed, but the checkpoint has no MTP head | Run only with target-only AR selected |
| Incompatible architecture | MTP markers exist for an unsupported architecture | Exit with roadmap pointer; experimental contract-gated backends exist for several of these families (DeepSeek V3/V4, GLM, MiMo, Nemotron-H, Step3.5, Hy-V3) |
| No MTP | No MTP head detected | Exit with a clear message |

The AR-only tier is narrow by design. It currently recognizes the exact
mixed-precision geometry and storage map of `mlx-community/Laguna-S-2.1-oQ4e` at
revision `8e3f5cad513746264940c1c4195de48d7ea345a5`. Local cache admission also
requires the pinned source marker, all 13 shards at their reviewed sizes, the
index, tokenizer, generation config, special tokens map, and Poolside chat
template. Other Laguna variants — including the earlier uniform-4bit build —
remain blocked until they have their own construction-time validation and
runtime evidence.

## Serving an existing Hugging Face snapshot

`serve --model` accepts a local directory or a Hugging Face repository ID.
For an ID, `resolve_model_path` checks the MTPLX cache, branded local builds,
then the shared `huggingface_hub` cache with `local_files_only=True`. The last
step respects the Hub cache configuration and does not download or copy
weights. An explicit snapshot path also avoids selecting a different cached
revision:

```sh
mtplx serve --model /path/to/hub/models--org--name/snapshots/REVISION --port 8001
mtplx serve --model org/name --port 8001
```

Resolving a complete snapshot proves that its files exist, not that its tensor
layout matches the backend. `can_run` is architecture admission; the first
weight load and a completed generation still have to succeed.

### Qwen3.8 Flash Next: architecture versus checkpoint layout

The native `qwen4_exp` / `qwen4_exp_text` backend does not require weights to
live in the MTPLX store. Its production n-gram loader does, however, use the
MTPLX `ngram-table.safetensors` sidecar, and its MTP loader expects
`mtp.safetensors`. Raw tiny-test n-gram tensors named `shard_N.weight` are a
separate supported input; they are not the quantized `shards.N.*` layout.

As checked on 2026-09-28, `grant-ai/Qwen3.8-Flash-Next-Abliterated-MLX-4bit`
revision `000544f8cddcbde27c1bc302deac2b5b4d45a5b1` uses 128 quantized
n-gram shards (`weight`, `scales`, `biases`) plus `weight_scale`. The backend
at source commit `452fad05688c571850727b06e838f6656235ae37` rejects those 385
parameters during weight loading. The snapshot also embeds `mtp.*` tensors
in its main shards; resolving its path cannot attach that head. An AR flag
does not solve the n-gram layout mismatch. Do not discard these parameters
or weaken strict weight loading to make startup pass.

The existing Forge path for already-MLX weights mirrors the source (normally
with symlinks) rather than translating this n-gram layout. It therefore does
not establish compatibility. `scripts/convert_qwen4_exp_fp8.py` builds the
house layout from an FP8 source with different tensor names; it is not a
converter for this quantized snapshot. Rebuilding from a different source
or quantization recipe also invalidates a same-checkpoint engine comparison.

A format-preserving route would need either a native reader for these packed
shards, including `weight_scale` and embedded MTP, or a separately validated
repack that keeps the quantized values and their semantics. Neither is supplied
by the cache resolver. The latter can reuse trunk files via links, but it must
exclude the old n-gram/MTP keys from loading; changing only the index is not
enough when the underlying loader reads all top-level safetensors files.

For capacity planning, this snapshot contains 114.07 GB of tensor files,
including 32.00 GB of n-gram tensors and 1.64 GB of MTP tensors (decimal GB).
A repack retaining the other tensors by reference needs about 33.64 GB of
additional tensor storage, plus metadata, scratch space and validation
headroom. A full duplicate needs about 114.07 GB. These are size estimates,
not a tested conversion recipe. Reading 114.07 GB and writing 33.64 GB would
take roughly 2.5–5 minutes at an assumed aggregate sequential throughput of
1–0.5 GB/s, before transformation and validation. No end-to-end Forge duration
has been measured for this snapshot.

## Embedded MTP heads and third-party loaders (#306)

An MTPLX-branded pack stores its MTP head as a standalone `mtp.safetensors`
sidecar. Do not brand or redistribute an artifact that keeps `mtp.*` tensors
embedded in the trunk shards with absolute norm gains: mlx-lm's qwen3.5-family
loader keys its +1.0 delta-norm restoration on the bare presence of those
keys, so it shifts every trunk RMSNorm of an already-absolute checkpoint a
second time. The model still loads and generates — with acceptance collapsed
to a few percent — so it benchmarks as "MTPLX models are slow" instead of
failing. MTPLX's own loader refuses such a trunk at load with the cause named
(the q-norm mean lands near 2.79 against a healthy 1.74–1.83 band). Rebuild
the pack through `mtplx forge`, which extracts the head into the sidecar and
decides the norm convention once per tensor set.
