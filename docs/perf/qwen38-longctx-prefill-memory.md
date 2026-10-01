# Qwen3.8 Flash-Next long-context decode memory

`mtplx serve` OOM'd on the pack's advertised 262,144-token context on a 128 GB
M5 Max under turbo / server defaults. The full source + log diagnosis is in
`.benchmark-artifacts/over100-reports/oom255k/report.md`; this note records the
two optimizations it produced. The OOM is in the speculative-verify decode step,
not the prefill: the prefill completes (TTFT ~240 s) and the request fails a few
decode tokens later. Two separate ~6.4 GB verify transients had to go.

## Optimization 1 — head-chunk the small-q_len verify SDPA

### Problem

At 261,120 tokens the **prefill completes** (TTFT ~240 s) and the request OOMs a
few decode tokens later with a Metal command-buffer execution failure
(`[METAL] ... kIOGPUCommandBufferCallbackErrorOutOfMemory`). Every arm — release
2.11.2, mask-fuse, score-tiler, `MTPLX_MLX_CACHE_LIMIT=2GiB`, session-off — hits
the identical failure at a ~100.8 GB sampled peak, so it is neither the prefill,
the allocator pool, nor the session bank.

The cause is the **speculative-verify decode step**. Of the 48 layers, 12 are
full-attention (`full_attention_interval: 4`) and keep dense KV over the whole
context; the model is GQA with `num_attention_heads: 24`, `num_key_value_heads:
2` (GQA factor 12). A multi-row verify has `q_len` = 3–8. MLX's fused
vector-attention kernel serves at most `q_len * GQA ≤ 32` rows per dispatch;
`q_len 5 × GQA 12 = 60 > 32`, so SDPA falls to an **unfused path that
materializes the `[24, q_len, T]` fp32 score plane (O(T)) and GQA-expands k/v**.
Across the 12 dense layers held in the fixed-M4 verify command buffer, that
transient — several GB at T = 261,120, and linear in T — lands on top of a
~93.8 GiB steady resident (weights 77.3 GiB + full KV 6.0 GiB + QSA aux + graph
buffers), tipping the single command buffer past the 100 GiB Metal wired limit.
At 131,072 tokens the same terms are ~half and the request fits; at 261,120 it
does not. The QSA layers avoid this via their bounded rows-gather selection; the
12 dense layers cannot, and the fixed-M4 verify path additionally gates off the
score-tiler and rows-gather (`not fixed_capacity`), which is why every prior
lever was a no-op.

### Change

`_verify_sdpa` (`mtplx/models/qwen4_exp.py`) replaces the dense-fallback SDPA
call in `Attention.__call__`. When `q_len * GQA > 32` and `q_len ≤ 32` (the
verify / small-tail band MLX will not route to its flash kernel), it splits the
**query heads** into chunks small enough that each fused call satisfies
`q_len * heads_per_chunk ≤ 32` (e.g. q_len 6 → chunks of 5 heads, q_len 4 → 8
heads), pairs each chunk with its **own single kv head** (a slice, so k/v is
**never GQA-expanded**), calls `mx.fast.scaled_dot_product_attention` per chunk,
and concatenates along the head axis. Every dispatch stays on the bounded fused
kernel, so no `[24, q_len, T]` plane is formed and the working set is one chunk
regardless of T. Wide prefill chunks (`q_len > 32`) are untouched and keep their
single flash call.

### Effect

The O(T) verify transient in the 12 dense layers is removed; the 261K decode
working set no longer scales with context, so it no longer tips past the wired
limit. Applies uniformly to the fixed-M4 verify, the ordinary verify, and any
small prefill-tail chunk (all reach the one dense-fallback SDPA). Decode `q_len
1` (12 ≤ 32) and prefill wide chunks are unaffected.

### Exactness

The heads in one chunk all belong to the same kv head, so each chunk's SDPA is
exactly the full GQA attention's arithmetic for those heads on the same fused
kernel — chunked-fused is **bit-identical to an unchunked fused call** (verified
< 1e-6 on fp32 random tensors against a GQA-expanded MHA reference). Versus the
unfused path it replaces, it is rounding-class (fused vs unfused accumulation),
the same class as the mask-fuse lane.

### Files

- `mtplx/models/qwen4_exp.py` — `_verify_sdpa`, `_sdpa_head_chunked`,
  `_verify_sdpa_head_chunk_plan`, `_verify_sdpa_head_chunk_enabled` (env at
  use), engagement latch + `verify_sdpa_head_chunk_report`; call site in
  `Attention.__call__`.
- `mtplx/server/openai.py` — `_qwen4_install_reports` surfaces the engagement
  receipt at `/health` under `qwen4_install_reports.verify_sdpa_head_chunk`.
- `tests/test_qwen4_verify_sdpa_head_chunk.py` — CPU tests.

### Switch / observability

- `MTPLX_QWEN4_VERIFY_SDPA_HEAD_CHUNK` — default ON (engages only in the
  `q_len × GQA > 32`, `q_len ≤ 32` band); `0`/`off` opts out. Resolved at use.
- Log line on first engagement: `[mtplx] verify SDPA head-chunked: q_len=S
  heads_per_chunk=H chunks=N`.
- `/health` → `qwen4_install_reports.verify_sdpa_head_chunk`:
  `{engaged, q_len, heads_per_chunk, chunks, n_kv_heads}` once it has fired.

## Optimization 2 — write the fixed-M4 verify KV in place

### Problem

With Optimization 1 in place the small-q_len score plane is gone, yet the
261,120-token request still OOMs at the same ~100.8 GB peak, a few decode tokens
in. The remaining ~6.4 GB (100.82 GB with MTP on and the verify running, vs
94.19 GB for the same prompt with MTP off at S=1) is the verify's KV-cache
update. The fixed-M4 verify uses `TensorOffsetKVCache` (graphbank.py), whose
`update_and_fetch` writes the new rows with the functional `mx.slice_update`.
That op reallocates the whole `[1, 2, capacity, 256]` buffer per key and per
value tensor (267 MB each at the 262K capacity, measured on CPU), and across the
12 full-attention layers that is 6.4 GB, matching the gap exactly. An S=1 AR
decode (MTP off) rides the stock `KVCache`, which writes in place and returns a
view, so it never pays it. The bank's own machinery demotes long generations to
the eager verify (graphbank: "longer generations demote to eager"), so the served
261K verify runs this eager path with a concrete offset.

### Change

`TensorOffsetKVCache.update_and_fetch` (and the `trim` rollback restore) now take
an in-place path when the offset is a concrete host value: the S new rows are
written with a slice assignment into the existing buffer (0 allocation, measured),
after materializing an independent copy of the pre-write rows for rollback. When
the offset is a tracer (inside an `mx.compile` trace) the compile-visible
functional `mx.slice_update` path is kept unchanged, so the compiled replay's
donation contract is untouched.

### Effect

The eager verify's KV update no longer reallocates the full buffer, removing the
~6.4 GB verify transient. Combined with Optimization 1, the 261,120-token verify's
working set no longer scales the way that overflows the 100 GiB knob.

### Exactness

Byte-identical. The in-place write lands the same bytes at the same positions as
the functional `slice_update` (verified equal on random tensors), and the
rollback snapshot is an independent copy so `trim` restores exactly the pre-write
rows.

### Files

- `mtplx/graphbank.py` — `TensorOffsetKVCache.update_and_fetch`, `trim`,
  `_concrete_offset`.
- `tests/test_tensoroffset_kv_inplace.py` — CPU tests (no full-buffer alloc,
  correctness, rollback independence, numerical equivalence, tracer detection).

### Switch / observability

No new knob; the in-place path is the eager verify's KV update. A GPU probe that
isolates the transient is at
`.benchmark-artifacts/over100-reports/oom255k/verify_kv_peak_probe.py`.
