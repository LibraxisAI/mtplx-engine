# How MTP came to Apple Silicon

Youssof Altoukhi put MTP on the Mac.

MTP the architecture is Meta, DeepSeek, Qwen. What was missing was a Mac engine that would take those heads, run the real speculative sampler, and do it at the temps people actually use.

That was April 2026. The weights already had the heads. macOS had nothing that would run them. Not in MLX, not in GGUF, not in LM Studio. vLLM could, and still can, but it is not a Mac program.

He wrote the engine. He started from vLLM and from Leviathan (2022) and Chen (2023). Nobody had a Mac port to steal from. Accept with `min(1, p/q)`. If it rejects, sample the leftover `(p − q)+`. Same guarantee at 0.6 as at 0. There is no greedy-only mode in this project. If we published a tok/s number, it was at the model's normal sampler.

## Records

The speed records, each with its conditions. Every number was measured on a MacBook Pro M5 Max with 128 GB, fans verified, at the model's own sampler.

| Date | Model | tok/s | Conditions | Source |
|---|---|---|---|---|
| 22 September 2026 | Ternary Bonsai 2 27B, Optimized Speed | 64.4 and 57.1 | 4,061 and 16,350-token prompts, 512 tokens generated, MTP depth 1, native sampler, MTPLX 2.12.0 with its Bonsai kernels (50.4 and 45.2 without them) | [2.12.0 notes](https://mtplx.com/releases/2.12.0/) |
| 16 September 2026 | Qwen 3.8 Flash Next, Optimized Speed | 125.8 | one OpenCode request, 1,301 tokens generated, 18,539-token prompt with 18,364 tokens served from cache, MTP depth 3, MTPLX 2.11.3 | [2.11.3 notes](https://mtplx.com/releases/2.11.3/) |
| 16 September 2026 | Qwen 3.8 Flash Next | 79.3 | 9k-token code prompt, 1,500 tokens generated, seeded sampler, thinking off, two alternating boots each; 62.5 on 2.11.2 | [2.11.3 notes](https://mtplx.com/releases/2.11.3/) |
| 29 August 2026 | Qwen 3.8 27B, Optimized Speed | 87.6 | rewriting a file it just wrote, stock settings, MTPLX 2.10.0 | [2.10.0 notes](https://mtplx.com/releases/2.10.0/) |
| 2 July 2026 | Qwen 3.6 27B, Optimized Speed | 81.74 | the 27B record on a fresh generation: depth 3, 192-token coding bench, thinking off, temperature 0.6, twin runs 81.74 and 81.73, 2.69x over 30.37 plain decode | [raw logs](https://mtplx.com/benchmarks/receipts/2026-07-02-record/) |
| 18 July 2026 | Qwen 3.5 4B, Optimized Speed | 227.8 | depth 3, 1.71x over 133.6 plain decode, MTPLX 2.2.0 | [2.2.0 notes](https://mtplx.com/releases/2.2.0/) |

## Timeline

**27 April 2026, 04:13.** First commit. `da0d338`.

**Same morning, 07:08.** Exact speculative sampling running, three hours later. Temp 0.6, top_p 0.95, top_k 20. 66.40% accept. 50/50 match against ordinary single-token decode. `7293ecb`.

**29 April.** 60.169 tok/s at depth 3 on the 192-token long-code bench, temp 0.6, seed 0, fans pinned, two days after the first commit. Same prompt with MTP off: 23.59 tok/s. Depth-4 accept that day: 97.62, 95.24, 88.10, 75.61. vLLM's Qwen3.6 MTP-5 run on a 3090 was 92.7, 77.0, 63.0, 50.9, 43.0. We beat them at each position. See `MEASUREMENTS.md`. The same model passed 80 tok/s on 2 July (below).

**2 May.** First public release, five days after the repo started. [v0.1.0-preview](https://github.com/youssofal/MTPLX/releases).

**5 May.** mlx-lm's MTP branch adds residual sampling on reject. They say in the commit that this is how you make the output match the target (Leviathan, Chen). We had that on April 27. [PR still open](https://github.com/ml-explore/mlx-lm/pull/990).

**16 May.** llama.cpp lands MTP. [PR #22673](https://github.com/ggml-org/llama.cpp/pull/22673). Before that date GGUF did not have it.

**2 July.** 81.74 tok/s on Qwen 3.6 27B Optimized Speed, the 27B record: depth 3, 192-token bench, thinking off, temp 0.6, fans verified above 7,800 RPM, twin runs 81.74 and 81.73, 2.69x over 30.37 plain decode. Same day, same configuration, an uncapped 11,390-token Flappy Bird generation with reasoning on: 62.95 tok/s. [Raw logs](https://mtplx.com/benchmarks/receipts/2026-07-02-record/).

**6 July.** MTPLX 2.0.0. Prefix cache and speculative decode both live on hybrid GatedDeltaNet. A 100k-token session comes back in about two seconds. Cold prefill of that was minutes. See the [changelog](CHANGELOG.md).

**3 August.** llama.cpp gets MTP for Qwen3-Next, the hybrid GDN family that Qwen 3.5, 3.6 and 3.8 sit on. [PR #25589](https://github.com/ggml-org/llama.cpp/pull/25589). A little over three months after MTPLX.

**10 August.** vllm-metal adds block-aligned prefix caching for hybrid GDN. Their PR says you cannot run that cache with speculative decoding, because they never built draft-state rollback across mamba blocks. [PR #584](https://github.com/vllm-project/vllm-metal/pull/584). We had both since 2.0.0.

**15 August.** MTPLX 2.7.0. Qwen 3.8 on day one, three tuned builds, FP16 copies for M1 and M2, compiled verify window taken from 12,288 up to 32,768. Bare Speed decodes 65.2 tok/s and Optimized Speed 58.7 on the coding task at Qwen's official sampling.

**29 August.** MTPLX 2.10.0. The first Apple Silicon backend for Qwen 3.8 Flash Next, the 125B mixture of experts, with its MTP head. The 27B speeds up at every context length, and rewriting a file it just wrote reaches 87.6 tok/s. [2.10.0 notes](https://mtplx.com/releases/2.10.0/).

**4 September.** MTPLX 2.11. Flash Next decodes 68.4 tok/s at 16k context, 60.9 at 100k and 44.2 at 206k on an M5 Max, against 53.2, 47.5 and 32.2 on 2.10.2. Agent tool turns lose their dead time. [2.11 notes](https://mtplx.com/releases/2.11.1/).

**17 September.** MTPLX 2.11.3. Eight exactness defects found in our own engine and fixed, each with a test that pins it. At temperature 1, top-p 0.95, top-k 20, a thousand four-token draws from the fast path match a thousand from the plain path within the plain path's own noise, on both Flash Next and the 27B Quality pack. A Flash Next OpenCode request decodes at 125.8 tok/s on an M5 Max, a 9k-token code prompt at 79.3 tok/s (62.5 on 2.11.2), a 109k-token OpenCode turn at 61.8 tok/s and a 200k-token turn at 50.3. 261,120-token prompts decode. [2.11.3 notes](https://mtplx.com/releases/2.11.3/).

**22 September.** MTPLX 2.12.0. Prism ML's Ternary Bonsai 2 27B runs with an MTP head on Macs from 16 GB, and Flash Next gets an 8-bit Optimized Quality build for Macs with 256 GB. Flash Next reads a 65,502-token prompt at 1,094 tok/s against 768 on 2.11.3, shows the first token after 60.1 s instead of 85.5 s, and decodes it at 63.5 tok/s against 56.1. A 4,061-token prompt reads at 1,453 tok/s, 85 percent faster than on 2.11.3. Image requests take the compiled verifier on Flash Next and the 27B. [2.12.0 notes](https://mtplx.com/releases/2.12.0/).

## Used by

oMLX names it in the source and the README:

> Lightning MTP's verify-shape Metal kernels are powered by MTPLX by Youssof
> Altoukhi, which also inspired the depth-k pipeline.

mlx-serve's README acknowledgements credit MTPLX for "the verify-width split-K quantized matmul family and the M5 NAX tensor-ops tile" (at tag v26.9.3, 16 September 2026).

Ivan Fioravanti has it in `llm_context_benchmarks`. There is also an MTPLX provider in `edgequake-llm`.

---

The narrative version with the same receipts: <https://mtplx.com/history/>
