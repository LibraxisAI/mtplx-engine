# Install

See [INSTALL.md](../INSTALL.md) for the short path.

MTPLX is Apple-Silicon-first:

- macOS 14.0 or newer
- native arm64 Python 3.11 or newer
- `python3 -m pip install mlx` in that same environment
- enough unified memory and disk for the selected model/profile, checked by `mtplx doctor`

On M3, M4 and M5, the app and CLI recommend Qwen 3.5 4B Optimized Speed
below 16 GB, Ternary Bonsai 2 27B from 16 GB, Qwen 3.8 27B Optimized Speed
from 32 GB, and Qwen 3.8 Flash-Next Optimized Speed from 256 GB, with
Flash-Next Optimized Quality second. Flash-Next Bare Speed is offered from
96 GB and Flash-Next Optimized Speed from 128 GB. M1 and M2 keep the FP16
policy: the 9B below 32 GB when it fits, then the 27B trio. An 8 GB M1 or M2
Mac has no fitting curated FP16 model. An explicit model selection always
wins.

Ternary Bonsai 2 27B and Flash-Next Optimized Quality need MTPLX 2.12.0 or
later. The app shows the Recommended badge when the Mac has at least 1.5
times a model's peak memory. Before a download, the app and `mtplx pull` ask
for the bytes still to download plus 5 GiB of free disk. Neither rule changes
engine memory limits, context windows or admission checks.

The quantized 27B and 9B flagships (the Qwen 3.8 trio, Optimized-Speed, Optimized-Quality, the legacy Optimized hybrid, and their FP16 siblings, plus the 9B Speed pair) launch on the Turbo profile by default — the same NAX verify-kernel + compiled-verify fast path the macOS app uses; every other model defaults to Sustained (`--profile sustained`). `stable` remains available as the conservative compatibility alias, and Burst is available explicitly as `--profile performance-cold --max` for short-context benchmark runs.

Do not install model weights into the source checkout. Use the MTPLX model cache or a Hugging Face cache.
