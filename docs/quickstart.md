# Quickstart

```bash
brew install youssofal/mtplx/mtplx

mtplx help
mtplx doctor --summary
mtplx start
```

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

Homebrew is the recommended macOS path. Python-only installs can use PyPI:

```bash
python3 -m pip install -U mtplx
```

The built-in downloader is the default. For faster parallel downloads, install aria2 and opt in:

```bash
brew install aria2
mtplx pull --download-backend aria2 Youssofal/Qwen3.8-27B-MTPLX-Optimized-Speed
```

`mtplx pull --download-backend aria2` requires aria2c and fails if it is missing; `--download-backend auto` uses aria2c when it is installed and the built-in downloader otherwise; `python` (the default) never touches aria2c.
Hugging Face credentials are passed to aria2c through standard input and are never placed in the process arguments.

**Behind a company proxy that inspects HTTPS.** If a download stops with `CERTIFICATE_VERIFY_FAILED` while `curl` reaches the same address, the proxy signs traffic with its own root certificate. macOS and `curl` trust it through the keychain; Python does not read the keychain. Either of these fixes it:

```bash
# 1. Let Python use the macOS keychain (what pip itself does). MTPLX picks it up when it is installed.
python3 -m pip install truststore

# 2. Or point Python at the proxy's root certificate (ask your IT team for the .pem file).
export SSL_CERT_FILE=/path/to/proxy-root.pem REQUESTS_CA_BUNDLE=/path/to/proxy-root.pem
```

`MTPLX_SYSTEM_TRUST=0` keeps MTPLX on the bundled certificates even when `truststore` is installed.

The GitHub release wheel remains available for reproducible installs:

```bash
gh release download --repo youssofal/mtplx --pattern '*.whl'   # latest tagged release
python3 -m pip install ./mtplx-*-py3-none-any.whl
```

The commands above are no-MLX-safe except generation and serving. A missing MLX runtime should appear in `doctor` as an actionable dependency issue, not a traceback.

After the verified model is available:

```bash
mtplx start
mtplx start cli
mtplx start cli --no-mtp
mtplx quickstart --port 8000 --no-stats-footer
```

`--no-mtp` switches generation to target-only AR. For MTP-equipped models the
MTP runtime stays loaded, so terminal chat can use `/mtp off`, `/mtp on`, and
`/mtp status` without reloading. Native AR-only models such as
`mlx-community/Laguna-S-2.1-oQ4e` instead install an unloaded AR route at
construction because there is no MTP head to retain.

For scheduler selection and backend-specific concurrent implementations, see
[Concurrency modes](concurrency.md).

The Laguna download is pinned automatically. It needs about 64.13 GB of disk
space, and the runtime's admission gate requires ≈85.3 GiB of unified memory
(weights plus runtime headroom and a 16 GiB system reserve) — in practice a
96 GB Mac, with 128 GB comfortable. Its default
context and maximum response are 32,768 tokens. A larger explicit server
context is accepted only when it fits the active Metal resident-memory cap.

Use `mtplx doctor --deep --json` for exhaustive diagnostics and `mtplx doctor --bundle` to create a redacted support bundle.
