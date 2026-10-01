#!/usr/bin/env python3
"""Audit the hand-pinned catalog size_bytes against live Hugging Face repos.

The Python catalog (mtplx/model_catalog.py) and the Swift mirror
(apps/MTPLXApp/Sources/MTPLXAppCore/Models/MTPLXModelOption.swift) pin each
official pack's exact download size. Head re-publishes change repo totals,
so this must run after every pack upload and both pins updated to match.

Prints one line per catalog entry: OK or MISMATCH with the exact new value
to pin. Exits 1 on a lookup error, missing size metadata, or any mismatch.
The two new 2.12.0 repos can each be explicitly marked with
``--not-yet-published REPO`` until upload; all other failures remain errors.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))


def main() -> None:
    from huggingface_hub import HfApi

    from mtplx.model_catalog import OFFICIAL_CATALOG

    from scripts.model_release_policy import add_not_yet_published_argument

    parser = argparse.ArgumentParser(description=__doc__)
    add_not_yet_published_argument(parser)
    args = parser.parse_args()
    api = HfApi()
    mismatches = 0
    errors = 0
    pending = 0
    for entry in OFFICIAL_CATALOG:
        repo = entry.hf_model_id
        try:
            info = api.model_info(repo_id=repo, files_metadata=True)
        except Exception as exc:
            if repo in args.not_yet_published:
                pending += 1
                print(f"NOT YET PUBLISHED {repo}: {exc}")
            else:
                errors += 1
                print(f"ERROR {repo}: {exc}")
            continue
        if not info.siblings or any(not isinstance(getattr(item, "size", None), int) for item in info.siblings):
            errors += 1
            print(f"ERROR {repo}: incomplete file-size metadata")
            continue
        total = sum(
            sibling.size
            for sibling in (info.siblings or [])
            if isinstance(getattr(sibling, "size", None), int)
        )
        if total == entry.size_bytes:
            print(f"OK       {repo}  {total:,}")
        else:
            mismatches += 1
            delta = total - entry.size_bytes
            print(
                f"MISMATCH {repo}\n"
                f"         pinned {entry.size_bytes:,}  live {total:,}  "
                f"(delta {delta:+,})\n"
                f"         pin -> size_bytes={total:_}"
            )
    if mismatches or errors:
        print(f"\n{mismatches} catalog pin(s) need updating; {errors} repository audit error(s).")
        raise SystemExit(1)
    print(f"\n{len(OFFICIAL_CATALOG) - pending} catalog pins match; {pending} explicitly not yet published.")


if __name__ == "__main__":
    main()
