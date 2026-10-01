#!/usr/bin/env python3
"""Write the launch-lane matrix the app's Swift parity test reads.

The matrix is every (client, model family, Adaptive depth switch) lane the
engine resolver (mtplx/launch_lane.py) produces for the app, plus the agent
env block per RAM seat (16 to 128 GB). The Swift test
LaunchLaneParityTests builds the app's daemon argv for the same cells and
compares. tests/test_launch_lane.py fails when this file is stale, so after
changing the resolver run:

    python scripts/write_launch_lane_matrix.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from mtplx.launch_lane import app_parity_fixture  # noqa: E402

FIXTURE = ROOT / "apps" / "MTPLXApp" / "Tests" / "Fixtures" / "launch_lane_matrix.json"


def render() -> str:
    return json.dumps(app_parity_fixture(), indent=1, sort_keys=True) + "\n"


def main() -> int:
    FIXTURE.parent.mkdir(parents=True, exist_ok=True)
    FIXTURE.write_text(render(), encoding="utf-8")
    print(f"wrote {FIXTURE}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
