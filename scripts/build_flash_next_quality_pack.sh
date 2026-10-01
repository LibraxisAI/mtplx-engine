#!/bin/bash
# All logic lives in the testable Python runner. No credentials are accepted.
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="${SCRIPT_DIR}/..${PYTHONPATH:+:${PYTHONPATH}}"
exec "${PY:-python3}" "${SCRIPT_DIR}/build_flash_next_quality_pack.py" "$@"
