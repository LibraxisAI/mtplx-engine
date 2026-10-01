"""A selected development runtime must not import another checkout from cwd."""

import os
from pathlib import Path
import subprocess
import sys


def test_source_wrapper_ignores_a_shadow_package_in_working_directory(tmp_path):
    selected = tmp_path / "selected"
    shadow = tmp_path / "working-directory"
    for root, message in ((selected, "selected runtime"), (shadow, "wrong runtime")):
        package = root / "mtplx"
        package.mkdir(parents=True)
        (package / "__init__.py").write_text("")
        (package / "cli.py").write_text(f"print({message!r})\n")

    env = {
        **os.environ,
        "MTPLX_RUNTIME_ENV_FILE": str(tmp_path / "absent.env"),
        "MTPLX_RUNTIME_VENV_PY": sys.executable,
        "MTPLX_RUNTIME_SOURCE_SHADOW": str(selected),
        "MTPLX_VLLM_METAL_REPO": "",
        "MTPLX_MLX_FORK_PYTHON_ROOT": "",
        "PYTHONPYCACHEPREFIX": str(tmp_path / "pycache"),
    }
    wrapper = Path(__file__).resolve().parents[1] / "bin/mtplx"
    result = subprocess.run(
        [str(wrapper), "--version"], cwd=shadow, env=env,
        capture_output=True, text=True, timeout=20, check=True,
    )
    assert result.stdout.strip() == "selected runtime"
    assert result.stderr == ""
