"""Optional real MicroMagnetic.jl integration check.

The unit test suite is deliberately runnable without Julia.  Set
``SOFTMAG_TEST_JULIA=1`` after ``julia/setup.jl`` to exercise the real
Python -> Julia -> Python contract with a tiny, non-quantitative mesh.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest


pytestmark = [
    pytest.mark.skipif(
        shutil.which("julia") is None,
        reason="Julia is not on PATH; run julia/setup.jl and refresh PATH",
    ),
    pytest.mark.skipif(
        os.environ.get("SOFTMAG_TEST_JULIA") != "1",
        reason="set SOFTMAG_TEST_JULIA=1 to run the real MicroMagnetic.jl check",
    ),
]


@pytest.mark.parametrize(
    "shape,mesh",
    [
        ("box", (8, 4, 2)),
        ("sphere", (16, 16, 8)),
        ("cluster", (16, 16, 8)),
    ],
)
def test_real_micromagnetics_contract(
    tmp_path: Path, shape: str, mesh: tuple[int, int, int]
) -> None:
    root = Path(__file__).resolve().parents[1]
    out = tmp_path / f"julia_check_{shape}"
    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "softmag",
            "julia-check",
            "--project",
            str(root / "configs/project.yaml"),
            "--output",
            str(out),
            "--shape",
            shape,
            "--mesh-nx",
            str(mesh[0]),
            "--mesh-ny",
            str(mesh[1]),
            "--mesh-nz",
            str(mesh[2]),
            "--timeout",
            "600",
        ],
        cwd=root,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    assert proc.returncode == 0, (
        f"julia-check failed\nstdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
    )
    report = json.loads((out / "julia_check.json").read_text(encoding="utf-8"))
    assert report["status"] == "passed"
    assert report["engine"] == "MicroMagnetic.jl"
    assert report["source"] == "julia"
    assert report["model_status"] == "conditional"
    assert report["shape"] == shape
    assert report["n_loop_points"] == 2 * report["field_steps"]
    assert report["max_abs_B_closure_error_T"] <= 1.0e-12
    assert report["package_version"]
    assert all(report["checks"].values())
    for name in ("job.json", "loop.csv", "result.json", "summary.json"):
        assert (out / name).is_file()
