"""Command-line interface for the soft magnetic powder-core predictor."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Sequence

from .configs import load_micromagnetics_config, load_project
from .io_utils import (
    ensure_dir,
    julia_version,
    read_json,
    software_versions,
    timestamp_utc,
    write_json,
)
from .micromagnetics import MicromagneticsError, run_micromagnetics
from .pipeline import run_all
from .schemas import ConfigError, load_config
from .units import MU0


def _julia_check(args: argparse.Namespace) -> int:
    """Run a tiny real MicroMagnetic.jl job through the Python bridge.

    This is a contract smoke test, not a quantitative material simulation.  It
    verifies that Julia can be located, the pinned environment loads, the
    JSON/CSV contract is honoured, and B = mu0 (H + phi M) survives the
    round-trip.
    """
    project, project_file = load_project(args.project)
    if project.micromagnetics_config is None:
        raise ConfigError("project.micromagnetics_config 未设置")
    prediction = load_config(project.prediction_config)
    mm = load_micromagnetics_config(project.micromagnetics_config)

    julia = julia_version(mm.julia_executable)
    if julia is None:
        raise MicromagneticsError(
            f"找不到 Julia 可执行文件 {mm.julia_executable!r}；"
            "请安装 Julia >= 1.11 并刷新 PATH（见 docs/deployment.md）。"
        )

    mm.mode = "julia"
    mm.particle_shape = str(args.shape)
    mm.mesh_nx = int(args.mesh_nx)
    mm.mesh_ny = int(args.mesh_ny)
    mm.mesh_nz = int(args.mesh_nz)
    mm.hmax_A_m = float(args.hmax)
    mm.n_field_steps = int(args.field_steps)
    mm.sweep = "full_loop"
    mm.timeout_s = int(args.timeout)
    prediction.solver.allow_fallback_models = False

    out = ensure_dir(
        Path(args.output).expanduser().resolve()
        if args.output
        else Path(project.output_dir) / "julia_check"
    )
    print(f"project: {project_file}")
    print(f"Julia:   {julia}")
    print(
        "mesh:    "
        f"{mm.mesh_nx}x{mm.mesh_ny}x{mm.mesh_nz}, shape={mm.particle_shape}, "
        f"{mm.n_field_steps} field steps, full_loop"
    )
    print(f"output:  {out}")
    print("running real MicroMagnetic.jl smoke job ...")

    result = run_micromagnetics(prediction, mm, out, run_id="julia_check")
    df = result["dataframe"]
    summary = result["summary"]
    meta = read_json(out / "result.json") if (out / "result.json").is_file() else {}
    job = read_json(out / "job.json") if (out / "job.json").is_file() else {}

    checks: dict[str, bool] = {
        "source_is_julia": result.get("source") == "julia",
        "summary_engine_is_micromagnetics": summary.get("engine")
        == "MicroMagnetic.jl",
        "model_status_is_conditional": summary.get("model_status") == "conditional",
        "result_json_engine_matches": meta.get("engine") == "MicroMagnetic.jl",
        "package_version_present": bool(meta.get("package_version")),
        "julia_version_present": bool(meta.get("julia_version")),
        "full_loop_rows": len(df) == 2 * mm.n_field_steps,
        "job_contract_present": bool(job.get("sweep", {}).get("H_A_m")),
        "job_full_loop": job.get("sweep", {}).get("full_loop") is True,
        "loop_csv_written": (out / "loop.csv").is_file(),
        "summary_json_written": (out / "summary.json").is_file(),
    }

    h = df["H_A_m"].to_numpy(dtype=float)
    n_forward = len(h) // 2
    checks["sweep_forward_then_reverse"] = bool(
        n_forward > 0
        and all(h[i] <= h[i + 1] for i in range(n_forward - 1))
        and all(h[i] >= h[i + 1] for i in range(n_forward, len(h) - 1))
    )
    phi = float(summary.get("magnetic_volume_fraction") or 0.0)
    checks["magnetic_volume_fraction_present"] = 0.0 < phi <= 1.0
    m_err = float(
        (df["M_macro_A_m"] - phi * df["M_A_m"]).abs().max()
    ) if "M_macro_A_m" in df and "M_A_m" in df else float("inf")
    b_err = float(
        (df["B_T"] - MU0 * (df["H_A_m"] + df["M_macro_A_m"])).abs().max()
    ) if "B_T" in df and "M_macro_A_m" in df else float("inf")
    ms = float(meta.get("Ms_A_m") or 0.0)
    checks["m_macro_matches_phi_times_phase_m"] = m_err <= max(1.0, 1.0e-9 * ms)
    checks["b_closure"] = b_err <= 1.0e-12
    checks["finite_values"] = bool(
        df[["H_A_m", "M_A_m", "M_macro_A_m", "B_T"]].notna().all().all()
    )

    status = "passed" if all(checks.values()) else "failed"
    failed = sorted(k for k, v in checks.items() if not v)
    validation: dict[str, Any] = {
        "schema_version": "1.0",
        "status": status,
        "checked_utc": timestamp_utc(),
        "scope": "python_julia_contract_smoke_not_quantitative",
        "project": str(project_file),
        "output_dir": str(out),
        "julia": julia,
        "package_version": meta.get("package_version"),
        "engine": summary.get("engine"),
        "source": result.get("source"),
        "model_status": summary.get("model_status"),
        "mesh": meta.get("mesh"),
        "shape": mm.particle_shape,
        "field_steps": mm.n_field_steps,
        "n_loop_points": int(len(df)),
        "magnetic_volume_fraction": phi,
        "max_abs_m_macro_error_A_m": m_err,
        "max_abs_B_closure_error_T": b_err,
        "checks": checks,
        "failed_checks": failed,
        "software": software_versions(),
        "warnings": [
            "smoke mesh 仅用于验证 Python→Julia→Python 契约，不代表网格收敛或定量精度。",
            "本检查使用 D 级占位材料参数，不得解释为真实 FeSiCr 性能。",
            "代表性子体积结果不能外推到 20 µm 全颗粒。",
        ],
    }
    write_json(out / "julia_check.json", validation)

    print()
    print(f"status:   {status.upper()}")
    print(f"engine:   {summary.get('engine')}")
    print(f"package:  MicroMagnetic.jl {meta.get('package_version')}")
    print(f"loop:     {len(df)} points")
    print(f"max |B - mu0(H+phi M)|: {b_err:.3e} T")
    if failed:
        print("failed checks: " + ", ".join(failed), file=sys.stderr)
        return 1
    print(f"report:   {out / 'julia_check.json'}")
    return 0


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="softmag",
        description="Multi-scale soft magnetic powder-core predictor (DEM + micromagnetics + calibration)",
    )
    sub = p.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run-all", help="run the configured end-to-end pipeline")
    run.add_argument("--project", default="configs/project.yaml")
    run.add_argument("--run-id", default=None)
    run.add_argument(
        "--steps",
        default=None,
        help="comma-separated subset of steps (default: project file)",
    )
    run.add_argument("--quiet", action="store_true")

    val = sub.add_parser("validate", help="validate the input contract without running models")
    val.add_argument("--config", default="configs/material.yaml")
    val.add_argument("--project", default=None)

    demo = sub.add_parser("demo", help="run the demo project (same as run-all)")
    demo.add_argument("--project", default="configs/project.yaml")
    demo.add_argument("--run-id", default="demo")

    check = sub.add_parser(
        "julia-check",
        help="run a tiny real MicroMagnetic.jl contract check (requires Julia setup)",
    )
    check.add_argument("--project", default="configs/project.yaml")
    check.add_argument(
        "--output",
        default=None,
        help="output directory (default: <project output>/julia_check)",
    )
    check.add_argument("--mesh-nx", type=int, default=8)
    check.add_argument("--mesh-ny", type=int, default=4)
    check.add_argument("--mesh-nz", type=int, default=2)
    check.add_argument("--shape", choices=["box", "sphere", "cluster"], default="box")
    check.add_argument("--field-steps", type=int, default=5)
    check.add_argument("--hmax", type=float, default=1.0e5)
    check.add_argument("--timeout", type=int, default=600)

    return p


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.command == "validate":
            if args.project:
                project, path = load_project(args.project)
                prediction = load_config(project.prediction_config)
                print(f"project: {path}")
            else:
                prediction = load_config(args.config)
            missing = prediction.missing_parameters()
            print(f"material: {prediction.material.material_id}")
            quality = prediction.material.parameter_quality
            print(f"quality: {getattr(quality, 'value', quality)}")
            print("missing:")
            for layer, items in missing.items():
                print(f"  {layer}: {items if items else 'none'}")
            for w in prediction.quality_warnings():
                print(f"WARNING: {w}")
            return 0
        if args.command == "julia-check":
            return _julia_check(args)
        project = args.project
        if args.command == "demo" and args.run_id == "demo":
            run_id = "demo"
        else:
            run_id = getattr(args, "run_id", None)
        steps = None
        if getattr(args, "steps", None):
            steps = [s.strip() for s in args.steps.split(",") if s.strip()]
        log = (lambda *a, **k: None) if getattr(args, "quiet", False) else print
        results = run_all(project, steps=steps, run_id=run_id, log=log)
        print(results.get("run_dir"))
        if results.get("report"):
            print(results["report"].get("markdown"))
        return 0
    except (ConfigError, MicromagneticsError, ValueError, FileNotFoundError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
