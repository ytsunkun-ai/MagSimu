"""Python -> Julia micromagnetics bridge with an explicitly labelled fallback.

The Julia script receives a JSON job file and writes a CSV loop plus a small JSON
metadata file.  The Python side never assumes that Julia is installed.  In
``auto`` mode it tries Julia first and falls back only when the input contract
allows it.  The fallback is a tanh major-loop model, not micromagnetics.
"""

from __future__ import annotations

import json
import math
import shutil
import subprocess
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd

from .configs import MicromagneticsConfig
from .homogenization import (
    TanhLoopParameters,
    default_tanh_parameters,
    estimate_initial_permeability_from_tanh,
    tanh_major_loop,
)
from .io_utils import (
    ensure_dir,
    julia_version,
    project_root,
    read_csv,
    read_json,
    software_versions,
    timestamp_utc,
    write_csv,
    write_json,
)
from .schemas import PredictionConfig
from .units import MU0


class MicromagneticsError(RuntimeError):
    pass


# ---------------------------------------------------------------------------
# Job contract
# ---------------------------------------------------------------------------
def _mesh_extent_m(mm: MicromagneticsConfig) -> tuple[float, float, float]:
    return (mm.mesh_nx * mm.dx_m, mm.mesh_ny * mm.dy_m, mm.mesh_nz * mm.dz_m)


def build_julia_job(
    prediction: PredictionConfig,
    mm: MicromagneticsConfig,
    *,
    run_id: str | None = None,
) -> dict[str, Any]:
    """Build the JSON document consumed by ``julia/hysteresis.jl``.

    The mesh is a small representative sub-volume.  A 5 nm mesh over a 20 µm
    particle would require 4000^3 cells; the mesh extent is therefore reported
    explicitly and must not be presented as a full-particle simulation.
    """
    contract = prediction.contract(run_id=run_id)
    material = prediction.material
    phi, phi_source, phi_warnings = prediction.resolve_magnetic_volume_fraction()
    ext = _mesh_extent_m(mm)
    phase_Ms = float(mm.Ms_A_m or material.Ms_A_m or 0.0)
    if phase_Ms <= 0:
        raise MicromagneticsError("缺少 material.Ms_A_m，无法创建微磁学作业")
    Hmax = float(mm.hmax_A_m)
    n_steps = int(mm.n_field_steps)
    Hs = np.linspace(-Hmax, Hmax, max(n_steps, 5)).tolist()
    return {
        "schema_version": "1.0",
        "run_id": run_id,
        "input_id": contract["input_id"],
        "material": {
            "material_id": material.material_id,
            "Ms_A_m": phase_Ms,
            "exchange_A_J_m": float(mm.exchange_A_J_m or material.exchange_A_J_m or 0.0),
            "anisotropy_K_J_m3": float(
                mm.anisotropy_K_J_m3
                if mm.anisotropy_K_J_m3 is not None
                else (material.anisotropy_K_J_m3 or 0.0)
            ),
            "damping_alpha": float(
                mm.damping_alpha
                if mm.damping_alpha is not None
                else (material.damping_alpha or 0.1)
            ),
            "parameter_quality": contract["material"]["parameter_quality"],
        },
        "mesh": {
            "nx": int(mm.mesh_nx),
            "ny": int(mm.mesh_ny),
            "nz": int(mm.mesh_nz),
            "dx_m": float(mm.dx_m),
            "dy_m": float(mm.dy_m),
            "dz_m": float(mm.dz_m),
            "extent_m": list(ext),
            "note": (
                "代表性子体积，不是完整颗粒；5 nm 网格不能覆盖 20 µm 颗粒。"
            ),
        },
        "shape": {
            "model": mm.particle_shape,
            "d50_m": prediction.particle.d50_m,
            "n_cluster_particles": int(mm.n_cluster_particles),
            "note": (
                "box 表示颗粒内部子体积；sphere 半径为网格可容纳的代表性球；"
                "cluster 为示意性颗粒簇，不是完整 DEM 团聚体。"
            ),
        },
        "sweep": {
            "H_A_m": Hs,
            "direction": [1.0, 0.0, 0.0],
            "full_loop": mm.sweep == "full_loop",
            "hmax_A_m": Hmax,
            "n_field_steps": n_steps,
        },
        "simulation": {
            "demag": bool(mm.demag),
            "m0": list(mm.m0),
            "stopping_dmdt": float(mm.stopping_dmdt),
            "driver": "LLG",
            "integrator": "BS23",
        },
        "microstructure": {
            "magnetic_volume_fraction": phi,
            "magnetic_volume_fraction_source": phi_source,
        },
        "warnings": [
            "Julia 结果是代表性子体积的微磁学计算，不是 20 µm 全颗粒。",
            "网格收敛性未在本作业中自动验证。",
            *phi_warnings,
        ],
        "output": {
            "loop_csv": "loop.csv",
            "metadata_json": "result.json",
        },
    }


# ---------------------------------------------------------------------------
# Julia execution
# ---------------------------------------------------------------------------
def _write_process_logs(outdir: Path, proc: subprocess.CompletedProcess[str]) -> None:
    (outdir / "julia_stdout.txt").write_text(proc.stdout or "", encoding="utf-8")
    (outdir / "julia_stderr.txt").write_text(proc.stderr or "", encoding="utf-8")


def run_julia_job(
    job_path: str | Path,
    *,
    outdir: str | Path,
    mm: MicromagneticsConfig,
) -> subprocess.CompletedProcess[str]:
    script = project_root() / "julia" / "hysteresis.jl"
    if not script.is_file():
        raise MicromagneticsError(f"找不到 Julia 脚本：{script}")
    exe = mm.julia_executable
    resolved = shutil.which(exe)
    if resolved is None:
        raise MicromagneticsError(f"找不到 Julia 可执行文件：{exe}")
    cmd = [
        resolved,
        f"--project={project_root() / 'julia'}",
        str(script),
        "--input",
        str(job_path),
        "--output",
        str(outdir),
    ]
    try:
        proc = subprocess.run(
            cmd,
            cwd=str(project_root()),
            capture_output=True,
            text=True,
            timeout=int(mm.timeout_s),
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        raise MicromagneticsError(
            f"Julia 作业超过 {mm.timeout_s} s；已在 timeout_s 处终止。"
        ) from exc
    _write_process_logs(Path(outdir), proc)
    if proc.returncode != 0:
        tail = (proc.stderr or proc.stdout or "")[-2000:]
        raise MicromagneticsError(
            f"Julia 作业返回码 {proc.returncode}。日志尾部：\n{tail}"
        )
    return proc


def _validate_loop(df: pd.DataFrame) -> pd.DataFrame:
    aliases = {
        "H": "H_A_m",
        "M": "M_phase_A_m",
        "M_A_m": "M_phase_A_m",
        "B": "B_T",
    }
    df = df.rename(columns={k: v for k, v in aliases.items() if k in df.columns})
    if "H_A_m" not in df.columns:
        raise MicromagneticsError("Julia 输出缺少 H_A_m 列")
    if "M_phase_A_m" not in df.columns:
        if "B_T" not in df.columns:
            raise MicromagneticsError("Julia 输出必须包含 M_phase_A_m 或 B_T")
    df = df.copy()
    df["H_A_m"] = pd.to_numeric(df["H_A_m"], errors="coerce")
    if "M_phase_A_m" in df.columns:
        df["M_phase_A_m"] = pd.to_numeric(df["M_phase_A_m"], errors="coerce")
    if "B_T" in df.columns:
        df["B_T"] = pd.to_numeric(df["B_T"], errors="coerce")
    df = df.dropna(subset=["H_A_m"])
    if len(df) < 5:
        raise MicromagneticsError(f"Julia 输出有效点太少（{len(df)}）")
    return df.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Fallback
# ---------------------------------------------------------------------------
def _fallback_loop(
    prediction: PredictionConfig,
    mm: MicromagneticsConfig,
    *,
    reason: str,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    material = prediction.material
    if material.Ms_A_m is None or material.Ms_A_m <= 0:
        raise MicromagneticsError("缺少 material.Ms_A_m，无法运行 tanh fallback")
    phi, phi_source, phi_warnings = prediction.resolve_magnetic_volume_fraction()
    if phi is None:
        phi = 1.0
        phi_warnings = [
            *phi_warnings,
            "缺少磁性体积分数；fallback 暂按 phi=1 计算，结果仅用于流程验证。",
        ]
    initial = dict(mm.loop_initial or {})
    Hc = float(initial.get("Hc_A_m", 100.0))
    a_value = initial.get("a_A_m")
    Ms_value = float(initial.get("Ms_A_m", material.Ms_A_m))
    if "Hc_A_m" not in initial:
        phi_warnings.append(
            "fallback 的 Hc=100 A/m 是 D 级占位假设；必须用实验回线或 MicroMagnetic 结果替换。"
        )
    if a_value is not None:
        params = TanhLoopParameters(
            Ms_A_m=Ms_value, Hc_A_m=Hc, a_A_m=float(a_value)
        )
    else:
        target_mu = material.initial_permeability
        params = default_tanh_parameters(
            Ms_value,
            coercivity_A_m=Hc,
            magnetic_volume_fraction=float(phi),
            target_initial_permeability=target_mu,
            squareness=0.30,
        )
        if target_mu is None:
            # No target: choose a soft but finite loop; this is not a material
            # inference, only a stable starting point.
            params = TanhLoopParameters(
                Ms_A_m=Ms_value, Hc_A_m=Hc, a_A_m=max(50.0 * Hc, 1.0)
            )
            phi_warnings.append(
                "缺少 initial_permeability；fallback 使用 a=50*Hc 的占位假设。"
            )
    n_points = max(201, 5 * int(mm.n_field_steps))
    df, diag = tanh_major_loop(
        params,
        Hmax_A_m=float(mm.hmax_A_m),
        magnetic_volume_fraction=float(phi),
        n_points=n_points,
    )
    df = df.rename(columns={"M_phase_A_m": "M_phase_A_m"})
    summary = {
        **diag,
        "engine": "python_tanh_fallback",
        "engine_status": "reduced_order_not_micromagnetics",
        "model_status": "fallback_reduced_order",
        "fallback_reason": reason,
        "magnetic_volume_fraction": float(phi),
        "magnetic_volume_fraction_source": phi_source,
        "initial_permeability_model": estimate_initial_permeability_from_tanh(
            params, magnetic_volume_fraction=float(phi)
        ),
        "warnings": [
            *diag.get("warnings", []),
            "未运行 MicroMagnetic.jl；所有回线结果均为唯象 fallback。",
            *phi_warnings,
        ],
        "software": software_versions(),
        "created_utc": timestamp_utc(),
    }
    return df, summary


# ---------------------------------------------------------------------------
# Public runner
# ---------------------------------------------------------------------------
def run_micromagnetics(
    prediction: PredictionConfig,
    mm: MicromagneticsConfig,
    outdir: str | Path,
    *,
    run_id: str | None = None,
) -> dict[str, Any]:
    """Run or fall back, write ``loop.csv`` and ``summary.json``, return results."""
    out = ensure_dir(outdir)
    job = build_julia_job(prediction, mm, run_id=run_id)
    job_path = write_json(out / "job.json", job)
    julia_present = julia_version(mm.julia_executable) is not None
    mode = mm.mode
    warnings: list[str] = []
    source = "fallback"
    julia_meta: dict[str, Any] | None = None
    fallback_reason = ""

    if mode in {"auto", "julia"} and not julia_present:
        if mode == "julia":
            if not prediction.solver.allow_fallback_models:
                raise MicromagneticsError(
                    "mode=julia 但找不到 Julia；且 solver.allow_fallback_models=false"
                )
            fallback_reason = "找不到 Julia 可执行文件"
        else:
            fallback_reason = "找不到 Julia 可执行文件（auto 模式回退）"
    elif mode in {"auto", "julia"}:
        try:
            run_julia_job(job_path, outdir=out, mm=mm)
            loop_path = out / "loop.csv"
            if not loop_path.is_file():
                raise MicromagneticsError("Julia 成功退出但未写出 loop.csv")
            df = _validate_loop(read_csv(loop_path))
            if "M_phase_A_m" in df.columns:
                phi = job["microstructure"]["magnetic_volume_fraction"]
                if phi is None:
                    phi = 1.0
                df["M_macro_A_m"] = float(phi) * df["M_phase_A_m"]
                if "B_T" not in df.columns:
                    df["B_T"] = MU0 * (df["H_A_m"] + df["M_macro_A_m"])
            elif "B_T" not in df.columns:
                raise MicromagneticsError("Julia 输出中没有可用的 B_T 或 M_phase_A_m")
            meta_path = out / "result.json"
            if meta_path.is_file():
                julia_meta = read_json(meta_path)
            source = "julia"
            warnings.append(
                "MicroMagnetic.jl 结果来自代表性子体积；网格收敛性需另行验证。"
            )
        except Exception as exc:  # noqa: BLE001
            if mode == "julia" and not prediction.solver.allow_fallback_models:
                raise
            fallback_reason = f"Julia 运行失败：{type(exc).__name__}: {exc}"
            warnings.append(fallback_reason)

    if source == "fallback":
        df, summary = _fallback_loop(
            prediction, mm, reason=fallback_reason or "mode=fallback"
        )
    else:
        # Build a summary from the Julia metadata and the actual loop.
        from .bh_curve import describe_curve

        phi = float(job["microstructure"]["magnetic_volume_fraction"] or 1.0)
        meta = julia_meta or {}
        summary = {
            "model_status": "conditional",
            "engine": "MicroMagnetic.jl",
            "engine_version": meta.get("package_version"),
            "mesh": job["mesh"],
            "shape": job["shape"],
            "julia": meta,
            "magnetic_volume_fraction": phi,
            "bh_metrics": describe_curve(df[["H_A_m", "B_T"]]),
            "warnings": [
                *warnings,
                "代表性子体积的微磁学结果不能直接代表完整 20 µm 颗粒；"
                "必须通过 RVE/网格收敛和实验校准外推。",
                *meta.get("warnings", []),
            ],
            "assumptions": [
                "LLG 阻尼动力学 + 给定材料参数",
                "网格尺度 5 nm 仅解析交换长度附近结构，不代表全颗粒",
                "未包含颗粒间磁相互作用（单颗粒/子体积模型）",
            ],
            "software": software_versions(),
            "created_utc": timestamp_utc(),
        }
    df = df.rename(columns={"M_phase_A_m": "M_A_m"})
    write_csv(df, out / "loop.csv")
    write_json(out / "summary.json", summary)
    return {
        "dataframe": df,
        "summary": summary,
        "paths": {
            "loop_csv": str(out / "loop.csv"),
            "summary_json": str(out / "summary.json"),
            "job_json": str(job_path),
        },
        "source": source,
    }
