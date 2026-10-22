"""End-to-end orchestration.

Every step writes its own summary and the final report is generated from those
summaries.  A step that cannot run because an input is missing returns a
``model_status='unavailable'`` record instead of substituting a fabricated value.
"""

from __future__ import annotations

import shutil
import time
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

from .bh_curve import describe_curve
from .calibration import calibrate
from .configs import (
    DemConfig,
    ExperimentConfig,
    MicromagneticsConfig,
    load_dem_config,
    load_experiment_config,
    load_micromagnetics_config,
    load_project,
)
from .dem_runner import load_structure, run_dem
from .dielectric import dielectric_breakdown
from .eddy_loss import (
    integrate_eddy_loss_over_distribution,
    interparticle_eddy_loss,
    separate_losses,
)
from .homogenization import TanhLoopParameters, homogenize, tanh_minor_loop
from .io_utils import (
    build_manifest,
    copy_if_exists,
    ensure_dir,
    make_run_id,
    new_run_dir,
    read_json,
    sha256_file,
    software_versions,
    timestamp_utc,
    to_jsonable,
    write_csv,
    write_json,
)
from .micromagnetics import MicromagneticsError, run_micromagnetics
from .microstructure import (
    electrical_network,
    magnetic_network,
    rve_convergence,
    structure_summary,
)
from .particle_generator import distribution_from_config, size_summary, write_particle_outputs
from .report import generate_report
from .schemas import ConfigError, PredictionConfig, load_config
from .uncertainty import uncertainty_analysis


def _run_directory(base: str | Path, run_id: str | None) -> tuple[Path, str]:
    base_path = Path(base)
    rid = run_id or make_run_id("run")
    candidate = base_path / rid
    if candidate.exists():
        rid = f"{rid}_{timestamp_utc().replace(':', '').replace('+00:00', 'Z')}"
    return new_run_dir(base_path, rid)


def _step(name: str, enabled: Iterable[str]) -> bool:
    return name in set(enabled)


def _json_write(path: Path, payload: Any) -> None:
    write_json(path, to_jsonable(payload))


def _load_experiment(path: str | None) -> tuple[ExperimentConfig | None, Path | None]:
    if path is None:
        return None, None
    p = Path(path)
    if not p.is_file():
        return None, p
    return load_experiment_config(p), p


def run_all(
    project_path: str | Path,
    *,
    steps: list[str] | None = None,
    run_id: str | None = None,
    log: Any = print,
) -> dict[str, Any]:
    project, project_file = load_project(project_path)
    pred_path = Path(project.prediction_config)
    prediction = load_config(pred_path)
    enabled = steps or list(project.steps)
    run_dir, rid = _run_directory(project.output_dir, run_id or project.run_id)
    log(f"[softmag] run {rid} -> {run_dir}")

    results: dict[str, Any] = {
        "run_id": rid,
        "run_dir": str(run_dir),
        "created_utc": timestamp_utc(),
        "steps_requested": enabled,
        "prediction": prediction.to_dict(),
        "warnings": [],
    }
    # ---- validate -----------------------------------------------------
    if _step("validate", enabled):
        contract = prediction.contract(run_id=rid)
        _json_write(run_dir / "input_contract.json", contract)
        _json_write(run_dir / "missing_parameters.json", prediction.missing_parameters())
        results["missing_parameters"] = prediction.missing_parameters()
        results["warnings"].extend(prediction.quality_warnings())
        results["warnings"].extend(contract.get("warnings", []))
        config_dir = ensure_dir(run_dir / "config_used")
        for src in (
            pred_path,
            Path(project.dem_config) if project.dem_config else None,
            Path(project.micromagnetics_config) if project.micromagnetics_config else None,
            Path(project.experiment_config) if project.experiment_config else None,
        ):
            if src and src.is_file():
                copy_if_exists(src, config_dir / src.name)
        _json_write(run_dir / "validate.json", {"missing_parameters": results["missing_parameters"], "warnings": results["warnings"]})
        log("[softmag] validate ok")

    # ---- particles ----------------------------------------------------
    dist = None
    particles_dir = ensure_dir(run_dir / "particles")
    if _step("particles", enabled):
        dist = distribution_from_config(prediction.particle)
        paths = write_particle_outputs(dist, particles_dir, n_sample=0)
        summary = size_summary(dist)
        results["particles"] = {"size_summary": summary, "paths": {k: str(v) for k, v in paths.items()}}
        _json_write(particles_dir / "summary.json", results["particles"])
        log("[softmag] particles ok")

    # ---- DEM ----------------------------------------------------------
    dem_dir = ensure_dir(run_dir / "dem")
    structure = None
    if _step("dem", enabled):
        if project.dem_config is None:
            raise ConfigError("project.dem_config 未设置，无法运行 DEM")
        if dist is None:
            dist = distribution_from_config(prediction.particle)
        dem_cfg = load_dem_config(project.dem_config)
        t0 = time.perf_counter()
        structure = run_dem(prediction, dem_cfg, dem_dir, distribution=dist)
        results["dem"] = {
            "n_particles": structure.n_particles,
            "n_contacts": int(len(structure.contacts)),
            "box_length_m": structure.box_length_m,
            "summary": structure.summary,
            "wall_time_s": float(time.perf_counter() - t0),
            "paths": {p.name: str(p) for p in dem_dir.glob("*")},
        }
        _json_write(dem_dir / "run_summary.json", results["dem"])
        log(f"[softmag] DEM ok: {structure.n_particles} particles, {len(structure.contacts)} contacts")

    if structure is None and _step("structure", enabled):
        structure = load_structure(dem_dir)
        results["dem"] = results.get("dem") or {
            "n_particles": structure.n_particles,
            "n_contacts": int(len(structure.contacts)),
            "box_length_m": structure.box_length_m,
            "summary": structure.summary,
        }

    # ---- structure ----------------------------------------------------
    structure_dir = ensure_dir(run_dir / "structure")
    st: dict[str, Any] = {}
    elec: dict[str, Any] = {}
    mag: dict[str, Any] = {}
    if _step("structure", enabled):
        if structure is None:
            raise ConfigError("structure 步骤需要先运行/提供 DEM 结构")
        st = structure_summary(
            structure,
            insulation_thickness_m=prediction.insulation.thickness_mean_m,
        )
        rng = np.random.default_rng(project.random_seed + 11)
        elec = electrical_network(
            structure,
            prediction.insulation,
            particle_resistivity_ohm_m=prediction.material.resistivity_ohm_m,
            rng=rng,
        )
        mag = magnetic_network(
            structure,
            particle_relative_permeability=prediction.material.initial_permeability,
            insulation_thickness_m=prediction.insulation.thickness_mean_m,
        )
        rve = rve_convergence(structure, rng=np.random.default_rng(project.random_seed + 12))
        st["electrical"] = elec
        st["magnetic"] = mag
        st["rve"] = rve
        results["structure"] = st
        _json_write(structure_dir / "summary.json", st)
        _json_write(structure_dir / "electrical_network.json", elec)
        _json_write(structure_dir / "magnetic_network.json", mag)
        _json_write(structure_dir / "rve.json", rve)
        log(f"[softmag] structure ok: mu_network={mag.get('effective_relative_permeability')}")

    # ---- micromagnetics ----------------------------------------------
    mm_summary: dict[str, Any] = {}
    mm_df: pd.DataFrame | None = None
    mm_dir = ensure_dir(run_dir / "hysteresis")
    if _step("micromagnetics", enabled):
        if project.micromagnetics_config is None:
            mm_summary = {"model_status": "unavailable", "reason": "未配置 micromagnetics_config"}
        else:
            mm_cfg = load_micromagnetics_config(project.micromagnetics_config)
            try:
                mm_result = run_micromagnetics(prediction, mm_cfg, mm_dir, run_id=rid)
                mm_summary = mm_result["summary"]
                mm_df = mm_result["dataframe"]
                results["micromagnetics"] = mm_summary
                log(f"[softmag] micromagnetics ok: {mm_summary.get('engine')}")
            except MicromagneticsError as exc:
                mm_summary = {
                    "model_status": "unavailable",
                    "reason": str(exc),
                    "warnings": ["微磁学步骤失败；后续均匀化只使用混合/网络模型。"],
                }
                results["micromagnetics"] = mm_summary
                results["warnings"].append(str(exc))
                log(f"[softmag] micromagnetics unavailable: {exc}")
        _json_write(mm_dir / "summary_pipeline.json", mm_summary)

    # Add B-H metrics from the actual loop if available.
    if mm_df is not None and not mm_df.empty:
        try:
            metrics = describe_curve(mm_df[["H_A_m", "B_T"]])
        except Exception as exc:  # noqa: BLE001
            metrics = {"model_status": "unavailable", "reason": str(exc)}
        mm_summary["bh_metrics"] = metrics
        results["micromagnetics"] = mm_summary
        mm_df.to_csv(mm_dir / "loop_used.csv", index=False)

    # ---- homogenize ---------------------------------------------------
    hom: dict[str, Any] = {}
    hom_dir = ensure_dir(run_dir / "homogenization")
    if _step("homogenize", enabled):
        phi, phi_source, phi_warnings = prediction.resolve_magnetic_volume_fraction()
        if phi is None:
            hom = {
                "model_status": "unavailable",
                "reason": "缺少磁性体积分数",
                "warnings": phi_warnings,
            }
        else:
            hom = homogenize(
                magnetic_volume_fraction=float(phi),
                Ms_A_m=prediction.material.Ms_A_m,
                particle_relative_permeability=prediction.material.initial_permeability,
                network=mag or None,
            )
            hom["magnetic_volume_fraction_source"] = phi_source
            hom["warnings"] = list(dict.fromkeys([*hom.get("warnings", []), *phi_warnings]))
            if mm_summary.get("model_status") == "fallback_reduced_order":
                hom["warnings"].append(
                    "回线来自 tanh fallback；均匀化磁导率只采用混合/网络模型，不采用 fallback 回线参数。"
                )
            if mm_df is not None and not mm_df.empty:
                write_csv(mm_df, hom_dir / "macro_bh.csv")
        results["homogenization"] = hom
        _json_write(hom_dir / "summary.json", hom)
        log(f"[softmag] homogenize ok: mu={hom.get('effective_relative_permeability')}")

    # ---- losses -------------------------------------------------------
    losses: dict[str, Any] = {}
    loss_dir = ensure_dir(run_dir / "losses")
    if _step("losses", enabled):
        f = prediction.excitation.frequency_hz
        B_peak, b_warn = prediction.effective_flux_peak_T()
        phi, phi_source, phi_warnings = prediction.resolve_magnetic_volume_fraction()
        warnings: list[str] = []
        if b_warn:
            warnings.append(b_warn)
        warnings.extend(phi_warnings)
        hysteresis = None
        minor_diag: dict[str, Any] = {}
        # The major-loop area at Hmax is NOT the loss at the excitation B peak.
        # Build a closed minor loop for the actual B_peak when the reduced-order
        # parameters are available.  A real MicroMagnetic major loop is not
        # silently truncated to represent a minor loop.
        params = mm_summary.get("parameters") or {}
        if (
            B_peak is not None
            and f is not None
            and phi is not None
            and {"Ms_A_m", "Hc_A_m", "a_A_m"}.issubset(params)
        ):
            try:
                minor_params = TanhLoopParameters(
                    Ms_A_m=float(params["Ms_A_m"]),
                    Hc_A_m=float(params["Hc_A_m"]),
                    a_A_m=float(params["a_A_m"]),
                )
                minor_df, minor_diag = tanh_minor_loop(
                    minor_params,
                    B_peak_T=float(B_peak),
                    magnetic_volume_fraction=float(phi),
                )
                area = float(minor_diag["loop_energy_density_J_m3_per_cycle"])
                hysteresis = area * float(f)
                write_csv(minor_df, loss_dir / "hysteresis_minor_loop.csv")
                warnings.extend(minor_diag.get("warnings", []))
            except Exception as exc:  # noqa: BLE001
                warnings.append(f"小回线构造失败：{type(exc).__name__}: {exc}")
        else:
            warnings.append(
                "缺少 B_peak 或 tanh 小回线参数；未用大回线面积冒充给定 B_peak 的磁滞损耗。"
            )
        if f is None:
            warnings.append("缺少频率；不能把回线面积转换为 W/m³。")
        eddy: dict[str, Any] = {"model_status": "unavailable", "reason": "缺少输入"}
        if (
            structure is not None
            and B_peak is not None
            and f is not None
            and prediction.material.resistivity_ohm_m is not None
            and phi is not None
        ):
            d = 2.0 * structure.radii
            volumes = (4.0 / 3.0) * np.pi * structure.radii**3
            eddy = integrate_eddy_loss_over_distribution(
                d,
                frequency_hz=float(f),
                B_peak_T=float(B_peak),
                resistivity_ohm_m=float(prediction.material.resistivity_ohm_m),
                weights=volumes,
                weight_basis="volume",
                magnetic_volume_fraction=float(phi),
                flux_concentration="none",
            )
        else:
            warnings.append("缺少结构/频率/B 峰值/电阻率/体积分数中的至少一项，不能计算颗粒涡流。")
        eddy_value = eddy.get("eddy_loss_sample_volume_average_W_m3") if eddy.get("model_status") != "unavailable" else None
        # Interparticle path only if the electrical network percolates.
        sigma_eff = (elec or {}).get("effective_conductivity_S_m")
        loop_len = None
        if structure is not None:
            d50 = float(np.median(2.0 * structure.radii))
            loop_len = min(structure.box_length_m, 10.0 * d50)
        inter = interparticle_eddy_loss(
            frequency_hz=float(f or 0.0),
            B_peak_T=float(B_peak or 0.0),
            effective_conductivity_S_m=sigma_eff,
            characteristic_loop_length_m=float(loop_len or 1e-6),
        )
        inter_value = inter.get("interparticle_eddy_loss_W_m3") if inter.get("model_status") != "unavailable" else None
        if inter.get("model_status") == "unavailable":
            warnings.extend(inter.get("warnings", []))
        else:
            warnings.extend(inter.get("warnings", []))
        known = [v for v in (hysteresis, eddy_value, inter_value) if v is not None]
        total = float(sum(known)) if known else None
        if hysteresis is None or eddy_value is None:
            warnings.append("总损耗只对可用分量求和；缺失分量未用零替代。")
        losses = {
            "model_status": "conditional" if known else "unavailable",
            "frequency_hz": f,
            "B_peak_T": B_peak,
            "hysteresis_loss_W_m3": hysteresis,
            "hysteresis_basis": (
                "normalized_closed_minor_loop_at_B_peak"
                if hysteresis is not None
                else None
            ),
            "hysteresis_minor_loop": minor_diag,
            "eddy_loss_W_m3": eddy_value,
            "interparticle_eddy_loss_W_m3": inter_value,
            "total_loss_W_m3": total,
            "total_is_partial": bool(hysteresis is None or eddy_value is None or inter_value is None),
            "eddy_loss": eddy,
            "interparticle_eddy": inter,
            "magnetic_volume_fraction": phi,
            "warnings": list(dict.fromkeys(warnings)),
            "assumptions": [
                "磁滞损耗 = 准静态回线面积 × 频率",
                "涡流损耗 = 孤立球粒径分布积分 + 可选颗粒间有效介质项",
                "三项未用单一 Steinmetz 公式替代；总损耗为可用项之和",
            ],
        }
        rows = [
            {"component": "hysteresis", "loss_W_m3": hysteresis},
            {"component": "eddy_particle", "loss_W_m3": eddy_value},
            {"component": "eddy_interparticle", "loss_W_m3": inter_value},
            {"component": "total", "loss_W_m3": total},
        ]
        write_csv(pd.DataFrame(rows), loss_dir / "components.csv")
        results["losses"] = losses
        _json_write(loss_dir / "summary.json", losses)
        log(f"[softmag] losses ok: total={total}")

    # ---- dielectric ---------------------------------------------------
    diel: dict[str, Any] = {}
    diel_dir = ensure_dir(run_dir / "dielectric")
    if _step("dielectric", enabled) and structure is not None:
        diel = dielectric_breakdown(
            structure,
            prediction.insulation,
            rng=np.random.default_rng(project.random_seed + 21),
            repeats=max(20, min(500, int(project.random_seed % 50 + 100))),
        )
        results["dielectric"] = diel
        _json_write(diel_dir / "summary.json", diel)
        log(f"[softmag] dielectric ok: {diel.get('model_status')}")

    # ---- calibration --------------------------------------------------
    cal: dict[str, Any] = {}
    cal_dir = ensure_dir(run_dir / "calibration")
    exp, exp_path = _load_experiment(project.experiment_config)
    if _step("calibrate", enabled) and exp is not None and exp_path is not None:
        cal = calibrate(
            prediction,
            exp,
            base_dir=exp_path.parent,
            run_id=rid,
        )
        results["calibration"] = cal
        _json_write(cal_dir / "summary.json", cal)
        # Attach the Steinmetz fit to the loss summary for the report.
        if losses and cal.get("fits", {}).get("steinmetz"):
            losses["steinmetz"] = cal["fits"]["steinmetz"]
            if (losses["steinmetz"].get("data_quality") == "synthetic_placeholder"):
                losses.setdefault("warnings", []).append(
                    "Steinmetz 拟合使用 synthetic placeholder 损耗数据；只用于流程演示，"
                    "不得用来替代第 5 节的分量损耗模型或做定量预测。"
                )
            _json_write(loss_dir / "summary.json", losses)
        log(f"[softmag] calibrate ok: {cal.get('model_status')}")
    elif _step("calibrate", enabled):
        results["calibration"] = {
            "model_status": "unavailable",
            "reason": f"实验配置文件不可用：{project.experiment_config}",
        }
        _json_write(cal_dir / "summary.json", results["calibration"])

    # ---- uncertainty --------------------------------------------------
    unc: dict[str, Any] = {}
    unc_dir = ensure_dir(run_dir / "uncertainty")
    if _step("uncertainty", enabled):
        bootstrap = 200
        if exp is not None:
            bootstrap = int(exp.calibration.bootstrap_samples)
        unc = uncertainty_analysis(
            prediction,
            homogenization=hom or None,
            losses=losses or None,
            dielectric=diel or None,
            n_samples=max(50, min(500, bootstrap)),
            random_seed=project.random_seed + 31,
        )
        results["uncertainty"] = unc
        _json_write(unc_dir / "summary.json", unc)
        log(f"[softmag] uncertainty ok: {unc.get('model_status')}")

    # ---- manifest + report -------------------------------------------
    config_paths: list[Path] = [project_file, pred_path]
    for value in (
        project.dem_config,
        project.micromagnetics_config,
        project.experiment_config,
    ):
        if value:
            config_paths.append(Path(value))
    manifest = build_manifest(
        run_id=rid,
        config_paths=config_paths,
        config=prediction.to_dict(),
        random_seed=project.random_seed,
        notes=project.notes,
        extra={
            "steps_requested": enabled,
            "input_contract_sha256": sha256_file(run_dir / "input_contract.json")
            if (run_dir / "input_contract.json").is_file()
            else None,
            "model_status": {
                "micromagnetics": (results.get("micromagnetics") or {}).get("model_status"),
                "homogenization": (results.get("homogenization") or {}).get("model_status"),
                "losses": (results.get("losses") or {}).get("model_status"),
                "dielectric": (results.get("dielectric") or {}).get("model_status"),
                "calibration": (results.get("calibration") or {}).get("model_status"),
                "uncertainty": (results.get("uncertainty") or {}).get("model_status"),
            },
        },
    )
    # manifest wants all config paths to exist; filter None already handled.
    results["manifest"] = manifest
    _json_write(run_dir / "manifest.json", manifest)
    # Save a machine-readable whole-run record before the Markdown report.
    _json_write(run_dir / "run_results.json", results)

    if _step("report", enabled):
        report_paths = generate_report(results, outdir=ensure_dir(run_dir / "reports"))
        results["report"] = report_paths
        # Rewrite the run record with the report paths.
        _json_write(run_dir / "run_results.json", results)
        log(f"[softmag] report -> {report_paths['markdown']}")

    return results
