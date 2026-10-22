"""Monte Carlo propagation of input uncertainty.

Only uncertainties that are actually present in ``material.uncertainty`` are
sampled.  Missing uncertainty entries are **not** silently filled with invented
values.  The module also reports a small scenario sweep for particle diameter and
flux density, because those inputs often lack a defensible probability
distribution; scenarios are labelled separately from probability intervals.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
from scipy.stats import spearmanr

from .eddy_loss import eddy_loss_density_sphere
from .homogenization import mixing_effective_permeability
from .io_utils import timestamp_utc
from .schemas import PredictionConfig


def _stats(values: list[float], *, prefix: str = "") -> dict[str, Any]:
    arr = np.asarray([v for v in values if np.isfinite(v)], dtype=float)
    if arr.size == 0:
        return {"model_status": "unavailable", "reason": "没有有效样本"}
    return {
        "model_status": "conditional",
        f"{prefix}n": int(arr.size),
        f"{prefix}mean": float(np.mean(arr)),
        f"{prefix}std": float(np.std(arr, ddof=1)) if arr.size > 1 else 0.0,
        f"{prefix}p05": float(np.percentile(arr, 5)),
        f"{prefix}p50": float(np.percentile(arr, 50)),
        f"{prefix}p95": float(np.percentile(arr, 95)),
        f"{prefix}min": float(np.min(arr)),
        f"{prefix}max": float(np.max(arr)),
    }


def _sample_positive(
    nominal: float,
    relative_std: float,
    rng: np.random.Generator,
    n: int,
) -> np.ndarray:
    if relative_std <= 0 or nominal <= 0:
        return np.full(n, float(nominal))
    # Lognormal with requested relative standard deviation, mean preserved.
    sigma = math.sqrt(math.log(1.0 + relative_std**2))
    return nominal * np.exp(sigma * rng.standard_normal(n) - 0.5 * sigma**2)


def uncertainty_analysis(
    prediction: PredictionConfig,
    *,
    homogenization: dict[str, Any] | None = None,
    losses: dict[str, Any] | None = None,
    dielectric: dict[str, Any] | None = None,
    n_samples: int = 200,
    random_seed: int | None = None,
) -> dict[str, Any]:
    """Propagate material uncertainty to selected engineering outputs.

    Scope: effective permeability (mixing model), isolated-particle eddy loss and
    the reported hysteresis loss.  It does **not** rerun DEM or MicroMagnetic.jl,
    so contact-network variability and microstructure uncertainty are absent.
    """
    n = max(10, int(n_samples))
    rng = np.random.default_rng(
        random_seed if random_seed is not None else prediction.compaction.random_seed
    )
    material = prediction.material
    unc = dict(material.uncertainty or {})
    phi, phi_source, phi_warnings = prediction.resolve_magnetic_volume_fraction()
    if phi is None:
        return {
            "model_status": "unavailable",
            "reason": "缺少磁性体积分数",
            "warnings": phi_warnings,
        }
    B_peak, b_warn = prediction.effective_flux_peak_T()
    f = prediction.excitation.frequency_hz
    d50 = prediction.particle.d50_m
    warnings: list[str] = []
    if b_warn:
        warnings.append(b_warn)
    if material.Ms_A_m is None:
        return {"model_status": "unavailable", "reason": "缺少 material.Ms_A_m"}
    if d50 is None:
        warnings.append("缺少 d50；不传播涡流损耗的粒径不确定性，仅做磁导率/磁滞传播。")
    if f is None:
        warnings.append("缺少频率；涡流损耗不参与传播。")
    if B_peak is None:
        warnings.append("缺少峰值磁通密度；涡流损耗不参与传播。")
    have_mu = material.initial_permeability is not None
    mu0 = float(material.initial_permeability or 0.0)
    if not have_mu:
        warnings.append("缺少颗粒本征初始磁导率；磁导率传播不可用。")
    sigma_mu = float(unc.get("initial_permeability", 0.0))
    if sigma_mu == 0:
        warnings.append(
            "material.uncertainty 未给出 initial_permeability；磁导率分布仅反映混合模型输入，"
            "不包含本征磁导率认知不确定性。"
        )
    # Sample material parameters only where uncertainty is declared.
    Ms = _sample_positive(material.Ms_A_m, float(unc.get("Ms_A_m", 0.0)), rng, n)
    rho = _sample_positive(
        material.resistivity_ohm_m or 0.0,
        float(unc.get("resistivity_ohm_m", 0.0)),
        rng,
        n,
    )
    mu_p = _sample_positive(mu0, sigma_mu, rng, n)
    # Keep the nominal/per-sample mapping for sensitivity.
    mu_eff: list[float] = []
    eddy: list[float] = []
    hyst: list[float] = []
    for k in range(n):
        if have_mu:
            mix = mixing_effective_permeability(
                float(phi),
                float(mu_p[k]),
                model="bruggeman",
            )
            value = mix.get("effective_relative_permeability")
            if value is not None:
                mu_eff.append(float(value))
        if d50 is not None and f is not None and B_peak is not None and rho[k] > 0:
            e = eddy_loss_density_sphere(
                diameter_m=float(d50),
                frequency_hz=float(f),
                B_peak_T=float(B_peak),
                resistivity_ohm_m=float(rho[k]),
            )
            eddy.append(float(e["eddy_loss_density_W_m3"]) * float(phi))
    # Hysteresis loss scaling: loop area scales with Ms if Hc/a are fixed.
    hyst0 = None
    if losses:
        hyst0 = losses.get("hysteresis_loss_W_m3")
    if hyst0 is not None:
        ratio = Ms / float(material.Ms_A_m)
        hyst = (float(hyst0) * ratio).tolist()
    else:
        warnings.append("没有磁滞损耗基线；不传播磁滞损耗。")
    outputs: dict[str, Any] = {
        "effective_relative_permeability": _stats(mu_eff),
        "eddy_loss_W_m3": _stats(eddy),
        "hysteresis_loss_W_m3": _stats(hyst),
    }
    # Bootstrap the sample breakdown median from the dielectric raw draws.
    v_raw = (dielectric or {}).get("raw_breakdown_voltage_V")
    if v_raw:
        arr = np.asarray(v_raw, dtype=float)
        med = [
            float(np.median(rng.choice(arr, size=arr.size, replace=True)))
            for _ in range(min(1000, max(200, n * 2)))
        ]
        outputs["breakdown_voltage_median_V"] = _stats(med)
    else:
        outputs["breakdown_voltage_median_V"] = {
            "model_status": "unavailable",
            "reason": "耐压结果没有原始样本，无法 bootstrap",
        }
    # Sensitivity: Spearman correlation of sampled inputs with effective mu and eddy loss.
    inputs = {"Ms_A_m": Ms, "resistivity_ohm_m": rho, "initial_permeability": mu_p}
    out_arrays = {
        "effective_relative_permeability": np.asarray(mu_eff, dtype=float)
        if len(mu_eff) == n
        else None,
        "eddy_loss_W_m3": np.asarray(eddy, dtype=float) if len(eddy) == n else None,
    }
    sensitivity: dict[str, Any] = {}
    for name, arr in out_arrays.items():
        if arr is None or np.std(arr) == 0:
            sensitivity[name] = {}
            continue
        row: dict[str, float] = {}
        for pname, parr in inputs.items():
            if np.std(parr) == 0:
                continue
            rho_s, _ = spearmanr(parr, arr)
            if np.isfinite(rho_s):
                row[pname] = float(rho_s)
        sensitivity[name] = dict(
            sorted(row.items(), key=lambda kv: abs(kv[1]), reverse=True)
        )
    # Scenario sweep: not a probability interval.
    scenarios: list[dict[str, Any]] = []
    if d50 is not None and f is not None and B_peak is not None and material.resistivity_ohm_m:
        for factor in (0.8, 1.0, 1.2):
            e = eddy_loss_density_sphere(
                diameter_m=float(d50) * factor,
                frequency_hz=float(f),
                B_peak_T=float(B_peak),
                resistivity_ohm_m=float(material.resistivity_ohm_m),
            )
            scenarios.append(
                {
                    "d50_factor": factor,
                    "d50_m": float(d50) * factor,
                    "eddy_loss_W_m3": float(e["eddy_loss_density_W_m3"]) * float(phi),
                }
            )
    warnings.extend(
        [
            "Monte Carlo 只传播 material.uncertainty 中声明的参数；未声明的参数保持名义值。",
            "未重跑 DEM/微磁学；接触网络与微结构认知不确定性不在本区间内。",
            "概率区间是输入假设下的条件区间，不是实验置信区间。",
            "粒径/磁通情景扫描与概率分布分开报告。",
        ]
    )
    if sigma_mu > 0:
        warnings.append(
            "initial_permeability 的相对不确定度来自 material.uncertainty；该值本身也需实验依据。"
        )
    return {
        "model_status": "conditional",
        "method": "monte_carlo_declared_input_uncertainty",
        "n_samples": n,
        "magnetic_volume_fraction": float(phi),
        "magnetic_volume_fraction_source": phi_source,
        "declared_relative_uncertainty": unc,
        "outputs": outputs,
        "sensitivity_spearman": sensitivity,
        "diameter_scenarios": scenarios,
        "warnings": list(dict.fromkeys([*warnings, *phi_warnings])),
        "assumptions": [
            "未声明不确定度的参数视为确定值（不是“零不确定度”的科学结论）",
            "正参数采用均值保持的对数正态采样",
            "磁导率采用 Bruggeman 混合模型重新计算",
            "涡流损耗采用 d50 孤立球基线并乘以磁性体积分数",
            "磁滞损耗按 Ms 线性缩放，Hc/a 固定",
        ],
        "created_utc": timestamp_utc(),
    }
