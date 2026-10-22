"""Loss models: classical eddy-current baseline and Steinmetz fitting.

The eddy-current calculation here is deliberately conservative and explicit:

* each particle is an isolated conducting sphere in a uniform sinusoidal flux
  density (low-frequency limit),
* the particle-size distribution is integrated with volume weights,
* the low-frequency solution is *not* applied silently when the skin depth is
  comparable to the particle radius,
* interparticle eddy paths are only estimated when the electrical contact
  network actually percolates.

None of these estimates replaces a measured loss map.  Total loss is reported as
a sum of separately labelled terms, never as a single unqualified number.
"""

from __future__ import annotations

import math
from typing import Any, Literal

import numpy as np
import pandas as pd

from .units import MU0, skin_depth_m


# ---------------------------------------------------------------------------
# Isolated-sphere classical eddy current baseline
# ---------------------------------------------------------------------------
def eddy_loss_density_sphere(
    *,
    diameter_m: float,
    frequency_hz: float,
    B_peak_T: float,
    resistivity_ohm_m: float,
    relative_permeability: float = 1.0,
    shape_factor: float = 1.0,
) -> dict[str, Any]:
    r"""Classical low-frequency eddy loss density in a sphere.

    .. math::

       P_e = \frac{\pi^2 d^2 f^2 B_p^2}{20 \rho}

    The coefficient 20 is the sphere result in the low-frequency limit.  It is
    an *isolated particle* baseline, not a powder-core loss model.  The returned
    skin-depth ratio tells the caller whether the low-frequency assumption is
    credible.
    """
    d = float(diameter_m)
    f = float(frequency_hz)
    B = float(B_peak_T)
    rho = float(resistivity_ohm_m)
    if d <= 0 or f < 0 or B < 0 or rho <= 0:
        raise ValueError("diameter 与 resistivity 必须为正，f/B 不能为负")
    if shape_factor <= 0:
        raise ValueError("shape_factor 必须为正")
    p = (math.pi**2) * d**2 * f**2 * B**2 / (20.0 * rho) * float(shape_factor)
    delta = skin_depth_m(max(f, 1e-30), rho, max(relative_permeability, 1e-12))
    ratio = d / (2.0 * delta) if delta > 0 else float("inf")
    warnings: list[str] = []
    status = "conditional"
    if f == 0:
        status = "direct"
    if ratio > 1.0:
        warnings.append(
            "颗粒直径与集肤深度之比 d/(2δ) > 1；低频涡流公式已超出适用范围，"
            "结果只能作为量级下界/趋势，不能当作定量预测。"
        )
        status = "outside_validity"
    elif ratio > 0.5:
        warnings.append(
            "d/(2δ) 接近 1；低频涡流近似开始失效，需用频域 Maxwell 或实测损耗修正。"
        )
        status = "needs_calibration"
    return {
        "model_status": status,
        "method": "isolated_sphere_low_frequency",
        "eddy_loss_density_W_m3": float(p),
        "diameter_m": d,
        "frequency_hz": f,
        "B_peak_T": B,
        "resistivity_ohm_m": rho,
        "skin_depth_m": float(delta),
        "diameter_over_2skin": float(ratio),
        "warnings": warnings,
        "assumptions": [
            "孤立球体、均匀正弦磁通、低频极限",
            "颗粒内部磁导率为给定值；未求解颗粒间磁通拥挤",
            "形状因子仅作经验修正，不是严格椭球解",
        ],
    }


def integrate_eddy_loss_over_distribution(
    diameters_m: np.ndarray,
    *,
    frequency_hz: float,
    B_peak_T: float,
    resistivity_ohm_m: float,
    weights: np.ndarray | None = None,
    weight_basis: Literal["number", "volume"] = "volume",
    relative_permeability: float = 1.0,
    magnetic_volume_fraction: float | None = None,
    flux_concentration: Literal["none", "by_phi"] = "none",
    shape_factor: float = 1.0,
) -> dict[str, Any]:
    """Integrate the isolated-sphere eddy baseline over a size distribution.

    ``weights`` are interpreted in ``weight_basis``.  Number weights are
    converted to volume weights using ``d^3`` before averaging.  The result is
    the loss per unit *sample* volume when ``magnetic_volume_fraction`` is
    supplied (the particle-volume-weighted loss is multiplied by the magnetic
    volume fraction); otherwise it is the loss per unit particle volume.

    ``flux_concentration='by_phi'`` evaluates the local particle flux as
    ``B_peak/phi``.  This is an upper-bound sensitivity scenario, not a
    validated correction.
    """
    d = np.asarray(diameters_m, dtype=float).reshape(-1)
    if d.size == 0 or np.any(d <= 0):
        raise ValueError("diameters_m 必须是非空正数组")
    if weights is None:
        w = d**3 if weight_basis == "volume" else np.ones_like(d)
    else:
        w = np.asarray(weights, dtype=float).reshape(-1)
        if w.size != d.size:
            raise ValueError("weights 与 diameters_m 长度必须一致")
        if np.any(w < 0) or not np.all(np.isfinite(w)) or w.sum() <= 0:
            raise ValueError("weights 必须是非负有限数且总和 > 0")
        if weight_basis == "number":
            w = w * d**3
    w = w / w.sum()
    order = np.argsort(d)
    d = d[order]
    w = w[order]
    cdf = np.cumsum(w)
    if flux_concentration == "by_phi":
        if magnetic_volume_fraction is None or magnetic_volume_fraction <= 0:
            raise ValueError("flux_concentration='by_phi' 需要正 magnetic_volume_fraction")
        B_local = float(B_peak_T) / float(magnetic_volume_fraction)
        conc_warning = (
            "局部磁通密度按 B_sample/phi 放大；这是敏感性上界，不是已验证的磁通拥挤模型。"
        )
    else:
        B_local = float(B_peak_T)
        conc_warning = None
    per_particle = np.array(
        [
            eddy_loss_density_sphere(
                diameter_m=float(di),
                frequency_hz=frequency_hz,
                B_peak_T=B_local,
                resistivity_ohm_m=resistivity_ohm_m,
                relative_permeability=relative_permeability,
                shape_factor=shape_factor,
            )["eddy_loss_density_W_m3"]
            for di in d
        ]
    )
    particle_volume_average = float(np.sum(w * per_particle))
    sample_average = (
        particle_volume_average * float(magnetic_volume_fraction)
        if magnetic_volume_fraction is not None
        else None
    )
    # Applicability uses the largest particle and the local flux scenario.
    worst = eddy_loss_density_sphere(
        diameter_m=float(np.max(d)),
        frequency_hz=frequency_hz,
        B_peak_T=B_local,
        resistivity_ohm_m=resistivity_ohm_m,
        relative_permeability=relative_permeability,
        shape_factor=shape_factor,
    )
    d50 = float(np.interp(0.5, cdf, d))
    warnings = list(worst["warnings"])
    if conc_warning:
        warnings.append(conc_warning)
    status = worst["model_status"]
    if status == "outside_validity":
        warnings.append(
            "至少最大颗粒超出低频涡流适用范围；分布积分结果只能作为下界/趋势。"
        )
    return {
        "model_status": status,
        "method": "isolated_sphere_size_distribution_integral",
        "eddy_loss_particle_volume_average_W_m3": particle_volume_average,
        "eddy_loss_sample_volume_average_W_m3": sample_average,
        "magnetic_volume_fraction": magnetic_volume_fraction,
        "n_particles": int(d.size),
        "diameter_m": {
            "d10": float(np.interp(0.1, cdf, d)),
            "d50": d50,
            "d90": float(np.interp(0.9, cdf, d)),
            "mean": float(np.sum(w * d)),
            "max": float(np.max(d)),
        },
        "B_local_T": B_local,
        "B_sample_T": float(B_peak_T),
        "skin_depth_m": worst["skin_depth_m"],
        "diameter_over_2skin": worst["diameter_over_2skin"],
        "warnings": warnings,
        "assumptions": [
            "每个颗粒按孤立球处理",
            "粒径分布按体积权重积分",
            "样品平均损耗 = 颗粒体积平均损耗 * 磁性体积分数",
            *worst["assumptions"],
        ],
    }


# ---------------------------------------------------------------------------
# Interparticle eddy-current path
# ---------------------------------------------------------------------------
def interparticle_eddy_loss(
    *,
    frequency_hz: float,
    B_peak_T: float,
    effective_conductivity_S_m: float | None,
    characteristic_loop_length_m: float,
    model_status: str = "conditional",
) -> dict[str, Any]:
    r"""Rough interparticle eddy loss estimate.

    Uses the same low-frequency scaling with the compact's effective conductivity
    and a characteristic current-loop length:

    .. math::

       P \sim \frac{1}{20}\sigma_\mathrm{eff}\omega^2 B_p^2 L_\mathrm{loop}^2

    This is an order-of-magnitude estimate.  If the contact network does not
    percolate, no DC path exists and the model returns zero with an explicit
    explanation; local capacitive/coating paths are outside this term.
    """
    if frequency_hz < 0:
        raise ValueError("frequency_hz 不能为负")
    if characteristic_loop_length_m <= 0:
        raise ValueError("characteristic_loop_length_m 必须为正")
    if effective_conductivity_S_m is None:
        return {
            "model_status": "unavailable",
            "method": "effective_medium_eddy_estimate",
            "interparticle_eddy_loss_W_m3": None,
            "reason": "电网络不逾渗或未求解，不能给出颗粒间涡流路径损耗",
            "warnings": [
                "绝缘良好的粉芯在 DC 下不导通，但这不排除高频容性/局部缺陷电流；"
                "本项未计入那些机制。"
            ],
        }
    sigma = float(effective_conductivity_S_m)
    if sigma < 0:
        raise ValueError("effective_conductivity_S_m 不能为负")
    omega = 2.0 * math.pi * float(frequency_hz)
    p = (1.0 / 20.0) * sigma * omega**2 * float(B_peak_T) ** 2 * characteristic_loop_length_m**2
    return {
        "model_status": model_status,
        "method": "effective_medium_eddy_estimate",
        "interparticle_eddy_loss_W_m3": float(p),
        "effective_conductivity_S_m": sigma,
        "characteristic_loop_length_m": float(characteristic_loop_length_m),
        "frequency_hz": float(frequency_hz),
        "B_peak_T": float(B_peak_T),
        "warnings": [
            "颗粒间涡流损耗为量级估计；特征回路长度是假设值，必须做敏感性分析。",
            "未求解三维频域电磁场，也未计入绝缘层电容耦合。",
        ],
        "assumptions": [
            "用有效电导率和单一特征回路长度代表颗粒间电流路径",
            "低频涡流标度关系",
        ],
    }


# ---------------------------------------------------------------------------
# Loss separation
# ---------------------------------------------------------------------------
def separate_losses(
    *,
    total_loss_W_m3: float,
    hysteresis_loss_W_m3: float | None,
    eddy_loss_W_m3: float | None,
    frequency_hz: float,
) -> dict[str, Any]:
    """Report hysteresis + eddy + residual/excess with an explicit residual."""
    warnings: list[str] = []
    if total_loss_W_m3 < 0:
        raise ValueError("total_loss_W_m3 不能为负")
    h = float(hysteresis_loss_W_m3) if hysteresis_loss_W_m3 is not None else None
    e = float(eddy_loss_W_m3) if eddy_loss_W_m3 is not None else None
    excess = None
    if h is not None and e is not None:
        excess = float(total_loss_W_m3 - h - e)
        if excess < 0:
            warnings.append(
                "总损耗小于磁滞+涡流基线之和；可能源于测量/单位/频率不一致，"
                "或基线模型不适用。未将负残差强行归零。"
            )
    return {
        "model_status": "conditional",
        "total_loss_W_m3": float(total_loss_W_m3),
        "hysteresis_loss_W_m3": h,
        "eddy_loss_W_m3": e,
        "excess_loss_W_m3": excess,
        "frequency_hz": float(frequency_hz),
        "warnings": warnings,
        "assumptions": [
            "损耗按磁滞 + 涡流 + 残差分离",
            "残差可能包含过剩损耗、测量误差和未建模机制，不能自动解释为某一物理过程",
        ],
    }


# ---------------------------------------------------------------------------
# Steinmetz fitting
# ---------------------------------------------------------------------------
def steinmetz_fit(
    df: pd.DataFrame,
    *,
    frequency_col: str = "frequency_hz",
    flux_col: str = "B_peak_T",
    loss_col: str = "loss_W_m3",
    density_kg_m3: float | None = None,
    min_points: int = 3,
) -> dict[str, Any]:
    """Fit ``P = k f^alpha B^beta`` by ordinary least squares in log space.

    ``loss_W_m3`` is required; if the CSV reports specific loss in W/kg, pass
    ``density_kg_m3`` and provide ``loss_W_kg`` as ``loss_col``.  The fit is a
    descriptive loss map.  Extrapolation outside the measured (f, B) box is
    flagged.
    """
    for col in (frequency_col, flux_col, loss_col):
        if col not in df.columns:
            raise ValueError(f"Steinmetz 数据缺少列 {col!r}；实际列：{list(df.columns)}")
    data = df[[frequency_col, flux_col, loss_col]].copy()
    data = data.replace([np.inf, -np.inf], np.nan).dropna()
    data = data[(data[frequency_col] > 0) & (data[flux_col] > 0) & (data[loss_col] > 0)]
    loss_is_specific = loss_col.lower().endswith("_w_kg") or "kg" in loss_col.lower()
    if loss_is_specific:
        if density_kg_m3 is None or density_kg_m3 <= 0:
            raise ValueError("loss_col 是 W/kg 时必须提供正 density_kg_m3 转为 W/m^3")
        data["loss_W_m3"] = data[loss_col] * float(density_kg_m3)
    else:
        data["loss_W_m3"] = data[loss_col]
    if len(data) < min_points:
        return {
            "model_status": "unavailable",
            "reason": f"有效数据点不足（{len(data)} < {min_points}）",
            "n_points": int(len(data)),
            "warnings": [],
        }
    X = np.column_stack(
        [
            np.ones(len(data)),
            np.log(data[frequency_col].to_numpy(float)),
            np.log(data[flux_col].to_numpy(float)),
        ]
    )
    y = np.log(data["loss_W_m3"].to_numpy(float))
    coeff, *_ = np.linalg.lstsq(X, y, rcond=None)
    pred = X @ coeff
    resid = y - pred
    ss_res = float(np.sum(resid**2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 1.0
    k = float(math.exp(coeff[0]))
    alpha = float(coeff[1])
    beta = float(coeff[2])
    warnings: list[str] = []
    if r2 < 0.9:
        warnings.append("Steinmetz 拟合 R^2 < 0.9；数据可能跨多个损耗机制或含异常点。")
    if len(data) < 6:
        warnings.append("数据点较少；拟合参数不确定度很大，不应外推。")
    fmin, fmax = float(data[frequency_col].min()), float(data[frequency_col].max())
    bmin, bmax = float(data[flux_col].min()), float(data[flux_col].max())
    return {
        "model_status": "conditional",
        "method": "steinmetz_log_ols",
        "k": k,
        "alpha": alpha,
        "beta": beta,
        "formula": "P_W_m3 = k * f_hz^alpha * B_peak_T^beta",
        "r2": r2,
        "rmse_log": float(math.sqrt(np.mean(resid**2))),
        "n_points": int(len(data)),
        "fit_domain": {
            "frequency_hz": [fmin, fmax],
            "B_peak_T": [bmin, bmax],
        },
        "warnings": warnings,
        "assumptions": [
            "损耗满足幂律分离 P=k f^alpha B^beta",
            "同一材料状态、同一波形、同一温度范围内的参数",
            "外推到拟合域外不保证有效",
        ],
    }


def steinmetz_predict(
    fit: dict[str, Any], *, frequency_hz: float, B_peak_T: float
) -> dict[str, Any]:
    if fit.get("model_status") == "unavailable" or "k" not in fit:
        return {
            "model_status": "unavailable",
            "loss_W_m3": None,
            "reason": "Steinmetz 拟合不可用",
        }
    if frequency_hz <= 0 or B_peak_T < 0:
        raise ValueError("frequency 必须为正，B 不能为负")
    p = (
        float(fit["k"])
        * float(frequency_hz) ** float(fit["alpha"])
        * float(B_peak_T) ** float(fit["beta"])
    )
    warnings = list(fit.get("warnings", []))
    dom = fit.get("fit_domain", {})
    fr = dom.get("frequency_hz")
    br = dom.get("B_peak_T")
    if fr and not (fr[0] <= frequency_hz <= fr[1]):
        warnings.append("预测频率超出 Steinmetz 拟合域。")
    if br and not (br[0] <= B_peak_T <= br[1]):
        warnings.append("预测 B 峰值超出 Steinmetz 拟合域。")
    return {
        "model_status": "conditional",
        "loss_W_m3": float(p),
        "frequency_hz": float(frequency_hz),
        "B_peak_T": float(B_peak_T),
        "warnings": warnings,
    }


def load_loss_csv(path: str | Any) -> pd.DataFrame:
    df = pd.read_csv(path)
    aliases = {
        "f": "frequency_hz",
        "frequency": "frequency_hz",
        "B": "B_peak_T",
        "B_peak": "B_peak_T",
        "Bm_T": "B_peak_T",
        "P": "loss_W_m3",
        "loss": "loss_W_m3",
    }
    return df.rename(columns={k: v for k, v in aliases.items() if k in df.columns})
