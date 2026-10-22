"""Statistical dielectric breakdown on the DEM contact network.

The model does **not** equate the largest local electric field with the breakdown
voltage.  Instead:

1. every candidate inter-particle contact gets a random coating thickness and a
   random coating breakdown voltage (Weibull scatter, optional defects),
2. for one realisation, the sample breakdown voltage is the *minimum over all
   conducting paths of the maximum edge threshold* (a minimax/bottleneck path),
3. Monte Carlo repetitions produce a breakdown-voltage distribution, which is
   summarised by a two-parameter Weibull fit.

The result is a network-statistical model.  It does not resolve local field
enhancement at asperities, space charge, partial discharge or thermal runaway.
Those mechanisms must be added or calibrated before quantitative use.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np
import pandas as pd
from scipy.stats import weibull_min

from .dem_runner import Structure
from .microstructure import electrode_indices, union_find_labels
from .schemas import InsulationParameters


# ---------------------------------------------------------------------------
# Distributions
# ---------------------------------------------------------------------------
def sample_coating_thickness(
    n: int,
    insulation: InsulationParameters,
    rng: np.random.Generator,
) -> tuple[np.ndarray, list[str]]:
    mean = insulation.thickness_mean_m
    if mean is None or mean <= 0:
        raise ValueError("缺少 insulation.thickness_mean_m")
    if n <= 0:
        raise ValueError("n 必须为正")
    std = insulation.thickness_std_m
    kind = insulation.thickness_distribution
    warnings: list[str] = []
    if kind == "constant" or not std:
        return np.full(n, float(mean)), warnings
    if std >= mean:
        raise ValueError("thickness_std_m 必须小于 thickness_mean_m")
    if kind == "normal":
        values = rng.normal(mean, std, size=n)
        floor = max(0.05 * mean, 1e-12)
        n_clip = int(np.sum(values < floor))
        values = np.maximum(values, floor)
        if n_clip:
            warnings.append(
                f"{n_clip} 个正态厚度样本被截断到 {floor:.3e} m；"
                "截断正态不是严格的正态分布。"
            )
        return values, warnings
    if kind == "lognormal":
        sigma_log = math.sqrt(math.log(1.0 + (std / mean) ** 2))
        mu_log = math.log(mean) - 0.5 * sigma_log**2
        return rng.lognormal(mu_log, sigma_log, size=n), warnings
    raise ValueError(f"未知厚度分布 {kind!r}")


def sample_edge_breakdown_voltage(
    *,
    n_edges: int,
    insulation: InsulationParameters,
    rng: np.random.Generator,
    weibull_shape: float = 8.0,
    defect_scale_range: tuple[float, float] = (0.05, 0.5),
) -> dict[str, Any]:
    """Sample coating thicknesses and edge breakdown voltages.

    The local coating breakdown field follows a Weibull distribution with scale
    ``insulation.breakdown_field_V_m`` and the requested shape; the edge voltage
    is ``E_i * t_i``.  A defect multiplies the voltage by a random factor in
    ``defect_scale_range``.  All of these are model assumptions.
    """
    if insulation.breakdown_field_V_m is None:
        raise ValueError("缺少 insulation.breakdown_field_V_m")
    if weibull_shape <= 0:
        raise ValueError("weibull_shape 必须为正")
    thickness, warnings = sample_coating_thickness(
        n_edges, insulation, rng
    )
    # Weibull with scale E_bd: E = E_bd * (-ln U)^(1/k).
    u = np.clip(rng.random(n_edges), 1e-300, 1.0 - 1e-15)
    E = float(insulation.breakdown_field_V_m) * (-np.log(u)) ** (1.0 / weibull_shape)
    p_defect = float(insulation.defect_probability or 0.0)
    defect = rng.random(n_edges) < p_defect
    if np.any(defect):
        lo, hi = defect_scale_range
        factor = rng.uniform(float(lo), float(hi), size=n_edges)
        E = np.where(defect, E * factor, E)
        warnings.append(
            f"{int(defect.sum())} 个接触被随机缺陷标记；缺陷电压折减为 "
            f"{lo:.2f}..{hi:.2f} 倍。缺陷模型未与显微缺陷统计校准。"
        )
    V = E * thickness
    return {
        "thickness_m": thickness,
        "breakdown_field_V_m": E,
        "breakdown_voltage_V": V,
        "is_defect": defect,
        "warnings": warnings,
    }


# ---------------------------------------------------------------------------
# Minimax path / network connectivity
# ---------------------------------------------------------------------------
def _connected_at_threshold(
    n: int,
    i: np.ndarray,
    j: np.ndarray,
    threshold: float,
    V: np.ndarray,
    electrodes_a: np.ndarray,
    electrodes_b: np.ndarray,
) -> bool:
    keep = V <= threshold + 1e-15
    labels = union_find_labels(n, i[keep], j[keep])
    if electrodes_a.size == 0 or electrodes_b.size == 0:
        return False
    return bool(set(labels[electrodes_a]).intersection(set(labels[electrodes_b])))


def minimax_breakdown_voltage(
    n: int,
    i: np.ndarray,
    j: np.ndarray,
    edge_voltage_V: np.ndarray,
    electrodes_a: np.ndarray,
    electrodes_b: np.ndarray,
) -> dict[str, Any]:
    """Smallest applied voltage that opens at least one complete path.

    This is the bottleneck value ``min_path max_edge V_edge``.  It is computed by
    sorting edge thresholds and binary-searching the first threshold at which the
    two electrode sets become connected.
    """
    i = np.asarray(i, dtype=int)
    j = np.asarray(j, dtype=int)
    V = np.asarray(edge_voltage_V, dtype=float)
    keep = np.isfinite(V) & (V >= 0)
    i, j, V = i[keep], j[keep], V[keep]
    if i.size == 0:
        return {
            "available": False,
            "breakdown_voltage_V": None,
            "reason": "没有可用的接触边",
        }
    if not _connected_at_threshold(
        n, i, j, float(np.max(V)), V, electrodes_a, electrodes_b
    ):
        return {
            "available": False,
            "breakdown_voltage_V": None,
            "reason": "即使所有接触都击穿，电极之间仍不连通（几何未逾渗）",
        }
    order = np.argsort(V)
    Vs = V[order]
    lo, hi = 0, len(Vs) - 1
    if _connected_at_threshold(n, i, j, float(Vs[0]), V, electrodes_a, electrodes_b):
        return {
            "available": True,
            "breakdown_voltage_V": float(Vs[0]),
            "n_edges": int(V.size),
            "method": "minimax_bottleneck_path",
        }
    while lo < hi:
        mid = (lo + hi) // 2
        if _connected_at_threshold(
            n, i, j, float(Vs[mid]), V, electrodes_a, electrodes_b
        ):
            hi = mid
        else:
            lo = mid + 1
    return {
        "available": True,
        "breakdown_voltage_V": float(Vs[lo]),
        "n_edges": int(V.size),
        "n_edges_at_breakdown": int(np.sum(V <= Vs[lo])),
        "method": "minimax_bottleneck_path",
    }


# ---------------------------------------------------------------------------
# Weibull fit
# ---------------------------------------------------------------------------
def fit_weibull(
    samples: Iterable[float],
    *,
    location_fixed_at_zero: bool = True,
    min_samples: int = 5,
) -> dict[str, Any]:
    data = np.asarray([float(v) for v in samples if np.isfinite(v) and v > 0])
    if data.size < min_samples:
        return {
            "model_status": "unavailable",
            "reason": f"有效样本不足（{data.size} < {min_samples}）",
            "n": int(data.size),
            "warnings": [],
        }
    try:
        shape, loc, scale = weibull_min.fit(
            data, floc=0.0 if location_fixed_at_zero else None
        )
        method = "mle_weibull_min"
    except Exception as exc:  # noqa: BLE001
        # Method-of-moments fallback (shape from CV).
        mean = float(np.mean(data))
        std = float(np.std(data, ddof=1))
        cv = std / mean if mean > 0 else 0.0
        shape = max(0.3, min(20.0, (cv + 1e-12) ** -1.086))
        scale = mean / math.gamma(1.0 + 1.0 / shape)
        loc = 0.0
        method = f"method_of_moments_fallback_after:{type(exc).__name__}"
    n = int(data.size)
    mean = float(scale * math.gamma(1.0 + 1.0 / shape))
    var = float(
        scale**2 * (math.gamma(1.0 + 2.0 / shape) - math.gamma(1.0 + 1.0 / shape) ** 2)
    )
    return {
        "model_status": "conditional",
        "method": method,
        "shape_k": float(shape),
        "scale_lambda": float(scale),
        "location": float(loc),
        "mean": mean,
        "std": math.sqrt(max(var, 0.0)),
        "p10": float(weibull_min.ppf(0.10, shape, loc=loc, scale=scale)),
        "p50": float(weibull_min.ppf(0.50, shape, loc=loc, scale=scale)),
        "p90": float(weibull_min.ppf(0.90, shape, loc=loc, scale=scale)),
        "n": n,
        "samples": data.tolist(),
        "warnings": (
            ["Weibull 形状参数很大，分布接近确定性阈值；需更多样本验证尾部。"]
            if shape > 15
            else []
        ),
    }


# ---------------------------------------------------------------------------
# Public dielectric model
# ---------------------------------------------------------------------------
def dielectric_breakdown(
    structure: Structure,
    insulation: InsulationParameters,
    *,
    rng: np.random.Generator | None = None,
    repeats: int = 200,
    weibull_shape: float = 8.0,
    near_gap_m: float | None = None,
    slab_fraction: float = 0.12,
    transport_axis: int = 0,
) -> dict[str, Any]:
    """Monte Carlo network breakdown model for one DEM structure.

    Returns a distribution of sample breakdown voltages, a Weibull fit and
    explicit warnings.  The electrode voltage is the potential difference between
    the two particle slabs; the sample field is reported as ``V/L`` with
    ``L`` the box length along the transport axis.
    """
    if insulation.thickness_mean_m is None:
        return {
            "model_status": "unavailable",
            "reason": "缺少绝缘层厚度，无法建立击穿网络",
            "warnings": [],
        }
    if insulation.breakdown_field_V_m is None:
        return {
            "model_status": "unavailable",
            "reason": "缺少绝缘层击穿场强，无法建立击穿网络",
            "warnings": [],
        }
    rng = rng or np.random.default_rng(12345)
    c = structure.contacts
    if c.empty:
        return {
            "model_status": "unavailable",
            "reason": "接触表为空",
            "warnings": [],
        }
    radii = structure.radii
    mean_d = float(np.mean(2.0 * radii))
    t_mean = float(insulation.thickness_mean_m)
    if near_gap_m is None:
        near_gap_m = max(5.0 * t_mean, 0.05 * mean_d)
    gap = c["surface_gap_m"].to_numpy(float)
    candidate = gap <= near_gap_m
    if not np.any(candidate):
        return {
            "model_status": "unavailable",
            "reason": "没有位于绝缘层近场范围内的接触",
            "warnings": [],
        }
    ci = c.loc[candidate, "particle_i"].to_numpy(int)
    cj = c.loc[candidate, "particle_j"].to_numpy(int)
    # Remove periodic contacts across the open electrode axis.
    wrap_col = f"wraps_{'xyz'[transport_axis]}"
    wraps = c.loc[candidate, wrap_col].to_numpy(bool)
    keep = ~wraps
    ci, cj = ci[keep], cj[keep]
    if ci.size == 0:
        return {
            "model_status": "unavailable",
            "reason": "去除跨电极周期接触后没有候选边",
            "warnings": [],
        }
    a, b = electrode_indices(
        structure.centers, structure.box_length_m, slab_fraction=slab_fraction
    )
    L = structure.box_length_m
    voltages: list[float] = []
    failures: list[str] = []
    all_warnings: list[str] = []
    for _ in range(max(1, int(repeats))):
        sampled = sample_edge_breakdown_voltage(
            n_edges=ci.size,
            insulation=insulation,
            rng=rng,
            weibull_shape=weibull_shape,
        )
        result = minimax_breakdown_voltage(
            structure.n_particles,
            ci,
            cj,
            sampled["breakdown_voltage_V"],
            a,
            b,
        )
        if result["available"]:
            voltages.append(float(result["breakdown_voltage_V"]))
        else:
            failures.append(str(result.get("reason", "unavailable")))
    # Defect sampling repeats have different counts; report the mechanism once.
    if insulation.defect_probability:
        all_warnings.append(
            f"每个 Monte Carlo 实现按 defect_probability={insulation.defect_probability} "
            "独立抽样缺陷；缺陷电压折减未与具体显微缺陷统计校准。"
        )
    # De-duplicate warning text to keep reports readable.
    all_warnings = list(dict.fromkeys(all_warnings))
    if not voltages:
        return {
            "model_status": "unavailable",
            "reason": "所有 Monte Carlo 实现均未形成击穿路径",
            "failure_reasons": list(dict.fromkeys(failures)),
            "warnings": all_warnings,
        }
    data = np.asarray(voltages, dtype=float)
    fit = fit_weibull(data)
    fields = data / L
    field_fit = fit_weibull(fields) if fit.get("model_status") == "conditional" else {
        "model_status": "unavailable"
    }
    return {
        "model_status": "conditional",
        "method": "monte_carlo_minimax_bottleneck_path",
        "n_repeats": int(max(1, int(repeats))),
        "n_successful": int(data.size),
        "n_failed": int(len(failures)),
        "sample_length_m": float(L),
        "electrode_a_size": int(a.size),
        "electrode_b_size": int(b.size),
        "n_candidate_edges": int(ci.size),
        "breakdown_voltage_V": fit,
        "breakdown_field_V_m": field_fit,
        "raw_breakdown_voltage_V": data.tolist(),
        "warnings": [
            *all_warnings,
            "击穿电压为统计性最弱路径模型；未解析局部场增强、空间电荷和热失控。",
            "绝缘厚度、缺陷率、击穿场强均为输入参数；示例配置是 D 级占位值。",
            "电极选取为粒子板层近似，不是真实电极几何。",
            *fit.get("warnings", []),
        ],
        "assumptions": [
            "每个接触的绝缘层独立同分布（或按输入分布）",
            "接触击穿阈值由 Weibull 场强与抽样厚度相乘",
            "样品击穿由最短/最弱瓶颈路径决定；所有路径中最小最大边阈值",
            "缺陷以随机电压折减表示，未映射到具体涂层缺陷类型",
        ],
    }
