"""B-H curve loading, branching, loop energy and permeability definitions.

A single outward field sweep is **not** a hysteresis loop.  This module therefore
validates closure and separates ascending/descending branches before computing
loop area, coercivity or permeability.  It also keeps all permeability
definitions explicit, because initial, amplitude, incremental, secant and
complex permeabilities are not interchangeable.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .io_utils import read_csv
from .units import format_quantity


class BHDataError(ValueError):
    pass


@dataclass
class BHBranches:
    ascending_H: np.ndarray
    ascending_B: np.ndarray
    descending_H: np.ndarray | None
    descending_B: np.ndarray | None
    closed: bool
    closure_error_H_A_m: float
    closure_error_B_T: float


def _as_arrays(H: Any, B: Any) -> tuple[np.ndarray, np.ndarray]:
    H = np.asarray(H, dtype=float).reshape(-1)
    B = np.asarray(B, dtype=float).reshape(-1)
    if H.size != B.size:
        raise BHDataError("H 与 B 长度必须相同")
    if H.size < 3:
        raise BHDataError("至少需要 3 个数据点")
    if not np.all(np.isfinite(H)) or not np.all(np.isfinite(B)):
        raise BHDataError("H/B 包含 NaN 或无穷值")
    if np.ptp(H) <= 0:
        raise BHDataError("H 没有变化，无法构成磁滞回线")
    return H, B


def load_bh_curve(path: str | Path) -> pd.DataFrame:
    df = read_csv(path)
    aliases = {
        "H": "H_A_m",
        "H_A_per_m": "H_A_m",
        "B": "B_T",
        "M": "M_A_m",
        "M_A_per_m": "M_A_m",
    }
    df = df.rename(columns={k: v for k, v in aliases.items() if k in df.columns})
    if "H_A_m" not in df.columns or "B_T" not in df.columns:
        raise BHDataError(
            f"{path} 必须包含 H_A_m 与 B_T 列（SI 单位）；实际列：{list(df.columns)}"
        )
    H, B = _as_arrays(df["H_A_m"], df["B_T"])
    df = df.iloc[: H.size].copy()
    if "M_A_m" not in df.columns:
        df["M_A_m"] = B / (4e-7 * math.pi) - H
    return df


def split_branches(H: Any, B: Any) -> BHBranches:
    """Split a measured loop into ascending and descending branches.

    Only one major loop is supported.  If no turning point exists, the data are
    treated as a single ascending branch and ``closed`` is False.
    """
    H, B = _as_arrays(H, B)
    imax = int(np.argmax(H))
    imin = int(np.argmin(H))
    closed_H = abs(H[-1] - H[0]) <= max(1e-9, 1e-6 * np.ptp(H))
    closed_B = abs(B[-1] - B[0]) <= max(1e-12, 1e-4 * max(np.ptp(B), 1e-12))
    closed = bool(closed_H and closed_B)
    if imax > 0 and imax < H.size - 1:
        up_H, up_B = H[: imax + 1], B[: imax + 1]
        down_H, down_B = H[imax:], B[imax:]
    elif imin > 0 and imin < H.size - 1:
        down_H, down_B = H[: imin + 1], B[: imin + 1]
        up_H, up_B = H[imin:], B[imin:]
    else:
        up_H, up_B = H, B
        down_H = down_B = None
    if up_H.size < 2:
        raise BHDataError("上升支数据不足")
    # Ensure the ascending branch is actually monotonic in H.
    if np.any(np.diff(up_H) < 0):
        order = np.argsort(up_H)
        up_H, up_B = up_H[order], up_B[order]
    return BHBranches(
        ascending_H=up_H,
        ascending_B=up_B,
        descending_H=down_H,
        descending_B=down_B,
        closed=closed,
        closure_error_H_A_m=float(abs(H[-1] - H[0])),
        closure_error_B_T=float(abs(B[-1] - B[0])),
    )


def loop_energy_density(
    H: Any, B: Any, *, require_closed: bool = True
) -> tuple[float, list[str]]:
    r"""Return :math:`\oint H\,dB` in J/m^3 per cycle.

    The sign convention is removed by taking the absolute value; the result is
    the dissipated energy density per cycle.  If the curve is not closed, a
    warning is returned (and an exception is raised when ``require_closed``).
    """
    H, B = _as_arrays(H, B)
    branches = split_branches(H, B)
    warnings: list[str] = []
    if require_closed and not branches.closed:
        raise BHDataError(
            "H-B 数据未闭合，不能作为完整周期计算磁滞损耗。"
            f" 闭合误差：ΔH={branches.closure_error_H_A_m:.3g} A/m, "
            f"ΔB={branches.closure_error_B_T:.3g} T"
        )
    if not branches.closed:
        warnings.append(
            "输入不是闭合回线；能量积分仅作诊断，不能当作每周期磁滞损耗。"
        )
    # Trapezoidal integral of H dB along the stored scan order.
    area = float(abs(np.sum(0.5 * (H[:-1] + H[1:]) * np.diff(B))))
    if area <= 0:
        warnings.append("回线面积为零；请检查数据顺序、单位和是否只有单支扫描。")
    return area, warnings


def hysteresis_loss_density(
    H: Any, B: Any, frequency_hz: float, *, require_closed: bool = True
) -> tuple[float, list[str]]:
    """Average hysteresis power density in W/m^3 (cycle energy * frequency)."""
    if frequency_hz < 0:
        raise ValueError("频率不能为负")
    area, warnings = loop_energy_density(H, B, require_closed=require_closed)
    return float(area * frequency_hz), warnings


def closure_report(H: Any, B: Any) -> dict[str, Any]:
    H, B = _as_arrays(H, B)
    branches = split_branches(H, B)
    return {
        "closed": branches.closed,
        "closure_error_H_A_m": branches.closure_error_H_A_m,
        "closure_error_B_T": branches.closure_error_B_T,
        "n_points": int(H.size),
        "n_ascending": int(branches.ascending_H.size),
        "n_descending": int(
            branches.descending_H.size if branches.descending_H is not None else 0
        ),
        "warning": (
            None
            if branches.closed
            else "回线未闭合：不能将积分面积解释为完整周期的磁滞损耗。"
        ),
    }


def _branch_arrays(H: np.ndarray, B: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    branches = split_branches(H, B)
    return branches.ascending_H, branches.ascending_B


def permeability_metrics(
    H: Any,
    B: Any,
    *,
    initial_field_fraction: float = 0.002,
    bias_field_A_m: float | None = None,
) -> dict[str, Any]:
    """Compute initial, amplitude, secant and incremental permeabilities.

    All values are relative permeabilities (:math:`\\mu_r`).  The initial value
    is obtained by a linear fit around zero field on the ascending branch; it is
    therefore a *small-signal, zero-bias* definition.  The amplitude value is
    ``B_peak/(mu0 H_peak)``.  The incremental value is the local slope at
    ``bias_field_A_m`` (or at zero by default).
    """
    H, B = _as_arrays(H, B)
    Hu, Bu = _branch_arrays(H, B)
    Hmax = float(np.max(np.abs(Hu)))
    if Hmax <= 0:
        raise BHDataError("H 幅值为零")
    near = np.abs(Hu) <= initial_field_fraction * Hmax
    if np.count_nonzero(near) >= 2:
        slope, intercept = np.polyfit(Hu[near], Bu[near], 1)
        # A fitted intercept means the branch is not centred; report R^2 anyway.
        pred = slope * Hu[near] + intercept
        ss_res = float(np.sum((Bu[near] - pred) ** 2))
        ss_tot = float(np.sum((Bu[near] - np.mean(Bu[near])) ** 2))
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 1.0
        mu_initial = max(float(slope) / (4e-7 * math.pi), 0.0)
    else:
        mu_initial = None
        r2 = None
    Bpeak = float(np.max(np.abs(B)))
    mu_amplitude = (
        float(Bpeak / (4e-7 * math.pi * Hmax)) if Hmax > 0 and Bpeak > 0 else None
    )
    bias = 0.0 if bias_field_A_m is None else float(bias_field_A_m)
    idx = int(np.argmin(np.abs(Hu - bias)))
    if 1 <= idx < Hu.size - 1:
        dH = Hu[idx + 1] - Hu[idx - 1]
        dB = Bu[idx + 1] - Bu[idx - 1]
        mu_incremental = float(dB / dH / (4e-7 * math.pi)) if dH != 0 else None
    else:
        mu_incremental = None
    return {
        "mu_initial_relative": mu_initial,
        "mu_initial_fit_r2": r2,
        "mu_amplitude_relative": mu_amplitude,
        "mu_secant_relative": mu_amplitude,
        "mu_incremental_relative": mu_incremental,
        "bias_field_A_m": bias,
        "H_peak_A_m": Hmax,
        "B_peak_T": Bpeak,
        "definitions": {
            "initial": f"ascending branch linear fit for |H| <= {initial_field_fraction:.0%} H_peak",
            "amplitude": "B_peak/(mu0*H_peak)",
            "secant": "same as amplitude at the stated peak",
            "incremental": "local dB/dH at the stated bias field",
            "complex": "not computed: requires small-signal AC measurement",
        },
        "warnings": (
            ["初始磁导率线性拟合的 R^2 较低，请检查低场数据质量。"]
            if r2 is not None and r2 < 0.95
            else []
        ),
    }


def saturation_metric(
    H: Any, B: Any, *, criterion: str = "B_at_Hmax"
) -> dict[str, Any]:
    """Report a saturation metric together with its criterion."""
    H, B = _as_arrays(H, B)
    Hu, Bu = _branch_arrays(H, B)
    i = int(np.argmax(np.abs(Hu)))
    B_s = float(Bu[i])
    return {
        "B_saturation_T": B_s,
        "H_at_saturation_A_m": float(Hu[i]),
        "criterion": criterion,
        "warning": (
            "饱和判据必须与数值一起报告；不同判据下的 B_s 不能直接比较。"
        ),
    }


def coercivity_remanence(H: Any, B: Any) -> dict[str, float | None]:
    """Extract H_c and B_r from the descending branch (major loop)."""
    H, B = _as_arrays(H, B)
    branches = split_branches(H, B)
    if branches.descending_H is None or branches.descending_B is None:
        return {"H_c_A_m": None, "B_r_T": None, "note": "没有下降支"}
    Hd, Bd = branches.descending_H, branches.descending_B
    order = np.argsort(Hd)
    Hd, Bd = Hd[order], Bd[order]
    Hc = None
    if Bd.min() <= 0 <= Bd.max():
        # B = 0 crossing by linear interpolation.
        idx = np.where(np.diff(np.signbit(Bd)))[0]
        if idx.size:
            k = int(idx[0])
            Hc = float(
                Hd[k] + (0.0 - Bd[k]) * (Hd[k + 1] - Hd[k]) / (Bd[k + 1] - Bd[k])
            )
    Br = None
    if Hd.min() <= 0 <= Hd.max():
        Br = float(np.interp(0.0, Hd, Bd))
    return {
        "H_c_A_m": Hc,
        "B_r_T": Br,
        "note": "H_c 为下降支 B=0 的场；B_r 为 H=0 的剩余磁通密度。",
    }


def describe_curve(df: pd.DataFrame, frequency_hz: float | None = None) -> dict[str, Any]:
    H = df["H_A_m"].to_numpy(float)
    B = df["B_T"].to_numpy(float)
    closure = closure_report(H, B)
    metrics = permeability_metrics(H, B)
    coercive = coercivity_remanence(H, B)
    sat = saturation_metric(H, B)
    result: dict[str, Any] = {
        "closure": closure,
        "permeability": metrics,
        "coercivity_remanence": coercive,
        "saturation": sat,
        "units": {
            "H_A_m": "A/m",
            "B_T": "T",
            "M_A_m": "A/m",
        },
    }
    if closure["closed"]:
        area, warn = loop_energy_density(H, B)
        result["loop_energy_density_J_m3_per_cycle"] = area
        result["loop_area_warnings"] = warn
        if frequency_hz is not None:
            result["hysteresis_loss_W_m3"] = area * frequency_hz
    else:
        result["loop_energy_density_J_m3_per_cycle"] = None
        result["hysteresis_loss_W_m3"] = None
        result["loop_area_warnings"] = [closure["warning"]]
    return result


def summary_markdown(df: pd.DataFrame) -> str:
    info = describe_curve(df)
    p = info["permeability"]
    return "\n".join(
        [
            "### B–H 曲线摘要",
            "",
            f"- 点数：{info['closure']['n_points']}",
            f"- 闭合：{info['closure']['closed']}",
            f"- 峰值 H：{format_quantity(p['H_peak_A_m'], 'A/m')}",
            f"- 峰值 B：{format_quantity(p['B_peak_T'], 'T')}",
            f"- 初始磁导率：{format_quantity(p['mu_initial_relative'], '')}",
            f"- 振幅磁导率：{format_quantity(p['mu_amplitude_relative'], '')}",
            f"- 矫顽力：{format_quantity(info['coercivity_remanence']['H_c_A_m'], 'A/m')}",
            f"- 剩磁：{format_quantity(info['coercivity_remanence']['B_r_T'], 'T')}",
        ]
    )
