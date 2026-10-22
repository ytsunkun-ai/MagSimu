"""Effective-medium homogenisation and reduced-order B-H generation.

The module contains three deliberately separate estimates:

1. analytical mixing laws for a magnetic particle fraction in a non-magnetic
   matrix (Maxwell-Garnett / Bruggeman / logarithmic),
2. a DEM-informed magnetic reluctance network (see :mod:`softmag.microstructure`),
3. a tanh major-loop reduced-order hysteresis model used as the documented
   fallback when MicroMagnetic.jl is unavailable.

The fallback is **not** a micromagnetics result.  Every function returns a
``model_status``/warnings so the report can distinguish the methods.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np
import pandas as pd
from scipy.optimize import brentq

from .bh_curve import permeability_metrics
from .units import MU0


# ---------------------------------------------------------------------------
# Mixing laws
# ---------------------------------------------------------------------------
def maxwell_garnett(
    phi: float,
    mu_particle: float,
    *,
    mu_matrix: float = 1.0,
    demag_factor: float = 1.0 / 3.0,
) -> float:
    """Maxwell-Garnett effective relative permeability for spherical inclusions."""
    _validate_mixing_inputs(phi, mu_particle, mu_matrix)
    if phi == 0:
        return float(mu_matrix)
    if phi == 1:
        return float(mu_particle)
    if abs(demag_factor - 1.0 / 3.0) > 1e-12:
        # Generalised MG with an anisotropic/demagnetising inclusion factor:
        # mu_eff = mu_m * [mu_m + (1-N)(mu_p-mu_m)phi] /
        #                 [mu_m + N(1-phi)(mu_p-mu_m)]  (sphere N=1/3).
        # This is a simple shape correction, not a full tensor model.
        return float(
            mu_matrix
            * (
                mu_matrix
                + (1.0 - demag_factor) * (mu_particle - mu_matrix) * phi
            )
            / (
                mu_matrix
                + demag_factor * (1.0 - phi) * (mu_particle - mu_matrix)
            )
        )
    num = mu_particle + 2 * mu_matrix + 2 * phi * (mu_particle - mu_matrix)
    den = mu_particle + 2 * mu_matrix - phi * (mu_particle - mu_matrix)
    return float(mu_matrix * num / den)


def bruggeman(
    phi: float,
    mu_particle: float,
    *,
    mu_matrix: float = 1.0,
    demag_factor: float = 1.0 / 3.0,
) -> float:
    """Symmetric Bruggeman effective medium for spherical inclusions."""
    _validate_mixing_inputs(phi, mu_particle, mu_matrix)
    if phi == 0:
        return float(mu_matrix)
    if phi == 1:
        return float(mu_particle)
    n = float(demag_factor)
    if abs(n - 1.0 / 3.0) > 1e-12:
        # Use the Maxwell-Garnett form as a documented approximation for
        # non-spherical inclusions; the symmetric tensor solution is out of scope.
        return maxwell_garnett(
            phi, mu_particle, mu_matrix=mu_matrix, demag_factor=n
        )

    def f(mu: float) -> float:
        return phi * (mu_particle - mu) / (mu_particle + 2 * mu) + (
            1 - phi
        ) * (mu_matrix - mu) / (mu_matrix + 2 * mu)

    lo, hi = sorted((float(mu_matrix), float(mu_particle)))
    # f(lo) and f(hi) have opposite signs for a physical mixture.
    flo, fhi = f(lo), f(hi)
    if abs(flo) < 1e-14:
        return lo
    if abs(fhi) < 1e-14:
        return hi
    if flo * fhi > 0:
        # Fallback to MG if the symmetric equation has no sign change.
        return maxwell_garnett(phi, mu_particle, mu_matrix=mu_matrix)
    return float(brentq(f, lo, hi, xtol=1e-12, rtol=1e-12, maxiter=200))


def logarithmic_mixing(
    phi: float, mu_particle: float, *, mu_matrix: float = 1.0
) -> float:
    """Lichtenecker logarithmic mixing law (trend estimate only)."""
    _validate_mixing_inputs(phi, mu_particle, mu_matrix)
    return float(mu_matrix ** (1 - phi) * mu_particle**phi)


def _validate_mixing_inputs(phi: float, mu_particle: float, mu_matrix: float) -> None:
    if not 0.0 <= phi <= 1.0:
        raise ValueError("磁性相体积分数 phi 必须位于 [0, 1]")
    if mu_particle <= 0 or mu_matrix <= 0:
        raise ValueError("磁导率必须为正")


def mixing_effective_permeability(
    phi: float,
    mu_particle: float | None,
    *,
    mu_matrix: float = 1.0,
    model: Literal["bruggeman", "maxwell_garnett", "logarithmic"] = "bruggeman",
    demag_factor: float = 1.0 / 3.0,
) -> dict[str, Any]:
    if mu_particle is None:
        return {
            "model_status": "unavailable",
            "effective_relative_permeability": None,
            "reason": "缺少颗粒本征磁导率",
            "warnings": [],
        }
    funcs = {
        "bruggeman": bruggeman,
        "maxwell_garnett": maxwell_garnett,
        "logarithmic": logarithmic_mixing,
    }
    values = {
        name: fn(
            phi,
            mu_particle,
            mu_matrix=mu_matrix,
            **({"demag_factor": demag_factor} if name != "logarithmic" else {}),
        )
        for name, fn in funcs.items()
    }
    chosen = values[model]
    return {
        "model_status": "conditional",
        "method": f"mixing:{model}",
        "effective_relative_permeability": float(chosen),
        "models": {k: float(v) for k, v in values.items()},
        "magnetic_volume_fraction": float(phi),
        "mu_particle": float(mu_particle),
        "demag_factor": float(demag_factor),
        "assumptions": [
            "球形（或给定退磁因子） inclusions",
            "线性、各向同性、无局部饱和",
            "颗粒间磁相互作用仅通过有效介质近似",
        ],
        "warnings": [
            "混合模型不能描述接触网络、局部磁通拥挤或饱和；高磁通下仅作趋势参考。"
        ],
    }


# ---------------------------------------------------------------------------
# Reduced-order tanh major-loop model
# ---------------------------------------------------------------------------
@dataclass
class TanhLoopParameters:
    """Three-parameter major-loop model.

    Ascending:  ``M = Ms tanh((H - Hc)/a)``
    Descending: ``M = Ms tanh((H + Hc)/a)``

    ``Hc`` is exactly the coercivity and ``a`` controls the approach to
    saturation.  This is a transparent reduced-order fallback, **not** a
    micromagnetics model.  Its purpose is to keep the data flow and calibration
    pipeline testable when MicroMagnetic.jl is unavailable.
    """

    Ms_A_m: float
    Hc_A_m: float
    a_A_m: float

    def validate(self) -> None:
        if self.Ms_A_m <= 0:
            raise ValueError("Ms 必须为正")
        if self.Hc_A_m <= 0:
            raise ValueError("Hc 必须为正")
        if self.a_A_m <= 0:
            raise ValueError("a 必须为正")

    def as_dict(self) -> dict[str, float]:
        return {
            "Ms_A_m": float(self.Ms_A_m),
            "Hc_A_m": float(self.Hc_A_m),
            "a_A_m": float(self.a_A_m),
            "Mr_over_Ms": float(np.tanh(self.Hc_A_m / self.a_A_m)),
        }


def default_tanh_parameters(
    Ms_A_m: float,
    *,
    coercivity_A_m: float = 100.0,
    magnetic_volume_fraction: float = 1.0,
    target_initial_permeability: float | None = None,
    squareness: float = 0.30,
) -> TanhLoopParameters:
    """Construct a plausible starting loop.

    If ``target_initial_permeability`` is supplied, ``a`` is solved so that the
    *core* initial permeability matches the target:
    ``mu_i = 1 + phi*Ms/a * sech^2(Hc/a)``.  Otherwise ``a`` is chosen from a
    target remanence ratio.  These are starting values, not calibrated data.
    """
    if Ms_A_m <= 0:
        raise ValueError("Ms 必须为正")
    if coercivity_A_m <= 0:
        raise ValueError("Hc 必须为正")
    if not 0.0 <= squareness < 1.0:
        raise ValueError("squareness 必须位于 [0, 1)")
    phi = float(magnetic_volume_fraction)
    if not 0.0 <= phi <= 1.0:
        raise ValueError("phi 必须位于 [0, 1]")
    Hc = float(coercivity_A_m)
    if target_initial_permeability is not None and target_initial_permeability > 1.0 and phi > 0:
        mu_target = float(target_initial_permeability)

        def residual(log_a: float) -> float:
            a = math.exp(log_a)
            return 1.0 + phi * Ms_A_m / a / math.cosh(Hc / a) ** 2 - mu_target

        # The low-field slope is non-monotonic in ``a``: very small ``a`` makes
        # an almost square loop (slope -> 0 at H=0), and very large ``a`` makes a
        # very soft loop (slope -> 0 as well).  We want the large-``a`` root,
        # because it represents a soft magnetic composite rather than a square
        # loop.  Start the bracket at the maximum of the slope.
        x_peak = 0.771702  # solves 1 - 2 x tanh(x) = 0
        a_peak = max(Hc / x_peak, 1e-9)
        hi = max(a_peak * 10.0, Ms_A_m)
        for _ in range(60):
            if residual(math.log(hi)) < 0:
                break
            hi *= 2.0
        if residual(math.log(a_peak)) > 0:
            a = math.exp(
                brentq(
                    residual,
                    math.log(a_peak),
                    math.log(hi),
                    xtol=1e-8,
                    rtol=1e-10,
                    maxiter=200,
                )
            )
        else:
            # Target is above the maximum achievable low-field slope.  Return
            # the closest physically meaningful loop and let the caller warn.
            a = a_peak
    else:
        if squareness <= 0:
            a = 10.0 * Hc
        else:
            a = Hc / math.atanh(min(squareness, 0.999))
    return TanhLoopParameters(Ms_A_m=float(Ms_A_m), Hc_A_m=Hc, a_A_m=float(a))


def _field_grid(Hmax_A_m: float, n_points: int, *, power: float = 3.0) -> np.ndarray:
    """Non-uniform field grid, dense near zero where magnetisation switches."""
    s = np.linspace(-1.0, 1.0, int(n_points))
    return Hmax_A_m * np.sign(s) * np.abs(s) ** float(power)


def tanh_major_loop(
    params: TanhLoopParameters,
    *,
    Hmax_A_m: float,
    magnetic_volume_fraction: float = 1.0,
    n_points: int = 201,
    label: str = "tanh_major_loop",
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Generate a symmetric major loop with the tanh model.

    The returned ``B_T`` is the **powder-core** flux density
    ``mu0 (H + phi M_phase)``; the phase magnetisation is retained separately.
    """
    params.validate()
    if Hmax_A_m <= 0:
        raise ValueError("Hmax_A_m 必须为正")
    if n_points < 5:
        raise ValueError("n_points 至少为 5")
    phi = float(magnetic_volume_fraction)
    if not 0.0 <= phi <= 1.0:
        raise ValueError("magnetic_volume_fraction 必须位于 [0, 1]")
    H_up = _field_grid(Hmax_A_m, n_points)
    H_down = H_up[::-1]
    M_up = params.Ms_A_m * np.tanh((H_up - params.Hc_A_m) / params.a_A_m)
    M_down = params.Ms_A_m * np.tanh((H_down + params.Hc_A_m) / params.a_A_m)
    H = np.concatenate([H_up, H_down[1:]])
    M_phase = np.concatenate([M_up, M_down[1:]])
    branch = np.array(
        ["ascending"] * H_up.size + ["descending"] * (H_down.size - 1)
    )
    M_macro = phi * M_phase
    B = MU0 * (H + M_macro)
    df = pd.DataFrame(
        {
            "H_A_m": H,
            "M_phase_A_m": M_phase,
            "M_macro_A_m": M_macro,
            "B_T": B,
            "branch": branch,
            "model": label,
        }
    )
    from .bh_curve import loop_energy_density

    area, area_warnings = loop_energy_density(H, B, require_closed=False)
    metrics = permeability_metrics(H, B)
    closure_error = abs(float(M_phase[-1] - M_phase[0])) / params.Ms_A_m
    diagnostics = {
        "model_status": "fallback_reduced_order",
        "model": "tanh_major_loop",
        "parameters": params.as_dict(),
        "magnetic_volume_fraction": phi,
        "Hmax_A_m": float(Hmax_A_m),
        "n_points": int(n_points),
        "loop_energy_density_J_m3_per_cycle": area,
        "permeability": metrics,
        "closure": {
            "closure_error_relative": float(closure_error),
            "closed": bool(closure_error < 0.02),
        },
        "warnings": [
            *area_warnings,
            "tanh 大回线是唯象 fallback，不是微磁学结果；不得替代 MicroMagnetic.jl。",
            "回线参数必须用独立实验数据校准后才可用于定量预测。",
        ],
        "assumptions": [
            "对称大回线、宏观各向同性",
            "Hc 与 a 吸收颗粒相互作用、缺陷和退磁效应",
            "不描述小回线、各向异性或频率相关动态回线",
        ],
    }
    return df, diagnostics


def tanh_minor_loop(
    params: TanhLoopParameters,
    *,
    B_peak_T: float,
    magnetic_volume_fraction: float,
    n_points: int = 241,
    shape_exponent: float = 2.0,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Closed symmetric minor loop at a requested peak flux density.

    A finite-amplitude minor loop cannot be obtained by simply truncating a
    major loop: the two branches would not meet at the turning points.  This
    function constructs a closed, symmetric normalised loop

    ``B/Bp = tanh(k x)/tanh(k) -/+ h (1-x^2)``, ``x=H/Ha``,

    with the ascending branch using ``-h`` and the descending branch ``+h``.
    The turning field ``Ha`` is solved from the anhysteretic tanh curve so that
    the loop peak is ``B_peak_T``.  ``h`` is bounded by ``Br/Bp`` and 0.48, and
    ``k`` is a fixed shape parameter.  This is a transparent phenomenological
    construction, not a Preisach model or a micromagnetics minor loop.
    """
    params.validate()
    phi = float(magnetic_volume_fraction)
    if not 0.0 <= phi <= 1.0:
        raise ValueError("magnetic_volume_fraction 必须位于 [0, 1]")
    Bp = float(B_peak_T)
    if Bp < 0:
        raise ValueError("B_peak_T 不能为负")
    if Bp == 0:
        df = pd.DataFrame(
            {
                "H_A_m": [0.0, 0.0],
                "M_phase_A_m": [0.0, 0.0],
                "M_macro_A_m": [0.0, 0.0],
                "B_T": [0.0, 0.0],
                "branch": ["ascending", "descending"],
            }
        )
        return df, {
            "model_status": "conditional",
            "model": "normalized_closed_minor_loop",
            "B_peak_T": 0.0,
            "H_turning_A_m": 0.0,
            "loop_energy_density_J_m3_per_cycle": 0.0,
            "closure": {"closed": True, "closure_error_relative": 0.0},
            "warnings": ["零激励；磁滞能量设为零。"],
        }
    if shape_exponent <= 0:
        raise ValueError("shape_exponent 必须为正")

    def anhysteretic_B(H: float) -> float:
        return MU0 * (H + phi * params.Ms_A_m * math.tanh(H / params.a_A_m))

    lo = 0.0
    hi = max(10.0 * params.a_A_m, 10.0 * params.Hc_A_m, 1.0)
    for _ in range(80):
        if anhysteretic_B(hi) >= Bp:
            break
        hi *= 2.0
    if anhysteretic_B(hi) < Bp:
        raise ValueError("B_peak_T 超过模型可表示的峰值范围")
    Ha = float(brentq(lambda H: anhysteretic_B(H) - Bp, lo, hi, xtol=1e-10, rtol=1e-12))
    Br = MU0 * phi * params.Ms_A_m * math.tanh(params.Hc_A_m / params.a_A_m)
    h = min(0.48, max(0.0, Br / Bp))
    s = _field_grid(Ha, n_points)
    x_up = s / Ha
    x_down = x_up[::-1]
    base_up = Bp * np.tanh(shape_exponent * x_up) / math.tanh(shape_exponent)
    base_down = Bp * np.tanh(shape_exponent * x_down) / math.tanh(shape_exponent)
    B_up = base_up - Bp * h * (1.0 - x_up**2) ** 3
    B_down = base_down + Bp * h * (1.0 - x_down**2) ** 3
    H_up = s
    H_down = s[::-1]
    H = np.concatenate([H_up, H_down[1:]])
    B = np.concatenate([B_up, B_down[1:]])
    branch = np.array(
        ["ascending"] * H_up.size + ["descending"] * (H_down.size - 1)
    )
    M_macro = B / MU0 - H
    M_phase = M_macro / phi if phi > 0 else np.zeros_like(M_macro)
    df = pd.DataFrame(
        {
            "H_A_m": H,
            "M_phase_A_m": M_phase,
            "M_macro_A_m": M_macro,
            "B_T": B,
            "branch": branch,
            "model": "normalized_closed_minor_loop",
        }
    )
    from .bh_curve import loop_energy_density

    area, area_warnings = loop_energy_density(H, B, require_closed=False)
    closure = float(abs(B[-1] - B[0]) / max(abs(Bp), 1e-30))
    return df, {
        "model_status": "conditional",
        "model": "normalized_closed_minor_loop",
        "B_peak_T": Bp,
        "H_turning_A_m": Ha,
        "B_turning_T": float(anhysteretic_B(Ha)),
        "remanence_ratio_h": float(h),
        "shape_exponent": float(shape_exponent),
        "loop_energy_density_J_m3_per_cycle": float(area),
        "closure": {"closed": bool(closure < 1e-6), "closure_error_relative": closure},
        "warnings": [
            *area_warnings,
            "小回线为归一化唯象闭合构造，不是微磁学小回线；必须用实测 B_peak 损耗校准。",
        ],
        "assumptions": [
            "对称小回线，峰值等于给定 B_peak",
            "回线形状由固定指数与 Br/Bp 控制，未从实验小回线拟合",
            "只用于损耗量级估算，不能替代损耗 map",
        ],
    }


def estimate_initial_permeability_from_tanh(
    params: TanhLoopParameters, *, magnetic_volume_fraction: float = 1.0
) -> float:
    phi = float(magnetic_volume_fraction)
    slope = (
        params.Ms_A_m
        / params.a_A_m
        / math.cosh(params.Hc_A_m / params.a_A_m) ** 2
    )
    return float(1.0 + phi * slope)


def macro_bh_from_phase_M(
    H_A_m: np.ndarray,
    M_phase_A_m: np.ndarray,
    *,
    magnetic_volume_fraction: float,
    branch: np.ndarray | None = None,
    demag_factor: float | None = None,
    label: str = "phase_average",
) -> pd.DataFrame:
    """Convert magnetic-phase magnetisation to a powder-core B(H) curve.

    ``B = mu0 (H + phi M_phase)``.  If ``demag_factor`` is supplied, the
    constitutive curve is evaluated at the internal field
    ``H_int = H - N phi M_phase``; the returned curve is still against the applied
    field.  This is a lumped demagnetisation correction, not a full field solver.
    """
    H = np.asarray(H_A_m, dtype=float)
    M = np.asarray(M_phase_A_m, dtype=float)
    if H.shape != M.shape:
        raise ValueError("H 与 M 长度必须一致")
    if not 0.0 <= magnetic_volume_fraction <= 1.0:
        raise ValueError("磁性相体积分数必须位于 [0,1]")
    M_macro = magnetic_volume_fraction * M
    B = MU0 * (H + M_macro)
    if demag_factor is not None:
        H_int = H - float(demag_factor) * M_macro
    else:
        H_int = H
    df = pd.DataFrame(
        {
            "H_A_m": H,
            "H_internal_A_m": H_int,
            "M_phase_A_m": M,
            "M_macro_A_m": M_macro,
            "B_T": B,
        }
    )
    if branch is not None:
        df["branch"] = branch
    df["model"] = label
    return df


# ---------------------------------------------------------------------------
# High-level homogenisation
# ---------------------------------------------------------------------------
def homogenize(
    *,
    magnetic_volume_fraction: float,
    Ms_A_m: float | None,
    particle_relative_permeability: float | None,
    demag_factor: float = 1.0 / 3.0,
    network: dict[str, Any] | None = None,
    mixing_model: Literal["bruggeman", "maxwell_garnett", "logarithmic"] = "bruggeman",
) -> dict[str, Any]:
    """Combine mixing and network estimates into one reported homogenisation."""
    phi = float(magnetic_volume_fraction)
    mixing = mixing_effective_permeability(
        phi,
        particle_relative_permeability,
        model=mixing_model,
        demag_factor=demag_factor,
    )
    if Ms_A_m is not None:
        Ms_eff = phi * float(Ms_A_m)
        Bs_eff = MU0 * Ms_eff
    else:
        Ms_eff = None
        Bs_eff = None
    network_mu = None
    if network and network.get("effective_relative_permeability") is not None:
        network_mu = float(network["effective_relative_permeability"])
    mixing_mu = mixing.get("effective_relative_permeability")
    if network_mu is not None:
        chosen = network_mu
        chosen_source = "dem_magnetic_network"
    else:
        chosen = mixing_mu
        chosen_source = f"mixing:{mixing_model}"
    warnings = list(mixing.get("warnings", []))
    if network_mu is not None and mixing_mu is not None:
        ratio = network_mu / mixing_mu if mixing_mu else None
        if ratio is not None and (ratio > 5 or ratio < 0.2):
            warnings.append(
                "磁路网络与混合模型的有效磁导率相差超过 5 倍；"
                "说明结果对接触/绝缘层假设高度敏感，必须实验校准。"
            )
    return {
        "model_status": "conditional" if chosen is not None else "unavailable",
        "method": "network+mixing" if network_mu is not None else f"mixing:{mixing_model}",
        "effective_relative_permeability": chosen,
        "effective_permeability_source": chosen_source,
        "mixing": mixing,
        "network": network,
        "magnetic_volume_fraction": phi,
        "Ms_phase_A_m": Ms_A_m,
        "Ms_effective_A_m": Ms_eff,
        "B_saturation_effective_T": Bs_eff,
        "demag_factor": float(demag_factor),
        "warnings": warnings,
        "assumptions": [
            "宏观各向同性",
            "磁性相体积分数是体积平均，不是质量分数",
            "有效饱和磁通密度按 mu0*phi*Ms 估计，不代表实测 B_s 判据",
        ],
    }

