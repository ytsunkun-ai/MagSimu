"""Physical constants, unit conversions and small validation helpers.

All internal calculations use SI units:

===============  =============
quantity         unit
===============  =============
length           m
magnetic field   A/m
flux density     T
frequency        Hz
resistivity      ohm m
density          kg/m^3
energy density   J/m^3
power density    W/m^3
===============  =============

The conversion helpers exist because experimental data are frequently provided in
Oe, G, emu/cm^3 or kg/mm^3.  Explicit conversion at the data boundary prevents the
most common class of multi-scale modelling errors.
"""

from __future__ import annotations

import math
from typing import Iterable

import numpy as np

MU0 = 4.0e-7 * math.pi  # vacuum permeability, H/m
MU0_OVER_4PI = 1.0e-7
PI = math.pi


# ---------------------------------------------------------------------------
# Unit conversions
# ---------------------------------------------------------------------------
def oe_to_a_per_m(h_oe: float | np.ndarray) -> float | np.ndarray:
    """Oersted -> A/m (1 Oe = 1000/(4*pi) A/m)."""
    return np.asarray(h_oe, dtype=float) * (1000.0 / (4.0 * PI))


def a_per_m_to_oe(h_a_m: float | np.ndarray) -> float | np.ndarray:
    return np.asarray(h_a_m, dtype=float) * (4.0 * PI / 1000.0)


def gauss_to_tesla(b_g: float | np.ndarray) -> float | np.ndarray:
    return np.asarray(b_g, dtype=float) * 1.0e-4


def tesla_to_gauss(b_t: float | np.ndarray) -> float | np.ndarray:
    return np.asarray(b_t, dtype=float) * 1.0e4


def emu_cm3_to_a_per_m(m_emu_cm3: float | np.ndarray) -> float | np.ndarray:
    """Magnetisation emu/cm^3 -> A/m (1 emu/cm^3 = 1000 A/m)."""
    return np.asarray(m_emu_cm3, dtype=float) * 1000.0


def a_per_m_to_emu_cm3(m_a_m: float | np.ndarray) -> float | np.ndarray:
    return np.asarray(m_a_m, dtype=float) / 1000.0


def um_to_m(d_um: float | np.ndarray) -> float | np.ndarray:
    return np.asarray(d_um, dtype=float) * 1.0e-6


def m_to_um(d_m: float | np.ndarray) -> float | np.ndarray:
    return np.asarray(d_m, dtype=float) * 1.0e6


def mm_to_m(d_mm: float | np.ndarray) -> float | np.ndarray:
    return np.asarray(d_mm, dtype=float) * 1.0e-3


def g_cm3_to_kg_m3(rho_g_cm3: float | np.ndarray) -> float | np.ndarray:
    return np.asarray(rho_g_cm3, dtype=float) * 1000.0


def kg_m3_to_g_cm3(rho_kg_m3: float | np.ndarray) -> float | np.ndarray:
    return np.asarray(rho_kg_m3, dtype=float) / 1000.0


def hz_to_angular(frequency_hz: float | np.ndarray) -> float | np.ndarray:
    return 2.0 * PI * np.asarray(frequency_hz, dtype=float)


# ---------------------------------------------------------------------------
# Excitation conventions
# ---------------------------------------------------------------------------
_FLUX_DEFINITIONS = {"peak", "amplitude", "peak_to_peak", "rms", "unknown"}


def amplitude_to_peak(value: float, definition: str) -> tuple[float, str | None]:
    """Convert a flux-density amplitude convention to a sine *peak* value.

    Returns ``(peak_value, warning)``.  The definition must be explicit: peak,
    amplitude, peak-to-peak, rms or unknown.  For ``unknown`` the value is passed
    through unchanged and a warning is returned.
    """
    definition = (definition or "unknown").strip().lower()
    if definition not in _FLUX_DEFINITIONS:
        raise ValueError(
            f"未知磁通密度定义 {definition!r}，应为 {sorted(_FLUX_DEFINITIONS)} 之一"
        )
    if value < 0:
        raise ValueError("磁通密度幅值不能为负")
    if definition in {"peak", "amplitude"}:
        return float(value), None
    if definition == "peak_to_peak":
        return float(value) / 2.0, None
    if definition == "rms":
        return float(value) * math.sqrt(2.0), None
    return float(value), (
        "磁通密度定义未指定（unknown），已按峰值处理；这会影响损耗绝对值，必须补充定义"
    )


def peak_to_rms(value: float) -> float:
    return float(value) / math.sqrt(2.0)


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------
def require_finite(value: float | None, name: str) -> float | None:
    if value is None:
        return None
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f"{name} 必须是有限数值，收到 {value!r}")
    return value


def require_positive(value: float | None, name: str) -> float | None:
    value = require_finite(value, name)
    if value is not None and value <= 0:
        raise ValueError(f"{name} 必须大于 0，收到 {value!r}")
    return value


def require_nonnegative(value: float | None, name: str) -> float | None:
    value = require_finite(value, name)
    if value is not None and value < 0:
        raise ValueError(f"{name} 不能为负，收到 {value!r}")
    return value


def require_fraction(
    value: float | None, name: str, *, allow_one: bool = True
) -> float | None:
    value = require_finite(value, name)
    if value is None:
        return None
    upper_ok = value <= 1.0 if allow_one else value < 1.0
    if not (0.0 <= value and upper_ok):
        interval = "(0, 1]" if allow_one else "(0, 1)"
        raise ValueError(f"{name} 必须位于 {interval}，收到 {value!r}")
    return value


# ---------------------------------------------------------------------------
# Lognormal distribution helpers
# ---------------------------------------------------------------------------
def lognormal_quantile(q: float, mu_log: float, sigma_log: float) -> float:
    """Quantile of a lognormal distribution parameterised in log-space."""
    if not 0.0 < q < 1.0:
        raise ValueError("分位数 q 必须位于 (0, 1)")
    from scipy.stats import norm

    return float(math.exp(mu_log + sigma_log * norm.ppf(q)))


def lognormal_moments(mu_log: float, sigma_log: float) -> dict[str, float]:
    mean = math.exp(mu_log + 0.5 * sigma_log**2)
    var = (math.exp(sigma_log**2) - 1.0) * math.exp(2.0 * mu_log + sigma_log**2)
    return {
        "mean_m": mean,
        "std_m": math.sqrt(var),
        "median_m": math.exp(mu_log),
    }


# ---------------------------------------------------------------------------
# Demagnetisation and characteristic scales
# ---------------------------------------------------------------------------
def demag_factor_sphere() -> float:
    return 1.0 / 3.0


def demag_factor_ellipsoid(axis_ratios: Iterable[float]) -> tuple[float, ...]:
    """Osborn demagnetisation factors for a general ellipsoid (SI, sum == 1).

    ``axis_ratios`` are the three semi-axis lengths in arbitrary units.  The
    numerical integration is stable for modest aspect ratios (0.05 .. 20), which
    covers the powder morphologies in this project.
    """
    axes = np.asarray(list(axis_ratios), dtype=float)
    if axes.shape != (3,) or np.any(axes <= 0):
        raise ValueError("需要三个正的半轴长度")
    a, b, c = axes
    # Use the general Carlson-form style numerical integral.  A simple adaptive
    # quadrature is sufficient here and avoids fragile closed-form branches.
    from scipy.integrate import quad

    factors = []
    for i in range(3):
        ai = axes[i]

        def integrand(s: float, ai: float = ai) -> float:
            denom = (s + a * a) * (s + b * b) * (s + c * c)
            return 1.0 / ((s + ai * ai) * math.sqrt(denom))

        value, _ = quad(integrand, 0.0, np.inf, limit=200)
        factors.append(0.5 * a * b * c * value)
    total = float(sum(factors))
    return tuple(float(f / total) for f in factors)


def exchange_length_m(
    exchange_stiffness_J_m: float, saturation_magnetisation_A_m: float
) -> float:
    r"""Estimate the exchange length :math:`\ell_{ex}=\sqrt{2A/(\mu_0 M_s^2)}`."""
    if exchange_stiffness_J_m <= 0 or saturation_magnetisation_A_m <= 0:
        raise ValueError("A 与 Ms 必须为正")
    return math.sqrt(
        2.0 * exchange_stiffness_J_m / (MU0 * saturation_magnetisation_A_m**2)
    )


def skin_depth_m(
    frequency_hz: float, resistivity_ohm_m: float, relative_permeability: float = 1.0
) -> float:
    """Classical skin depth ``sqrt(2 rho / (omega mu0 mu_r))``."""
    if frequency_hz <= 0 or resistivity_ohm_m <= 0 or relative_permeability <= 0:
        raise ValueError("频率、电阻率和相对磁导率必须为正")
    omega = 2.0 * PI * frequency_hz
    return math.sqrt(
        2.0 * resistivity_ohm_m / (omega * MU0 * relative_permeability)
    )


# ---------------------------------------------------------------------------
# Formatting
# ---------------------------------------------------------------------------
def format_number(value: float | None, digits: int = 4) -> str:
    """Compact engineering formatting for Markdown reports."""
    if value is None:
        return "—"
    try:
        value = float(value)
    except (TypeError, ValueError):
        return str(value)
    if not math.isfinite(value):
        return "—"
    if value == 0:
        return "0"
    magnitude = abs(value)
    if magnitude >= 1e5 or magnitude < 1e-3:
        return f"{value:.{digits - 1}e}"
    return f"{value:.{digits}g}"


def format_quantity(value: float | None, unit: str, digits: int = 4) -> str:
    if value is None:
        return "—"
    return f"{format_number(value, digits)} {unit}".strip()
