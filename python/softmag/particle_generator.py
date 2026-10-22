"""Particle-size distributions and particle-table generation.

The module treats the *basis* of a size distribution explicitly.  Laser
diffraction D10/D50/D90 values are usually volume-based, while DEM needs a
number-weighted particle sample.  For a lognormal distribution, weighting by
volume (d^3) shifts the log-mean by ``+3 sigma^2``; the code performs that shift
instead of silently mixing number and volume statistics.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from scipy.stats import norm

from .io_utils import ensure_dir, read_csv, resolve_path, write_csv, write_json
from .schemas import ParticleParameters

Basis = Literal["number", "volume"]


# ---------------------------------------------------------------------------
# Fitting
# ---------------------------------------------------------------------------
def fit_lognormal_quantiles(
    d10_m: float | None,
    d50_m: float,
    d90_m: float | None,
) -> tuple[float, float, dict[str, Any]]:
    """Fit a lognormal distribution while keeping D50 exact.

    Returns ``(mu_log, sigma_log, diagnostics)``.  The median fixes ``mu``; the
    width is chosen to minimise the squared log-error of D10/D90.  This avoids a
    fitted median drifting away from a measured D50 when the measured quantiles
    are not perfectly lognormal.
    """
    if d50_m is None or d50_m <= 0:
        raise ValueError("D50 必须为正")
    if d10_m is not None and d10_m <= 0:
        raise ValueError("D10 必须为正")
    if d90_m is not None and d90_m <= 0:
        raise ValueError("D90 必须为正")
    if d10_m is not None and d10_m > d50_m:
        raise ValueError("D10 不能大于 D50")
    if d90_m is not None and d90_m < d50_m:
        raise ValueError("D90 不能小于 D50")

    mu = math.log(float(d50_m))
    z90 = float(norm.ppf(0.90))

    if d10_m is not None or d90_m is not None:

        def loss(log_sigma: float) -> float:
            sigma = math.exp(log_sigma)
            total = 0.0
            count = 0
            if d10_m is not None:
                pred = math.exp(mu - z90 * sigma)
                total += (math.log(pred / d10_m)) ** 2
                count += 1
            if d90_m is not None:
                pred = math.exp(mu + z90 * sigma)
                total += (math.log(pred / d90_m)) ** 2
                count += 1
            return total / max(count, 1)

        result = minimize_scalar(loss, bounds=(-3.2, 1.7), method="bounded")
        sigma = float(math.exp(result.x))
        assumed = False
    else:
        sigma = 0.30
        assumed = True

    predicted = {
        "d10_m": math.exp(mu - z90 * sigma),
        "d50_m": math.exp(mu),
        "d90_m": math.exp(mu + z90 * sigma),
    }
    targets = {"d10_m": d10_m, "d50_m": d50_m, "d90_m": d90_m}
    rel_errors = {}
    for key, target in targets.items():
        if target is not None:
            rel_errors[key] = predicted[key] / target - 1.0
    diagnostics = {
        "method": "lognormal_median_preserving",
        "assumed_width": assumed,
        "predicted_quantiles_m": predicted,
        "relative_errors": rel_errors,
        "max_abs_relative_error": max((abs(v) for v in rel_errors.values()), default=0.0),
        "warning": (
            "仅给出 D50 时采用 sigma_log=0.30 的假设分布宽度；这不代表真实粒径分布。"
            if assumed
            else None
        ),
    }
    return mu, sigma, diagnostics


# ---------------------------------------------------------------------------
# Distribution object
# ---------------------------------------------------------------------------
@dataclass
class ParticleSizeDistribution:
    """Number- or volume-based particle size distribution.

    Parameters
    ----------
    kind:
        ``lognormal``, ``histogram`` or ``monodisperse``.
    basis:
        Basis in which the source parameters/weights are expressed.
    """

    kind: Literal["lognormal", "histogram", "monodisperse"]
    basis: Basis = "volume"
    mu_log: float | None = None
    sigma_log: float | None = None
    diameters_m: np.ndarray | None = None
    weights: np.ndarray | None = None
    source: str = "unknown"
    input_quantiles_m: dict[str, float | None] = field(default_factory=dict)
    fit_diagnostics: dict[str, Any] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)

    # -- constructors ---------------------------------------------------
    @classmethod
    def from_quantiles(
        cls,
        d10_m: float | None,
        d50_m: float,
        d90_m: float | None,
        *,
        basis: Basis = "volume",
        source: str = "D10/D50/D90",
    ) -> "ParticleSizeDistribution":
        mu, sigma, diag = fit_lognormal_quantiles(d10_m, d50_m, d90_m)
        warnings = []
        if diag.get("warning"):
            warnings.append(str(diag["warning"]))
        if diag["max_abs_relative_error"] > 0.25:
            warnings.append(
                "输入分位数与对数正态假设偏差较大（最大相对偏差 "
                f"{diag['max_abs_relative_error']:.1%}）；建议改用实测直方图。"
            )
        return cls(
            kind="lognormal",
            basis=basis,
            mu_log=mu,
            sigma_log=sigma,
            source=source,
            input_quantiles_m={"d10_m": d10_m, "d50_m": d50_m, "d90_m": d90_m},
            fit_diagnostics=diag,
            warnings=warnings,
        )

    @classmethod
    def monodisperse(cls, diameter_m: float, *, source: str = "monodisperse") -> "ParticleSizeDistribution":
        if diameter_m <= 0:
            raise ValueError("直径必须为正")
        return cls(
            kind="monodisperse",
            basis="number",
            diameters_m=np.array([float(diameter_m)]),
            weights=np.array([1.0]),
            source=source,
            input_quantiles_m={
                "d10_m": diameter_m,
                "d50_m": diameter_m,
                "d90_m": diameter_m,
            },
            warnings=["单分散假设忽略了粒径分布；真实粉末通常至少需要 D10/D50/D90。"],
        )

    @classmethod
    def from_histogram(
        cls,
        diameters_m: np.ndarray | list[float],
        weights: np.ndarray | list[float],
        *,
        basis: Basis = "number",
        source: str = "histogram",
    ) -> "ParticleSizeDistribution":
        d = np.asarray(diameters_m, dtype=float)
        w = np.asarray(weights, dtype=float)
        if d.ndim != 1 or w.ndim != 1 or d.size != w.size:
            raise ValueError("diameters_m 与 weights 必须是等长一维数组")
        if d.size < 2:
            raise ValueError("直方图至少需要 2 个粒径组")
        if np.any(d <= 0) or np.any(w < 0):
            raise ValueError("粒径必须为正，权重不能为负")
        if not np.isfinite(w).all() or w.sum() <= 0:
            raise ValueError("权重必须为有限非负数且总和大于 0")
        order = np.argsort(d)
        d = d[order]
        w = w[order]
        w = w / w.sum()
        return cls(
            kind="histogram",
            basis=basis,
            diameters_m=d,
            weights=w,
            source=source,
            input_quantiles_m={
                "d10_m": cls._hist_quantile(d, w, basis, 0.1),
                "d50_m": cls._hist_quantile(d, w, basis, 0.5),
                "d90_m": cls._hist_quantile(d, w, basis, 0.9),
            },
        )

    # -- parameter transforms ------------------------------------------
    def _lognormal_params(self, weighting: Basis) -> tuple[float, float]:
        if self.kind != "lognormal" or self.mu_log is None or self.sigma_log is None:
            raise RuntimeError("不是对数正态分布")
        mu, sigma = self.mu_log, self.sigma_log
        if weighting == self.basis:
            return mu, sigma
        # volume weighting of a lognormal multiplies the density by d^3.
        if self.basis == "volume" and weighting == "number":
            return mu - 3.0 * sigma**2, sigma
        if self.basis == "number" and weighting == "volume":
            return mu + 3.0 * sigma**2, sigma
        return mu, sigma

    def _number_probabilities(self) -> np.ndarray:
        if self.kind != "histogram":
            raise RuntimeError("不是直方图分布")
        assert self.diameters_m is not None and self.weights is not None
        if self.basis == "volume":
            p = self.weights / self.diameters_m**3
        else:
            p = self.weights.copy()
        return p / p.sum()

    def _probabilities(self, weighting: Basis) -> np.ndarray:
        p = self._number_probabilities()
        if weighting == "number":
            return p
        assert self.diameters_m is not None
        pv = p * self.diameters_m**3
        return pv / pv.sum()

    @staticmethod
    def _hist_quantile(d: np.ndarray, w: np.ndarray, basis: Basis, q: float) -> float:
        if basis == "volume":
            p = (w * d**3) / np.sum(w * d**3)
        else:
            p = w / np.sum(w)
        cdf = np.clip(np.cumsum(p), 0.0, 1.0)
        return float(np.interp(q, cdf, d))

    # -- public API -----------------------------------------------------
    def sample(
        self,
        n: int,
        rng: np.random.Generator | None = None,
        *,
        weighting: Basis = "number",
    ) -> np.ndarray:
        """Draw ``n`` diameters from the requested weighting."""
        if n <= 0:
            raise ValueError("n 必须为正整数")
        rng = rng or np.random.default_rng()
        if self.kind == "monodisperse":
            assert self.diameters_m is not None
            return np.full(int(n), float(self.diameters_m[0]))
        if self.kind == "lognormal":
            mu, sigma = self._lognormal_params(weighting)
            return np.exp(rng.normal(mu, sigma, size=int(n)))
        assert self.diameters_m is not None
        p = self._probabilities(weighting)
        idx = rng.choice(self.diameters_m.size, size=int(n), p=p)
        return self.diameters_m[idx]

    def quantile(self, q: float, *, weighting: Basis = "number") -> float:
        if not 0.0 < q < 1.0:
            raise ValueError("q 必须位于 (0, 1)")
        if self.kind == "monodisperse":
            assert self.diameters_m is not None
            return float(self.diameters_m[0])
        if self.kind == "lognormal":
            mu, sigma = self._lognormal_params(weighting)
            return float(math.exp(mu + sigma * norm.ppf(q)))
        assert self.diameters_m is not None
        p = self._probabilities(weighting)
        cdf = np.clip(np.cumsum(p), 0.0, 1.0)
        return float(np.interp(q, cdf, self.diameters_m))

    def cdf(self, diameter_m: float | np.ndarray, *, weighting: Basis = "number") -> np.ndarray:
        d = np.asarray(diameter_m, dtype=float)
        if self.kind == "monodisperse":
            assert self.diameters_m is not None
            return (d >= self.diameters_m[0]).astype(float)
        if self.kind == "lognormal":
            mu, sigma = self._lognormal_params(weighting)
            out = np.where(d > 0, norm.cdf((np.log(np.maximum(d, 1e-300)) - mu) / sigma), 0.0)
            return out
        assert self.diameters_m is not None
        p = self._probabilities(weighting)
        cdf = np.clip(np.cumsum(p), 0.0, 1.0)
        return np.interp(d, self.diameters_m, cdf, left=0.0, right=1.0)

    def stats(self, *, weighting: Basis = "number") -> dict[str, float]:
        if self.kind == "lognormal":
            mu_number, sigma = self._lognormal_params("number")
            # All moments are handled analytically in the requested basis.
            mu, sigma = self._lognormal_params(weighting)
            mean = math.exp(mu + 0.5 * sigma**2)
            second = math.exp(2 * mu + 2 * sigma**2)
            third = math.exp(3 * mu_number + 4.5 * sigma**2)
            std = math.sqrt(max(second - mean**2, 0.0))
            d_vm = third ** (1.0 / 3.0)
            d10 = math.exp(mu - 1.2815515655 * sigma)
            d50 = math.exp(mu)
            d90 = math.exp(mu + 1.2815515655 * sigma)
        elif self.kind == "histogram":
            assert self.diameters_m is not None
            p = self._probabilities(weighting)
            d = self.diameters_m
            mean = float(np.sum(p * d))
            third = float(np.sum(self._probabilities("number") * d**3))
            d_vm = third ** (1.0 / 3.0)
            std = float(math.sqrt(max(np.sum(p * (d - mean) ** 2), 0.0)))
            cdf = np.clip(np.cumsum(p), 0.0, 1.0)
            d10 = float(np.interp(0.1, cdf, d))
            d50 = float(np.interp(0.5, cdf, d))
            d90 = float(np.interp(0.9, cdf, d))
        else:
            assert self.diameters_m is not None
            mean = d_vm = d10 = d50 = d90 = float(self.diameters_m[0])
            std = 0.0
        return {
            "basis": weighting,
            "mean_m": mean,
            "std_m": std,
            "d10_m": d10,
            "d50_m": d50,
            "d90_m": d90,
            "volume_weighted_mean_m": d_vm,
            "span": (d90 - d10) / d50 if d50 > 0 else float("nan"),
        }

    @property
    def representative_diameter_m(self) -> float:
        """Number-weighted mean diameter (used as a fallback length scale)."""
        return self.stats(weighting="number")["mean_m"]

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "kind": self.kind,
            "basis": self.basis,
            "source": self.source,
            "input_quantiles_m": self.input_quantiles_m,
            "fit_diagnostics": self.fit_diagnostics,
            "warnings": self.warnings,
            "stats_number": self.stats(weighting="number"),
            "stats_volume": self.stats(weighting="volume"),
        }
        if self.kind == "lognormal":
            payload["mu_log"] = self.mu_log
            payload["sigma_log"] = self.sigma_log
        if self.diameters_m is not None:
            payload["histogram"] = {
                "diameters_m": self.diameters_m.tolist(),
                "weights": self.weights.tolist() if self.weights is not None else None,
            }
        return payload


# ---------------------------------------------------------------------------
# Loading from configuration
# ---------------------------------------------------------------------------
_DIAMETER_COLUMNS = {
    "diameter_m": 1.0,
    "d_m": 1.0,
    "diameter_um": 1.0e-6,
    "d_um": 1.0e-6,
    "diameter_um_m": 1.0e-6,
    "diameter_mm": 1.0e-3,
    "d_mm": 1.0e-3,
}
_WEIGHT_COLUMNS = (
    "count",
    "number",
    "number_fraction",
    "volume_fraction",
    "frequency",
    "fraction",
    "weight",
)


def load_histogram_csv(
    path: str | Path,
    *,
    basis: Basis | None = None,
    diameter_unit: str | None = None,
) -> ParticleSizeDistribution:
    df = read_csv(path)
    if df.empty:
        raise ValueError(f"粒径直方图文件为空：{path}")
    lower = {str(c).strip().lower(): c for c in df.columns}
    diameter_col = None
    scale = 1.0
    for name, factor in _DIAMETER_COLUMNS.items():
        if name in lower:
            diameter_col = lower[name]
            scale = factor
            break
    if diameter_col is None:
        raise ValueError(
            f"{path} 缺少粒径列；支持 {sorted(_DIAMETER_COLUMNS)}，实际列：{list(df.columns)}"
        )
    if diameter_unit:
        scale = {"m": 1.0, "um": 1e-6, "mm": 1e-3}[diameter_unit]
    weight_col = None
    for name in _WEIGHT_COLUMNS:
        if name in lower:
            weight_col = lower[name]
            break
    if weight_col is None:
        raise ValueError(f"{path} 缺少计数/权重列；支持 {_WEIGHT_COLUMNS}")
    inferred_basis: Basis
    if basis is not None:
        inferred_basis = basis
    elif "volume" in str(weight_col).lower():
        inferred_basis = "volume"
    elif "number" in str(weight_col).lower() or str(weight_col).lower() == "count":
        inferred_basis = "number"
    else:
        inferred_basis = "number"
    d = df[diameter_col].to_numpy(dtype=float) * scale
    w = df[weight_col].to_numpy(dtype=float)
    return ParticleSizeDistribution.from_histogram(
        d, w, basis=inferred_basis, source=f"file:{path}"
    )


def distribution_from_config(
    particle: ParticleParameters,
    *,
    base_dir: str | Path | None = None,
) -> ParticleSizeDistribution:
    """Build a :class:`ParticleSizeDistribution` from the input contract."""
    if particle.distribution_file:
        path = resolve_path(particle.distribution_file, base_dir)
        return load_histogram_csv(
            path,
            basis=particle.distribution_weight_type,
            diameter_unit=particle.distribution_diameter_unit,
        )
    if particle.size_distribution_type == "monodisperse":
        if particle.d50_m is None:
            raise ValueError("单分散分布必须给出 d50_m")
        return ParticleSizeDistribution.monodisperse(particle.d50_m)
    if particle.d50_m is None:
        raise ValueError(
            "缺少 particle.d50_m 或 distribution_file，无法建立粒径分布。"
            "系统不会根据材料名称自动假定粒径。"
        )
    dist = ParticleSizeDistribution.from_quantiles(
        particle.d10_m,
        particle.d50_m,
        particle.d90_m,
        basis=particle.quantile_basis,
        source="D10/D50/D90",
    )
    if particle.d10_m is None or particle.d90_m is None:
        dist.warnings.append(
            "缺少 D10 或 D90；分布宽度是假设值。请补充完整分位数或直方图。"
        )
    return dist


# ---------------------------------------------------------------------------
# Particle table
# ---------------------------------------------------------------------------
def generate_particle_table(
    dist: ParticleSizeDistribution,
    n: int,
    *,
    rng: np.random.Generator | None = None,
    aspect_ratio: float | None = None,
    shape_model: str = "sphere",
) -> pd.DataFrame:
    """Generate a number-weighted particle table with equivalent-volume axes.

    DEM contact detection in this prototype uses volume-equivalent spheres (or
    their circumscribed equivalent radius).  Shape information is retained for
    demagnetisation and loss corrections, but ellipsoidal contact mechanics is
    out of scope and is reported as a limitation.
    """
    rng = rng or np.random.default_rng(12345)
    d_eq = dist.sample(n, rng, weighting="number")
    alpha = float(aspect_ratio) if aspect_ratio else 1.0
    if alpha < 1.0:
        raise ValueError("aspect_ratio 必须 >= 1")
    if shape_model in {"sphere", "irregular", "measured_mesh"}:
        alpha = 1.0
    if alpha == 1.0:
        a = b = c = d_eq
    else:
        # spheroid: a/b = alpha; a*b*c = d_eq^3
        a = d_eq * alpha ** (2.0 / 3.0)
        b = d_eq * alpha ** (-1.0 / 3.0)
        c = b
    volume = (math.pi / 6.0) * a * b * c
    theta = np.arccos(rng.uniform(-1.0, 1.0, size=n)) if alpha > 1.0 else np.zeros(n)
    phi = rng.uniform(0.0, 2.0 * math.pi, size=n) if alpha > 1.0 else np.zeros(n)
    return pd.DataFrame(
        {
            "particle_id": np.arange(n, dtype=int),
            "diameter_eq_m": d_eq,
            "axis_a_m": a,
            "axis_b_m": b,
            "axis_c_m": c,
            "volume_m3": volume,
            "aspect_ratio": alpha,
            "shape_model": shape_model,
            "orientation_theta_rad": theta,
            "orientation_phi_rad": phi,
        }
    )


def size_summary(
    dist: ParticleSizeDistribution,
    sample_m: np.ndarray | None = None,
) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "distribution": dist.to_dict(),
        "quantiles_input": dist.input_quantiles_m,
        "warnings": list(dist.warnings),
        "model_status": "conditional",
        "note": (
            "粒径分布统计量必须注明 number/volume 归一化基。"
            "D10/D50/D90 的输入基由 particle.quantile_basis 给出。"
        ),
    }
    if sample_m is not None and len(sample_m) > 0:
        sample = np.asarray(sample_m, dtype=float)
        summary["sample"] = {
            "n": int(sample.size),
            "mean_m": float(np.mean(sample)),
            "std_m": float(np.std(sample)),
            "min_m": float(np.min(sample)),
            "max_m": float(np.max(sample)),
            "d10_m": float(np.quantile(sample, 0.1)),
            "d50_m": float(np.quantile(sample, 0.5)),
            "d90_m": float(np.quantile(sample, 0.9)),
            "volume_weighted_mean_m": float(
                (np.sum(sample**3) / sample.size) ** (1.0 / 3.0)
            ),
        }
    return summary


def write_particle_outputs(
    dist: ParticleSizeDistribution,
    outdir: str | Path,
    *,
    n_sample: int = 0,
    rng: np.random.Generator | None = None,
) -> dict[str, Path]:
    out = ensure_dir(outdir)
    paths: dict[str, Path] = {}
    grid = np.geomspace(
        max(dist.quantile(0.001), 1e-9), dist.quantile(0.999), 400
    )
    hist_df = pd.DataFrame(
        {
            "diameter_m": grid,
            "cdf_number": dist.cdf(grid, weighting="number"),
            "cdf_volume": dist.cdf(grid, weighting="volume"),
            "pdf_number_per_m": np.gradient(
                dist.cdf(grid, weighting="number"), grid
            ),
            "pdf_volume_per_m": np.gradient(
                dist.cdf(grid, weighting="volume"), grid
            ),
        }
    )
    paths["distribution_csv"] = write_csv(hist_df, out / "particle_size_distribution.csv")
    summary = size_summary(dist)
    if n_sample > 0:
        sample = dist.sample(n_sample, rng, weighting="number")
        summary["sample"] = size_summary(dist, sample)["sample"]
        paths["sample_csv"] = write_csv(
            pd.DataFrame({"particle_id": np.arange(n_sample), "diameter_m": sample}),
            out / "particle_sample.csv",
        )
    paths["summary_json"] = write_json(out / "particle_size_summary.json", summary)
    return paths
