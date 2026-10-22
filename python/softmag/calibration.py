"""Calibration of reduced-order models against experimental CSV files.

The calibration module never hides a poor fit.  It returns parameters together
with fit domain, R^2, residual scale, bound-hit warnings and a data-quality flag.
Synthetic/placeholder files are detected from an optional ``source`` column and
from the project configuration; they are labelled as demo-only.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.optimize import least_squares

from .configs import CalibrationSettings, ExperimentConfig
from .eddy_loss import load_loss_csv, separate_losses, steinmetz_fit
from .io_utils import read_csv, sha256_file, timestamp_utc
from .schemas import PredictionConfig
from .units import MU0


def _resolve(path: str | None, base: Path) -> Path | None:
    if path is None:
        return None
    p = Path(path)
    return p if p.is_absolute() else (base / p).resolve()


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def _branch_array(df: pd.DataFrame) -> np.ndarray:
    if "branch" in df.columns:
        labels = df["branch"].astype(str).str.lower().str.strip().to_numpy()
        out = np.array(["ascending"] * len(df), dtype=object)
        out[np.isin(labels, ["desc", "descending", "down", "decreasing"])] = "descending"
        return out
    H = df["H_A_m"].to_numpy(float)
    # A single monotonic branch is the only safe inference here.
    d = np.diff(H)
    if np.all(d >= 0):
        return np.array(["ascending"] * len(df), dtype=object)
    if np.all(d <= 0):
        return np.array(["descending"] * len(df), dtype=object)
    # Full loop: use the first index of the maximum and minimum as turn points.
    imax = int(np.argmax(H))
    imin = int(np.argmin(H))
    labels = np.array(["ascending"] * len(df), dtype=object)
    if imax < imin:
        labels[: imax + 1] = "ascending"
        labels[imax + 1 : imin + 1] = "descending"
        labels[imin + 1 :] = "ascending"
    else:
        labels[: imin + 1] = "descending"
        labels[imin + 1 : imax + 1] = "ascending"
        labels[imax + 1 :] = "descending"
    return labels


def _source_quality(df: pd.DataFrame) -> str:
    if "source" not in df.columns:
        return "unknown"
    texts = " ".join(df["source"].astype(str).str.lower().unique())
    if any(token in texts for token in ("placeholder", "synthetic", "demo")):
        return "synthetic_placeholder"
    return "declared_source"


def load_bh_csv(path: str | Path) -> pd.DataFrame:
    df = read_csv(path)
    aliases = {
        "H": "H_A_m",
        "H_A_per_m": "H_A_m",
        "B": "B_T",
        "B_T": "B_T",
        "B_mT": "B_mT",
    }
    df = df.rename(columns={k: v for k, v in aliases.items() if k in df.columns})
    if "H_A_m" not in df.columns:
        raise ValueError(f"B-H 数据缺少 H_A_m/H 列；实际列：{list(df.columns)}")
    if "B_T" not in df.columns and "B_mT" in df.columns:
        df["B_T"] = df["B_mT"] * 1.0e-3
    if "B_T" not in df.columns:
        raise ValueError(f"B-H 数据缺少 B_T/B 列；实际列：{list(df.columns)}")
    df["H_A_m"] = pd.to_numeric(df["H_A_m"], errors="coerce")
    df["B_T"] = pd.to_numeric(df["B_T"], errors="coerce")
    df = df.dropna(subset=["H_A_m", "B_T"]).copy()
    if len(df) < 5:
        raise ValueError("B-H 数据有效点不足 5 个")
    df["branch"] = _branch_array(df)
    return df.reset_index(drop=True)


def fit_tanh_loop(
    df: pd.DataFrame,
    *,
    magnetic_volume_fraction: float,
    initial_guess: dict[str, float] | None = None,
    train_fraction: float = 0.7,
    random_seed: int = 12345,
) -> dict[str, Any]:
    """Fit ``Ms, Hc, a`` of the symmetric tanh major loop.

    ``B = mu0 (H + phi Ms tanh((H -/+ Hc)/a))``.  The ascending branch uses
    ``-Hc`` and the descending branch ``+Hc``.  If both branches exist, the model
    is trained on one branch and tested on the other: a random point split would
    leak nearly identical neighbouring points and overstate predictive skill.
    """
    if not 0.0 < magnetic_volume_fraction <= 1.0:
        raise ValueError("magnetic_volume_fraction 必须位于 (0,1]")
    data = df.copy()
    data["branch"] = _branch_array(data)
    H = data["H_A_m"].to_numpy(float)
    B = data["B_T"].to_numpy(float)
    branches = data["branch"].to_numpy(object)
    has_both = bool(np.any(branches == "ascending") and np.any(branches == "descending"))
    warnings: list[str] = []
    if has_both:
        train_mask = branches == "ascending"
        test_mask = branches == "descending"
        split = "train_ascending_test_descending"
        warnings.append(
            "拟合在升支上完成、在降支上检验；这不是独立的温度/频率外推验证。"
        )
    else:
        rng = np.random.default_rng(random_seed)
        idx = rng.permutation(len(data))
        n_train = max(5, int(round(train_fraction * len(data))))
        train_mask = np.zeros(len(data), dtype=bool)
        train_mask[idx[:n_train]] = True
        test_mask = ~train_mask
        split = f"random_point_split_{train_fraction:.2f}"
        warnings.append(
            "只有一个磁化分支：采用随机点划分，邻近点相关会使测试 R^2 偏乐观。"
        )
    if np.sum(train_mask) < 5:
        raise ValueError("训练点不足 5 个")

    Hmax = float(np.max(np.abs(H)))
    if "Ms_A_m" in (initial_guess or {}):
        Ms0 = float(initial_guess["Ms_A_m"])
    else:
        Ms0 = max(
            float(np.max(np.abs(B))) / (MU0 * max(magnetic_volume_fraction, 1e-6)),
            1.0e4,
        )
    Hc0 = float((initial_guess or {}).get("Hc_A_m", max(Hmax * 0.01, 1.0)))
    a0 = float((initial_guess or {}).get("a_A_m", max(Hmax * 0.1, Hc0 * 2.0)))
    Ms_hi = max(4.0 * Ms0, 1.0e6)
    Hc_hi = max(Hmax, 1.0)
    a_hi = max(20.0 * Hmax, 1.0)
    bounds = (
        np.array([1.0e2, 0.0, 1.0e-3]),
        np.array([Ms_hi, Hc_hi, a_hi]),
    )
    x0 = np.clip(np.array([Ms0, Hc0, a0]), bounds[0] * 1.0001, bounds[1] * 0.9999)
    B_scale = max(float(np.std(B[train_mask])), 1.0e-6)

    def model_B(x: np.ndarray, H_in: np.ndarray, br: np.ndarray) -> np.ndarray:
        Ms, Hc, a = x
        sign = np.where(br == "ascending", 1.0, -1.0)
        M = Ms * np.tanh((H_in - sign * Hc) / a)
        return MU0 * (H_in + magnetic_volume_fraction * M)

    def residual(x: np.ndarray) -> np.ndarray:
        return (model_B(x, H[train_mask], branches[train_mask]) - B[train_mask]) / B_scale

    result = least_squares(
        residual,
        x0,
        bounds=bounds,
        x_scale=np.maximum(np.abs(x0), 1e-9),
        max_nfev=5000,
        ftol=1e-12,
        xtol=1e-12,
        gtol=1e-12,
    )
    x = result.x
    pred_all = model_B(x, H, branches)
    pred_train = pred_all[train_mask]
    pred_test = pred_all[test_mask]
    resid_all = B - pred_all
    ss_res = float(np.sum(resid_all**2))
    ss_tot = float(np.sum((B - np.mean(B)) ** 2))
    r2_all = 1.0 - ss_res / ss_tot if ss_tot > 0 else 1.0
    r2_train = (
        1.0
        - float(np.sum((B[train_mask] - pred_train) ** 2))
        / max(float(np.sum((B[train_mask] - np.mean(B[train_mask])) ** 2)), 1e-30)
    )
    r2_test = None
    test_rmse = None
    if np.any(test_mask):
        test_rmse = float(np.sqrt(np.mean((B[test_mask] - pred_test) ** 2)))
        ss_t = float(np.sum((B[test_mask] - np.mean(B[test_mask])) ** 2))
        if ss_t > 0:
            r2_test = 1.0 - float(np.sum((B[test_mask] - pred_test) ** 2)) / ss_t
    # Parameter covariance from the Jacobian, reported only when well-conditioned.
    param_std: dict[str, float | None] = {k: None for k in ("Ms_A_m", "Hc_A_m", "a_A_m")}
    try:
        J = result.jac
        dof = max(len(B[train_mask]) - 3, 1)
        s2 = float(np.sum(result.fun**2)) / dof
        cov = np.linalg.inv(J.T @ J) * s2 * B_scale**2
        if np.all(np.isfinite(cov)) and np.all(np.diag(cov) >= 0):
            std = np.sqrt(np.diag(cov))
            for key, value in zip(param_std, std):
                param_std[key] = float(value)
    except Exception:  # noqa: BLE001
        pass
    at_bound = []
    for name, value, lo, hi in zip(
        ("Ms_A_m", "Hc_A_m", "a_A_m"), x, bounds[0], bounds[1]
    ):
        if (value - lo) <= 1e-3 * max(abs(lo), 1.0) or (hi - value) <= 1e-3 * max(
            abs(hi), 1.0
        ):
            at_bound.append(name)
    if r2_all < 0.9:
        warnings.append("tanh 回线拟合 R^2 < 0.9；模型形状与数据不匹配，不应外推。")
    if at_bound:
        warnings.append(f"拟合参数触及边界：{at_bound}；参数不可辨识或数据范围不足。")
    source_quality = "unknown"
    if "source" in data.columns:
        texts = " ".join(data["source"].astype(str).str.lower().unique())
        if "placeholder" in texts or "synthetic" in texts or "demo" in texts:
            source_quality = "synthetic_placeholder"
    if source_quality == "synthetic_placeholder":
        warnings.append("校准数据含 placeholder/synthetic 标记；拟合仅用于代码验证。")
    return {
        "model_status": "conditional",
        "method": "tanh_major_loop_least_squares",
        "parameters": {
            "Ms_A_m": float(x[0]),
            "Hc_A_m": float(x[1]),
            "a_A_m": float(x[2]),
        },
        "parameter_std": param_std,
        "magnetic_volume_fraction": float(magnetic_volume_fraction),
        "fit_formula": "B = mu0*(H + phi*Ms*tanh((H -/+ Hc)/a))",
        "r2_all": float(r2_all),
        "r2_train": float(r2_train),
        "r2_test": float(r2_test) if r2_test is not None else None,
        "test_rmse_T": test_rmse,
        "n_points": int(len(data)),
        "n_train": int(np.sum(train_mask)),
        "n_test": int(np.sum(test_mask)),
        "split": split,
        "bounds": {"lower": bounds[0].tolist(), "upper": bounds[1].tolist()},
        "at_bound": at_bound,
        "data_quality": source_quality,
        "warnings": warnings,
        "assumptions": [
            "对称大回线，Ms/Hc/a 为集总有效参数",
            "忽略频率、温度、小回线和各向异性",
            "使用体积分数 phi 将相磁化转为宏观 B",
        ],
        "created_utc": timestamp_utc(),
    }


def _fit_steinmetz(path: Path, prediction: PredictionConfig) -> dict[str, Any]:
    df = load_loss_csv(path)
    density = prediction.material.density_kg_m3
    if "loss_W_kg" in df.columns:
        fit = steinmetz_fit(df, loss_col="loss_W_kg", density_kg_m3=density)
    else:
        fit = steinmetz_fit(df)
    quality = _source_quality(df)
    fit["data_quality"] = quality
    if quality == "synthetic_placeholder":
        fit.setdefault("warnings", []).append(
            "Steinmetz 数据含 placeholder/synthetic 标记；拟合仅用于代码验证。"
        )
    fit["source_file"] = str(path)
    fit["source_sha256"] = sha256_file(path)
    return fit


def _fit_breakdown(path: Path) -> dict[str, Any]:
    df = read_csv(path)
    aliases = {
        "V": "breakdown_voltage_V",
        "voltage_V": "breakdown_voltage_V",
        "E": "breakdown_field_V_m",
        "field_V_m": "breakdown_field_V_m",
        "thickness_m": "thickness_m",
    }
    df = df.rename(columns={k: v for k, v in aliases.items() if k in df.columns})
    if "breakdown_voltage_V" in df.columns and "thickness_m" in df.columns:
        field = df["breakdown_voltage_V"].to_numpy(float) / df["thickness_m"].to_numpy(float)
        col = "breakdown_field_V_m"
        data = field
    elif "breakdown_field_V_m" in df.columns:
        col = "breakdown_field_V_m"
        data = df[col].to_numpy(float)
    elif "breakdown_voltage_V" in df.columns:
        col = "breakdown_voltage_V"
        data = df[col].to_numpy(float)
    else:
        return {
            "model_status": "unavailable",
            "reason": f"击穿数据缺少可识别列；实际列：{list(df.columns)}",
            "warnings": [],
        }
    from .dielectric import fit_weibull

    fit = fit_weibull(data)
    fit["source_file"] = str(path)
    fit["source_sha256"] = sha256_file(path)
    fit["column_used"] = col
    quality = _source_quality(df)
    fit["data_quality"] = quality
    if quality == "synthetic_placeholder":
        fit.setdefault("warnings", []).append(
            "击穿数据含 placeholder/synthetic 标记；Weibull 参数仅用于代码验证。"
        )
    return fit


def _separation_from_csv(path: Path, prediction: PredictionConfig) -> dict[str, Any]:
    df = load_loss_csv(path)
    required = {"frequency_hz", "loss_total_W_m3"}
    if not required.issubset(df.columns):
        return {
            "model_status": "unavailable",
            "reason": "损耗分离需要 loss_total_W_m3 与 frequency_hz",
        }
    hcol = next((c for c in ("loss_hysteresis_W_m3", "hysteresis_loss_W_m3") if c in df.columns), None)
    ecol = next((c for c in ("loss_eddy_W_m3", "eddy_loss_W_m3") if c in df.columns), None)
    rows = []
    for _, row in df.iterrows():
        rows.append(
            separate_losses(
                total_loss_W_m3=float(row["loss_total_W_m3"]),
                hysteresis_loss_W_m3=float(row[hcol]) if hcol else None,
                eddy_loss_W_m3=float(row[ecol]) if ecol else None,
                frequency_hz=float(row["frequency_hz"]),
            )
        )
    excess = [r["excess_loss_W_m3"] for r in rows if r["excess_loss_W_m3"] is not None]
    quality = _source_quality(df)
    warnings = [w for r in rows for w in r.get("warnings", [])]
    if quality == "synthetic_placeholder":
        warnings.append("损耗分离数据含 placeholder/synthetic 标记；仅用于流程验证。")
    return {
        "model_status": "conditional",
        "data_quality": quality,
        "n_rows": len(rows),
        "n_negative_excess": int(sum(1 for v in excess if v < 0)),
        "excess_loss_W_m3": {
            "mean": float(np.mean(excess)) if excess else None,
            "min": float(np.min(excess)) if excess else None,
            "max": float(np.max(excess)) if excess else None,
        },
        "rows": rows,
        "warnings": warnings,
    }


def calibrate(
    prediction: PredictionConfig,
    experiment: ExperimentConfig,
    *,
    base_dir: str | Path,
    run_id: str | None = None,
) -> dict[str, Any]:
    """Run all enabled calibration fits and return a single provenance record."""
    base = Path(base_dir)
    settings: CalibrationSettings = experiment.calibration
    warnings: list[str] = []
    phi, phi_source, phi_warnings = prediction.resolve_magnetic_volume_fraction()
    warnings.extend(phi_warnings)
    result: dict[str, Any] = {
        "model_status": "conditional",
        "run_id": run_id,
        "magnetic_volume_fraction": phi,
        "magnetic_volume_fraction_source": phi_source,
        "settings": settings.model_dump(mode="json"),
        "fits": {},
        "data_files": {},
        "warnings": warnings,
        "created_utc": timestamp_utc(),
    }
    if settings.fit_loop_model and experiment.bh_curve_csv:
        p = _resolve(experiment.bh_curve_csv, base)
        if p and p.is_file():
            df = load_bh_csv(p)
            result["data_files"]["bh_curve_csv"] = {
                "path": str(p),
                "sha256": sha256_file(p),
                "rows": int(len(df)),
            }
            if phi is None:
                result["fits"]["loop"] = {
                    "model_status": "unavailable",
                    "reason": "缺少磁性体积分数，无法拟合宏观回线",
                }
            else:
                result["fits"]["loop"] = fit_tanh_loop(
                    df,
                    magnetic_volume_fraction=float(phi),
                    initial_guess={
                        "Ms_A_m": prediction.material.Ms_A_m or 1.0e6,
                    },
                    train_fraction=settings.train_fraction,
                    random_seed=settings.random_seed,
                )
        else:
            warnings.append(f"B-H 数据不存在：{p}")
    if settings.fit_steinmetz and experiment.loss_csv:
        p = _resolve(experiment.loss_csv, base)
        if p and p.is_file():
            result["fits"]["steinmetz"] = _fit_steinmetz(p, prediction)
            result["data_files"]["loss_csv"] = {
                "path": str(p),
                "sha256": sha256_file(p),
            }
        else:
            warnings.append(f"损耗数据不存在：{p}")
        if settings.fit_loss_separation:
            result["fits"]["loss_separation"] = _separation_from_csv(p, prediction) if p and p.is_file() else {
                "model_status": "unavailable",
                "reason": "损耗数据不存在",
            }
    if settings.fit_weibull and experiment.breakdown_csv:
        p = _resolve(experiment.breakdown_csv, base)
        if p and p.is_file():
            result["fits"]["weibull"] = _fit_breakdown(p)
            result["data_files"]["breakdown_csv"] = {
                "path": str(p),
                "sha256": sha256_file(p),
            }
        else:
            warnings.append(f"击穿数据不存在：{p}")
    fit_statuses = [fit.get("model_status") for fit in result["fits"].values()]
    if not result["fits"]:
        result["model_status"] = "unavailable"
        warnings.append("没有启用或找到任何校准数据。")
    elif all(status == "unavailable" for status in fit_statuses):
        result["model_status"] = "unavailable"
    elif prediction.material.parameter_quality.value in {"D", "UNKNOWN"}:
        result["model_status"] = "demo_only"
        warnings.append(
            "材料证据等级为 D/unknown；校准结果只用于验证拟合流程，不能提升材料参数证据等级。"
        )
    result["warnings"] = list(dict.fromkeys(warnings))
    return result
