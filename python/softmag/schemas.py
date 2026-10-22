"""Validated input data contract for the powder-core predictor.

The schema deliberately uses SI units and explicit ``null`` values.  Missing
parameters are *allowed*: the modelling layers must then report a reduced
``model_status`` instead of inventing numbers.  See the project description,
section 2.4 and Appendix A.
"""

from __future__ import annotations

import hashlib
import json
import math
from enum import Enum
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator

from . import units
from .io_utils import read_yaml, to_jsonable, write_yaml


class ConfigError(RuntimeError):
    """Raised when an input configuration is invalid or inconsistent."""


class EvidenceLevel(str, Enum):
    """Evidence grade for a material parameter (section 4.3)."""

    A = "A"  # direct measurement on the target material + process
    B = "B"  # traceable literature on the same material system
    C = "C"  # similar material / process
    D = "D"  # assumption used only for code tests
    UNKNOWN = "unknown"


class ModelStatus(str, Enum):
    DIRECT = "direct"  # 可直接计算
    CONDITIONAL = "conditional"  # 条件预测
    NEEDS_CALIBRATION = "needs_calibration"  # 需要校准
    UNAVAILABLE = "unavailable"  # 暂不可预测


MODEL_STATUS_LABELS = {
    ModelStatus.DIRECT: "可直接计算",
    ModelStatus.CONDITIONAL: "条件预测",
    ModelStatus.NEEDS_CALIBRATION: "需要校准",
    ModelStatus.UNAVAILABLE: "暂不可预测",
}


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", validate_assignment=True, use_enum_values=False)


class MaterialParameters(_StrictModel):
    material_id: str = Field(min_length=1)
    composition: str | None = None
    density_kg_m3: float | None = None
    Ms_A_m: float | None = None
    exchange_A_J_m: float | None = None
    anisotropy_K_J_m3: float | None = None
    damping_alpha: float | None = None
    magnetostriction_lambda_s: float | None = None
    resistivity_ohm_m: float | None = None
    initial_permeability: float | None = None
    grain_size_m: float | None = None
    heat_treatment: str | None = None
    parameter_source: str = "待填写"
    parameter_quality: EvidenceLevel = EvidenceLevel.UNKNOWN
    source_type: Literal["measured", "literature", "estimated", "assumed", "unknown"] = (
        "unknown"
    )
    source_ref: str | None = None
    uncertainty: dict[str, float] = Field(default_factory=dict)
    notes: str = ""

    @field_validator(
        "density_kg_m3",
        "Ms_A_m",
        "exchange_A_J_m",
        "resistivity_ohm_m",
        "initial_permeability",
        "grain_size_m",
    )
    @classmethod
    def _positive(cls, value: float | None, info: Any) -> float | None:
        return units.require_positive(value, info.field_name)

    @field_validator("anisotropy_K_J_m3")
    @classmethod
    def _nonnegative(cls, value: float | None) -> float | None:
        return units.require_nonnegative(value, "anisotropy_K_J_m3")

    @field_validator("damping_alpha")
    @classmethod
    def _alpha(cls, value: float | None) -> float | None:
        if value is None:
            return None
        value = float(value)
        if not 0.0 < value <= 1.0:
            raise ValueError("damping_alpha 应位于 (0, 1]（无量纲）")
        return value

    @field_validator("uncertainty")
    @classmethod
    def _uncertainty(cls, value: dict[str, float]) -> dict[str, float]:
        for key, item in value.items():
            if not math.isfinite(float(item)) or float(item) < 0:
                raise ValueError(f"uncertainty[{key!r}] 必须是非负有限数")
        return {k: float(v) for k, v in value.items()}


class ParticleParameters(_StrictModel):
    size_distribution_type: Literal[
        "measured", "lognormal", "histogram", "monodisperse", "bimodal", "unknown"
    ] = "unknown"
    d10_m: float | None = None
    d50_m: float | None = None
    d90_m: float | None = None
    quantile_basis: Literal["number", "volume"] = "volume"
    distribution_file: str | None = None
    distribution_weight_type: Literal["number", "volume"] = "number"
    distribution_diameter_unit: Literal["m", "um", "mm"] = "m"
    shape_model: Literal["sphere", "spheroid", "ellipsoid", "irregular", "measured_mesh"] = (
        "sphere"
    )
    aspect_ratio: float | None = None
    measurement_method: str | None = None
    dispersion_state: str | None = None
    notes: str = ""

    @field_validator("d10_m", "d50_m", "d90_m")
    @classmethod
    def _positive_diameter(cls, value: float | None, info: Any) -> float | None:
        return units.require_positive(value, info.field_name)

    @field_validator("aspect_ratio")
    @classmethod
    def _aspect_ratio(cls, value: float | None) -> float | None:
        if value is None:
            return None
        value = float(value)
        if value < 1.0:
            raise ValueError("aspect_ratio 定义为长轴/短轴，必须 >= 1")
        return value

    @model_validator(mode="after")
    def _check_order(self) -> "ParticleParameters":
        available = [self.d10_m, self.d50_m, self.d90_m]
        if all(v is not None for v in available):
            d10, d50, d90 = available  # type: ignore[misc]
            if not (d10 <= d50 <= d90):
                raise ValueError("粒径分位数必须满足 d10 <= d50 <= d90")
        if self.size_distribution_type == "monodisperse" and self.d50_m is None:
            raise ValueError("单分散分布必须给出 d50_m")
        if self.size_distribution_type in {"histogram", "measured"} and (
            self.distribution_file is None
            and not all(v is not None for v in available)
        ):
            raise ValueError(
                "histogram/measured 分布需要 distribution_file，或同时给出 d10/d50/d90"
            )
        return self

    @property
    def has_quantiles(self) -> bool:
        return all(v is not None for v in (self.d10_m, self.d50_m, self.d90_m))


class InsulationParameters(_StrictModel):
    thickness_mean_m: float | None = None
    thickness_std_m: float | None = None
    thickness_distribution: Literal["constant", "normal", "lognormal"] = "lognormal"
    relative_permittivity: float | None = None
    resistivity_ohm_m: float | None = None
    defect_probability: float | None = None
    breakdown_field_V_m: float | None = None
    coating_material: str | None = None
    data_source: str = "待填写"
    notes: str = ""

    @field_validator("thickness_mean_m", "resistivity_ohm_m", "breakdown_field_V_m")
    @classmethod
    def _positive(cls, value: float | None, info: Any) -> float | None:
        return units.require_positive(value, info.field_name)

    @field_validator("thickness_std_m")
    @classmethod
    def _nonnegative(cls, value: float | None) -> float | None:
        return units.require_nonnegative(value, "thickness_std_m")

    @field_validator("relative_permittivity")
    @classmethod
    def _permittivity(cls, value: float | None) -> float | None:
        if value is None:
            return None
        value = float(value)
        if value < 1.0:
            raise ValueError("relative_permittivity 必须 >= 1")
        return value

    @field_validator("defect_probability")
    @classmethod
    def _probability(cls, value: float | None) -> float | None:
        return units.require_fraction(value, "defect_probability")

    @model_validator(mode="after")
    def _std_consistency(self) -> "InsulationParameters":
        if (
            self.thickness_mean_m is not None
            and self.thickness_std_m is not None
            and self.thickness_std_m >= self.thickness_mean_m
        ):
            raise ValueError(
                "thickness_std_m 不应大于等于 thickness_mean_m；否则会产生非物理负厚度"
            )
        return self


class CompactionParameters(_StrictModel):
    target_bulk_density_kg_m3: float | None = None
    target_relative_density: float | None = None
    pressure_Pa: float | None = None
    friction_coefficient: float | None = None
    dem_engine: str = "待选择"
    random_seed: int = 12345
    notes: str = ""

    @field_validator("target_bulk_density_kg_m3", "pressure_Pa")
    @classmethod
    def _nonnegative(cls, value: float | None, info: Any) -> float | None:
        return units.require_nonnegative(value, info.field_name)

    @field_validator("target_relative_density")
    @classmethod
    def _relative_density(cls, value: float | None) -> float | None:
        return units.require_fraction(value, "target_relative_density")

    @field_validator("friction_coefficient")
    @classmethod
    def _friction(cls, value: float | None) -> float | None:
        return units.require_fraction(value, "friction_coefficient")

    @field_validator("random_seed")
    @classmethod
    def _seed(cls, value: int) -> int:
        if value < 0:
            raise ValueError("random_seed 必须是非负整数")
        return int(value)


class MicrostructureParameters(_StrictModel):
    magnetic_volume_fraction: float | None = None
    porosity_fraction: float | None = None
    coating_volume_fraction: float | None = None
    rve_size_m: float | None = None
    notes: str = ""

    @field_validator(
        "magnetic_volume_fraction", "porosity_fraction", "coating_volume_fraction"
    )
    @classmethod
    def _fraction(cls, value: float | None, info: Any) -> float | None:
        return units.require_fraction(value, info.field_name, allow_one=True)

    @field_validator("rve_size_m")
    @classmethod
    def _rve(cls, value: float | None) -> float | None:
        return units.require_positive(value, "rve_size_m")

    @model_validator(mode="after")
    def _phase_sum(self) -> "MicrostructureParameters":
        phi = self.magnetic_volume_fraction
        pore = self.porosity_fraction
        coat = self.coating_volume_fraction
        if phi is not None and pore is not None and phi + pore > 1.0 + 1e-9:
            raise ValueError("magnetic_volume_fraction + porosity_fraction 不能超过 1")
        if (
            phi is not None
            and pore is not None
            and coat is not None
            and phi + pore + coat > 1.0 + 1e-9
        ):
            raise ValueError("磁性相、孔隙与包覆层体积分数之和不能超过 1")
        return self


class ExcitationParameters(_StrictModel):
    frequency_hz: float | None = None
    flux_density_definition: Literal[
        "peak", "amplitude", "peak_to_peak", "rms", "unknown"
    ] = "unknown"
    flux_density_peak_T: float | None = None
    temperature_K: float = 293.15
    waveform: Literal["sinusoidal", "triangular", "square", "arbitrary", "unknown"] = (
        "sinusoidal"
    )
    dc_bias_A_m: float = 0.0
    notes: str = ""

    @field_validator("frequency_hz", "temperature_K")
    @classmethod
    def _positive(cls, value: float | None, info: Any) -> float | None:
        return units.require_positive(value, info.field_name)

    @field_validator("flux_density_peak_T")
    @classmethod
    def _flux(cls, value: float | None) -> float | None:
        return units.require_nonnegative(value, "flux_density_peak_T")

    @field_validator("dc_bias_A_m")
    @classmethod
    def _bias(cls, value: float) -> float:
        return float(value)


class SolverParameters(_StrictModel):
    micromagnetic_package: str = "MicroMagnetic.jl"
    dem_model: str = "待配置"
    mesh_convergence_required: bool = True
    uncertainty_analysis: bool = True
    allow_fallback_models: bool = True
    notes: str = ""


class Provenance(_StrictModel):
    experiment_id: str | None = None
    operator: str | None = None
    date: str | None = None
    notes: str = ""
    source_refs: list[str] = Field(default_factory=list)


class PredictionConfig(_StrictModel):
    """Root input document (Appendix A)."""

    schema_version: str = "1.0"
    material: MaterialParameters
    particle: ParticleParameters
    insulation: InsulationParameters = Field(default_factory=InsulationParameters)
    compaction: CompactionParameters = Field(default_factory=CompactionParameters)
    microstructure: MicrostructureParameters = Field(
        default_factory=MicrostructureParameters
    )
    excitation: ExcitationParameters = Field(default_factory=ExcitationParameters)
    solver: SolverParameters = Field(default_factory=SolverParameters)
    provenance: Provenance = Field(default_factory=Provenance)

    # -- derived values -------------------------------------------------
    def effective_flux_peak_T(self) -> tuple[float | None, str | None]:
        if self.excitation.flux_density_peak_T is None:
            return None, "缺少 excitation.flux_density_peak_T，无法估算涡流损耗"
        value, warning = units.amplitude_to_peak(
            self.excitation.flux_density_peak_T,
            self.excitation.flux_density_definition,
        )
        return value, warning

    def resolve_magnetic_volume_fraction(self) -> tuple[float | None, str, list[str]]:
        """Resolve the magnetic phase fraction with an explicit provenance flag.

        Returns ``(value, source, warnings)``.  A bulk-density-derived value is
        only an *estimate*: coating mass, closed porosity and phase densities are
        not necessarily known.  Never silently equate relative density with the
        magnetic volume fraction (section 2.3).
        """
        phi = self.microstructure.magnetic_volume_fraction
        if phi is not None:
            return float(phi), "input:microstructure.magnetic_volume_fraction", []
        rho_bulk = self.compaction.target_bulk_density_kg_m3
        rho_true = self.material.density_kg_m3
        if rho_bulk is not None and rho_true is not None:
            estimate = float(rho_bulk) / float(rho_true)
            return estimate, "estimated:bulk_density/true_density", [
                "磁性相体积分数由成品体积密度/材料真密度估算；未扣除绝缘层质量与闭孔，"
                "不能视为精确值。若已知各相密度，应直接输入 magnetic_volume_fraction。"
            ]
        return None, "missing", [
            "缺少 magnetic_volume_fraction，且无法用 bulk density / true density 估算；"
            "宏观磁性能和损耗只能给出条件预测。"
        ]

    def target_particle_volume_fraction(self) -> tuple[float | None, str, list[str]]:
        """Particle (solid) fraction used by the DEM compaction step."""
        phi, source, warnings = self.resolve_magnetic_volume_fraction()
        if phi is not None:
            return phi, source, warnings
        rel = self.compaction.target_relative_density
        if rel is not None:
            return (
                float(rel),
                "compaction.target_relative_density",
                [
                    "将目标相对密度直接当作颗粒体积分数；若样品含绝缘层或闭孔，"
                    "该假设会高估颗粒体积分数。"
                ],
            )
        return None, "missing", ["缺少 DEM 压实目标密度（体积分数或相对密度）"]

    def missing_parameters(self) -> dict[str, list[str]]:
        """List inputs required for each model layer, by absolute path."""
        m, p, ins, ex = self.material, self.particle, self.insulation, self.excitation
        missing: dict[str, list[str]] = {}
        eddy: list[str] = []
        if p.d50_m is None and not p.distribution_file and not p.has_quantiles:
            eddy.append("particle.d50_m / particle.distribution_file")
        if m.resistivity_ohm_m is None:
            eddy.append("material.resistivity_ohm_m")
        if ex.frequency_hz is None:
            eddy.append("excitation.frequency_hz")
        if ex.flux_density_peak_T is None:
            eddy.append("excitation.flux_density_peak_T")
        missing["eddy_loss"] = eddy

        mm: list[str] = []
        for name, value in (
            ("material.Ms_A_m", m.Ms_A_m),
            ("material.exchange_A_J_m", m.exchange_A_J_m),
            ("material.anisotropy_K_J_m3", m.anisotropy_K_J_m3),
            ("material.damping_alpha", m.damping_alpha),
            ("particle.d50_m", p.d50_m),
        ):
            if value is None:
                mm.append(name)
        missing["micromagnetics"] = mm

        hom: list[str] = []
        if m.Ms_A_m is None:
            hom.append("material.Ms_A_m")
        phi, _, _ = self.resolve_magnetic_volume_fraction()
        if phi is None:
            hom.append("microstructure.magnetic_volume_fraction")
        missing["homogenization"] = hom

        diel: list[str] = []
        for name, value in (
            ("insulation.thickness_mean_m", ins.thickness_mean_m),
            ("insulation.relative_permittivity", ins.relative_permittivity),
            ("insulation.breakdown_field_V_m", ins.breakdown_field_V_m),
        ):
            if value is None:
                diel.append(name)
        missing["dielectric"] = diel

        cal: list[str] = []
        if m.parameter_quality in {EvidenceLevel.D, EvidenceLevel.UNKNOWN}:
            cal.append("material.parameter_quality (D/unknown -> 仅用于代码验证)")
        missing["calibration"] = cal
        return missing

    def quality_warnings(self) -> list[str]:
        warnings: list[str] = []
        quality = self.material.parameter_quality
        if quality == EvidenceLevel.D:
            warnings.append(
                "材料参数证据等级为 D（调试假设值），结果只能用于代码/流程验证，"
                "不得作为真实材料性能预测。"
            )
        elif quality == EvidenceLevel.UNKNOWN:
            warnings.append(
                "材料参数证据等级未知；请在 material.parameter_source / "
                "parameter_quality 中记录来源。"
            )
        if self.material.source_type in {"assumed", "unknown"}:
            warnings.append("material.source_type 未明确为 measured/literature/estimated。")
        if self.particle.size_distribution_type == "unknown":
            warnings.append("未声明粒径分布类型；默认按分位数拟合对数正态分布。")
        if self.excitation.flux_density_definition == "unknown":
            warnings.append("磁通密度定义未知，已按峰值处理。")
        if self.insulation.thickness_mean_m is None:
            warnings.append("缺少绝缘层厚度；耐压模型只能给出参数化趋势。")
        return warnings

    def contract(self, *, run_id: str | None = None) -> dict[str, Any]:
        """Python -> Julia data contract (section 10.4)."""
        payload = self.model_dump(mode="json")
        digest = hashlib.sha256(
            json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
        ).hexdigest()[:16]
        b_peak, b_warning = self.effective_flux_peak_T()
        phi, phi_source, phi_warnings = self.resolve_magnetic_volume_fraction()
        return {
            "schema_version": self.schema_version,
            "input_id": digest,
            "run_id": run_id,
            "material": {
                "material_id": self.material.material_id,
                "composition": self.material.composition,
                "Ms_A_m": self.material.Ms_A_m,
                "A_J_m": self.material.exchange_A_J_m,
                "K_J_m3": self.material.anisotropy_K_J_m3,
                "alpha": self.material.damping_alpha,
                "resistivity_ohm_m": self.material.resistivity_ohm_m,
                "density_kg_m3": self.material.density_kg_m3,
                "initial_permeability": self.material.initial_permeability,
                "parameter_quality": self.material.parameter_quality.value
                if hasattr(self.material.parameter_quality, "value")
                else str(self.material.parameter_quality),
                "source_type": self.material.source_type,
                "source_ref": self.material.source_ref,
            },
            "geometry": {
                "shape_model": self.particle.shape_model,
                "d10_m": self.particle.d10_m,
                "d50_m": self.particle.d50_m,
                "d90_m": self.particle.d90_m,
                "aspect_ratio": self.particle.aspect_ratio,
                "distribution_file": self.particle.distribution_file,
            },
            "microstructure": {
                "magnetic_volume_fraction": phi,
                "magnetic_volume_fraction_source": phi_source,
                "porosity_fraction": self.microstructure.porosity_fraction,
                "rve_size_m": self.microstructure.rve_size_m,
                "insulation_thickness_mean_m": self.insulation.thickness_mean_m,
                "insulation_thickness_std_m": self.insulation.thickness_std_m,
                "insulation_relative_permittivity": self.insulation.relative_permittivity,
                "insulation_defect_probability": self.insulation.defect_probability,
            },
            "excitation": {
                "frequency_hz": self.excitation.frequency_hz,
                "flux_density_definition": self.excitation.flux_density_definition,
                "flux_density_value_T": self.excitation.flux_density_peak_T,
                "flux_density_peak_T": b_peak,
                "temperature_K": self.excitation.temperature_K,
                "waveform": self.excitation.waveform,
            },
            "solver": self.solver.model_dump(mode="json"),
            "provenance": self.provenance.model_dump(mode="json"),
            "warnings": [
                *self.quality_warnings(),
                *([b_warning] if b_warning else []),
                *phi_warnings,
            ],
        }

    def to_dict(self) -> dict[str, Any]:
        return self.model_dump(mode="json")


# ---------------------------------------------------------------------------
# Loading / saving
# ---------------------------------------------------------------------------
def _format_validation_error(exc: ValidationError) -> str:
    lines = ["输入配置校验失败："]
    for err in exc.errors():
        location = ".".join(str(part) for part in err.get("loc", ()))
        message = err.get("msg", "invalid value")
        lines.append(f"  - {location}: {message}")
    return "\n".join(lines)


def load_config(path: str | Path) -> PredictionConfig:
    path = Path(path)
    if not path.is_file():
        raise ConfigError(f"配置文件不存在：{path}")
    try:
        data = read_yaml(path)
    except Exception as exc:  # noqa: BLE001
        raise ConfigError(f"无法读取 YAML 配置 {path}: {exc}") from exc
    if data is None:
        raise ConfigError(f"配置文件为空：{path}")
    if not isinstance(data, dict):
        raise ConfigError("配置根节点必须是映射（mapping）")
    # Convenience: allow a wrapper key ``prediction:`` in combined files.
    if "prediction" in data and isinstance(data["prediction"], dict):
        data = data["prediction"]
    try:
        return PredictionConfig.model_validate(data)
    except ValidationError as exc:
        raise ConfigError(_format_validation_error(exc)) from exc


def save_config(config: PredictionConfig, path: str | Path) -> Path:
    return write_yaml(path, config.model_dump(mode="json"))


def config_hash(config: PredictionConfig) -> str:
    payload = config.model_dump(mode="json")
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
    ).hexdigest()
