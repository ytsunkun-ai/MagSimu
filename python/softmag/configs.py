"""Configuration models for the DEM, Julia/micromagnetics, experiment and project layers.

The material/particle/excitation contract lives in :mod:`softmag.schemas`.  This
module holds *engine* settings that are intentionally separate, because they can
change when a different DEM solver or micromagnetics package is used without
touching the material data contract.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator

from .io_utils import read_yaml, write_yaml
from .schemas import ConfigError


class _Config(BaseModel):
    model_config = ConfigDict(extra="forbid", validate_assignment=True)


class DemConfig(_Config):
    """Settings for the built-in simplified DEM and optional external DEM import."""

    engine: Literal["simplified", "external"] = "simplified"
    n_particles: int = 1200
    target_particle_volume_fraction: float | None = None
    initial_particle_volume_fraction: float = 0.25
    max_relax_iterations: int = 2500
    relaxation_factor: float = 0.55
    overlap_tolerance_rel: float = 2.0e-3
    max_compaction_steps: int = 20
    compaction_step_fraction: float = 0.03
    contact_gap_tolerance_rel: float = 0.08
    boundary: Literal["periodic"] = "periodic"
    youngs_modulus_Pa: float = 2.0e11
    poisson_ratio: float = 0.30
    friction_coefficient: float = 0.30
    restitution: float = 0.50
    target_pressure_Pa: float | None = None
    random_seed: int = 12345
    external_particles_csv: str | None = None
    external_contacts_csv: str | None = None
    rve_fractions: list[float] = Field(default_factory=lambda: [0.25, 0.35, 0.5, 0.75, 1.0])
    rve_repeats: int = 5
    notes: str = ""

    @field_validator("n_particles")
    @classmethod
    def _n(cls, value: int) -> int:
        if not 1 <= value <= 200_000:
            raise ValueError("n_particles 应位于 1 .. 200000")
        return int(value)

    @field_validator(
        "target_particle_volume_fraction", "initial_particle_volume_fraction"
    )
    @classmethod
    def _fraction(cls, value: float | None) -> float | None:
        if value is None:
            return None
        value = float(value)
        if not 0.02 <= value <= 0.72:
            raise ValueError("体积分数应位于 0.02 .. 0.72（球形颗粒随机堆积上限约 0.64）")
        return value

    @field_validator(
        "overlap_tolerance_rel",
        "compaction_step_fraction",
        "contact_gap_tolerance_rel",
    )
    @classmethod
    def _small_positive(cls, value: float) -> float:
        value = float(value)
        if not 0.0 < value < 0.5:
            raise ValueError("该参数应位于 (0, 0.5)")
        return value

    @field_validator("relaxation_factor")
    @classmethod
    def _relax(cls, value: float) -> float:
        value = float(value)
        if not 0.0 < value <= 1.0:
            raise ValueError("relaxation_factor 应位于 (0, 1]")
        return value

    @field_validator("poisson_ratio")
    @classmethod
    def _poisson(cls, value: float) -> float:
        value = float(value)
        if not 0.0 <= value < 0.5:
            raise ValueError("poisson_ratio 应位于 [0, 0.5)")
        return value

    @field_validator("friction_coefficient", "restitution")
    @classmethod
    def _unit(cls, value: float, info: Any) -> float:
        value = float(value)
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"{info.field_name} 应位于 [0, 1]")
        return value

    @field_validator("random_seed")
    @classmethod
    def _seed(cls, value: int) -> int:
        if value < 0:
            raise ValueError("random_seed 必须是非负整数")
        return int(value)

    @field_validator("rve_fractions")
    @classmethod
    def _rve(cls, value: list[float]) -> list[float]:
        if not value:
            raise ValueError("rve_fractions 不能为空")
        out = []
        for item in value:
            item = float(item)
            if not 0.1 <= item <= 1.0:
                raise ValueError("rve_fractions 的每个值应位于 [0.1, 1.0]")
            out.append(item)
        return sorted(set(out))

    @model_validator(mode="after")
    def _order(self) -> "DemConfig":
        if (
            self.target_particle_volume_fraction is not None
            and self.initial_particle_volume_fraction
            >= self.target_particle_volume_fraction
        ):
            raise ValueError(
                "initial_particle_volume_fraction 必须小于 target_particle_volume_fraction"
            )
        if self.engine == "external" and not self.external_particles_csv:
            raise ValueError("engine=external 时必须给出 external_particles_csv")
        return self


class MicromagneticsConfig(_Config):
    """Settings for the Python -> Julia micromagnetics job."""

    package: str = "MicroMagnetic.jl"
    mode: Literal["auto", "julia", "fallback"] = "auto"
    julia_executable: str = "julia"
    timeout_s: int = 3600
    particle_shape: Literal["sphere", "box", "cluster"] = "sphere"
    mesh_nx: int = 64
    mesh_ny: int = 16
    mesh_nz: int = 16
    dx_m: float = 5.0e-9
    dy_m: float = 5.0e-9
    dz_m: float = 5.0e-9
    n_cluster_particles: int = 7
    Ms_A_m: float | None = None
    exchange_A_J_m: float | None = None
    anisotropy_K_J_m3: float | None = None
    damping_alpha: float | None = None
    demag: bool = True
    m0: tuple[float, float, float] = (-1.0, 0.0, 0.0)
    stopping_dmdt: float = 0.01
    hmax_A_m: float = 4.0e5
    n_field_steps: int = 41
    sweep: Literal["full_loop", "up_only"] = "full_loop"
    fallback_model: Literal["tanh_major_loop"] = "tanh_major_loop"
    loop_initial: dict[str, float] | None = None
    output_subdir: str = "hysteresis"
    notes: str = ""

    @field_validator("mesh_nx", "mesh_ny", "mesh_nz")
    @classmethod
    def _cells(cls, value: int, info: Any) -> int:
        if value < 1:
            raise ValueError(f"{info.field_name} 必须 >= 1")
        return int(value)

    @field_validator("dx_m", "dy_m", "dz_m", "hmax_A_m", "stopping_dmdt")
    @classmethod
    def _positive(cls, value: float, info: Any) -> float:
        value = float(value)
        if value <= 0:
            raise ValueError(f"{info.field_name} 必须 > 0")
        return value

    @field_validator("n_field_steps")
    @classmethod
    def _steps(cls, value: int) -> int:
        if value < 5:
            raise ValueError("n_field_steps 至少为 5（完整回线）")
        return int(value)

    @field_validator("damping_alpha")
    @classmethod
    def _alpha(cls, value: float | None) -> float | None:
        if value is None:
            return None
        value = float(value)
        if not 0.0 < value <= 1.0:
            raise ValueError("damping_alpha 应位于 (0, 1]")
        return value

    @field_validator("m0")
    @classmethod
    def _m0(cls, value: tuple[float, float, float]) -> tuple[float, float, float]:
        if len(value) != 3 or sum(v * v for v in value) == 0:
            raise ValueError("m0 必须是长度为 3 的非零向量")
        norm = sum(v * v for v in value) ** 0.5
        return tuple(float(v / norm) for v in value)  # type: ignore[return-value]

    @field_validator("n_cluster_particles")
    @classmethod
    def _cluster(cls, value: int) -> int:
        if value < 1:
            raise ValueError("n_cluster_particles 必须 >= 1")
        return int(value)


class CalibrationSettings(_Config):
    fit_loop_model: bool = True
    fit_loss_separation: bool = True
    fit_steinmetz: bool = True
    fit_weibull: bool = True
    bootstrap_samples: int = 300
    random_seed: int = 12345
    train_fraction: float = 0.7
    group_column: str | None = None
    notes: str = ""

    @field_validator("bootstrap_samples")
    @classmethod
    def _bootstrap(cls, value: int) -> int:
        if not 0 <= value <= 100_000:
            raise ValueError("bootstrap_samples 应位于 0 .. 100000")
        return int(value)

    @field_validator("train_fraction")
    @classmethod
    def _train(cls, value: float) -> float:
        value = float(value)
        if not 0.1 <= value <= 0.95:
            raise ValueError("train_fraction 应位于 [0.1, 0.95]")
        return value


class ExperimentConfig(_Config):
    bh_curve_csv: str | None = None
    loss_csv: str | None = None
    breakdown_csv: str | None = None
    calibration: CalibrationSettings = Field(default_factory=CalibrationSettings)
    notes: str = ""


class ProjectConfig(_Config):
    """Optional orchestrator that points at the individual config files."""

    schema_version: str = "1.0"
    run_id: str | None = None
    prediction_config: str = "configs/material.yaml"
    dem_config: str | None = "configs/dem.yaml"
    micromagnetics_config: str | None = "configs/micromagnetics.yaml"
    experiment_config: str | None = "configs/experiment.yaml"
    output_dir: str = "results"
    steps: list[str] = Field(
        default_factory=lambda: [
            "validate",
            "particles",
            "dem",
            "structure",
            "micromagnetics",
            "homogenize",
            "losses",
            "dielectric",
            "report",
        ]
    )
    random_seed: int = 12345
    notes: str = ""

    @field_validator("steps")
    @classmethod
    def _steps(cls, value: list[str]) -> list[str]:
        allowed = {
            "validate",
            "particles",
            "dem",
            "structure",
            "micromagnetics",
            "homogenize",
            "losses",
            "dielectric",
            "calibrate",
            "uncertainty",
            "report",
        }
        unknown = sorted(set(value) - allowed)
        if unknown:
            raise ValueError(f"未知步骤 {unknown}; 允许 {sorted(allowed)}")
        return value


def _load_yaml_model(path: str | Path, model: type[BaseModel]) -> Any:
    path = Path(path)
    if not path.is_file():
        raise ConfigError(f"配置文件不存在：{path}")
    try:
        data = read_yaml(path)
    except Exception as exc:  # noqa: BLE001
        raise ConfigError(f"无法读取 YAML 配置 {path}: {exc}") from exc
    if data is None:
        raise ConfigError(f"配置文件为空：{path}")
    try:
        return model.model_validate(data)
    except ValidationError as exc:
        lines = [f"配置校验失败：{path}"]
        for err in exc.errors():
            loc = ".".join(str(p) for p in err.get("loc", ()))
            lines.append(f"  - {loc}: {err.get('msg', 'invalid')}")
        raise ConfigError("\n".join(lines)) from exc


def load_dem_config(path: str | Path) -> DemConfig:
    return _load_yaml_model(path, DemConfig)


def load_micromagnetics_config(path: str | Path) -> MicromagneticsConfig:
    return _load_yaml_model(path, MicromagneticsConfig)


def load_experiment_config(path: str | Path) -> ExperimentConfig:
    return _load_yaml_model(path, ExperimentConfig)


def load_project(path: str | Path) -> tuple[ProjectConfig, Path]:
    """Load a project file and resolve its relative paths against its directory."""
    path = Path(path).resolve()
    project = _load_yaml_model(path, ProjectConfig)
    base = path.parent

    def _resolve(value: str | None) -> str | None:
        if value is None:
            return None
        p = Path(value)
        return str(p if p.is_absolute() else (base / p).resolve())

    project.prediction_config = _resolve(project.prediction_config)  # type: ignore[assignment]
    project.dem_config = _resolve(project.dem_config)
    project.micromagnetics_config = _resolve(project.micromagnetics_config)
    project.experiment_config = _resolve(project.experiment_config)
    project.output_dir = _resolve(project.output_dir) or str((base / "results").resolve())  # type: ignore[assignment]
    return project, path


def save_config_model(model: BaseModel, path: str | Path) -> Path:
    return write_yaml(path, model.model_dump(mode="json"))
