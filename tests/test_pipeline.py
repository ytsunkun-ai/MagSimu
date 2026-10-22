from pathlib import Path

import yaml

from softmag.pipeline import run_all


def test_end_to_end_demo(tmp_path):
    root = Path.cwd()
    src = root / "configs"
    mat = yaml.safe_load((src / "material.yaml").read_text(encoding="utf-8"))
    dem = yaml.safe_load((src / "dem.yaml").read_text(encoding="utf-8"))
    mm = yaml.safe_load((src / "micromagnetics.yaml").read_text(encoding="utf-8"))
    exp = yaml.safe_load((src / "experiment.yaml").read_text(encoding="utf-8"))
    dem["n_particles"] = 100
    dem["target_particle_volume_fraction"] = 0.55
    dem["initial_particle_volume_fraction"] = 0.25
    dem["max_relax_iterations"] = 300
    mm["mode"] = "fallback"
    exp["bh_curve_csv"] = str((root / "data/raw/example_bh_placeholder.csv").resolve())
    exp["loss_csv"] = str((root / "data/raw/example_loss_placeholder.csv").resolve())
    exp["breakdown_csv"] = str((root / "data/raw/example_breakdown_placeholder.csv").resolve())
    for name, payload in (("material.yaml", mat), ("dem.yaml", dem), ("micromagnetics.yaml", mm), ("experiment.yaml", exp)):
        (tmp_path / name).write_text(yaml.safe_dump(payload, allow_unicode=True), encoding="utf-8")
    project = {
        "schema_version": "1.0",
        "run_id": "pytest",
        "prediction_config": str(tmp_path / "material.yaml"),
        "dem_config": str(tmp_path / "dem.yaml"),
        "micromagnetics_config": str(tmp_path / "micromagnetics.yaml"),
        "experiment_config": str(tmp_path / "experiment.yaml"),
        "output_dir": str(tmp_path / "results"),
        "steps": [
            "validate", "particles", "dem", "structure", "micromagnetics",
            "homogenize", "losses", "dielectric", "calibrate", "uncertainty", "report",
        ],
        "random_seed": 4,
    }
    project_path = tmp_path / "project.yaml"
    project_path.write_text(yaml.safe_dump(project, allow_unicode=True), encoding="utf-8")
    result = run_all(project_path, run_id="pytest", log=lambda *a, **k: None)
    assert result["micromagnetics"]["model_status"] == "fallback_reduced_order"
    assert result["homogenization"]["effective_relative_permeability"] is not None
    assert result["losses"]["hysteresis_loss_W_m3"] is not None
    assert result["losses"]["eddy_loss_W_m3"] is not None
    assert result["dielectric"]["model_status"] in {"conditional", "unavailable"}
    assert result["calibration"]["model_status"] == "demo_only"
    assert result["uncertainty"]["model_status"] == "conditional"
    report = Path(result["report"]["markdown"])
    assert report.is_file()
    text = report.read_text(encoding="utf-8")
    assert "D 级" in text
    assert "不是最大局部电场" in text
    assert (Path(result["run_dir"]) / "manifest.json").is_file()
