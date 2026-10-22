import pytest

from softmag.schemas import ConfigError, load_config


def test_load_demo_config():
    p = load_config("configs/material.yaml")
    assert p.material.material_id
    assert p.material.parameter_quality.value == "D"
    assert p.resolve_magnetic_volume_fraction()[0] == pytest.approx(0.6)
    assert p.target_particle_volume_fraction()[0] == pytest.approx(0.6)


def test_invalid_quantile_order_is_rejected(tmp_path):
    text = """
schema_version: "1.0"
material: {material_id: test}
particle:
  size_distribution_type: measured
  d10_m: 3.0e-5
  d50_m: 2.0e-5
  d90_m: 1.0e-5
"""
    p = tmp_path / "bad.yaml"
    p.write_text(text, encoding="utf-8")
    with pytest.raises(ConfigError):
        load_config(p)


def test_missing_parameter_is_reported_not_filled(tmp_path):
    text = """
schema_version: "1.0"
material: {material_id: test}
particle:
  size_distribution_type: monodisperse
  d50_m: 2.0e-5
"""
    p = tmp_path / "partial.yaml"
    p.write_text(text, encoding="utf-8")
    cfg = load_config(p)
    missing = cfg.missing_parameters()
    assert "material.Ms_A_m" in missing["micromagnetics"]
    assert "material.resistivity_ohm_m" in missing["eddy_loss"]
