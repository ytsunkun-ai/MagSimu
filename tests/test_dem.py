import numpy as np
import pytest

from softmag.configs import DemConfig
from softmag.dem_runner import generate_packing
from softmag.schemas import load_config


def test_small_dem_reaches_target_and_has_contacts():
    p = load_config("configs/material.yaml")
    cfg = DemConfig(
        n_particles=120,
        target_particle_volume_fraction=0.55,
        initial_particle_volume_fraction=0.25,
        max_relax_iterations=300,
        max_compaction_steps=20,
        random_seed=7,
    )
    rng = np.random.default_rng(7)
    d = np.full(120, 2.0e-5)
    # Polydispersity avoids a crystallised monodisperse artefact.
    d *= np.exp(rng.normal(0.0, 0.15, 120))
    s = generate_packing(d, cfg, target_particle_volume_fraction=0.55, random_seed=7)
    phi = float(s.particles["volume_m3"].sum() / s.box_length_m**3)
    assert 0.50 < phi < 0.60
    assert len(s.contacts) > 0
    for col in ("particle_i", "particle_j", "surface_gap_m", "overlap_m"):
        assert col in s.contacts.columns
    assert s.summary["warnings"]
