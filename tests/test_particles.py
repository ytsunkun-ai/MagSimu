import numpy as np
import pytest

from softmag.particle_generator import ParticleSizeDistribution


def test_volume_quantiles_preserve_d50_and_shift_number_distribution():
    dist = ParticleSizeDistribution.from_quantiles(
        1.0e-5, 2.0e-5, 3.6e-5, basis="volume"
    )
    assert dist.quantile(0.5, weighting="volume") == pytest.approx(2.0e-5, rel=1e-6)
    # Number-weighted d50 must be smaller than the volume-weighted d50.
    assert dist.quantile(0.5, weighting="number") < 2.0e-5
    v = dist.stats(weighting="volume")
    n = dist.stats(weighting="number")
    assert v["d50_m"] > n["d50_m"]
    assert v["basis"] == "volume"
    assert n["basis"] == "number"


def test_sampling_reproducible_and_positive():
    dist = ParticleSizeDistribution.from_quantiles(
        1.0e-5, 2.0e-5, 3.6e-5, basis="volume"
    )
    a = dist.sample(20, np.random.default_rng(1), weighting="number")
    b = dist.sample(20, np.random.default_rng(1), weighting="number")
    assert np.all(a > 0)
    assert np.allclose(a, b)
