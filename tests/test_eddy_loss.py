import numpy as np
import pandas as pd
import pytest

from softmag.eddy_loss import (
    eddy_loss_density_sphere,
    integrate_eddy_loss_over_distribution,
    steinmetz_fit,
    steinmetz_predict,
)
from softmag.units import skin_depth_m


def test_sphere_eddy_scaling():
    a = eddy_loss_density_sphere(
        diameter_m=20e-6, frequency_hz=1e5, B_peak_T=0.05, resistivity_ohm_m=1.2e-6
    )
    b = eddy_loss_density_sphere(
        diameter_m=40e-6, frequency_hz=1e5, B_peak_T=0.05, resistivity_ohm_m=1.2e-6
    )
    assert b["eddy_loss_density_W_m3"] == pytest.approx(
        4.0 * a["eddy_loss_density_W_m3"], rel=1e-12
    )
    assert a["skin_depth_m"] == pytest.approx(
        skin_depth_m(1e5, 1.2e-6, 1.0), rel=1e-12
    )


def test_distribution_integral_uses_volume_weights():
    d = np.array([1e-5, 2e-5, 4e-5])
    r = integrate_eddy_loss_over_distribution(
        d,
        frequency_hz=1e5,
        B_peak_T=0.05,
        resistivity_ohm_m=1.2e-6,
        weights=np.array([1, 1, 1]),
        weight_basis="volume",
        magnetic_volume_fraction=0.6,
    )
    assert r["eddy_loss_sample_volume_average_W_m3"] is not None
    assert r["diameter_m"]["d50"] > 1e-5


def test_steinmetz_roundtrip():
    rows = []
    for f in (5e4, 1e5, 2e5, 4e5):
        for B in (0.02, 0.05, 0.1):
            rows.append({"frequency_hz": f, "B_peak_T": B, "loss_W_m3": 2e-4 * f**1.2 * B**1.8})
    df = pd.DataFrame(rows)
    fit = steinmetz_fit(df)
    assert fit["model_status"] == "conditional"
    assert fit["r2"] > 0.999
    pred = steinmetz_predict(fit, frequency_hz=1e5, B_peak_T=0.05)
    assert pred["loss_W_m3"] > 0
