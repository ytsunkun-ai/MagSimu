import numpy as np
import pandas as pd
import pytest

from softmag.calibration import fit_tanh_loop, load_bh_csv


def _synthetic_loop():
    Ms, Hc, a, phi = 1.2e6, 100.0, 5000.0, 0.6
    mu0 = 4e-7 * np.pi
    H = np.linspace(-4e5, 4e5, 301)
    Hd = H[::-1]
    B = mu0 * (H + phi * Ms * np.tanh((H - Hc) / a))
    Bd = mu0 * (Hd + phi * Ms * np.tanh((Hd + Hc) / a))
    return pd.DataFrame(
        {
            "H_A_m": np.concatenate([H, Hd]),
            "B_T": np.concatenate([B, Bd]),
            "branch": ["ascending"] * H.size + ["descending"] * Hd.size,
        }
    )


def test_fit_recovers_tanh_parameters():
    fit = fit_tanh_loop(_synthetic_loop(), magnetic_volume_fraction=0.6)
    assert fit["model_status"] == "conditional"
    assert fit["r2_all"] > 0.999
    assert fit["parameters"]["Ms_A_m"] == pytest.approx(1.2e6, rel=0.02)
    assert fit["parameters"]["Hc_A_m"] == pytest.approx(100.0, rel=0.02)
    assert fit["parameters"]["a_A_m"] == pytest.approx(5000.0, rel=0.02)


def test_bh_loader_rejects_too_few_points(tmp_path):
    p = tmp_path / "bad.csv"
    pd.DataFrame({"H_A_m": [0, 1], "B_T": [0, 1]}).to_csv(p, index=False)
    with pytest.raises(ValueError):
        load_bh_csv(p)
