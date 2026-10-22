import numpy as np

from softmag.dielectric import fit_weibull, minimax_breakdown_voltage
from softmag.microstructure import solve_network


def test_minimax_path_uses_bottleneck_not_maximum_edge():
    # Two paths: one with a very high edge threshold, one with a lower bottleneck.
    i = np.array([0, 1, 0, 2])
    j = np.array([1, 3, 2, 3])
    V = np.array([100.0, 100.0, 3.0, 3.0])
    r = minimax_breakdown_voltage(4, i, j, V, np.array([0]), np.array([3]))
    assert r["available"]
    assert r["breakdown_voltage_V"] == 3.0


def test_disconnected_geometry_is_unavailable():
    r = minimax_breakdown_voltage(
        3, np.array([0]), np.array([1]), np.array([1.0]), np.array([0]), np.array([2])
    )
    assert not r["available"]


def test_weibull_fit_positive_parameters():
    rng = np.random.default_rng(3)
    data = 10.0 * (-np.log(rng.random(200))) ** (1.0 / 5.0)
    fit = fit_weibull(data)
    assert fit["model_status"] == "conditional"
    assert fit["shape_k"] > 0
    assert fit["scale_lambda"] > 0
    assert fit["p10"] < fit["p50"] < fit["p90"]
