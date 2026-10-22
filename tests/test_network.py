import numpy as np
import pytest

from softmag.microstructure import solve_network


def test_series_chain_resistance():
    sol = solve_network(
        3,
        np.array([0, 1]),
        np.array([1, 2]),
        np.array([1.0, 1.0]),
        np.array([0]),
        np.array([2]),
        length_m=2.0,
        area_m2=1.0,
    )
    assert sol.percolating
    assert sol.effective_resistance == pytest.approx(2.0)
    assert sol.effective_resistivity == pytest.approx(1.0)
    assert sol.effective_conductivity == pytest.approx(1.0)
    assert sol.potential[1] == pytest.approx(0.5)


def test_disconnected_electrodes_are_not_percolating():
    sol = solve_network(
        4,
        np.array([0]),
        np.array([1]),
        np.array([1.0]),
        np.array([0]),
        np.array([3]),
        length_m=1.0,
        area_m2=1.0,
    )
    assert not sol.percolating
    assert sol.effective_resistivity is None


def test_parallel_paths_choose_lower_resistance():
    # Path 0-1-3 and path 0-2-3, each with unit conductance.
    sol = solve_network(
        4,
        np.array([0, 1, 0, 2]),
        np.array([1, 3, 2, 3]),
        np.array([1.0, 1.0, 1.0, 1.0]),
        np.array([0]),
        np.array([3]),
        length_m=1.0,
        area_m2=1.0,
    )
    assert sol.percolating
    assert sol.effective_resistance == pytest.approx(1.0)
