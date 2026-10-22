import math

import numpy as np
import pytest

from softmag import units
from softmag.bh_curve import closure_report, loop_energy_density
from softmag.homogenization import (
    TanhLoopParameters,
    bruggeman,
    default_tanh_parameters,
    maxwell_garnett,
    tanh_major_loop,
    tanh_minor_loop,
)


def test_unit_conversions_roundtrip():
    h = units.oe_to_a_per_m(10.0)
    assert h == pytest.approx(10.0 * 1000.0 / (4.0 * math.pi))
    assert units.a_per_m_to_oe(h) == pytest.approx(10.0)
    assert units.gauss_to_tesla(10000.0) == pytest.approx(1.0)


def test_amplitude_to_peak():
    value, warning = units.amplitude_to_peak(1.0, "peak")
    assert value == 1.0 and warning is None
    value, warning = units.amplitude_to_peak(1.0, "amplitude")
    assert value == 1.0 and warning is None
    value, warning = units.amplitude_to_peak(1.0, "rms")
    assert value == pytest.approx(math.sqrt(2.0))


def test_mixing_boundaries():
    for fn in (bruggeman, maxwell_garnett):
        assert fn(0.0, 100.0) == pytest.approx(1.0)
        assert fn(1.0, 100.0) == pytest.approx(100.0)
    assert maxwell_garnett(0.3, 100.0) < bruggeman(0.3, 100.0)


def test_tanh_major_loop_closed_and_area_positive():
    p = TanhLoopParameters(Ms_A_m=1.2e6, Hc_A_m=100.0, a_A_m=4000.0)
    df, diag = tanh_major_loop(p, Hmax_A_m=4e5, magnetic_volume_fraction=0.6)
    assert diag["closure"]["closed"]
    assert diag["loop_energy_density_J_m3_per_cycle"] > 0
    rep = closure_report(df["H_A_m"], df["B_T"])
    assert rep["closed"]
    area, _ = loop_energy_density(df["H_A_m"], df["B_T"])
    assert area > 0


def test_minor_loop_peak_and_closure():
    p = default_tanh_parameters(
        1.25e6,
        coercivity_A_m=100.0,
        magnetic_volume_fraction=0.6,
        target_initial_permeability=150.0,
    )
    for Bp in (0.01, 0.05, 0.1):
        df, diag = tanh_minor_loop(p, B_peak_T=Bp, magnetic_volume_fraction=0.6)
        assert diag["closure"]["closed"]
        assert np.max(np.abs(df["B_T"])) <= Bp * 1.01
        assert diag["loop_energy_density_J_m3_per_cycle"] > 0
