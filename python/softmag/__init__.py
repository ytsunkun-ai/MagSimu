"""Soft magnetic powder-core multi-scale performance predictor.

The package is a research prototype.  It deliberately separates:

* L0 analytical baselines (:mod:`softmag.eddy_loss`, :mod:`softmag.homogenization`),
* L1 structure-aware models (:mod:`softmag.dem_runner`, :mod:`softmag.microstructure`),
* L2 local high-fidelity models (Julia / MicroMagnetic.jl bridge,
  :mod:`softmag.micromagnetics`),
* L3 calibration (:mod:`softmag.calibration`),
* reporting and uncertainty (:mod:`softmag.report`, :mod:`softmag.uncertainty`).

Every result carries a ``model_status`` and a list of assumptions.  The code never
silently turns a placeholder parameter into a validated material prediction.
"""

from __future__ import annotations

__version__ = "0.1.0"
__all__ = ["__version__"]
