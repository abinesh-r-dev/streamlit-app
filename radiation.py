"""
radiation.py
Vectorised UNSCEAR 1988 five-step stoichiometric radiation assessment.
All functions accept scalar or NumPy array inputs.
"""

from __future__ import annotations
import numpy as np
from config import (TH_ACTIVITY, U_ACTIVITY, CF_TH, CF_U, CF_K,
                    TISSUE_WT, HBRA_THRESHOLD)


def compute(monazite_pct, k40, occupancy: float) -> dict:
    """
    Compute all seven radiological parameters from monazite percentage.

    Parameters
    ----------
    monazite_pct : scalar or array-like  (weight %)
    k40          : scalar or array-like  (Bq/kg)
    occupancy    : float                  (dimensionless, 0–1)

    Returns
    -------
    dict with keys: a_th, a_u, dose_ngy, gamma, ra_eq, aed, elcr
    """
    w     = np.asarray(monazite_pct, dtype=float) / 100.0
    a_th  = w * TH_ACTIVITY
    a_u   = w * U_ACTIVITY
    dose  = CF_TH * a_th + CF_U * a_u + CF_K * np.asarray(k40, dtype=float)
    gamma = (dose / 1000.0) * TISSUE_WT
    ra_eq = a_u + 1.43 * a_th + 0.077 * np.asarray(k40, dtype=float)
    aed   = gamma * 8760.0 * occupancy * TISSUE_WT
    elcr  = aed * 0.05 * 70.0 / 1_000_000.0
    return dict(a_th=a_th, a_u=a_u, dose_ngy=dose,
                gamma=gamma, ra_eq=ra_eq, aed=aed, elcr=elcr)


def risk_label(ra_eq: float) -> tuple[str, str]:
    """Return (label, css-class) for a given Ra_eq value."""
    if ra_eq > 740:
        return "Critical (> 740 Bq/kg)", "danger"
    if ra_eq > HBRA_THRESHOLD:
        return "HBRA (> 370 Bq/kg)", "warning"
    return "Safe", "success"


def sweep(monazite_range: tuple[float, float] = (0.0, 2.0),
          k40: float = 350.0, occupancy: float = 0.2,
          n: int = 300) -> dict:
    """Return a parameter sweep across monazite% for What-If analysis."""
    mon = np.linspace(*monazite_range, n)
    m   = compute(mon, k40, occupancy)
    return {"monazite": mon, **m}