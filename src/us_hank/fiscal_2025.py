"""Fiscal paths from Auclert, Rognlie, and Straub (2025).

Clean-room implementation of the fiscal accounting and shock rules stated in
the published paper. It does not copy the authors' unlicensed replication
repository and does not yet claim to reproduce the heterogeneous-agent IRFs.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Calibration:
    """Published quarterly baseline calibration (Table 1)."""

    annual_real_rate: float = 0.02
    annual_assets_to_gdp: float = 5.0
    annual_debt_to_gdp: float = 1.0
    government_spending_to_gdp: float = 0.20
    taxes_to_gdp: float = 0.22
    impact_mpc: float = 0.20
    beta_low: float = 0.91
    beta_high: float = 1.00
    patient_share: float = 0.49
    beta_redraw_probability: float = 0.01
    markup: float = 1.11
    debt_persistence: float = 0.975
    shock_persistence: float = 0.90

    @property
    def quarterly_real_rate(self) -> float:
        return self.annual_real_rate / 4

    @property
    def debt_level(self) -> float:
        """Debt in units of quarterly GDP when annual-GDP debt is 100%."""
        return 4 * self.annual_debt_to_gdp


PUBLISHED_CALIBRATION = Calibration()
SUPPORTED_KINDS = ("deficit_tax_cut", "deficit_spending")


def fiscal_paths(
    kind: str,
    size: float = 0.01,
    shock_persistence: float = PUBLISHED_CALIBRATION.shock_persistence,
    debt_persistence: float = PUBLISHED_CALIBRATION.debt_persistence,
    T: int = 300,
    quarterly_real_rate: float = PUBLISHED_CALIBRATION.quarterly_real_rate,
) -> dict[str, np.ndarray | str]:
    """Construct the paper's mean-reverting debt-financed fiscal paths.

    ``size`` is in units of steady-state quarterly GDP. Positive ``tax_cut``
    denotes a reduction in tax revenue received by households.
    """
    if kind not in SUPPORTED_KINDS:
        raise ValueError(f"unknown fiscal kind {kind!r}; available: {SUPPORTED_KINDS}")
    if T < 2:
        raise ValueError("T must be at least 2")
    if size < 0:
        raise ValueError("size must be non-negative; direction is set by kind")
    if not 0 <= shock_persistence < 1:
        raise ValueError("shock_persistence must be in [0, 1)")
    if not 0 <= debt_persistence < 1:
        raise ValueError("debt_persistence must be in [0, 1)")

    impulse = size * shock_persistence ** np.arange(T)
    tax_cut = impulse if kind == "deficit_tax_cut" else np.zeros(T)
    spending = impulse if kind == "deficit_spending" else np.zeros(T)
    financing_need = tax_cut + spending

    debt = np.zeros(T)
    lagged_debt = np.zeros(T)
    for t in range(T):
        if t:
            lagged_debt[t] = debt[t - 1]
        debt[t] = debt_persistence * lagged_debt[t] + financing_need[t]

    tax_revenue = spending + (1 + quarterly_real_rate) * lagged_debt - debt
    budget_residual = (
        spending
        + (1 + quarterly_real_rate) * lagged_debt
        - debt
        - tax_revenue
    )
    return {
        "kind": kind,
        "tax_cut": tax_cut,
        "spending": spending,
        "debt": debt,
        "tax_revenue": tax_revenue,
        "budget_residual": budget_residual,
    }


def shock(*args, **kwargs):
    """Refuse to present an unvalidated HA response as a shipped model."""
    raise RuntimeError(
        "fiscal_2025 heterogeneous-agent responses are not available: "
        "the paper omits exact household calibration inputs and Figure 2 "
        "numerical targets, while the companion repository declares no "
        "license. Use fiscal_paths() for tested policy paths; enable shock() "
        "only after licensed numerical replication."
    )
