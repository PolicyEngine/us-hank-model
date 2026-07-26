"""One-asset HANK (fast variant), same interface as ``us_hank.model``.

Uses ``sequence_jacobian.examples.hank``. Exogenous instruments are ``rstar``
and ``Z`` only — this DAG has no G, transfer, or tax-rate instrument
(Tax = r*B is endogenous), so only 'monetary' and 'productivity' shocks are
available. There is no capital, so no 'I' in the IRFs.
"""

from __future__ import annotations

import numpy as np
from sequence_jacobian.examples import hank as oa

UNKNOWNS = ['w', 'Y', 'pi']
TARGETS = ['asset_mkt', 'goods_mkt', 'nkpc_res']
EXOGENOUS = ['rstar', 'Z']

SHOCK_KINDS = {'monetary': 'rstar', 'productivity': 'Z'}

_CACHE: dict = {}


def solve_steady_state(use_cache: bool = True):
    if use_cache and 'ss' in _CACHE:
        return _CACHE['ss']
    ss_model, ss, model, unknowns, targets, exogenous = oa.dag()
    _CACHE['model'] = model
    _CACHE['ss'] = ss
    return ss


def solve_jacobian(ss, T: int = 300, use_cache: bool = True):
    key = ('jac', id(ss), T)
    if use_cache and key in _CACHE:
        return _CACHE[key]
    if 'model' not in _CACHE:
        solve_steady_state()
    G = _CACHE['model'].solve_jacobian(ss, UNKNOWNS, TARGETS, EXOGENOUS, T=T)
    if use_cache:
        _CACHE[key] = G
    return G


def shock(kind: str, size: float, persistence: float, T: int = 300,
          ss=None, G_jac=None) -> dict:
    """Linear IRFs; 'Y' and 'C' in % deviations, 'pi' and 'r' in percentage
    points (quarterly rates). 'fiscal_spending' is not available in the
    one-asset DAG (no exogenous G)."""
    if kind not in SHOCK_KINDS:
        raise ValueError(f"unknown shock kind {kind!r} for one-asset HANK; "
                         f"available: {sorted(SHOCK_KINDS)} (no exogenous G, "
                         "transfer, or tax-rate instrument in this DAG)")
    if ss is None:
        ss = solve_steady_state()
    if G_jac is None:
        G_jac = solve_jacobian(ss, T=T)
    inp = SHOCK_KINDS[kind]
    dx = size * persistence ** np.arange(T)
    out = {'shock': dx, 'shock_input': inp}
    for v in ('Y', 'C'):
        out[v] = 100 * (G_jac[v][inp] @ dx) / ss[v]
    for v in ('pi', 'r'):
        out[v] = 100 * (G_jac[v][inp] @ dx)
    return out
