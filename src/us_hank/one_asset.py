"""One-asset HANK (fast variant), same interface as ``us_hank.model``.

Uses the blocks of ``sequence_jacobian.examples.hank``, solved on the grid of
the toolkit's own one-asset notebook (``notebooks/hank.ipynb``: 7 income
states, 500 asset points) rather than the packaged ``dag()`` demo grid (2
income states, 10 asset points). The demo grid is not converged: it moves
the impact output response to a -25bp-per-quarter cut (rho 0.61) from 0.191
to 0.246 (+29%) and to a 0.01 TFP shock (rho 0.9) from 0.576 to 0.450
(-22%). The answer is grid-stable from 100 asset points.

Exogenous instruments are ``rstar``
and ``Z`` only — this DAG has no G, transfer, or tax-rate instrument
(Tax = r*B is endogenous), so only 'monetary' and 'productivity' shocks are
available. There is no capital, so no 'I' in the IRFs.
"""

from __future__ import annotations

import numpy as np
from sequence_jacobian import create_model
from sequence_jacobian.examples import hank as oa

#: The toolkit example's calibration with the notebook's grid (nS=7, nA=500)
#: in place of the demo grid (nS=2, nA=10). Every other value is the
#: example's own.
PRODUCTION_CALIBRATION = {
    'r': 0.005, 'rstar': 0.005, 'eis': 0.5, 'frisch': 0.5, 'B': 5.6,
    'mu': 1.2, 'rho_s': 0.966, 'sigma_s': 0.5, 'kappa': 0.1, 'phi': 1.5,
    'Y': 1., 'Z': 1., 'pi': 0., 'nS': 7, 'amax': 150, 'nA': 500,
}

UNKNOWNS = ['w', 'Y', 'pi']
TARGETS = ['asset_mkt', 'goods_mkt', 'nkpc_res']
EXOGENOUS = ['rstar', 'Z']

SHOCK_KINDS = {'monetary': 'rstar', 'productivity': 'Z'}

_CACHE: dict = {}


def solve_steady_state(use_cache: bool = True):
    if use_cache and 'ss' in _CACHE:
        return _CACHE['ss']
    household = oa.hh.add_hetinputs(
        [oa.transfers, oa.wages, oa.make_grids]).add_hetoutputs(
        [oa.labor_supply])
    model = create_model([household, oa.firm, oa.monetary, oa.fiscal,
                          oa.mkt_clearing, oa.nkpc], name='One-Asset HANK')
    ss_model = create_model([household, oa.firm, oa.monetary, oa.fiscal,
                             oa.mkt_clearing, oa.nkpc_ss],
                            name='One-Asset HANK SS')
    cali = ss_model.solve_steady_state(
        dict(PRODUCTION_CALIBRATION), unknowns={'beta': 0.986, 'vphi': 0.8},
        targets={'asset_mkt': 0., 'NE': 1.}, solver='broyden_custom')
    ss = model.steady_state(cali)
    _CACHE['model'] = model
    _CACHE['ss'] = ss
    return ss


def solve_jacobian(ss, T: int = 300, use_cache: bool = True):
    """T x T GE Jacobian around ``ss``, cached on ``(id(ss), T)``.

    The entry keeps a reference to ``ss`` so its address cannot be recycled by
    a later steady state that would then be served this one's Jacobian.
    """
    key = ('jac', id(ss), T)
    if use_cache and key in _CACHE:
        return _CACHE[key][1]
    if 'model' not in _CACHE:
        solve_steady_state()
    G = _CACHE['model'].solve_jacobian(ss, UNKNOWNS, TARGETS, EXOGENOUS, T=T)
    if use_cache:
        _CACHE[key] = (ss, G)
    return G


def shock(kind: str, size: float, persistence: float, T: int = 300,
          ss=None, G_jac=None) -> dict:
    """Linear IRFs; 'Y' and 'C' in % deviations, 'pi' and 'r' in percentage
    points (quarterly rates). 'fiscal_spending' is not available in the
    one-asset DAG (no exogenous G).

    ``size`` is a level change in the driving variable, as in
    ``us_hank.model.shock``; ``shock_pct_of_ss`` reports it as a share of that
    variable's steady state. This DAG's Z_ss differs from the two-asset one,
    so the same ``size`` is a different percentage TFP shock in each.
    """
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
    out = {'shock': dx, 'shock_input': inp,
           'shock_pct_of_ss': 100 * dx / float(ss[inp])}
    for v in ('Y', 'C'):
        out[v] = 100 * (G_jac[v][inp] @ dx) / ss[v]
    for v in ('pi', 'r'):
        out[v] = 100 * (G_jac[v][inp] @ dx)
    return out
