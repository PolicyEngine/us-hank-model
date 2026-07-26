"""Two-asset HANK (Auclert, Bardoczy, Rognlie & Straub, Econometrica 2021).

Builds the two-asset HANK DAG from ``sequence_jacobian.examples.two_asset``
at the paper's PRODUCTION grid sizes (nB=50, nA=70, nK=50; the packaged
example's ``dag()`` uses coarse demo grids), solves the steady state and the
general-equilibrium Jacobian, and produces linear impulse responses to
stylized shocks.

Exogenous instruments in this DAG are ``rstar`` (monetary policy shock in the
Taylor rule), ``G`` (real government spending), and ``Z`` (TFP). There is NO
exogenous transfer or tax-rate instrument: the labor tax rate ``tax`` is
determined endogenously by the ``fiscal`` block to balance the government
budget (tax = (r*Bg + G)/(w*N)), so a G shock is implicitly tax-financed.
Shock kinds exposed here: 'monetary', 'fiscal_spending', 'productivity'.
"""

from __future__ import annotations

import numpy as np
from sequence_jacobian import combine, create_model
from sequence_jacobian.examples import two_asset as ta

# Paper calibration with PRODUCTION grids (nB=50, nA=70, nK=50), unlike the
# packaged example dag() which uses nB=10, nA=16, nK=4 for demo speed.
PRODUCTION_CALIBRATION = {
    'Y': 1., 'N': 1.0, 'K': 10., 'r': 0.0125, 'rstar': 0.0125,
    'tot_wealth': 14, 'delta': 0.02, 'pi': 0.,
    'kappap': 0.1, 'muw': 1.1, 'Bh': 1.04, 'Bg': 2.8, 'G': 0.2, 'eis': 0.5,
    'frisch': 1, 'chi0': 0.25, 'chi2': 2, 'epsI': 4, 'omega': 0.005,
    'kappaw': 0.1, 'phi': 1.5,
    'nZ': 3, 'nB': 50, 'nA': 70, 'nK': 50,
    'bmax': 50, 'amax': 4000, 'kmax': 1, 'rho_z': 0.966, 'sigma_z': 0.92,
}

UNKNOWNS = ['r', 'w', 'Y']
TARGETS = ['asset_mkt', 'fisher', 'wnkpc']
EXOGENOUS = ['rstar', 'Z', 'G']

SHOCK_KINDS = {
    'monetary': 'rstar',
    'fiscal_spending': 'G',
    'productivity': 'Z',
}

_CACHE: dict = {}


def build_models():
    """Build (ss_model, model) DAGs, mirroring two_asset.dag() block structure."""
    if 'models' in _CACHE:
        return _CACHE['models']
    household = ta.hh.add_hetinputs([ta.income, ta.make_grids])
    production = combine([ta.labor, ta.investment])
    production_solved = production.solved(
        unknowns={'Q': 1., 'K': 10.}, targets=['inv', 'val'],
        solver='broyden_custom')
    blocks = [household, ta.pricing_solved, ta.arbitrage_solved,
              production_solved, ta.dividend, ta.taylor, ta.fiscal,
              ta.share_value, ta.finance, ta.wage, ta.union, ta.mkt_clearing]
    model = create_model(blocks, name='Two-Asset HANK')
    blocks_ss = [household, ta.partial_ss, ta.dividend, ta.taylor, ta.fiscal,
                 ta.share_value, ta.finance, ta.union_ss, ta.mkt_clearing]
    ss_model = create_model(blocks_ss, name='Two-Asset HANK SS')
    _CACHE['models'] = (ss_model, model)
    return ss_model, model


def solve_steady_state(use_cache: bool = True):
    """Solve the general-equilibrium steady state at production grids.

    Calibrates (beta, chi1) to hit the asset-market clearing and liquid-asset
    (B = 1.04) targets. Slow (tens of seconds at full grids); the result is
    cached at module level.
    """
    if use_cache and 'ss' in _CACHE:
        return _CACHE['ss']
    ss_model, model = build_models()
    cali = ss_model.solve_steady_state(
        PRODUCTION_CALIBRATION,
        unknowns={'beta': 0.976, 'chi1': 6.5},
        targets={'asset_mkt': 0., 'B': 'Bh'},
        solver='broyden_custom')
    ss = model.steady_state(cali)
    _CACHE['ss'] = ss
    return ss


def solve_jacobian(ss, T: int = 300, use_cache: bool = True):
    """Solve the T x T general-equilibrium Jacobian around ``ss`` (cached by (id(ss), T))."""
    key = ('jac', id(ss), T)
    if use_cache and key in _CACHE:
        return _CACHE[key]
    _, model = build_models()
    G = model.solve_jacobian(ss, UNKNOWNS, TARGETS, EXOGENOUS, T=T)
    if use_cache:
        _CACHE[key] = G
    return G


def shock(kind: str, size: float, persistence: float, T: int = 300,
          ss=None, G_jac=None) -> dict:
    """Linear IRFs to an AR(1) shock ``size * persistence**t``.

    kind: 'monetary' (rstar, size in levels, e.g. -0.0025 = -25bp quarterly
    easing), 'fiscal_spending' (G, size in levels of G, e.g. 0.01 = 1% of
    ss Y since Y=1), or 'productivity' (Z). There is no exogenous transfer or
    tax-rate instrument in this DAG (see module docstring).

    Returns dict with 'Y', 'C', 'I' as % deviations from steady state,
    'pi', 'r' as level (quarterly rate) deviations in percentage points,
    'shock' (the exogenous path in levels), and 'shock_input' (variable name).
    """
    if kind not in SHOCK_KINDS:
        raise ValueError(f"unknown shock kind {kind!r}; available: "
                         f"{sorted(SHOCK_KINDS)} (no transfer/tax-rate "
                         "instrument exists in this DAG; tax is endogenous)")
    if ss is None:
        ss = solve_steady_state()
    if G_jac is None:
        G_jac = solve_jacobian(ss, T=T)
    inp = SHOCK_KINDS[kind]
    dx = size * persistence ** np.arange(T)
    out = {'shock': dx, 'shock_input': inp}
    for v in ('Y', 'C', 'I'):
        out[v] = 100 * (G_jac[v][inp] @ dx) / ss[v]
    for v in ('pi', 'r'):
        out[v] = 100 * (G_jac[v][inp] @ dx)
    return out
