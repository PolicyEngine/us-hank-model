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

# ``size`` in shock() is ALWAYS a level change in the driving variable, never
# a percentage. That reads naturally for two of the three kinds and is a trap
# for the third, so state it once here and let callers introspect rather than
# hard-coding a units string of their own:
#
#   monetary        rstar_ss = 0.0125. A level change in the quarterly policy
#                   rate: size=-0.0025 is a 25bp-per-quarter easing. Natural.
#   fiscal_spending G_ss = 0.2, and Y_ss = 1, so a level change in G is also a
#                   share of quarterly GDP: size=0.01 is 1% of GDP. Natural --
#                   but note it is a 5% increase in G itself.
#   productivity    Z_ss = 0.4678, NOT 1. A level change in Z is therefore NOT
#                   a percentage TFP change: size=0.01 is a 2.14% TFP
#                   improvement, not 1%. Use ``size_pct_of_ss`` below, or read
#                   ``shock_pct_of_ss`` off the result, before quoting a
#                   productivity shock as "an x% TFP shock".
SHOCK_UNITS = {
    'monetary': {
        'input': 'rstar',
        'units': 'level change in the quarterly real policy rate '
                 '(-0.0025 = 25bp per quarter easing on impact)',
        'level_equals_pct_of_ss': False,
    },
    'fiscal_spending': {
        'input': 'G',
        'units': 'level change in real government spending; steady-state '
                 'Y = 1, so 0.01 = 1% of quarterly GDP (= a 5% rise in G)',
        'level_equals_pct_of_ss': False,
    },
    'productivity': {
        'input': 'Z',
        'units': 'level change in TFP around Z_ss = 0.468; a level change is '
                 'NOT a percent change -- 0.01 is a 2.14% TFP improvement',
        'level_equals_pct_of_ss': False,
    },
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
    # Pass a copy: the calibration dict is module-level shared state, and the
    # solver adds its solved unknowns to what it is given. Handing over the
    # original would let the first solve seed every later one.
    cali = ss_model.solve_steady_state(
        dict(PRODUCTION_CALIBRATION),
        unknowns={'beta': 0.976, 'chi1': 6.5},
        targets={'asset_mkt': 0., 'B': 'Bh'},
        solver='broyden_custom')
    ss = model.steady_state(cali)
    _CACHE['ss'] = ss
    return ss


def solve_jacobian(ss, T: int = 300, use_cache: bool = True):
    """Solve the T x T general-equilibrium Jacobian around ``ss``.

    Cached on ``(id(ss), T)``. The cache entry keeps a reference to ``ss``
    itself: without it the steady state could be garbage collected, its
    address reused by a *different* steady state, and that second object
    silently handed the first one's Jacobian.
    """
    key = ('jac', id(ss), T)
    if use_cache and key in _CACHE:
        return _CACHE[key][1]
    _, model = build_models()
    G = model.solve_jacobian(ss, UNKNOWNS, TARGETS, EXOGENOUS, T=T)
    if use_cache:
        _CACHE[key] = (ss, G)
    return G


def shock(kind: str, size: float, persistence: float, T: int = 300,
          ss=None, G_jac=None) -> dict:
    """Linear IRFs to an AR(1) shock ``size * persistence**t``.

    ``size`` is a LEVEL change in the driving variable for every kind -- see
    ``SHOCK_UNITS``, and mind that a level change in Z is not a percent change
    in TFP (Z_ss = 0.468, so size=0.01 is a 2.14% TFP shock). The returned
    ``shock_pct_of_ss`` states the shock as a percentage of the driving
    variable's steady state so the reading is never left to the caller.

    kind: 'monetary' (rstar), 'fiscal_spending' (G) or 'productivity' (Z).
    There is no exogenous transfer or tax-rate instrument in this DAG (see
    the module docstring).

    Responses are first-order: the IRF is a matrix-vector product against a
    fixed Jacobian, so it scales EXACTLY with ``size`` and carries no
    state-dependence or asymmetry. Doubling ``size`` doubles every response;
    that is a property of the method, not a finding about the economy.

    Returns a dict with:
      'Y', 'C', 'I'   % deviations from steady state
      'pi', 'r'       level deviations of QUARTERLY rates, percentage points
      'tax'           labour tax rate, level deviation in percentage points.
                      Included because it is the financing: this DAG has a
                      fixed bond supply Bg, so a G shock is paid for by this
                      tax path, period by period, with no deficit. Reading a
                      spending IRF without it invites a deficit-financed
                      interpretation the model cannot produce.
      'shock'         the exogenous path in levels
      'shock_input'   name of the driving variable
      'shock_pct_of_ss'  the same path as % of that variable's steady state
      'units'         human-readable units string for ``size``
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
    out = {
        'shock': dx,
        'shock_input': inp,
        'shock_pct_of_ss': 100 * dx / float(ss[inp]),
        'units': SHOCK_UNITS[kind]['units'],
    }
    for v in ('Y', 'C', 'I'):
        out[v] = 100 * (G_jac[v][inp] @ dx) / ss[v]
    for v in ('pi', 'r', 'tax'):
        out[v] = 100 * (G_jac[v][inp] @ dx)
    return out
