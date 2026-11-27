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
#   monetary        rstar_ss = 0.0125. A level change in the intercept of the
#                   quarterly nominal Taylor rule, i = rstar + phi*pi:
#                   size=-0.0025 is a 25bp-per-quarter cut in the intercept,
#                   about 100bp annualised. A 25bp annualised move is
#                   size=-0.000625. The policy rate i itself moves less,
#                   because the rule responds to the inflation the shock
#                   causes: -13bp on impact for size=-0.0025, rho 0.61.
#                   WARNING: the sign of the policy-rate response depends
#                   on the persistence. Above a persistence of about 0.75 an
#                   intercept CUT raises the policy rate on impact: at 0.9
#                   (PolicyEngine Macro's default) size=-0.0025 gives
#                   i = +0.25pp a quarter, about +1pp annualised, because
#                   inflation rises 0.34pp and phi*pi outweighs the cut. It
#                   is still an easing: the ex-ante real rate falls and Y and
#                   C rise. Do not describe it as "a 25bp rate cut"; read the
#                   returned ``i``. See docs/SHOCKS.md, "A monetary easing
#                   can raise the policy rate".
#   fiscal_spending G_ss = 0.2, and Y_ss = 1, so a level change in G is also a
#                   share of quarterly GDP: size=0.01 is 1% of GDP. Natural --
#                   but note it is a 5% increase in G itself.
#   productivity    Z_ss = 0.4678, NOT 1. A level change in Z is therefore NOT
#                   a percentage TFP change: size=0.01 is a 2.14% TFP
#                   improvement, not 1%. Use ``size_pct_of_ss()`` below, or
#                   read ``shock_pct_of_ss`` off the result, before quoting a
#                   productivity shock as "an x% TFP shock".
SHOCK_UNITS = {
    'monetary': {
        'input': 'rstar',
        'units': 'level change in the intercept of the quarterly nominal '
                 'Taylor rule, i = rstar + phi*pi (-0.0025 = a 25bp-per-'
                 'quarter cut in the intercept, about 100bp annualised; '
                 '-0.000625 = 25bp annualised). The policy rate itself moves '
                 'less, because the rule responds to the inflation the shock '
                 'causes: 13bp on impact for a 25bp cut with persistence '
                 '0.61. With persistence above about 0.75 the policy rate '
                 'moves the other way on impact: at 0.9 a 25bp-per-quarter '
                 'cut in the intercept RAISES the policy rate by 0.25pp a '
                 'quarter, while the ex-ante real rate falls and output '
                 'rises. Read the returned i before calling a shock a rate '
                 'cut of a given size',
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

#: The model period is a quarter, so rates are quarterly rates.
PERIODS_PER_YEAR = 4

_CACHE: dict = {}


def annualised_bp(size: float) -> float:
    """A quarterly rate change, in annualised basis points.

    ``size`` for a monetary shock, and the ``pi`` and ``r`` responses divided
    by 100, are level changes in quarterly rates. The annualised figure is the
    simple one, four times the quarterly change: ``annualised_bp(-0.0025)`` is
    -100, and ``annualised_bp(-0.000625)`` is -25.
    """
    return 1e4 * PERIODS_PER_YEAR * size


def monetary_size(annualised_basis_points: float) -> float:
    """The monetary ``size`` for a move quoted in annualised basis points.

    ``monetary_size(-25)`` is -0.000625, the intercept cut that a central bank
    would announce as 25bp. ``size=-0.0025`` is four times that.
    """
    return annualised_basis_points / (1e4 * PERIODS_PER_YEAR)


def size_pct_of_ss(kind: str, size: float, ss=None) -> float:
    """``size`` as a percentage of the driving variable's steady-state level.

    ``size_pct_of_ss('productivity', 0.01)`` is 2.14, not 1. For a monetary
    shock the result is a share of the steady-state quarterly rate (0.0125),
    which is rarely the number wanted; use ``annualised_bp`` there.
    """
    if kind not in SHOCK_KINDS:
        raise ValueError(f"unknown shock kind {kind!r}; available: "
                         f"{sorted(SHOCK_KINDS)}")
    if ss is None:
        ss = solve_steady_state()
    return 100 * size / float(ss[SHOCK_KINDS[kind]])


def size_from_pct_of_ss(kind: str, pct: float, ss=None) -> float:
    """The ``size`` that is ``pct`` percent of the driving variable's steady
    state: ``size_from_pct_of_ss('productivity', 1)`` is 0.00468, a 1% TFP
    improvement."""
    return pct / size_pct_of_ss(kind, 1.0, ss=ss)


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
    (B = 1.04) targets. Takes a few seconds at full grids; the result is
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

    A monetary ``size`` is a shock to the rule's intercept, not to the policy
    rate, and the two can have opposite signs. ``i = rstar + phi*pi`` with
    phi = 1.5, so on impact ``i_0 = size + phi*pi_0``. A persistent cut raises
    inflation by enough that the policy rate RISES: with size=-0.0025 the
    impact response of ``i`` is -0.13pp a quarter at persistence 0.61, zero at
    about 0.752 and +0.25pp a quarter (about +1pp annualised) at 0.9, which is
    PolicyEngine Macro's default persistence. The shock is still an easing at
    0.9: the ex-ante real rate ``i_t - pi_{t+1}`` falls 0.07pp and output
    rises 0.36% on impact. Read ``i`` off the result instead of inferring the
    policy-rate move from ``size`` (tests/test_units.py, docs/SHOCKS.md).

    Responses are first-order: the IRF is a matrix-vector product against a
    fixed Jacobian, so it scales EXACTLY with ``size`` and carries no
    state-dependence or asymmetry. Doubling ``size`` doubles every response;
    that is a property of the method, not a finding about the economy.

    Returns a dict with:
      'Y', 'C', 'I'   % deviations from steady state
      'pi', 'r'       level deviations of QUARTERLY rates, percentage points
                      (multiply by 4 for the annualised figure)
      'tax'           labour tax rate, level deviation in percentage points.
                      Included because it is the financing: this DAG has a
                      fixed bond supply Bg, so a G shock is paid for by this
                      tax path, period by period, with no deficit. Reading a
                      spending IRF without it invites a deficit-financed
                      interpretation the model cannot produce.
      'w', 'N'        real wage and hours, % deviations. Their sum is the
                      first-order % deviation of the PRE-tax wage bill.
      'i'             nominal policy rate, quarterly, percentage points
      'labour_income_post_tax'
                      % deviation of (1 - tax)*w*N, the wage bill after the
                      labour tax: w + N - tax/(1 - tax_ss), to first order.
                      It can have the opposite sign to the pre-tax wage bill:
                      a spending rise lifts w*N and lowers this, because the
                      tax that pays for it rises more. Use this, not w + N,
                      for what a shock does to workers' incomes.
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
    for v in ('w', 'N'):
        out[v] = 100 * (G_jac[v][inp] @ dx) / ss[v]
    out['i'] = 100 * (G_jac['i'][inp] @ dx)
    # First-order % deviation of (1 - tax)*w*N: the wage bill households keep.
    out['labour_income_post_tax'] = (
        out['w'] + out['N'] - out['tax'] / (1 - float(ss['tax'])))
    return out
