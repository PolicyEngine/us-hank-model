"""Opt-in variant: discount-factor (beta) heterogeneity. NOT the default.

The published model (``us_hank.model``) is the ABRS (2021) replication and is
left exactly as it is. This module builds a second, separate DAG in which
households draw one of several permanent discount-factor types::

    beta_z = beta + beta_spread * beta_offsets      (type masses beta_masses)

The shipped design has two types: a patient type with discount factor
``beta`` (two thirds of households) and an impatient type with
``beta - beta_spread`` (one third). This is in the spirit of Carroll,
Slacalek, Tokuoka & White (2017, Quantitative Economics) and the beta-heterogeneity robustness in Kaplan & Violante (2022,
Annual Review of Economics). Its purpose is the one scoped in issue #8: the
default calibration's quarterly MPC out of a $500-1,500 rebate is 0.08-0.10,
below the 0.15-0.25 range in the rebate evidence, and an impatient type that
holds little wealth of either kind is the standard way to raise it.

Why beta heterogeneity rather than a fixed or linear illiquid adjustment
cost: the household solver (``sequence_jacobian.hetblocks.hh_twoasset``) is
an endogenous-grid method that solves ``W_a / W_b = 1 + Psi_1`` by
interpolation, which needs Psi to be smooth with a monotone derivative. A
fixed cost makes the problem non-convex and a linear (Kaplan-Moll-Violante)
term puts a kink in Psi_1, so either would need a new household solver.
Beta heterogeneity changes one line of the household problem and adds an
exogenous state; every other block, the Jacobian method and the IRF code are
reused unchanged.

Construction
------------
* ``hh_beta`` is ``hh_twoasset.hh`` verbatim except that the discount factor
  is the per-state vector ``beta_z`` (one entry per (beta type, income)
  state) instead of the scalar ``beta``.
* The exogenous state is (beta type) x (income), with
  ``Pi = kron(Pi_beta, Pi_e)``. ``Pi_beta`` lets a household redraw its
  type, from the type masses, with probability ``beta_switch`` per quarter
  (default 1/200, an expected 50 years), so the masses are the stationary
  distribution. The redraw only exists to make the
  chain ergodic: with permanent types ``Pi`` is block-diagonal, has no unique
  stationary distribution, and the toolkit would silently pick the uniform
  seed. It moves the results very little at this frequency.
* The scalar ``beta`` stays a top-level parameter: the discount factor of
  the type with offset 0 (the patient type). It is the steady-state unknown
  (with chi1) exactly as in the default, and it is the discount factor the
  union's wage Phillips curve uses, which keeps that curve comparable with
  the default (the patient type holds nearly all the wealth).
* Why not three symmetric types around a centre (cstwMPC style): the patient
  type's beta is capped near 1/(1 + r) = 0.988 while the centre is already
  0.976, so the spread runs out at about 0.012, and on demo grids that moves
  the MPC by less than 0.01. Raising the MPC to the target needs an
  impatient type far below the centre, which an asymmetric design gives
  without pushing the patient type to the cap.
* ``beta_spread`` is set OUTSIDE the general-equilibrium solve: for a given
  spread the steady state solves (beta, chi1) for the same targets as the
  default (asset market clears at total wealth 14; liquid wealth B = 1.04).
  ``calibrate_beta_spread`` wraps that in a 1-D root find on the spread so
  that the refined-grid quarterly MPC out of a $1,000-per-household rebate
  hits ``MPC_TARGET``. The resulting spread is stored in
  ``BETA_HET_CALIBRATION`` so ordinary use is a single steady-state solve.

With one type (``beta_offsets=(0.0,)``, ``beta_masses=(1.0,)``) the variant
reduces to the default model and reproduces its household block exactly; a
test checks that.
"""

from __future__ import annotations

import numpy as np
from sequence_jacobian import combine, create_model, grids
from sequence_jacobian.blocks.het_block import het
from sequence_jacobian import interpolate
from sequence_jacobian.examples import two_asset as ta
from sequence_jacobian.hetblocks.hh_twoasset import (
    addouter, adjustment_costs, get_Psi_and_deriv, hh_init,
    lhs_equals_rhs_interpolate, marginal_cost_grid)

from us_hank import model as _model
from us_hank import distributional as _dist

#: Target for the quarterly MPC out of a $1,000-per-household rebate, measured
#: on the refined liquid grid (``refined_aggregate_mpc_beta_het``). 0.20 is the
#: middle of the 0.15-0.25 range of the US rebate evidence. The top end is the
#: empirical estimate Kaplan & Violante (2014, Econometrica 82(4), 1199-1239)
#: compare their model with: about 25% of a stimulus payment spent on
#: nondurables in the quarter of receipt (the 2001 rebates; Johnson, Parker &
#: Souleles 2006). KV do NOT target it: they calibrate beta to median illiquid
#: wealth and the transaction cost to a hand-to-mouth share of about 1/3, and
#: their model's untargeted output is a rebate coefficient of 15% and an MPC
#: out of a surprise $500 of about 20%, falling with the payment size (to 3%
#: at $5,000). So 0.20 at $1,000 is at the high end of what their model gives.
#: The bottom end is the smaller response to the 2008 payments (Kaplan &
#: Violante 2014, AER Papers & Proceedings 104(5); Parker et al. 2013). The
#: default model gives 0.091 on the same measure (issue #8).
MPC_TARGET = 0.20
MPC_TARGET_DOLLARS_PER_HOUSEHOLD = 1000.0

# US GDP about $31T a year (2026), 133M households: the scale model units
# (steady-state quarterly Y = 1) are mapped to dollars with.
Q_GDP_PER_HOUSEHOLD = 31e12 / 133e6 / 4

#: Variant calibration. Everything from ``PRODUCTION_CALIBRATION`` (grids,
#: targets, every structural parameter) plus the beta-type block. ``beta_spread``
#: is the value ``calibrate_beta_spread`` returns at ``MPC_TARGET``.
BETA_HET_CALIBRATION = {
    **_model.PRODUCTION_CALIBRATION,
    'beta_offsets': (-1.0, 0.0),
    'beta_masses': (1 / 3, 2 / 3),
    # calibrate_beta_spread() at MPC_TARGET, production grids (CI run
    # 42145287339): impatient beta = 0.98283 - 0.13041 = 0.85242.
    'beta_spread': 0.1304144180026553,
    'beta_switch': 1 / 200,
}

#: Median-wealth targets (``liquid_target='median'``). Wealth ratios are to
#: MEAN quarterly pre-tax labour income (``w * N`` in the model), the
#: normalisation of Kaplan & Violante (2014, Econometrica 82(4), Table III and
#: fn. 28), from the 2001 Survey of Consumer Finances, households aged 22-59,
#: top 5% by net worth excluded: mean earnings plus benefits $52,745, median
#: net liquid wealth $2,629, median net illiquid wealth $54,600.
SCF_2001_KV = {'mean_earnings': 52745.0, 'median_liquid': 2629.0,
               'median_illiquid': 54600.0}
MEDIAN_TARGETS = {
    # 4 x 2,629 / 52,745 = 0.199 quarters of mean labour income
    'median_liquid_to_income': 4 * SCF_2001_KV['median_liquid'] / SCF_2001_KV['mean_earnings'],
    # 4 x 54,600 / 52,745 = 4.14 quarters of mean labour income
    'median_illiquid_to_income': 4 * SCF_2001_KV['median_illiquid'] / SCF_2001_KV['mean_earnings'],
    # Poor hand-to-mouth share: Kaplan, Violante & Weidner (2014, Brookings
    # Papers on Economic Activity, Table 3 baseline, SCF pooled 1989-2010,
    # biweekly pay period): 0.121 poor, 0.192 wealthy, 0.312 in total. A
    # different sample from the 2001 medians above.
    'poor_htm': 0.12,
    # Quarterly MPC out of $500 per household, target band 0.15-0.25: the
    # 2001 rebate coefficient on nondurables is 0.375 (JPS 2006), 0.22-0.24
    # with the CEX outliers trimmed and 0.244 by IV quantile regression
    # (Kaplan & Violante 2014, Table I); for 2008, households spent 12-30% of
    # the payments on nondurables in the quarter of receipt (Parker,
    # Souleles, Johnson & McClelland 2013, AER 103(6)). KV's own model gives
    # about 20% out of a surprise $500. These are nondurables; the model has
    # one consumption good.
    'mpc_500': 0.20,
}

#: Hand-to-mouth cutoff, model units, for both assets: liquid wealth below it
#: is hand-to-mouth, and hand-to-mouth with illiquid wealth below it too is
#: POOR hand-to-mouth (the Kaplan-Violante-Weidner split). 0.05 is about
#: $1,000 at the 2001 SCF scale (0.05 / 0.66 x 52,745 / 4). KVW's data rule is
#: liquid wealth at most half of earnings per pay period; at median earnings
#: ($41,000) that is about $790 for their biweekly baseline and $1,700 for a
#: monthly pay period, so $1,000 sits between the two. It equals
#: ``distributional.DEFAULT_HTM_THRESHOLD``. The mass exactly at the liquid
#: kink (b = 0) is reported too, but it moves with the grid (lottery
#: weights), so it is not the target.
HTM_CUTOFF = 0.05

#: Second calibration of the variant (issue #8, round 4): median wealth
#: targets, ``liquid_target='median'``. Three equal-mass types on a uniform
#: beta distribution of half-width ``beta_spread`` around the centre ``beta``
#: (Carroll, Slacalek, Tokuoka & White 2017), ``chi1`` set outside the solve.
#: (beta_spread, chi1) were chosen on a grid (CI runs 42149673949,
#: 42150101068, 42150572003, 42150783547) to fit the two medians with every
#: type inside or near the Carroll et al. liquid-wealth range
#: (0.936-0.978/quarter). Fits, refined grid: median liquid 0.205 (target
#: 0.199), median illiquid 4.69 (4.14), poor HtM 0.060 (0.12), MPC out of
#: $500 0.126 (0.15-0.25). The last two MISS, and cannot be reached: a wider
#: spread (0.034) or a fourth, more impatient type at 12-15% mass leaves no
#: centre beta that clears the asset market at A + B = 14 with the most
#: patient type below 1 / (1 + r). The most patient type sits at that cap
#: here (0.9875 vs 0.98765). chi0 in 0.02-0.25 moves poor HtM by under 1pp.
MEDIAN_CALIBRATION = {
    **_model.PRODUCTION_CALIBRATION,
    'beta_offsets': (-2 / 3, 0.0, 2 / 3),
    'beta_masses': (1 / 3, 1 / 3, 1 / 3),
    'beta_spread': 0.033,
    'beta_switch': 1 / 200,
    'chi1': 6.3,
}

#: Starting guess for the (beta, chi1) solve: the default's.
_SS_GUESS = {'beta': 0.976, 'chi1': 6.5}
#: The calibrated variant's solution, used as the guess at the calibrated
#: spread: Broyden from the default's guess does not converge that far out.
_CALIBRATED_GUESS = {'beta': 0.982831393610993, 'chi1': 8.065188075253275}

_CACHE: dict = {}


def make_grids_beta(bmax, amax, kmax, nB, nA, nK, nZ, rho_z, sigma_z,
                    beta_offsets, beta_masses, beta, beta_spread, beta_switch):
    """``two_asset.make_grids`` with the exogenous state extended by beta type.

    Returns the usual grids with ``e_grid`` tiled over types, the joint
    transition ``Pi = kron(Pi_beta, Pi_e)``, and ``beta_z``, the discount
    factor of each joint state. Type-major ordering: state ``j * nZ + i`` is
    type ``j``, income state ``i``.
    """
    b_grid = grids.agrid(amax=bmax, n=nB)
    a_grid = grids.agrid(amax=amax, n=nA)
    k_grid = grids.agrid(amax=kmax, n=nK)[::-1].copy()
    e, _, Pi_e = grids.markov_rouwenhorst(rho=rho_z, sigma=sigma_z, N=nZ)
    offsets = np.asarray(beta_offsets, dtype=float)
    masses = np.asarray(beta_masses, dtype=float)
    n_beta = len(offsets)
    betas = beta + beta_spread * offsets
    # Redraw the type with probability beta_switch per quarter, from the type
    # masses, so the masses are the chain's unique stationary distribution.
    Pi_beta = ((1 - beta_switch) * np.eye(n_beta)
               + beta_switch * np.outer(np.ones(n_beta), masses))
    e_grid = np.tile(e, n_beta)
    Pi = np.kron(Pi_beta, Pi_e)
    beta_z = np.repeat(betas, nZ)
    return b_grid, a_grid, k_grid, e_grid, Pi, beta_z


def _make_grids_beta_refined(bmax, amax, kmax, nB, nA, nK, nZ, rho_z, sigma_z,
                             beta_offsets, beta_masses, beta, beta_spread,
                             beta_switch, b_pivot):
    """``make_grids_beta`` on the refined liquid grid of
    ``distributional._make_grids_refined`` (same pivot construction)."""
    _, a_grid, k_grid, e_grid, Pi, beta_z = make_grids_beta(
        bmax, amax, kmax, nB, nA, nK, nZ, rho_z, sigma_z, beta_offsets,
        beta_masses, beta, beta_spread, beta_switch)
    b_grid = np.geomspace(b_pivot, bmax + b_pivot, nB) - b_pivot
    b_grid[0] = 0.0
    # (the toolkit reads output names off this return statement)
    return b_grid, a_grid, k_grid, e_grid, Pi, beta_z


@het(exogenous='Pi', policy=['b', 'a'], backward=['Vb', 'Va'],
     hetinputs=[marginal_cost_grid], hetoutputs=[adjustment_costs],
     backward_init=hh_init)
def hh(Va_p, Vb_p, a_grid, b_grid, z_grid, e_grid, k_grid, beta_z, eis,
            rb, ra, chi0, chi1, chi2, Psi1):
    """``hh_twoasset.hh`` with a per-state discount factor ``beta_z``.

    The body is the toolkit's verbatim; the only change is the two lines that
    discount the expected continuation value. The function keeps the name
    ``hh`` so the block's internals sit under ``ss.internals['hh']``, where
    every function in ``us_hank.distributional`` looks for them.
    """
    Wb = beta_z[:, np.newaxis, np.newaxis] * Vb_p
    Wa = beta_z[:, np.newaxis, np.newaxis] * Va_p
    W_ratio = Wa / Wb

    i, pi = lhs_equals_rhs_interpolate(W_ratio, 1 + Psi1)
    a_endo_unc = interpolate.apply_coord(i, pi, a_grid)
    c_endo_unc = interpolate.apply_coord(i, pi, Wb) ** (-eis)

    b_endo = (c_endo_unc + a_endo_unc + addouter(-z_grid, b_grid, -(1 + ra) * a_grid)
              + get_Psi_and_deriv(a_endo_unc, a_grid, ra, chi0, chi1, chi2)[0]) / (1 + rb)
    i, pi = interpolate.interpolate_coord(b_endo.swapaxes(1, 2), b_grid)
    a_unc = interpolate.apply_coord(i, pi, a_endo_unc.swapaxes(1, 2)).swapaxes(1, 2)
    b_unc = interpolate.apply_coord(i, pi, b_grid).swapaxes(1, 2)

    lhs_con = W_ratio[:, 0:1, :] / (1 + k_grid[np.newaxis, :, np.newaxis])
    i, pi = lhs_equals_rhs_interpolate(lhs_con, 1 + Psi1)
    a_endo_con = interpolate.apply_coord(i, pi, a_grid)
    c_endo_con = ((1 + k_grid[np.newaxis, :, np.newaxis]) ** (-eis)
                  * interpolate.apply_coord(i, pi, Wb[:, 0:1, :]) ** (-eis))

    b_endo = (c_endo_con + a_endo_con
              + addouter(-z_grid, np.full(len(k_grid), b_grid[0]), -(1 + ra) * a_grid)
              + get_Psi_and_deriv(a_endo_con, a_grid, ra, chi0, chi1, chi2)[0]) / (1 + rb)
    a_con = interpolate.interpolate_y(b_endo.swapaxes(1, 2), b_grid,
                                      a_endo_con.swapaxes(1, 2)).swapaxes(1, 2)

    a, b = a_unc.copy(), b_unc.copy()
    b[b <= b_grid[0]] = b_grid[0]
    a[b <= b_grid[0]] = a_con[b <= b_grid[0]]

    Psi, _, Psi2 = get_Psi_and_deriv(a, a_grid, ra, chi0, chi1, chi2)
    c = addouter(z_grid, (1 + rb) * b_grid, (1 + ra) * a_grid) - Psi - a - b
    uc = c ** (-1 / eis)
    uce = e_grid[:, np.newaxis, np.newaxis] * uc

    Va = (1 + ra - Psi2) * uc
    Vb = (1 + rb) * uc
    return Va, Vb, a, b, c, uce


hh_beta = hh


def household_block():
    return hh_beta.add_hetinputs([ta.income, make_grids_beta])


def build_models():
    """(ss_model, model) for the variant: the default DAG with ``hh_beta``."""
    if 'models' in _CACHE:
        return _CACHE['models']
    household = household_block()
    production = combine([ta.labor, ta.investment])
    production_solved = production.solved(
        unknowns={'Q': 1., 'K': 10.}, targets=['inv', 'val'],
        solver='broyden_custom')
    blocks = [household, ta.pricing_solved, ta.arbitrage_solved,
              production_solved, ta.dividend, ta.taylor, ta.fiscal,
              ta.share_value, ta.finance, ta.wage, ta.union, ta.mkt_clearing]
    m = create_model(blocks, name='Two-Asset HANK, beta heterogeneity')
    blocks_ss = [household, ta.partial_ss, ta.dividend, ta.taylor, ta.fiscal,
                 ta.share_value, ta.finance, ta.union_ss, ta.mkt_clearing]
    ss_m = create_model(blocks_ss, name='Two-Asset HANK SS, beta heterogeneity')
    _CACHE['models'] = (ss_m, m)
    return ss_m, m


def solve_steady_state(beta_spread: float | None = None,
                       guess: dict | None = None,
                       use_cache: bool = True,
                       liquid_target: str = 'aggregate', **overrides):
    """General-equilibrium steady state of the variant.

    ``liquid_target='aggregate'`` (the first version of this variant, and the
    default model's targets): solves (beta, chi1) so the asset market clears
    at total wealth 14 and liquid wealth B = Bh = 1.04.

    ``liquid_target='median'`` (``MEDIAN_CALIBRATION``): drops the aggregate
    liquid-wealth target, as Kaplan & Violante (2014, fn. 28) do. ``chi1`` is
    then a parameter (set outside the solve, with the beta spread, to the
    median-wealth targets), ``beta`` alone clears the asset market, and
    ``Bh`` is reset to the household liquid wealth the solve produces, so the
    illiquid fund's equity share ``pshare = p / (tot_wealth - Bh)`` stays
    consistent with what households hold. In the steady state ``Bh`` enters
    nothing else (``ra = r`` whatever ``pshare`` is).

    At the given ``beta_spread`` (default: the calibrated one). ``guess``
    seeds the solver (continuation); ``overrides`` replace calibration
    entries (e.g. ``beta_offsets``, ``beta_masses``, ``chi1``, grid sizes).
    """
    if liquid_target not in ('aggregate', 'median'):
        raise ValueError(f"liquid_target must be 'aggregate' or 'median', "
                         f"not {liquid_target!r}")
    base = MEDIAN_CALIBRATION if liquid_target == 'median' else BETA_HET_CALIBRATION
    cal = dict(base)
    cal.update(overrides)
    if beta_spread is not None:
        cal['beta_spread'] = float(beta_spread)
    key = ('ss', liquid_target, tuple(sorted((k, v) for k, v in cal.items())))
    if use_cache and key in _CACHE:
        return _CACHE[key]
    ss_m, m = build_models()
    if liquid_target == 'aggregate':
        if guess is None:
            guess = (_CALIBRATED_GUESS if cal['beta_spread'] > 0.05
                     else _SS_GUESS)
        cali = ss_m.solve_steady_state(
            cal, unknowns=dict(guess),
            targets={'asset_mkt': 0., 'B': 'Bh'}, solver='broyden_custom')
    else:
        cal.pop('beta', None)
        lo, hi = beta_bracket(cal) if guess is None else guess
        cali = ss_m.solve_steady_state(
            cal, unknowns={'beta': (lo, hi)}, targets={'asset_mkt': 0.},
            solver='brentq', ttol=1e-12)
        cali = dict(cali.toplevel)
        cali['Bh'] = cali['B']
    ss = m.steady_state(cali)
    if use_cache:
        _CACHE[key] = ss
    return ss


def beta_cap(cal) -> float:
    """1 / (1 + r): no type's discount factor may reach it (the illiquid
    return is r in the steady state, so a type at the cap never stops
    accumulating)."""
    return 1.0 / (1.0 + cal['r'])


def beta_bracket(cal) -> tuple:
    """Bracket for the centre ``beta`` in the median-mode solve: the most
    patient type stays strictly below ``beta_cap``."""
    top = cal['beta_spread'] * max(cal['beta_offsets'])
    return 0.90, beta_cap(cal) - top - 1e-4


def uniform_offsets(n: int) -> tuple:
    """Offsets of an ``n``-point equal-mass discretisation of a uniform
    distribution on [beta - spread, beta + spread] (Carroll, Slacalek,
    Tokuoka & White 2017): the midpoints of ``n`` equal-probability bins."""
    return tuple((2 * j + 1 - n) / n for j in range(n))


def solve_jacobian(ss, T: int = 300):
    """General-equilibrium Jacobian of the variant around ``ss``.

    Pass it with ``ss`` to ``us_hank.model.shock(kind, size, rho, ss=ss,
    G_jac=...)``; the IRF code is shared with the default model.
    """
    key = ('jac', id(ss), T)
    if key in _CACHE:
        return _CACHE[key][1]
    _, m = build_models()
    G = m.solve_jacobian(ss, _model.UNKNOWNS, _model.TARGETS, _model.EXOGENOUS, T=T)
    _CACHE[key] = (ss, G)
    return G


def shock(kind: str, size: float, persistence: float, T: int = 300, ss=None):
    """``us_hank.model.shock`` on the variant's steady state and Jacobian."""
    if ss is None:
        ss = solve_steady_state()
    return _model.shock(kind, size, persistence, T=T, ss=ss,
                        G_jac=solve_jacobian(ss, T=T))


# --- MPCs ---------------------------------------------------------------------

def refined_household_ss(ss, nB: int | None = None, b_pivot: float | None = None):
    """The variant's household block alone on the refined liquid grid, at
    ``ss`` prices: the analogue of ``distributional.refined_household_ss``."""
    nB = _dist.REFINED_LIQUID_GRID['nB'] if nB is None else int(nB)
    b_pivot = (_dist.REFINED_LIQUID_GRID['b_pivot'] if b_pivot is None
               else float(b_pivot))
    key = ('refined', id(ss), nB, b_pivot)
    if key in _CACHE:
        return _CACHE[key][1]
    block = hh_beta.add_hetinputs([ta.income, _make_grids_beta_refined])
    inputs = dict(ss.toplevel)
    inputs.update(nB=nB, b_pivot=b_pivot)
    out = block.steady_state(inputs)
    _CACHE[key] = (ss, out)
    return out


def rebate_frac(ss, dollars_per_household: float) -> float:
    """A rebate in dollars per household as a fraction of mean quarterly
    post-tax labour income, the unit ``distributional`` MPCs take."""
    return (dollars_per_household / Q_GDP_PER_HOUSEHOLD
            / _dist.mean_quarterly_labour_income(ss))


def refined_aggregate_mpc_beta_het(ss, windfall_frac: float, **kw) -> float:
    return _dist.aggregate_mpc(refined_household_ss(ss, **kw), windfall_frac)


def mpc_by_type(ss, windfall_frac: float) -> dict:
    """Mass and mean MPC of each beta type (on whatever grid ``ss`` carries)."""
    mpc, D, _ = _dist.mpc_liquid(ss, windfall_frac)
    n = len(ss.toplevel['beta_offsets'])
    nZ = int(ss.toplevel['nZ'])
    out = {}
    for j in range(n):
        sl = slice(j * nZ, (j + 1) * nZ)
        m = float(D[sl].sum())
        out[f'type{j}'] = {'mass': m, 'mpc': float(np.vdot(D[sl], mpc[sl]) / m),
                           'beta': float(ss.internals['hh']['beta_z'][j * nZ])}
    return out


def calibrate_beta_spread(target: float = MPC_TARGET,
                          dollars: float = MPC_TARGET_DOLLARS_PER_HOUSEHOLD,
                          sweep=(0.0, 0.02, 0.04, 0.06, 0.08, 0.1, 0.12, 0.15,
                                 0.18, 0.21, 0.24, 0.27, 0.3),
                          tol: float = 2e-4, log=print, **overrides):
    """Find ``beta_spread`` so the refined-grid MPC out of ``dollars`` per
    household equals ``target``.

    Walks up ``sweep`` (continuation: each (beta, chi1) solve starts from the
    previous one) until the MPC crosses the target, then refines inside that
    bracket by Illinois regula falsi. Returns (spread, ss, path), where
    ``path`` lists every evaluation. Raises if the sweep never crosses.
    """
    path = []
    guess = dict(_SS_GUESS)

    def f(x, g):
        ss = solve_steady_state(x, guess=g, use_cache=False, **overrides)
        mpc = refined_aggregate_mpc_beta_het(ss, rebate_frac(ss, dollars))
        path.append({'beta_spread': float(x), 'beta': float(ss['beta']),
                     'chi1': float(ss['chi1']), 'mpc': mpc})
        log(path[-1])
        return mpc - target, ss, {'beta': float(ss['beta']),
                                  'chi1': float(ss['chi1'])}

    lo = flo = glo = None
    for x in sweep:
        fx, ss_x, g = f(x, guess)
        if abs(fx) < tol:
            return x, ss_x, path
        if fx > 0 and lo is not None:
            hi, fhi, ghi = x, fx, g
            break
        lo, flo, glo, guess = x, fx, g, g
    else:
        raise ValueError(f"MPC target {target} not reached on the sweep: {path}")
    # Illinois regula falsi inside [lo, hi].
    for _ in range(25):
        x = hi - fhi * (hi - lo) / (fhi - flo)
        g = glo if abs(x - lo) < abs(x - hi) else ghi
        fx, ss_x, gx = f(x, g)
        if abs(fx) < tol:
            break
        if fx * fhi < 0:
            lo, flo, glo = hi, fhi, ghi
        else:
            flo /= 2
        hi, fhi, ghi = x, fx, gx
    return x, ss_x, path


# --- distribution moments (median-wealth targets) -----------------------------

def _weighted_median(grid, mass):
    """Median of a distribution on ``grid`` (lottery weights ``mass``),
    linear inside the cell where the cumulative mass crosses one half."""
    cum = np.cumsum(mass) / mass.sum()
    j = int(np.searchsorted(cum, 0.5))
    if j == 0:
        return float(grid[0])
    lo = cum[j - 1]
    return float(grid[j - 1] + (grid[j] - grid[j - 1]) * (0.5 - lo) / (cum[j] - lo))


def wealth_moments(ss) -> dict:
    """Median and mean liquid and illiquid wealth over mean quarterly pre-tax
    labour income (``w * N``), and the hand-to-mouth shares at the liquid
    kink: the moments ``MEDIAN_TARGETS`` refers to. Works on ``ss`` or on a
    ``refined_household_ss`` output (whose ``toplevel`` carries the same
    prices)."""
    h = ss.internals['hh']
    D, b_grid, a_grid = h['D'], h['b_grid'], h['a_grid']
    inc = float(ss['w'] * ss['N'])
    liq = D[:, b_grid < HTM_CUTOFF, :]
    htm = float(liq.sum())
    poor = float(liq[:, :, a_grid < HTM_CUTOFF].sum())
    return {
        'income': inc,
        'median_liquid_to_income': _weighted_median(b_grid, D.sum(axis=(0, 2))) / inc,
        'median_illiquid_to_income': _weighted_median(a_grid, D.sum(axis=(0, 1))) / inc,
        'mean_liquid_to_income': float(np.vdot(D.sum(axis=(0, 2)), b_grid)) / inc,
        'mean_illiquid_to_income': float(np.vdot(D.sum(axis=(0, 1)), a_grid)) / inc,
        'htm': htm,
        'poor_htm': poor,
        'wealthy_htm': htm - poor,
        'htm_kink': float(D[:, 0, :].sum()),
    }
