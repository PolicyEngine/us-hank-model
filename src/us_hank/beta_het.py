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
#: middle of the 0.15-0.25 range for nondurable spending in the quarter of
#: receipt that Kaplan & Violante (2014, Econometrica 82(4)) take from the
#: 2001 and 2008 US rebate studies (Johnson, Parker & Souleles 2006, AER;
#: Parker, Souleles, Johnson & McClelland 2013, AER) as their calibration
#: target. The default model gives 0.091 on the same measure (issue #8).
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
    'beta_spread': None,  # set below once calibrated; see calibrate_beta_spread
    'beta_switch': 1 / 200,
}

#: Starting guess for the (beta, chi1) solve. The default's solution.
_SS_GUESS = {'beta': 0.976, 'chi1': 6.5}

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


def solve_steady_state(beta_spread: float | None = None, guess: dict | None = None,
                       use_cache: bool = True, **overrides):
    """General-equilibrium steady state of the variant.

    Solves (beta, chi1) for the default's targets (asset market clears; B =
    Bh = 1.04) at the given ``beta_spread`` (default: the calibrated one).
    ``guess`` seeds the solver (continuation); ``overrides`` replace
    calibration entries (e.g. ``beta_offsets``, ``beta_masses``, grid sizes).
    """
    cal = dict(BETA_HET_CALIBRATION)
    cal.update(overrides)
    if beta_spread is not None:
        cal['beta_spread'] = float(beta_spread)
    if cal['beta_spread'] is None:
        raise ValueError("beta_spread is not calibrated yet; pass one")
    key = ('ss', tuple(sorted((k, v) for k, v in cal.items())))
    if use_cache and key in _CACHE:
        return _CACHE[key]
    ss_m, m = build_models()
    cali = ss_m.solve_steady_state(
        cal, unknowns=dict(guess or _SS_GUESS),
        targets={'asset_mkt': 0., 'B': 'Bh'}, solver='broyden_custom')
    ss = m.steady_state(cali)
    if use_cache:
        _CACHE[key] = ss
    return ss


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
