"""Distributional outputs from the solved two-asset HANK steady state.

All functions work off ``ss.internals['hh']``: the joint distribution D and
the consumption policy c on the (z, b, a) grid, where ``b`` and ``a`` index
*beginning-of-period* liquid and illiquid assets.

Two measurement caveats govern everything in this module, and both are
reproducible with the checks in ``tests/test_distributional.py``:

1. **MPCs are windfall-size dependent and only partly grid-converged.** The
   consumption policy has a kink at the liquid borrowing constraint, where
   just over half the population sits. The MPC out of a *small* windfall is
   the slope right at that kink, and nB=50 does not resolve it: the aggregate
   MPC out of a one-grid-step windfall rises monotonically from 0.076 at
   nB=50 to 0.100 at nB=150 with no plateau. This module therefore defines
   MPCs out of an explicit windfall of stated economic size (see
   ``DEFAULT_WINDFALL_FRAC``), which is far more stable across grids, and
   never reports an unlabelled "the MPC".

2. **The hand-to-mouth share is an atom, not a threshold.** 51.2% of
   households hold exactly zero liquid assets, so the reported share is
   driven by that atom rather than by the cutoff; see
   ``hand_to_mouth_breakdown``.
"""

from __future__ import annotations

import numpy as np

# Liquid-asset cutoff for the hand-to-mouth count, in model units where
# steady-state QUARTERLY output Y = 1. So 0.05 is 5% of quarterly GDP, 1.25%
# of annual GDP, and about 11.8% of mean quarterly post-tax labour income
# (which is (1 - tax) * w * N = 0.425 at the paper calibration) -- roughly a
# fortnight of earnings, the usual hand-to-mouth cutoff. The count is
# insensitive to this number over [0, 0.028] because it is dominated by the
# atom of households exactly at the constraint.
DEFAULT_HTM_THRESHOLD = 0.05

# Windfall used to define MPCs, as a fraction of mean quarterly post-tax
# labour income. 0.25 is 0.106 in model units (10.6% of quarterly GDP per
# model household). Mapped to US GDP (about $31T a year, 2026) that is about
# $2,400 per head (342M people) or about $6,200 per household (133M
# households) -- LARGER than the 2001 ($300-600) and 2008 (about $900 on
# average) rebates the empirical MPC literature measures. With a concave
# policy that biases the MPC down, but only slightly: at $1,000-1,500 per
# head it is 0.069-0.064 (vs 0.058), and per household such a rebate is
# below the first liquid grid step, where the arc MPC is the grid-step slope
# (0.075 at nB=50, not grid-converged). 0.25 is kept because it is above that
# grid step and so measures economics rather than grid spacing; see
# test_rebate_scale_mpc_is_still_far_below_the_empirical_range.
DEFAULT_WINDFALL_FRAC = 0.25


def _hh(ss):
    return ss.internals['hh']


def mean_quarterly_labour_income(ss) -> float:
    """Mean post-tax quarterly labour income, the scale windfalls are quoted in."""
    return float((1.0 - ss['tax']) * ss['w'] * ss['N'])


def mpc_liquid(ss, windfall_frac: float | None = DEFAULT_WINDFALL_FRAC):
    """Quarterly MPC out of a liquid-asset windfall. Returns (mpc, D, b_grid).

    ``windfall_frac`` is the windfall size as a fraction of mean quarterly
    post-tax labour income. The MPC is the arc slope of the consumption
    policy, ``[c(b + windfall) - c(b)] / windfall``, with ``c`` interpolated
    linearly along the liquid grid at fixed (z, a). Because the windfall has
    a fixed economic size, the answer does not move with the grid the way a
    one-grid-step difference does.

    ``windfall_frac=None`` selects the legacy one-grid-step difference (a
    centred difference in the interior). That is the derivative of the policy
    at grid resolution and is NOT grid-converged at nB=50 -- it rises from
    0.076 to 0.100 as nB goes 50 -> 150. It is kept for comparison and for
    the grid-sensitivity test; do not quote it as a structural MPC.

    MPCs are capped above at 1. They are not floored: a negative value would
    signal a policy-function or grid problem rather than a quantity to hide,
    and ``tests/test_distributional.py`` asserts none occur.
    """
    h = _hh(ss)
    D, c, b_grid = h['D'], h['c'], h['b_grid']
    if windfall_frac is None:
        mpc = np.empty_like(c)
        db = b_grid[2:] - b_grid[:-2]
        mpc[:, 1:-1, :] = (c[:, 2:, :] - c[:, :-2, :]) / db[None, :, None]
        mpc[:, 0, :] = (c[:, 1, :] - c[:, 0, :]) / (b_grid[1] - b_grid[0])
        mpc[:, -1, :] = (c[:, -1, :] - c[:, -2, :]) / (b_grid[-1] - b_grid[-2])
        return np.minimum(mpc, 1.0), D, b_grid

    if windfall_frac <= 0:
        raise ValueError("windfall_frac must be positive (or None for the "
                         "legacy one-grid-step difference)")
    windfall = windfall_frac * mean_quarterly_labour_income(ss)
    # Interpolate c(z, b + windfall, a) along b at fixed (z, a). The windfall
    # pushes the top of the grid past b_grid[-1], where np.interp would CLAMP
    # and so report a zero MPC for the richest households; extrapolate along
    # the last interval's slope instead. (Negligible mass lives there, but a
    # clamp would be a silent zero rather than an approximation.)
    shifted = _shift_along_b(c, b_grid, windfall)
    return np.minimum((shifted - c) / windfall, 1.0), D, b_grid


def aggregate_mpc(ss, windfall_frac: float | None = DEFAULT_WINDFALL_FRAC) -> float:
    """Population-weighted quarterly MPC out of a liquid windfall.

    See ``mpc_liquid`` for the windfall convention: this number is only
    meaningful alongside the windfall size it was computed at.
    """
    mpc, D, _ = mpc_liquid(ss, windfall_frac)
    return float(np.vdot(D, mpc))


def mpc_by_liquid_quartile(
    ss, windfall_frac: float | None = DEFAULT_WINDFALL_FRAC
) -> dict:
    """Mean quarterly MPC within each liquid-wealth quartile (Q1 = poorest).

    Over half the population holds exactly zero liquid assets, so the 25th and
    50th percentiles of liquid wealth are both *inside* that single atom: Q1
    and Q2 are both entirely constrained households and, by construction here,
    report the same MPC. Any apparent Q1-vs-Q2 gradient would be an artifact
    of how tied households happen to be ordered, not economics -- see
    ``_mean_by_quantile``.
    """
    mpc, D, b_grid = mpc_liquid(ss, windfall_frac)
    b = np.broadcast_to(b_grid[None, :, None], D.shape).ravel()
    return _mean_by_quantile(mpc.ravel(), D.ravel(), b, 4, 'Q')


def hand_to_mouth_share(ss, threshold: float = DEFAULT_HTM_THRESHOLD) -> float:
    """Population share with liquid assets b < ``threshold`` (incl. wealthy HtM)."""
    h = _hh(ss)
    D, b_grid = h['D'], h['b_grid']
    return float(D[:, b_grid < threshold, :].sum())


def hand_to_mouth_breakdown(ss, threshold: float = DEFAULT_HTM_THRESHOLD) -> dict:
    """Hand-to-mouth share split poor vs wealthy, the Kaplan-Violante-Weidner cut.

    'Poor' HtM hold little of *either* asset; 'wealthy' HtM are liquidity-poor
    but hold illiquid wealth. ``constrained_share`` is the mass exactly at the
    liquid borrowing constraint -- report that alongside the thresholded share,
    because it is what actually drives the number: the threshold only matters
    once it exceeds the first positive grid point.

    The illiquid cutoff is the same ``threshold`` in the same model units.
    """
    h = _hh(ss)
    D, b_grid, a_grid = h['D'], h['b_grid'], h['a_grid']
    liquid_poor = b_grid < threshold
    illiquid_poor = a_grid < threshold
    total = float(D[:, liquid_poor, :].sum())
    poor = float(D[:, liquid_poor, :][:, :, illiquid_poor].sum())
    return {
        'hand_to_mouth_share': total,
        'poor_hand_to_mouth_share': poor,
        'wealthy_hand_to_mouth_share': total - poor,
        'constrained_share': float(D[:, b_grid <= 0.0, :].sum()),
        'threshold': float(threshold),
    }


def consumption_response_by_wealth_quantile(ss, irf: dict, n: int = 4) -> dict:
    """Impact consumption response (% dev) allocated across TOTAL-wealth quantiles.

    This is an ACCOUNTING ALLOCATION of an already-computed aggregate, not a
    solved household-level response. It takes the aggregate impact consumption
    response ``irf['C'][0]`` and splits it across quantiles in proportion to
    each household's steady-state liquid-asset MPC, normalised so the
    population-weighted mean reproduces the aggregate exactly.

    What it therefore does and does not capture:

    * It captures the MPC-heterogeneity channel: high-MPC households are
      assigned a larger share of the aggregate response.
    * It does NOT solve for household-level dynamics, and it does NOT use any
      household-specific income exposure to the shock. Every household's
      income is implicitly assumed to move in proportion to the aggregate, so
      the allocation is the same shape for a monetary, a spending and a
      productivity shock -- only the aggregate it is scaled to differs. Shocks
      with genuinely different incidence across the wealth distribution (a
      spending shock financed by a labour tax falls on earners, not on
      rentiers) are NOT distinguished by this statistic.
    * Because the scaling is by relative MPC, the SIGN is common to all
      quantiles: this cannot show one group gaining while another loses.

    Use it to say "the consumption response is concentrated among low-wealth,
    high-MPC households", not to quote a group-specific number as a solved
    result. A full answer needs the household-level Jacobians, which this
    package does not compute.
    """
    mpc, D, b_grid = mpc_liquid(ss)
    h = _hh(ss)
    a_grid = h['a_grid']
    wealth = (np.broadcast_to(b_grid[None, :, None], D.shape)
              + np.broadcast_to(a_grid[None, None, :], D.shape)).ravel()
    Df, mf = D.ravel(), mpc.ravel()
    agg_mpc = float(np.vdot(Df, mf))
    dC_agg = float(irf['C'][0])
    # Scale group responses in proportion to relative MPC so that the
    # population-weighted mean is exactly dC_agg.
    rel = mf / agg_mpc if agg_mpc != 0 else np.ones_like(mf)
    return _mean_by_quantile(dC_agg * rel, Df, wealth, n, 'Q')


def _mean_by_quantile(x, weights, key, n, prefix):
    """Weighted mean of ``x`` within ``n`` equal-population bins of ``key``.

    Households sharing a ``key`` value form one indivisible block: its mass is
    split across bins in proportion to overlap, and every bin it touches
    receives that block's own weighted mean of ``x``.

    That tie handling is essential here rather than fussy. Over half the
    population sits exactly at the liquid borrowing constraint, so two of the
    three quartile edges fall inside a single atom. Assigning whole grid cells
    to bins instead -- the obvious implementation -- lets the arbitrary
    ordering of tied cells manufacture a gradient: re-sorting ties at random
    moves Q1 between 0.11 and 0.13 and can invert the Q1-Q2 ordering. With
    blocks, every bin inside the atom reports the atom's mean, which is the
    honest answer.

    Each bin holds exactly 1/n of the mass, so the unweighted mean of the n
    returned values equals the population mean of ``x`` to machine precision.
    """
    order = np.argsort(key, kind='stable')
    k, w, xs = np.asarray(key)[order], np.asarray(weights)[order], np.asarray(x)[order]
    # Collapse ties into blocks of equal key.
    starts = np.flatnonzero(np.r_[True, k[1:] != k[:-1]])
    block_w = np.add.reduceat(w, starts)
    block_wx = np.add.reduceat(w * xs, starts)
    total = block_w.sum()
    if total <= 0:
        raise ValueError("distribution has no mass; cannot form quantiles")
    cum = np.r_[0.0, np.cumsum(block_w)]
    edges = np.linspace(0.0, 1.0, n + 1) * total
    out = {}
    for i in range(n):
        lo, hi = edges[i], edges[i + 1]
        overlap = np.clip(np.minimum(cum[1:], hi) - np.maximum(cum[:-1], lo),
                          0.0, None)
        mass = overlap.sum()
        if mass <= 0:
            raise ValueError(f"quantile bin {i + 1} of {n} is empty")
        frac = np.divide(overlap, block_w, out=np.zeros_like(overlap),
                         where=block_w > 0)
        out[f'{prefix}{i + 1}'] = float(np.sum(frac * block_wx) / mass)
    return out


# --- MPCs on a refined liquid grid ------------------------------------------
#
# The production liquid grid (nB=50, pivot 0.25) has its first positive point
# at b = 0.0286, i.e. 6.7% of mean quarterly post-tax income. Every windfall
# smaller than that -- which includes a $500-1,500 rebate per household --
# lands inside the first grid segment, so its arc MPC is that segment's slope
# and not an economic quantity. The functions below re-solve the HOUSEHOLD
# block alone on a liquid grid that is denser near the constraint, holding
# every price and parameter (beta, chi1, rb, ra, tax, w, N) at the solved
# general-equilibrium steady state. Nothing is recalibrated and no published
# number from the production steady state changes.
#
# Caveat: at fixed prices the finer grid's stationary distribution holds
# slightly less liquid wealth (B = 1.019 vs the 1.04 target at nB=200), a
# partial-equilibrium drift of 2% that the MPC statistic tolerates but a GE
# object would not. Use these for MPC statistics only, never for IRFs.

#: Refined liquid grid used by ``refined_household_ss`` by default. Converged:
#: rebate-sized MPCs move by less than 1e-4 from here to nB=400.
REFINED_LIQUID_GRID = {'nB': 200, 'b_pivot': 0.01}

_REFINED_CACHE: dict = {}


def _make_grids_refined(bmax, amax, kmax, nB, nA, nK, nZ, rho_z, sigma_z,
                        b_pivot):
    """``two_asset.make_grids`` with the liquid-grid pivot exposed.

    The toolkit's ``agrid`` spaces points evenly in log(b + 0.25); a smaller
    pivot packs more of them near the constraint at b = 0. With
    ``b_pivot=0.25`` this reproduces the production grid exactly.
    """
    from sequence_jacobian import grids
    b_grid = np.geomspace(b_pivot, bmax + b_pivot, nB) - b_pivot
    b_grid[0] = 0.0
    a_grid = grids.agrid(amax=amax, n=nA)
    k_grid = grids.agrid(amax=kmax, n=nK)[::-1].copy()
    e_grid, _, Pi = grids.markov_rouwenhorst(rho=rho_z, sigma=sigma_z, N=nZ)
    return b_grid, a_grid, k_grid, e_grid, Pi


def refined_household_ss(ss, nB: int | None = None,
                         b_pivot: float | None = None):
    """Household steady state on a refined liquid grid, at ``ss`` prices.

    Returns a steady-state dict whose ``internals['hh']`` can be passed to
    every MPC function in this module. Cached on ``(id(ss), nB, b_pivot)``;
    the entry keeps ``ss`` alive so an address can never be reused.
    """
    nB = REFINED_LIQUID_GRID['nB'] if nB is None else int(nB)
    b_pivot = (REFINED_LIQUID_GRID['b_pivot'] if b_pivot is None
               else float(b_pivot))
    key = (id(ss), nB, b_pivot)
    if key in _REFINED_CACHE:
        return _REFINED_CACHE[key][1]
    from sequence_jacobian.examples import two_asset as ta
    block = ta.hh.add_hetinputs([ta.income, _make_grids_refined])
    inputs = dict(ss.toplevel)
    inputs.update(nB=nB, b_pivot=b_pivot)
    out = block.steady_state(inputs)
    _REFINED_CACHE[key] = (ss, out)
    return out


def refined_aggregate_mpc(ss, windfall_frac: float = DEFAULT_WINDFALL_FRAC,
                          nB: int | None = None,
                          b_pivot: float | None = None) -> float:
    """Aggregate quarterly MPC out of ``windfall_frac`` on a refined liquid grid.

    Use this for rebate-sized windfalls (below ~6.7% of quarterly income),
    where the production-grid number is the first grid segment's slope. At
    the 25% default it agrees with ``aggregate_mpc`` to about 0.001.
    """
    if windfall_frac is None or windfall_frac <= 0:
        raise ValueError("refined MPCs need a positive windfall_frac")
    return aggregate_mpc(refined_household_ss(ss, nB, b_pivot), windfall_frac)


def windfall_allocation(ss, windfall_frac: float = DEFAULT_WINDFALL_FRAC,
                        illiquid_threshold: float = DEFAULT_HTM_THRESHOLD
                        ) -> dict:
    """Where a liquid windfall goes, by group: consumed, kept liquid, or
    moved into the illiquid account.

    For each group returns its population ``mass`` and the population-mean
    arc responses per unit of windfall: ``mpc`` (consumption),
    ``mps_liquid`` (end-of-period b'), ``mps_illiquid`` (end-of-period a')
    and ``adj_cost`` (change in portfolio adjustment cost). The four add up
    to 1 + rb, because the windfall arrives in beginning-of-period liquid
    assets and earns rb before it is spent; that is the household budget
    constraint, and a test checks it.

    Groups split households by whether their liquid policy is AT the
    borrowing limit (b' = 0), and the constrained ones by illiquid wealth.
    This is the diagnostic behind the model's low MPC: constrained
    households, half the population, move most of a windfall into the
    illiquid account rather than consuming it.

    Pass ``refined_household_ss(ss)`` as ``ss`` for rebate-sized windfalls.
    """
    if windfall_frac is None or windfall_frac <= 0:
        raise ValueError("windfall_allocation needs a positive windfall_frac")
    h = _hh(ss)
    D, b_grid, a_grid = h['D'], h['b_grid'], h['a_grid']
    w = windfall_frac * mean_quarterly_labour_income(ss)
    resp = {name: (_shift_along_b(h[pol], b_grid, w) - h[pol]) / w
            for name, pol in (('mpc', 'c'), ('mps_liquid', 'b'),
                              ('mps_illiquid', 'a'), ('adj_cost', 'chi'))}
    constrained = h['b'] <= b_grid[0]
    has_illiquid = np.broadcast_to(a_grid[None, None, :] >= illiquid_threshold,
                                   D.shape)
    groups = {
        'all': np.ones(D.shape, bool),
        'constrained': constrained,
        'constrained_with_illiquid': constrained & has_illiquid,
        'constrained_without_illiquid': constrained & ~has_illiquid,
        'unconstrained': ~constrained,
    }
    out = {'windfall_frac': float(windfall_frac),
           'windfall_model_units': float(w)}
    for g, m in groups.items():
        mass = float(D[m].sum())
        out[g] = {'mass': mass}
        for k, r in resp.items():
            out[g][k] = float((D[m] * r[m]).sum() / mass) if mass > 0 else None
    return out


def _shift_along_b(pol, b_grid, w):
    """pol(z, b + w, a), linear along b, extrapolated past the top point."""
    out = np.empty_like(pol)
    target = b_grid + w
    over = target > b_grid[-1]
    span = b_grid[-1] - b_grid[-2]
    for iz in range(pol.shape[0]):
        for ia in range(pol.shape[2]):
            col = pol[iz, :, ia]
            vals = np.interp(target, b_grid, col)
            if over.any():
                slope = (col[-1] - col[-2]) / span
                vals = np.where(over,
                                col[-1] + slope * (target - b_grid[-1]), vals)
            out[iz, :, ia] = vals
    return out


def summary(ss, windfall_frac: float | None = DEFAULT_WINDFALL_FRAC) -> dict:
    """Steady-state distributional statistics, each labelled with its convention.

    MPCs carry the windfall size they were measured at, because the MPC out of
    a small windfall is not grid-converged at production grids (see
    ``mpc_liquid``). The hand-to-mouth count carries its threshold and the
    share exactly at the constraint that drives it.
    """
    htm = hand_to_mouth_breakdown(ss)
    income = mean_quarterly_labour_income(ss)
    return {
        'aggregate_quarterly_mpc_liquid': aggregate_mpc(ss, windfall_frac),
        'mpc_by_liquid_quartile': mpc_by_liquid_quartile(ss, windfall_frac),
        'hand_to_mouth_share': htm['hand_to_mouth_share'],
        'mpc_windfall_frac_of_quarterly_income': windfall_frac,
        'mpc_windfall_model_units': (None if windfall_frac is None
                                     else windfall_frac * income),
        'mean_quarterly_labour_income': income,
        'hand_to_mouth_breakdown': htm,
    }
