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
# labour income. 0.25 is about a $1,000-1,500 rebate against US median
# household income -- the scale of the 2001/2008 US tax rebates that the
# empirical MPC literature actually measures.
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
    shifted = np.empty_like(c)
    target = b_grid + windfall
    over = target > b_grid[-1]
    span = b_grid[-1] - b_grid[-2]
    for iz in range(c.shape[0]):
        for ia in range(c.shape[2]):
            col = c[iz, :, ia]
            vals = np.interp(target, b_grid, col)
            if over.any():
                slope = (col[-1] - col[-2]) / span
                vals = np.where(over,
                                col[-1] + slope * (target - b_grid[-1]), vals)
            shifted[iz, :, ia] = vals
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
