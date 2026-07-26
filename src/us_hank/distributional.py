"""Distributional outputs from the solved two-asset HANK steady state.

All functions work off ``ss.internals['hh']``: the joint distribution D and
the consumption policy c on the (z, b, a) grid.
"""

from __future__ import annotations

import numpy as np

DEFAULT_HTM_THRESHOLD = 0.05  # liquid assets below ~1.25% of quarterly income scale


def _hh(ss):
    return ss.internals['hh']


def mpc_liquid(ss):
    """Pointwise quarterly MPC out of liquid assets (finite difference of the
    consumption policy along the b grid, capped at 1). Returns (mpc, D, b_grid)."""
    h = _hh(ss)
    D, c, b_grid = h['D'], h['c'], h['b_grid']
    mpc = np.empty_like(c)
    mpc[:, 1:-1, :] = (c[:, 2:, :] - c[:, :-2, :]) / (b_grid[2:] - b_grid[:-2])[None, :, None]
    mpc[:, 0, :] = (c[:, 1, :] - c[:, 0, :]) / (b_grid[1] - b_grid[0])
    mpc[:, -1, :] = (c[:, -1, :] - c[:, -2, :]) / (b_grid[-1] - b_grid[-2])
    return np.minimum(mpc, 1.0), D, b_grid


def aggregate_mpc(ss) -> float:
    """Population-weighted aggregate quarterly MPC out of liquid assets."""
    mpc, D, _ = mpc_liquid(ss)
    return float(np.vdot(D, mpc))


def mpc_by_liquid_quartile(ss) -> dict:
    """Mean quarterly MPC within each liquid-wealth quartile (Q1 = poorest)."""
    mpc, D, b_grid = mpc_liquid(ss)
    b = np.broadcast_to(b_grid[None, :, None], D.shape).ravel()
    return _mean_by_quantile(mpc.ravel(), D.ravel(), b, 4, 'Q')


def hand_to_mouth_share(ss, threshold: float = DEFAULT_HTM_THRESHOLD) -> float:
    """Population share with liquid assets b < threshold (incl. wealthy HtM)."""
    h = _hh(ss)
    D, b_grid = h['D'], h['b_grid']
    return float(D[:, b_grid < threshold, :].sum())


def consumption_response_by_wealth_quantile(ss, irf: dict, n: int = 4) -> dict:
    """Approximate impact consumption response (% dev) by TOTAL-wealth quantile.

    First-order approximation using steady-state policies: each household's
    impact consumption response to the aggregate shock is proxied by its
    liquid-asset MPC times its share of the aggregate impact income change,
    plus the common (representative) component. This uses only steady-state
    objects; it captures the MPC-heterogeneity channel of the response, not
    full household-level dynamics.

    irf: output of ``model.shock`` (uses irf['C'][0], the aggregate impact
    consumption response in %).
    """
    mpc, D, b_grid = mpc_liquid(ss)
    h = _hh(ss)
    a_grid = h['a_grid']
    wealth = (np.broadcast_to(b_grid[None, :, None], D.shape)
              + np.broadcast_to(a_grid[None, None, :], D.shape)).ravel()
    Df, mf = D.ravel(), mpc.ravel()
    agg_mpc = float(np.vdot(Df, mf))
    dC_agg = float(irf['C'][0])
    # scale group responses in proportion to relative MPC, so weighted mean = dC_agg
    rel = mf / agg_mpc if agg_mpc != 0 else np.ones_like(mf)
    return _mean_by_quantile(dC_agg * rel, Df, wealth, n, 'Q')


def _mean_by_quantile(x, weights, key, n, prefix):
    order = np.argsort(key, kind='stable')
    w = weights[order]
    cw = np.cumsum(w) / np.sum(w)
    edges = np.linspace(0, 1, n + 1)
    out = {}
    for i in range(n):
        m = (cw > edges[i]) & (cw <= edges[i + 1] + 1e-12)
        out[f'{prefix}{i + 1}'] = float(np.sum((w * x[order])[m]) / np.sum(w[m]))
    return out


def summary(ss) -> dict:
    return {
        'aggregate_quarterly_mpc_liquid': aggregate_mpc(ss),
        'mpc_by_liquid_quartile': mpc_by_liquid_quartile(ss),
        'hand_to_mouth_share': hand_to_mouth_share(ss),
    }
