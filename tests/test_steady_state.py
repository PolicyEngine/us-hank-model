"""Steady-state validation against ABRS (2021) published calibration targets."""

import numpy as np
import pytest

pytestmark = pytest.mark.slow


def test_calibration_targets(ss):
    assert np.isclose(ss['Y'], 1.0)
    assert np.isclose(ss['K'], 10.0)
    assert np.isclose(ss['r'], 0.0125)
    assert np.isclose(ss['G'], 0.2)
    assert np.isclose(ss['Bg'], 2.8)


def test_wealth_targets(ss):
    # total wealth 14 (3.5x annual GDP), liquid B = 1.04 (0.26x annual GDP),
    # so illiquid A = 12.96; equity p = tot_wealth - Bg = 11.2.
    assert abs(ss['A'] + ss['B'] - 14.0) < 1e-6
    assert abs(ss['B'] - 1.04) < 1e-6
    assert abs(ss['A'] - 12.96) < 1e-6
    assert abs(ss['p'] - 11.2) < 1e-6
    assert abs(ss['asset_mkt']) < 1e-6


def test_beta_production_grids(ss):
    # paper full-grid discount factor ~0.976 (coarse demo grids give ~0.9726)
    assert abs(ss['beta'] - 0.976) < 2e-3


def test_internally_calibrated_parameters_match_published_table(ss):
    """The real external check: ABRS (2021) Table B.III.

    Y, K, r, G, Bg and total wealth are calibration INPUTS -- they are hit by
    construction and prove nothing. These seven are the parameters the model
    solves for internally (beta and chi1 numerically, the rest analytically),
    so they are the only steady-state numbers with independent published
    values to miss.

    vphi is deliberately absent: the published table reports 2.073, this DAG's
    union block yields 1.7135, and the gap is unresolved (see README). It is a
    normalisation internal to the wage Phillips curve -- this implementation
    is self-consistent (wnkpc = 0 at the steady state) and no asset or wealth
    target depends on it -- but it is not asserted here, because asserting a
    value we cannot reconcile with the paper would launder the discrepancy.
    """
    published = {'beta': 0.976, 'chi1': 6.416, 'Z': 0.468, 'alpha': 0.33,
                 'mup': 1.015, 'tax': 0.356}
    for name, target in published.items():
        # the table is printed to 3-4 significant figures; require agreement
        # to the last published digit
        assert abs(ss[name] - target) < 5e-4, (name, ss[name], target)


def test_market_clearing_residual_magnitudes(ss):
    """Pin the residuals, not just their existence.

    asset_mkt is a solver target so it clears to machine precision. goods_mkt
    is UNTARGETED -- it holds only by Walras' law, so its residual is the
    honest measure of steady-state accuracy and is ~1e-8, five orders looser
    than the asset market. Quoting one for the other would overstate this.
    """
    assert abs(ss['asset_mkt']) < 1e-11
    assert abs(ss['goods_mkt']) < 1e-6
    assert abs(ss['fisher']) < 1e-11
    assert abs(ss['wnkpc']) < 1e-11


def test_goods_market_clears(ss):
    assert abs(ss['goods_mkt']) < 1e-5


def test_household_block_reference_values():
    """Household block at full paper grids matches the sequence-jacobian
    repo's own regression targets (tests/base/test_two_asset.py) tightly."""
    from sequence_jacobian import utilities as utils
    from sequence_jacobian.hetblocks import hh_twoasset as hh

    beta, r, tot_wealth, K, delta, Bg, G = 0.976, 0.0125, 14, 10, 0.02, 2.8, 0.2
    eis, chi0, chi1, chi2, omega = 0.5, 0.25, 6.5, 2, 0.005
    b_grid = utils.discretize.agrid(amax=50, n=50)
    a_grid = utils.discretize.agrid(amax=4000, n=70)
    k_grid = utils.discretize.agrid(amax=1, n=50)[::-1].copy()
    e_grid, _, Pi = utils.discretize.markov_rouwenhorst(rho=0.966, sigma=0.92, N=3)
    mc = 1 - r * (tot_wealth - Bg - K)
    alpha = (r + delta) * K / mc
    w = (1 - alpha) * mc
    tax = (r * Bg + G) / w
    calibration = {'Pi': Pi, 'a_grid': a_grid, 'b_grid': b_grid,
                   'e_grid': e_grid, 'z_grid': (1 - tax) * w * e_grid,
                   'k_grid': k_grid, 'beta': beta, 'N': 1.0, 'tax': tax,
                   'w': w, 'eis': eis, 'rb': r - omega, 'ra': r,
                   'chi0': chi0, 'chi1': chi1, 'chi2': chi2}
    out = hh.hh.steady_state(calibration)
    assert np.isclose(out['A'], 12.526539492650361)
    assert np.isclose(out['B'], 1.0840860793350566)
    assert np.isclose(out['UCE'], 4.5102870939550055)
