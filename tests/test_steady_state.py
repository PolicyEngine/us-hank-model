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
    # total wealth 14 (3.5x annual GDP), liquid B = 1.04 (0.26x annual GDP)
    assert abs(ss['A'] + ss['B'] - 14.0) < 1e-6
    assert abs(ss['B'] - 1.04) < 1e-6
    assert abs(ss['asset_mkt']) < 1e-6


def test_beta_production_grids(ss):
    # paper full-grid discount factor ~0.976 (coarse demo grids give ~0.9726)
    assert abs(ss['beta'] - 0.976) < 2e-3


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
