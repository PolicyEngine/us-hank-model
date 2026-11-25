"""Units of ``size`` and of the rate responses, derived from the model.

PolicyEngine/macro#211 found two wrappers describing ``size=-0.0025`` as "a
25bp easing" (it is 25bp a quarter, 100bp annualised) and ``size=0.01`` on
productivity as "1% TFP" (2.14% in the two-asset model). The conversions are
asserted here against the model's own equations and steady state, so a
description can be checked against a function rather than against prose.
"""

import numpy as np
import pytest

from us_hank import model, one_asset

T = 300


# --- fast: arithmetic and the one-asset variant ------------------------------

def test_the_model_period_is_a_quarter():
    """rho_z = 0.966 and delta = 0.02 are quarterly values (ABRS Table B.III),
    and the steady-state rate 0.0125 is the paper's 5% a year over four."""
    assert model.PERIODS_PER_YEAR == 4
    cal = model.PRODUCTION_CALIBRATION
    assert cal['r'] * model.PERIODS_PER_YEAR == pytest.approx(0.05)
    assert cal['rstar'] == cal['r']


@pytest.mark.parametrize("size, bp", [(-0.0025, -100.0), (-0.000625, -25.0),
                                      (0.01, 400.0), (-0.0125, -500.0)])
def test_monetary_size_in_annualised_basis_points(size, bp):
    """-0.0025 is 100bp annualised, not 25bp; a 25bp move is -0.000625.

    The last case is the paper's Fig. 7a shock as this repo reads it: -5pp
    annualised is size=-0.0125 (tests/test_published_magnitude.py).
    """
    assert model.annualised_bp(size) == pytest.approx(bp)
    assert model.monetary_size(bp) == pytest.approx(size)


def test_unit_strings_quote_the_conversions_the_helpers_give():
    """The prose in SHOCK_UNITS must agree with the helper functions."""
    for units in (model.SHOCK_UNITS['monetary']['units'],
                  one_asset.SHOCK_UNITS['monetary']['units']):
        assert '-0.0025 = a 25bp-per-quarter cut' in ' '.join(units.split())
        assert 'about 100bp annualised' in units
        assert '-0.000625 = 25bp annualised' in units
    assert model.annualised_bp(-0.0025) == pytest.approx(-100)
    assert model.annualised_bp(-0.000625) == pytest.approx(-25)


def test_one_asset_units_are_reported():
    assert set(one_asset.SHOCK_UNITS) == set(one_asset.SHOCK_KINDS)
    for kind, inp in one_asset.SHOCK_KINDS.items():
        irf = one_asset.shock(kind, 0.01, 0.9, T=T)
        assert irf['units'] == one_asset.SHOCK_UNITS[kind]['units']
        assert one_asset.SHOCK_UNITS[kind]['input'] == inp == irf['shock_input']


def test_one_asset_tfp_level_is_a_percentage_but_two_asset_is_not():
    """Z_ss is 1 in the one-asset DAG, so there 0.01 IS a 1% TFP shock. The
    same size in the two-asset model is 2.14% (slow test below)."""
    ss = one_asset.solve_steady_state()
    assert ss['Z'] == pytest.approx(1.0)
    irf = one_asset.shock('productivity', 0.01, 0.9, T=T)
    assert irf['shock_pct_of_ss'][0] == pytest.approx(1.0)
    assert one_asset.SHOCK_UNITS['productivity']['level_equals_pct_of_ss']
    assert not model.SHOCK_UNITS['productivity']['level_equals_pct_of_ss']


def test_one_asset_steady_state_rate_differs_from_two_asset():
    """0.5% a quarter (2% a year) against 1.25% (5% a year): the same
    ``size`` is the same number of basis points in both, but a different
    share of the steady-state rate."""
    ss = one_asset.solve_steady_state()
    assert ss['rstar'] == pytest.approx(0.005)
    irf = one_asset.shock('monetary', -0.0025, 0.61, T=T)
    assert irf['shock_pct_of_ss'][0] == pytest.approx(-50.0)
    assert model.PRODUCTION_CALIBRATION['rstar'] == pytest.approx(0.0125)


def test_one_asset_rule_is_lagged_one_quarter():
    """1 + r_t = (1 + rstar_{t-1} + phi*pi_{t-1}) / (1 + pi_t): on impact the
    real return moves only with the inflation surprise, r_0 = -(1 + r)*pi_0,
    and the intercept itself arrives in quarter 1 (to first order,
    r_1 = rstar_0 + phi*pi_0 - (1 + r)*pi_1)."""
    ss = one_asset.solve_steady_state()
    size, rho = -0.0025, 0.61
    irf = one_asset.shock('monetary', size, rho, T=T)
    r, pi = irf['r'] / 100, irf['pi'] / 100
    assert r[0] == pytest.approx(-(1 + ss['r']) * pi[0], rel=1e-9)
    expected = size + ss['phi'] * pi[0] - (1 + ss['r']) * pi[1]
    assert r[1] == pytest.approx(expected, rel=1e-9)


def test_size_pct_of_ss_rejects_an_unknown_kind():
    with pytest.raises(ValueError, match="unknown shock kind"):
        model.size_pct_of_ss('transfer', 0.01, ss={})


# --- slow: the two-asset model at production grids ---------------------------

@pytest.mark.slow
def test_size_pct_of_ss_matches_the_irf(ss, G_jac):
    for kind, size, rho in (('productivity', 0.01, 0.9),
                            ('fiscal_spending', 0.01, 0.8),
                            ('monetary', -0.0025, 0.61)):
        irf = model.shock(kind, size, rho, T=T, ss=ss, G_jac=G_jac)
        assert model.size_pct_of_ss(kind, size, ss=ss) == pytest.approx(
            irf['shock_pct_of_ss'][0])
    assert model.size_pct_of_ss('productivity', 0.01, ss=ss) == pytest.approx(
        2.138, abs=1e-3)
    assert model.size_pct_of_ss('fiscal_spending', 0.01, ss=ss) == pytest.approx(5.0)
    assert model.size_pct_of_ss('monetary', -0.0025, ss=ss) == pytest.approx(-20.0)


@pytest.mark.slow
def test_a_one_percent_tfp_shock_is_size_0_00468(ss, G_jac):
    size = model.size_from_pct_of_ss('productivity', 1.0, ss=ss)
    assert size == pytest.approx(0.004678, abs=1e-6)
    irf = model.shock('productivity', size, 0.9, T=T, ss=ss, G_jac=G_jac)
    assert irf['shock_pct_of_ss'][0] == pytest.approx(1.0)
    # First-order responses scale exactly, so the 0.01 response is 2.138
    # times the 1% response: quoting it as "a 1% TFP shock" overstates by that.
    big = model.shock('productivity', 0.01, 0.9, T=T, ss=ss, G_jac=G_jac)
    assert big['Y'][0] / irf['Y'][0] == pytest.approx(2.138, abs=1e-3)


@pytest.mark.slow
def test_monetary_size_is_the_intercept_of_the_quarterly_rule(ss, G_jac):
    """i_t - phi*pi_t = rstar_t in every quarter of the response.

    So ``size`` is a move in the intercept of a rule for the QUARTERLY nominal
    rate, and the nominal rate moves by ``size`` only if inflation does not
    respond. Derived from the general-equilibrium Jacobian, not from the
    units string.
    """
    size, rho = model.monetary_size(-100), 0.61
    assert size == pytest.approx(-0.0025)
    dx = size * rho ** np.arange(T)
    di = G_jac['i']['rstar'] @ dx
    dpi = G_jac['pi']['rstar'] @ dx
    assert np.allclose(di - ss['phi'] * dpi, dx, rtol=0, atol=1e-12)
    # The annualised intercept move is 100bp; the annualised policy rate
    # falls 53bp on impact (13bp a quarter, as SHOCK_UNITS says).
    assert model.annualised_bp(dx[0]) == pytest.approx(-100.0)
    assert model.annualised_bp(di[0]) == pytest.approx(-52.6, abs=1.0)
    assert 100 * di[0] == pytest.approx(-0.13, abs=5e-3)


@pytest.mark.slow
def test_rate_responses_are_quarterly(ss, G_jac):
    """``pi`` and ``r`` are quarterly rates in percentage points: the ex-post
    return identity r_0 = -(1 + r_ss)*pi_0 holds in those units, with the
    quarterly r_ss = 0.0125, and fails with an annualised one."""
    irf = model.shock('monetary', -0.0025, 0.61, T=T, ss=ss, G_jac=G_jac)
    assert ss['r'] == pytest.approx(0.0125)
    assert irf['r'][0] == pytest.approx(-(1 + ss['r']) * irf['pi'][0], rel=1e-9)
    assert irf['r'][0] != pytest.approx(-(1 + 4 * ss['r']) * irf['pi'][0], rel=1e-3)
    # Annualised: inflation +0.32pp, ex-post real return -0.32pp on impact.
    assert 4 * irf['pi'][0] == pytest.approx(0.316, abs=5e-3)
