import numpy as np
import pytest

from us_hank import one_asset


def test_steady_state():
    ss = one_asset.solve_steady_state()
    assert ss['Y'] == pytest.approx(1.0)
    assert ss['r'] == pytest.approx(0.005)
    assert abs(ss['asset_mkt']) < 1e-6


def test_monetary_easing_expansionary():
    irf = one_asset.shock('monetary', -0.0025, 0.61, T=300)
    assert irf['Y'][0] > 0
    assert irf['C'][0] > 0
    assert irf['pi'][0] > 0


def test_monetary_tightening_is_contractionary():
    irf = one_asset.shock('monetary', 0.0025, 0.61, T=300)
    assert irf['Y'][0] < 0
    assert irf['C'][0] < 0
    assert irf['pi'][0] < 0


def test_productivity_shock_signs():
    irf = one_asset.shock('productivity', 0.01, 0.9, T=300)
    assert irf['Y'][0] > 0
    assert irf['C'][0] > 0
    assert irf['pi'][0] < 0


def test_responses_are_exactly_linear():
    base = one_asset.shock('monetary', -0.001, 0.61, T=300)
    scaled = one_asset.shock('monetary', -0.005, 0.61, T=300)
    for v in ('Y', 'C', 'pi', 'r'):
        assert np.allclose(scaled[v] / 5.0, base[v], rtol=1e-12, atol=1e-14)


def test_no_investment_series():
    """No capital in this DAG, so no investment IRF is invented."""
    irf = one_asset.shock('monetary', -0.0025, 0.61, T=300)
    assert 'I' not in irf


def test_no_fiscal_instrument():
    with pytest.raises(ValueError):
        one_asset.shock('fiscal_spending', 0.01, 0.8)


def test_solved_on_the_notebook_grid_not_the_demo_grid():
    """The packaged example's dag() uses 2 income states and 10 asset points;
    the toolkit's own one-asset notebook uses 7 and 500. Regression pins on
    the notebook grid (measured 14 Nov 2026, sequence-jacobian 1.0.0); the
    demo grid gave 0.246 and 0.450, which these bounds reject."""
    ss = one_asset.solve_steady_state()
    assert (ss['nS'], ss['nA']) == (7, 500)
    m = one_asset.shock('monetary', -0.0025, 0.61, T=300)
    z = one_asset.shock('productivity', 0.01, 0.9, T=300)
    assert m['Y'][0] == pytest.approx(0.1908, abs=2e-3)
    assert z['Y'][0] == pytest.approx(0.5755, abs=2e-3)
