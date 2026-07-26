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


def test_no_fiscal_instrument():
    with pytest.raises(ValueError):
        one_asset.shock('fiscal_spending', 0.01, 0.8)
