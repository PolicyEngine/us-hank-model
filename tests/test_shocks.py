"""IRF sanity checks at production grids."""

import numpy as np
import pytest

from us_hank import model

pytestmark = pytest.mark.slow


def test_monetary_easing_expansionary(ss, G_jac):
    irf = model.shock('monetary', -0.0025, 0.61, T=300, ss=ss, G_jac=G_jac)
    assert irf['Y'][0] > 0
    assert irf['C'][0] > 0
    assert irf['I'][0] > 0
    assert irf['pi'][0] > 0
    assert irf['r'][0] < 0  # real rate falls


def test_fiscal_spending_crowds_out_investment(ss, G_jac):
    irf = model.shock('fiscal_spending', 0.01, 0.8, T=300, ss=ss, G_jac=G_jac)
    assert irf['Y'][0] > 0
    assert irf['I'][0] < 0  # crowding out under active Taylor rule
    assert irf['C'][0] < 0  # tax-financed G crowds out consumption
    assert irf['pi'][0] > 0


def test_irfs_revert_to_steady_state(ss, G_jac):
    irf = model.shock('monetary', -0.0025, 0.61, T=300, ss=ss, G_jac=G_jac)
    for v in ('Y', 'C', 'I', 'pi', 'r'):
        assert abs(irf[v][-1]) < 5e-3
        assert abs(irf[v][-1]) < 0.05 * np.max(np.abs(irf[v]))


def test_goods_market_clears_along_irf(ss, G_jac):
    dx = -0.0025 * 0.61 ** np.arange(300)
    dgoods = G_jac['goods_mkt']['rstar'] @ dx
    # goods_mkt is untargeted (implied by Walras' law); residual is small
    # relative to the Y response (~5e-5 vs ~5e-3 level dY)
    assert np.max(np.abs(dgoods)) < 1e-4


def test_unknown_kind_raises():
    with pytest.raises(ValueError):
        model.shock('transfer', 0.01, 0.8)
