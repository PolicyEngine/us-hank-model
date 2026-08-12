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


def test_monetary_tightening_is_contractionary(ss, G_jac):
    """The mirror image of the easing test, checked explicitly.

    It is implied by linearity, but a contractionary shock is the experiment
    users actually run, and an accidental sign flip in the plumbing would be
    invisible if only easing were ever tested.
    """
    irf = model.shock('monetary', 0.0025, 0.61, T=300, ss=ss, G_jac=G_jac)
    assert irf['Y'][0] < 0
    assert irf['C'][0] < 0
    assert irf['I'][0] < 0
    assert irf['pi'][0] < 0
    assert irf['r'][0] > 0  # real rate rises


def test_productivity_shock_signs(ss, G_jac):
    """A positive TFP shock raises output, consumption and investment and is
    disinflationary. Previously untested despite being a shipped shock kind."""
    irf = model.shock('productivity', 0.01, 0.9, T=300, ss=ss, G_jac=G_jac)
    assert irf['Y'][0] > 0
    assert irf['C'][0] > 0
    assert irf['I'][0] > 0
    assert irf['pi'][0] < 0  # supply expansion is disinflationary
    # output rises by more than TFP does: a level shock to Z is a 2.14% TFP
    # improvement, and the response is amplified beyond it
    assert irf['Y'][0] > irf['shock_pct_of_ss'][0]


def test_fiscal_spending_crowds_out_investment(ss, G_jac):
    irf = model.shock('fiscal_spending', 0.01, 0.8, T=300, ss=ss, G_jac=G_jac)
    assert irf['Y'][0] > 0
    assert irf['I'][0] < 0  # crowding out under active Taylor rule
    assert irf['C'][0] < 0  # tax-financed G crowds out consumption
    assert irf['pi'][0] > 0
    # impact multiplier is well below 1: this is a BALANCED-BUDGET multiplier
    # under an active Taylor rule, not a deficit-financed one
    assert 0 < irf['Y'][0] / 100 / 0.01 < 0.5


def test_fiscal_spending_is_tax_financed_not_deficit_financed(ss, G_jac):
    """The scope claim that most invites misreading, asserted numerically.

    Bond supply Bg is a fixed scalar -- it is not an input to the GE Jacobian
    and not one of its outputs -- so there is no deficit channel in this DAG
    at all. The labour tax does all the work, and the government budget
    tax*w*N = r*Bg + G must hold period by period, not just in present value.
    """
    T = 300
    dx = 0.01 * 0.8 ** np.arange(T)
    assert 'Bg' not in G_jac.inputs
    assert 'Bg' not in G_jac.outputs

    irf = model.shock('fiscal_spending', 0.01, 0.8, T=T, ss=ss, G_jac=G_jac)
    assert irf['tax'][0] > 0  # the tax rate rises on impact
    assert np.all(irf['tax'][:40] > 0)

    dtax = G_jac['tax']['G'] @ dx
    dw = G_jac['w']['G'] @ dx
    dN = G_jac['N']['G'] @ dx
    dr = G_jac['r']['G'] @ dx
    revenue = (dtax * ss['w'] * ss['N'] + ss['tax'] * dw * ss['N']
               + ss['tax'] * ss['w'] * dN)
    outlays = dr * ss['Bg'] + dx
    assert np.max(np.abs(revenue - outlays)) < 1e-12


def test_responses_are_exactly_linear_in_shock_size(ss, G_jac):
    """Linearity is claimed; it is exact by construction, so assert it exactly.

    The IRF is a fixed Jacobian times the shock path, so scaling the size must
    scale every response by the same factor to machine precision. A tolerance
    any looser than this would let a genuine nonlinearity or a stray
    size-dependent branch slip through unnoticed.
    """
    for kind, rho in [('monetary', 0.61), ('fiscal_spending', 0.8),
                      ('productivity', 0.9)]:
        base = model.shock(kind, 0.001, rho, T=300, ss=ss, G_jac=G_jac)
        for factor in (10.0, -3.0):
            scaled = model.shock(kind, 0.001 * factor, rho, T=300, ss=ss,
                                 G_jac=G_jac)
            for v in ('Y', 'C', 'I', 'pi', 'r', 'tax'):
                assert np.allclose(scaled[v] / factor, base[v],
                                   rtol=1e-12, atol=1e-14), (kind, v, factor)


def test_shock_units_are_reported_and_productivity_is_not_a_percent(ss, G_jac):
    """size is a LEVEL change, and for TFP that is not a percent change.

    Z_ss = 0.468, so size=0.01 is a 2.14% TFP shock. Anything describing it as
    "a 1% TFP shock" is wrong by more than a factor of two, which is why the
    result carries shock_pct_of_ss rather than leaving callers to guess.
    """
    assert set(model.SHOCK_UNITS) == set(model.SHOCK_KINDS)
    irf = model.shock('productivity', 0.01, 0.9, T=300, ss=ss, G_jac=G_jac)
    assert irf['shock_pct_of_ss'][0] == pytest.approx(100 * 0.01 / ss['Z'])
    assert irf['shock_pct_of_ss'][0] == pytest.approx(2.138, abs=1e-2)
    assert not model.SHOCK_UNITS['productivity']['level_equals_pct_of_ss']

    # G is the benign case only because Y_ss = 1
    irf_g = model.shock('fiscal_spending', 0.01, 0.8, T=300, ss=ss, G_jac=G_jac)
    assert irf_g['shock'][0] / ss['Y'] == pytest.approx(0.01)
    assert irf_g['shock_pct_of_ss'][0] == pytest.approx(5.0)  # 5% rise in G


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
