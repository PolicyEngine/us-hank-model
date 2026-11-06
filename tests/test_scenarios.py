"""Scenario sweep: monetary and fiscal shocks of several sizes, signs and
persistences, checked against properties that must hold in this model.

Values pinned here were produced on CI (production grids, sequence-jacobian
1.0.0) on 6 Nov 2026. They are regression pins on this implementation, not
published ABRS results.
"""

import numpy as np
import pytest

from us_hank import distributional, model

pytestmark = pytest.mark.slow

T = 300
VARS = ('Y', 'C', 'I', 'pi', 'r', 'tax')
CASES = [('monetary', s, rho)
         for s in (-0.01, -0.0025, 0.0025, 0.01) for rho in (0.0, 0.61, 0.9)]
CASES += [('fiscal_spending', s, rho)
          for s in (-0.02, -0.005, 0.005, 0.02) for rho in (0.0, 0.5, 0.8, 0.95)]
CASES += [('productivity', s, 0.9) for s in (-0.01, 0.01)]


# --- Nonlinear checks -------------------------------------------------------
# model.shock is a matrix-vector product against a fixed Jacobian, so in it a
# zero shock is zero and opposite shocks mirror by construction; tests of
# those properties cannot fail. The checks below compare it with the model's
# own nonlinear perfect-foresight transition (sequence-jacobian's
# solve_impulse_nonlinear on the same DAG), which can. Values measured on CI
# (production grids, T=300) on 6 Nov 2026; bounds are measured + margin.

NL_VARS = VARS + tuple(model.TARGETS) + ('goods_mkt',)


def _nonlinear(ss, kind, size, rho):
    _, dag = model.build_models()
    inp = model.SHOCK_KINDS[kind]
    nl = dag.solve_impulse_nonlinear(
        ss, model.UNKNOWNS, model.TARGETS, {inp: size * rho ** np.arange(T)},
        outputs=list(NL_VARS))
    out = {}
    for v in VARS:
        scale = 100 / ss[v] if v in ('Y', 'C', 'I') else 100
        out[v] = np.asarray(nl[v]) * scale
    for v in tuple(model.TARGETS) + ('goods_mkt',):
        out[v] = np.asarray(nl[v])
    return out


def _gap(nl, lin, v):
    """max |nonlinear - linear| over the path, relative to the linear peak."""
    return np.max(np.abs(nl[v] - lin[v])) / np.max(np.abs(lin[v]))


NL_CASES = [('monetary', 1e-5, 0.61), ('fiscal_spending', 1e-4, 0.8),
            ('monetary', 0.0025, 0.61), ('monetary', -0.0025, 0.61),
            ('fiscal_spending', 0.01, 0.8), ('fiscal_spending', -0.01, 0.8)]


@pytest.fixture(scope='module')
def nonlinear(ss):
    return {c: _nonlinear(ss, *c) for c in NL_CASES}


def test_steady_state_is_a_fixed_point_of_the_dynamic_model(ss):
    """Feed the dynamic DAG the steady state (every exogenous input and
    unknown at its ss value for T quarters): every target must stay at zero.
    This catches a steady-state DAG that disagrees with the dynamic one.
    Measured: asset_mkt 3.9e-6 (the household block re-iterated from the ss
    policy, against 1e-11 in the ss solve itself), fisher 0, wnkpc 4.7e-8,
    goods_mkt 7.8e-8."""
    _, dag = model.build_models()
    zero = {k: np.zeros(T) for k in model.EXOGENOUS + model.UNKNOWNS}
    out = dag.impulse_nonlinear(ss, zero, outputs=list(model.TARGETS)
                                + ['goods_mkt'])
    bounds = {'asset_mkt': 1e-5, 'fisher': 1e-10, 'wnkpc': 1e-7,
              'goods_mkt': 2e-7}
    for k, b in bounds.items():
        assert np.max(np.abs(out[k])) < b, k


@pytest.mark.parametrize('kind,size,rho,bound', [
    ('monetary', 1e-5, 0.61, 0.01),        # measured max 0.0077 (pi)
    ('fiscal_spending', 1e-4, 0.8, 0.025),  # measured max 0.0174 (Y)
])
def test_linear_irf_matches_the_nonlinear_transition_for_small_shocks(
        ss, G_jac, nonlinear, kind, size, rho, bound):
    """For a tiny shock the nonlinear transition must equal the Jacobian IRF:
    this validates G_jac and the unit scaling in model.shock (a 1% scaling
    error fails the monetary case). The remaining gap is the steady-state
    offset above, which is large only relative to a tiny shock."""
    nl = nonlinear[(kind, size, rho)]
    lin = model.shock(kind, size, rho, T=T, ss=ss, G_jac=G_jac)
    for v in VARS:
        assert _gap(nl, lin, v) < bound, v
    for k in model.TARGETS:
        assert np.max(np.abs(nl[k])) < 1e-8, k


def test_monetary_nonlinearity_at_100bp_annualised_is_small(ss, G_jac,
                                                            nonlinear):
    """A 25bp-per-quarter shock (about 100bp annualised): the linear IRF is
    within 5% of its peak of the nonlinear path for every variable (measured
    max 0.037, Y on the easing), and easing moves output more than
    tightening: Y on impact +0.511 vs -0.477 (linear 0.493), a 7.1% gap
    between the two paths. Pinned two-sided, so a model that became linear
    here would fail too."""
    up, dn = (nonlinear[('monetary', s, 0.61)] for s in (0.0025, -0.0025))
    for s, nl in ((0.0025, up), (-0.0025, dn)):
        lin = model.shock('monetary', s, 0.61, T=T, ss=ss, G_jac=G_jac)
        for v in VARS:
            assert _gap(nl, lin, v) < 0.05, (s, v)
    assert up['Y'][0] == pytest.approx(-0.4773, abs=3e-3)
    assert dn['Y'][0] == pytest.approx(0.5110, abs=3e-3)
    asym = np.max(np.abs(up['Y'] + dn['Y'])) / np.max(np.abs(up['Y']))
    assert 0.05 < asym < 0.09


def test_spending_multiplier_is_strongly_nonlinear_at_1pct_of_gdp(
        ss, G_jac, nonlinear):
    """The README's linear impact multiplier, 0.175 for a 1%-of-GDP G rise
    (rho 0.8), is a first-order number. On the nonlinear transition the same
    shock gives 0.251 and a 1% cut gives -0.125: a second-order term of about
    +0.06% of GDP that does not change sign with the shock. The linear
    multiplier is small, so this term is 44% of the linear peak. A +2%-of-GDP
    rise does not solve nonlinearly at all (a solved block's backward
    iteration fails to converge), so beyond about 1% of GDP only the linear
    IRF exists and it understates a rise and overstates a cut. Pinned; the
    tax response is the other side of it (+0.30pp impact vs 0.51pp linear)."""
    up, dn = (nonlinear[('fiscal_spending', s, 0.8)] for s in (0.01, -0.01))
    lin = model.shock('fiscal_spending', 0.01, 0.8, T=T, ss=ss, G_jac=G_jac)
    assert lin['Y'][0] == pytest.approx(0.1753, abs=2e-3)
    assert up['Y'][0] == pytest.approx(0.2508, abs=5e-3)
    assert dn['Y'][0] == pytest.approx(-0.1251, abs=5e-3)
    assert up['tax'][0] == pytest.approx(0.3012, abs=5e-3)
    assert 0.35 < _gap(up, lin, 'Y') < 0.55       # measured 0.441
    asym = np.max(np.abs(up['Y'] + dn['Y'])) / np.max(np.abs(up['Y']))
    assert 0.40 < asym < 0.60                     # measured 0.501
    for nl in (up, dn):
        for k in model.TARGETS:
            assert np.max(np.abs(nl[k])) < 1e-7, k


@pytest.mark.parametrize('kind,size,rho', CASES)
def test_equilibrium_conditions_hold_along_every_shock(ss, G_jac, kind, size,
                                                       rho):
    """The three targeted conditions hold to solver precision; goods market
    clearing is untargeted (Walras' law) and holds to a small residual."""
    inp = model.SHOCK_KINDS[kind]
    dx = size * rho ** np.arange(T)
    for target in model.TARGETS:
        assert np.max(np.abs(G_jac[target][inp] @ dx)) < 1e-12, target
    # observed max 9e-5 at 100bp/rho=0.9, against level dY of order 1e-2
    assert np.max(np.abs(G_jac['goods_mkt'][inp] @ dx)) < 2e-4 * max(
        1.0, abs(size) / 0.01)


@pytest.mark.parametrize('rho', [0.0, 0.61, 0.9])
def test_monetary_easing_signs_at_every_persistence(ss, G_jac, rho):
    irf = model.shock('monetary', -0.0025, rho, T=T, ss=ss, G_jac=G_jac)
    assert irf['Y'][0] > 0 and irf['C'][0] > 0 and irf['I'][0] > 0
    assert irf['tax'][0] < 0  # lower r*Bg lets the labour tax fall
    if rho > 0:
        assert irf['pi'][0] > 0
    else:
        # A one-quarter cut leaves impact inflation marginally NEGATIVE
        # (-0.13bp for 25bp). Pinned so a change in this corner is noticed.
        assert irf['pi'][0] == pytest.approx(-0.0013, abs=3e-4)


@pytest.mark.parametrize('rho', [0.0, 0.5, 0.8, 0.95])
def test_spending_rise_signs_at_every_persistence(ss, G_jac, rho):
    irf = model.shock('fiscal_spending', 0.01, rho, T=T, ss=ss, G_jac=G_jac)
    assert irf['Y'][0] > 0 and irf['C'][0] < 0 and irf['I'][0] < 0
    assert irf['pi'][0] > 0 and irf['tax'][0] > 0
    assert 0 < irf['Y'][0] / 100 / 0.01 < 1  # balanced-budget multiplier


@pytest.mark.parametrize('kind,size,rho', CASES)
def test_impact_real_rate_is_the_inflation_surprise(ss, G_jac, kind, size,
                                                    rho):
    """r is the ex-post real return, r_0 = (1+i_{-1})/(1+pi_0) - 1 with i_{-1}
    predetermined, so dr_0 = -(1+r_ss) * dpi_0 exactly to first order. This
    is why a spending rise LOWERS the impact debt-service bill r*Bg."""
    irf = model.shock(kind, size, rho, T=T, ss=ss, G_jac=G_jac)
    assert irf['r'][0] == pytest.approx(-(1 + ss['r']) * irf['pi'][0],
                                        rel=1e-6, abs=1e-12)


def test_spending_tax_path_and_multiplier_are_pinned(ss, G_jac):
    """README figures for a 1%-of-GDP G shock, rho=0.8.

    The tax rate rises only 0.51pp on impact, not the ~1.5pp that dG/(wN)
    alone would need, because the impact ex-post real rate falls 0.21pp and
    r*Bg (Bg = 2.8) frees 0.59% of GDP. From quarter 1 the real rate is
    positive and the tax rise jumps to 1.59pp.
    """
    irf = model.shock('fiscal_spending', 0.01, 0.8, T=T, ss=ss, G_jac=G_jac)
    assert irf['tax'][0] == pytest.approx(0.5073, abs=2e-3)
    assert irf['tax'][1] == pytest.approx(1.5859, abs=5e-3)
    assert irf['r'][0] == pytest.approx(-0.2113, abs=2e-3)
    assert irf['Y'][0] / 100 / 0.01 == pytest.approx(0.1753, abs=2e-3)
    assert irf['C'][0] == pytest.approx(-0.5360, abs=3e-3)
    assert irf['I'][0] == pytest.approx(-2.5525, abs=1e-2)


def test_monetary_reference_shock_is_pinned(ss, G_jac):
    """25bp-per-quarter easing, rho=0.61 (the README/test convention)."""
    irf = model.shock('monetary', -0.0025, 0.61, T=T, ss=ss, G_jac=G_jac)
    assert irf['Y'][0] == pytest.approx(0.4926, abs=3e-3)
    assert irf['C'][0] == pytest.approx(0.2997, abs=3e-3)
    assert irf['I'][0] == pytest.approx(1.6976, abs=1e-2)
    assert irf['pi'][0] == pytest.approx(0.0789, abs=1e-3)
    assert irf['tax'][0] == pytest.approx(-0.6331, abs=5e-3)


@pytest.mark.parametrize('kind,size,rho', CASES)
def test_wealth_quartile_allocation_sums_to_the_aggregate(ss, G_jac, kind,
                                                          size, rho):
    """Equal-population quartiles, so their mean must be the aggregate. The
    allocation shares the aggregate's sign in every quartile (it is an
    MPC-proportional split, not a solved incidence)."""
    irf = model.shock(kind, size, rho, T=T, ss=ss, G_jac=G_jac)
    q = distributional.consumption_response_by_wealth_quantile(ss, irf)
    assert sum(q.values()) / 4 == pytest.approx(irf['C'][0], rel=1e-10,
                                                abs=1e-12)
    assert all(np.sign(v) == np.sign(irf['C'][0]) for v in q.values())


# Y (% dev) for a 1%-of-GDP G shock, quarters 0-8 (CI, 6 Nov 2026).
_Y_G = {
    0.8: [0.1753, 0.0685, 0.0112, -0.0182, -0.0315, -0.0359, -0.0355,
          -0.0328, -0.0292],
    0.9: [0.0749, 0.0294, 0.0025, -0.0130, -0.0216, -0.0260, -0.0280,
          -0.0285, -0.0282],
}
_CUM_G = {0.8: {4: 0.0802, 20: -0.0151, 40: -0.0244},
          0.9: {4: 0.0273, 20: -0.0323, 40: -0.0510}}


@pytest.mark.parametrize('rho', [0.8, 0.9])
def test_spending_output_path_turns_negative_while_g_is_raised(ss, G_jac,
                                                               rho):
    """Issue #5. A tax-financed G rise lifts output only briefly. From quarter
    3, with G still 51-73% of its impact level, the persistently higher labour
    tax and crowded-out investment push output below steady state, and the
    20- and 40-quarter cumulative multipliers are negative. These are regression
    pins on this implementation; no published IRF gates them."""
    irf = model.shock('fiscal_spending', 0.01, rho, T=T, ss=ss, G_jac=G_jac)
    assert np.allclose(irf['Y'][:9], _Y_G[rho], atol=1.5e-3)
    assert int(np.argmax(irf['Y'] < 0)) == 3
    assert np.all(irf['shock'][:9] > 0)
    dY = irf['Y'] / 100 * ss['Y']
    for h, want in _CUM_G[rho].items():
        assert dY[:h].sum() / irf['shock'][:h].sum() == pytest.approx(
            want, abs=2e-3), h


def test_monetary_inflation_peaks_on_impact(ss, G_jac):
    """Forward-looking NKPC: for a persistent (rho 0.9) 25bp-per-quarter
    tightening, inflation's largest move is on impact (-0.336pp per quarter,
    about -1.34pp annualised), not after a lag. Model class, not a bug."""
    irf = model.shock('monetary', 0.0025, 0.9, T=T, ss=ss, G_jac=G_jac)
    assert int(np.argmax(np.abs(irf['pi']))) == 0
    assert irf['pi'][0] == pytest.approx(-0.3356, abs=3e-3)
