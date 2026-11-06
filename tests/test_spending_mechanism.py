"""Why tax-financed G turns output negative from quarter 3 (issue #5).

Two checks rule out a bug, and two counterfactuals locate the mechanism:

* the household budget constraint holds along the G IRF, so no resources
  leak in the household block (the government budget is already gated in
  test_shocks.py);
* holding the labour tax OUT of the union's wage Phillips curve (households
  still pay it) keeps output positive every quarter: the negative output
  comes from the tax distorting wage setting, which pushes up wage and price
  inflation, which the Taylor rule meets with a higher real rate;
* a weaker Taylor rule (phi = 1.1) keeps output positive too, and a stronger
  one (phi = 3) makes it negative on impact.

So the q3 sign change is a property of a balanced-budget spending rise
financed by a distortionary labour tax under an active Taylor rule.
"""
import numpy as np
import pytest
from sequence_jacobian import combine, create_model, simple
from sequence_jacobian.examples import two_asset as ta

from us_hank import model

pytestmark = pytest.mark.slow

T = 300
DG = 0.01 * 0.8 ** np.arange(T)


@simple
def union_tax_neutral(piw, N, tax_bar, w, UCE, kappaw, muw, vphi, frisch,
                      beta):
    """The union's wage Phillips curve with the labour tax held at tax_bar."""
    wnkpc = kappaw * (vphi * N ** (1 + 1 / frisch)
                      - (1 - tax_bar) * w * N * UCE / muw) + beta * \
        (1 + piw(+1)).apply(np.log) - (1 + piw).apply(np.log)
    return wnkpc


def _variant_jacobian(ss, union_block=None, **overrides):
    household = ta.hh.add_hetinputs([ta.income, ta.make_grids])
    production = combine([ta.labor, ta.investment]).solved(
        unknowns={'Q': 1., 'K': 10.}, targets=['inv', 'val'],
        solver='broyden_custom')
    blocks = [household, ta.pricing_solved, ta.arbitrage_solved, production,
              ta.dividend, ta.taylor, ta.fiscal, ta.share_value, ta.finance,
              ta.wage, union_block or ta.union, ta.mkt_clearing]
    m = create_model(blocks, name='variant')
    s = ss.copy()
    for k, v in overrides.items():
        s[k] = v
    return m.solve_jacobian(s, model.UNKNOWNS, model.TARGETS, model.EXOGENOUS,
                            T=T)


def _y(G_jac):
    return 100 * (G_jac['Y']['G'] @ DG)


def test_household_budget_holds_along_the_spending_irf(ss, G_jac):
    """c + a' + b' + Psi = (1 - tax) w N + (1 + ra) a + (1 + rb) b, to first
    order, every quarter."""
    v = {k: np.asarray(G_jac[k]['G'] @ DG)
         for k in ('C', 'A', 'B', 'CHI', 'tax', 'w', 'N', 'ra', 'rb')}
    s = {k: float(ss[k]) for k in ('A', 'B', 'tax', 'w', 'N', 'ra', 'rb')}
    lag = lambda x: np.r_[0.0, x[:-1]]
    uses = v['C'] + v['A'] + v['B'] + v['CHI']
    labour = (-v['tax'] * s['w'] * s['N']
              + (1 - s['tax']) * (v['w'] * s['N'] + s['w'] * v['N']))
    sources = (labour + v['ra'] * s['A'] + (1 + s['ra']) * lag(v['A'])
               + v['rb'] * s['B'] + (1 + s['rb']) * lag(v['B']))
    assert np.max(np.abs(uses - sources)) < 1e-8


def test_output_stays_positive_when_the_tax_does_not_distort_wage_setting(ss,
                                                                          G_jac):
    base = _y(G_jac)
    assert base[3] < 0  # the published path: negative from q3
    neutral = _y(_variant_jacobian(ss, union_tax_neutral,
                                   tax_bar=float(ss['tax'])))
    assert np.all(neutral[:40] > 0)
    assert neutral[0] == pytest.approx(0.678, abs=5e-3)
    cum20 = neutral[:20].sum() / 100 / DG[:20].sum()
    assert cum20 == pytest.approx(0.728, abs=5e-3)


@pytest.mark.parametrize('phi, sign_q0, positive_through_q8',
                         [(1.1, 1, True), (3.0, -1, False)])
def test_taylor_rule_strength_sets_the_sign(ss, phi, sign_q0,
                                            positive_through_q8):
    y = _y(_variant_jacobian(ss, phi=phi))
    assert np.sign(y[0]) == sign_q0
    assert bool(np.all(y[:8] > 0)) is positive_through_q8
