"""The one impulse-response magnitude the paper states in its text.

ABRS (2021), NBER working paper w26123, pp. 40-41 (Figure 8a; Figure 7a in
the Econometrica version): the two-asset model's impact consumption response
to a -5pp shock to the Taylor rule with quarterly persistence 0.6 is 1.46%
on the linear solution and 1.36% on the nonlinear one, and 5 x the nonlinear
response to a -1pp shock is "almost identical" to the linear one.

The paper does not state the shock's units. Annualised percentage points
(rstar -0.0125 per quarter) is inferred: the quarterly reading (-0.05) gives
5.94%, four times the published number. The 3% relative band covers the
paper's two-decimal rounding (about 0.35%) and the measured gap, and still
rejects the quarterly reading by a wide margin.

Measured locally on 16 Nov 2026 (production grids, T=300, sequence-jacobian
1.0.0): linear 1.485 (+1.7%), nonlinear 1.388 (+2.1%), ratio 0.935 against
0.932, 5 x nonlinear(-1pp) 1.467. Part of the linear gap is the wage-markup
convention (see docs/VALIDATION.md, "phi and the wage-markup convention"):
with kappaw x muw**2, the paper-era form, the linear response is 1.468.

Only consumption on impact to this one shock is compared. Output,
investment, inflation and fiscal responses have no published magnitude.
"""

import numpy as np
import pytest

from us_hank import model

pytestmark = pytest.mark.slow

T = 300
RHO = 0.6
SHOCK = -0.05 / 4              # -5pp annualised, as a quarterly rstar level
PAPER_LINEAR = 1.46
PAPER_NONLINEAR = 1.36


def _c0_linear(ss, G, size):
    dx = size * RHO ** np.arange(T)
    return 100 * (G['C']['rstar'] @ dx)[0] / ss['C']


def _c0_nonlinear(ss, size):
    _, dag = model.build_models()
    nl = dag.solve_impulse_nonlinear(
        ss, model.UNKNOWNS, model.TARGETS,
        {'rstar': size * RHO ** np.arange(T)})
    for k in model.TARGETS:
        assert np.max(np.abs(nl[k])) < 1e-7, k
    return 100 * nl['C'][0] / ss['C']


@pytest.fixture(scope="module")
def responses(ss, G_jac):
    return {
        'linear': _c0_linear(ss, G_jac, SHOCK),
        'nonlinear': _c0_nonlinear(ss, SHOCK),
        'nonlinear_1pp_x5': 5 * _c0_nonlinear(ss, SHOCK / 5),
    }


def test_linear_impact_consumption_matches_the_paper(responses):
    assert responses['linear'] == pytest.approx(PAPER_LINEAR, rel=0.03)


def test_nonlinear_impact_consumption_matches_the_paper(responses):
    assert responses['nonlinear'] == pytest.approx(PAPER_NONLINEAR, rel=0.03)


def test_nonlinear_to_linear_ratio_matches_the_paper(responses):
    ratio = responses['nonlinear'] / responses['linear']
    assert ratio == pytest.approx(PAPER_NONLINEAR / PAPER_LINEAR, abs=0.01)


def test_small_shock_is_close_to_linear(responses):
    """The paper: 5 x nonlinear(-1pp) is almost identical to the linear
    response. Compared here with this model's own linear response."""
    assert responses['nonlinear_1pp_x5'] == pytest.approx(
        responses['linear'], rel=0.02)


def test_paper_era_wage_convention_narrows_the_linear_gap(ss):
    """Same steady state with kappaw x muw**2, the working paper's wage
    Phillips curve. Measured 1.468."""
    ss2 = ss.copy()
    ss2['kappaw'] = ss['kappaw'] * ss['muw'] ** 2
    _, dag = model.build_models()
    G2 = dag.solve_jacobian(ss2, model.UNKNOWNS, model.TARGETS,
                            model.EXOGENOUS, T=T)
    assert _c0_linear(ss2, G2, SHOCK) == pytest.approx(PAPER_LINEAR,
                                                       rel=0.015)
