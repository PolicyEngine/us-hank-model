import numpy as np
import pytest

from us_hank import distributional, model

pytestmark = pytest.mark.slow


def test_mpc_declines_in_liquid_wealth(ss):
    q = distributional.mpc_by_liquid_quartile(ss)
    assert set(q) == {'Q1', 'Q2', 'Q3', 'Q4'}
    assert q['Q1'] > q['Q4']
    assert q['Q3'] > q['Q4']
    assert all(0 <= v <= 1 for v in q.values())


def test_bottom_two_quartiles_are_the_same_constrained_households(ss):
    """Q1 and Q2 must report the SAME MPC, and that is not a bug.

    Over half the population holds exactly zero liquid assets, so the 25th and
    50th percentiles of liquid wealth both fall inside a single atom: Q1 and
    Q2 are the same households at the same (zero) liquid wealth. Any gradient
    between them would be manufactured by the arbitrary ordering of tied grid
    cells, so the quantile helper splits tied blocks proportionally and both
    bins report the atom's mean.
    """
    h = ss.internals['hh']
    constrained = float(h['D'][:, h['b_grid'] <= 0.0, :].sum())
    assert constrained > 0.5, constrained

    q = distributional.mpc_by_liquid_quartile(ss)
    assert q['Q1'] == pytest.approx(q['Q2'], rel=1e-12)


def test_quantile_bins_preserve_the_population_mean(ss):
    """Every bin holds exactly 1/n of the mass, so the unweighted mean of the
    bin means is the population mean to machine precision."""
    q = distributional.mpc_by_liquid_quartile(ss)
    assert sum(q.values()) / 4 == pytest.approx(
        distributional.aggregate_mpc(ss), rel=1e-12)


def test_mpcs_are_positive_and_uncapped(ss):
    """No household should show a negative MPC, and the cap at 1 should not
    be doing any work. Both would signal a policy-function or grid problem,
    and the cap would otherwise hide it."""
    mpc, D, _ = distributional.mpc_liquid(ss)
    assert mpc.min() > 0.0
    assert float(D[mpc >= 1.0].sum()) == 0.0


def test_aggregate_mpc_production_grids(ss):
    m = distributional.aggregate_mpc(ss)
    assert 0.02 < m < 0.6


def test_mpc_is_windfall_size_dependent(ss):
    """MPCs fall as the windfall grows -- the consumption policy is concave.

    This is why no unlabelled "the MPC" is reported: the number is only
    meaningful next to the windfall it was measured at.
    """
    small = distributional.aggregate_mpc(ss, windfall_frac=0.05)
    mid = distributional.aggregate_mpc(ss, windfall_frac=0.25)
    large = distributional.aggregate_mpc(ss, windfall_frac=1.0)
    assert small > mid > large


def test_grid_step_mpc_is_not_grid_converged(ss):
    """Guard rail on a known limitation, not a target to hit.

    The legacy one-grid-step MPC differences the policy over b_grid[1] -
    b_grid[0], an interval set by nB rather than by economics. It rises from
    0.076 at nB=50 to 0.100 at nB=150 with no plateau, because nB=50 does not
    resolve the kink at the borrowing constraint. It therefore sits well above
    the fixed-windfall MPC, and must not be quoted as a structural MPC.
    """
    grid_step = distributional.aggregate_mpc(ss, windfall_frac=None)
    windfall = distributional.aggregate_mpc(ss, windfall_frac=0.25)
    assert grid_step > windfall
    h = ss.internals['hh']
    first_step = float(h['b_grid'][1] - h['b_grid'][0])
    income = distributional.mean_quarterly_labour_income(ss)
    # the implied windfall is a grid artifact: ~6.7% of quarterly income
    assert 0.02 < first_step / income < 0.15


def test_hand_to_mouth_share(ss):
    s = distributional.hand_to_mouth_share(ss)
    assert 0.2 < s < 0.8  # large HtM share is the point of the model


def test_hand_to_mouth_is_an_atom_not_a_threshold(ss):
    """The reported share is driven by households exactly at the constraint,
    not by where the cutoff is put: it is flat below the first positive grid
    point and only then starts to rise."""
    b = distributional.hand_to_mouth_breakdown(ss)
    assert b['constrained_share'] == pytest.approx(0.5116, abs=5e-3)
    assert b['hand_to_mouth_share'] == pytest.approx(0.5372, abs=5e-3)
    # insensitive to the cutoff below the first positive grid point
    assert (distributional.hand_to_mouth_share(ss, 0.001)
            == pytest.approx(b['constrained_share'], rel=1e-12))
    # and monotone in it
    shares = [distributional.hand_to_mouth_share(ss, t)
              for t in (0.001, 0.05, 0.2, 1.0)]
    assert shares == sorted(shares)


def test_hand_to_mouth_is_overwhelmingly_wealthy_htm(ss):
    """The model's HtM households almost all hold illiquid wealth.

    This is the model's own split, NOT a replication of a published number:
    ABRS (2021) reports no hand-to-mouth share anywhere. It is also far from
    the US evidence -- Kaplan, Violante & Weidner (2014) put total HtM near
    one third, two thirds of them wealthy HtM, against ~54% and ~96% here.
    """
    b = distributional.hand_to_mouth_breakdown(ss)
    total = b['hand_to_mouth_share']
    assert b['poor_hand_to_mouth_share'] + b['wealthy_hand_to_mouth_share'] \
        == pytest.approx(total)
    assert b['wealthy_hand_to_mouth_share'] / total > 0.9
    # well above the ~1/3 US empirical estimate; assert the gap knowingly
    assert total > 0.45


def test_consumption_response_by_wealth_quantile(ss, G_jac):
    irf = model.shock('monetary', -0.0025, 0.61, T=300, ss=ss, G_jac=G_jac)
    q = distributional.consumption_response_by_wealth_quantile(ss, irf)
    # weighted responses average to the aggregate; poor respond more than rich
    assert q['Q1'] > q['Q4'] > 0
    h = ss.internals['hh']
    assert h['D'].sum() == pytest.approx(1.0)
    # bins hold exactly 1/4 of the mass each, so this is exact, not approximate
    assert sum(q.values()) / 4 == pytest.approx(irf['C'][0], rel=1e-12)


def test_consumption_allocation_is_a_rescaling_not_a_solved_response(ss, G_jac):
    """Make the approximation's limits executable.

    The allocation scales one aggregate by relative MPCs, so its SHAPE is
    identical across shock kinds and its sign is common to all quantiles. It
    cannot show a shock that helps one group and hurts another, and it does
    not use any household-specific exposure to the shock.
    """
    mon = model.shock('monetary', -0.0025, 0.61, T=300, ss=ss, G_jac=G_jac)
    fis = model.shock('fiscal_spending', 0.01, 0.8, T=300, ss=ss, G_jac=G_jac)
    qm = distributional.consumption_response_by_wealth_quantile(ss, mon)
    qf = distributional.consumption_response_by_wealth_quantile(ss, fis)

    # opposite-signed aggregates, yet the shapes are proportional
    assert mon['C'][0] > 0 > fis['C'][0]
    ratio = [qf[k] / qm[k] for k in ('Q1', 'Q2', 'Q3', 'Q4')]
    assert np.allclose(ratio, fis['C'][0] / mon['C'][0], rtol=1e-12)
    # sign is common across quantiles -- no distributional disagreement
    assert all(v < 0 for v in qf.values())


def test_summary_keys(ss):
    s = distributional.summary(ss)
    assert set(s) == {
        'aggregate_quarterly_mpc_liquid', 'mpc_by_liquid_quartile',
        'hand_to_mouth_share', 'mpc_windfall_frac_of_quarterly_income',
        'mpc_windfall_model_units', 'mean_quarterly_labour_income',
        'hand_to_mouth_breakdown',
    }
    # the MPC never travels without the windfall it was measured at
    assert s['mpc_windfall_frac_of_quarterly_income'] == 0.25
    assert s['mpc_windfall_model_units'] == pytest.approx(
        0.25 * s['mean_quarterly_labour_income'])
