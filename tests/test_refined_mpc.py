"""MPCs on a refined liquid grid, and where a windfall goes.

The production grid's first liquid step (b = 0.0286, 6.7% of quarterly
income) is larger than a $500-1,500 rebate per household, so on that grid
every rebate-sized MPC is the slope of one grid segment (0.0751). These tests
re-solve the household block alone, at the GE steady-state prices, on liquid
grids that get denser near the constraint, and check that the answer
converges. The production steady state and its published numbers are not
touched.
"""
import pytest

from us_hank import distributional

pytestmark = pytest.mark.slow

# US GDP ~$31T a year (2026), 133M households.
_Q_GDP_PER_HOUSEHOLD = 31e12 / 133e6 / 4


def _frac(ss, dollars):
    return (dollars / _Q_GDP_PER_HOUSEHOLD
            / distributional.mean_quarterly_labour_income(ss))


def test_refined_solver_reproduces_the_production_grid(ss):
    """With the production grid (nB=50, pivot 0.25) the re-solve must give the
    production MPC exactly, so any difference elsewhere is the grid alone."""
    for f in (_frac(ss, 1000), 0.25):
        assert distributional.refined_aggregate_mpc(
            ss, f, nB=50, b_pivot=0.25
        ) == pytest.approx(distributional.aggregate_mpc(ss, f), rel=1e-6)


def test_rebate_mpc_converges_in_liquid_grid_size(ss):
    """$500 per household: 0.0751 on the production grid, 0.104 converged.

    Doubling nB from 200 to 400 at pivot 0.01 moves it by less than 1e-4, and
    the default refined grid agrees with pivot 0.05 at nB=400 too, so the
    answer is a property of the model, not of the grid.
    """
    f = _frac(ss, 500)
    seq = [distributional.refined_aggregate_mpc(ss, f, nB=n, b_pivot=0.01)
           for n in (50, 100, 200, 400)]
    steps = [abs(b - a) for a, b in zip(seq, seq[1:])]
    assert steps == sorted(steps, reverse=True)  # shrinking corrections
    assert steps[-1] < 2e-4
    other = distributional.refined_aggregate_mpc(ss, f, nB=400, b_pivot=0.05)
    assert other == pytest.approx(seq[-1], abs=5e-4)
    assert seq[2] == pytest.approx(0.1040, abs=1e-3)
    # and the production-grid figure is materially lower: a grid artifact
    assert distributional.aggregate_mpc(ss, f) == pytest.approx(0.0751, abs=1e-3)
    assert seq[2] - distributional.aggregate_mpc(ss, f) > 0.025


def test_refined_rebate_mpcs_are_pinned_and_still_below_the_literature(ss):
    """Converged MPCs: $500/1,000/1,500 per household 0.104/0.091/0.083;
    the 25% default 0.059 (production grid 0.058). All sit below the
    0.15-0.25 quarterly range of Parker et al. (2013) and the 2001/2008
    rebate studies, so the grid explains part of the gap, not most of it."""
    got = {d: distributional.refined_aggregate_mpc(ss, _frac(ss, d))
           for d in (500, 1000, 1500)}
    assert got[500] == pytest.approx(0.1040, abs=1e-3)
    assert got[1000] == pytest.approx(0.0913, abs=1e-3)
    assert got[1500] == pytest.approx(0.0834, abs=1e-3)
    default = distributional.refined_aggregate_mpc(ss, 0.25)
    assert default == pytest.approx(0.0593, abs=1e-3)
    assert abs(default - distributional.aggregate_mpc(ss)) < 2e-3
    assert max(got.values()) < 0.15


def test_refined_grid_keeps_prices_and_reports_its_drift(ss):
    """The refined block is partial equilibrium at the GE prices: same tax,
    wage and returns. Its stationary liquid wealth drifts a little below the
    1.04 target, which is why it is for MPC statistics only."""
    r = distributional.refined_household_ss(ss)
    for k in ('tax', 'w', 'N', 'rb', 'ra', 'beta', 'chi1'):
        assert float(r[k]) == pytest.approx(float(ss[k]), rel=1e-12)
    assert float(r['B']) == pytest.approx(1.019, abs=5e-3)


def test_constrained_households_move_most_of_a_windfall_into_illiquid_assets(ss):
    """The diagnosis of the low MPC.

    Half the population ends the period at the liquid borrowing limit, which
    would ordinarily mean a high MPC. Here the illiquid account's adjustment
    cost is smooth and convex (zero marginal cost at zero adjustment, no
    linear term), so a constrained household with illiquid wealth simply
    withdraws less from it: 59% of a $1,000 rebate ends up in the illiquid
    account and 11% is consumed. Only the 1.8% of households that are
    constrained with no illiquid wealth behave close to hand-to-mouth.
    """
    a = distributional.windfall_allocation(ss, _frac(ss, 1000))
    rb = float(ss['rb'])
    for g in ('all', 'constrained', 'unconstrained'):
        tot = sum(a[g][k] for k in ('mpc', 'mps_liquid', 'mps_illiquid',
                                    'adj_cost'))
        assert tot == pytest.approx(1 + rb, abs=1e-6)  # budget constraint
    assert a['constrained']['mass'] == pytest.approx(0.497, abs=5e-3)
    cw = a['constrained_with_illiquid']
    assert cw['mass'] == pytest.approx(0.479, abs=5e-3)
    assert cw['mps_illiquid'] == pytest.approx(0.593, abs=1e-2)
    assert cw['mpc'] == pytest.approx(0.111, abs=5e-3)
    assert a['constrained_without_illiquid']['mass'] == pytest.approx(
        0.0182, abs=2e-3)
    assert a['unconstrained']['mpc'] == pytest.approx(0.035, abs=3e-3)
    assert cw['mps_illiquid'] > 5 * cw['mpc']
