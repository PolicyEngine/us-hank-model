"""The opt-in beta-heterogeneity variant (issue #8).

Fast tests run on the toolkit's demo grids (nB=10, nA=16, nK=4) and check
the construction: with one type the variant IS the default model, bit for
bit, and building the variant leaves the default's calibration and caches
alone. Slow tests solve the calibrated variant at production grids.
"""
import numpy as np
import pytest

from us_hank import beta_het, distributional, model

DEMO = dict(nB=10, nA=16, nK=4)


def _default_demo_ss():
    """The DEFAULT model's steady state on demo grids, solved without touching
    ``model.PRODUCTION_CALIBRATION`` or ``model``'s cache."""
    ss_m, m = model.build_models()
    cal = dict(model.PRODUCTION_CALIBRATION, **DEMO)
    cali = ss_m.solve_steady_state(
        cal, unknowns={'beta': 0.976, 'chi1': 6.5},
        targets={'asset_mkt': 0., 'B': 'Bh'}, solver='broyden_custom')
    return m.steady_state(cali)


def test_one_type_variant_is_the_default_model_bit_for_bit():
    ss0 = _default_demo_ss()
    ss1 = beta_het.solve_steady_state(0.0, beta_offsets=(0.0,),
                                      beta_masses=(1.0,), use_cache=False,
                                      **DEMO)
    h0, h1 = ss0.internals['hh'], ss1.internals['hh']
    for k in ('D', 'c', 'a', 'b', 'Va', 'Vb'):
        assert np.array_equal(h0[k], h1[k]), k
    for k in ('beta', 'chi1', 'A', 'B', 'C', 'tax', 'goods_mkt'):
        assert ss0[k] == ss1[k], k


def test_building_the_variant_leaves_the_default_alone():
    before = dict(model.PRODUCTION_CALIBRATION)
    cache_keys = set(model._CACHE)
    beta_het.solve_steady_state(0.05, use_cache=False, **DEMO)
    assert model.PRODUCTION_CALIBRATION == before
    assert set(model._CACHE) == cache_keys
    # and the variant's own calibration starts from the published one
    for k, v in model.PRODUCTION_CALIBRATION.items():
        assert beta_het.BETA_HET_CALIBRATION[k] == v, k


def test_type_masses_are_the_stationary_distribution():
    grids = beta_het.make_grids_beta(
        50, 4000, 1, 10, 16, 4, 3, 0.966, 0.92, (-1.0, 0.0), (1 / 3, 2 / 3),
        0.976, 0.1, 1 / 200)
    Pi, beta_z = grids[4], grids[5]
    assert np.allclose(Pi.sum(axis=1), 1.0)
    from sequence_jacobian.utilities.discretize import stationary
    pi = stationary(Pi)
    assert pi[:3].sum() == pytest.approx(1 / 3, abs=1e-8)
    assert np.allclose(beta_z, [0.876] * 3 + [0.976] * 3)


def test_variant_demo_steady_state_clears_markets_and_raises_the_mpc():
    """On demo grids: same targets as the default, markets clear, the type
    masses carry through to the distribution, and an impatient type raises
    the MPC above the one-type model's."""
    ss = beta_het.solve_steady_state(0.05, use_cache=False, **DEMO)
    assert abs(ss['asset_mkt']) < 1e-10
    assert abs(ss['B'] - 1.04) < 1e-9
    assert abs(ss['A'] + ss['B'] - 14.0) < 1e-8
    assert abs(ss['goods_mkt']) < 1e-6
    D = ss.internals['hh']['D']
    assert D.sum() == pytest.approx(1.0, abs=1e-12) and D.min() >= 0
    by_type = beta_het.mpc_by_type(ss, 0.25)
    assert by_type['type0']['mass'] == pytest.approx(1 / 3, abs=1e-8)
    assert by_type['type0']['mpc'] > 2 * by_type['type1']['mpc']
    one = beta_het.solve_steady_state(0.0, beta_offsets=(0.0,),
                                      beta_masses=(1.0,), use_cache=False,
                                      **DEMO)
    assert (distributional.aggregate_mpc(ss)
            > distributional.aggregate_mpc(one) + 0.01)


# --- production grids (slow) --------------------------------------------------
# Pinned from CI run 42145287339 (r5 diag): impatient type one third of
# households, beta_L = 0.852 vs patient beta = 0.983.


@pytest.fixture(scope="module")
def ss_v():
    return beta_het.solve_steady_state()


@pytest.fixture(scope="module")
def G_v(ss_v):
    return beta_het.solve_jacobian(ss_v, T=300)


@pytest.mark.slow
def test_variant_steady_state_clears_markets_at_the_default_targets(ss_v):
    assert abs(ss_v['asset_mkt']) < 1e-11
    assert abs(ss_v['fisher']) < 1e-11
    assert abs(ss_v['wnkpc']) < 1e-11
    assert abs(ss_v['goods_mkt']) < 1e-6  # untargeted (Walras), as default
    assert abs(ss_v['B'] - 1.04) < 1e-9
    assert abs(ss_v['A'] + ss_v['B'] - 14.0) < 1e-8
    D = ss_v.internals['hh']['D']
    assert D.sum() == pytest.approx(1.0, abs=1e-10) and D.min() >= 0
    for k in ('Y', 'K', 'r', 'G', 'Bg'):
        assert ss_v[k] == pytest.approx(model.PRODUCTION_CALIBRATION[k])
    assert ss_v['beta'] == pytest.approx(0.98283, abs=1e-4)
    assert ss_v['chi1'] == pytest.approx(8.065, abs=2e-3)


@pytest.mark.slow
def test_variant_hits_the_mpc_target_and_reports_by_rebate_size(ss_v):
    ref = beta_het.refined_household_ss(ss_v)
    mpc = {d: distributional.aggregate_mpc(ref, beta_het.rebate_frac(ss_v, d))
           for d in (500, 1000, 1500)}
    assert mpc[1000] == pytest.approx(beta_het.MPC_TARGET, abs=1e-3)
    assert mpc[500] == pytest.approx(0.227, abs=2e-3)
    assert mpc[1500] == pytest.approx(0.186, abs=2e-3)
    assert 0.15 <= mpc[1500] < mpc[1000] < mpc[500] <= 0.25
    htm = distributional.hand_to_mouth_breakdown(ss_v)
    assert htm['poor_hand_to_mouth_share'] == pytest.approx(0.203, abs=2e-3)
    assert htm['hand_to_mouth_share'] == pytest.approx(0.572, abs=2e-3)
    by_type = beta_het.mpc_by_type(ref, beta_het.rebate_frac(ss_v, 1000))
    assert by_type['type0']['mpc'] == pytest.approx(0.432, abs=3e-3)
    assert by_type['type1']['mpc'] == pytest.approx(0.084, abs=2e-3)


@pytest.mark.slow
def test_variant_spending_multiplier_and_the_issue_5_sign(ss_v, G_v):
    """Higher MPC: impact multiplier 0.175 -> 0.376 (rho 0.8). Output stays
    positive for 23 quarters (default: negative from q3), and at rho 0.9 it
    never turns negative within 40 quarters."""
    r = model.shock('fiscal_spending', 0.01, 0.8, ss=ss_v, G_jac=G_v)
    g = 100 * r['shock']
    assert r['Y'][0] / g[0] == pytest.approx(0.376, abs=2e-3)
    assert r['Y'][:20].sum() / g[:20].sum() == pytest.approx(0.309, abs=2e-3)
    assert np.flatnonzero(r['Y'][:40] < 0)[0] == 23
    assert r['C'][0] < 0  # still crowds out consumption on impact
    r9 = model.shock('fiscal_spending', 0.01, 0.9, ss=ss_v, G_jac=G_v)
    assert (r9['Y'][:40] > 0).all()
    assert r9['Y'][0] / (100 * r9['shock'][0]) == pytest.approx(0.319, abs=2e-3)


@pytest.mark.slow
def test_variant_monetary_easing_is_expansionary_but_smaller(ss_v, G_v, ss, G_jac):
    r = model.shock('monetary', -0.0025, 0.61, ss=ss_v, G_jac=G_v)
    d = model.shock('monetary', -0.0025, 0.61, ss=ss, G_jac=G_jac)
    assert r['Y'][0] == pytest.approx(0.244, abs=2e-3)
    assert d['Y'][0] == pytest.approx(0.493, abs=2e-3)
    assert r['C'][0] > d['C'][0] and r['I'][0] < d['I'][0]
    assert r['pi'][0] > 0


# --- median-wealth targets (fast, demo grids) ---------------------------------

def test_uniform_offsets_are_equal_mass_bin_midpoints():
    assert beta_het.uniform_offsets(1) == (0.0,)
    assert np.allclose(beta_het.uniform_offsets(3), (-2 / 3, 0.0, 2 / 3))
    assert np.allclose(beta_het.uniform_offsets(5), (-0.8, -0.4, 0.0, 0.4, 0.8))


def test_median_targets_are_the_kv2014_scf_ratios():
    t = beta_het.MEDIAN_TARGETS
    assert t['median_liquid_to_income'] == pytest.approx(4 * 2629 / 52745)
    assert t['median_illiquid_to_income'] == pytest.approx(4 * 54600 / 52745)
    assert t['poor_htm'] == pytest.approx(0.12)
    assert 0.15 <= t['mpc_500'] <= 0.25


def test_median_mode_clears_the_asset_market_with_beta_alone():
    """chi1 stays where it is put, beta clears the asset market, and Bh is
    reset to what households hold (so it is no longer 1.04)."""
    ss = beta_het.solve_steady_state(liquid_target='median', use_cache=False,
                                     chi1=9.0, **DEMO)
    assert ss['chi1'] == 9.0
    assert abs(ss['asset_mkt']) < 1e-8
    assert abs(ss['A'] + ss['B'] - 14.0) < 1e-8
    assert ss['Bh'] == pytest.approx(ss['B'], abs=1e-10)
    assert abs(ss['B'] - 1.04) > 0.1
    assert abs(ss['goods_mkt']) < 1e-6
    assert abs(ss['fisher']) < 1e-10 and abs(ss['wnkpc']) < 1e-10
    top = max(ss.internals['hh']['beta_z'])
    assert top < beta_het.beta_cap(ss.toplevel)


def test_wealth_moments_are_consistent_with_the_aggregates():
    ss = _default_demo_ss()
    m = beta_het.wealth_moments(ss)
    assert m['income'] == pytest.approx(ss['w'] * ss['N'])
    assert m['mean_liquid_to_income'] * m['income'] == pytest.approx(ss['B'], rel=1e-8)
    assert m['mean_illiquid_to_income'] * m['income'] == pytest.approx(ss['A'], rel=1e-8)
    assert 0 <= m['poor_htm'] <= m['htm'] <= 1
    assert m['median_liquid_to_income'] <= m['mean_liquid_to_income']
    htm = distributional.hand_to_mouth_breakdown(ss, beta_het.HTM_CUTOFF)
    assert m['poor_htm'] == pytest.approx(htm['poor_hand_to_mouth_share'])
    assert m['htm'] == pytest.approx(htm['hand_to_mouth_share'])


def test_weighted_median_interpolates_inside_the_crossing_cell():
    g = np.array([0.0, 1.0, 2.0])
    assert beta_het._weighted_median(g, np.array([0.6, 0.2, 0.2])) == 0.0
    assert beta_het._weighted_median(g, np.array([0.25, 0.5, 0.25])) == pytest.approx(0.5)
