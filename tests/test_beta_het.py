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
