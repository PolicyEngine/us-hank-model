import numpy as np
import pytest

from us_hank import distributional, model

pytestmark = pytest.mark.slow


def test_mpc_declines_in_liquid_wealth(ss):
    q = distributional.mpc_by_liquid_quartile(ss)
    assert set(q) == {'Q1', 'Q2', 'Q3', 'Q4'}
    assert q['Q1'] > q['Q4']
    assert all(0 <= v <= 1 for v in q.values())


def test_aggregate_mpc_production_grids(ss):
    # full grids resolve the constraint kink; quarterly MPC should be sizable
    m = distributional.aggregate_mpc(ss)
    assert 0.05 < m < 0.6


def test_hand_to_mouth_share(ss):
    s = distributional.hand_to_mouth_share(ss)
    assert 0.2 < s < 0.8  # large HtM share is the point of the model


def test_consumption_response_by_wealth_quantile(ss, G_jac):
    irf = model.shock('monetary', -0.0025, 0.61, T=300, ss=ss, G_jac=G_jac)
    q = distributional.consumption_response_by_wealth_quantile(ss, irf)
    # weighted responses average to the aggregate; poor respond more than rich
    assert q['Q1'] > q['Q4'] > 0
    h = ss.internals['hh']
    D = h['D']
    assert np.isclose(sum(q.values()) / 4, irf['C'][0], rtol=0.5)
    assert D.sum() == pytest.approx(1.0)


def test_summary_keys(ss):
    s = distributional.summary(ss)
    assert set(s) == {'aggregate_quarterly_mpc_liquid',
                      'mpc_by_liquid_quartile', 'hand_to_mouth_share'}
