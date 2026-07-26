import numpy as np
import pytest

from us_hank import fiscal_2025


def test_published_calibration_units():
    calibration = fiscal_2025.PUBLISHED_CALIBRATION
    assert calibration.quarterly_real_rate == pytest.approx(0.005)
    assert calibration.debt_level == pytest.approx(4.0)
    assert calibration.impact_mpc == pytest.approx(0.20)
    assert calibration.debt_persistence == pytest.approx(0.975)


def test_deficit_tax_cut_matches_published_shock_rule():
    paths = fiscal_2025.fiscal_paths("deficit_tax_cut", T=80)
    assert paths["tax_cut"][0] == pytest.approx(0.01)
    assert paths["tax_cut"][1] == pytest.approx(0.009)
    assert paths["debt"][0] == pytest.approx(0.01)
    assert paths["debt"][1] == pytest.approx(0.975 * 0.01 + 0.009)
    assert paths["tax_revenue"][0] == pytest.approx(-0.01)


def test_deficit_spending_is_untaxed_on_impact():
    paths = fiscal_2025.fiscal_paths("deficit_spending", T=80)
    assert paths["spending"][0] == pytest.approx(0.01)
    assert paths["debt"][0] == pytest.approx(0.01)
    assert paths["tax_revenue"][0] == pytest.approx(0.0)


@pytest.mark.parametrize("kind", fiscal_2025.SUPPORTED_KINDS)
def test_government_budget_clears_every_period(kind):
    paths = fiscal_2025.fiscal_paths(kind, T=300)
    assert np.max(np.abs(paths["budget_residual"])) < 1e-14
    assert paths["debt"][-1] < 0.01 * np.max(paths["debt"])


@pytest.mark.parametrize(
    "argument,value",
    [
        ("shock_persistence", 1.0),
        ("debt_persistence", -0.1),
        ("T", 1),
        ("size", -0.01),
    ],
)
def test_invalid_paths_are_refused(argument, value):
    with pytest.raises(ValueError):
        fiscal_2025.fiscal_paths("deficit_tax_cut", **{argument: value})


def test_unvalidated_ha_response_is_hard_blocked():
    with pytest.raises(RuntimeError, match="not available"):
        fiscal_2025.shock("deficit_tax_cut", 0.01)
