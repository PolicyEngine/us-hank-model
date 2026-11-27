# Changelog

Versions are the `version` in `pyproject.toml`. `v0.2.0` is tagged at
`58353e5` (the #19 merge); 0.1.0 was not tagged. PolicyEngine Macro pins a
commit, not a tag.

## Unreleased

After the `v0.2.0` tag; no change to model behaviour or results, and
`pyproject.toml` still reads 0.2.0.

Docs, tests, comments and CI
- Pointers to README sections that moved to `docs/` corrected; the test
  named for each claim in `docs/VALIDATION.md` corrected.
- Pins added for figures the docs quote that no test gated: the real-rate
  path after the reference monetary shock, a tighter upper bound on the
  goods-market residual, and the remaining cells of the two tables in
  `docs/BETA_HET.md`.
- CI: a job timeout, read-only token permissions, and superseded
  pull-request runs are cancelled.

Added
- `model.annualised_bp`, `model.monetary_size`, `model.size_pct_of_ss` and
  `model.size_from_pct_of_ss` convert between `size` and annualised basis
  points or a percentage of the steady state. A comment in `model.py`
  already pointed to `size_pct_of_ss`, which did not exist.
- `one_asset.SHOCK_UNITS`, and a `units` string in every one-asset impulse
  response. In that variant `Z` is 1, so `0.01` on `productivity` is a 1% TFP
  shock, not 2.14%.
- `model.shock()` returns `w`, `N`, `i` and `labour_income_post_tax`, the
  first-order % change in `(1 − tax)·w·N`. After a 1%-of-GDP spending rise
  (ρ 0.8) the pre-tax wage bill rises 0.31% on impact and the post-tax one
  falls 0.48%.
- `tests/test_units.py`: the conversions, that `size` is the intercept of the
  quarterly rule (`i − φ·π = rstar` along the response), and that `pi` and
  `r` are quarterly rates.
- A warning that a persistent monetary easing raises the policy rate. With
  `size = -0.0025` the impact response of `i` is −0.13pp a quarter at
  persistence 0.61, zero at about 0.752 and +0.25pp a quarter at 0.9
  (PolicyEngine Macro's default). The ex-ante real rate still falls and
  output rises. The note is in the `model.shock` docstring, the monetary
  `units` string (which every monetary response returns) and
  `docs/SHOCKS.md`; five tests in `tests/test_units.py` pin it. No default
  and no response changes.
- `tests/test_beta_het.py`: three fast tests run `calibrate_beta_spread` on
  the demo grids with a short sweep (bracket and refine, return on a sweep
  point, no crossing). The search was untested because it takes minutes at
  production grids. They add about 13 seconds to the fast suite.

## 0.2.0

Version set on 17 November 2026 (#19). Everything merged after the initial
package is listed here. The default model's calibration, grids and solver pin
are the same as in 0.1.0. What is reported changed: the MPC is measured out
of an explicit windfall (#3), and every impulse response carries the labour
tax path and the shock's units (#3).

Added
- `us_hank.fiscal_2025`: clean-room debt and tax paths from Auclert, Rognlie
  & Straub (2025); its heterogeneous-agent `shock()` is blocked (#2).
- MPCs out of a windfall of stated size (default 25% of mean quarterly
  post-tax labour income) in place of the one-grid-step MPC, the
  hand-to-mouth breakdown, the `tax` path, `SHOCK_UNITS` and
  `shock_pct_of_ss` in every impulse response, and the
  published-parameter test against Table B.III (#3).
- Scenario sweep and nonlinear-transition tests; pinned spending-shock and
  MPC figures (#4).
- `distributional.refined_aggregate_mpc` and `windfall_allocation`:
  rebate-sized MPCs on a refined liquid grid, and the tests of the
  spending-shock mechanism (#7).
- `us_hank.beta_het`: opt-in discount-factor heterogeneity, with an
  aggregate-target and a median-wealth calibration; Jacobian sums made
  independent of `PYTHONHASHSEED` (#9, #10).
- `high_mpc_share` in `distributional.summary()`, beside the hand-to-mouth
  share (#17).
- A test of the one impulse-response magnitude the paper states, impact
  consumption after the Fig. 7a monetary shock, to 3% (#16).

Changed
- The one-asset variant is solved on the toolkit notebook's grid (7 income
  states, 500 asset points) instead of the 2 x 10 demo grid; its impact
  responses changed (monetary output 0.246 to 0.191) (#13).
- Python 3.11 or later is required; CI tests 3.11, 3.12 and 3.15 (#14).
- The monetary shock is described as a shock to the Taylor-rule intercept:
  for `size=-0.0025`, persistence 0.61, the policy rate falls 0.13pp on
  impact, not 0.25pp. The label and units string changed; the responses did
  not (#18).
- The φ gap to Table B.III is explained as a wage-markup convention and
  asserted (#15).
- README shortened; the long text moved to `docs/` (#11, #19).

## 0.1.0

26 July 2026. Initial package: the two-asset HANK of Auclert, Bardóczy,
Rognlie & Straub (2021) at production grids (nB=50, nA=70, nK=50) on
`sequence-jacobian` 1.0.0, linear impulse responses to monetary, government
spending and TFP shocks, the one-asset variant, and steady-state
distributional statistics.
