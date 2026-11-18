# Changelog

Versions are the `version` in `pyproject.toml`. No version has been tagged
on GitHub yet; PolicyEngine Macro pins a commit.

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

Checked after the version was set (docs, tests and CI only)
- Pointers to README sections that moved to `docs/` corrected; the test
  named for each claim in `docs/VALIDATION.md` corrected.
- Pins added for figures the docs quote that no test gated: the real-rate
  path after the reference monetary shock, the order of magnitude of the
  goods-market residual, and the remaining cells of the two tables in
  `docs/BETA_HET.md`.
- CI: a job timeout, read-only token permissions, and superseded
  pull-request runs are cancelled.

## 0.1.0

26 July 2026. Initial package: the two-asset HANK of Auclert, Bardóczy,
Rognlie & Straub (2021) at production grids (nB=50, nA=70, nK=50) on
`sequence-jacobian` 1.0.0, linear impulse responses to monetary, government
spending and TFP shocks, the one-asset variant, and steady-state
distributional statistics.
