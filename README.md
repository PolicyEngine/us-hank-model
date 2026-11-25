# us-hank-model

US two-asset HANK model: a **calibrated replication** of Auclert, Bardóczy,
Rognlie & Straub, *"Using the Sequence-Space Jacobian to Solve and Estimate
Heterogeneous-Agent Models"* (Econometrica, 2021), packaged for PolicyEngine
Macro. It scores stylised aggregate shocks (monetary, government spending,
TFP) with distributional detail. It is built on the authors' MIT-licensed
[sequence-jacobian](https://github.com/shade-econ/sequence-jacobian) toolkit
(pinned at 1.0.0), solved at the paper's production grids (nB=50, nA=70,
nK=50) rather than the toolkit's demo grids.

## Status

| What | Status | Detail |
|---|---|---|
| Steady-state parameters | Six of seven solved parameters match ABRS (2021) Table B.III to the last published digit; the seventh (φ) matches after a wage-markup convention conversion. Only β, χ1 and φ test the household solve; Z, α, μp and τ are closed-form in the inputs | [VALIDATION.md](docs/VALIDATION.md) |
| Calibration targets | Y, K, r, total wealth, Bg and G are **inputs**, imposed by construction, not results | [VALIDATION.md](docs/VALIDATION.md) |
| Impulse-response magnitudes | The one magnitude the paper states (impact consumption after a monetary shock, Fig. 7a) is matched within about 2%, reading the shock in annualised points, which the paper does not state. Every other response is sign-checked and regression-pinned only | [VALIDATION.md](docs/VALIDATION.md) |
| MPCs and hand-to-mouth share | This implementation's own numbers; ABRS report neither | [DISTRIBUTIONAL.md](docs/DISTRIBUTIONAL.md) |

## Install

```bash
pip install git+https://github.com/PolicyEngine/us-hank-model   # Python >= 3.11
```

## Quickstart

```python
from us_hank import model, distributional, one_asset

ss = model.solve_steady_state()        # a few seconds at production grids (cached)
G = model.solve_jacobian(ss, T=300)    # a few seconds (cached)

# -25bp-per-quarter cut in the Taylor-rule intercept, AR(1) persistence 0.61
irf = model.shock('monetary', -0.0025, 0.61, T=300, ss=ss, G_jac=G)
irf['Y']               # % deviation from steady state, quarters 0..299
irf['shock_pct_of_ss'] # the shock as % of the driving variable's ss level
irf['units']           # what `size` meant

# G shock worth 1% of GDP, persistence 0.8
irf_g = model.shock('fiscal_spending', 0.01, 0.8, T=300, ss=ss, G_jac=G)
irf_g['tax']           # the labour tax path that pays for it, in pp
irf_g['labour_income_post_tax']  # % change in (1 - tax)*w*N: falls, though w*N rises

distributional.summary(ss)             # MPCs (with windfall size), HtM split

irf1 = one_asset.shock('monetary', -0.0025, 0.61)  # fast one-asset variant
                                                     # (toolkit notebook grid: 7 income states, 500 asset points)
```

`size` is always a **level** change in the driving variable: `0.01` on
`productivity` is a 2.14% TFP rise, not 1%. `Y`, `C`, `I` are % deviations;
`pi`, `r` and `tax` are percentage-point deviations of quarterly rates. `r`
is the ex-post real return. Details: [SHOCKS.md](docs/SHOCKS.md).

## Headline results

Default calibration, production grids, linear (first-order) responses on
impact unless stated.

| Result | Value | Backed by |
|---|---|---|
| Monetary, −25bp/quarter intercept cut, ρ 0.61 | Y +0.493%, C +0.300%, I +1.698%, π +0.079pp/q, policy rate −0.132pp | regression pin, `tests/test_scenarios.py` |
| Fig. 7a shock (−5pp annualised, ρ 0.6): consumption | 1.485% linear, 1.388% nonlinear (paper: 1.46%, 1.36%) | `tests/test_published_magnitude.py` |
| G +1% of GDP, ρ 0.8: impact / 20-quarter multiplier | 0.175 / −0.015 (nonlinear impact 0.251) | regression pin, `tests/test_scenarios.py` |
| G +1% of GDP, ρ 0.9: impact / 20-quarter multiplier | 0.075 / −0.032 | regression pin, `tests/test_scenarios.py` |
| Quarterly MPC, windfall 25% of income (~$6,200 per household) | 0.058 | `tests/test_distributional.py` |
| Quarterly MPC, $500 / $1,000 / $1,500 per household | 0.104 / 0.091 / 0.083 | `tests/test_refined_mpc.py` |
| Hand-to-mouth share by liquid wealth (poor / wealthy) | 0.537 (0.019 / 0.518) | `tests/test_distributional.py` |
| Share with MPC > 0.25 out of $1,000 | 3.1% | `tests/test_refined_mpc.py` |

Every claim with its test and status: [VALIDATION.md](docs/VALIDATION.md#every-claim-its-number-and-what-backs-it).

## Limits

- **Not a forecaster** and **not a tax-reform scorer**: the DAG has no
  exogenous transfer or tax-rate instrument. For US tax-benefit reforms, use
  PolicyEngine's microsimulation models.
- **G shocks are balanced-budget**: the labour tax adjusts every period and
  there is no deficit channel, so the multiplier must not be compared with
  deficit-financed ones. Output turns negative from quarter 3 at ρ 0.8; the
  mechanism is tested. Quote ρ with any multiplier.
- **First-order**: responses are exactly linear in shock size. The model is
  not: at 1% of GDP the nonlinear spending multiplier is 0.251, not 0.175.
- **Peaks are on impact**: no hump-shaped responses.
- **MPCs sit below the US rebate evidence** (0.15–0.25), even on a converged
  grid, because the illiquid adjustment cost has zero marginal cost at zero
  adjustment.
- **The 54% hand-to-mouth share counts liquid wealth, not spending
  behaviour**: most of those households hold illiquid wealth and spend little
  of a rebate.

## Variants

- **`us_hank.one_asset`**: the toolkit's one-asset HANK on its notebook grid.
  Monetary and TFP shocks only; no capital, no fiscal instrument.
- **`us_hank.beta_het`** (opt-in, not the default): discount-factor
  heterogeneity calibrated to a 0.20 rebate MPC, plus a median-wealth
  calibration. [BETA_HET.md](docs/BETA_HET.md).
- **`us_hank.fiscal_2025`**: clean-room debt and tax paths from Auclert,
  Rognlie & Straub (2025). Its heterogeneous-agent `shock()` is deliberately
  blocked until a licensed numerical replication exists.
  [FISCAL_2025.md](docs/FISCAL_2025.md).

## Tests

```bash
pip install -e .[dev]
pytest                # fast: fiscal paths, one-asset, determinism, β-het demo grids
pytest --runslow      # full production-grid validation (about 5 minutes on CI)
```

CI runs both on every pull request, on Python 3.11, 3.12 and 3.15. The
`--runslow` gate covers the published-parameter comparison, market-clearing
residual magnitudes, the paper's one stated impulse-response magnitude, IRF
signs for all three shock kinds in both directions, exact linearity, the
period-by-period tax financing of G shocks, and the distributional
conventions.

## Pins

`sequence-jacobian==1.0.0` exactly, with major-version ceilings on NumPy,
SciPy and Numba (see `pyproject.toml`): the replication asserts published
parameters to the last digit, so the solver version is part of the result.
PolicyEngine Macro pins a commit of this repository. Changes by version:
[CHANGELOG.md](CHANGELOG.md).

## License and attribution

MIT. The model and solution method are due to Adrien Auclert, Bence Bardóczy,
Matthew Rognlie and Ludwig Straub (2021); their `sequence-jacobian` code is
MIT-licensed. Please cite the paper when using this model.
