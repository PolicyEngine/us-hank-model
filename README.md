# us-hank-model

US two-asset HANK model — a **validated replication** of Auclert, Bardóczy,
Rognlie & Straub, *"Using the Sequence-Space Jacobian to Solve and Estimate
Heterogeneous-Agent Models"* (Econometrica, 2021) — packaged for
PolicyEngine MacroMod. Built on the authors' MIT-licensed
[sequence-jacobian](https://github.com/shade-econ/sequence-jacobian) toolkit
(pinned at 1.0.0), with the model solved at the paper's production grid sizes
(nB=50, nA=70, nK=50) rather than the toolkit's coarse demo grids.

## Validation status

- Steady state reproduces the paper's calibration targets: Y=1, K=10,
  r=1.25% quarterly, total wealth = 14 (3.5× annual GDP), liquid assets
  B = 1.04, solved discount factor β ≈ 0.976.
- The household block at full grids matches the upstream repo's regression
  targets (A, B, UCE) to machine precision.
- IRF signs and magnitudes match the paper: monetary easing is expansionary;
  tax-financed G spending crowds out consumption and investment under an
  active Taylor rule.

## Scope — read this

This model scores **stylized aggregate shocks** (monetary policy `rstar`,
government spending `G`, TFP `Z`) with distributional detail (MPCs by liquid
wealth, hand-to-mouth share). It is **not a forecaster** and it does **not**
score detailed tax reforms: the DAG has no exogenous transfer or tax-rate
instrument — the labor tax rate adjusts endogenously to balance the
government budget, so G shocks are implicitly tax-financed. For detailed
US tax-benefit reform scoring, use PolicyEngine's microsimulation models.

## Fiscal 2025 variant

`us_hank.fiscal_2025` implements the debt, tax-cut, spending, and repayment
paths published in Auclert, Rognlie, and Straub (2025), *“Fiscal and Monetary
Policy with Heterogeneous Agents.”* It is a clean-room implementation from the
paper: the companion `shade-econ/annual-review` repository currently declares
no license, so none of its notebook/module code or calibration inputs are
copied here.

```python
from us_hank import fiscal_2025

paths = fiscal_2025.fiscal_paths("deficit_tax_cut", size=0.01, T=300)
paths["debt"]
paths["tax_revenue"]
```

The heterogeneous-agent `fiscal_2025.shock()` response is intentionally hard
blocked. Shipping it requires licensed calibration inputs or author permission,
followed by numerical reproduction of the paper's Figure 2. The tested fiscal
paths are infrastructure for that validation; they are not presented as a
validated HANK impulse response.

## Quickstart

```python
from us_hank import model, distributional, one_asset

ss = model.solve_steady_state()        # ~1-2 min at production grids (cached)
G = model.solve_jacobian(ss, T=300)    # cached

# -25bp monetary easing, AR(1) persistence 0.61
irf = model.shock('monetary', -0.0025, 0.61, T=300, ss=ss, G_jac=G)
irf['Y']   # % deviation from steady state, quarters 0..299

# G shock worth 1% of GDP, persistence 0.8
irf_g = model.shock('fiscal_spending', 0.01, 0.8, T=300, ss=ss, G_jac=G)

distributional.summary(ss)             # MPC by liquid quartile, HtM share

irf1 = one_asset.shock('monetary', -0.0025, 0.61)  # fast one-asset variant
```

Units: `Y`, `C`, `I` are % deviations from steady state; `pi` and `r` are
quarterly-rate deviations in percentage points.

## Tests

```bash
pip install -e .[dev]
pytest                # fast tests only
pytest --runslow      # full production-grid validation (several minutes)
```

## License and attribution

MIT. The model and solution method are due to Adrien Auclert, Bence Bardóczy,
Matthew Rognlie and Ludwig Straub (2021); their `sequence-jacobian` code is
MIT-licensed. Please cite the paper when using this model.
