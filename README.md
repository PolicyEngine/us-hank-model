# us-hank-model

US two-asset HANK model — a **validated replication** of Auclert, Bardóczy,
Rognlie & Straub, *"Using the Sequence-Space Jacobian to Solve and Estimate
Heterogeneous-Agent Models"* (Econometrica, 2021) — packaged for
PolicyEngine MacroMod. Built on the authors' MIT-licensed
[sequence-jacobian](https://github.com/shade-econ/sequence-jacobian) toolkit
(pinned at 1.0.0), with the model solved at the paper's production grid sizes
(nB=50, nA=70, nK=50) rather than the toolkit's coarse demo grids.

## Validation status

**What is genuinely validated.** The parameters the model solves for
internally reproduce ABRS (2021) Table B.III to the last published digit:

| Parameter | Published | Solved here | Deviation |
|---|---|---|---|
| β discount factor | 0.976 | 0.9762739 | +2.7e-4 |
| χ1 portfolio adj. cost scale | 6.416 | 6.4164196 | +4.2e-4 |
| Z TFP | 0.468 | 0.4677898 | −2.1e-4 |
| α capital share | 0.33 | 0.3299492 | −5.1e-5 |
| μp price markup | 1.015 | 1.0152284 | +2.3e-4 |
| τ labour tax | 0.356 | 0.3560606 | +6.1e-5 |

χ1 is the strongest check: it is solved numerically from a 6.5 starting guess
and lands on the published 6.416.

Market clearing: asset market 3.2e-13 (a solver target), goods market 3.3e-08
(**untargeted** — it holds only by Walras' law, so it is the honest measure of
steady-state accuracy and is five orders looser than the asset market).

The household block at full grids reproduces the upstream regression targets
(A, B, UCE) bit for bit; those are verbatim from `sequence-jacobian` v1.0.0's
`tests/base/test_two_asset.py`, and that test is *partial* equilibrium at the
solver's χ1=6.5 **guess**, so it validates the household code, not the
calibration.

**What is hit by construction and proves nothing.** Y=1, K=10, r=1.25%
quarterly, total wealth 14 (3.5× annual GDP), Bg=2.8 and G=0.2 are calibration
*inputs*. They are imposed, and (Z, α, μp, τ, φ) are backed out to make them
consistent. Do not read them as a successful replication; the table above is
the replication.

**Known discrepancy — φ (labour disutility).** Table B.III publishes
φ = 2.073; this DAG's `union_ss` block yields 1.7135, a 17% gap, and it does
not reconcile under either the earnings-weighted or unweighted definition of
`UCE`. It appears to be a reporting-convention difference: φ is a pure
normalisation chosen so N=1 is optimal, this implementation is self-consistent
(the wage Phillips residual is 7e-15 at the steady state), and no asset,
wealth or market-clearing target depends on it. It is nonetheless unexplained
and is deliberately **not** asserted in the test suite.

**IRF signs are checked; magnitudes are not validated against the paper.**
Monetary easing is expansionary, tax-financed G crowds out C and I under an
active Taylor rule, and TFP shocks are expansionary and disinflationary — all
gated by tests. But ABRS (2021) contains **no** figure plotting Y/C/I/π
impulse responses for the two-asset model against which magnitudes could be
checked; its only two-asset monetary figure (Fig. 7a) plots consumption alone
and exists to compare the linear and nonlinear solvers. The `-25bp, ρ=0.61`
convention used in the examples here comes from the upstream repo's notebook
and its *one-asset* test, not from the paper. Treat IRF magnitudes as this
implementation's output, not as a reproduced published result.

Responses are exactly linear in shock size (machine precision) — that is a
property of the sequence-space method, not evidence about the economy.

## Scope — read this

This model scores **stylized aggregate shocks** (monetary policy `rstar`,
government spending `G`, TFP `Z`) with distributional detail (MPCs by liquid
wealth, hand-to-mouth share). It is **not a forecaster** and it does **not**
score detailed tax reforms: the DAG has no exogenous transfer or tax-rate
instrument — the labor tax rate adjusts endogenously to balance the
government budget, so G shocks are implicitly tax-financed. For detailed
US tax-benefit reform scoring, use PolicyEngine's microsimulation models.

### Shock sizes are LEVELS, and for TFP that is not a percentage

`size` is a level change in the driving variable for every kind. Two of the
three read naturally; the third is a trap:

| kind | drives | ss level | `size=0.01` means |
|---|---|---|---|
| `monetary` | `rstar` | 0.0125 | +100bp per quarter |
| `fiscal_spending` | `G` | 0.2 | 1% of quarterly GDP (Y=1), i.e. a 5% rise in G |
| `productivity` | `Z` | **0.468** | a **2.14%** TFP improvement — *not* 1% |

Every IRF returns `shock_pct_of_ss` and a `units` string, and
`model.SHOCK_UNITS` carries the same metadata, so no caller needs to hard-code
a units description. Describing `size=0.01` on `productivity` as "a 1% TFP
shock" is wrong by more than a factor of two.

### G shocks are tax-financed, and the IRF shows it

Bond supply `Bg` is a fixed scalar — it is neither an input nor an output of
the GE Jacobian, so **no deficit channel exists in this DAG**. The labour tax
does all the financing, and `tax*w*N = r*Bg + G` holds period by period (to
1e-18 along the IRF, gated by a test). Every IRF therefore returns `tax`, the
labour-tax-rate path in percentage points: a 1%-of-GDP spending shock raises
the tax rate by 0.51pp on impact. The resulting impact multiplier (~0.18) is a
**balanced-budget** multiplier under an active Taylor rule, and must not be
compared with deficit-financed multipliers.

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

ss = model.solve_steady_state()        # ~10s at production grids (cached)
G = model.solve_jacobian(ss, T=300)    # ~3s (cached)

# -25bp monetary easing, AR(1) persistence 0.61
irf = model.shock('monetary', -0.0025, 0.61, T=300, ss=ss, G_jac=G)
irf['Y']               # % deviation from steady state, quarters 0..299
irf['shock_pct_of_ss'] # the shock as % of the driving variable's ss level
irf['units']           # what `size` meant

# G shock worth 1% of GDP, persistence 0.8
irf_g = model.shock('fiscal_spending', 0.01, 0.8, T=300, ss=ss, G_jac=G)
irf_g['tax']           # the labour tax path that pays for it, in pp

distributional.summary(ss)             # MPCs (with windfall size), HtM split

irf1 = one_asset.shock('monetary', -0.0025, 0.61)  # fast one-asset variant
```

Units: `Y`, `C`, `I` are % deviations from steady state; `pi`, `r` and `tax`
are quarterly-rate/rate deviations in percentage points. `size` is always a
LEVEL change in the driving variable — see the table above.

## Distributional output — read the conventions

ABRS (2021) reports **no MPC and no hand-to-mouth share anywhere**. These are
this implementation's own computations from the steady state; there is no
published number to replicate, and they should never be presented as one.

**Hand-to-mouth share ≈ 0.537** at the default cutoff, of which 0.512 is an
atom of households holding *exactly zero* liquid assets. The cutoff barely
matters below the first positive grid point — the number is an atom, not a
threshold. Almost all of it (0.518 of 0.537, ~96%) is **wealthy** HtM.
Against the US evidence this is high: Kaplan, Violante & Weidner (2014) put
total HtM near **one third** of US households, about two thirds of them
wealthy HtM. `hand_to_mouth_breakdown()` returns the split.

**MPCs are windfall-size dependent and only partly grid-converged.** The
consumption policy kinks at the borrowing constraint, where over half the
population sits, and nB=50 does not resolve that kink. The legacy
one-grid-step MPC rises monotonically 0.076 → 0.100 as nB goes 50 → 150 with
no plateau, because its implied windfall is set by grid spacing rather than by
economics. MPCs are therefore measured out of an **explicit windfall** of
stated economic size (default: 25% of mean quarterly post-tax labour income,
about a $1,000–1,500 rebate), giving an aggregate quarterly MPC of **0.058**.
Either way this sits below the US empirical range of roughly **0.15–0.25** for
a quarterly MPC out of a rebate. Quote the number with its windfall size, or
not at all — `summary()` always ships both.

**MPC by liquid quartile: Q1 and Q2 are identical by design.** Both quartile
edges fall inside the zero-liquid-wealth atom, so Q1 and Q2 are the same
households at the same liquid wealth. Any Q1-vs-Q2 gradient would be an
artifact of how tied grid cells happen to be ordered — randomising the tie
order moves it around and can invert it — so tied blocks are split
proportionally and both bins report the atom's mean.

**The by-wealth consumption allocation is an accounting split, not a solved
response.** `consumption_response_by_wealth_quantile` takes the *aggregate*
impact response and divides it across quantiles in proportion to steady-state
MPCs. It captures the MPC-heterogeneity channel only. It uses no
household-specific exposure to the shock, so its shape is identical across
shock kinds, and its sign is common to every quantile — it cannot represent a
policy that helps one group and hurts another. Household-level Jacobians would
be needed for that, and this package does not compute them.

## Tests

```bash
pip install -e .[dev]
pytest                # fast tests only (fiscal-path and one-asset checks)
pytest --runslow      # full production-grid validation: 47 tests
```

The `--runslow` gate covers the published-parameter comparison, market-clearing
residual magnitudes, IRF signs for all three shock kinds in both directions,
exact linearity, the period-by-period tax financing of G shocks, and the
distributional conventions above.

## License and attribution

MIT. The model and solution method are due to Adrien Auclert, Bence Bardóczy,
Matthew Rognlie and Ludwig Straub (2021); their `sequence-jacobian` code is
MIT-licensed. Please cite the paper when using this model.
