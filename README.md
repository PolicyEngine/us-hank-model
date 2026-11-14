# us-hank-model

US two-asset HANK model — a **calibrated replication** of Auclert, Bardóczy,
Rognlie & Straub, *"Using the Sequence-Space Jacobian to Solve and Estimate
Heterogeneous-Agent Models"* (Econometrica, 2021): six of the seven solved
steady-state parameters match the paper, and impulse-response magnitudes are
not validated (see below) — packaged for
PolicyEngine MacroMod. Built on the authors' MIT-licensed
[sequence-jacobian](https://github.com/shade-econ/sequence-jacobian) toolkit
(pinned at 1.0.0), with the model solved at the paper's production grid sizes
(nB=50, nA=70, nK=50) rather than the toolkit's coarse demo grids.

## Validation status

**What is genuinely validated.** Six of the seven parameters the model
solves for internally reproduce ABRS (2021) Table B.III to the last published
digit; the seventh, φ, is 17% off (below):

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
consistent. Do not read them as a successful replication. Four of the six
parameters in the table above are also closed-form in these inputs
(`partial_ss` and `fiscal`): mc = 1 − r(p − K)/Y = 0.985 with p = 14 − 2.8,
so μp = 1/mc = 1.015228, α = (r + δ)K/(Y·mc) = 0.329949, Z = Y·K^(−α) =
0.46779, w = mc(1 − α)Y = 0.66 and τ = (r·Bg + G)/(w·N) = 0.356061. They
match whenever the inputs do. Only β and χ1 test the household solve.

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
property of the sequence-space method, not evidence about the economy. The
model itself is not linear at the sizes quoted below; see "The spending
numbers are first-order".

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
| `monetary` | `rstar` | 0.0125 | +100bp per quarter, about +400bp annualised (25bp annualised is `0.000625`) |
| `fiscal_spending` | `G` | 0.2 | 1% of quarterly GDP (Y=1), i.e. a 5% rise in G |
| `productivity` | `Z` | **0.468** | a **2.14%** TFP improvement — *not* 1% |

Every IRF returns `shock_pct_of_ss` and a `units` string, and
`model.SHOCK_UNITS` carries the same metadata, so no caller needs to hard-code
a units description. Describing `size=0.01` on `productivity` as "a 1% TFP
shock" is wrong by more than a factor of two.

### G shocks are tax-financed, and the IRF shows it

Bond supply `Bg` is a fixed scalar — it is neither an input nor an output of
the GE Jacobian, so **no deficit channel exists in this DAG**. The labour tax
does all the financing, and `tax*w*N = r*Bg + G` holds period by period (gated
by a test at 1e-12 along the IRF). Every IRF therefore returns `tax`, the
labour-tax-rate path in percentage points: a 1%-of-GDP spending shock
(persistence 0.8) raises the tax rate by 0.51pp on impact and by 1.59pp in
quarter 1. The impact rise is small because the ex-post real rate falls 0.21pp
on the inflation surprise, which cuts debt service `r*Bg` by 0.59% of GDP;
from quarter 1 the real rate is above steady state and the tax does the full
work. The resulting impact multiplier (0.175) is a
**balanced-budget** multiplier under an active Taylor rule, and must not be
compared with deficit-financed multipliers.

The multiplier depends strongly on persistence, and output turns negative
from quarter 3 while G is still raised: the labour tax stays 1–1.6pp higher
and investment is crowded out. This is a model property, not a bug
(`tests/test_spending_mechanism.py`). The household and government budget
constraints hold along the IRF. The sign change comes from the labour tax
entering the union's wage Phillips curve: a higher tax raises the wage the
union sets, which pushes up wage and price inflation, and the Taylor rule
answers with a higher real rate. If the tax is held out of wage setting
(households still pay it), output stays positive every quarter, and the
20-quarter cumulative multiplier is 0.73 instead of −0.015. A weaker Taylor
rule (φ = 1.1) also keeps output positive, while φ = 3 makes output negative
on impact. At rho 0.8 the impact multiplier is 0.175 and
the cumulative multiplier over 20 quarters is −0.015. At rho 0.9 (PolicyEngine
Macro's hosted default) they are 0.075 and −0.032. No published IRF for this
calibration gates these numbers; `tests/test_scenarios.py` pins them as
regression values only.

### The spending numbers are first-order

Every spending figure in this README, including the β-het tables, is from the
linear (first-order) IRF. The nonlinear check below is for the default
calibration only. On the model's own
nonlinear perfect-foresight transition, a 1%-of-GDP rise in G (ρ 0.8) gives
an impact multiplier of 0.251, not 0.175, and the labour tax rises 0.30pp on
impact, not 0.51pp. A 1%-of-GDP cut lowers output by 0.125% on impact (a
multiplier of 0.125). The second-order term is
about +0.06% of GDP and keeps its sign when the shock flips, so at this size
the linear IRF understates the output gain from a rise by about 30% and
overstates the loss from a cut. At 2% of GDP the nonlinear solve does not
converge at default settings, so only the linear IRF exists there. A
25bp-per-quarter monetary shock (ρ 0.61) is close to linear: impact output +0.511
for the cut and −0.477 for the rise, against 0.493 linear. These are pinned
in `tests/test_scenarios.py`
(`test_spending_multiplier_is_strongly_nonlinear_at_1pct_of_gdp`,
`test_monetary_nonlinearity_at_100bp_annualised_is_small`).

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

## Opt-in variant: discount-factor heterogeneity (issue #8)

`us_hank.beta_het` is a second, separate DAG for users who need consumption
responses to transfers that match the rebate evidence. **It is not the
default**: `us_hank.model` and every number above are the published
calibration and are unchanged.

```python
from us_hank import beta_het, model
ss_v = beta_het.solve_steady_state()          # calibrated variant
irf = beta_het.shock('fiscal_spending', 0.01, 0.8, ss=ss_v)
```

Design. Two permanent discount-factor types: one third of households are
impatient (quarterly β 0.852), two thirds patient (β 0.983); a household
redraws its type with probability 1/200 a quarter. The steady state hits
the default's targets (total wealth 14, liquid B 1.04) by solving (β, χ1) as
the default does, and the gap between the types is set so the quarterly MPC
out of a $1,000-per-household rebate (refined liquid grid) is 0.20, the
middle of the 0.15–0.25 range of the US rebate evidence (Kaplan & Violante
2014; source notes in `beta_het.MPC_TARGET`). β heterogeneity was chosen
over a fixed or linear illiquid adjustment cost because the toolkit's
household solver needs a smooth cost; see the module docstring. Three
symmetric types cannot reach the target: the patient type hits
β ≈ 1/(1+r) first.

| | Default | β-het variant |
|---|---|---|
| MPC, $500 / $1,000 / $1,500 per household (refined grid) | 0.104 / 0.091 / 0.083 | 0.227 / 0.200 / 0.186 |
| Hand-to-mouth share (poor / wealthy) | 0.537 (0.019 / 0.518) | 0.572 (0.203 / 0.369) |
| G shock ρ 0.8: impact multiplier (first-order) | 0.175 | 0.376 |
| G shock ρ 0.8: 20-quarter cumulative multiplier | −0.015 | 0.309 |
| G shock ρ 0.8: first quarter with Y < 0 | 3 | 23 |
| G shock ρ 0.9: impact / first Y < 0 | 0.075 / q3 | 0.319 / none in 40q |
| Monetary −25bp/q, ρ 0.61: impact Y / C / I (%) | 0.493 / 0.300 / 1.698 | 0.244 / 0.308 / 0.478 |

So with an MPC in the evidence range the spending multiplier doubles on
impact and the negative output path of issue #5 largely goes away: the
same tax-and-wage-markup channel is still there, but higher-MPC households
spend more of the extra income. Monetary easing moves output by half as
much, and investment responds far less; consumption is about the same.
The variant is calibrated on three numbers, not estimated, and the impatient
type's β is low (0.852 a quarter, 0.53 a year): it is a device to put the
MPC in range, not a measured preference. It sits well below published
discount-factor distributions: Carroll, Slacalek, Tokuoka & White (2017)
estimate quarterly β between about 0.94 and 0.98 when matching liquid
assets, and Kaplan & Violante (2014) use a single β of 0.941 a year. Sensitivity (CI run 42145287339):
an impatient mass of 1/4 or 1/2 needs β gaps of 0.160 or 0.099 and gives
impact multipliers 0.403 and 0.385; halving the type-switch rate moves the
$1,000 MPC 0.200 → 0.207.

### Second calibration: median wealth targets (`liquid_target='median'`)

The first calibration keeps the aggregate targets (total wealth 14, liquid
B 1.04), and that forces the low impatient β. Kaplan & Violante (2014,
fn. 28) target median wealth instead of aggregate wealth. The second
calibration targets the medians but keeps ABRS's total wealth 14 (about 21
quarters of mean labour income, against a mean net worth of 11.4 quarters
in KV's 2001 SCF sample, which excludes the top 5%), so it is a hybrid of
the two approaches, closer to Carroll et al. (2017) than to KV:

```python
ss_m = beta_het.solve_steady_state(liquid_target='median')
```

It uses three equal-mass types on a uniform β distribution. It drops the
B = 1.04 target, so B comes out at 1.21 and `Bh` is reset to it. χ1 is set
outside the solve, and β alone clears the asset market at A + B = 14. The
targets are from the 2001 SCF as reported in KV (2014, Table III), ratios to
mean labour income: median liquid wealth 0.199 quarters and median illiquid
wealth 4.14 quarters. The poor hand-to-mouth target is 12.1% (Kaplan,
Violante & Weidner 2014, Table 3). The MPC band out of $500 is 0.15–0.25
(sources in `beta_het.MEDIAN_TARGETS`). Grid search: CI runs 42149673949,
42150101068, 42150572003 and 42150783547.

| | Default | β-het, aggregate targets | β-het, median targets | Target |
|---|---|---|---|---|
| β by type (quarterly) | 0.976 | 0.852 / 0.983 | 0.9435 / 0.9655 / 0.9875 | Carroll et al. liquid 0.936–0.978 |
| Median liquid / income (quarters) | 0.000 | 0.008 | 0.205 | 0.199 |
| Median illiquid / income (quarters) | 12.64 | 6.14 | 4.69 | 4.14 |
| Poor / all hand-to-mouth (b, a < 0.05) | 0.019 / 0.546 | 0.204 / 0.579 | 0.060 / 0.450 | 0.121 / 0.312 |
| MPC $500 / $1,000 / $1,500 | 0.104 / 0.091 / 0.083 | 0.227 / 0.200 / 0.186 | 0.126 / 0.110 / 0.101 | 0.15–0.25 at $500 |
| G ρ 0.8: impact / 20q cumulative multiplier | 0.175 / −0.015 | 0.376 / 0.309 | 0.274 / 0.213 | |
| G ρ 0.8 / 0.9: first quarter with Y < 0 | q3 / q3 | q23 / none | none / none (40q) | |
| Monetary −25bp/q: impact Y / C / I (%) | 0.493 / 0.300 / 1.698 | 0.244 / 0.308 / 0.478 | 0.360 / 0.240 / 1.191 | |

The moments are measured on the refined liquid grid, with the hand-to-mouth
shares using the 0.05 (about $1,000) cutoff. Both medians come close
(illiquid 13% high). The β points are not inside the Carroll et al. range:
the top point, 0.9875, is above 0.978, and the uniform distribution they
discretise runs from 0.932 to 0.998 (Carroll et al.: 0.936–0.978). β levels
are not comparable across models with different returns either; the top
type's β(1 + r) is 0.9998. Poor hand-to-mouth (6%, half the data) and the
MPC (0.126) stay below their targets, and wealthy hand-to-mouth (39%) is
twice the data (19.2%). With total wealth held at 14 they could not be
pushed further: a wider spread (0.034), or a fourth, impatient type with
12–15% of households, gave no root for the centre β in the solver's
bracket, which keeps the most patient type at least 1e-4 below 1/(1+r).
That is a property of the bracket, not a proof that no steady state exists:
with types redrawn every 200 quarters on average, a top β at or slightly
above 1/(1+r) need not be explosive, and dropping the total wealth target,
as KV do, was not tried. The fit is knife-edge: spread 0.033 solves and
0.034 does not, with the top type 2e-4 below the cap holding 73% of
illiquid wealth. The MPC is quoted per household at 2025 GDP per household
($58,000 a quarter); KV's $500 in 2001 is 3.8% of quarterly earnings, which
here is about $1,460, where the MPC is about 0.10. χ0 between 0.02 and
0.25 moves poor hand-to-mouth by under 1 point. The spending-multiplier
sign result survives: output stays positive for 40 quarters at ρ 0.8 and
0.9 without the extreme impatient type. The $500 MPC rises only from 0.104
to 0.126, so the average MPC alone does not explain the sign change; the
rest comes from the wealth distribution (not decomposed here). This
calibration is a median-wealth sensitivity, not a higher-MPC one.

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

`r` is the ex-post real
return on bonds held into the quarter, r_t = (1 + i_{t−1})/(1 + π_t) − 1. On
impact it moves only with the inflation surprise (r_0 = −(1 + r)·π_0); the
policy move itself shows from quarter 1. For the −25bp-per-quarter cut
(ρ 0.61), `r` falls 0.08pp on impact and 0.18pp in quarter 1, while the
ex-ante real rate i_t − π_{t+1} falls 0.18pp on impact.

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
which is 0.106 in model units, i.e. 10.6% of quarterly GDP per model
household), giving an aggregate quarterly MPC of **0.058**. Mapped to US
GDP (about $31T a year) that windfall is about $2,400 per head or about
$6,200 per household —
larger than the 2001 ($300–600) and 2008 (about $900 on average) rebates
the empirical MPC literature measures. MPCs fall with windfall size (the
policy is concave, and a test asserts it), but the size mismatch explains
little of the gap to the US empirical range of roughly **0.15–0.25**: at a
$1,000–1,500 rebate per head the MPC is 0.069–0.064, and per household that
rebate is smaller than the first liquid grid step, where the MPC is a flat
0.075 set by the grid (the grid-step MPC). Quote the number with its windfall size, or
not at all — `summary()` always ships both.

**Rebate-sized MPCs need the refined liquid grid.**
`distributional.refined_aggregate_mpc()` re-solves the household block alone,
at the GE steady-state prices, on a liquid grid packed near the constraint
(nB=200, pivot 0.01; nothing is recalibrated). It converges (the change from
nB=200 to 400 is under 1e-4). Per household, $500 / $1,000 / $1,500 give
**0.104 / 0.091 / 0.083**, against a flat 0.075 on the production grid. At the
25% default the refined figure is 0.059 against 0.058, so the published number
stands. `tests/test_refined_mpc.py` pins these numbers.

**Why the MPC is low, even converged.** The grid explains part of the gap to
0.15–0.25. It does not explain most of it. The model has plenty of
constrained households: half the population ends the quarter at the liquid
limit. But the illiquid account's adjustment cost is smooth and convex. It
has zero marginal cost at zero adjustment, and no linear term of the kind
Kaplan, Moll & Violante (2018) use to create an inaction region. So a
constrained household that holds illiquid wealth just withdraws less: on the
refined grid, 75% of a $1,000 rebate goes into the illiquid account, and 14%
is consumed (`distributional.windfall_allocation()`). For a larger windfall
(25% of income), the household leaves the constraint, keeps 60% of it liquid,
and consumes 8%. Only 1.8% of households are constrained with no illiquid
wealth, the group that acts close to hand-to-mouth (MPC 0.39 at $1,000).
Raising the hand-to-mouth share does not help. Recalibrated to liquid wealth
Bh = 0.5, the constrained share rises to 64%, yet the MPC *falls*; that
recalibration also cuts χ1 from 6.4 to 3.8, cheapening adjustment, which is
the same channel. Seven income states instead of three also lower it
slightly. ABRS (2021) report no MPC for this calibration, so none of this is
a replication failure. It is a property of the calibration.

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
pytest --runslow      # full production-grid validation
```

The `--runslow` gate covers the published-parameter comparison, market-clearing
residual magnitudes, IRF signs for all three shock kinds in both directions,
exact linearity, the period-by-period tax financing of G shocks, and the
distributional conventions above.

## License and attribution

MIT. The model and solution method are due to Adrien Auclert, Bence Bardóczy,
Matthew Rognlie and Ludwig Straub (2021); their `sequence-jacobian` code is
MIT-licensed. Please cite the paper when using this model.
