# Validation

[← README](../README.md)

**What is genuinely validated.** Six of the seven parameters the model
solves for internally reproduce ABRS (2021) Table B.III to the last published
digit. The seventh, φ, is 17% below the table as solved, and matches it after
multiplying by μw² (see "φ and the wage-markup convention" below):

| Parameter | Published | Solved here | Deviation |
|---|---|---|---|
| β discount factor | 0.976 | 0.9762739 | +2.7e-4 |
| χ1 portfolio adj. cost scale | 6.416 | 6.4164196 | +4.2e-4 |
| Z TFP | 0.468 | 0.4677898 | −2.1e-4 |
| α capital share | 0.33 | 0.3299492 | −5.1e-5 |
| μp price markup | 1.015 | 1.0152284 | +2.3e-4 |
| τ labour tax | 0.356 | 0.3560606 | +6.1e-5 |
| φ labour disutility (× μw²) | 2.073 | 2.0733059 | +3.1e-4 |

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
consistent. Do not read them as a successful replication. Four of the seven
parameters in the table above are also closed-form in these inputs
(`partial_ss` and `fiscal`): mc = 1 − r(p − K)/Y = 0.985 with p = 14 − 2.8,
so μp = 1/mc = 1.015228, α = (r + δ)K/(Y·mc) = 0.329949, Z = Y·K^(−α) =
0.46779, w = mc(1 − α)Y = 0.66 and τ = (r·Bg + G)/(w·N) = 0.356061. They
match whenever the inputs do. β and χ1 test the household solve, and so does
φ, whose closed form uses UCE, an output of the household block.

**φ and the wage-markup convention.** This DAG's `union_ss` block solves
φ = (1 − τ)·w·UCE / (μw·N^(1+ν)) = 1.7134759, where UCE = ∫ e·c^(−σ). That is
the steady state of the wage Phillips curve as the published Econometrica
online appendix writes it (Appendix B.3, Unions), with the after-tax wage term
divided by μw, so the wage is μw times the marginal rate of substitution. The
NBER working-paper version (w26123, eq. 69) writes the same curve with μw
multiplying that term instead. Its steady state is φ = μw·(1 − τ)·w·UCE/N^(1+ν)
= 2.0733059 here, which is exactly μw² = 1.21 times ours. Both versions print
φ = 2.073 in the calibration table. So Table B.III's φ follows the working
paper's convention, and this implementation follows the published equation,
which is also the one that puts the wage above the MRS. The two φ values
describe the same steady state (same β, χ1, prices and allocations). In
dynamics, the multiplying form scales the union's wage-gap term by μw² for a
given κw, so IRFs from code written in that convention would not be
comparable one for one. One case is measured: impact consumption after the
paper's monetary shock is 1.468% under the multiplying form against 1.485%
here (see the next paragraph); nothing else is. The test suite asserts
vphi·μw² against 2.073 and states the convention.

**IRF signs are checked; one stated magnitude is matched within about 2%.**
Monetary easing is expansionary, tax-financed G crowds out C and I under an
active Taylor rule, and TFP shocks are expansionary and disinflationary — all
gated by tests. ABRS (2021) plots no Y/C/I/π impulse responses for the
two-asset model, but its text states one magnitude: the impact consumption
response to a −5pp Taylor-rule shock with quarterly persistence 0.6 is 1.46%
on the linear solution and 1.36% on the nonlinear one. These numbers are from
the working paper's text (NBER w26123, pp. 40–41, Fig. 8a); the figure is
Fig. 7a in the Econometrica version, whose text was not checked for them. This model
gives 1.485% (+1.7%) and 1.388% (+2.1%), and the nonlinear/linear ratio is
0.935 against 0.932. The paper does not state the shock's units; annualised
percentage points is inferred, since the quarterly reading gives 5.94%. Part
of the linear gap is the wage-markup convention above (1.468% under the
paper-era form). `tests/test_published_magnitude.py` gates these to 3%.
No other variable, horizon or shock is compared with a published magnitude:
output, investment, inflation and the fiscal and productivity responses are
sign-checked only, and the impact peaks here have no hump. The
`-25bp, ρ=0.61` convention used in the examples comes from the upstream
repo's notebook and its *one-asset* test, not from the paper.

Responses are exactly linear in shock size (machine precision) — that is a
property of the sequence-space method, not evidence about the economy. The
model itself is not linear at the sizes quoted in the README; see "The spending
numbers are first-order" in [SHOCKS.md](SHOCKS.md).

## Every claim, its number, and what backs it

| Claim | Number | Test | Status |
|---|---|---|---|
| β, χ1 match Table B.III | 0.9762739, 6.4164196 | `test_steady_state.py::test_internally_calibrated_parameters_match_published_table` | matches a published value |
| φ × μw² matches Table B.III | 2.0733059 | `test_steady_state.py::test_vphi_matches_published_table_in_the_working_paper_convention` | matches a published value (after the convention conversion) |
| Impact consumption, Fig. 7a shock | 1.485% / 1.388% vs 1.46% / 1.36% | `test_published_magnitude.py` | matches a published value (units inferred) |
| Z, α, μp, τ match Table B.III | 0.4677898, 0.3299492, 1.0152284, 0.3560606 | `test_steady_state.py` | closed-form in inputs |
| Goods market clears (untargeted) | 3.3e-08 | `test_steady_state.py::test_market_clearing_residual_magnitudes` | identity or property test |
| G is labour-tax financed period by period | 1e-12 | `test_shocks.py`, `test_scenarios.py` | identity or property test |
| Linearity, r₀ = −(1 + r)π₀, household budget, ss fixed point | — | `test_scenarios.py`, `test_spending_mechanism.py` | identity or property test |
| Monetary reference shock: Y / C / I / π / i on impact | 0.493 / 0.300 / 1.698 / 0.079 / −0.132 | `test_scenarios.py::test_monetary_reference_shock_is_pinned` | regression pin, no external benchmark |
| Spending multipliers (ρ 0.8 / 0.9), nonlinear 0.251 | 0.175 / 0.075 | `test_scenarios.py` | regression pin, no external benchmark |
| MPCs, hand-to-mouth, high-MPC share | 0.058; 0.104 / 0.091 / 0.083; 0.537; 3.1% at $1,000 | `test_distributional.py`, `test_refined_mpc.py` | regression pin, no external benchmark |
| β-het rows | see [BETA_HET.md](BETA_HET.md) | `test_beta_het.py` | regression pin, no external benchmark |
| β-het sensitivity; median-mode bracket; Bh = 0.5 recalibration; 2%-of-GDP non-convergence | — | none | measured once, not gated |
