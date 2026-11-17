# Distributional output: read the conventions

[← README](../README.md)

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

**The 54% counts liquid wealth, not spending behaviour.** It is a
liquid-wealth definition in the spirit of Kaplan, Violante & Weidner, and in
this calibration its members are not high spenders, so do not read it as the
share of households with high MPCs. 52 of the 54 points are households with
illiquid wealth above the same 0.05 cutoff. The illiquid adjustment cost has zero marginal cost
at zero adjustment, so these households absorb a windfall mainly by
withdrawing less: about 75% of a $1,000 rebate goes to the illiquid account
and 14% is consumed (`windfall_allocation()`). 1.9% of households are
below the cutoff in both assets (poor hand-to-mouth). `summary()` also returns
`high_mpc_share`, the population share whose MPC out of the same windfall
exceeds 0.25. It is 0.0 at the 25% default windfall. On the refined liquid
grid it is 4.9% / 3.1% / 2.1% at $500 / $1,000 / $1,500 per household, while
46% of households have an MPC above 0.1 at $1,000; both are converged in the
liquid grid and pinned in `tests/test_refined_mpc.py`. Read the hand-to-mouth
share and `high_mpc_share` together. The gap is a property of the ABRS
calibration, not something this package corrects; the opt-in β-heterogeneity
variant (`us_hank.beta_het`) is calibrated to a higher rebate MPC.

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
25% default the refined figure is 0.059 against 0.058, so the headline number
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
Raising the hand-to-mouth share does not help (measured once, not gated by
a test). Recalibrated to liquid wealth
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
