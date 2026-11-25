# Shocks, units and scope

[← README](../README.md)

This model scores **stylized aggregate shocks** (monetary policy `rstar`,
government spending `G`, TFP `Z`) with distributional detail (MPCs by liquid
wealth, hand-to-mouth share). It is **not a forecaster** and it does **not**
score detailed tax reforms: the DAG has no exogenous transfer or tax-rate
instrument — the labor tax rate adjusts endogenously to balance the
government budget, so G shocks are implicitly tax-financed. For detailed
US tax-benefit reform scoring, use PolicyEngine's microsimulation models.

## Shock sizes are LEVELS, and for TFP that is not a percentage

`size` is a level change in the driving variable for every kind. Two of the
three read naturally; the third is a trap:

| kind | drives | ss level | `size=0.01` means |
|---|---|---|---|
| `monetary` | `rstar` | 0.0125 | +100bp per quarter, about +400bp annualised (25bp annualised is `0.000625`) |
| `fiscal_spending` | `G` | 0.2 | 1% of quarterly GDP (Y=1), i.e. a 5% rise in G |
| `productivity` | `Z` | **0.468** | a **2.14%** TFP improvement — *not* 1% |

A monetary `size` moves the rule's intercept (`i = rstar + φ·π`, with φ here
the Taylor coefficient 1.5). The policy rate falls less than `size` because
the rule reacts to the higher inflation: 0.13pp on impact for the
−25bp-per-quarter example.

Every IRF returns `shock_pct_of_ss` and a `units` string, and
`model.SHOCK_UNITS` carries the same metadata, so no caller needs to hard-code
a units description. Four helpers do the conversions
(`tests/test_units.py`):

```python
model.monetary_size(-25)                       # -0.000625: a 25bp annualised cut
model.annualised_bp(-0.0025)                   # -100.0
model.size_pct_of_ss('productivity', 0.01)     # 2.14 (% of Z_ss)
model.size_from_pct_of_ss('productivity', 1)   # 0.00468: a 1% TFP improvement
```

The one-asset variant has different steady-state levels, and
`one_asset.SHOCK_UNITS` states them: `Z` is 1 there, so `0.01` on
`productivity` **is** a 1% TFP improvement, and the steady-state rate is 0.5%
a quarter (2% a year) against 1.25% (5% a year). A monetary `size` is the
same number of basis points in both. The one-asset rule is lagged a quarter,
`1 + r_t = (1 + rstar_{t−1} + φ·π_{t−1})/(1 + π_t)`, so the intercept reaches
the real return in quarter 1. Describing `size=0.01` on `productivity` as "a 1% TFP
shock" is wrong by more than a factor of two.

## G shocks are tax-financed, and the IRF shows it

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
rule (inflation coefficient 1.1 instead of 1.5; `phi` in the code, not the
labour-disutility φ in [VALIDATION.md](VALIDATION.md)) also keeps output positive, while a coefficient
of 3 makes output negative on impact. At rho 0.8 the impact multiplier is 0.175 and
the cumulative multiplier over 20 quarters is −0.015. At rho 0.9 (PolicyEngine
Macro's hosted default) they are 0.075 and −0.032. No published IRF for this
calibration gates these numbers; `tests/test_scenarios.py` pins them as
regression values only.

## The spending numbers are first-order

Every spending figure in the README and these docs, including the β-het tables, is from the
linear (first-order) IRF. The nonlinear check below is for the default
calibration only. On the model's own
nonlinear perfect-foresight transition, a 1%-of-GDP rise in G (ρ 0.8) gives
an impact multiplier of 0.251, not 0.175, and the labour tax rises 0.30pp on
impact, not 0.51pp. A 1%-of-GDP cut lowers output by 0.125% on impact (a
multiplier of 0.125). The second-order term is
about +0.06% of GDP and keeps its sign when the shock flips, so at this size
the linear IRF understates the output gain from a rise by about 30% and
overstates the loss from a cut. At 2% of GDP the nonlinear solve does not
converge at default settings (measured once, not gated by a test), so only
the linear IRF exists there. A
25bp-per-quarter monetary shock (ρ 0.61) is close to linear: impact output +0.511
for the cut and −0.477 for the rise, against 0.493 linear. These are pinned
in `tests/test_scenarios.py`
(`test_spending_multiplier_is_strongly_nonlinear_at_1pct_of_gdp`,
`test_monetary_nonlinearity_at_100bp_annualised_is_small`).

## Output units and `r`

Units: `Y`, `C`, `I` are % deviations from steady state; `pi`, `r` and `tax`
are quarterly-rate/rate deviations in percentage points. Multiply `pi` and `r`
by 4 for annualised figures: the −25bp-per-quarter example raises inflation by
0.08pp a quarter on impact, 0.32pp annualised. `size` is always a
LEVEL change in the driving variable — see the table above.

`r` is the ex-post real
return on bonds held into the quarter, r_t = (1 + i_{t−1})/(1 + π_t) − 1. On
impact it moves only with the inflation surprise (r_0 = −(1 + r)·π_0); the
policy move itself shows from quarter 1. For the −25bp-per-quarter intercept cut
(ρ 0.61), `r` falls 0.08pp on impact and 0.18pp in quarter 1, while the
ex-ante real rate i_t − π_{t+1} falls 0.18pp on impact.
