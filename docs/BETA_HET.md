# Opt-in variant: discount-factor heterogeneity (issue #8)

[← README](../README.md)

`us_hank.beta_het` is a second, separate DAG for users who need consumption
responses to transfers that match the rebate evidence. **It is not the
default**: `us_hank.model` and every number in the README and the other docs are the published
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
| Hand-to-mouth share, poor / wealthy (production grid) | 0.537 (0.019 / 0.518) | 0.572 (0.203 / 0.369) |
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
assets, and Kaplan & Violante (2014) use a single β of 0.941 a year. Sensitivity (CI run 42145287339;
measured once, not gated by a test): an impatient mass of 1/4 or 1/2 needs
β gaps of 0.160 or 0.099 and gives impact multipliers 0.403 and 0.385;
halving the type-switch rate moves the $1,000 MPC 0.200 → 0.207.

## Second calibration: median wealth targets (`liquid_target='median'`)

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
| Poor / all hand-to-mouth (b, a < 0.05; refined grid) | 0.019 / 0.546 | 0.204 / 0.579 | 0.060 / 0.450 | 0.121 / 0.312 |
| MPC $500 / $1,000 / $1,500 | 0.104 / 0.091 / 0.083 | 0.227 / 0.200 / 0.186 | 0.126 / 0.110 / 0.101 | 0.15–0.25 at $500 |
| G ρ 0.8: impact / 20q cumulative multiplier | 0.175 / −0.015 | 0.376 / 0.309 | 0.274 / 0.213 | |
| G ρ 0.8 / 0.9: first quarter with Y < 0 | q3 / q3 | q23 / none | none / none (40q) | |
| Monetary −25bp/q: impact Y / C / I (%) | 0.493 / 0.300 / 1.698 | 0.244 / 0.308 / 0.478 | 0.360 / 0.240 / 1.191 | |

The moments are measured on the refined liquid grid, which is why the
default's hand-to-mouth share reads 0.546 here and 0.537 (production grid)
elsewhere. The hand-to-mouth cutoff is 0.05 in model units, 7.6% of mean
quarterly pre-tax earnings: about $1,000 at the 2001 SCF earnings the wealth
targets are scaled by, and about $2,900 at the GDP-per-household scale the
MPC rows use. The two dollar scales differ by a factor of 2.9 and are not
interchangeable. Both medians come close
(illiquid 13% high). The β points are not inside the Carroll et al. range:
the top point, 0.9875, is above 0.978, and the uniform distribution they
discretise runs from 0.932 to 0.998 (Carroll et al.: 0.936–0.978). β levels
are not comparable across models with different returns either; the top
type's β(1 + r) is 0.9998. Poor hand-to-mouth (6%, half the data) and the
MPC (0.126) stay below their targets, and wealthy hand-to-mouth (39%) is
twice the data (19.2%). With total wealth held at 14 they could not be
pushed further (measured once, not gated by a test): a wider spread (0.034), or a fourth, impatient type with
12–15% of households, gave no root for the centre β in the solver's
bracket, which keeps the most patient type at least 1e-4 below 1/(1+r).
That is a property of the bracket, not a proof that no steady state exists:
with types redrawn every 200 quarters on average, a top β at or slightly
above 1/(1+r) need not be explosive, and dropping the total wealth target,
as KV do, was not tried. The fit is knife-edge: spread 0.033 solves and
0.034 does not, with the top type 2e-4 below the cap holding 73% of
illiquid wealth. The MPC is quoted per household at US GDP of about $31T a year and
133M households ($58,000 a quarter per household); KV's $500 in 2001 is 3.8% of quarterly earnings, which
here is about $1,460, where the MPC is about 0.10. χ0 between 0.02 and
0.25 moves poor hand-to-mouth by under 1 point (measured once, not gated by
a test). The spending-multiplier
sign result survives: output stays positive for 40 quarters at ρ 0.8 and
0.9 without the extreme impatient type. The $500 MPC rises only from 0.104
to 0.126, so the average MPC alone does not explain the sign change; the
rest comes from the wealth distribution (not decomposed here). This
calibration is a median-wealth sensitivity, not a higher-MPC one.
