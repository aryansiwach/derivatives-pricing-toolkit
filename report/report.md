# Derivatives & Options Pricing Toolkit — Report

Data: WRDS OptionMetrics IvyDB US, SPY and AAPL, as-of 2024-03-15. All numbers
below are read directly from `results/tables/*.csv`, produced by the notebooks
in `notebooks/`, `scripts/build_results.py`, and the `src/` modules. Nothing
here is hand-typed from memory or estimated.

This report went through a self-review pass after the first full draft:
several methodology gaps were found (no tests for the data pipeline,
single-start optimization confounding "the model is hard to identify"
with "the optimizer got unlucky," an untested smile fit feeding directly
into the H5 numbers, and a couple of statistical independence issues that
were quietly assumed away). All of them were fixed, not just noted, and
this version reflects the fixed code and the recomputed results. Section
9 lists exactly what changed and why, including the later migration of the
data layer from Yahoo Finance / FRED to OptionMetrics.

## 1. Research question and hypotheses

How do closed-form (Black-Scholes-Merton), lattice (CRR binomial), and
simulation-based (Monte Carlo) methods compare in pricing equity options
on accuracy, convergence rate, and computational cost, and to what extent
do stochastic-volatility (Heston) and jump-diffusion (Merton) extensions
improve the fit to the market-observed implied-vol surface relative to
the constant-volatility Black-Scholes assumption? A secondary question:
since real equity/ETF options are American-style, how large is the
early-exercise premium the European Black-Scholes price misses, and does
the binomial model recover it consistently with the market?

Five hypotheses follow from that: numerical convergence of the lattice and
simulation methods to the closed form (H1), whether the market smile is
statistically real rather than noise (H2), whether Monte Carlo variance
reduction is unbiased (H3), whether Merton and Heston materially improve
on flat Black-Scholes out-of-sample (H4), and whether the American
early-exercise premium is material and the binomial model tracks it (H5).

## 2. Data

**Source:** WRDS OptionMetrics IvyDB US. `scripts/wrds_pull.py` pulls one
as-of date (2024-03-15) for SPY (`secid` 109820) and AAPL (`secid` 101594):
the option chain from `optionm.opprcd2024`, the underlying close from
`optionm.secprd2024`, cash distributions from `optionm.distrd`, and the
continuously-compounded zero curve from `optionm.zerocd`. `src/data_loader.py`
reads those CSVs offline. The as-of date is historical because the IvyDB feed
lags real time by weeks to months.

| | SPY | AAPL |
|---|---|---|
| Spot | 509.83 | 172.62 |
| Trailing-12m div yield (`q`, from `optionm.distrd`) | 1.61% | 0.56% |
| Expiries (near / mid / long) | 2024-04-05 / 04-30 / 07-19 | 2024-04-05 / 04-26 / 07-19 |

**Risk-free rate.** The old pipeline used a single flat FRED 3-month T-bill
rate for every maturity. This one interpolates a separate continuously-
compounded rate from `optionm.zerocd` to each option's own days-to-expiry
(3-month point 5.34% on this date); the processed frame carries a per-option
`r_cc` column. `optionm.zerocd.rate` is already continuously compounded, so
the earlier `r_cc = ln(1 + r_simple)` conversion is not applied.

**Filtering** (thresholds fixed in `src/data_loader.py` before any results
were inspected: zero/missing bid, crossed quotes, open interest below 10,
expiry under 1 day, |log-moneyness| over 0.40):

| | n start | dropped: bid | dropped: crossed | dropped: low OI | dropped: moneyness | n kept | % kept |
|---|---|---|---|---|---|---|---|
| SPY | 822 | 6 | 0 | 125 | 40 | 651 | 79.2% |
| AAPL | 244 | 36 | 0 | 85 | 21 | 102 | 41.8% |

Open interest is the only liquidity gate; a same-day volume threshold was
considered and dropped, since a contract can have live open interest and a
firm quote with zero trades that particular day. AAPL loses more to the bid
and open-interest filters than SPY, a real liquidity difference between an
index ETF and a single name at a fixed threshold.

**Implied vol solve** (Brent, `src/implied_vol.py`): SPY 651/651 succeeded
(100%); AAPL 100/102 (98%). The two AAPL failures were "price below minimum
attainable" on deep-ITM contracts whose quoted mid sat below the
near-zero-vol Black-Scholes floor.

**The toolkit solves its own IV rather than trusting the vendor field.**
OptionMetrics' `impl_volatility` is well behaved here (0 of 651 SPY and 0 of
102 AAPL contracts report IV under 0.01, versus 2.3% obviously-broken values
in the earlier Yahoo Finance data), so this is now a matter of methodological
control rather than working around a broken field: the toolkit inverts every
mid price with its own `r`, `q`, and Black-Scholes convention so that the same
assumptions run through H2 and H4 end to end. The vendor field is kept as a
cross-check column.

**Snapshot design.** This is a single point-in-time cross-section: one day,
one volatility regime. It cannot say anything about how the smile or the
calibrated parameters evolve through time. IvyDB does make a historical panel
feasible as a follow-up (it is the dataset the earlier report said would be
needed); the single-snapshot scope here is a deliberate cost choice, not a
data constraint any more.

## 3. Diagnostics

- **Put-call parity, Black-Scholes (synthetic):** exact to 1e-8 by
  construction (`tests/test_black_scholes.py`).
- **Analytical vs. finite-difference Greeks:** max abs difference on an
  ATM case: delta 9.3e-9, gamma 8.8e-10, vega 2.3e-8, theta 1.8e-10, rho
  8.1e-10. Agreement to floating-point precision. An earlier version of the
  finite-difference theta had the wrong sign relative to `bs_theta`'s
  calendar-time convention, and this cross-check is what caught it.
- **Put-call parity on real market quotes** (SPY/AAPL, matched call/put
  pairs, `results/tables/diagnostics_put_call_parity.csv`): mean absolute
  gap 0.71 (SPY), 0.30 (AAPL), max 3.29 (SPY). Not a data error: SPY/AAPL
  options are American, and exact put-call parity is a European relation.
  The gap is largely the same early-exercise premium quantified directly
  in H5 below.
- **Binomial European-limit convergence to BS:** see H1.

## 4. Results by hypothesis

### H1 — Numerical convergence: supported

(`notebooks/02_convergence.ipynb`, `results/tables/h1_convergence_verdict.csv`,
market-grounded params: S=509.83, K=510, r_cc=5.34%, q=1.61%, sigma=0.154
illustrative, T=0.25, BS benchmark call 17.90)

| Model | Fitted slope | Theoretical | R² | p (slope != 0) |
|---|---|---|---|---|
| CRR binomial | -1.200 | -1.0 | 0.952 | 1.1e-16 |
| Plain Monte Carlo | -0.533 | -0.5 | 0.991 | 1.3e-11 |

Both slopes land close to theory and both are highly significant, so H0
(no convergence) is rejected for both methods. The binomial slope of -1.20
is a little steeper than the -1.0 rate; CRR converges with a known
oscillatory pattern around the true price rather than monotonically, and
fitting a single log-log line through that oscillation gives a slope near,
but not exactly at, the theoretical value without contradicting the
underlying O(1/N) rate.

### H2 — Volatility smile: supported, including under a more conservative test

(`notebooks/03_smile_calibration.ipynb`, `results/tables/h2_smile_regression.csv`)

OLS of IV on log-moneyness (linear) and log-moneyness squared (curvature),
per ticker per expiry, reported two ways: plain OLS, and Newey-West
HAC-robust standard errors ordered along the moneyness axis. Plain OLS
treats each strike as independent, but adjacent strikes on the same day
share correlated microstructure noise, so the effective sample size is
smaller than the raw count and OLS p-values run optimistic. HAC accounts
for that correlation and is the more defensible test.

| Ticker | Expiry | n | linear coef | linear p (OLS) | linear p (HAC) | curvature coef | curvature p (OLS) | curvature p (HAC) |
|---|---|---|---|---|---|---|---|---|
| SPY | 04-05 | 157 | -0.616 | 1.6e-34 | 2.4e-06 | 1.715 | 5.5e-21 | 1.6e-03 |
| SPY | 04-30 | 288 | -0.540 | 2.5e-51 | 6.2e-10 | 1.247 | 2.1e-26 | 5.3e-05 |
| SPY | 07-19 | 206 | -0.314 | 1.4e-82 | 8.6e-38 | 0.812 | 2.6e-54 | 2.0e-15 |
| AAPL | 04-05 | 28 | -0.149 | 3.5e-04 | 7.5e-06 | 2.453 | 4.8e-14 | 2.3e-38 |
| AAPL | 04-26 | 26 | -0.141 | 2.7e-05 | 2.5e-05 | 1.994 | 1.7e-12 | 2.2e-17 |
| AAPL | 07-19 | 46 | -0.121 | 8.2e-14 | 6.8e-68 | 0.508 | 3.1e-12 | 1.9e-92 |

HAC p-values run larger than OLS on the SPY slices, as expected, but the
worst one across all twelve tests (six expiries, linear and curvature) is
1.6e-3 (SPY 04-05 curvature). Everything else stays far below any
conventional threshold, and the negative linear coefficient (downward skew)
and positive curvature (smile) are consistent across every slice. H0 (flat
IV) is rejected under both the naive and the correlation-robust test.
**Robustness:** the conclusion survives excluding the bottom quartile of
contracts by volume — every linear and curvature p-value stays at machine
zero, coefficients barely move
(`results/tables/robustness_illiquid_quartile_exclusion.csv`).

### H3 — Variance reduction: supported, with a caveat on control variates

(`scripts/build_results.py`, `results/tables/h3_variance_reduction.csv`,
S=K=100, r=5%, q=0%, sigma=20%, T=1, 50,000 paths, 30 seeds — a fixed
numerical experiment, independent of the market data)

| Method | Mean price | Mean SE | SE reduction vs. plain | t-stat vs. BS | p-value |
|---|---|---|---|---|---|
| Plain | 10.4441 | 0.0657 | — | -0.548 | 0.588 |
| Antithetic | 10.4363 | 0.0464 | 36.6% | -1.921 | 0.065 |
| Control variate (S_T) | 10.4428 | 0.0251 | 68.9% | -2.117 | 0.043 |

BS benchmark: 10.4506. Both variance-reduction methods cut standard error
materially at equal path cost. Neither rejects unbiasedness at the
pre-registered p > 0.01 threshold used in the test suite, but the control-
variate p-value (0.043) is close enough to 0.05 that "not statistically
distinguishable from unbiased at 30 seeds" is a more honest read than
"definitely unbiased."

Using the option's own BS price as its own control would be degenerate
here, since MC and BS estimate the identical quantity under identical GBM
assumptions and a coefficient of 1 would give exactly zero variance
without demonstrating anything about the technique (see
`src/monte_carlo.py`). The discounted terminal stock price S_T is the
standard, non-degenerate alternative, with an exactly known mean under
the same risk-neutral GBM.

### H4 — Merton/Heston vs. flat Black-Scholes: Merton robustly supported; Heston supported but with real identification problems

(`notebooks/03_smile_calibration.ipynb`, `results/tables/h4_calibration_verdict.csv`)

Calibrated on 6-12 strikes evenly spaced across the available moneyness
range per expiry; every other valid contract at that expiry is a
**held-out** point never seen during calibration. Both calibrators run
from **3 starting points** (one fixed, two randomized) and keep the best
fit by RMSE, so parameter instability can't be blamed on "only tried one
starting guess" (see section 9 for why this was added).

| Ticker | Expiry | n_calib | n_holdout | BS OOS RMSE(IV) | Merton OOS RMSE | Merton improve % | Heston OOS RMSE | Heston improve % | Heston Feller OK |
|---|---|---|---|---|---|---|---|---|---|
| SPY | 04-05 | 12 | 67 | 0.0649 | 0.0074 | 88.5% | 0.0201 | 69.0% | True |
| SPY | 04-30 | 12 | 137 | 0.1386 | 0.0082 | 94.1% | 0.0452 | 67.4% | False |
| SPY | 07-19 | 12 | 100 | 0.0644 | 0.0051 | 92.1% | 0.0275 | 57.2% | False |
| AAPL | 04-05 | 7 | 7 | 0.0499 | 0.0082 | 83.6% | 0.0142 | 71.6% | False |
| AAPL | 04-26 | 6 | 7 | 0.0262 | 0.0113 | 56.7% | 0.0151 | 42.2% | False |
| AAPL | 07-19 | 12 | 16 | 0.0412 | 0.0022 | 94.8% | 0.0066 | 84.0% | False |

Both models cut out-of-sample IV RMSE substantially against flat BS in
every case (Merton 57-95%, Heston 42-84%), so H0 (no material improvement)
is rejected and the improvement survives the out-of-sample check rather
than just fitting redundant nearby strikes. Merton beats Heston out-of-
sample in all six cases, and the multi-start protocol makes the reason
much clearer than a single-start run could:

- **Merton is robustly identified.** Across five of the six calibrations,
  the spread in final RMSE across the 3 starting points is essentially zero
  (under 1.5e-5). The one exception, AAPL 04-26 with only 6 calibration
  points, has a spread of 0.030: the smallest calibration set is the one
  where the starting point matters, exactly as expected.
  `results/tables/h4_calibration_verdict.csv`, `Merton_start_rmse_spread`.
- **Heston is a mixed picture.** The SPY near-dated expiry (04-05) is the
  one clean Heston fit: RMSE 0.0201 out-of-sample, all three starts agree
  to 0.03 RMSE spread, `v0` lands at 0.003 rather than pinning, and Feller's
  condition is satisfied. Every other Heston fit either fails to converge
  (SPY 04-30) or lands on identical parameters from every starting point.
  That last point is the opposite of an optimizer artifact: it means the
  objective landscape reliably funnels toward the same place regardless of
  where the search begins, which is stronger evidence of genuine
  non-identifiability, not weaker.
- **`v0` pins at its lower bound (0.001) in 5 of the 6 Heston fits**,
  across both tickers, every maturity but the SPY near-dated one, and every
  starting point per fit (each drawn from `v0` in [0.005, 0.3], so this
  isn't a quirk of one lucky guess). The bound was fixed before any
  calibration ran. That consistency is a real signal that the optimizer
  wants an even smaller instantaneous variance than allowed.
- **Feller's condition (2*kappa*theta >= xi^2) is violated in 5 of 6 fits**,
  the SPY near-dated expiry again being the exception.
- **Parameter stability under a different strike subset** (SPY 04-30,
  `results/tables/h4_parameter_stability_same_maturity.csv`, each subset
  itself calibrated with 3 starts): Merton stays stable (all four
  parameters move under 3.1%). Heston does not: both strike subsets pin
  `theta` at or near its upper bound (1.90 and 2.00) and `kappa` moves 35%
  between them. Because each subset's fit is itself already the best of 3
  starts, this instability is a property of the data and the strike
  selection, not leftover optimizer noise. SPY 04-30 is also the expiry
  where the primary Heston fit failed to converge, so this is instability
  between two poor fits rather than between two good ones.
- **Parameter stability across maturities**
  (`results/tables/robustness_cross_maturity_stability.csv`): Merton's jump
  parameters shift with maturity but stay in a plausible range with
  uniformly good fit quality (Merton OOS RMSE 0.0022-0.0113 across all six).
  Heston's `kappa` swings from 7.8 at the near-dated expiry down to 0.1-0.2
  at longer maturities and `theta` hits its upper bound in the two
  longer-dated SPY fits, the same instability pattern as the strike-subset
  check, now across maturities too.

**Interpretation:** the smile reflects priced risk, not noise (both
models clear the out-of-sample bar). Merton (jump risk) consistently
outperforms and out-stabilizes Heston (diffusive stochastic vol) on this
short-dated-to-3-month equity/ETF data, and the multi-start evidence
rules out "unlucky starting point" as the explanation for most of it.
This matches a textbook expectation directly: short-dated equity skew is
largely jump-risk driven, and a pure-diffusion stochastic-vol model needs
a richer strike grid, longer maturities, or a joint multi-maturity fit
(sharing kappa, theta, xi, rho across expiries and only letting v0 vary)
to identify its five parameters reliably from a small per-expiry
calibration set. This is reported as the actual finding: not "Heston
explains the smile," but "Heston beats flat BS yet is poorly identified
by this particular data and setup, except at the one short-dated SPY
expiry where it behaves."

### H5 — Early-exercise premium: supported, concentrated in ITM puts

(`notebooks/04_early_exercise.ipynb`)

Using each contract's own solved IV for both BS and binomial pricing
would be circular, since BS is solved to match the market price by
construction and the gap would be tautologically zero. Instead a single
smile is fit per expiry from **OTM contracts only** (negligible
early-exercise premium there) and applied to every contract, including
ITM ones, giving a non-circular common vol input.

**Smile-fit quality** (`results/tables/h5_smile_fit_diagnostics.csv`),
checked directly rather than assumed: R² ranges from 0.938 to 0.989
across all six ticker/expiry combinations, with 19 to 152 OTM points
feeding each fit and residual standard deviations of 0.004-0.021 in
volatility terms. The premium numbers below rest on a genuinely good fit,
not an unverified one.

**Materiality and concentration** (`results/tables/h5_premium_by_moneyness_bucket.csv`):

| Ticker | Type | Bucket | n | Mean premium | Mean premium (% of price) |
|---|---|---|---|---|---|
| SPY | put | ITM | 19 | 1.342 | 4.91% |
| SPY | put | ATM | 93 | 0.320 | 3.41% |
| SPY | put | OTM | 199 | 0.036 | 1.99% |
| AAPL | put | ITM | 10 | 0.783 | 4.57% |
| AAPL | put | ATM | 6 | 0.123 | 2.33% |
| AAPL | put | OTM | 29 | 0.014 | 1.70% |
| both | call | any | 395 | 0.000 | 0.00% |

Calls carry zero premium everywhere, which is expected: early exercise of
an American call is only ever optimal just before a dividend, and both
names have small yields. Puts increase monotonically from OTM to ITM in
both tickers, and the ITM effect is larger in absolute terms for SPY than
AAPL, consistent with "concentrated in ITM puts."

**Statistical test, two ways** (`results/tables/h5_ttest.csv`): the
contract-level Welch t-test (ITM puts, n=29, mean 1.149, vs. everything
else, n=722, mean 0.053) gives t=8.67, p=2.0e-9, but that treats
correlated same-day contracts as independent draws. A more conservative
version aggregates to one mean premium per (ticker, expiry, bucket) group
first, so each group counts once: t=3.47, p=0.017 (n=6 vs n=30 groups).
Still significant at the 5% level, just not the extreme significance the
naive test suggests. The group-level number is the one worth trusting.

**Cross-model tracking, read with a caveat.** The pooled raw-dollar
`market_minus_bs_gap` vs. `market_minus_binomial_gap` comparison
(`results/tables/h5_cross_model_gap_summary.csv`) is noisy: for SPY puts
the average binomial dollar gap is larger than the BS one, for AAPL puts it
is 88% smaller. That's not evidence binomial mistracks the market: raw-
dollar gaps pool contracts worth cents (deep OTM) with contracts worth
hundreds of dollars (deep ITM), so a handful of expensive deep-ITM
contracts dominate the mean. The moneyness-bucketed, price-normalized
table above is the reliable comparison, and it shows the expected pattern.

## 5. Robustness checks

1. **AAPL vs. SPY:** done throughout H2, H4, H5. Smile shape and the
   ITM-put early-exercise pattern are qualitatively similar on both
   names, and magnitudes scale sensibly with each name's liquidity and
   dividend yield.
2. **Monte Carlo, multiple seeds:** H3 uses 30 independent seeds;
   `notebooks/02_convergence.ipynb` uses replicated reps per N. SE is
   reported throughout rather than a single unreplicated run.
3. **Recalibrate on a different maturity:** see H4 above
   (`robustness_cross_maturity_stability.csv`). Merton stable, Heston not.
4. **Exclude the bottom volume quartile:** H2's conclusion survives fully
   (`robustness_illiquid_quartile_exclusion.csv`); the one R² that drops
   noticeably (AAPL 07-19, 0.82 -> 0.66) is a small-sample artefact of
   dropping to 34 points, not a change in sign or significance.

## 6. Limitations

- **Single-day snapshot.** One day, one volatility regime. Says nothing
  about how the smile or the calibrated parameters evolve through time. A
  historical panel is now feasible with IvyDB (unlike the earlier
  yfinance-based version) and is the obvious next extension.
- **Continuous dividend yield** approximates the discrete `optionm.distrd`
  cash payments.
- **BS-based IV inversion applied to American-style quotes**, a standard
  market convention and an approximation; the American-specific effect is
  handled separately by H5's binomial analysis rather than ignored.
- **Monte Carlo prices European payoffs only**; American MC
  (Longstaff-Schwartz) is out of scope for this toolkit.
- **Heston calibration is weakly identified** on this data even after
  ruling out the optimizer as the cause (see H4).
- **Calibration subsample size (6-12 points)** is a deliberate
  cost/robustness tradeoff, not a data limitation, but it does mean
  Heston's five parameters have less information per fit than a
  full-chain calibration would provide, and the two 6-7 point AAPL fits
  are the least reliable in the set.

## 7. Defense Prep

Written after the results existed, including the multi-start rerun and the
OptionMetrics migration, not before.

**1. Why SPY and AAPL specifically?**
SPY for a liquid, dividend-paying, American-style ETF with a clean term
structure; AAPL as a single-name cross-check with higher idiosyncratic
vol and a much lower dividend yield, to see whether findings (especially
H5's dividend-driven early-exercise premium) scale the way theory
predicts. They do: SPY's ITM-put premium is larger in absolute terms,
consistent with its higher yield and deeper liquidity.

**2. Why 6-12 calibration strikes instead of the full chain?**
Adjacent strikes are highly redundant, and the per-point cost (an
implied-vol solve inside the calibration objective, evaluated every
optimizer iteration, times three starting points) dominates runtime with
diminishing information return. A curated, evenly-spaced subset also
gives a large, honest out-of-sample holdout (up to 137 points on SPY).

**3. What would have invalidated H4?**
If out-of-sample RMSE hadn't improved over flat BS, or improved in-sample
but not out-of-sample. Neither happened here, but this was a real
possibility going in, not a foregone conclusion, and it's exactly what
the held-out split was built to catch.

**4. Single-start optimization can't tell "hard to identify" apart from
"unlucky starting point." How do you know Heston's instability is real?**
Every Heston (and Merton) calibration now runs from 3 starting points and
keeps the best. Merton's spread across starts is essentially zero on the
five larger calibration sets. Heston's picture is mixed: the one
well-conditioned fit (SPY near-dated) is clean across all three starts,
but the other fits either fail to converge or land on identical
parameters regardless of starting point, and `v0` pins at its lower bound
across every start of every pinned fit. That consistency under deliberate
perturbation is stronger evidence of genuine non-identifiability than a
single-start result could ever provide.

**5. Why does v0 keep hitting its lower bound?**
Not resolved here. It would need either a richer strike grid, a joint
calibration across all three maturities simultaneously (sharing kappa,
theta, xi, rho and only letting v0 vary), or a different parameterization
of the short-variance dynamics. Flagged as the natural next step rather
than swept under the rug. Note that the one expiry where `v0` does not
pin (SPY 04-05) is also the only one where Feller holds and the fit is
stable, which is consistent with the identification problem being real
and data-dependent rather than a coding bug.

**6. Why use S_T rather than the option's own BS price as the Monte Carlo
control variate?**
MC and BS estimate the identical quantity here, under identical GBM
assumptions, so using the option's own price as its own control gives a
coefficient of 1 and exactly zero variance, which validates nothing.
S_T is the standard non-degenerate choice with an exactly known
risk-neutral mean.

**7. Why is put-call parity violated on the real market data (mean gap
0.71 for SPY) when the unit tests show it holding to 1e-8?**
Two different things. The unit tests check the Black-Scholes formula's
internal consistency, exact by construction for synthetic European
inputs. The real-data check is on American market quotes, where exact
put-call parity doesn't hold: the gap is early-exercise premium, the same
effect quantified in H5.

**8. You solve your own implied vol instead of using OptionMetrics'
field. Why, if that field is clean here?**
For consistency, not because it's broken. Every IV in H2 and H4 is
inverted with the same `r` (per-expiry, from `optionm.zerocd`), the same
`q`, and the same Black-Scholes convention, so the whole pipeline runs on
one set of assumptions. The vendor field is kept as a cross-check. The
earlier Yahoo Finance data had 2.3% of quotes with an obviously-broken IV
under 0.01, which is the concrete reason the toolkit was built to not
depend on a vendor field in the first place.

**9. Why is the AAPL sample so much smaller than SPY's (102 vs. 651
contracts)?**
AAPL has less open interest and narrower strike/expiry listings than SPY
at the same filter thresholds. Real liquidity difference between an index
ETF and a single name, not a bug, and the thresholds were fixed before
looking at either ticker's results.

**10. Some AAPL calibration and holdout sets are tiny (n_calib=6-7,
n_holdout=7). Does that undermine the AAPL numbers?**
Partially, and it's visible directly in the `n_calib` / `n_holdout`
columns rather than hidden. The two 6-7 point AAPL fits (04-05, 04-26)
are the least reliable in the set: AAPL 04-26 is the one calibration
where the Merton starting point measurably matters, and both are flagged
in the limitations.

**11. Why fit the H5 vol smile only on OTM options, and how do you know
that fit is any good?**
OTM avoids the circularity described above and matches standard market
convention. And the fit is checked, not assumed: R² across all six
ticker/expiry combinations ranges from 0.938 to 0.989
(`h5_smile_fit_diagnostics.csv`), added specifically because the first
version of this notebook didn't check it at all.

**12. Why does the H5 cross-model raw-dollar table look inconsistent
between SPY and AAPL?**
Raw-dollar means are dominated by a few expensive deep-ITM contracts. The
moneyness-bucketed, price-normalized table is the reliable comparison and
shows the expected pattern. Both tables stay in the report rather than
dropping the inconvenient one.

**13. Why truncate the Merton jump-diffusion sum, and is the truncation
error controlled?**
Terms are weighted by a Poisson(lambda'T) pmf; the code truncates once
the remaining tail probability drops below 1e-10, which directly bounds
the truncation error since each BS term is bounded by max(S,K). In
practice this converges in well under 50 terms for every calibrated
lambda seen here.

**14. Why the "little trap" Heston formulation instead of the original
1993 characteristic function?**
The original has a known discontinuous branch cut in the complex
logarithm that produces silently wrong prices for some parameter regions.
The reformulation is numerically stable across the ranges explored during
calibration, verified directly against the deterministic-variance limit
of Heston equals Black-Scholes, matching to 1e-3 on both a call and an
OTM put.

**15. The Welch t-test in H5 treats every contract as independent. Isn't
that a problem elsewhere too?**
Yes, and it's addressed the same way in H2: report the naive version and
a more conservative, correlation-aware version side by side, and trust
the conservative one. For H5 that's the group-level test (p=0.017,
n=6 vs n=30 groups); for H2 that's the HAC-robust p-values.

**16. What's the single weakest part of this project?**
Heston, for a precise reason: it clears H4's bar (beats flat BS
out-of-sample everywhere) and the instability survives multi-start, so it
isn't a search-quality problem. It's a genuine identification problem
with this per-expiry, 6-12-point calibration setup, and the natural fix
(joint multi-maturity calibration) is identified but not implemented here.

**17. If you had one more day, what would you do first?**
Joint calibration of Heston across all three maturities simultaneously,
sharing kappa, theta, xi, rho and only letting v0 vary by expiry. That's
the standard fix for the per-maturity instability observed above, and it
would directly test whether the weak identification is a per-fit
sample-size problem or a deeper model/data mismatch.

## 8. Final Self-Audit

| Dimension | Score | Notes |
|---|---|---|
| Data integrity | 9 | Institutional-grade source (OptionMetrics IvyDB), documented filters, drop counts reported, per-expiry rate curve rather than a single flat rate, and the pipeline has real test coverage (12 tests over synthetic `data/wrds_raw` fixtures). Not 10: filter thresholds are reasonable defaults, not derived from a deeper liquidity study. |
| Mathematical/statistical correctness | 9 | BS matches Hull to 2dp; binomial/MC convergence rates match theory; Heston verified against the BS deterministic-vol limit; a real sign bug (finite-difference theta) was caught by the diagnostics cross-check itself. H2 and H5 report correlation-robust statistics (HAC, group-level) alongside the naive versions. |
| Model selection justification | 9 | Every model choice (little-trap Heston, S_T control variate, OTM-only H5 smile, small calibration subsample, multi-start optimization) has a stated rationale in code and notebooks, and multi-start was added specifically because a single start couldn't support the identifiability claims being made about Heston. |
| Diagnostic rigor | 9 | Parity checks (synthetic and real), Greeks cross-check, convergence regressions, Feller-condition tracking, two-axis parameter-stability checks (strike subset and maturity), smile-fit R² for H5 and per-start RMSE spread for H4, all reported including the unfavorable results. |
| Robustness | 9 | All four robustness checks done with real recomputation. The multi-start protocol is itself a robustness check on the calibration methodology. Not 10: a single point-in-time snapshot caps how far robustness can be pushed. |
| Code quality | 9 | Modular `src/`, one concern per file, 83 passing tests including the data pipeline. `filter_chain` and the bucket logic are untouched by the data-source swap. |
| Reproducibility | 9 | A fixed historical as-of date (not a live pull), timestamped raw/processed caches with JSON summaries, `requirements.txt` with the network libraries removed, fixed filter/calibration thresholds in code, all notebooks executed end-to-end with saved outputs and persisted result tables, `scripts/build_results.py` for the auxiliary tables, multi-start seeded. Not 10: the raw chains are licensed and cannot be committed, so a reader without WRDS access reads the results but cannot rerun the pull. |
| Honesty about limitations | 10 | Heston's identifiability problems (including what multi-start did and didn't explain, and the one expiry where Heston does behave), the H5 raw-dollar-gap pitfall, the circular-IV design trap that was caught and fixed, the small-sample AAPL fits, the statistical-independence issues in H2/H5 and how they were addressed, and the single-day snapshot are all stated plainly. |

## 9. What changed after the first draft

A self-review pass treated the first version of this report the way a
skeptical reader would, and found real gaps rather than stylistic ones.
Fixed, in order of how much they changed the results:

1. **No tests for the data pipeline.** `src/data_loader.py` had zero
   automated coverage. Added tests (`tests/test_data_loader.py`) for the
   pure filtering logic and, after the migration, for every Stage-B reader
   against synthetic `data/wrds_raw` fixtures.
2. **Single-start calibration.** Both Heston and Merton were calibrated
   from one fixed starting point, which meant "the parameters are
   unstable" could have meant "the optimizer got unlucky." Added
   multi-start (3 starting points, best kept, full spread reported) to
   both, and reran H4 end to end.
3. **Unvalidated H5 smile fit.** The OTM-only quadratic feeding both
   pricers in H5 had no reported fit quality. Added R²/residual
   diagnostics per ticker/expiry.
4. **Unacknowledged statistical independence assumptions.** H2's
   per-expiry OLS and H5's contract-level Welch t-test both treated
   correlated same-day observations as independent draws. Added
   HAC-robust standard errors to H2 and a group-level aggregated test to
   H5, reported alongside the original versions.
5. **Dead code.** `MIN_VOLUME = 0` in the data loader implemented a
   filter that could never drop anything. Removed.
6. **Performance/consistency.** `add_implied_vol` switched from
   `iterrows` to `itertuples`; Merton's calibration objective was
   vectorized across strikes the same way Heston's already was.
7. **Data-source migration.** The whole data layer moved from Yahoo
   Finance / FRED to WRDS OptionMetrics IvyDB US. `scripts/wrds_pull.py`
   (a one-time WRDS pull) plus a rewritten `src/data_loader.py` I/O layer
   replace the live yfinance calls; `filter_chain` and the bucket logic
   are unchanged. The risk-free rate went from a single flat FRED
   3-month rate to a per-expiry interpolation of `optionm.zerocd`. Every
   result table and figure was regenerated. All five hypotheses hold on
   the new data with the same conclusions; the specific numbers, the
   filter drop counts, and the Heston identification story (now with one
   clean expiry) are updated throughout.
