# Derivatives & Options Pricing Toolkit

Closed-form (Black-Scholes-Merton), lattice (CRR binomial), and simulation
(Monte Carlo) option pricing compared on accuracy, convergence, and cost,
extended with Merton jump-diffusion and Heston stochastic-volatility
calibration to real SPY/AAPL implied-vol surfaces, plus an American
early-exercise premium analysis. Data is WRDS OptionMetrics IvyDB US
(SPY and AAPL, as-of 2024-03-15). See `report/report.md` for the full
write-up and findings.

## Summary of findings

Full detail, numbers, and defense-prep Q&A live in `report/report.md`. Headline:

- **H1 (convergence): supported.** Binomial error scales as N^-1.20, Monte
  Carlo standard error as N^-0.53, both close to theory (-1, -0.5) and
  highly significant.
- **H2 (smile): supported**, including under a Newey-West HAC-robust test
  that accounts for correlated strikes (not just plain OLS). Every
  SPY/AAPL expiry shows significant skew and curvature (worst HAC p-value
  1.6e-3); survives excluding the bottom volume quartile.
- **H3 (variance reduction): supported.** Antithetic variates cut Monte
  Carlo standard error 37%, control variates cut it 69%, and neither
  estimator is distinguishable from unbiased at the 1% level.
- **H4 (Merton/Heston vs. flat BS): Merton robustly supported; Heston
  supported but weakly identified.** Both beat flat Black-Scholes
  out-of-sample in every ticker/expiry combination (Merton 57-95%, Heston
  42-84% lower IV RMSE). Both calibrators run from three starting points
  and keep the best fit, so parameter instability can't be blamed on an
  unlucky initial guess. Merton finds the same optimum from every start on
  the five larger calibration sets. Heston has exactly one clean fit (SPY
  near-dated: stable across starts, Feller satisfied, `v0` not pinned);
  the other five either fail to converge or land on identical parameters
  from every start with `v0` pinned at its lower bound. Reported as what it
  is: real evidence the data doesn't identify Heston's five parameters
  well at this calibration size, not an optimizer problem.
- **H5 (early-exercise premium): supported.** ITM puts carry a materially
  larger premium than the rest of the surface (SPY 4.9% of price, AAPL
  4.6%), while calls carry none. Tested two ways: a contract-level test
  (p=2e-9) and a more conservative group-level test that treats each
  ticker/expiry/bucket as one observation (p=0.017). Both reject the null;
  the group-level number is the one worth trusting.

## Setup

```bash
python -m venv venv
venv\Scripts\pip install -r requirements.txt          # Windows
# source venv/bin/activate && pip install -r requirements.txt   # macOS/Linux
```

The option data is WRDS OptionMetrics IvyDB US, pulled once by
`scripts/wrds_pull.py` (needs an authenticated WRDS session). Everything
downstream reads local CSVs -- no network, no API keys. `data/` is git-ignored;
the OptionMetrics licence does not permit redistribution.

```bash
python scripts/wrds_pull.py --asof 2024-03-15
```

Pick a date you know IvyDB has loaded -- the WRDS feed lags real time by weeks
to months. This writes `data/wrds_raw/` (SPY and AAPL chains, spot, dividends,
and the `optionm.zerocd` zero curve).

To run the notebooks from the command line:

```bash
venv\Scripts\python -m ipykernel install --user --name derivtoolkit --display-name "Python (derivtoolkit)"
venv\Scripts\python -m jupyter nbconvert --to notebook --execute --inplace ^
    --ExecutePreprocessor.kernel_name=derivtoolkit notebooks\01_eda.ipynb
```

Repeat per notebook. `03_smile_calibration.ipynb` is the slow one: it runs
Merton and Heston calibration, each from three starting points, across six
ticker/expiry combinations plus a stability check, against a
characteristic-function pricer. Budget 30-45 minutes for that one.

## Execution order

1. **WRDS pull** (once): `python scripts/wrds_pull.py --asof <date>` -> `data/wrds_raw/`.
2. **Process**: `python src/data_loader.py`. Reads the pull, picks three
   expiries per ticker (near/mid/long), applies the documented filters (zero
   bid, crossed quotes, low open interest, near-expiry, extreme moneyness),
   attaches spot, trailing dividend yield, and a continuously-compounded rate
   interpolated from `optionm.zerocd` to each option's own days-to-expiry.
   Caches raw and processed data under `data/` with a timestamp plus a JSON
   summary sidecar per ticker.
3. **`notebooks/01_eda.ipynb`**: market snapshot, chain composition,
   filter drop counts, and why this toolkit solves its own implied vol
   instead of trusting the OptionMetrics field.
4. **Unit tests** (`pytest`, 83 tests): Black-Scholes against Hull
   textbook values, put-call parity, convergence sanity, edge cases, and
   a mocked test suite for the data pipeline, before anything downstream
   depends on any of it.
5. **`notebooks/02_convergence.ipynb`** (H1): binomial and Monte Carlo
   convergence to Black-Scholes, log-log regression of error against N.
6. **`notebooks/03_smile_calibration.ipynb`** (H2, H4): Brent-solved
   implied-vol surface, smile regression with HAC-robust standard errors,
   and multi-start Merton/Heston calibration with an in-sample/out-of-
   sample strike split and two parameter-stability checks.
7. **`notebooks/04_early_exercise.ipynb`** (H5): American early-exercise
   premium, binomial versus Black-Scholes gap to the market, by moneyness
   bucket, using a non-circular OTM-fitted smile vol with its own reported
   fit diagnostics.
8. **Robustness checks**: illiquid-quartile exclusion
   (`results/tables/robustness_illiquid_quartile_exclusion.csv`),
   multi-seed Monte Carlo variance (`tests/test_monte_carlo.py`,
   notebook 02), and parameter stability across both strike subsets and
   maturities (`results/tables/h4_parameter_stability_same_maturity.csv`,
   `robustness_cross_maturity_stability.csv`).
9. **`report/report.md`**: final write-up. Per-hypothesis verdicts,
   limitations, defense-prep Q&A, a self-audit, and a changelog of what a
   self-review pass found and fixed after the first draft.

## Repo layout

`src/` holds one module per model or concern, `tests/` mirrors it,
`notebooks/` holds the per-hypothesis analysis, `results/` holds saved
figures and tables, and `report/` holds the write-up.

## Known limitations (see report for full discussion)

- Single point-in-time snapshot: one day, one vol regime. A historical
  panel is feasible with IvyDB and is the obvious next extension.
- Continuous dividend yield approximates the discrete `optionm.distrd`
  cash payments.
- Implied vol and calibration use the Black-Scholes (European) formula as
  the inversion model even though SPY/AAPL options are American. Standard
  market-convention simplification, addressed directly by the
  binomial-based early-exercise analysis in notebook 04.
- Monte Carlo in this toolkit prices European payoffs only; American MC
  (Longstaff-Schwartz) is out of scope.
- Heston's five parameters are weakly identified by a 6-12-point
  per-expiry calibration on this data, confirmed rather than assumed by
  running each fit from three starting points (see report section 4/7).
- The raw OptionMetrics chains are licensed and not committed; `data/` is
  git-ignored. `results/` (the aggregate tables and figures) is committed.
