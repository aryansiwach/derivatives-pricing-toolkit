# Derivatives & Options Pricing Toolkit

Closed-form (Black-Scholes-Merton), lattice (CRR binomial), and simulation
(Monte Carlo) option pricing compared on accuracy, convergence, and cost,
extended with Merton jump-diffusion and Heston stochastic-volatility
calibration to real SPY/AAPL implied-vol surfaces, plus an American
early-exercise premium analysis. See `report/report.md` for the full
write-up and findings.

## Summary of findings

Full detail, numbers, and defense-prep Q&A live in `report/report.md`. Headline:

- **H1 (convergence): supported.** Binomial error scales as N^-1.08, Monte
  Carlo standard error as N^-0.53, both close to theory (-1, -0.5) and
  highly significant.
- **H2 (smile): supported**, including under a Newey-West HAC-robust test
  that accounts for correlated strikes (not just plain OLS). Every
  SPY/AAPL expiry shows significant skew and curvature; survives excluding
  the bottom volume quartile.
- **H3 (variance reduction): supported.** Antithetic variates cut Monte
  Carlo standard error 29%, control variates cut it 62%, and neither
  estimator is distinguishable from unbiased at the 1% level.
- **H4 (Merton/Heston vs. flat BS): Merton robustly supported; Heston
  supported but weakly identified.** Both beat flat Black-Scholes
  out-of-sample in every ticker/expiry combination. Both calibrators run
  from three starting points and keep the best fit, specifically so
  parameter instability can't be blamed on an unlucky initial guess.
  Merton finds the same optimum from every start (spread in RMSE near
  zero everywhere). Heston is more interesting: one fit improved with a
  better starting point, but five of six landed on identical parameters
  regardless of where the search began, and the instantaneous variance
  pinned at its lower calibration bound across all eighteen individual
  optimization attempts. That's reported as what it is: real evidence the
  data doesn't identify Heston's five parameters well at this calibration
  size, not an optimizer problem.
- **H5 (early-exercise premium): supported.** ITM puts carry a materially
  larger premium than the rest of the surface, concentrated more in SPY
  (higher dividend yield) than AAPL (lower yield), matching theory. Tested
  two ways: a contract-level test (p=1.9e-7) and a more conservative
  group-level test that treats each ticker/expiry/bucket as one
  observation instead of pooling correlated contracts (p=0.030). Both
  reject the null; the group-level number is the one worth trusting.

## Setup

```bash
python -m venv venv
venv\Scripts\pip install -r requirements.txt          # Windows
# source venv/bin/activate && pip install -r requirements.txt   # macOS/Linux
```

Data pulls need network access (Yahoo Finance via `yfinance`, and FRED via
`pandas_datareader` for the risk-free rate). No API keys required.

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

1. **Data pull**: `python src/data_loader.py`. Pulls SPY and AAPL option
   chains (three expiries each: near/mid/long), spot price, trailing
   dividend yield, and the FRED 3-month T-bill rate. Applies documented
   filters (zero bid, crossed quotes, low open interest, near-expiry,
   extreme moneyness) and caches raw and processed data under `data/`
   with a timestamp, plus a JSON summary sidecar per ticker.
2. **`notebooks/01_eda.ipynb`**: market snapshot, chain composition,
   filter drop counts, and why this toolkit solves its own implied vol
   instead of trusting yfinance's field.
3. **Unit tests** (`pytest`, 83 tests): Black-Scholes against Hull
   textbook values, put-call parity, convergence sanity, edge cases, and
   a mocked test suite for the data pipeline, before anything downstream
   depends on any of it.
4. **`notebooks/02_convergence.ipynb`** (H1): binomial and Monte Carlo
   convergence to Black-Scholes, log-log regression of error against N.
5. **`notebooks/03_smile_calibration.ipynb`** (H2, H4): Brent-solved
   implied-vol surface, smile regression with HAC-robust standard errors,
   and multi-start Merton/Heston calibration with an in-sample/out-of-
   sample strike split and two parameter-stability checks.
6. **`notebooks/04_early_exercise.ipynb`** (H5): American early-exercise
   premium, binomial versus Black-Scholes gap to the market, by moneyness
   bucket, using a non-circular OTM-fitted smile vol with its own reported
   fit diagnostics.
7. **Robustness checks**: illiquid-quartile exclusion
   (`results/tables/robustness_illiquid_quartile_exclusion.csv`),
   multi-seed Monte Carlo variance (`tests/test_monte_carlo.py`,
   notebook 02), and parameter stability across both strike subsets and
   maturities (`results/tables/h4_parameter_stability_same_maturity.csv`,
   `robustness_cross_maturity_stability.csv`).
8. **`report/report.md`**: final write-up. Per-hypothesis verdicts,
   limitations, defense-prep Q&A, a self-audit, and a changelog of what a
   self-review pass found and fixed after the first draft.

## Repo layout

`src/` holds one module per model or concern, `tests/` mirrors it,
`notebooks/` holds the per-hypothesis analysis, `results/` holds saved
figures and tables, and `report/` holds the write-up.

## Known limitations (see report for full discussion)

- Single point-in-time snapshot: one day, one vol regime, not a
  historical panel.
- Continuous dividend yield and continuous-compounding rate conversion
  both approximate discrete real-world quantities.
- Implied vol and calibration use the Black-Scholes (European) formula as
  the inversion model even though SPY/AAPL options are American. Standard
  market-convention simplification, addressed directly by the
  binomial-based early-exercise analysis in notebook 04.
- Monte Carlo in this toolkit prices European payoffs only; American MC
  (Longstaff-Schwartz) is out of scope.
- Heston's five parameters are weakly identified by a 10-12-point
  per-expiry calibration on this data, confirmed rather than assumed by
  running each fit from three starting points (see report section 4/7).
