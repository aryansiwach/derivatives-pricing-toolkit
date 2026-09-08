"""Regenerate the auxiliary result tables that the notebooks do not write
directly: real-quote put-call parity, the analytic-vs-finite-difference Greeks
cross-check, the H3 Monte Carlo variance-reduction experiment, and the two
robustness checks (illiquid-quartile exclusion for H2, cross-maturity
parameter stability for H4).

Run after processing the data (`python src/data_loader.py`) and after
`notebooks/03_smile_calibration.ipynb` (this reads its
`results/tables/h4_calibrated_params.csv`).

    python scripts/build_results.py
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
warnings.filterwarnings("ignore")

from black_scholes import bs_price  # noqa: E402
from data_loader import latest_pull  # noqa: E402
from diagnostics import greeks_cross_check, put_call_parity_check  # noqa: E402
from implied_vol import add_implied_vol  # noqa: E402
from monte_carlo import (  # noqa: E402
    mc_price_antithetic, mc_price_control_variate, mc_price_plain,
)

TABLES = ROOT / "results" / "tables"


def real_quote_parity():
    frames = []
    for tkr in ("SPY", "AAPL"):
        _, df = latest_pull(tkr)
        p = put_call_parity_check(df)
        frames.append(p)
        print(f"  {tkr} parity: mean|gap| {p['parity_gap'].abs().mean():.3f}, "
              f"max|gap| {p['parity_gap'].abs().max():.3f}  (n={len(p)})")
    out = pd.concat(frames, ignore_index=True)
    out.to_csv(TABLES / "diagnostics_put_call_parity.csv", index=False)


def greeks_check():
    d = greeks_cross_check(S=100.0, K=100.0, r=0.03, q=0.01, sigma=0.2, T=0.5)
    diff = d["max_abs_diff"]
    pd.DataFrame([diff]).to_csv(TABLES / "diagnostics_greeks_cross_check.csv",
                                index=False)
    print("  greeks max abs diff:", {k: f"{v:.1e}" for k, v in diff.items()})


def h3_variance_reduction():
    """Fixed-parameter numerical experiment (independent of the market data):
    S=K=100, r=5%, q=0, sigma=20%, T=1y, 30 seeds x 50k paths."""
    S = K = 100.0
    r, q, sigma, T = 0.05, 0.0, 0.2, 1.0
    n, seeds = 50_000, list(range(30))
    bs = float(bs_price(S, K, r, q, sigma, T, "call"))

    rows = []
    for name, fn in [("plain", mc_price_plain),
                     ("antithetic", mc_price_antithetic),
                     ("control_variate", mc_price_control_variate)]:
        est = np.array([fn(S, K, r, q, sigma, T, n, "call", s)["price"] for s in seeds])
        se = np.array([fn(S, K, r, q, sigma, T, n, "call", s)["se"] for s in seeds])
        t, p = stats.ttest_1samp(est - bs, 0.0)
        rows.append({
            "method": name, "n_paths": n, "n_seeds": len(seeds),
            "mean_price": est.mean(), "bs_benchmark": bs,
            "mean_error": est.mean() - bs, "mean_reported_se": se.mean(),
            "realized_std_across_seeds": est.std(ddof=1),
            "t_stat_vs_bs": t, "p_value_vs_bs": p,
        })
    df = pd.DataFrame(rows)
    base = df.loc[df.method == "plain", "realized_std_across_seeds"].iloc[0]
    df["se_reduction_pct_vs_plain"] = 100 * (1 - df["realized_std_across_seeds"] / base)
    df.to_csv(TABLES / "h3_variance_reduction.csv", index=False)
    print("  H3 SE reduction:", df.set_index("method")["se_reduction_pct_vs_plain"].round(1).to_dict())


def _smile_regression(grp: pd.DataFrame) -> dict:
    grp = grp.sort_values("log_moneyness")
    X = sm.add_constant(pd.DataFrame({"lm": grp.log_moneyness,
                                      "lm2": grp.log_moneyness ** 2}))
    y = grp["iv_computed"]
    ols = sm.OLS(y, X).fit()
    return {"n": len(grp), "linear_coef": ols.params["lm"],
            "linear_p": ols.pvalues["lm"], "curvature_coef": ols.params["lm2"],
            "curvature_p": ols.pvalues["lm2"], "R2": ols.rsquared}


def robustness_illiquid_exclusion():
    rows = []
    for tkr in ("SPY", "AAPL"):
        _, df = latest_pull(tkr)
        out, _ = add_implied_vol(df)
        out = out[out["iv_success"]]
        for expiry, grp in out.groupby("expiry"):
            rows.append({"ticker": tkr, "dataset": "full", "expiry": expiry,
                         **_smile_regression(grp)})
            cut = grp["volume"].quantile(0.25)
            keep = grp[grp["volume"] > cut]
            if len(keep) >= 6:
                rows.append({"ticker": tkr,
                             "dataset": "excl_bottom_quartile_volume",
                             "expiry": expiry, **_smile_regression(keep)})
    pd.DataFrame(rows).to_csv(
        TABLES / "robustness_illiquid_quartile_exclusion.csv", index=False)
    print(f"  robustness (illiquid exclusion): {len(rows)} rows")


def robustness_cross_maturity():
    src = TABLES / "h4_calibrated_params.csv"
    if not src.exists():
        print("  skip cross-maturity: run notebook 03 first")
        return
    cal = pd.read_csv(src)
    rows = []
    for (tkr, model), g in cal.groupby(["ticker", "model"]):
        g = g.sort_values("expiry")
        params = " || ".join(g["params"].tolist())
        v0_pinned = None
        if model == "Heston":
            v0s = [float(p.split("v0=")[1]) for p in g["params"]]
            v0_pinned = bool(all(abs(v - 0.001) < 1e-6 for v in v0s))
        rows.append({"ticker": tkr, "model": model,
                     "n_expiries": len(g),
                     "expiries": " | ".join(g["expiry"]),
                     "params_by_expiry": params,
                     "v0_all_pinned_at_lower_bound": v0_pinned})
    pd.DataFrame(rows).to_csv(
        TABLES / "robustness_cross_maturity_stability.csv", index=False)
    print(f"  robustness (cross-maturity): {len(rows)} rows")


def main():
    TABLES.mkdir(parents=True, exist_ok=True)
    print("real-quote put-call parity ...");     real_quote_parity()
    print("Greeks cross-check ...");              greeks_check()
    print("H3 variance reduction ...");           h3_variance_reduction()
    print("robustness: illiquid exclusion ...");  robustness_illiquid_exclusion()
    print("robustness: cross-maturity ...");      robustness_cross_maturity()
    print(f"done -> {TABLES}")


if __name__ == "__main__":
    main()
