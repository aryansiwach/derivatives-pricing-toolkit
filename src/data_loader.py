"""
Options chain acquisition and cleaning.

Data sources:
- yfinance: spot price, dividend history, options chains (bid/ask/volume/OI)
- FRED (DGS3MO via pandas_datareader): 3-month T-bill secondary market rate

Filter thresholds below are fixed in code, not tuned against results (see
no p-hacking). If they need to change, that is a
deliberate methodology change, documented in the report, not a silent
after-the-fact adjustment.
"""
from __future__ import annotations

import datetime as dt
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf
import pandas_datareader.data as web

REPO_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = REPO_ROOT / "data" / "raw"
PROCESSED_DIR = REPO_ROOT / "data" / "processed"

# --- documented filter thresholds, fixed before inspecting any results ---
MIN_OPEN_INTEREST = 10          # below this, quote is illiquid / IV unreliable.
                                 # This is the sole liquidity gate: a same-day
                                 # volume>=k threshold was considered and
                                 # deliberately dropped, since a contract can
                                 # legitimately have zero trades on the pull
                                 # day yet still carry live open interest and a
                                 # firm two-sided quote -- open interest is the
                                 # more meaningful liquidity signal for a
                                 # single-day snapshot than same-day volume.
MAX_ABS_LOG_MONEYNESS = 0.40    # drop |ln(K/S)| > 0.40 (~K/S outside [0.67, 1.49]):
                                 # deep ITM/OTM contracts have near-zero vega,
                                 # so IV inversion is numerically unstable and
                                 # tiny bid-ask noise maps to huge IV swings
MIN_DAYS_TO_EXPIRY = 1           # T must be strictly positive and not a same-day
                                 # expiry, where BS assumptions break down


def fetch_spot(ticker: str) -> float:
    t = yf.Ticker(ticker)
    fi = t.fast_info
    price = fi.get("lastPrice") if hasattr(fi, "get") else None
    if price is None:
        price = fi["lastPrice"]
    return float(price)


def fetch_dividend_yield(ticker: str, spot: float, asof: dt.date | None = None) -> dict:
    """Trailing-12-month dividends / spot, used as a continuous yield q.

    This is an approximation: real dividends are discrete lump payments,
    not a continuous flow. Documented explicitly here and in the report
    as a limitation.
    """
    asof = asof or dt.date.today()
    t = yf.Ticker(ticker)
    divs = t.dividends  # pandas Series indexed by tz-aware Timestamp
    trailing_sum = 0.0
    if divs is not None and len(divs) > 0:
        idx = divs.index.tz_localize(None) if divs.index.tz is not None else divs.index
        divs = pd.Series(divs.values, index=idx)
        cutoff = pd.Timestamp(asof) - pd.Timedelta(days=365)
        trailing = divs[(divs.index > cutoff) & (divs.index <= pd.Timestamp(asof))]
        trailing_sum = float(trailing.sum())
    q_trailing = trailing_sum / spot if spot > 0 else 0.0

    info_yield = None
    try:
        info = t.info
        raw = info.get("dividendYield")
        if raw is not None:
            # yfinance has historically been inconsistent about percent vs
            # decimal here; normalize by magnitude.
            info_yield = raw / 100.0 if raw > 1.0 else float(raw)
    except Exception:
        info_yield = None

    return {
        "ticker": ticker,
        "trailing_12m_dividends": trailing_sum,
        "q_trailing_12m": q_trailing,
        "q_info_field": info_yield,
        "q_used": q_trailing,
        "note": "continuous-yield approximation of discrete dividends; see report limitations",
    }


def fetch_risk_free_rate(asof: dt.date | None = None) -> dict:
    """3-month T-bill secondary market rate (FRED DGS3MO), converted to
    continuous compounding: r_cc = ln(1 + r_simple)."""
    asof = asof or dt.date.today()
    start = asof - dt.timedelta(days=14)
    df = web.DataReader("DGS3MO", "fred", start, asof)
    df = df.dropna()
    if df.empty:
        raise RuntimeError("No DGS3MO observations returned from FRED in lookback window")
    last_obs_date = df.index[-1].date()
    r_simple_pct = float(df.iloc[-1, 0])
    r_simple = r_simple_pct / 100.0
    r_cc = float(np.log(1.0 + r_simple))
    return {
        "asof_requested": str(asof),
        "fred_obs_date": str(last_obs_date),
        "r_simple_pct": r_simple_pct,
        "r_simple": r_simple,
        "r_cc": r_cc,
        "conversion": "r_cc = ln(1 + r_simple)",
    }


def select_expiries(ticker: str, asof: dt.date | None = None) -> dict:
    """Pick 3 expiries spanning near (~2-4wk), mid (~1-2mo), long (~3mo+)."""
    asof = asof or dt.date.today()
    t = yf.Ticker(ticker)
    all_expiries = [dt.datetime.strptime(e, "%Y-%m-%d").date() for e in t.options]
    if not all_expiries:
        raise RuntimeError(f"No option expiries available for {ticker}")

    def days_out(e):
        return (e - asof).days

    buckets = {
        "near": (14, 28),
        "mid": (29, 62),
        "long": (63, 10_000),
    }
    chosen = {}
    for label, (lo, hi) in buckets.items():
        candidates = [e for e in all_expiries if lo <= days_out(e) <= hi]
        if candidates:
            # pick the one closest to the bucket midpoint for a representative pick
            mid_target = (lo + min(hi, 200)) / 2
            chosen[label] = min(candidates, key=lambda e: abs(days_out(e) - mid_target))

    missing = [label for label in buckets if label not in chosen]
    return {
        "ticker": ticker,
        "asof": str(asof),
        "all_expiries": [str(e) for e in all_expiries],
        "chosen": {k: str(v) for k, v in chosen.items()},
        "missing_buckets": missing,
    }


def fetch_raw_chain(ticker: str, expiries: list[str]) -> pd.DataFrame:
    """Pull raw calls+puts for the given expiries, tag with type/expiry, no filtering."""
    t = yf.Ticker(ticker)
    frames = []
    for exp in expiries:
        oc = t.option_chain(exp)
        calls = oc.calls.copy()
        calls["option_type"] = "call"
        puts = oc.puts.copy()
        puts["option_type"] = "put"
        for df in (calls, puts):
            df["expiry"] = exp
            df["ticker"] = ticker
        frames.append(calls)
        frames.append(puts)
    raw = pd.concat(frames, ignore_index=True)
    return raw


def filter_chain(raw: pd.DataFrame, spot: float, asof: dt.date | None = None) -> tuple[pd.DataFrame, dict]:
    """Apply documented filters, return (filtered_df, drop_report).

    Filters applied in order, each counted independently against the
    incoming set at that stage (sequential, not overlapping double-counts).
    """
    asof = asof or dt.date.today()
    df = raw.copy()
    n0 = len(df)
    report = {"n_start": n0}

    df["mid"] = (df["bid"] + df["ask"]) / 2.0

    # 1. zero/missing bid
    mask = df["bid"] > 0
    report["dropped_zero_or_missing_bid"] = int((~mask).sum())
    df = df[mask]

    # 2. crossed/locked quote (bid > ask)
    mask = df["ask"] >= df["bid"]
    report["dropped_bid_gt_ask"] = int((~mask).sum())
    df = df[mask]

    # 3. illiquid: open interest below threshold
    mask = df["openInterest"].fillna(0) >= MIN_OPEN_INTEREST
    report["dropped_low_open_interest"] = int((~mask).sum())
    df = df[mask]

    # 4. time to expiry too small
    exp_dates = pd.to_datetime(df["expiry"]).dt.date
    days_to_exp = exp_dates.apply(lambda e: (e - asof).days)
    df = df.assign(days_to_expiry=days_to_exp.values)
    mask = df["days_to_expiry"] >= MIN_DAYS_TO_EXPIRY
    report["dropped_too_close_to_expiry"] = int((~mask).sum())
    df = df[mask]

    # 5. deep ITM/OTM (IV inversion numerically unstable here)
    log_moneyness = np.log(df["strike"] / spot)
    df = df.assign(log_moneyness=log_moneyness.values)
    mask = df["log_moneyness"].abs() <= MAX_ABS_LOG_MONEYNESS
    report["dropped_extreme_moneyness"] = int((~mask).sum())
    df = df[mask]

    df = df.assign(T_years=df["days_to_expiry"] / 365.0)

    report["n_end"] = len(df)
    report["n_dropped_total"] = n0 - len(df)
    report["pct_kept"] = round(100.0 * len(df) / n0, 1) if n0 else 0.0

    keep_cols = [
        "ticker", "expiry", "option_type", "strike", "bid", "ask", "mid",
        "lastPrice", "volume", "openInterest", "impliedVolatility",
        "days_to_expiry", "T_years", "log_moneyness", "contractSymbol",
    ]
    keep_cols = [c for c in keep_cols if c in df.columns]
    df = df[keep_cols].reset_index(drop=True)

    return df, report


def pull_and_process(ticker: str, out_tag: str | None = None) -> dict:
    """Full pipeline for one underlying: spot, dividend yield, rate, chain,
    filter, and save both raw and processed data with a timestamp."""
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    asof = dt.date.today()
    ts = dt.datetime.now().strftime("%Y%m%dT%H%M%S")
    tag = out_tag or ts

    spot = fetch_spot(ticker)
    div_info = fetch_dividend_yield(ticker, spot, asof)
    rate_info = fetch_risk_free_rate(asof)
    exp_info = select_expiries(ticker, asof)
    expiries = list(exp_info["chosen"].values())
    if len(expiries) < 3:
        print(f"WARNING: {ticker} only had {len(expiries)} expiry buckets filled: "
              f"{exp_info['chosen']}, missing {exp_info['missing_buckets']}")

    raw = fetch_raw_chain(ticker, expiries)
    raw_path = RAW_DIR / f"{ticker}_raw_{tag}.csv"
    raw.to_csv(raw_path, index=False)

    filtered, drop_report = filter_chain(raw, spot, asof)
    filtered["spot"] = spot
    filtered["q"] = div_info["q_used"]
    filtered["r_cc"] = rate_info["r_cc"]
    processed_path = PROCESSED_DIR / f"{ticker}_processed_{tag}.csv"
    filtered.to_csv(processed_path, index=False)

    summary = {
        "ticker": ticker,
        "asof": str(asof),
        "spot": spot,
        "dividend_yield": div_info,
        "risk_free_rate": rate_info,
        "expiry_selection": exp_info,
        "raw_path": str(raw_path),
        "processed_path": str(processed_path),
        "filter_report": drop_report,
    }

    import json
    summary_path = PROCESSED_DIR / f"{ticker}_summary_{tag}.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    summary["summary_path"] = str(summary_path)

    latest_path = PROCESSED_DIR / f"{ticker}_latest.json"
    with open(latest_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)

    return summary


def latest_pull(ticker: str) -> dict:
    """Load the most recent pull's summary + processed dataframe."""
    import json
    latest_path = PROCESSED_DIR / f"{ticker}_latest.json"
    with open(latest_path) as f:
        summary = json.load(f)
    df = pd.read_csv(summary["processed_path"])
    return summary, df


if __name__ == "__main__":
    import json
    for tkr in ["SPY", "AAPL"]:
        result = pull_and_process(tkr)
        print(json.dumps(result, indent=2, default=str))
