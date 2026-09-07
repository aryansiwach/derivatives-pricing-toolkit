"""
Options chain acquisition and cleaning.

Data source: WRDS OptionMetrics IvyDB US, pulled by scripts/wrds_pull.py into
data/wrds_raw/. This module reads those CSVs offline -- no network, no API keys.
It replaces an earlier yfinance/FRED pipeline; the downstream schema (the
columns produced by filter_chain and the summary JSON) is unchanged, so the
notebooks and model modules are unaffected by the source swap.

    optionm.opprcd  -> option quotes (best bid/offer, OptionMetrics IV, greeks)
    optionm.secprd  -> underlying close
    optionm.distrd  -> cash distributions (trailing-yield input)
    optionm.zerocd  -> continuously-compounded zero curve, interpolated to each
                       option's own days-to-expiry (a strict upgrade over a
                       single flat FRED 3-month rate)

Filter thresholds below are fixed in code, not tuned against results (see
no p-hacking). Changing them is a deliberate methodology change, documented in
the report, not a silent after-the-fact adjustment.
"""
from __future__ import annotations

import datetime as dt
import json
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
WRDS_RAW_DIR = REPO_ROOT / "data" / "wrds_raw"
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


# --------------------------------------------------------------------------- #
# Stage-B readers: consume scripts/wrds_pull.py output
# --------------------------------------------------------------------------- #

def _raw_dir() -> Path:
    if not (WRDS_RAW_DIR / "pull_meta.json").exists():
        raise FileNotFoundError(
            f"{WRDS_RAW_DIR} has no pull -- run scripts/wrds_pull.py --asof <date>")
    return WRDS_RAW_DIR


def pull_asof() -> dt.date:
    with open(_raw_dir() / "pull_meta.json") as f:
        return dt.date.fromisoformat(json.load(f)["asof"])


def fetch_spot(ticker: str) -> float:
    df = pd.read_csv(_raw_dir() / f"{ticker}_spot.csv")
    return float(df.iloc[0]["close"])


def fetch_dividend_yield(ticker: str, spot: float, asof: dt.date | None = None) -> dict:
    """Trailing-12-month cash distributions / spot, used as a continuous yield q.

    An approximation: real dividends are discrete lump payments, not a
    continuous flow. Documented here and in the report as a limitation. Only
    ordinary cash distributions (distr_type == 1) are counted; special
    distributions and stock splits are excluded.
    """
    asof = asof or pull_asof()
    path = _raw_dir() / f"{ticker}_dividends.csv"
    trailing_sum = 0.0
    if path.exists():
        d = pd.read_csv(path, dtype={"ex_date": str})
        if not d.empty:
            cutoff = (asof - dt.timedelta(days=365)).isoformat()
            in_window = d[(d["ex_date"] > cutoff) & (d["ex_date"] <= asof.isoformat())]
            if "distr_type" in in_window.columns:
                in_window = in_window[in_window["distr_type"].astype(str) == "1"]
            trailing_sum = float(in_window["amount"].sum())
    q_trailing = trailing_sum / spot if spot > 0 else 0.0
    return {
        "ticker": ticker,
        "trailing_12m_dividends": trailing_sum,
        "q_trailing_12m": q_trailing,
        "q_used": q_trailing,
        "note": "continuous-yield approximation of discrete OptionMetrics "
                "distrd cash dividends; see report limitations",
    }


def load_zero_curve() -> pd.DataFrame:
    """optionm.zerocd for the pull date: columns ``days`` and ``rate_cc``
    (annualised, continuously compounded -- the raw ``rate`` column is a
    PERCENT, divided by 100 here; no ln(1 + r_simple) conversion, unlike the
    old FRED simple-rate path)."""
    d = pd.read_csv(_raw_dir() / "zero_curve.csv")
    d = d.sort_values("days").reset_index(drop=True)
    d["rate_cc"] = d["rate"].astype(float) / 100.0
    return d


def interpolate_zero_rate(curve: pd.DataFrame, days: float) -> float:
    """Linear interpolation of the cc zero curve at a day count. Flat
    extrapolation outside the quoted tenors (documented, not silent)."""
    return float(np.interp(days, curve["days"].to_numpy(float),
                           curve["rate_cc"].to_numpy(float)))


def fetch_risk_free_rate(asof: dt.date | None = None) -> dict:
    """Summary of the zero curve used. The pipeline interpolates a separate
    continuously-compounded rate for every option's own days-to-expiry
    (see pull_and_process); ``r_cc_3m`` below is only a representative value
    for reporting, the 3-month point on the curve."""
    asof = asof or pull_asof()
    curve = load_zero_curve()
    return {
        "asof": str(asof),
        "source": "optionm.zerocd",
        "curve_days": curve["days"].tolist(),
        "curve_rate_cc": curve["rate_cc"].round(6).tolist(),
        "r_cc_3m": interpolate_zero_rate(curve, 91),
        "conversion": "rate_cc = optionm.zerocd.rate / 100 (already continuously "
                      "compounded); interpolated per option expiry",
    }


EXPIRY_BUCKETS = {
    "near": (14, 28),
    "mid": (29, 62),
    "long": (63, 10_000),
}


def pick_expiry_buckets(all_expiries: list[dt.date], asof: dt.date) -> dict:
    """Pure bucket-selection logic, independent of the data source: pick one
    expiry per bucket closest to that bucket's midpoint, spanning near
    (~2-4wk) / mid (~1-2mo) / long (~3mo+). Takes a plain list of dates, so it
    has no dependency on where those dates came from."""
    def days_out(e):
        return (e - asof).days

    chosen = {}
    for label, (lo, hi) in EXPIRY_BUCKETS.items():
        candidates = [e for e in all_expiries if lo <= days_out(e) <= hi]
        if candidates:
            mid_target = (lo + min(hi, 200)) / 2
            chosen[label] = min(candidates, key=lambda e: abs(days_out(e) - mid_target))

    missing = [label for label in EXPIRY_BUCKETS if label not in chosen]
    return {"chosen": chosen, "missing_buckets": missing}


def select_expiries(ticker: str, asof: dt.date | None = None) -> dict:
    """Pick 3 expiries spanning near (~2-4wk), mid (~1-2mo), long (~3mo+) from
    the pulled chain's distinct expiry dates."""
    asof = asof or pull_asof()
    chain = pd.read_csv(_raw_dir() / f"{ticker}_chain.csv", dtype={"exdate": str})
    all_expiries = sorted({dt.date.fromisoformat(e[:10]) for e in chain["exdate"]})
    if not all_expiries:
        raise RuntimeError(f"No option expiries available for {ticker}")

    picked = pick_expiry_buckets(all_expiries, asof)
    return {
        "ticker": ticker,
        "asof": str(asof),
        "all_expiries": [str(e) for e in all_expiries],
        "chosen": {k: str(v) for k, v in picked["chosen"].items()},
        "missing_buckets": picked["missing_buckets"],
    }


def fetch_raw_chain(ticker: str, expiries: list[str]) -> pd.DataFrame:
    """Read the pulled chain for the given expiries and map OptionMetrics
    columns onto the schema filter_chain expects. impl_volatility becomes the
    ``impliedVolatility`` cross-check column; this toolkit re-inverts prices
    itself in implied_vol.py rather than trusting the vendor field."""
    chain = pd.read_csv(_raw_dir() / f"{ticker}_chain.csv", dtype={"exdate": str})
    chain = chain[chain["exdate"].str.slice(0, 10).isin(expiries)].copy()

    strike = chain["strike"].astype(float)
    cp = np.where(chain["option_type"].to_numpy() == "call", "C", "P")
    exp = chain["exdate"].str.slice(0, 10)
    out = pd.DataFrame({
        "ticker": ticker,
        "expiry": exp.to_numpy(),
        "option_type": chain["option_type"].to_numpy(),
        "strike": strike.to_numpy(),
        "bid": chain["best_bid"].astype(float).to_numpy(),
        "ask": chain["best_offer"].astype(float).to_numpy(),
        "volume": chain["volume"].fillna(0).astype(float).to_numpy(),
        "openInterest": chain["open_interest"].fillna(0).astype(float).to_numpy(),
        "impliedVolatility": chain["impl_volatility"].to_numpy(),
        "contractSymbol": [f"{ticker}_{e}_{c}_{k:g}"
                           for e, c, k in zip(exp, cp, strike)],
    })
    return out.reset_index(drop=True)


# --------------------------------------------------------------------------- #
# filter_chain: pure logic, unchanged by the data-source swap
# --------------------------------------------------------------------------- #

def filter_chain(raw: pd.DataFrame, spot: float, asof: dt.date | None = None) -> tuple[pd.DataFrame, dict]:
    """Apply documented filters, return (filtered_df, drop_report).

    Filters applied in order, each counted independently against the
    incoming set at that stage (sequential, not overlapping double-counts).
    """
    asof = asof or pull_asof()
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

    # 4. time to expiry too small. Day count via numpy datetime64 rather than
    # pd.to_datetime, which is unstable on this pandas build.
    exp64 = np.asarray(df["expiry"].astype(str).str.slice(0, 10),
                       dtype="datetime64[D]")
    days_to_exp = (exp64 - np.datetime64(asof)).astype("timedelta64[D]").astype(int)
    df = df.assign(days_to_expiry=days_to_exp)
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


# --------------------------------------------------------------------------- #
# orchestration
# --------------------------------------------------------------------------- #

def pull_and_process(ticker: str, out_tag: str | None = None) -> dict:
    """Full pipeline for one underlying: read the pull, pick expiries, filter,
    attach spot / dividend yield / per-expiry interpolated rate, and save raw
    and processed data with a timestamp."""
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    asof = pull_asof()
    ts = dt.datetime.now().strftime("%Y%m%dT%H%M%S")
    tag = out_tag or ts

    spot = fetch_spot(ticker)
    div_info = fetch_dividend_yield(ticker, spot, asof)
    rate_info = fetch_risk_free_rate(asof)
    curve = load_zero_curve()
    exp_info = select_expiries(ticker, asof)
    expiries = list(exp_info["chosen"].values())
    if len(expiries) < 3:
        print(f"WARNING: {ticker} only had {len(expiries)} expiry buckets filled: "
              f"{exp_info['chosen']}, missing {exp_info['missing_buckets']}")

    raw = fetch_raw_chain(ticker, expiries)
    raw_path = RAW_DIR / f"{ticker}_raw_{tag}.csv"
    raw.to_csv(raw_path, index=False)

    filtered, drop_report = filter_chain(raw, spot, asof)
    filtered = filtered.assign(
        spot=spot,
        q=div_info["q_used"],
        # per-expiry cc rate, interpolated from optionm.zerocd
        r_cc=[interpolate_zero_rate(curve, d) for d in filtered["days_to_expiry"]],
    )
    processed_path = PROCESSED_DIR / f"{ticker}_processed_{tag}.csv"
    filtered.to_csv(processed_path, index=False)

    summary = {
        "ticker": ticker,
        "asof": str(asof),
        "source": "wrds_optionmetrics",
        "spot": spot,
        "dividend_yield": div_info,
        "risk_free_rate": rate_info,
        "expiry_selection": exp_info,
        "raw_path": str(raw_path),
        "processed_path": str(processed_path),
        "filter_report": drop_report,
    }

    summary_path = PROCESSED_DIR / f"{ticker}_summary_{tag}.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    summary["summary_path"] = str(summary_path)

    latest_path = PROCESSED_DIR / f"{ticker}_latest.json"
    with open(latest_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)

    return summary


def latest_pull(ticker: str) -> tuple[dict, pd.DataFrame]:
    """Load the most recent pull's summary + processed dataframe."""
    latest_path = PROCESSED_DIR / f"{ticker}_latest.json"
    with open(latest_path) as f:
        summary = json.load(f)
    df = pd.read_csv(summary["processed_path"])
    return summary, df


if __name__ == "__main__":
    for tkr in ["SPY", "AAPL"]:
        result = pull_and_process(tkr)
        print(json.dumps(result, indent=2, default=str))
