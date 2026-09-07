"""Stage A -- pull raw SPY and AAPL option data from WRDS OptionMetrics
(IvyDB US) for one as-of date. Requires an authenticated WRDS session.

    python scripts/wrds_pull.py --asof 2024-03-15

OptionMetrics data through WRDS lags real time by weeks to months, so pick a
date you know is populated. Writes to data/wrds_raw/:

    {TICKER}_chain.csv      raw option quotes for the as-of date, DTE <= 130
    {TICKER}_spot.csv       one row: the underlying close
    {TICKER}_dividends.csv  ex-dividend cash amounts, trailing ~400 days
    zero_curve.csv          optionm.zerocd for the as-of date (days, rate)
    pull_meta.json          as-of date, secids, row counts

src/data_loader.py consumes these offline. Nothing else in the repo needs a
network connection.

Schema (IvyDB US, verified against the current WRDS layout):
  optionm.securd          secid <- ticker
  optionm.opprcd{YYYY}    daily option quotes; strike_price is in 1/1000 USD;
                          best_bid / best_offer are the EOD NBBO; impl_volatility
                          is OptionMetrics' own inversion (kept as a cross-check)
  optionm.secprd{YYYY}    daily underlying close
  optionm.distrd          cash distributions, by ex_date
  optionm.zerocd          continuously-compounded zero curve, PERCENT, by
                          (date, days-to-maturity)
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
from pathlib import Path

import pandas as pd

RAW_DIR = Path(__file__).resolve().parents[1] / "data" / "wrds_raw"
TICKERS = ("SPY", "AAPL")
MAX_DTE = 130          # keep the pull small; data_loader picks 3 expiry buckets


def connect():
    import wrds
    return wrds.Connection()


def resolve_secid(db, ticker: str) -> int:
    df = db.raw_sql("SELECT secid FROM optionm.securd WHERE ticker = %(t)s",
                    params={"t": ticker})
    if df.empty:
        raise RuntimeError(f"no OptionMetrics secid for ticker {ticker!r}")
    return int(df.iloc[0]["secid"])


def pull_chain(db, secid: int, asof: dt.date) -> pd.DataFrame:
    year = asof.year
    df = db.raw_sql(
        f"""
        SELECT date, exdate, cp_flag, strike_price, best_bid, best_offer,
               volume, open_interest, impl_volatility, delta,
               am_settlement, ss_flag
        FROM optionm.opprcd{year}
        WHERE secid = %(s)s AND date = %(d)s
          AND exdate <= %(d)s + INTERVAL '{MAX_DTE} days'
        """,
        params={"s": secid, "d": asof},
    )
    if df.empty:
        return df
    df["strike"] = df["strike_price"].astype(float) / 1000.0
    df["option_type"] = df["cp_flag"].map({"C": "call", "P": "put"})
    return df.drop(columns=["strike_price", "cp_flag"])


def pull_spot(db, secid: int, asof: dt.date) -> pd.DataFrame:
    year = asof.year
    df = db.raw_sql(
        f"SELECT date, close FROM optionm.secprd{year} "
        f"WHERE secid = %(s)s AND date = %(d)s",
        params={"s": secid, "d": asof},
    )
    if df.empty:
        raise RuntimeError(f"no underlying close for secid {secid} on {asof}")
    return df


def pull_dividends(db, secid: int, asof: dt.date) -> pd.DataFrame:
    start = asof - dt.timedelta(days=400)
    return db.raw_sql(
        """
        SELECT ex_date, amount, distr_type, currency
        FROM optionm.distrd
        WHERE secid = %(s)s AND ex_date > %(a)s AND ex_date <= %(d)s
        ORDER BY ex_date
        """,
        params={"s": secid, "a": start, "d": asof},
    )


def pull_zero_curve(db, asof: dt.date) -> pd.DataFrame:
    df = db.raw_sql(
        "SELECT days, rate FROM optionm.zerocd WHERE date = %(d)s ORDER BY days",
        params={"d": asof},
    )
    if df.empty:
        raise RuntimeError(f"no zero curve for {asof}")
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--asof", required=True, help="YYYY-MM-DD, a populated IvyDB date")
    args = ap.parse_args()
    asof = dt.date.fromisoformat(args.asof)

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    db = connect()
    meta = {"asof": str(asof), "tickers": {}}
    try:
        pull_zero_curve(db, asof).to_csv(RAW_DIR / "zero_curve.csv", index=False)
        for tkr in TICKERS:
            secid = resolve_secid(db, tkr)
            chain = pull_chain(db, secid, asof)
            if chain.empty:
                print(f"{tkr}: no option rows for {asof} -- pick another date")
                continue
            chain.to_csv(RAW_DIR / f"{tkr}_chain.csv", index=False)
            pull_spot(db, secid, asof).to_csv(RAW_DIR / f"{tkr}_spot.csv", index=False)
            pull_dividends(db, secid, asof).to_csv(
                RAW_DIR / f"{tkr}_dividends.csv", index=False)
            meta["tickers"][tkr] = {"secid": secid, "n_quotes": len(chain),
                                    "n_expiries": int(chain["exdate"].nunique())}
            print(f"{tkr}: secid {secid}, {len(chain)} quotes, "
                  f"{chain['exdate'].nunique()} expiries")
    finally:
        db.close()

    with open(RAW_DIR / "pull_meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    print(f"\nwrote {RAW_DIR}")


if __name__ == "__main__":
    main()
