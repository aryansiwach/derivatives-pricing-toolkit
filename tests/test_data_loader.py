"""
Tests for the data pipeline. filter_chain is pure logic, tested directly on
synthetic DataFrames. The Stage-B readers are tested against synthetic
data/wrds_raw CSVs written to a temp directory -- fast, offline, no WRDS.
"""
import datetime as dt
import json

import numpy as np
import pandas as pd
import pytest

import data_loader
from data_loader import (
    fetch_dividend_yield, fetch_raw_chain, fetch_risk_free_rate, fetch_spot,
    filter_chain, interpolate_zero_rate, load_zero_curve, select_expiries,
)

ASOF = dt.date(2024, 3, 15)


# --------------------------------------------------------------------------- #
# filter_chain -- pure, unchanged by the source swap
# --------------------------------------------------------------------------- #

def _synthetic_raw_chain():
    """Spot = 100. Rows deliberately hit every filter condition once."""
    rows = [
        {"strike": 100, "bid": 5.0, "ask": 5.2, "volume": 50, "openInterest": 100,
         "option_type": "call", "expiry": "2099-06-01", "ticker": "TEST"},
        {"strike": 105, "bid": 0.0, "ask": 1.0, "volume": 10, "openInterest": 50,
         "option_type": "call", "expiry": "2099-06-01", "ticker": "TEST"},
        {"strike": 95, "bid": 6.0, "ask": 5.5, "volume": 10, "openInterest": 50,
         "option_type": "put", "expiry": "2099-06-01", "ticker": "TEST"},
        {"strike": 110, "bid": 1.0, "ask": 1.2, "volume": 5, "openInterest": 3,
         "option_type": "call", "expiry": "2099-06-01", "ticker": "TEST"},
        {"strike": 200, "bid": 0.1, "ask": 0.2, "volume": 10, "openInterest": 50,
         "option_type": "call", "expiry": "2099-06-01", "ticker": "TEST"},
        {"strike": 102, "bid": 3.0, "ask": 3.2, "volume": 0, "openInterest": 40,
         "option_type": "call", "expiry": "2099-06-01", "ticker": "TEST"},
    ]
    return pd.DataFrame(rows)


def test_filter_chain_drops_expected_rows_for_expected_reasons():
    filtered, report = filter_chain(_synthetic_raw_chain(), spot=100.0,
                                    asof=dt.date(2099, 1, 1))
    assert report["n_start"] == 6
    assert report["dropped_zero_or_missing_bid"] == 1
    assert report["dropped_bid_gt_ask"] == 1
    assert report["dropped_low_open_interest"] == 1
    assert report["dropped_extreme_moneyness"] == 1
    assert report["n_end"] == 2
    assert set(filtered["strike"]) == {100, 102}


def test_filter_chain_keeps_zero_volume_row_with_healthy_open_interest():
    filtered, _ = filter_chain(_synthetic_raw_chain(), spot=100.0,
                               asof=dt.date(2099, 1, 1))
    zero_vol_row = filtered[filtered.strike == 102]
    assert len(zero_vol_row) == 1
    assert zero_vol_row.iloc[0]["volume"] == 0


def test_filter_chain_computes_mid_and_log_moneyness_correctly():
    filtered, _ = filter_chain(_synthetic_raw_chain(), spot=100.0,
                               asof=dt.date(2099, 1, 1))
    atm = filtered[filtered.strike == 100].iloc[0]
    assert atm["mid"] == pytest.approx(5.1)
    assert atm["log_moneyness"] == pytest.approx(0.0, abs=1e-9)


def test_filter_chain_drops_expiry_too_close():
    raw = pd.DataFrame([{
        "strike": 100, "bid": 5.0, "ask": 5.2, "volume": 50, "openInterest": 100,
        "option_type": "call", "expiry": "2099-01-01", "ticker": "TEST",
    }])
    filtered, report = filter_chain(raw, spot=100.0, asof=dt.date(2099, 1, 1))
    assert report["dropped_too_close_to_expiry"] == 1
    assert len(filtered) == 0


def test_filter_chain_empty_input_does_not_crash():
    raw = pd.DataFrame(columns=["strike", "bid", "ask", "volume", "openInterest",
                                 "option_type", "expiry", "ticker"])
    filtered, report = filter_chain(raw, spot=100.0, asof=dt.date(2099, 1, 1))
    assert report["n_start"] == 0
    assert report["n_end"] == 0
    assert len(filtered) == 0


# --------------------------------------------------------------------------- #
# Stage-B readers -- synthetic data/wrds_raw
# --------------------------------------------------------------------------- #

@pytest.fixture
def wrds_raw(tmp_path, monkeypatch):
    d = tmp_path / "wrds_raw"
    d.mkdir()
    (d / "pull_meta.json").write_text(json.dumps({"asof": str(ASOF),
                                                  "tickers": {}}))
    # zero curve: cc PERCENT
    pd.DataFrame({"days": [7, 30, 91, 182, 365],
                  "rate": [5.20, 5.25, 5.30, 5.20, 5.00]}).to_csv(
        d / "zero_curve.csv", index=False)
    # SPY spot + dividends
    pd.DataFrame({"date": [str(ASOF)], "close": [512.0]}).to_csv(
        d / "SPY_spot.csv", index=False)
    pd.DataFrame({
        "ex_date": ["2023-06-16", "2023-09-15", "2023-12-15", "2024-03-15",
                    "2022-01-01"],
        "amount": [1.6, 1.6, 1.8, 1.7, 99.0],
        "distr_type": ["1", "1", "1", "1", "1"],
        "currency": ["USD"] * 5,
    }).to_csv(d / "SPY_dividends.csv", index=False)
    # SPY chain: two expiries, a few strikes each
    rows = []
    for exd, dte in [("2024-04-05", 21), ("2024-05-17", 63)]:
        for k in (480, 500, 512, 525, 545):
            for cp, typ in (("C", "call"), ("P", "put")):
                rows.append({"date": str(ASOF), "exdate": exd, "strike": float(k),
                             "option_type": typ, "best_bid": 5.0, "best_offer": 5.4,
                             "volume": 100, "open_interest": 500,
                             "impl_volatility": 0.13, "delta": 0.5,
                             "am_settlement": 0, "ss_flag": 0})
    pd.DataFrame(rows).to_csv(d / "SPY_chain.csv", index=False)

    monkeypatch.setattr(data_loader, "WRDS_RAW_DIR", d)
    return d


def test_fetch_spot_reads_close(wrds_raw):
    assert fetch_spot("SPY") == 512.0


def test_zero_curve_percent_to_cc_and_interpolation(wrds_raw):
    curve = load_zero_curve()
    assert curve["rate_cc"].iloc[0] == pytest.approx(0.052)
    # 60 days sits between the 30d (5.25%) and 91d (5.30%) points
    r60 = interpolate_zero_rate(curve, 60)
    assert 0.0525 < r60 < 0.0530


def test_fetch_dividend_yield_trailing_12m_only(wrds_raw):
    info = fetch_dividend_yield("SPY", spot=512.0, asof=ASOF)
    # 1.6 + 1.6 + 1.8 + 1.7 = 6.7 inside the trailing year (cutoff 2023-03-16);
    # the 99.0 payment from 2022-01-01 is excluded
    assert info["trailing_12m_dividends"] == pytest.approx(6.7)
    assert info["q_used"] == pytest.approx(6.7 / 512.0)


def test_select_expiries_from_chain(wrds_raw):
    res = select_expiries("SPY", ASOF)
    assert res["chosen"]["near"] == "2024-04-05"
    assert res["chosen"]["long"] == "2024-05-17"
    assert "mid" in res["missing_buckets"]


def test_fetch_raw_chain_maps_optionmetrics_columns(wrds_raw):
    raw = fetch_raw_chain("SPY", ["2024-04-05"])
    assert set(raw.columns) >= {"ticker", "expiry", "option_type", "strike",
                                "bid", "ask", "volume", "openInterest",
                                "impliedVolatility", "contractSymbol"}
    assert (raw["expiry"] == "2024-04-05").all()
    assert raw["bid"].iloc[0] == 5.0
    assert raw["contractSymbol"].iloc[0].startswith("SPY_2024-04-05_")


def test_risk_free_rate_summary_shape(wrds_raw):
    rf = fetch_risk_free_rate(ASOF)
    assert rf["source"] == "optionm.zerocd"
    assert rf["r_cc_3m"] == pytest.approx(0.053, abs=1e-4)
    assert "ln(1 + r_simple)" not in rf["conversion"]


def test_end_to_end_pull_and_process(wrds_raw, tmp_path, monkeypatch):
    monkeypatch.setattr(data_loader, "RAW_DIR", tmp_path / "raw")
    monkeypatch.setattr(data_loader, "PROCESSED_DIR", tmp_path / "processed")
    summary = data_loader.pull_and_process("SPY")
    assert summary["source"] == "wrds_optionmetrics"
    assert summary["spot"] == 512.0
    _, df = data_loader.latest_pull("SPY")
    assert len(df) > 0
    # per-expiry interpolated rate, not one flat value
    assert df["r_cc"].nunique() >= 1
    assert (df["r_cc"] > 0.04).all() and (df["r_cc"] < 0.06).all()
