"""
Tests for the data pipeline. filter_chain is pure logic, tested directly
on synthetic DataFrames (no network). The yfinance/FRED-touching functions
are tested with mocks so the suite stays fast and offline.
"""
import datetime as dt
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from data_loader import filter_chain, select_expiries, fetch_dividend_yield, fetch_risk_free_rate


def _synthetic_raw_chain():
    """Spot = 100. Rows deliberately hit every filter condition once."""
    rows = [
        # normal, should survive everything
        {"strike": 100, "bid": 5.0, "ask": 5.2, "volume": 50, "openInterest": 100,
         "option_type": "call", "expiry": "2099-06-01", "ticker": "TEST"},
        # zero bid -> dropped
        {"strike": 105, "bid": 0.0, "ask": 1.0, "volume": 10, "openInterest": 50,
         "option_type": "call", "expiry": "2099-06-01", "ticker": "TEST"},
        # crossed quote (bid > ask) -> dropped
        {"strike": 95, "bid": 6.0, "ask": 5.5, "volume": 10, "openInterest": 50,
         "option_type": "put", "expiry": "2099-06-01", "ticker": "TEST"},
        # low open interest -> dropped
        {"strike": 110, "bid": 1.0, "ask": 1.2, "volume": 5, "openInterest": 3,
         "option_type": "call", "expiry": "2099-06-01", "ticker": "TEST"},
        # deep OTM, extreme moneyness -> dropped (ln(200/100)=0.69 > 0.40)
        {"strike": 200, "bid": 0.1, "ask": 0.2, "volume": 10, "openInterest": 50,
         "option_type": "call", "expiry": "2099-06-01", "ticker": "TEST"},
        # zero volume but healthy OI -> should survive (OI is the liquidity gate, not volume)
        {"strike": 102, "bid": 3.0, "ask": 3.2, "volume": 0, "openInterest": 40,
         "option_type": "call", "expiry": "2099-06-01", "ticker": "TEST"},
    ]
    return pd.DataFrame(rows)


def test_filter_chain_drops_expected_rows_for_expected_reasons():
    raw = _synthetic_raw_chain()
    asof = dt.date(2099, 1, 1)  # far before expiry, so nothing trips the expiry filter
    filtered, report = filter_chain(raw, spot=100.0, asof=asof)

    assert report["n_start"] == 6
    assert report["dropped_zero_or_missing_bid"] == 1
    assert report["dropped_bid_gt_ask"] == 1
    assert report["dropped_low_open_interest"] == 1
    assert report["dropped_extreme_moneyness"] == 1
    assert report["n_end"] == 2
    assert set(filtered["strike"]) == {100, 102}


def test_filter_chain_keeps_zero_volume_row_with_healthy_open_interest():
    # This is the specific behavior the module's docstring claims: open
    # interest is the liquidity gate, not same-day volume.
    raw = _synthetic_raw_chain()
    filtered, _ = filter_chain(raw, spot=100.0, asof=dt.date(2099, 1, 1))
    zero_vol_row = filtered[filtered.strike == 102]
    assert len(zero_vol_row) == 1
    assert zero_vol_row.iloc[0]["volume"] == 0


def test_filter_chain_computes_mid_and_log_moneyness_correctly():
    raw = _synthetic_raw_chain()
    filtered, _ = filter_chain(raw, spot=100.0, asof=dt.date(2099, 1, 1))
    atm = filtered[filtered.strike == 100].iloc[0]
    assert atm["mid"] == pytest.approx(5.1)
    assert atm["log_moneyness"] == pytest.approx(0.0, abs=1e-9)


def test_filter_chain_drops_expiry_too_close():
    raw = pd.DataFrame([{
        "strike": 100, "bid": 5.0, "ask": 5.2, "volume": 50, "openInterest": 100,
        "option_type": "call", "expiry": "2099-01-01", "ticker": "TEST",
    }])
    filtered, report = filter_chain(raw, spot=100.0, asof=dt.date(2099, 1, 1))  # same day
    assert report["dropped_too_close_to_expiry"] == 1
    assert len(filtered) == 0


def test_filter_chain_empty_input_does_not_crash():
    raw = pd.DataFrame(columns=["strike", "bid", "ask", "volume", "openInterest",
                                 "option_type", "expiry", "ticker"])
    filtered, report = filter_chain(raw, spot=100.0)
    assert report["n_start"] == 0
    assert report["n_end"] == 0
    assert len(filtered) == 0


def test_select_expiries_picks_near_mid_long_buckets():
    mock_ticker = MagicMock()
    asof = dt.date(2099, 1, 1)
    mock_ticker.options = (
        "2099-01-10",  # 9 days: below near bucket (14-28)
        "2099-01-20",  # 19 days: near
        "2099-02-15",  # 45 days: mid
        "2099-04-01",  # 90 days: long
    )
    with patch("data_loader.yf.Ticker", return_value=mock_ticker):
        result = select_expiries("TEST", asof)

    assert result["chosen"]["near"] == "2099-01-20"
    assert result["chosen"]["mid"] == "2099-02-15"
    assert result["chosen"]["long"] == "2099-04-01"
    assert result["missing_buckets"] == []


def test_select_expiries_reports_missing_bucket_explicitly():
    mock_ticker = MagicMock()
    asof = dt.date(2099, 1, 1)
    mock_ticker.options = ("2099-01-20",)  # only a near-dated expiry exists
    with patch("data_loader.yf.Ticker", return_value=mock_ticker):
        result = select_expiries("TEST", asof)

    assert "near" in result["chosen"]
    assert "mid" in result["missing_buckets"]
    assert "long" in result["missing_buckets"]


def test_select_expiries_raises_on_no_expiries_at_all():
    mock_ticker = MagicMock()
    mock_ticker.options = ()
    with patch("data_loader.yf.Ticker", return_value=mock_ticker):
        with pytest.raises(RuntimeError):
            select_expiries("TEST", dt.date(2099, 1, 1))


def test_fetch_dividend_yield_computes_trailing_12m_sum():
    mock_ticker = MagicMock()
    idx = pd.to_datetime(["2098-03-01", "2098-06-01", "2098-09-01", "2098-12-01", "2097-01-01"])
    mock_ticker.dividends = pd.Series([0.25, 0.25, 0.25, 0.25, 99.0], index=idx)
    mock_ticker.info = {"dividendYield": 1.5}
    with patch("data_loader.yf.Ticker", return_value=mock_ticker):
        result = fetch_dividend_yield("TEST", spot=100.0, asof=dt.date(2099, 1, 1))

    # only the four 0.25 payments fall in the trailing-12m window; the 99.0
    # payment from 2097 must NOT be included
    assert result["trailing_12m_dividends"] == pytest.approx(1.0)
    assert result["q_trailing_12m"] == pytest.approx(0.01)
    assert result["q_used"] == result["q_trailing_12m"]


def test_fetch_dividend_yield_normalizes_percent_vs_decimal_info_field():
    mock_ticker = MagicMock()
    mock_ticker.dividends = pd.Series([], index=pd.DatetimeIndex([]), dtype=float)
    mock_ticker.info = {"dividendYield": 35.0}  # a yfinance-style bogus percent value
    with patch("data_loader.yf.Ticker", return_value=mock_ticker):
        result = fetch_dividend_yield("TEST", spot=100.0, asof=dt.date(2099, 1, 1))
    assert result["q_info_field"] == pytest.approx(0.35)  # normalized, not used for q_used
    assert result["q_used"] == 0.0  # no dividends in the window -> trailing yield is 0


def test_fetch_risk_free_rate_converts_to_continuous_compounding():
    mock_df = pd.DataFrame({"DGS3MO": [3.85, 3.87]},
                            index=pd.to_datetime(["2099-01-05", "2099-01-06"]))
    with patch("data_loader.web.DataReader", return_value=mock_df):
        result = fetch_risk_free_rate(dt.date(2099, 1, 10))

    assert result["r_simple_pct"] == pytest.approx(3.87)
    assert result["r_simple"] == pytest.approx(0.0387)
    assert result["r_cc"] == pytest.approx(np.log(1.0387))


def test_fetch_risk_free_rate_raises_when_no_data_in_window():
    empty_df = pd.DataFrame({"DGS3MO": []}, index=pd.DatetimeIndex([]))
    with patch("data_loader.web.DataReader", return_value=empty_df):
        with pytest.raises(RuntimeError):
            fetch_risk_free_rate(dt.date(2099, 1, 10))
