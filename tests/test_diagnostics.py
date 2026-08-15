import numpy as np
import pandas as pd
import pytest

from diagnostics import put_call_parity_check, finite_difference_greeks, greeks_cross_check, early_exercise_premium
from black_scholes import bs_price, bs_greeks


def test_put_call_parity_check_zero_gap_for_consistent_quotes():
    S, K, r, q, T = 100.0, 100.0, 0.03, 0.01, 0.5
    sigma = 0.2
    c = bs_price(S, K, r, q, sigma, T, "call")
    p = bs_price(S, K, r, q, sigma, T, "put")
    df = pd.DataFrame([
        {"ticker": "TEST", "expiry": "2026-01-01", "strike": K, "option_type": "call",
         "mid": c, "spot": S, "q": q, "r_cc": r, "T_years": T},
        {"ticker": "TEST", "expiry": "2026-01-01", "strike": K, "option_type": "put",
         "mid": p, "spot": S, "q": q, "r_cc": r, "T_years": T},
    ])
    result = put_call_parity_check(df)
    assert len(result) == 1
    assert result.iloc[0]["parity_gap"] == pytest.approx(0.0, abs=1e-8)


def test_put_call_parity_check_detects_violation():
    S, K, r, q, T = 100.0, 100.0, 0.03, 0.0, 0.5
    df = pd.DataFrame([
        {"ticker": "TEST", "expiry": "2026-01-01", "strike": K, "option_type": "call",
         "mid": 10.0, "spot": S, "q": q, "r_cc": r, "T_years": T},
        {"ticker": "TEST", "expiry": "2026-01-01", "strike": K, "option_type": "put",
         "mid": 1.0, "spot": S, "q": q, "r_cc": r, "T_years": T},  # deliberately inconsistent
    ])
    result = put_call_parity_check(df)
    assert abs(result.iloc[0]["parity_gap"]) > 0.5


def test_finite_difference_greeks_agree_with_analytical():
    S, K, r, q, sigma, T = 100.0, 105.0, 0.04, 0.01, 0.25, 0.75
    for opt in ["call", "put"]:
        analytical = bs_greeks(S, K, r, q, sigma, T, opt)
        fd = finite_difference_greeks(S, K, r, q, sigma, T, opt)
        assert analytical["delta"] == pytest.approx(fd["delta"], abs=1e-4)
        assert analytical["gamma"] == pytest.approx(fd["gamma"], abs=1e-3)
        assert analytical["vega"] == pytest.approx(fd["vega"], abs=1e-3)
        assert analytical["rho"] == pytest.approx(fd["rho"], abs=1e-3)
        assert analytical["theta"] == pytest.approx(fd["theta"], abs=1e-2)


def test_greeks_cross_check_reports_small_max_abs_diff():
    result = greeks_cross_check(100, 100, 0.03, 0.0, 0.2, 1.0, "call")
    for greek, diff in result["max_abs_diff"].items():
        assert diff < 1e-2


def test_early_exercise_premium_nonnegative_for_put():
    df = pd.DataFrame([{
        "ticker": "TEST", "expiry": "2026-01-01", "strike": 110.0, "option_type": "put",
        "mid": 12.0, "spot": 100.0, "q": 0.0, "r_cc": 0.05, "T_years": 1.0, "iv_computed": 0.2,
    }])
    result = early_exercise_premium(df, N=200)
    assert result.iloc[0]["binomial_early_exercise_premium"] >= -1e-9
