import numpy as np
import pandas as pd
import pytest

from black_scholes import bs_price
from implied_vol import solve_implied_vol, add_implied_vol


@pytest.mark.parametrize("true_sigma", [0.05, 0.15, 0.30, 0.60, 1.20])
@pytest.mark.parametrize("option_type", ["call", "put"])
def test_round_trip_recovers_known_sigma(true_sigma, option_type):
    S, K, r, q, T = 100.0, 105.0, 0.04, 0.01, 0.5
    price = bs_price(S, K, r, q, true_sigma, T, option_type)
    result = solve_implied_vol(price, S, K, r, q, T, option_type)
    assert result["success"]
    assert result["iv"] == pytest.approx(true_sigma, abs=1e-4)


def test_price_below_intrinsic_fails_gracefully():
    # An arbitrage-violating quote (price below the lower no-arb bound) has
    # no valid implied vol -- solver should report failure, not crash or
    # silently return a placeholder.
    S, K, r, q, T = 100.0, 50.0, 0.03, 0.0, 1.0
    bogus_price = 1e-8  # far below intrinsic S - K*exp(-rT)
    result = solve_implied_vol(bogus_price, S, K, r, q, T, "call")
    assert not result["success"]
    assert np.isnan(result["iv"])
    assert result["message"] is not None


def test_zero_or_negative_price_fails_gracefully():
    result = solve_implied_vol(0.0, 100, 100, 0.03, 0.0, 1.0, "call")
    assert not result["success"]
    result2 = solve_implied_vol(-5.0, 100, 100, 0.03, 0.0, 1.0, "call")
    assert not result2["success"]


def test_zero_maturity_fails_gracefully():
    result = solve_implied_vol(5.0, 100, 100, 0.03, 0.0, 0.0, "call")
    assert not result["success"]


def test_batch_application_on_synthetic_chain():
    rows = []
    for K in [80, 90, 100, 110, 120]:
        for opt in ["call", "put"]:
            true_sigma = 0.15 + 0.002 * (100 - K) ** 2 / 100  # fake smile
            price = bs_price(100, K, 0.03, 0.01, true_sigma, 0.5, opt)
            rows.append({"spot": 100, "strike": K, "r_cc": 0.03, "q": 0.01,
                         "T_years": 0.5, "option_type": opt, "mid": price,
                         "true_sigma": true_sigma})
    df = pd.DataFrame(rows)
    out, report = add_implied_vol(df)
    assert report["n_failed"] == 0
    assert report["pct_success"] == 100.0
    np.testing.assert_allclose(out["iv_computed"], out["true_sigma"], atol=1e-4)


def test_batch_reports_failures_without_dropping_rows():
    df = pd.DataFrame([
        {"spot": 100, "strike": 100, "r_cc": 0.03, "q": 0.0, "T_years": 0.5,
         "option_type": "call", "mid": bs_price(100, 100, 0.03, 0.0, 0.2, 0.5, "call")},
        {"spot": 100, "strike": 50, "r_cc": 0.03, "q": 0.0, "T_years": 0.5,
         "option_type": "call", "mid": 1e-9},  # bogus/arbitrage-violating quote
    ])
    out, report = add_implied_vol(df)
    assert len(out) == 2  # no silent dropping
    assert report["n_failed"] == 1
    assert out.loc[0, "iv_success"] and not out.loc[1, "iv_success"]
