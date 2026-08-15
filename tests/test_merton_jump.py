import numpy as np
import pandas as pd
import pytest

from merton_jump import merton_jump_price, merton_jump_price_value, calibrate_merton
from black_scholes import bs_price
from implied_vol import solve_implied_vol


def test_zero_jump_intensity_reduces_to_black_scholes():
    S, K, r, q, sigma, T = 100.0, 100.0, 0.03, 0.0, 0.2, 1.0
    bs = bs_price(S, K, r, q, sigma, T, "call")
    merton, info = merton_jump_price(S, K, r, q, sigma, T, lam=0.0, mu_j=-0.1, sigma_j=0.2, option_type="call")
    assert merton == pytest.approx(bs, abs=1e-8)


def test_truncation_report_has_small_remaining_tail_mass():
    S, K, r, q, sigma, T = 100.0, 100.0, 0.03, 0.0, 0.2, 1.0
    _, info = merton_jump_price(S, K, r, q, sigma, T, lam=1.0, mu_j=-0.1, sigma_j=0.2, option_type="call")
    assert info["tail_mass_remaining"] < 1e-8
    assert info["n_terms_used"] < 50  # should truncate well before max_terms


def test_price_is_positive_and_finite():
    S, K, r, q, sigma, T = 100.0, 105.0, 0.03, 0.01, 0.25, 0.5
    price = merton_jump_price_value(S, K, r, q, sigma, T, lam=0.8, mu_j=-0.15, sigma_j=0.25, option_type="put")
    assert np.isfinite(price)
    assert price > 0


def test_put_call_parity_approximately_holds():
    # Merton jump-diffusion is still a risk-neutral model of S_T, so
    # put-call parity on the *model* prices should hold (it's a
    # model-independent no-arbitrage relation given the same S,K,r,q,T).
    S, K, r, q, sigma, T = 100.0, 95.0, 0.04, 0.02, 0.22, 0.75
    lam, mu_j, sigma_j = 0.6, -0.1, 0.2
    c = merton_jump_price_value(S, K, r, q, sigma, T, lam, mu_j, sigma_j, "call")
    p = merton_jump_price_value(S, K, r, q, sigma, T, lam, mu_j, sigma_j, "put")
    lhs = c - p
    rhs = S * np.exp(-q * T) - K * np.exp(-r * T)
    assert lhs == pytest.approx(rhs, abs=1e-4)


def test_calibration_recovers_known_smile_reasonably():
    # Generate a synthetic "market" from known Merton params, then check
    # calibration finds parameters that reproduce the IV surface closely
    # (not necessarily the exact same parameters -- jump params can be
    # weakly identified -- but the fit quality should be tight).
    S, r, q = 100.0, 0.03, 0.01
    true_params = dict(sigma=0.18, lam=0.7, mu_j=-0.12, sigma_j=0.20)
    rows = []
    for T in [0.1, 0.25, 0.5]:
        for K in [80, 90, 95, 100, 105, 110, 120]:
            opt = "call"
            price = merton_jump_price_value(S, K, r, q, T=T, option_type=opt, **true_params)
            iv = solve_implied_vol(price, S, K, r, q, T, opt)["iv"]
            rows.append({"strike": K, "T_years": T, "option_type": opt, "iv_computed": iv})
    df = pd.DataFrame(rows)

    result = calibrate_merton(df, spot=S, r=r, q=q)
    assert result["final_rmse_iv"] < 0.01  # tight fit to synthetic ground truth


def test_multi_start_returns_best_and_reports_all_attempts():
    S, r, q = 100.0, 0.03, 0.01
    true_params = dict(sigma=0.18, lam=0.7, mu_j=-0.12, sigma_j=0.20)
    rows = []
    for T in [0.25, 0.5]:
        for K in [85, 95, 100, 105, 115]:
            price = merton_jump_price_value(S, K, r, q, T=T, option_type="call", **true_params)
            iv = solve_implied_vol(price, S, K, r, q, T, "call")["iv"]
            rows.append({"strike": K, "T_years": T, "option_type": "call", "iv_computed": iv})
    df = pd.DataFrame(rows)

    result = calibrate_merton(df, spot=S, r=r, q=q, n_starts=3, random_seed=1)
    assert result["n_starts"] == 3
    assert len(result["all_start_rmse"]) == 3
    assert result["final_rmse_iv"] == pytest.approx(min(result["all_start_rmse"]))


def test_calibration_reports_failure_state_explicitly():
    result = calibrate_merton(
        pd.DataFrame({"strike": [100], "T_years": [0.5], "option_type": ["call"], "iv_computed": [0.2]}),
        spot=100, r=0.03, q=0.0,
    )
    assert "success" in result and isinstance(result["success"], bool)
