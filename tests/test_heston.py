import numpy as np
import pandas as pd
import pytest

from heston import heston_price, heston_price_batch, feller_condition, calibrate_heston
from black_scholes import bs_price
from implied_vol import solve_implied_vol


def test_heston_reduces_to_black_scholes_when_vol_is_deterministic():
    # If v0 = theta and xi -> 0, dv_t = kappa*(theta - v0) dt = 0, so
    # variance stays at v0 for the whole path -- Heston must collapse to
    # plain BS with sigma = sqrt(v0). This is the core correctness check
    # for the characteristic-function implementation.
    S, K, r, q, T = 100.0, 100.0, 0.03, 0.01, 0.5
    v0 = 0.04
    bs = bs_price(S, K, r, q, np.sqrt(v0), T, "call")
    heston = heston_price(S, K, r, q, T, kappa=2.0, theta=v0, xi=1e-6, rho=-0.5, v0=v0, option_type="call")
    assert heston == pytest.approx(bs, abs=1e-3)


def test_heston_reduces_to_black_scholes_otm_put():
    S, K, r, q, T = 100.0, 110.0, 0.03, 0.0, 0.75
    v0 = 0.09
    bs = bs_price(S, K, r, q, np.sqrt(v0), T, "put")
    heston = heston_price(S, K, r, q, T, kappa=1.5, theta=v0, xi=1e-6, rho=0.2, v0=v0, option_type="put")
    assert heston == pytest.approx(bs, abs=1e-3)


def test_put_call_parity():
    S, K, r, q, T = 100.0, 95.0, 0.04, 0.02, 1.0
    params = dict(kappa=2.0, theta=0.05, xi=0.4, rho=-0.6, v0=0.04)
    c = heston_price(S, K, r, q, T, option_type="call", **params)
    p = heston_price(S, K, r, q, T, option_type="put", **params)
    lhs = c - p
    rhs = S * np.exp(-q * T) - K * np.exp(-r * T)
    assert lhs == pytest.approx(rhs, abs=1e-3)


def test_batch_matches_scalar():
    S, r, q, T = 100.0, 0.03, 0.01, 0.5
    params = dict(kappa=2.0, theta=0.04, xi=0.5, rho=-0.5, v0=0.045)
    Ks = [80, 90, 100, 110, 120]
    batch = heston_price_batch(Ks, S, r, q, T, option_type="call", **params)
    scalars = [heston_price(S, K, r, q, T, option_type="call", **params) for K in Ks]
    np.testing.assert_allclose(batch, scalars, atol=1e-6)


def test_price_positive_and_monotonic_in_v0():
    S, K, r, q, T = 100.0, 100.0, 0.03, 0.0, 0.5
    low = heston_price(S, K, r, q, T, kappa=2.0, theta=0.02, xi=0.3, rho=-0.5, v0=0.01, option_type="call")
    high = heston_price(S, K, r, q, T, kappa=2.0, theta=0.02, xi=0.3, rho=-0.5, v0=0.10, option_type="call")
    assert 0 < low < high


def test_feller_condition_sign():
    assert feller_condition(kappa=2.0, theta=0.04, xi=0.1) > 0   # 2*2*0.04=0.16 > 0.01
    assert feller_condition(kappa=1.0, theta=0.02, xi=1.0) < 0   # 2*1*0.02=0.04 < 1.0


def test_calibration_recovers_synthetic_smile():
    S, r, q = 100.0, 0.03, 0.01
    true_params = dict(kappa=2.0, theta=0.04, xi=0.4, rho=-0.6, v0=0.045)
    rows = []
    for T in [0.25, 0.5]:
        for K in [85, 95, 100, 105, 115]:
            price = heston_price(S, K, r, q, T, option_type="call", **true_params)
            iv = solve_implied_vol(price, S, K, r, q, T, "call")["iv"]
            rows.append({"strike": K, "T_years": T, "option_type": "call", "iv_computed": iv})
    df = pd.DataFrame(rows)

    result = calibrate_heston(df, spot=S, r=r, q=q, x0=[1.5, 0.03, 0.3, -0.4, 0.04],
                               enforce_feller=False, maxiter=100)
    assert result["final_rmse_iv"] < 0.01


def test_multi_start_returns_best_and_reports_all_attempts():
    S, r, q = 100.0, 0.03, 0.01
    true_params = dict(kappa=2.0, theta=0.04, xi=0.4, rho=-0.6, v0=0.045)
    rows = []
    for T in [0.25, 0.5]:
        for K in [85, 95, 100, 105, 115]:
            price = heston_price(S, K, r, q, T, option_type="call", **true_params)
            iv = solve_implied_vol(price, S, K, r, q, T, "call")["iv"]
            rows.append({"strike": K, "T_years": T, "option_type": "call", "iv_computed": iv})
    df = pd.DataFrame(rows)

    result = calibrate_heston(df, spot=S, r=r, q=q, x0=[1.5, 0.03, 0.3, -0.4, 0.04],
                               enforce_feller=False, maxiter=80, n_starts=3, random_seed=1)
    assert result["n_starts"] == 3
    assert len(result["all_start_rmse"]) == 3
    # the returned fit must be at least as good as every individual attempt
    assert result["final_rmse_iv"] == pytest.approx(min(result["all_start_rmse"]))


def test_multi_start_is_reproducible_given_same_seed():
    df = pd.DataFrame({
        "strike": [95, 100, 105], "T_years": [0.5, 0.5, 0.5],
        "option_type": ["call", "call", "call"], "iv_computed": [0.22, 0.20, 0.19],
    })
    r1 = calibrate_heston(df, spot=100, r=0.03, q=0.0, enforce_feller=False, maxiter=30,
                           n_starts=3, random_seed=7)
    r2 = calibrate_heston(df, spot=100, r=0.03, q=0.0, enforce_feller=False, maxiter=30,
                           n_starts=3, random_seed=7)
    assert r1["all_start_rmse"] == r2["all_start_rmse"]


def test_calibration_reports_feller_status_explicitly():
    df = pd.DataFrame({
        "strike": [95, 100, 105], "T_years": [0.5, 0.5, 0.5],
        "option_type": ["call", "call", "call"], "iv_computed": [0.22, 0.20, 0.19],
    })
    result = calibrate_heston(df, spot=100, r=0.03, q=0.0, enforce_feller=True, maxiter=30)
    assert "feller_satisfied" in result
    assert isinstance(result["feller_satisfied"], bool)
    assert "success" in result
