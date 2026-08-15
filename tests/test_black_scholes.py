import numpy as np
import pytest

from black_scholes import bs_price, bs_greeks


def test_hull_example_call():
    # Hull, "Options, Futures, and Other Derivatives", Ch. 15 worked example:
    # S=42, K=40, r=10%, sigma=20%, T=0.5, no dividends -> c ~= 4.76
    c = bs_price(S=42, K=40, r=0.10, q=0.0, sigma=0.20, T=0.5, option_type="call")
    assert c == pytest.approx(4.76, abs=0.01)


def test_hull_example_put():
    # Same inputs, put-call parity gives p ~= 0.81
    p = bs_price(S=42, K=40, r=0.10, q=0.0, sigma=0.20, T=0.5, option_type="put")
    assert p == pytest.approx(0.81, abs=0.01)


def test_put_call_parity_no_dividend():
    S, K, r, q, sigma, T = 100.0, 105.0, 0.03, 0.0, 0.25, 0.75
    c = bs_price(S, K, r, q, sigma, T, "call")
    p = bs_price(S, K, r, q, sigma, T, "put")
    lhs = c - p
    rhs = S * np.exp(-q * T) - K * np.exp(-r * T)
    assert lhs == pytest.approx(rhs, abs=1e-10)


def test_put_call_parity_with_dividend():
    S, K, r, q, sigma, T = 250.0, 240.0, 0.04, 0.02, 0.30, 1.5
    c = bs_price(S, K, r, q, sigma, T, "call")
    p = bs_price(S, K, r, q, sigma, T, "put")
    lhs = c - p
    rhs = S * np.exp(-q * T) - K * np.exp(-r * T)
    assert lhs == pytest.approx(rhs, abs=1e-10)


def test_deep_itm_call_converges_to_intrinsic():
    S, K, r, q, sigma, T = 1000.0, 10.0, 0.03, 0.0, 0.2, 1.0
    c = bs_price(S, K, r, q, sigma, T, "call")
    intrinsic = S - K * np.exp(-r * T)
    assert c == pytest.approx(intrinsic, rel=1e-6)


def test_deep_otm_call_near_zero():
    S, K, r, q, sigma, T = 10.0, 1000.0, 0.03, 0.0, 0.2, 1.0
    c = bs_price(S, K, r, q, sigma, T, "call")
    assert c == pytest.approx(0.0, abs=1e-6)


def test_zero_vol_call_is_discounted_intrinsic():
    # As sigma -> 0, the option becomes deterministic: max(S_T - K, 0) discounted.
    S, K, r, q, T = 100.0, 90.0, 0.05, 0.0, 1.0
    sigma = 1e-6
    c = bs_price(S, K, r, q, sigma, T, "call")
    forward = S * np.exp((r - q) * T)
    expected = np.exp(-r * T) * max(forward - K, 0.0)
    assert c == pytest.approx(expected, abs=1e-3)


def test_near_zero_maturity_call_is_intrinsic():
    S, K, r, q, sigma = 105.0, 100.0, 0.03, 0.0, 0.2
    T = 1e-6
    c = bs_price(S, K, r, q, sigma, T, "call")
    assert c == pytest.approx(max(S - K, 0.0), abs=1e-2)


def test_call_delta_bounds():
    greeks = bs_greeks(S=100, K=100, r=0.03, q=0.0, sigma=0.2, T=1.0, option_type="call")
    assert 0.0 <= greeks["delta"] <= 1.0


def test_put_delta_bounds():
    greeks = bs_greeks(S=100, K=100, r=0.03, q=0.0, sigma=0.2, T=1.0, option_type="put")
    assert -1.0 <= greeks["delta"] <= 0.0


def test_gamma_positive_and_shared_across_call_put():
    g_call = bs_greeks(S=100, K=95, r=0.03, q=0.01, sigma=0.25, T=0.5, option_type="call")
    g_put = bs_greeks(S=100, K=95, r=0.03, q=0.01, sigma=0.25, T=0.5, option_type="put")
    assert g_call["gamma"] > 0
    assert g_call["gamma"] == pytest.approx(g_put["gamma"], rel=1e-10)


def test_invalid_option_type_raises():
    with pytest.raises(ValueError):
        bs_price(S=100, K=100, r=0.03, q=0.0, sigma=0.2, T=1.0, option_type="straddle")
