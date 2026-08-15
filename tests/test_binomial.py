import numpy as np
import pytest

from binomial import crr_binomial_price
from black_scholes import bs_price


def test_european_converges_to_black_scholes_no_dividend():
    S, K, r, q, sigma, T = 100.0, 100.0, 0.05, 0.0, 0.2, 1.0
    bs = bs_price(S, K, r, q, sigma, T, "call")
    binom = crr_binomial_price(S, K, r, q, sigma, T, N=2000, option_type="call", exercise="european")
    assert binom == pytest.approx(bs, abs=0.02)


def test_european_converges_to_black_scholes_with_dividend_and_put():
    S, K, r, q, sigma, T = 100.0, 105.0, 0.03, 0.02, 0.25, 0.75
    bs = bs_price(S, K, r, q, sigma, T, "put")
    binom = crr_binomial_price(S, K, r, q, sigma, T, N=2000, option_type="put", exercise="european")
    assert binom == pytest.approx(bs, abs=0.02)


def test_american_call_no_dividend_equals_european():
    # Classic result: never optimal to early-exercise an American call on a
    # non-dividend-paying stock, so American == European in that case.
    S, K, r, q, sigma, T = 100.0, 95.0, 0.05, 0.0, 0.2, 1.0
    euro = crr_binomial_price(S, K, r, q, sigma, T, N=500, option_type="call", exercise="european")
    amer = crr_binomial_price(S, K, r, q, sigma, T, N=500, option_type="call", exercise="american")
    assert amer == pytest.approx(euro, abs=1e-8)


def test_american_put_premium_nonnegative():
    S, K, r, q, sigma, T = 100.0, 110.0, 0.05, 0.0, 0.2, 1.0
    euro = crr_binomial_price(S, K, r, q, sigma, T, N=500, option_type="put", exercise="european")
    amer = crr_binomial_price(S, K, r, q, sigma, T, N=500, option_type="put", exercise="american")
    assert amer >= euro - 1e-9


def test_american_call_with_dividend_can_exceed_european():
    # With a dividend yield, early exercise of a call can become optimal,
    # so American >= European should hold (and typically strictly greater).
    S, K, r, q, sigma, T = 100.0, 90.0, 0.03, 0.06, 0.2, 1.0
    euro = crr_binomial_price(S, K, r, q, sigma, T, N=500, option_type="call", exercise="european")
    amer = crr_binomial_price(S, K, r, q, sigma, T, N=500, option_type="call", exercise="american")
    assert amer >= euro - 1e-9


def test_convergence_rate_is_approximately_order_1_over_n():
    # H1: CRR error should decay ~ O(1/N). Check the log-log slope is
    # negative and roughly in the ballpark of -1 (loose bound here; the
    # precise regression lives in notebook 02).
    S, K, r, q, sigma, T = 100.0, 100.0, 0.05, 0.0, 0.2, 1.0
    bs = bs_price(S, K, r, q, sigma, T, "call")
    Ns = [50, 100, 200, 400, 800, 1600]
    errors = [abs(crr_binomial_price(S, K, r, q, sigma, T, N, "call", "european") - bs) for N in Ns]
    logN = np.log(Ns)
    logErr = np.log(errors)
    slope, _ = np.polyfit(logN, logErr, 1)
    assert -1.6 < slope < -0.5


def test_deep_itm_intrinsic_floor():
    S, K, r, q, sigma, T = 200.0, 50.0, 0.03, 0.0, 0.2, 1.0
    price = crr_binomial_price(S, K, r, q, sigma, T, N=200, option_type="call", exercise="american")
    assert price >= (S - K) - 1e-8


def test_invalid_n_raises():
    with pytest.raises(ValueError):
        crr_binomial_price(100, 100, 0.05, 0.0, 0.2, 1.0, N=0)


def test_invalid_option_type_raises():
    with pytest.raises(ValueError):
        crr_binomial_price(100, 100, 0.05, 0.0, 0.2, 1.0, N=10, option_type="strangle")
